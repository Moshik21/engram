"""FastAPI application factory and server startup."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from engram import __version__
from engram.api.activation import router as activation_router
from engram.api.admin import router as admin_router
from engram.api.consolidation import router as consolidation_router
from engram.api.conversations import router as conversations_router
from engram.api.entities import router as entities_router
from engram.api.episodes import router as episodes_router
from engram.api.evaluation import router as evaluation_router
from engram.api.graph import router as graph_router
from engram.api.health import router as health_router
from engram.api.hygiene import router as hygiene_router
from engram.api.ingest_ws import router as ingest_ws_router
from engram.api.knowledge import router as knowledge_router
from engram.api.lifecycle import router as lifecycle_router
from engram.api.loop import router as loop_router
from engram.api.stats import router as stats_router
from engram.api.storage import router as storage_router
from engram.api.websocket import router as ws_router
from engram.config import EngramConfig
from engram.consolidation_trigger import run_shutdown_consolidation
from engram.events.bus import get_event_bus
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.security.middleware import TenantContextMiddleware
from engram.storage.bootstrap import (
    close_if_supported,
    create_atlas_store_for_graph,
    create_consolidation_store_for_graph,
    create_conversation_store_for_graph,
    create_evaluation_store_for_graph,
    initialize_search_index_for_graph,
    stop_if_supported,
    stop_task_if_running,
)
from engram.storage.factory import create_stores
from engram.storage.protocols import AtlasStore, ConsolidationStore
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)

# Module-level app state for dependency injection
_app_state: dict = {}
_startup_background_tasks: set[asyncio.Task] = set()


async def _wait_for_brain_window(config: EngramConfig) -> None:
    """Never open the graph while a cold-brain window holds the flock.

    Helix native must not be multi-opened; the brain pauses the shell for its
    window, so a serve starting mid-window (user, login RunAtLoad) must wait
    for the lock to clear instead of racing the brain's writes.
    """
    from engram.brain_runtime import brain_lock_is_held, brain_lock_path

    try:
        wait_budget = float(os.environ.get("ENGRAM_BRAIN_LOCK_WAIT_SECONDS", "300"))
    except ValueError:
        wait_budget = 300.0
    if wait_budget <= 0 or not brain_lock_is_held():
        return
    logger.warning(
        "Brain window active (%s held); waiting up to %.0fs before opening the graph",
        brain_lock_path(),
        wait_budget,
    )
    deadline = asyncio.get_event_loop().time() + wait_budget
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(2.0)
        if not brain_lock_is_held():
            logger.info("Brain window cleared; continuing startup")
            return
    raise RuntimeError(
        "brain window still active after "
        f"{wait_budget:.0f}s (ENGRAM_BRAIN_LOCK_WAIT_SECONDS); refusing to "
        "multi-open the graph — retry once 'engram brain run' finishes"
    )


def _start_capture_queue_drain(manager: GraphManager, config: EngramConfig) -> None:
    """Replay offline captures (~/.engram/capture-queue.jsonl) after readiness.

    Captures queued while the shell was down (brain windows, crashes) were
    previously never replayed automatically — the queue rotted until someone
    called POST /api/knowledge/replay-queue by hand.
    """

    async def _drain() -> None:
        try:
            from engram.api.knowledge import _dedup_check
            from engram.ingestion.offline_replay import (
                build_api_manager_offline_replay_surface,
            )
            from engram.utils.offline_queue import drain_queue

            payload = await build_api_manager_offline_replay_surface(
                manager,
                drain_queue=drain_queue,
                dedup_check=_dedup_check,
                group_id=config.default_group_id,
            )
            replayed = payload.get("replayed")
            if replayed:
                logger.info("Capture queue drained on startup: %s", payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Capture queue startup drain failed", exc_info=True)

    _track_startup_background_task(asyncio.create_task(_drain()))


async def _startup(app: FastAPI, config: EngramConfig) -> None:
    """Initialize storage backends and services."""
    await _wait_for_brain_window(config)
    mode = await resolve_mode(config.mode)

    # I3: the native LMDB env must have a single local owner. The MCP stdio
    # path takes the same advisory flock; whichever process starts second is
    # refused fast with the holder's PID instead of silently sharing the env
    # (macOS LMDB's writer lock is a non-robust semaphore — see
    # docs/product/investigations/I3_mcp_concurrent_open.md).
    from engram.storage.native_lock import (
        _acquire_native_shell_lock,
        _native_backend_selected,
        _native_data_dir,
    )

    if _native_backend_selected(config, mode):
        _acquire_native_shell_lock(_native_data_dir(config))

    graph_store, activation_store, search_index = create_stores(mode, config)

    # ACT-R access history lives in the in-memory activation store; restore
    # the last snapshot so 2h brain-window restarts stop wiping it.
    if hasattr(activation_store, "load_from_file"):
        loaded = activation_store.load_from_file(_activation_snapshot_path())
        if loaded:
            logger.info("Restored activation snapshot: %d entities", loaded)

    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )

    extractor = create_extractor(config)
    event_bus = get_event_bus()

    # Extract embedding provider from search index for lifecycle management
    embedding_provider = None
    if hasattr(search_index, "_provider"):
        embedding_provider = search_index._provider

    # Optional reranker (B1)
    reranker = None
    if config.activation.reranker_enabled:
        from engram.retrieval.reranker import create_reranker

        cohere_key = os.environ.get("COHERE_API_KEY", "")
        reranker = create_reranker(
            api_key=cohere_key or None,
            provider=config.activation.reranker_provider,
            local_model=config.activation.reranker_local_model,
        )

    # Optional community store (B2)
    community_store = None
    if config.activation.community_spreading_enabled:
        from engram.activation.community import CommunityStore

        community_store = CommunityStore(
            stale_seconds=config.activation.community_stale_seconds,
            max_iterations=config.activation.community_max_iterations,
        )

    # Optional predicate cache for context-gated spreading (B3)
    predicate_cache = None
    if config.activation.context_gating_enabled and embedding_provider is not None:
        from engram.activation.context_gate import PredicateEmbeddingCache

        predicate_cache = PredicateEmbeddingCache()

    config.configure_runtime_packet_cache(mode.value)
    config.configure_runtime_cue_index_outbox(mode.value)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        extractor,
        cfg=config.activation,
        event_bus=event_bus,
        reranker=reranker,
        community_store=community_store,
        predicate_cache=predicate_cache,
        nerve_center_cfg=config.nerve_center,
        runtime_mode=mode.value,
    )

    atlas_store = cast(
        AtlasStore,
        await create_atlas_store_for_graph(
            config,
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            mode=mode,
        ),
    )

    from engram.atlas.builder import AtlasBuilder
    from engram.atlas.service import AtlasService

    atlas_builder = AtlasBuilder(
        graph_store,
        activation_store,
        config.activation,
        community_store=community_store,
    )
    atlas_service = AtlasService(
        atlas_store,
        atlas_builder,
        graph_store,
    )

    from engram.consolidation.engine import ConsolidationEngine

    consolidation_store = cast(
        ConsolidationStore,
        await create_consolidation_store_for_graph(
            config,
            graph_store=graph_store,
            mode=mode,
        ),
    )

    consolidation_engine = ConsolidationEngine(
        graph_store,
        activation_store,
        search_index,
        cfg=config.activation,
        consolidation_store=consolidation_store,
        event_bus=event_bus,
        extractor=extractor,
        graph_manager=manager,
    )

    evaluation_store = await create_evaluation_store_for_graph(
        config,
        graph_store=graph_store,
        mode=mode,
    )

    # Pressure accumulator (optional)
    from engram.consolidation.pressure import PressureAccumulator

    pressure_accumulator = None
    if config.activation.consolidation_pressure_enabled:
        pressure_accumulator = PressureAccumulator()
        pressure_accumulator.start(config.default_group_id, event_bus)

    # Proactive notification surfacing
    from engram.notifications.collector import NotificationCollector
    from engram.notifications.store import NotificationStore
    from engram.notifications.surface import NotificationSurfaceService
    from engram.notifications.temporal import TemporalIntentionScanner

    notification_store = NotificationStore(
        max_per_group=config.activation.notification_max_per_group,
    )
    notification_surface_service = NotificationSurfaceService(notification_store)
    notification_collector = None
    if config.activation.notification_surfacing_enabled:
        notification_collector = NotificationCollector(
            notification_store, config.activation, consolidation_store
        )
        event_bus.add_on_publish_hook(notification_collector.on_event)

    temporal_scanner = None
    if config.activation.notification_temporal_enabled:
        temporal_scanner = TemporalIntentionScanner(notification_store, config.activation)

    # Consolidation scheduler + EpisodeWorker only in monolith role.
    # Shell role keeps the golden-loop API small; cold brain is a separate process.
    from engram.consolidation.scheduler import ConsolidationScheduler

    in_process_brain = config.shell_runs_in_process_brain()
    if not in_process_brain:
        logger.info(
            "Runtime role=%s: in-process consolidation scheduler and "
            "EpisodeWorker disabled (cold brain is external)",
            config.runtime_role,
        )

    consolidation_scheduler = ConsolidationScheduler(
        consolidation_engine,
        config.activation,
        default_group_id=config.default_group_id,
        pressure=pressure_accumulator,
        temporal_scanner=temporal_scanner,
        graph_store=graph_store,
        consolidation_store=consolidation_store,
    )
    if in_process_brain and config.activation.consolidation_enabled:
        consolidation_scheduler.start()

    # Background episode worker
    from engram.ingestion.worker_runtime import EpisodeWorkerRuntimeStores
    from engram.worker import EpisodeWorker

    episode_worker = None
    if in_process_brain and config.activation.worker_enabled:
        episode_worker = EpisodeWorker(
            manager,
            config.activation,
            stores=EpisodeWorkerRuntimeStores(
                graph=graph_store,
                activation=activation_store,
                search=search_index,
            ),
        )
        episode_worker.start(config.default_group_id, event_bus)

    cue_index_outbox_task = None
    if config.activation.cue_index_outbox_enabled:
        cue_index_outbox_task = asyncio.create_task(
            manager.drain_cue_index_outbox(
                limit=config.activation.cue_index_outbox_replay_limit,
                include_failed=False,
            ),
        )

    # Rate limiter + usage meter (Redis-backed in full mode)
    from engram.security.rate_limit import RateLimiter
    from engram.security.usage import UsageMeter

    redis_for_metering = None
    if mode == EngineMode.FULL:
        import redis.asyncio as _aioredis

        metering_kwargs: dict = {"decode_responses": False}
        if config.redis.ssl_cert_reqs:
            import ssl as _ssl

            _reqs_map = {
                "required": _ssl.CERT_REQUIRED,
                "optional": _ssl.CERT_OPTIONAL,
                "none": _ssl.CERT_NONE,
            }
            metering_kwargs["ssl_cert_reqs"] = _reqs_map.get(
                config.redis.ssl_cert_reqs,
                _ssl.CERT_REQUIRED,
            )
        redis_for_metering = _aioredis.from_url(config.redis.url, **metering_kwargs)

    rate_limiter = RateLimiter(
        redis_client=redis_for_metering if config.rate_limit.enabled else None,
        limits={
            "observe": (config.rate_limit.observe_per_min, 60),
            "remember": (config.rate_limit.remember_per_min, 60),
            "recall": (config.rate_limit.recall_per_min, 60),
            "trigger": (config.rate_limit.trigger_per_hour, 3600),
            "chat": (config.rate_limit.chat_per_min, 60),
        }
        if config.rate_limit.enabled
        else None,
    )
    usage_meter = UsageMeter(redis_client=redis_for_metering)

    # In full mode, subscribe to Redis events from MCP processes
    redis_subscriber = None
    if mode == EngineMode.FULL:
        from engram.events.redis_bridge import create_subscriber

        redis_subscriber = await create_subscriber(
            config.default_group_id,
            event_bus,
            redis_url=config.redis.url,
        )
        if redis_subscriber:
            await redis_subscriber.start()

    conversation_store = await create_conversation_store_for_graph(
        config,
        graph_store=graph_store,
        mode=mode,
    )

    from engram.storage.diagnostics import StorageDiagnostics

    storage_diagnostics = await StorageDiagnostics.create(
        config=config,
        mode=mode.value,
        graph_store=graph_store,
        group_id=config.default_group_id,
    )
    manager.attach_storage_diagnostics(storage_diagnostics)

    if (
        mode.value == "helix"
        and getattr(config.helix, "transport", "") == "native"
        and config.activation.capture_startup_warmup_enabled
    ):
        try:
            warmup_timings = await _warm_capture_store_bounded(manager, config)
            logger.info("Native capture warmup completed: %s", warmup_timings)
        except TimeoutError:
            logger.warning(
                "Native capture warmup exceeded %.1fs; continuing startup",
                _capture_startup_warmup_timeout_seconds(config),
            )
        except Exception:
            logger.debug("Native capture warmup failed", exc_info=True)

    if mode.value == "helix" and getattr(config.helix, "transport", "") == "native":
        _start_graph_stats_warmup(manager, config)
    if predicate_cache is not None and embedding_provider is not None:
        _start_predicate_cache_warmup(predicate_cache, config, embedding_provider)
    if config.activation.continuity_startup_warmup_enabled:
        _start_continuity_warmup(manager, config)
    _start_capture_queue_drain(manager, config)
    _start_activation_snapshot_task(activation_store, config)

    _app_state.update(
        {
            "config": config,
            "conversation_store": conversation_store,
            "mode": mode.value,
            "graph_store": graph_store,
            "activation_store": activation_store,
            "search_index": search_index,
            "graph_manager": manager,
            "atlas_store": atlas_store,
            "atlas_service": atlas_service,
            "event_bus": event_bus,
            "embedding_provider": embedding_provider,
            "consolidation_engine": consolidation_engine,
            "consolidation_store": consolidation_store,
            "evaluation_store": evaluation_store,
            "consolidation_scheduler": consolidation_scheduler,
            "pressure_accumulator": pressure_accumulator,
            "episode_worker": episode_worker,
            "redis_subscriber": redis_subscriber,
            "rate_limiter": rate_limiter,
            "usage_meter": usage_meter,
            "redis_metering": redis_for_metering,
            "notification_store": notification_store,
            "notification_surface_service": notification_surface_service,
            "notification_collector": notification_collector,
            "temporal_scanner": temporal_scanner,
            "storage_diagnostics": storage_diagnostics,
            "cue_index_outbox_task": cue_index_outbox_task,
            "predicate_cache": predicate_cache,
        }
    )

    logger.info(
        "Engram v%s started in %s mode",
        __version__,
        mode.value,
    )


def _activation_snapshot_path():
    from engram.storage.memory.activation import activation_snapshot_path

    return activation_snapshot_path()


def _start_activation_snapshot_task(activation_store, config: EngramConfig) -> asyncio.Task | None:
    """RF M4.1: periodic activation-snapshot saves in the shell lifespan.

    The shell is the primary snapshot owner, so it saves without ownership
    probes; the store gates each attempt on elapsed >= interval AND dirty >=
    min, bounding kill -9 access loss to ~interval instead of losing
    everything since the last clean shutdown.
    """
    interval = float(config.activation.activation_snapshot_interval_seconds)
    dirty_min = int(config.activation.activation_snapshot_dirty_min)
    if interval <= 0 or not hasattr(activation_store, "maybe_save_periodic"):
        return None

    async def _loop() -> None:
        # Poll faster than the interval so a save fires promptly once due.
        poll = min(60.0, interval)
        while True:
            await asyncio.sleep(poll)
            try:
                saved = activation_store.maybe_save_periodic(
                    _activation_snapshot_path(),
                    interval_seconds=interval,
                    dirty_min=dirty_min,
                )
                if saved:
                    logger.info("Periodic activation snapshot: %d entities", saved)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("Periodic activation snapshot failed", exc_info=True)

    task = asyncio.create_task(_loop())
    _track_startup_background_task(task)
    return task


async def _shutdown() -> None:
    """Cleanup on shutdown."""
    activation_store = _app_state.get("activation_store")
    if activation_store is not None and hasattr(activation_store, "save_to_file"):
        saved = activation_store.save_to_file(_activation_snapshot_path())
        if saved:
            logger.info("Saved activation snapshot: %d entities", saved)

    cue_index_outbox_task = _app_state.get("cue_index_outbox_task")
    await stop_task_if_running(cue_index_outbox_task)
    for task in list(_startup_background_tasks):
        await stop_task_if_running(task)

    # Stop Redis event subscriber
    await stop_if_supported(_app_state.get("redis_subscriber"))

    # Stop episode worker
    await stop_if_supported(_app_state.get("episode_worker"))

    # Stop pressure accumulator
    await stop_if_supported(_app_state.get("pressure_accumulator"))

    # Stop consolidation scheduler
    await stop_if_supported(_app_state.get("consolidation_scheduler"))

    # Only the monolith role may run in-process consolidation. A shell-role
    # stop is usually the brain pausing us — running a live cycle here races
    # the brain's exclusive graph open (observed as zombie 'shutdown' cycles).
    shutdown_config = _app_state.get("config")
    if shutdown_config is not None and shutdown_config.shell_runs_in_process_brain():
        await run_shutdown_consolidation(
            _app_state.get("consolidation_engine"),
            config=shutdown_config,
            logger=logger,
        )
    else:
        engine = _app_state.get("consolidation_engine")
        if engine is not None and getattr(engine, "is_running", False):
            engine.cancel()

    await close_if_supported(_app_state.get("redis_metering"))
    await close_if_supported(_app_state.get("consolidation_store"))
    await close_if_supported(_app_state.get("evaluation_store"))

    # Close OIDC validator (httpx client)
    from engram.security.middleware import _oidc_validator

    await close_if_supported(_oidc_validator)

    await close_if_supported(_app_state.get("embedding_provider"))
    await close_if_supported(_app_state.get("atlas_store"))
    await close_if_supported(_app_state.get("conversation_store"))

    manager = _app_state.get("graph_manager")
    if manager and hasattr(manager, "close_runtime_resources"):
        await manager.close_runtime_resources()
    else:
        await close_if_supported(_app_state.get("search_index"))
        await close_if_supported(_app_state.get("activation_store"))
        await close_if_supported(_app_state.get("graph_store"))


async def _warm_capture_store_bounded(
    manager: GraphManager,
    config: EngramConfig,
) -> dict[str, float]:
    task = asyncio.create_task(manager.warm_capture_store())
    _track_startup_background_task(task)
    timeout_seconds = _capture_startup_warmup_timeout_seconds(config)
    if timeout_seconds <= 0:
        return await task
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=timeout_seconds)
    except TimeoutError:
        # Keep the create/delete probe running so a late create can still clean itself up.
        raise


def _capture_startup_warmup_timeout_seconds(config: EngramConfig) -> float:
    timeout_ms = getattr(config.activation, "capture_startup_warmup_timeout_ms", 2000)
    return max(0.0, float(timeout_ms) / 1000.0)


def _start_predicate_cache_warmup(
    predicate_cache: object,
    config: EngramConfig,
    embedding_provider: object,
) -> None:
    """Warm predicate embeddings after startup readiness is no longer blocked."""

    async def _warm() -> None:
        try:
            await predicate_cache.initialize(config.activation, embedding_provider)  # type: ignore[attr-defined]
            count = len(predicate_cache.get_embeddings())  # type: ignore[attr-defined]
            logger.info("Predicate embedding cache warmup completed: predicates=%d", count)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Failed to warm predicate cache", exc_info=True)

    task = asyncio.create_task(_warm())
    _track_startup_background_task(task)
    logger.info("Predicate embedding cache warmup started")


def _start_graph_stats_warmup(manager: GraphManager, config: EngramConfig) -> None:
    """Start graph stats cache refresh without blocking startup readiness."""
    warm_graph_stats = getattr(manager, "warm_graph_stats", None)
    if not callable(warm_graph_stats):
        return
    try:
        task = warm_graph_stats(config.default_group_id)
    except TypeError:
        task = warm_graph_stats(group_id=config.default_group_id)
    except Exception:
        logger.debug("Native graph stats warmup failed to start", exc_info=True)
        return
    if asyncio.iscoroutine(task):
        task = asyncio.create_task(task)
    if isinstance(task, asyncio.Task):
        _track_startup_background_task(task)
        logger.info("Native graph stats warmup started")


def _start_continuity_warmup(manager: GraphManager, config: EngramConfig) -> None:
    """Background-warm product continuity paths after readiness.

    LaunchAgent cold boots previously paid multi-second first-hit latency on
    get_context/recall. This primes identity_core listing, durable context packs,
    and Decision-name recall without blocking /health.
    """

    async def _warm() -> None:
        from engram.retrieval.context_builder import build_mcp_context_surface
        from engram.retrieval.recall_surface import build_api_recall_surface

        group_id = config.default_group_id
        timings: dict[str, float] = {}

        async def _stage(name: str, coro):
            started = asyncio.get_running_loop().time()
            try:
                return await coro
            finally:
                timings[name] = round(
                    (asyncio.get_running_loop().time() - started) * 1000,
                    1,
                )

        try:
            # 1) Live storage counts (seeds write-through baseline for AXI)
            diagnostics = getattr(manager, "_storage_diagnostics", None)
            if diagnostics is not None and hasattr(diagnostics, "snapshot"):
                await _stage(
                    "storage_live",
                    diagnostics.snapshot(group_id=group_id, live=True, timeout_seconds=5.0),
                )

            # 2) Identity-core list (fast durable path used by get_context)
            graph = getattr(manager, "_graph", None)
            probe_name = "Cold Decision hit requires healthy search index"
            if graph is not None and hasattr(graph, "get_identity_core_entities"):
                core = await _stage(
                    "identity_core",
                    graph.get_identity_core_entities(group_id),
                )
                if isinstance(core, list) and core:
                    for ent in core:
                        name = getattr(ent, "name", None) or (
                            ent.get("name") if isinstance(ent, dict) else None
                        )
                        et = getattr(ent, "entity_type", None) or (
                            ent.get("entity_type") if isinstance(ent, dict) else None
                        )
                        if name and str(et or "") in {
                            "Decision",
                            "Preference",
                            "Goal",
                            "Commitment",
                            "Correction",
                        }:
                            probe_name = str(name)
                            break

            # 3) Durable get_context pack
            await _stage(
                "get_context",
                build_mcp_context_surface(
                    manager,
                    group_id=group_id,
                    max_tokens=1200,
                    topic_hint=probe_name,
                    project_path=None,
                    format="structured",
                    operation_source="startup_warmup",
                ),
            )

            # 4) Exact Decision recall (primes packet cache)
            await _stage(
                "recall",
                build_api_recall_surface(
                    manager,
                    group_id=group_id,
                    query=probe_name,
                    limit=5,
                    project_path=None,
                    operation_source="startup_warmup",
                ),
            )

            logger.info(
                "Continuity startup warmup completed: probe=%r timings_ms=%s",
                probe_name,
                timings,
            )
            # Product readiness: first Decision/identity path is warm.
            _app_state["product_ready"] = True
            _app_state["product_ready_timings_ms"] = dict(timings)
            _app_state.pop("product_ready_degraded", None)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Continuity startup warmup failed (timings_ms=%s)",
                timings,
                exc_info=True,
            )
            # Mark ready-ish so clients prefer durable/packet paths over hybrid thrash.
            _app_state["product_ready"] = True
            _app_state["product_ready_degraded"] = True

    async def _warm_bounded() -> None:
        timeout_ms = int(
            getattr(config.activation, "continuity_startup_warmup_timeout_ms", 15000) or 15000
        )
        timeout_seconds = max(1.0, timeout_ms / 1000.0)
        try:
            await asyncio.wait_for(_warm(), timeout=timeout_seconds)
        except TimeoutError:
            logger.warning(
                "Continuity startup warmup exceeded %.1fs; continuing in background cancel",
                timeout_seconds,
            )
            _app_state["product_ready"] = True
            _app_state["product_ready_degraded"] = True
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(_warm_bounded())
    _track_startup_background_task(task)
    logger.info("Continuity startup warmup started")


def _track_startup_background_task(task: asyncio.Task) -> None:
    _startup_background_tasks.add(task)

    def _discard(done: asyncio.Task) -> None:
        _startup_background_tasks.discard(done)
        if done.cancelled():
            return
        try:
            done.exception()
        except Exception:
            logger.debug("Startup background task failed", exc_info=True)

    task.add_done_callback(_discard)


def create_app(config: EngramConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = EngramConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _startup(app, config)
        mcp_server_module = getattr(app.state, "mcp_server_module", None)
        if mcp_server_module is not None:
            mcp_server_module.attach_external_runtime(
                manager=_app_state["graph_manager"],
                config=_app_state["config"],
                mode=_app_state["mode"],
                graph_store=_app_state["graph_store"],
                group_id=_app_state["config"].default_group_id,
            )
            mcp_server_module.set_background_runtime_managed_externally(True)
        try:
            async with AsyncExitStack() as stack:
                mcp_session_manager = getattr(app.state, "mcp_session_manager", None)
                if mcp_session_manager is not None:
                    await stack.enter_async_context(mcp_session_manager.run())
                yield
        finally:
            if mcp_server_module is not None:
                mcp_server_module.clear_external_runtime()
                mcp_server_module.set_background_runtime_managed_externally(False)
            await _shutdown()

    app = FastAPI(
        title="Engram",
        version=__version__,
        description="Activation-based memory layer for AI agents",
        lifespan=lifespan,
    )

    # CORS
    origins = list(config.cors.allowed_origins)
    if config.cors.production_origin:
        origins.append(config.cors.production_origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
    )

    # Tenant context middleware
    app.add_middleware(TenantContextMiddleware, config=config.auth)

    @app.middleware("http")
    async def mcp_http_health_probe(request, call_next):
        accept = request.headers.get("accept", "")
        if (
            request.method == "GET"
            and request.url.path == "/mcp"
            and "text/event-stream" not in accept
        ):
            return JSONResponse(
                {
                    "status": "ok",
                    "transport": "streamable-http",
                    "path": "/mcp",
                }
            )
        return await call_next(request)

    # Routes
    app.include_router(health_router)
    app.include_router(graph_router)
    app.include_router(entities_router)
    app.include_router(episodes_router)
    app.include_router(stats_router)
    app.include_router(storage_router)
    app.include_router(lifecycle_router)
    app.include_router(activation_router)
    app.include_router(admin_router)
    app.include_router(consolidation_router)
    app.include_router(hygiene_router)
    app.include_router(loop_router)
    app.include_router(evaluation_router)
    app.include_router(ws_router, tags=["websocket"])
    app.include_router(ingest_ws_router, tags=["websocket"])
    app.include_router(knowledge_router)
    app.include_router(conversations_router)

    # FastMCP's streamable HTTP ASGI app already exposes its protocol route at
    # /mcp. Mount it at the REST app root so the advertised URL stays /mcp.
    if os.environ.get("ENGRAM_MCP_ENABLED", "1") != "0":
        try:
            from engram.mcp import server as mcp_server_module

            mcp_server = mcp_server_module.mcp

            mcp_server.settings.stateless_http = True
            # FastMCP session managers are single-run lifecycle objects. The
            # REST app factory may be called repeatedly in tests and operator
            # restarts, so each app instance needs its own manager.
            mcp_server._session_manager = None
            mcp_app = mcp_server.streamable_http_app()
            app.state.mcp_server_module = mcp_server_module
            app.state.mcp_session_manager = mcp_server.session_manager
            app.mount("/", mcp_app)
            logger.info("MCP streamable-http mounted at /mcp")
        except Exception:
            logger.warning(
                "Failed to mount MCP transport",
                exc_info=True,
            )

    return app


# Default app instance for uvicorn
app = create_app()
