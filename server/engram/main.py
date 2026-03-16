"""FastAPI application factory and server startup."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engram import __version__
from engram.api.activation import router as activation_router
from engram.api.admin import router as admin_router
from engram.api.consolidation import router as consolidation_router
from engram.api.conversations import router as conversations_router
from engram.api.entities import router as entities_router
from engram.api.episodes import router as episodes_router
from engram.api.graph import router as graph_router
from engram.api.health import router as health_router
from engram.api.knowledge import router as knowledge_router
from engram.api.stats import router as stats_router
from engram.api.websocket import router as ws_router
from engram.config import EngramConfig
from engram.events.bus import get_event_bus
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.security.middleware import TenantContextMiddleware
from engram.storage.factory import create_stores
from engram.storage.protocols import AtlasStore, ConsolidationStore
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)

# Module-level app state for dependency injection
_app_state: dict = {}


async def _startup(app: FastAPI, config: EngramConfig) -> None:
    """Initialize storage backends and services."""
    mode = await resolve_mode(config.mode)

    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    search_initializer = cast(Any, search_index).initialize
    if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
        await search_initializer(db=graph_store._db)
    else:
        await search_initializer()

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
        try:
            await predicate_cache.initialize(config.activation, embedding_provider)
        except Exception:
            logger.warning("Failed to initialize predicate cache", exc_info=True)
            predicate_cache = None

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
        runtime_mode=mode.value,
    )

    if mode == EngineMode.HELIX:
        from engram.storage.helix.atlas import HelixAtlasStore

        # Share the async HelixClient across all Helix stores
        helix_client = getattr(graph_store, "_helix_client", None)
        atlas_store: AtlasStore = HelixAtlasStore(config.helix, client=helix_client)
        await atlas_store.initialize()
    elif mode == EngineMode.LITE:
        from engram.storage.sqlite.atlas import SQLiteAtlasStore

        atlas_store = SQLiteAtlasStore(str(config.get_sqlite_path()))
        if hasattr(graph_store, "_db"):
            await atlas_store.initialize(db=graph_store._db)
        else:
            await atlas_store.initialize()
    else:
        from engram.storage.redis.atlas import RedisAtlasStore

        redis_client = getattr(search_index, "_redis", None)
        if redis_client is None:
            redis_client = getattr(activation_store, "_redis", None)
        atlas_store = RedisAtlasStore(redis_client)
        await atlas_store.initialize()

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

    # Consolidation engine + store
    from engram.consolidation.engine import ConsolidationEngine

    if mode == EngineMode.HELIX:
        from engram.storage.helix.consolidation import HelixConsolidationStore

        consolidation_store: ConsolidationStore = cast(
            ConsolidationStore,
            HelixConsolidationStore(config.helix, client=helix_client),
        )
        await consolidation_store.initialize()
    elif config.postgres.dsn:
        from engram.storage.postgres.consolidation import PostgresConsolidationStore

        consolidation_store: ConsolidationStore = cast(
            ConsolidationStore,
            PostgresConsolidationStore(
                config.postgres.dsn,
                min_pool_size=config.postgres.min_pool_size,
                max_pool_size=config.postgres.max_pool_size,
            ),
        )
        await consolidation_store.initialize()
    else:
        from engram.consolidation.store import SQLiteConsolidationStore

        consolidation_store = cast(
            ConsolidationStore,
            SQLiteConsolidationStore(str(config.get_sqlite_path())),
        )
        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await cast(Any, consolidation_store).initialize(db=graph_store._db)
        else:
            await consolidation_store.initialize()

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

    # Pressure accumulator (optional)
    from engram.consolidation.pressure import PressureAccumulator

    pressure_accumulator = None
    if config.activation.consolidation_pressure_enabled:
        pressure_accumulator = PressureAccumulator()
        pressure_accumulator.start(config.default_group_id, event_bus)

    # Consolidation scheduler
    from engram.consolidation.scheduler import ConsolidationScheduler

    consolidation_scheduler = ConsolidationScheduler(
        consolidation_engine,
        config.activation,
        default_group_id=config.default_group_id,
        pressure=pressure_accumulator,
    )
    if config.activation.consolidation_enabled:
        consolidation_scheduler.start()

    # Background episode worker
    from engram.worker import EpisodeWorker

    episode_worker = None
    if config.activation.worker_enabled:
        episode_worker = EpisodeWorker(manager, config.activation)
        episode_worker.start(config.default_group_id, event_bus)

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

    # Conversation store
    if mode == EngineMode.HELIX:
        from engram.storage.helix.conversations import HelixConversationStore

        conversation_store = HelixConversationStore(config.helix, client=helix_client)
        await conversation_store.initialize()
    else:
        from engram.storage.sqlite.conversations import SQLiteConversationStore

        conversation_store = SQLiteConversationStore(str(config.get_sqlite_path()))
        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await conversation_store.initialize(db=graph_store._db)
        else:
            await conversation_store.initialize()

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
            "consolidation_scheduler": consolidation_scheduler,
            "pressure_accumulator": pressure_accumulator,
            "episode_worker": episode_worker,
            "redis_subscriber": redis_subscriber,
            "rate_limiter": rate_limiter,
            "usage_meter": usage_meter,
            "redis_metering": redis_for_metering,
        }
    )

    logger.info(
        "Engram v%s started in %s mode",
        __version__,
        mode.value,
    )


async def _shutdown() -> None:
    """Cleanup on shutdown."""
    # Stop Redis event subscriber
    redis_sub = _app_state.get("redis_subscriber")
    if redis_sub:
        await redis_sub.stop()

    # Stop episode worker
    worker = _app_state.get("episode_worker")
    if worker:
        await worker.stop()

    # Stop pressure accumulator
    pressure = _app_state.get("pressure_accumulator")
    if pressure:
        await pressure.stop()

    # Stop consolidation scheduler
    scheduler = _app_state.get("consolidation_scheduler")
    if scheduler:
        await scheduler.stop()

    # Run final consolidation cycle or cancel running one
    engine = _app_state.get("consolidation_engine")
    config = _app_state.get("config")

    if engine:
        if engine.is_running:
            engine.cancel()
        elif config and config.activation.consolidation_enabled:
            try:
                await engine.run_cycle(
                    group_id=config.default_group_id,
                    trigger="shutdown",
                    dry_run=False,
                )
            except Exception:
                logger.warning("Shutdown consolidation failed", exc_info=True)

    # Close metering Redis client
    redis_metering = _app_state.get("redis_metering")
    if redis_metering and hasattr(redis_metering, "aclose"):
        await redis_metering.aclose()

    # Close consolidation store (asyncpg pool in Postgres mode)
    consolidation_store = _app_state.get("consolidation_store")
    if consolidation_store and hasattr(consolidation_store, "close"):
        await consolidation_store.close()

    # Close OIDC validator (httpx client)
    from engram.security.middleware import _oidc_validator

    if _oidc_validator and hasattr(_oidc_validator, "close"):
        await _oidc_validator.close()

    # Close embedding provider
    provider = _app_state.get("embedding_provider")
    if provider and hasattr(provider, "close"):
        await provider.close()

    # Close activation store (Redis client in full mode)
    activation_store = _app_state.get("activation_store")
    if activation_store and hasattr(activation_store, "close"):
        await activation_store.close()

    search_index = _app_state.get("search_index")
    if search_index and hasattr(search_index, "close"):
        await search_index.close()

    atlas_store = _app_state.get("atlas_store")
    if atlas_store and hasattr(atlas_store, "close"):
        await atlas_store.close()

    graph_store = _app_state.get("graph_store")
    if graph_store and hasattr(graph_store, "close"):
        await graph_store.close()


def create_app(config: EngramConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = EngramConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _startup(app, config)
        try:
            yield
        finally:
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

    # Routes
    app.include_router(health_router)
    app.include_router(graph_router)
    app.include_router(entities_router)
    app.include_router(episodes_router)
    app.include_router(stats_router)
    app.include_router(activation_router)
    app.include_router(admin_router)
    app.include_router(consolidation_router)
    app.include_router(ws_router)
    app.include_router(knowledge_router)
    app.include_router(conversations_router)

    # Mount MCP streamable-http transport at /mcp
    if os.environ.get("ENGRAM_MCP_ENABLED", "1") != "0":
        try:
            from engram.mcp.server import mcp as mcp_server

            mcp_server.settings.stateless_http = True
            mcp_app = mcp_server.streamable_http_app()
            app.mount("/mcp", mcp_app)
            logger.info("MCP streamable-http mounted at /mcp")
        except Exception:
            logger.warning(
                "Failed to mount MCP transport",
                exc_info=True,
            )

    return app


# Default app instance for uvicorn
app = create_app()
