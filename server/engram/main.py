"""FastAPI application factory and server startup."""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from engram import __version__
from engram.api.activation import router as activation_router
from engram.api.admin import router as admin_router
from engram.api.consolidation import router as consolidation_router
from engram.api.entities import router as entities_router
from engram.api.episodes import router as episodes_router
from engram.api.graph import router as graph_router
from engram.api.health import router as health_router
from engram.api.stats import router as stats_router
from engram.api.websocket import router as ws_router
from engram.config import EngramConfig
from engram.events.bus import get_event_bus
from engram.extraction.extractor import EntityExtractor
from engram.graph_manager import GraphManager
from engram.security.middleware import TenantContextMiddleware
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)

# Module-level app state for dependency injection
_app_state: dict = {}


async def _startup(app: FastAPI, config: EngramConfig) -> None:
    """Initialize storage backends and services."""
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    if hasattr(search_index, "initialize"):
        # In lite mode, share the SQLite connection across stores
        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await search_index.initialize(db=graph_store._db)
        else:
            await search_index.initialize()

    extractor = EntityExtractor()
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
        reranker = create_reranker(api_key=cohere_key or None)

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
    )

    # Consolidation engine + store
    from engram.consolidation.engine import ConsolidationEngine
    from engram.consolidation.store import SQLiteConsolidationStore

    consolidation_store = SQLiteConsolidationStore(str(config.get_sqlite_path()))
    if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
        await consolidation_store.initialize(db=graph_store._db)
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

    _app_state.update(
        {
            "config": config,
            "mode": mode.value,
            "graph_store": graph_store,
            "activation_store": activation_store,
            "search_index": search_index,
            "graph_manager": manager,
            "event_bus": event_bus,
            "embedding_provider": embedding_provider,
            "consolidation_engine": consolidation_engine,
            "consolidation_store": consolidation_store,
            "consolidation_scheduler": consolidation_scheduler,
            "pressure_accumulator": pressure_accumulator,
        }
    )

    logger.info(
        "Engram v%s started in %s mode",
        __version__,
        mode.value,
    )


async def _shutdown() -> None:
    """Cleanup on shutdown."""
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

    graph_store = _app_state.get("graph_store")
    if graph_store and hasattr(graph_store, "close"):
        await graph_store.close()


def create_app(config: EngramConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = EngramConfig()

    app = FastAPI(
        title="Engram",
        version=__version__,
        description="Activation-based memory layer for AI agents",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
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

    @app.on_event("startup")
    async def startup():
        await _startup(app, config)

    @app.on_event("shutdown")
    async def shutdown():
        await _shutdown()

    return app


# Default app instance for uvicorn
app = create_app()
