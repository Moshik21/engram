"""Shared storage initialization helpers for runtime entrypoints."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)


def shared_sqlite_db(graph_store: Any, mode: EngineMode) -> Any | None:
    """Return the active SQLite graph connection that companion stores can borrow."""
    if mode != EngineMode.LITE:
        return None
    return borrowed_sqlite_db(graph_store)


def borrowed_sqlite_db(graph_store: Any) -> Any | None:
    """Return a caller-owned SQLite connection from a graph-like store if present."""
    return getattr(graph_store, "_db", None)


def shared_helix_client(graph_store: Any) -> Any | None:
    """Return the active Helix client that companion stores can share."""
    return getattr(graph_store, "_helix_client", None)


async def initialize_store_for_graph(
    store: Any,
    *,
    graph_store: Any,
    mode: EngineMode,
) -> None:
    """Initialize a store, borrowing the lite graph DB when available."""
    initializer = getattr(store, "initialize", None)
    if initializer is None:
        return

    db = shared_sqlite_db(graph_store, mode)
    result = initializer(db=db) if db is not None else initializer()
    if inspect.isawaitable(result):
        await result


async def initialize_search_index_for_graph(
    search_index: Any,
    *,
    graph_store: Any,
    mode: EngineMode,
) -> None:
    """Initialize a search index beside the active graph store."""
    await initialize_store_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )


def create_local_runtime_stores(
    mode: EngineMode,
    config: EngramConfig,
) -> tuple[Any, Any, Any]:
    """Create graph, activation, and search stores for local CLI runtime reads."""
    if mode == EngineMode.LITE:
        from engram.storage.memory.activation import MemoryActivationStore
        from engram.storage.sqlite.graph import SQLiteGraphStore
        from engram.storage.sqlite.search import FTS5SearchIndex

        db_path = str(config.get_sqlite_path())
        return (
            SQLiteGraphStore(db_path),
            MemoryActivationStore(cfg=config.activation),
            FTS5SearchIndex(db_path),
        )

    from engram.storage.factory import create_stores

    return create_stores(mode, config)


async def create_atlas_store_for_graph(
    config: EngramConfig,
    *,
    graph_store: Any,
    activation_store: Any,
    search_index: Any,
    mode: EngineMode,
) -> Any:
    """Create the atlas store that matches an active graph runtime."""
    if mode == EngineMode.HELIX:
        from engram.storage.helix.atlas import HelixAtlasStore

        store = HelixAtlasStore(
            config.helix,
            client=shared_helix_client(graph_store),
        )
        await store.initialize()
        return store

    if mode == EngineMode.LITE:
        from engram.storage.sqlite.atlas import SQLiteAtlasStore

        store = SQLiteAtlasStore(str(config.get_sqlite_path()))
        await initialize_store_for_graph(store, graph_store=graph_store, mode=mode)
        return store

    from engram.storage.redis.atlas import RedisAtlasStore

    redis_client = getattr(search_index, "_redis", None)
    if redis_client is None:
        redis_client = getattr(activation_store, "_redis", None)
    store = RedisAtlasStore(redis_client)
    await store.initialize()
    return store


async def create_consolidation_store_for_graph(
    config: EngramConfig,
    *,
    graph_store: Any,
    mode: EngineMode,
    sqlite_path: str | Path | None = None,
) -> Any:
    """Create the consolidation audit store that matches an active graph store."""
    if mode == EngineMode.HELIX:
        from engram.storage.helix.consolidation import HelixConsolidationStore

        store = HelixConsolidationStore(
            config.helix,
            client=shared_helix_client(graph_store),
        )
        await store.initialize()
        return store

    if config.postgres.dsn:
        from engram.storage.postgres.consolidation import PostgresConsolidationStore

        store = PostgresConsolidationStore(
            config.postgres.dsn,
            min_pool_size=config.postgres.min_pool_size,
            max_pool_size=config.postgres.max_pool_size,
        )
        await store.initialize()
        return store

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(str(sqlite_path or config.get_sqlite_path()))
    await initialize_store_for_graph(store, graph_store=graph_store, mode=mode)
    return store


async def create_borrowed_sqlite_consolidation_store(db: Any | None) -> Any | None:
    """Create a consolidation store over a caller-owned SQLite connection."""
    if db is None:
        return None

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(":memory:")
    await store.initialize(db=db)
    return store


async def create_borrowed_consolidation_store_for_graph(graph_store: Any | None) -> Any | None:
    """Create a consolidation store over a graph store's borrowed SQLite DB."""
    if graph_store is None:
        return None
    return await create_borrowed_sqlite_consolidation_store(borrowed_sqlite_db(graph_store))


async def create_conversation_store_for_graph(
    config: EngramConfig,
    *,
    graph_store: Any,
    mode: EngineMode,
) -> Any:
    """Create the conversation store that matches an active graph runtime."""
    if mode == EngineMode.HELIX:
        from engram.storage.helix.conversations import HelixConversationStore

        store = HelixConversationStore(
            config.helix,
            client=shared_helix_client(graph_store),
        )
        await store.initialize()
        return store

    from engram.storage.sqlite.conversations import SQLiteConversationStore

    store = SQLiteConversationStore(str(config.get_sqlite_path()))
    await initialize_store_for_graph(store, graph_store=graph_store, mode=mode)
    return store


async def create_evaluation_store_for_graph(
    config: EngramConfig,
    *,
    graph_store: Any,
    mode: EngineMode,
    sqlite_path: str | Path | None = None,
) -> Any:
    """Create the local evaluation label store beside an active graph store."""
    from engram.evaluation.store import SQLiteEvaluationStore

    store = SQLiteEvaluationStore(str(sqlite_path or config.get_sqlite_path()))
    await initialize_store_for_graph(store, graph_store=graph_store, mode=mode)
    return store


async def close_if_supported(resource: Any) -> None:
    """Close a runtime resource if it exposes a sync or async close/aclose method."""
    if resource is None:
        return
    close = getattr(resource, "aclose", None) or getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result


async def stop_if_supported(resource: Any) -> None:
    """Stop a runtime resource if it exposes a sync or async stop method."""
    if resource is None:
        return
    stop = getattr(resource, "stop", None)
    if stop is None:
        return
    result = stop()
    if inspect.isawaitable(result):
        await result


async def stop_task_if_running(task: asyncio.Task | None) -> None:
    """Stop and await a background task if it is still running."""
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@dataclass
class LocalStores:
    """Opened local stores for a one-shot CLI/eval run."""

    mode: EngineMode
    graph_store: Any
    activation_store: Any
    search_index: Any
    consolidation_store: Any | None = None


@asynccontextmanager
async def open_local_stores(
    config: EngramConfig,
    *,
    mode: EngineMode | None = None,
    with_consolidation: bool = False,
    consolidation_sqlite_path: str | Path | None = None,
    local_runtime: bool = False,
) -> AsyncIterator[LocalStores]:
    """Open graph/activation/search (+ optional consolidation) stores for a one-shot run.

    Resolves the engine mode when not supplied, initializes the graph store and
    search index, and guarantees reverse-order close on exit — including when
    setup or the body fails partway.
    """
    if mode is None:
        mode = await resolve_mode(config.mode)

    opened: list[Any] = []
    try:
        if local_runtime:
            graph_store, activation_store, search_index = create_local_runtime_stores(
                mode,
                config,
            )
        else:
            from engram.storage.factory import create_stores

            graph_store, activation_store, search_index = create_stores(mode, config)
        opened = [graph_store, activation_store, search_index]
        await graph_store.initialize()
        await initialize_search_index_for_graph(
            search_index,
            graph_store=graph_store,
            mode=mode,
        )
        consolidation_store = None
        if with_consolidation:
            consolidation_store = await create_consolidation_store_for_graph(
                config,
                graph_store=graph_store,
                mode=mode,
                sqlite_path=consolidation_sqlite_path,
            )
            opened.append(consolidation_store)
        yield LocalStores(
            mode=mode,
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            consolidation_store=consolidation_store,
        )
    finally:
        for resource in reversed(opened):
            try:
                await close_if_supported(resource)
            except Exception:  # silent-ok: unwind must still close remaining stores
                logger.warning("Failed to close local store during unwind", exc_info=True)
