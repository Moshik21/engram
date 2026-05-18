"""Shared storage initialization helpers for runtime entrypoints."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.storage.resolver import EngineMode


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
    return await create_borrowed_sqlite_consolidation_store(
        borrowed_sqlite_db(graph_store)
    )


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
    """Close a runtime resource if it exposes a sync or async close method."""
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if inspect.isawaitable(result):
        await result
