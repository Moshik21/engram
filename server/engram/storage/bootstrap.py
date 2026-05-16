"""Shared storage initialization helpers for runtime entrypoints."""

from __future__ import annotations

import inspect
from typing import Any

from engram.storage.resolver import EngineMode


def shared_sqlite_db(graph_store: Any, mode: EngineMode) -> Any | None:
    """Return the active SQLite graph connection that companion stores can borrow."""
    if mode != EngineMode.LITE:
        return None
    return getattr(graph_store, "_db", None)


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
