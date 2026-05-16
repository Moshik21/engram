from __future__ import annotations

import pytest

from engram.storage.bootstrap import (
    close_if_supported,
    initialize_search_index_for_graph,
    initialize_store_for_graph,
    shared_sqlite_db,
)
from engram.storage.resolver import EngineMode


class FakeGraphStore:
    def __init__(self, db: object | None = None) -> None:
        if db is not None:
            self._db = db


class FakeStore:
    def __init__(self) -> None:
        self.calls: list[object | None] = []

    async def initialize(self, db: object | None = None) -> None:
        self.calls.append(db)


class SyncClosable:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class AsyncClosable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_initialize_store_for_graph_borrows_lite_db() -> None:
    db = object()
    store = FakeStore()
    graph = FakeGraphStore(db)

    await initialize_store_for_graph(store, graph_store=graph, mode=EngineMode.LITE)

    assert shared_sqlite_db(graph, EngineMode.LITE) is db
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_initialize_store_for_graph_does_not_borrow_outside_lite() -> None:
    db = object()
    store = FakeStore()
    graph = FakeGraphStore(db)

    await initialize_store_for_graph(store, graph_store=graph, mode=EngineMode.HELIX)

    assert shared_sqlite_db(graph, EngineMode.HELIX) is None
    assert store.calls == [None]


@pytest.mark.asyncio
async def test_initialize_search_index_for_graph_uses_same_contract() -> None:
    db = object()
    search = FakeStore()

    await initialize_search_index_for_graph(
        search,
        graph_store=FakeGraphStore(db),
        mode=EngineMode.LITE,
    )

    assert search.calls == [db]


@pytest.mark.asyncio
async def test_close_if_supported_accepts_missing_close() -> None:
    await close_if_supported(object())
    await close_if_supported(None)


@pytest.mark.asyncio
async def test_close_if_supported_handles_sync_and_async_close() -> None:
    sync_resource = SyncClosable()
    async_resource = AsyncClosable()

    await close_if_supported(sync_resource)
    await close_if_supported(async_resource)

    assert sync_resource.closed is True
    assert async_resource.closed is True
