from __future__ import annotations

import pytest

from engram.config import EngramConfig
from engram.storage.bootstrap import (
    borrowed_sqlite_db,
    close_if_supported,
    create_atlas_store_for_graph,
    create_borrowed_consolidation_store_for_graph,
    create_borrowed_sqlite_consolidation_store,
    create_consolidation_store_for_graph,
    create_conversation_store_for_graph,
    create_evaluation_store_for_graph,
    initialize_search_index_for_graph,
    initialize_store_for_graph,
    shared_helix_client,
    shared_sqlite_db,
    stop_if_supported,
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


class FakePathStore(FakeStore):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path


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


class AsyncAClosable:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class SyncStoppable:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


class AsyncStoppable:
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


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
async def test_create_consolidation_store_for_graph_borrows_lite_db(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engram.consolidation.store.SQLiteConsolidationStore",
        FakePathStore,
    )
    db = object()
    path = tmp_path / "consolidation.db"

    store = await create_consolidation_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=FakeGraphStore(db),
        mode=EngineMode.LITE,
        sqlite_path=path,
    )

    assert isinstance(store, FakePathStore)
    assert store.path == str(path)
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_atlas_store_for_graph_borrows_lite_db(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("engram.storage.sqlite.atlas.SQLiteAtlasStore", FakePathStore)
    db = object()
    config = EngramConfig(_env_file=None)
    config.sqlite.path = str(tmp_path / "engram.db")

    store = await create_atlas_store_for_graph(
        config,
        graph_store=FakeGraphStore(db),
        activation_store=object(),
        search_index=object(),
        mode=EngineMode.LITE,
    )

    assert isinstance(store, FakePathStore)
    assert store.path == str(config.get_sqlite_path())
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_conversation_store_for_graph_borrows_lite_db(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engram.storage.sqlite.conversations.SQLiteConversationStore",
        FakePathStore,
    )
    db = object()
    config = EngramConfig(_env_file=None)
    config.sqlite.path = str(tmp_path / "engram.db")

    store = await create_conversation_store_for_graph(
        config,
        graph_store=FakeGraphStore(db),
        mode=EngineMode.LITE,
    )

    assert isinstance(store, FakePathStore)
    assert store.path == str(config.get_sqlite_path())
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_evaluation_store_for_graph_borrows_lite_db(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("engram.evaluation.store.SQLiteEvaluationStore", FakePathStore)
    db = object()
    path = tmp_path / "evaluation.db"

    store = await create_evaluation_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=FakeGraphStore(db),
        mode=EngineMode.LITE,
        sqlite_path=path,
    )

    assert isinstance(store, FakePathStore)
    assert store.path == str(path)
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_consolidation_store_for_graph_shares_helix_client(
    monkeypatch,
) -> None:
    class FakeHelixConsolidationStore:
        def __init__(self, config, client=None) -> None:
            self.config = config
            self.client = client
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    monkeypatch.setattr(
        "engram.storage.helix.consolidation.HelixConsolidationStore",
        FakeHelixConsolidationStore,
    )
    client = object()
    graph = FakeGraphStore()
    graph._helix_client = client

    store = await create_consolidation_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=graph,
        mode=EngineMode.HELIX,
    )

    assert isinstance(store, FakeHelixConsolidationStore)
    assert shared_helix_client(graph) is client
    assert store.client is client
    assert store.initialized is True


@pytest.mark.asyncio
async def test_create_atlas_store_for_graph_shares_helix_client(monkeypatch) -> None:
    class FakeHelixAtlasStore:
        def __init__(self, config, client=None) -> None:
            self.config = config
            self.client = client
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    monkeypatch.setattr("engram.storage.helix.atlas.HelixAtlasStore", FakeHelixAtlasStore)
    client = object()
    graph = FakeGraphStore()
    graph._helix_client = client

    store = await create_atlas_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=graph,
        activation_store=object(),
        search_index=object(),
        mode=EngineMode.HELIX,
    )

    assert isinstance(store, FakeHelixAtlasStore)
    assert store.client is client
    assert store.initialized is True


@pytest.mark.asyncio
async def test_create_conversation_store_for_graph_shares_helix_client(
    monkeypatch,
) -> None:
    class FakeHelixConversationStore:
        def __init__(self, config, client=None) -> None:
            self.config = config
            self.client = client
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    monkeypatch.setattr(
        "engram.storage.helix.conversations.HelixConversationStore",
        FakeHelixConversationStore,
    )
    client = object()
    graph = FakeGraphStore()
    graph._helix_client = client

    store = await create_conversation_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=graph,
        mode=EngineMode.HELIX,
    )

    assert isinstance(store, FakeHelixConversationStore)
    assert store.client is client
    assert store.initialized is True


@pytest.mark.asyncio
async def test_create_atlas_store_for_graph_uses_runtime_redis(monkeypatch) -> None:
    class FakeRedisAtlasStore:
        def __init__(self, redis) -> None:
            self.redis = redis
            self.initialized = False

        async def initialize(self) -> None:
            self.initialized = True

    monkeypatch.setattr("engram.storage.redis.atlas.RedisAtlasStore", FakeRedisAtlasStore)
    redis = object()
    search_index = type("SearchIndex", (), {"_redis": redis})()

    store = await create_atlas_store_for_graph(
        EngramConfig(_env_file=None),
        graph_store=FakeGraphStore(),
        activation_store=object(),
        search_index=search_index,
        mode=EngineMode.FULL,
    )

    assert isinstance(store, FakeRedisAtlasStore)
    assert store.redis is redis
    assert store.initialized is True


@pytest.mark.asyncio
async def test_create_borrowed_sqlite_consolidation_store_uses_supplied_db(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engram.consolidation.store.SQLiteConsolidationStore",
        FakePathStore,
    )
    db = object()

    store = await create_borrowed_sqlite_consolidation_store(db)

    assert isinstance(store, FakePathStore)
    assert store.path == ":memory:"
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_borrowed_consolidation_store_for_graph_uses_graph_db(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "engram.consolidation.store.SQLiteConsolidationStore",
        FakePathStore,
    )
    db = object()
    graph = FakeGraphStore(db)

    store = await create_borrowed_consolidation_store_for_graph(graph)

    assert borrowed_sqlite_db(graph) is db
    assert isinstance(store, FakePathStore)
    assert store.calls == [db]


@pytest.mark.asyncio
async def test_create_borrowed_consolidation_store_for_graph_handles_missing_db() -> None:
    assert await create_borrowed_consolidation_store_for_graph(FakeGraphStore()) is None
    assert await create_borrowed_sqlite_consolidation_store(None) is None


@pytest.mark.asyncio
async def test_close_if_supported_accepts_missing_close() -> None:
    await close_if_supported(object())
    await close_if_supported(None)


@pytest.mark.asyncio
async def test_close_if_supported_handles_sync_and_async_close() -> None:
    sync_resource = SyncClosable()
    async_resource = AsyncClosable()
    async_aclose_resource = AsyncAClosable()

    await close_if_supported(sync_resource)
    await close_if_supported(async_resource)
    await close_if_supported(async_aclose_resource)

    assert sync_resource.closed is True
    assert async_resource.closed is True
    assert async_aclose_resource.closed is True


@pytest.mark.asyncio
async def test_stop_if_supported_accepts_missing_stop() -> None:
    await stop_if_supported(object())
    await stop_if_supported(None)


@pytest.mark.asyncio
async def test_stop_if_supported_handles_sync_and_async_stop() -> None:
    sync_resource = SyncStoppable()
    async_resource = AsyncStoppable()

    await stop_if_supported(sync_resource)
    await stop_if_supported(async_resource)

    assert sync_resource.stopped is True
    assert async_resource.stopped is True
