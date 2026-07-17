"""Tests for the one-shot local store lifecycle helper ``open_local_stores``."""

from __future__ import annotations

import pytest

from engram.config import EngramConfig
from engram.storage.bootstrap import open_local_stores
from engram.storage.resolver import EngineMode


class FakeGraphStore:
    def __init__(self, events: list[str], fail_initialize: bool = False) -> None:
        self.events = events
        self.fail_initialize = fail_initialize

    async def initialize(self) -> None:
        if self.fail_initialize:
            raise RuntimeError("graph init failed")
        self.events.append("graph.initialize")

    async def close(self) -> None:
        self.events.append("graph.close")


class FakeActivationStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def close(self) -> None:
        self.events.append("activation.close")


class FakeSearchIndex:
    def __init__(self, events: list[str], fail_close: bool = False) -> None:
        self.events = events
        self.fail_close = fail_close

    async def initialize(self, db: object | None = None) -> None:
        self.events.append("search.initialize")

    async def close(self) -> None:
        if self.fail_close:
            raise RuntimeError("search close failed")
        self.events.append("search.close")


class FakeConsolidationStore:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def close(self) -> None:
        self.events.append("consolidation.close")


def _patch_create_stores(monkeypatch, events: list[str], **graph_kwargs):
    triple = (
        FakeGraphStore(events, **graph_kwargs),
        FakeActivationStore(events),
        FakeSearchIndex(events),
    )

    def fake_create_stores(mode: EngineMode, config: EngramConfig):
        events.append(f"create_stores:{mode.value}")
        return triple

    monkeypatch.setattr("engram.storage.factory.create_stores", fake_create_stores)
    return triple


@pytest.mark.asyncio
async def test_yields_bundle_and_closes_in_reverse_order(monkeypatch) -> None:
    events: list[str] = []
    graph, activation, search = _patch_create_stores(monkeypatch, events)
    config = EngramConfig(_env_file=None)

    async with open_local_stores(config, mode=EngineMode.LITE) as stores:
        assert stores.mode == EngineMode.LITE
        assert stores.graph_store is graph
        assert stores.activation_store is activation
        assert stores.search_index is search
        assert stores.consolidation_store is None
        events.append("body")

    assert events == [
        "create_stores:lite",
        "graph.initialize",
        "search.initialize",
        "body",
        "search.close",
        "activation.close",
        "graph.close",
    ]


@pytest.mark.asyncio
async def test_with_consolidation_creates_after_init_and_closes_first(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    graph, _activation, _search = _patch_create_stores(monkeypatch, events)
    sqlite_path = tmp_path / "consolidation.db"
    seen: dict[str, object] = {}

    async def fake_create_consolidation_store_for_graph(
        config,
        *,
        graph_store,
        mode,
        sqlite_path=None,
    ):
        seen["graph_store"] = graph_store
        seen["mode"] = mode
        seen["sqlite_path"] = sqlite_path
        events.append("consolidation.create")
        return FakeConsolidationStore(events)

    monkeypatch.setattr(
        "engram.storage.bootstrap.create_consolidation_store_for_graph",
        fake_create_consolidation_store_for_graph,
    )

    async with open_local_stores(
        EngramConfig(_env_file=None),
        mode=EngineMode.LITE,
        with_consolidation=True,
        consolidation_sqlite_path=sqlite_path,
    ) as stores:
        assert isinstance(stores.consolidation_store, FakeConsolidationStore)

    assert seen == {
        "graph_store": graph,
        "mode": EngineMode.LITE,
        "sqlite_path": sqlite_path,
    }
    assert events == [
        "create_stores:lite",
        "graph.initialize",
        "search.initialize",
        "consolidation.create",
        "consolidation.close",
        "search.close",
        "activation.close",
        "graph.close",
    ]


@pytest.mark.asyncio
async def test_body_error_still_closes_in_reverse_order(monkeypatch) -> None:
    events: list[str] = []
    _patch_create_stores(monkeypatch, events)

    with pytest.raises(RuntimeError, match="body failed"):
        async with open_local_stores(EngramConfig(_env_file=None), mode=EngineMode.LITE):
            raise RuntimeError("body failed")

    assert events[-3:] == ["search.close", "activation.close", "graph.close"]


@pytest.mark.asyncio
async def test_setup_error_closes_created_stores(monkeypatch) -> None:
    events: list[str] = []
    _patch_create_stores(monkeypatch, events, fail_initialize=True)

    with pytest.raises(RuntimeError, match="graph init failed"):
        async with open_local_stores(EngramConfig(_env_file=None), mode=EngineMode.LITE):
            pytest.fail("body must not run when setup fails")

    assert events == [
        "create_stores:lite",
        "search.close",
        "activation.close",
        "graph.close",
    ]


@pytest.mark.asyncio
async def test_close_failure_is_suppressed_and_unwind_continues(monkeypatch) -> None:
    events: list[str] = []
    triple = (
        FakeGraphStore(events),
        FakeActivationStore(events),
        FakeSearchIndex(events, fail_close=True),
    )
    monkeypatch.setattr(
        "engram.storage.factory.create_stores",
        lambda mode, config: triple,
    )

    async with open_local_stores(EngramConfig(_env_file=None), mode=EngineMode.LITE):
        pass

    assert events == [
        "graph.initialize",
        "search.initialize",
        "activation.close",
        "graph.close",
    ]


@pytest.mark.asyncio
async def test_resolves_mode_when_not_supplied(monkeypatch) -> None:
    events: list[str] = []
    _patch_create_stores(monkeypatch, events)
    resolved: list[str] = []

    async def fake_resolve_mode(mode: str) -> EngineMode:
        resolved.append(mode)
        return EngineMode.LITE

    monkeypatch.setattr("engram.storage.bootstrap.resolve_mode", fake_resolve_mode)
    config = EngramConfig(mode="lite", _env_file=None)

    async with open_local_stores(config) as stores:
        assert stores.mode == EngineMode.LITE

    assert resolved == ["lite"]


@pytest.mark.asyncio
async def test_explicit_mode_skips_resolution(monkeypatch) -> None:
    events: list[str] = []
    _patch_create_stores(monkeypatch, events)

    async def fail_resolve_mode(mode: str) -> EngineMode:
        pytest.fail("resolve_mode must not be called when mode is supplied")

    monkeypatch.setattr("engram.storage.bootstrap.resolve_mode", fail_resolve_mode)

    async with open_local_stores(EngramConfig(_env_file=None), mode=EngineMode.LITE) as stores:
        assert stores.mode == EngineMode.LITE


@pytest.mark.asyncio
async def test_local_runtime_uses_local_runtime_stores(monkeypatch) -> None:
    events: list[str] = []
    triple = (
        FakeGraphStore(events),
        FakeActivationStore(events),
        FakeSearchIndex(events),
    )

    def fake_create_local_runtime_stores(mode: EngineMode, config: EngramConfig):
        events.append(f"create_local_runtime_stores:{mode.value}")
        return triple

    monkeypatch.setattr(
        "engram.storage.bootstrap.create_local_runtime_stores",
        fake_create_local_runtime_stores,
    )

    async with open_local_stores(
        EngramConfig(_env_file=None),
        mode=EngineMode.LITE,
        local_runtime=True,
    ) as stores:
        assert stores.graph_store is triple[0]

    assert events == [
        "create_local_runtime_stores:lite",
        "graph.initialize",
        "search.initialize",
        "search.close",
        "activation.close",
        "graph.close",
    ]
