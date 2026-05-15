from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram import consolidation_trigger as trigger_module
from engram.config import ActivationConfig
from engram.consolidation_trigger import ConsolidationTriggerService


class FakeGraphStore:
    def __init__(self) -> None:
        self._db = object()
        self.get_stats_calls: list[str] = []

    async def get_stats(self, group_id: str) -> dict:
        self.get_stats_calls.append(group_id)
        return {"episodes": 3}


@pytest.mark.asyncio
async def test_consolidation_trigger_service_runs_cycle(monkeypatch) -> None:
    captured: dict[str, object] = {}
    cycle = SimpleNamespace(id="cyc_1", status="completed")

    class FakeConsolidationEngine:
        def __init__(self, *args, **kwargs) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        async def run_cycle(self, **kwargs):
            captured["run_cycle"] = kwargs
            return cycle

    monkeypatch.setattr(
        trigger_module,
        "_build_consolidation_engine",
        FakeConsolidationEngine,
    )

    graph = FakeGraphStore()
    activation = SimpleNamespace()
    search = SimpleNamespace()
    extractor = SimpleNamespace()
    store = SimpleNamespace()
    service = ConsolidationTriggerService(
        graph_store=graph,
        activation_store=activation,
        search_index=search,
        cfg=ActivationConfig(),
        extractor=extractor,
    )

    result = await service.trigger_consolidation_cycle(
        group_id="brain",
        trigger="mcp",
        dry_run=True,
        consolidation_store=store,
    )

    assert result.cycle is cycle
    assert result.graph_stats == {"episodes": 3}
    assert graph.get_stats_calls == ["brain"]
    assert captured["args"][:3] == (graph, activation, search)
    assert captured["kwargs"]["consolidation_store"] is store
    assert captured["kwargs"]["extractor"] is extractor
    assert captured["run_cycle"] == {
        "group_id": "brain",
        "trigger": "mcp",
        "dry_run": True,
    }


def test_consolidation_trigger_service_exposes_shared_sqlite_db() -> None:
    graph = FakeGraphStore()
    service = ConsolidationTriggerService(
        graph_store=graph,
        activation_store=SimpleNamespace(),
        search_index=SimpleNamespace(),
        cfg=ActivationConfig(),
        extractor=None,
    )

    assert service.shared_sqlite_db() is graph._db
