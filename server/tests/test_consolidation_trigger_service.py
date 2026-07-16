from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram import consolidation_trigger as trigger_module
from engram.config import ActivationConfig, NerveCenterConfig
from engram.consolidation.audit_reader import ConsolidationCycleDetail
from engram.consolidation_trigger import (
    ConsolidationTriggerResult,
    ConsolidationTriggerService,
    build_api_consolidation_cycle_detail_surface,
    build_api_consolidation_history_surface,
    build_api_consolidation_status_response_surface,
    build_api_consolidation_status_surface,
    build_api_consolidation_trigger_response_surface,
    build_api_consolidation_trigger_surface,
    build_mcp_consolidation_status_surface,
    build_mcp_consolidation_trigger_surface,
    resolve_mcp_consolidation_trigger_store,
    run_api_consolidation_cycle,
)
from engram.models.consolidation import ConsolidationCycle, PhaseResult


class FakeGraphStore:
    def __init__(self, stats: dict | None = None) -> None:
        self._db = object()
        self._stats = stats or {"episodes": 3}
        self.get_stats_calls: list[str] = []

    async def get_stats(self, group_id: str) -> dict:
        self.get_stats_calls.append(group_id)
        return dict(self._stats)


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


@pytest.mark.asyncio
async def test_autonomous_pressure_consolidation_blocks_below_cortical_unlock(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeConsolidationEngine:
        def __init__(self, *args, **kwargs) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        async def run_cycle(self, **kwargs):
            captured["run_cycle"] = kwargs
            return SimpleNamespace(id="blocked")

    monkeypatch.setattr(
        trigger_module,
        "_build_consolidation_engine",
        FakeConsolidationEngine,
    )

    graph = FakeGraphStore({"episodes": 3, "total_entities": 0})
    service = ConsolidationTriggerService(
        graph_store=graph,
        activation_store=SimpleNamespace(),
        search_index=SimpleNamespace(),
        cfg=ActivationConfig(),
        extractor=None,
        nerve_center_cfg=NerveCenterConfig(level_unlock_autonomous_consolidation=20),
    )

    with pytest.raises(PermissionError, match="Level 20"):
        await service.trigger_consolidation_cycle(
            group_id="brain",
            trigger="pressure",
            dry_run=False,
        )

    assert "run_cycle" not in captured
    assert graph.get_stats_calls == ["brain"]


@pytest.mark.asyncio
async def test_autonomous_pressure_consolidation_runs_at_cortical_unlock(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    cycle = SimpleNamespace(id="cyc_20", status="completed")

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

    graph = FakeGraphStore({"episodes": 3, "total_entities": 950})
    service = ConsolidationTriggerService(
        graph_store=graph,
        activation_store=SimpleNamespace(),
        search_index=SimpleNamespace(),
        cfg=ActivationConfig(),
        extractor=None,
        nerve_center_cfg=NerveCenterConfig(level_unlock_autonomous_consolidation=20),
    )

    result = await service.trigger_consolidation_cycle(
        group_id="brain",
        trigger="pressure",
        dry_run=False,
    )

    assert result.cycle is cycle
    assert result.graph_stats == {"episodes": 3, "total_entities": 950}
    assert captured["run_cycle"] == {
        "group_id": "brain",
        "trigger": "pressure",
        "dry_run": False,
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


def _cycle(cycle_id: str = "cyc_1") -> ConsolidationCycle:
    cycle = ConsolidationCycle(
        id=cycle_id,
        group_id="native_brain",
        trigger="manual",
        dry_run=True,
        status="completed",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="completed",
                items_processed=2,
                items_affected=1,
                duration_ms=12.5,
            )
        ],
    )
    cycle.completed_at = cycle.started_at + 1
    cycle.total_duration_ms = 1000.0
    return cycle


def test_api_consolidation_trigger_surface_handles_running_and_triggered() -> None:
    running = build_api_consolidation_trigger_surface(
        SimpleNamespace(is_running=True),
        group_id="native_brain",
        dry_run=True,
    )
    assert running.status_code == 409
    assert running.payload == {"detail": "A consolidation cycle is already running"}
    assert running.should_run is False

    ready = build_api_consolidation_trigger_surface(
        SimpleNamespace(is_running=False),
        group_id="native_brain",
        dry_run=False,
    )
    assert ready.status_code == 200
    assert ready.payload == {
        "status": "triggered",
        "group_id": "native_brain",
        "dry_run": False,
    }
    assert ready.should_run is True


def test_api_consolidation_trigger_response_surface_schedules_background_cycle() -> None:
    background_tasks = MagicMock()
    logger = MagicMock()
    engine = SimpleNamespace(is_running=False)

    result = build_api_consolidation_trigger_response_surface(
        engine,
        group_id="native_brain",
        dry_run=True,
        background_tasks=background_tasks,
        logger=logger,
    )

    assert result.status_code == 200
    background_tasks.add_task.assert_called_once_with(
        run_api_consolidation_cycle,
        engine,
        group_id="native_brain",
        dry_run=True,
        logger=logger,
    )


def test_api_consolidation_trigger_response_surface_skips_background_when_running() -> None:
    background_tasks = MagicMock()

    result = build_api_consolidation_trigger_response_surface(
        SimpleNamespace(is_running=True),
        group_id="native_brain",
        dry_run=True,
        background_tasks=background_tasks,
    )

    assert result.status_code == 409
    background_tasks.add_task.assert_not_called()


@pytest.mark.asyncio
async def test_run_api_consolidation_cycle_runs_manual_background_cycle() -> None:
    engine = SimpleNamespace(run_cycle=AsyncMock())

    await run_api_consolidation_cycle(
        engine,
        group_id="native_brain",
        dry_run=True,
    )

    engine.run_cycle.assert_awaited_once_with(
        group_id="native_brain",
        trigger="manual",
        dry_run=True,
    )


@pytest.mark.asyncio
async def test_api_consolidation_status_surface_includes_scheduler_pressure_and_latest() -> None:
    engine = SimpleNamespace(
        is_running=True,
        get_latest_cycle=AsyncMock(return_value=_cycle("cyc_status")),
    )
    scheduler = SimpleNamespace(is_active=True)
    pressure = MagicMock()
    pressure.get_snapshot.return_value = SimpleNamespace(
        episodes_since_last=3,
        entities_created=2,
        last_cycle_time=10.0,
    )
    pressure.get_pressure.return_value = 123.456
    cfg = ActivationConfig()

    result = await build_api_consolidation_status_surface(
        engine,
        group_id="native_brain",
        scheduler=scheduler,
        pressure=pressure,
        activation_cfg=cfg,
    )

    assert result["is_running"] is True
    assert result["scheduler_active"] is True
    assert result["pressure"] == {
        "value": 123.46,
        "threshold": cfg.consolidation_pressure_threshold,
        "episodes_since_last": 3,
        "entities_created": 2,
        "last_cycle_time": 10.0,
        # Hygiene debt now folds into pressure; no graph store here -> 0.0.
        "hygiene_debt_pressure": 0.0,
    }
    assert result["latest_cycle"]["id"] == "cyc_status"
    engine.get_latest_cycle.assert_awaited_once_with("native_brain")
    pressure.get_snapshot.assert_called_once_with("native_brain")
    pressure.get_pressure.assert_called_once_with("native_brain", cfg, hygiene_debt=None)


@pytest.mark.asyncio
async def test_api_consolidation_status_surface_degrades_when_latest_cycle_times_out(
    monkeypatch,
) -> None:
    async def slow_latest_cycle(_group_id: str):
        await asyncio.sleep(0.05)
        return _cycle("cyc_late")

    monkeypatch.setattr(
        "engram.consolidation_trigger._API_CONSOLIDATION_STATUS_LATEST_CYCLE_TIMEOUT_SECONDS",
        0.01,
    )
    engine = SimpleNamespace(
        is_running=True,
        get_latest_cycle=AsyncMock(side_effect=slow_latest_cycle),
    )

    result = await build_api_consolidation_status_surface(
        engine,
        group_id="native_brain",
    )

    assert result["is_running"] is True
    assert result["degraded"] is True
    assert result["skip_reason"] == "latest_cycle_timeout"
    assert result["latest_cycle_status"] == "timeout"
    assert "latest_cycle" not in result


@pytest.mark.asyncio
async def test_api_consolidation_status_response_surface_uses_config_activation() -> None:
    engine = SimpleNamespace(
        is_running=False,
        get_latest_cycle=AsyncMock(return_value=None),
    )
    scheduler = SimpleNamespace(is_active=False)
    pressure = MagicMock()
    pressure.get_snapshot.return_value = SimpleNamespace(
        episodes_since_last=1,
        entities_created=2,
        last_cycle_time=3.0,
    )
    pressure.get_pressure.return_value = 7.0
    cfg = SimpleNamespace(activation=ActivationConfig())

    result = await build_api_consolidation_status_response_surface(
        engine,
        group_id="native_brain",
        scheduler=scheduler,
        pressure=pressure,
        config=cfg,
    )

    assert result["pressure"]["threshold"] == cfg.activation.consolidation_pressure_threshold
    pressure.get_pressure.assert_called_once_with("native_brain", cfg.activation, hygiene_debt=None)


@pytest.mark.asyncio
async def test_api_consolidation_status_response_surface_omits_pressure_without_config() -> None:
    engine = SimpleNamespace(
        is_running=False,
        get_latest_cycle=AsyncMock(return_value=None),
    )
    pressure = MagicMock()

    result = await build_api_consolidation_status_response_surface(
        engine,
        group_id="native_brain",
        pressure=pressure,
        config=None,
    )

    assert "pressure" not in result
    pressure.get_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_api_consolidation_history_surface_serializes_cycles() -> None:
    engine = SimpleNamespace(get_recent_cycles=AsyncMock(return_value=[_cycle("cyc_hist")]))

    result = await build_api_consolidation_history_surface(
        engine,
        group_id="native_brain",
        limit=5,
    )

    assert result["cycles"][0]["id"] == "cyc_hist"
    assert result["cycles"][0]["summary"]["total_processed"] == 2
    engine.get_recent_cycles.assert_awaited_once_with("native_brain", limit=5)


@pytest.mark.asyncio
async def test_api_consolidation_cycle_detail_surface_statuses() -> None:
    unavailable = SimpleNamespace(audit_store_available=False)
    unavailable_result = await build_api_consolidation_cycle_detail_surface(
        unavailable,
        group_id="native_brain",
        cycle_id="cyc_missing",
    )
    assert unavailable_result.status_code == 404
    assert unavailable_result.payload == {"detail": "Consolidation store not available"}

    missing = SimpleNamespace(
        audit_store_available=True,
        get_cycle_detail=AsyncMock(return_value=None),
    )
    missing_result = await build_api_consolidation_cycle_detail_surface(
        missing,
        group_id="native_brain",
        cycle_id="cyc_missing",
    )
    assert missing_result.status_code == 404
    assert missing_result.payload == {"detail": "Cycle not found"}

    detail = ConsolidationCycleDetail(cycle=_cycle("cyc_detail"))
    engine = SimpleNamespace(
        audit_store_available=True,
        get_cycle_detail=AsyncMock(return_value=detail),
    )
    detail_result = await build_api_consolidation_cycle_detail_surface(
        engine,
        group_id="native_brain",
        cycle_id="cyc_detail",
    )
    assert detail_result.status_code == 200
    assert detail_result.payload["id"] == "cyc_detail"
    assert detail_result.payload["summary"]["total_affected"] == 1
    engine.get_cycle_detail.assert_awaited_once_with("cyc_detail", "native_brain")


@pytest.mark.asyncio
async def test_mcp_consolidation_status_surface_includes_latest_cycle() -> None:
    cycle = SimpleNamespace(
        id="cyc_status",
        status="failed",
        error="calibration failed",
        dry_run=True,
        trigger="mcp",
        started_at=1.0,
        completed_at=2.0,
        total_duration_ms=7.5,
        phase_results=[],
    )
    store = MagicMock()
    store.get_recent_cycles = AsyncMock(return_value=[cycle])

    result = await build_mcp_consolidation_status_surface(
        store,
        group_id="native_brain",
    )

    assert result["is_running"] is False
    assert result["latest_cycle"]["id"] == "cyc_status"
    assert result["latest_cycle"]["error"] == "calibration failed"
    store.get_recent_cycles.assert_awaited_once_with("native_brain", limit=1)


@pytest.mark.asyncio
async def test_mcp_consolidation_trigger_surface_formats_cycle_payload() -> None:
    cycle = SimpleNamespace(
        id="cyc_1",
        status="completed",
        error=None,
        dry_run=True,
        trigger="mcp",
        started_at=1.0,
        completed_at=2.0,
        total_duration_ms=7.5,
        phase_results=[],
    )
    store = SimpleNamespace(name="active-store")
    manager = MagicMock()
    manager.trigger_consolidation_cycle = AsyncMock(
        return_value=ConsolidationTriggerResult(
            cycle=cycle,
            graph_stats={"episodes": 3},
        )
    )

    result = await build_mcp_consolidation_trigger_surface(
        manager,
        group_id="native_brain",
        dry_run=True,
        consolidation_store=store,
    )

    assert result["cycle_id"] == "cyc_1"
    assert "id" not in result
    assert result["graph_stats"] == {"episodes": 3}
    manager.trigger_consolidation_cycle.assert_awaited_once_with(
        group_id="native_brain",
        trigger="mcp",
        dry_run=True,
        consolidation_store=store,
    )


@pytest.mark.asyncio
async def test_resolve_mcp_consolidation_trigger_store_prefers_active_store() -> None:
    manager = MagicMock()
    manager.get_consolidation_shared_db.return_value = object()
    active_store = SimpleNamespace(name="active")

    result = await resolve_mcp_consolidation_trigger_store(manager, active_store)

    assert result is active_store
    manager.get_consolidation_shared_db.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_mcp_consolidation_trigger_store_uses_shared_db(monkeypatch) -> None:
    initialized: dict[str, object] = {}

    class FakeSQLiteConsolidationStore:
        def __init__(self, path: str) -> None:
            self.path = path

        async def initialize(self, *, db) -> None:
            initialized["db"] = db

    monkeypatch.setattr(
        "engram.consolidation.store.SQLiteConsolidationStore",
        FakeSQLiteConsolidationStore,
    )
    db = object()
    manager = MagicMock()
    manager.get_consolidation_shared_db.return_value = db

    result = await resolve_mcp_consolidation_trigger_store(manager, None)

    assert isinstance(result, FakeSQLiteConsolidationStore)
    assert result.path == ":memory:"
    assert initialized["db"] is db
    manager.get_consolidation_shared_db.assert_called_once_with()
