from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.evaluation.report_service import (
    build_brain_loop_evaluation_surface,
    build_mcp_evaluation_report_surface,
    load_consolidation_evaluation_inputs,
)


@pytest.mark.asyncio
async def test_build_brain_loop_evaluation_surface_persists_live_recall_metrics() -> None:
    manager = AsyncMock()
    manager.get_graph_state.return_value = {
        "stats": {
            "episodes": {"total": 1},
            "entities": {"total": 2, "active": 1},
            "relationships": {"total": 3},
            "recall_metrics": {"total_analyses": 5, "trigger_count": 2},
        }
    }
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=50,
        snapshot_source="rest_report",
    )

    assert report["group_id"] == "brain_a"
    assert report["recall"]["total_analyses"] == 5
    store.save_recall_metrics_snapshot.assert_awaited_once()
    snapshot = store.save_recall_metrics_snapshot.await_args.args[0]
    assert snapshot.group_id == "brain_a"
    assert snapshot.metrics["total_analyses"] == 5
    assert snapshot.source == "rest_report"
    store.get_latest_recall_metrics_snapshot.assert_not_awaited()
    store.get_recall_samples.assert_awaited_once_with("brain_a", limit=50)
    store.get_session_samples.assert_awaited_once_with("brain_a", limit=50)


@pytest.mark.asyncio
async def test_build_brain_loop_evaluation_surface_uses_saved_runtime_metrics() -> None:
    manager = AsyncMock()
    manager.get_graph_state.return_value = {
        "stats": {
            "episodes": {"total": 1},
            "entities": {"total": 2, "active": 1},
            "relationships": {"total": 3},
            "recall_metrics": {},
        }
    }
    store = AsyncMock()
    store.get_latest_recall_metrics_snapshot.return_value = {
        "total_analyses": 7,
        "trigger_count": 3,
    }
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=25,
        snapshot_source="mcp_report",
    )

    assert report["recall"]["total_analyses"] == 7
    assert report["recall"]["trigger_count"] == 3
    store.save_recall_metrics_snapshot.assert_not_awaited()
    store.get_latest_recall_metrics_snapshot.assert_awaited_once_with("brain_a")
    store.get_recall_samples.assert_awaited_once_with("brain_a", limit=25)
    store.get_session_samples.assert_awaited_once_with("brain_a", limit=25)


@pytest.mark.asyncio
async def test_load_consolidation_evaluation_inputs_reads_cycle_snapshots() -> None:
    cycle = SimpleNamespace(id="cyc_1", phase_results=[])
    store = AsyncMock()
    store.get_recent_cycles.return_value = [cycle]
    store.get_calibration_snapshots.return_value = ["snapshot"]

    cycles, snapshots = await load_consolidation_evaluation_inputs(
        store,
        group_id="brain_a",
        cycle_limit=0,
    )

    assert cycles == [cycle]
    assert snapshots == ["snapshot"]
    store.get_recent_cycles.assert_awaited_once_with("brain_a", limit=1)
    store.get_calibration_snapshots.assert_awaited_once_with("cyc_1", "brain_a")


@pytest.mark.asyncio
async def test_mcp_evaluation_report_surface_uses_active_consolidation_store() -> None:
    cycle = SimpleNamespace(id="cyc_1", status="completed", phase_results=[])
    manager = AsyncMock()
    manager.get_graph_state.return_value = {
        "stats": {
            "episodes": {"total": 1},
            "entities": {"total": 2, "active": 1},
            "relationships": {"total": 3},
            "recall_metrics": {},
        }
    }
    evaluation_store = AsyncMock()
    evaluation_store.get_latest_recall_metrics_snapshot.return_value = {}
    evaluation_store.get_recall_samples.return_value = []
    evaluation_store.get_session_samples.return_value = []
    consolidation_store = AsyncMock()
    consolidation_store.get_recent_cycles.return_value = [cycle]
    consolidation_store.get_calibration_snapshots.return_value = ["snapshot"]

    report = await build_mcp_evaluation_report_surface(
        manager,
        evaluation_store,
        consolidation_store=consolidation_store,
        group_id="brain_a",
        cycle_limit=2,
        sample_limit=0,
    )

    assert report["group_id"] == "brain_a"
    consolidation_store.get_recent_cycles.assert_awaited_once_with("brain_a", limit=2)
    evaluation_store.get_recall_samples.assert_awaited_once_with("brain_a", limit=1)
