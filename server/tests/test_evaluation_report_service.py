from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.evaluation.report_service import (
    build_api_brain_loop_evaluation_surface,
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
async def test_build_brain_loop_evaluation_surface_prefers_stats_only_reader() -> None:
    graph_state_called = False

    class Manager:
        async def get_graph_stats(self, group_id: str) -> dict:
            assert group_id == "brain_a"
            return {
                "episodes": 2,
                "entities": 1,
                "relationships": 0,
                "cue_metrics": {
                    "cue_count": 2,
                    "cue_surfaced_count": 1,
                    "cue_used_count": 1,
                },
                "projection_metrics": {
                    "state_counts": {"projected": 2},
                    "yield": {
                        "avg_linked_entities_per_projected_episode": 1.5,
                    },
                },
                "recall_metrics": {"total_analyses": 1, "trigger_count": 1},
            }

        async def get_graph_state(self, **_kwargs):
            nonlocal graph_state_called
            graph_state_called = True
            return {"stats": {}}

    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        Manager(),
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=50,
        snapshot_source="rest_report",
    )

    assert graph_state_called is False
    assert report["totals"]["episodes"] == 2
    assert report["evaluation_signals"]["cue_usefulness"]["status"] == "measured"
    assert report["evaluation_signals"]["projection_yield"]["status"] == "measured"
    assert report["evaluation_signals"]["projection_yield"]["metric"] == 1.5


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
async def test_build_brain_loop_evaluation_surface_persists_live_memory_operation_metrics() -> None:
    manager = AsyncMock()
    manager.get_graph_state.return_value = {
        "stats": {
            "episodes": {"total": 1},
            "entities": {"total": 2, "active": 1},
            "relationships": {"total": 3},
            "recall_metrics": {},
            "memory_operation_metrics": {
                "operation_count": 3,
                "duration_ms": {"avg": 5.0, "p95": 13.0},
            },
        }
    }
    store = AsyncMock()
    store.get_latest_recall_metrics_snapshot.return_value = {}
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

    assert report["memory_value"]["cost"]["operation_count"] == 3
    assert report["memory_value"]["cost"]["p95_added_latency_ms"] == 13.0
    store.save_memory_operation_metrics_snapshot.assert_awaited_once()
    snapshot = store.save_memory_operation_metrics_snapshot.await_args.args[0]
    assert snapshot.group_id == "brain_a"
    assert snapshot.metrics["operation_count"] == 3
    assert snapshot.source == "rest_report"
    store.get_latest_memory_operation_metrics_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_brain_loop_evaluation_surface_uses_saved_memory_operation_metrics() -> None:
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
    store.get_latest_recall_metrics_snapshot.return_value = {}
    store.get_latest_memory_operation_metrics_snapshot.return_value = {
        "operation_count": 7,
        "duration_ms": {"avg": 8.0, "p95": 17.0},
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

    assert report["memory_value"]["cost"]["operation_count"] == 7
    assert report["memory_value"]["cost"]["p95_added_latency_ms"] == 17.0
    store.save_memory_operation_metrics_snapshot.assert_not_awaited()
    store.get_latest_memory_operation_metrics_snapshot.assert_awaited_once_with("brain_a")


@pytest.mark.asyncio
async def test_brain_loop_surface_can_skip_saved_memory_operation_metrics() -> None:
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
    store.get_latest_recall_metrics_snapshot.return_value = {}
    store.get_latest_memory_operation_metrics_snapshot.return_value = {
        "operation_count": 7,
        "duration_ms": {"avg": 8.0, "p95": 17.0},
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
        snapshot_source="rest_report",
        merge_saved_memory_operation_metrics=False,
    )

    assert report["memory_value"]["cost"]["operation_count"] == 0
    assert report["memory_value"]["cost"]["status"] == "needs_samples"
    store.save_memory_operation_metrics_snapshot.assert_not_awaited()
    store.get_latest_memory_operation_metrics_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_live_cost_report_skips_graph_state_scan() -> None:
    graph_state_called = False

    async def get_graph_state(**_kwargs):
        nonlocal graph_state_called
        graph_state_called = True
        return {"stats": {}}

    manager = SimpleNamespace(
        get_graph_state=get_graph_state,
        get_recall_metrics=lambda _group_id: {"total_analyses": 1, "trigger_count": 1},
        get_memory_operation_metrics=lambda _group_id: {
            "operation_count": 4,
            "duration_ms": {"avg": 12.0, "p95": 32.0},
        },
        get_memory_packet_cache_summary=lambda _group_id: {"entry_count": 2},
    )
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []
    engine = AsyncMock()
    engine.get_recent_evaluation_context.return_value = ([], [])

    report = await build_api_brain_loop_evaluation_surface(
        manager,
        store,
        engine,
        group_id="brain_a",
        cycle_limit=1,
        sample_limit=1,
        live_memory_operation_cost=True,
    )

    assert graph_state_called is False
    assert report["degraded"] is True
    assert report["degradations"] == [
        {
            "stage": "graph_state",
            "status": "skipped",
            "skip_reason": "live_cost_runtime_only",
        }
    ]
    assert report["memory_value"]["cost"]["operation_count"] == 4
    assert report["memory_value"]["cost"]["p95_added_latency_ms"] == 32.0
    engine.get_recent_evaluation_context.assert_not_awaited()
    store.save_memory_operation_metrics_snapshot.assert_awaited_once()
    store.get_latest_memory_operation_metrics_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_live_cost_report_uses_cached_graph_stats_without_scan() -> None:
    graph_state_called = False

    async def get_graph_state(**_kwargs):
        nonlocal graph_state_called
        graph_state_called = True
        return {"stats": {}}

    manager = SimpleNamespace(
        get_graph_state=get_graph_state,
        get_cached_graph_stats=lambda _group_id: {
            "episodes": 12,
            "entities": 7,
            "relationships": 5,
        },
        get_recall_metrics=lambda _group_id: {"total_analyses": 1, "trigger_count": 1},
        get_memory_operation_metrics=lambda _group_id: {
            "operation_count": 4,
            "duration_ms": {"avg": 12.0, "p95": 32.0},
        },
        get_memory_packet_cache_summary=lambda _group_id: {"entry_count": 2},
    )
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []
    engine = AsyncMock()
    engine.get_recent_evaluation_context.return_value = ([], [])

    report = await build_api_brain_loop_evaluation_surface(
        manager,
        store,
        engine,
        group_id="brain_a",
        cycle_limit=1,
        sample_limit=1,
        live_memory_operation_cost=True,
    )

    assert graph_state_called is False
    assert "degraded" not in report
    assert report["totals"] == {
        "episodes": 12,
        "entities": 7,
        "relationships": 5,
        "active_entities": 0,
    }
    assert report["memory_value"]["cost"]["operation_count"] == 4
    assert report["memory_value"]["cost"]["p95_added_latency_ms"] == 32.0
    engine.get_recent_evaluation_context.assert_not_awaited()
    store.save_memory_operation_metrics_snapshot.assert_awaited_once()
    store.get_latest_memory_operation_metrics_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_brain_loop_evaluation_surface_degrades_on_graph_state_timeout() -> None:
    async def slow_graph_state(**_kwargs):
        await asyncio.sleep(1)
        return {"stats": {}}

    manager = SimpleNamespace(
        get_graph_state=slow_graph_state,
        get_recall_metrics=lambda _group_id: {"total_analyses": 1, "trigger_count": 1},
        get_memory_operation_metrics=lambda _group_id: {
            "operation_count": 2,
            "duration_ms": {"avg": 20.0, "p95": 44.0},
        },
        get_memory_packet_cache_summary=lambda _group_id: {"entry_count": 3},
    )
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=25,
        snapshot_source="rest_report",
        graph_state_timeout_seconds=0.01,
    )

    assert report["degraded"] is True
    assert report["degradations"] == [
        {
            "stage": "graph_state",
            "status": "degraded",
            "skip_reason": "graph_state_timeout",
            "timeout_ms": 10,
        }
    ]
    assert report["recall"]["total_analyses"] == 1
    assert report["memory_value"]["cost"]["operation_count"] == 2
    assert report["memory_value"]["cost"]["p95_added_latency_ms"] == 44.0
    store.save_recall_metrics_snapshot.assert_awaited_once()
    store.save_memory_operation_metrics_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_brain_loop_surface_marks_runtime_only_graph_stats_degraded() -> None:
    class Manager:
        async def get_graph_stats(self, group_id: str) -> dict:
            assert group_id == "brain_a"
            return {
                "recall_metrics": {"total_analyses": 1, "trigger_count": 1},
                "memory_operation_metrics": {
                    "operation_count": 2,
                    "duration_ms": {"avg": 20.0, "p95": 44.0},
                },
            }

    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        Manager(),
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=25,
        snapshot_source="rest_report",
    )

    assert report["degraded"] is True
    assert report["degradations"] == [
        {
            "stage": "graph_state",
            "status": "degraded",
            "skip_reason": "graph_state_unavailable",
        }
    ]
    assert report["totals"] == {
        "episodes": 0,
        "entities": 0,
        "relationships": 0,
        "active_entities": 0,
    }
    assert report["recall"]["total_analyses"] == 1
    assert report["memory_value"]["cost"]["operation_count"] == 2


@pytest.mark.asyncio
async def test_brain_loop_surface_uses_cached_graph_stats_on_timeout() -> None:
    async def slow_graph_stats(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return {"stats": {}}

    cached_calls = 0

    def get_cached_graph_stats(_group_id):
        nonlocal cached_calls
        cached_calls += 1
        if cached_calls == 1:
            return None
        return {
            "episodes": 3,
            "entities": 2,
            "relationships": 1,
            "cue_metrics": {
                "cue_count": 3,
                "cue_surfaced_count": 2,
                "cue_used_count": 1,
            },
            "projection_metrics": {
                "state_counts": {"projected": 3},
                "yield": {"avg_linked_entities_per_projected_episode": 2.0},
            },
        }

    manager = SimpleNamespace(
        get_graph_stats=slow_graph_stats,
        get_cached_graph_stats=get_cached_graph_stats,
        get_recall_metrics=lambda _group_id: {"total_analyses": 1, "trigger_count": 1},
        get_memory_operation_metrics=lambda _group_id: {
            "operation_count": 2,
            "duration_ms": {"avg": 20.0, "p95": 44.0},
        },
        get_memory_packet_cache_summary=lambda _group_id: {"entry_count": 3},
    )
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=25,
        snapshot_source="rest_report",
        graph_state_timeout_seconds=0.01,
    )

    assert report["degraded"] is True
    assert report["totals"]["episodes"] == 3
    assert report["evaluation_signals"]["cue_usefulness"]["status"] == "measured"
    assert report["evaluation_signals"]["projection_yield"]["status"] == "measured"
    assert report["recall"]["total_analyses"] == 1
    assert report["memory_value"]["cost"]["operation_count"] == 2
    await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_brain_loop_surface_uses_fresh_cached_graph_stats_without_refresh() -> None:
    async def unexpected_graph_stats(*_args, **_kwargs):
        raise AssertionError("cache hit should avoid graph stats refresh")

    manager = SimpleNamespace(
        get_graph_stats=unexpected_graph_stats,
        get_cached_graph_stats=lambda _group_id: {
            "episodes": 3,
            "entities": 2,
            "relationships": 1,
            "cue_metrics": {
                "cue_count": 3,
                "cue_surfaced_count": 2,
                "cue_used_count": 1,
            },
            "projection_metrics": {
                "state_counts": {"projected": 3},
                "yield": {"avg_linked_entities_per_projected_episode": 2.0},
            },
        },
        get_recall_metrics=lambda _group_id: {"total_analyses": 1, "trigger_count": 1},
        get_memory_operation_metrics=lambda _group_id: {
            "operation_count": 2,
            "duration_ms": {"avg": 20.0, "p95": 44.0},
        },
        get_memory_packet_cache_summary=lambda _group_id: {"entry_count": 3},
    )
    store = AsyncMock()
    store.get_recall_samples.return_value = []
    store.get_session_samples.return_value = []

    report = await build_brain_loop_evaluation_surface(
        manager,
        store,
        group_id="brain_a",
        recent_cycles=[],
        calibration_snapshots=[],
        sample_limit=25,
        snapshot_source="rest_report",
        graph_state_timeout_seconds=0.01,
    )

    assert report.get("degraded") is not True
    assert report["totals"]["episodes"] == 3
    assert report["evaluation_signals"]["projection_yield"]["status"] == "measured"


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
async def test_api_evaluation_report_surface_loads_engine_context() -> None:
    from engram.evaluation import report_service as report_module

    report_module._CONSOLIDATION_CONTEXT_CACHE.clear()
    report_module._CONSOLIDATION_CONTEXT_TASKS.clear()
    report_module._CONSOLIDATION_CONTEXT_TASK_STARTED_AT.clear()

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
    engine = AsyncMock()
    engine.get_recent_evaluation_context.return_value = (
        [SimpleNamespace(id="cyc_1", phase_results=[])],
        ["snapshot"],
    )

    report = await build_api_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        engine,
        group_id="brain_a",
        cycle_limit=0,
        sample_limit=0,
    )

    assert report["group_id"] == "brain_a"
    engine.get_recent_evaluation_context.assert_awaited_once_with(
        "brain_a",
        cycle_limit=1,
    )
    evaluation_store.get_recall_samples.assert_awaited_once_with("brain_a", limit=1)
    evaluation_store.get_session_samples.assert_awaited_once_with("brain_a", limit=1)


@pytest.mark.asyncio
async def test_api_evaluation_report_uses_cached_consolidation_context_after_timeout(
    monkeypatch,
) -> None:
    from engram.evaluation import report_service as report_module

    report_module._CONSOLIDATION_CONTEXT_CACHE.clear()
    report_module._CONSOLIDATION_CONTEXT_TASKS.clear()
    report_module._CONSOLIDATION_CONTEXT_TASK_STARTED_AT.clear()
    monkeypatch.setattr(report_module, "REPORT_CONTEXT_TIMEOUT_SECONDS", 0.01)

    manager = SimpleNamespace(
        get_cached_graph_stats=lambda _group_id: {
            "episodes": 1,
            "entities": 1,
            "relationships": 0,
            "cue_metrics": {"cue_count": 1, "cue_surfaced_count": 1},
            "projection_metrics": {"state_counts": {"projected": 1}},
        },
        get_recall_metrics=lambda _group_id: {},
        get_memory_operation_metrics=lambda _group_id: {},
        get_memory_packet_cache_summary=lambda _group_id: {},
    )
    evaluation_store = AsyncMock()
    evaluation_store.get_latest_recall_metrics_snapshot.return_value = {}
    evaluation_store.get_latest_memory_operation_metrics_snapshot.return_value = {}
    evaluation_store.get_recall_samples.return_value = []
    evaluation_store.get_session_samples.return_value = []

    async def slow_context(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return ([SimpleNamespace(id="cyc_cached", phase_results=[])], ["snapshot"])

    engine = SimpleNamespace(get_recent_evaluation_context=AsyncMock(side_effect=slow_context))

    first = await build_api_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        engine,
        group_id="brain_cached",
        cycle_limit=1,
        sample_limit=1,
    )
    assert first["degradations"][0]["skip_reason"] == "evaluation_context_timeout"

    await asyncio.sleep(0.06)
    second = await build_api_brain_loop_evaluation_surface(
        manager,
        evaluation_store,
        engine,
        group_id="brain_cached",
        cycle_limit=1,
        sample_limit=1,
    )

    assert second.get("degraded") is not True
    assert second["consolidate"]["cycle_count"] == 1
    engine.get_recent_evaluation_context.assert_awaited_once()


@pytest.mark.asyncio
async def test_api_evaluation_report_replaces_stale_consolidation_context_task(
    monkeypatch,
) -> None:
    from engram.evaluation import report_service as report_module

    report_module._CONSOLIDATION_CONTEXT_CACHE.clear()
    report_module._CONSOLIDATION_CONTEXT_TASKS.clear()
    report_module._CONSOLIDATION_CONTEXT_TASK_STARTED_AT.clear()
    monkeypatch.setattr(report_module, "REPORT_CONTEXT_TASK_MAX_SECONDS", 0.01)

    async def never_finishes():
        await asyncio.sleep(3600)
        return [], []

    stale_task = asyncio.create_task(never_finishes())
    cache_key = ("rest_report", "brain_stale")
    report_module._CONSOLIDATION_CONTEXT_TASKS[cache_key] = stale_task
    report_module._CONSOLIDATION_CONTEXT_TASK_STARTED_AT[cache_key] = (
        asyncio.get_running_loop().time() - 1
    )

    try:
        fresh_task = report_module._warm_consolidation_context(
            cache_key,
            lambda: asyncio.sleep(
                0,
                result=([SimpleNamespace(id="cyc_fresh", phase_results=[])], []),
            ),
        )
        await asyncio.sleep(0)

        assert fresh_task is not stale_task
        assert stale_task.cancelled()
        cycles, snapshots = await fresh_task
        assert [cycle.id for cycle in cycles] == ["cyc_fresh"]
        assert snapshots == []
        cached_cycles, cached_snapshots = report_module._cached_consolidation_context(cache_key)
        assert cached_cycles[0].id == "cyc_fresh"
        assert cached_snapshots == []
    finally:
        stale_task.cancel()


@pytest.mark.asyncio
async def test_mcp_evaluation_report_surface_uses_active_consolidation_store() -> None:
    from engram.evaluation import report_service as report_module

    report_module._CONSOLIDATION_CONTEXT_CACHE.clear()
    report_module._CONSOLIDATION_CONTEXT_TASKS.clear()
    report_module._CONSOLIDATION_CONTEXT_TASK_STARTED_AT.clear()

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
