from types import SimpleNamespace

from engram.consolidation.events import ConsolidationEventPublisher
from engram.consolidation.lifecycle import build_cycle_plan
from engram.events.bus import EventBus
from engram.models.consolidation import ConsolidationCycle, PhaseResult


def _drain(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_event_publisher_emits_lifecycle_contract_payloads():
    bus = EventBus()
    queue = bus.subscribe("test")
    publisher = ConsolidationEventPublisher(bus)
    cycle = ConsolidationCycle(
        group_id="test",
        trigger="manual",
        dry_run=False,
        status="running",
        id="cyc_test",
    )
    cycle_plan = build_cycle_plan(
        cycle=cycle,
        phases=[SimpleNamespace(name="merge")],
    )
    phase_plan = cycle_plan.phase_plan("merge")
    result = PhaseResult(
        phase="merge",
        status="success",
        items_processed=4,
        items_affected=1,
        duration_ms=12.5,
    )

    publisher.cycle_started("test", cycle_plan)
    publisher.phase_started("test", cycle_id=cycle.id, phase_plan=phase_plan)
    publisher.phase_completed(
        "test",
        cycle_id=cycle.id,
        result=result,
        phase_plan=phase_plan,
    )
    publisher.graph_delta(
        "test",
        removed_node_ids=("ent_drop",),
        dry_run=False,
    )
    cycle.status = "completed"
    cycle.phase_results = [result]
    cycle.total_duration_ms = 30.0
    publisher.cycle_completed(
        "test",
        cycle,
        finalization={"refreshedPinnedContexts": 2},
    )

    events = _drain(queue)
    assert [event["type"] for event in events] == [
        "consolidation.started",
        "consolidation.phase.merge.started",
        "consolidation.phase.merge.completed",
        "graph.delta",
        "consolidation.completed",
    ]
    assert events[0]["payload"] == {
        "cycle_id": "cyc_test",
        "dry_run": False,
        "trigger": "manual",
        "lifecycleStage": "consolidate",
        "phaseCount": 1,
        "phases": ["merge"],
    }
    assert events[1]["payload"] == {
        "cycle_id": "cyc_test",
        "phase": "merge",
        "phaseOrdinal": 0,
        "lifecycleStage": "consolidate",
    }
    assert events[2]["payload"] == {
        "cycle_id": "cyc_test",
        "phase": "merge",
        "status": "success",
        "items_processed": 4,
        "items_affected": 1,
        "duration_ms": 12.5,
        "lifecycleStage": "consolidate",
        "phaseOrdinal": 0,
    }
    assert events[3]["payload"] == {"nodesRemoved": ["ent_drop"]}
    assert events[4]["payload"] == {
        "cycle_id": "cyc_test",
        "status": "completed",
        "duration_ms": 30.0,
        "phases": 1,
        "trigger": "manual",
        "dry_run": False,
        "lifecycleStage": "consolidate",
        "finalization": {"refreshedPinnedContexts": 2},
    }


def test_event_publisher_emits_failed_phase_and_learning_updates():
    bus = EventBus()
    queue = bus.subscribe("test")
    publisher = ConsolidationEventPublisher(bus)
    cycle = ConsolidationCycle(group_id="test", id="cyc_test")
    cycle_plan = build_cycle_plan(
        cycle=cycle,
        phases=[SimpleNamespace(name="merge")],
    )
    phase_plan = cycle_plan.phase_plan("merge")

    publisher.phase_failed(
        "test",
        cycle_id=cycle.id,
        result=PhaseResult(phase="merge", status="error", error="boom"),
        phase_plan=phase_plan,
    )
    publisher.learning_updated(
        "test",
        cycle_id=cycle.id,
        distillation_examples=2,
        calibration_snapshots=1,
    )

    events = _drain(queue)
    assert [event["type"] for event in events] == [
        "consolidation.phase.merge.failed",
        "consolidation.learning.updated",
    ]
    assert events[0]["payload"] == {
        "cycle_id": "cyc_test",
        "phase": "merge",
        "status": "error",
        "items_processed": 0,
        "items_affected": 0,
        "duration_ms": 0.0,
        "lifecycleStage": "consolidate",
        "phaseOrdinal": 0,
        "error": "boom",
    }
    assert events[1]["payload"] == {
        "cycle_id": "cyc_test",
        "distillation_examples": 2,
        "calibration_snapshots": 1,
    }


def test_event_publisher_skips_empty_deltas_and_learning_updates():
    bus = EventBus()
    queue = bus.subscribe("test")
    publisher = ConsolidationEventPublisher(bus)

    publisher.graph_delta("test", removed_node_ids=("ent_drop",), dry_run=True)
    publisher.graph_delta("test", removed_node_ids=(), dry_run=False)
    publisher.learning_updated(
        "test",
        cycle_id="cyc_test",
        distillation_examples=0,
        calibration_snapshots=0,
    )
    ConsolidationEventPublisher().cycle_completed(
        "test",
        ConsolidationCycle(group_id="test", id="cyc_test"),
    )

    assert _drain(queue) == []
