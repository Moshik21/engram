"""Tests for consolidation cycle completion orchestration."""

from __future__ import annotations

from engram.consolidation.completion import ConsolidationCycleCompletionService
from engram.consolidation.events import ConsolidationEventPublisher
from engram.consolidation.finalization import ConsolidationFinalizationResult
from engram.consolidation.learning import ConsolidationLearningResult
from engram.events.bus import EventBus
from engram.models.consolidation import ConsolidationCycle, CycleContext


class FakeCompletionStore:
    def __init__(self) -> None:
        self.updated_cycles: list[ConsolidationCycle] = []

    async def update_cycle(self, cycle: ConsolidationCycle) -> None:
        self.updated_cycles.append(cycle)


class FakeLearningService:
    def __init__(self) -> None:
        self.calls: list[tuple[ConsolidationCycle, CycleContext]] = []

    async def analyze_cycle(
        self,
        cycle: ConsolidationCycle,
        context: CycleContext,
    ) -> ConsolidationLearningResult:
        self.calls.append((cycle, context))
        return ConsolidationLearningResult(
            distillation_examples=1,
            calibration_snapshots=1,
        )


class FakeFinalizationService:
    def __init__(self) -> None:
        self.groups: list[str] = []

    async def refresh_after_cycle(self, group_id: str) -> ConsolidationFinalizationResult:
        self.groups.append(group_id)
        return ConsolidationFinalizationResult(refreshed_pinned_contexts=2)


async def test_completion_service_updates_store_learning_finalization_and_events():
    bus = EventBus()
    queue = bus.subscribe("test")
    store = FakeCompletionStore()
    learning = FakeLearningService()
    finalization = FakeFinalizationService()
    service = ConsolidationCycleCompletionService(
        consolidation_store=store,
        learning_service=learning,
        finalization_service=finalization,
        event_publisher=ConsolidationEventPublisher(bus),
    )
    cycle = ConsolidationCycle(
        id="cyc_completion",
        group_id="test",
        status="completed",
        started_at=1.0,
    )
    context = CycleContext(trigger="manual")

    await service.complete_cycle(group_id="test", cycle=cycle, context=context)

    assert cycle.completed_at is not None
    assert cycle.total_duration_ms is not None
    assert store.updated_cycles == [cycle]
    assert learning.calls == [(cycle, context)]
    assert finalization.groups == ["test"]

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event["type"] for event in events] == [
        "consolidation.learning.updated",
        "consolidation.completed",
    ]
    assert events[-1]["payload"]["finalization"] == {"refreshedPinnedContexts": 2}


async def test_completion_service_skips_finalization_for_failed_cycle():
    bus = EventBus()
    queue = bus.subscribe("test")
    finalization = FakeFinalizationService()
    service = ConsolidationCycleCompletionService(
        consolidation_store=None,
        learning_service=FakeLearningService(),
        finalization_service=finalization,
        event_publisher=ConsolidationEventPublisher(bus),
    )
    cycle = ConsolidationCycle(
        id="cyc_failed_completion",
        group_id="test",
        status="failed",
        error="capability failed",
        started_at=1.0,
    )

    await service.complete_cycle(
        group_id="test",
        cycle=cycle,
        context=CycleContext(trigger="manual"),
    )

    assert finalization.groups == []
    completed = queue.get_nowait()
    assert completed["type"] == "consolidation.completed"
    assert completed["payload"]["status"] == "failed"
    assert completed["payload"]["error"] == "capability failed"
