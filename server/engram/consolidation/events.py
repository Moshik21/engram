"""Event publishing boundary for consolidation lifecycle updates."""

from __future__ import annotations

from engram.consolidation.lifecycle import (
    ConsolidationCycleLifecycleResult,
    ConsolidationCyclePlan,
    ConsolidationPhaseLifecycleResult,
    ConsolidationPhasePlan,
)
from engram.events.bus import EventBus
from engram.models.consolidation import ConsolidationCycle, PhaseResult


class ConsolidationEventPublisher:
    """Publish consolidation lifecycle events from typed contracts."""

    def __init__(self, event_bus: EventBus | None = None) -> None:
        self._event_bus = event_bus

    def cycle_started(self, group_id: str, cycle_plan: ConsolidationCyclePlan) -> None:
        self._publish(
            group_id,
            "consolidation.started",
            cycle_plan.started_payload(),
        )

    def phase_started(
        self,
        group_id: str,
        *,
        cycle_id: str,
        phase_plan: ConsolidationPhasePlan,
    ) -> None:
        self._publish(
            group_id,
            f"consolidation.phase.{phase_plan.name}.started",
            {
                "cycle_id": cycle_id,
                "phase": phase_plan.name,
                "phaseOrdinal": phase_plan.ordinal,
                "lifecycleStage": "consolidate",
            },
        )

    def phase_completed(
        self,
        group_id: str,
        *,
        cycle_id: str,
        result: PhaseResult,
        phase_plan: ConsolidationPhasePlan,
    ) -> None:
        self._publish(
            group_id,
            f"consolidation.phase.{phase_plan.name}.completed",
            ConsolidationPhaseLifecycleResult.from_phase_result(
                cycle_id,
                result,
                ordinal=phase_plan.ordinal,
            ).event_payload(),
        )

    def phase_failed(
        self,
        group_id: str,
        *,
        cycle_id: str,
        result: PhaseResult,
        phase_plan: ConsolidationPhasePlan,
    ) -> None:
        self._publish(
            group_id,
            f"consolidation.phase.{phase_plan.name}.failed",
            ConsolidationPhaseLifecycleResult.from_phase_result(
                cycle_id,
                result,
                ordinal=phase_plan.ordinal,
            ).event_payload(),
        )

    def graph_delta(
        self,
        group_id: str,
        *,
        removed_node_ids: tuple[str, ...],
        dry_run: bool,
    ) -> None:
        if dry_run or not removed_node_ids:
            return
        self._publish(
            group_id,
            "graph.delta",
            {"nodesRemoved": list(removed_node_ids)},
        )

    def cycle_completed(
        self,
        group_id: str,
        cycle: ConsolidationCycle,
        *,
        finalization: dict[str, object] | None = None,
    ) -> None:
        self._publish(
            group_id,
            "consolidation.completed",
            ConsolidationCycleLifecycleResult.from_cycle(
                cycle,
                finalization=finalization,
            ).event_payload(),
        )

    def learning_updated(
        self,
        group_id: str,
        *,
        cycle_id: str,
        distillation_examples: int,
        calibration_snapshots: int,
    ) -> None:
        if not distillation_examples and not calibration_snapshots:
            return
        self._publish(
            group_id,
            "consolidation.learning.updated",
            {
                "cycle_id": cycle_id,
                "distillation_examples": distillation_examples,
                "calibration_snapshots": calibration_snapshots,
            },
        )

    def _publish(self, group_id: str, event_type: str, payload: dict) -> None:
        if self._event_bus:
            self._event_bus.publish(group_id, event_type, payload)
