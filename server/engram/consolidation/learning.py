"""Post-cycle learning artifact generation for consolidation."""

from __future__ import annotations

from dataclasses import dataclass

from engram.config import ActivationConfig
from engram.consolidation.calibration import (
    build_calibration_snapshots,
    build_distillation_examples,
)
from engram.models.consolidation import (
    ConsolidationCycle,
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
)
from engram.storage.protocols import ConsolidationStore


@dataclass(frozen=True)
class ConsolidationLearningResult:
    """Counts of post-cycle learning artifacts produced for a cycle."""

    distillation_examples: int = 0
    calibration_snapshots: int = 0

    @property
    def updated(self) -> bool:
        return bool(self.distillation_examples or self.calibration_snapshots)


class ConsolidationLearningService:
    """Build and persist post-cycle distillation/calibration artifacts."""

    def __init__(
        self,
        cfg: ActivationConfig,
        consolidation_store: ConsolidationStore | None = None,
    ) -> None:
        self._cfg = cfg
        self._store = consolidation_store

    async def analyze_cycle(
        self,
        cycle: ConsolidationCycle,
        context: CycleContext,
    ) -> ConsolidationLearningResult:
        if self._store is None:
            return ConsolidationLearningResult()

        distillation_examples = []
        if self._cfg.consolidation_distillation_enabled and context.decision_traces:
            distillation_examples = build_distillation_examples(
                cycle.id,
                cycle.group_id,
                context.decision_traces,
                context.decision_outcome_labels,
            )
            for example in distillation_examples:
                await self._store.save_distillation_example(example)

        snapshots = []
        if self._cfg.consolidation_calibration_enabled:
            recent_cycles = await self._store.get_recent_cycles(
                cycle.group_id,
                limit=self._cfg.consolidation_calibration_window_cycles,
            )
            traces, labels = await self._collect_calibration_examples(
                cycle,
                context,
                recent_cycles,
            )

            if traces:
                snapshots = build_calibration_snapshots(
                    cycle.id,
                    cycle.group_id,
                    traces,
                    labels,
                    window_cycles=len(recent_cycles),
                    min_examples=self._cfg.consolidation_calibration_min_examples,
                    bins=self._cfg.consolidation_calibration_bins,
                )
                for snapshot in snapshots:
                    snapshot.summary.setdefault(
                        "requested_window_cycles",
                        self._cfg.consolidation_calibration_window_cycles,
                    )
                    snapshot.summary.setdefault(
                        "cycles_observed",
                        len(recent_cycles),
                    )
                    await self._store.save_calibration_snapshot(snapshot)

        return ConsolidationLearningResult(
            distillation_examples=len(distillation_examples),
            calibration_snapshots=len(snapshots),
        )

    async def _collect_calibration_examples(
        self,
        cycle: ConsolidationCycle,
        context: CycleContext,
        recent_cycles: list[ConsolidationCycle],
    ) -> tuple[list[DecisionTrace], list[DecisionOutcomeLabel]]:
        assert self._store is not None
        traces: list[DecisionTrace] = []
        labels: list[DecisionOutcomeLabel] = []
        for recent_cycle in recent_cycles:
            if recent_cycle.id == cycle.id:
                traces.extend(context.decision_traces)
                labels.extend(context.decision_outcome_labels)
                continue
            traces.extend(await self._store.get_decision_traces(recent_cycle.id, cycle.group_id))
            labels.extend(
                await self._store.get_decision_outcome_labels(recent_cycle.id, cycle.group_id)
            )
        return traces, labels
