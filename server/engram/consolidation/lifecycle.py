"""Typed lifecycle contracts for consolidation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from engram.consolidation.presenter import cycle_phase_issue_text
from engram.models.consolidation import ConsolidationCycle, PhaseResult


@dataclass(frozen=True)
class ConsolidationPhasePlan:
    """One planned phase in a consolidation cycle."""

    name: str
    ordinal: int
    selected: bool = True


@dataclass(frozen=True)
class ConsolidationCyclePlan:
    """Stable plan for the Consolidate stage before phase execution."""

    cycle_id: str
    group_id: str
    trigger: str
    dry_run: bool
    phases: tuple[ConsolidationPhasePlan, ...]
    requested_phase_names: frozenset[str] | None = None

    @property
    def selected_phases(self) -> tuple[ConsolidationPhasePlan, ...]:
        return tuple(phase for phase in self.phases if phase.selected)

    @property
    def selected_phase_names(self) -> set[str]:
        return {phase.name for phase in self.selected_phases}

    @property
    def known_phase_names(self) -> frozenset[str]:
        return frozenset(phase.name for phase in self.phases)

    @property
    def unknown_phase_names(self) -> frozenset[str]:
        if self.requested_phase_names is None:
            return frozenset()
        return self.requested_phase_names - self.known_phase_names

    def phase_plan(self, phase_name: str) -> ConsolidationPhasePlan | None:
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        return None

    def validate_requested_phase_names(self) -> None:
        """Fail fast when a caller asks for phases outside the engine contract."""
        unknown = sorted(self.unknown_phase_names)
        if unknown:
            names = ", ".join(unknown)
            raise ValueError(f"Unknown consolidation phase(s): {names}")

    def started_payload(self) -> dict[str, object]:
        return {
            "cycle_id": self.cycle_id,
            "dry_run": self.dry_run,
            "trigger": self.trigger,
            "lifecycleStage": "consolidate",
            "phaseCount": len(self.selected_phases),
            "phases": [phase.name for phase in self.selected_phases],
        }


@dataclass(frozen=True)
class ConsolidationPhaseLifecycleResult:
    """Stable lifecycle result for one consolidation phase."""

    cycle_id: str
    phase: str
    status: str
    items_processed: int = 0
    items_affected: int = 0
    duration_ms: float = 0.0
    error: str | None = None
    ordinal: int | None = None

    @classmethod
    def from_phase_result(
        cls,
        cycle_id: str,
        result: PhaseResult,
        *,
        ordinal: int | None = None,
    ) -> ConsolidationPhaseLifecycleResult:
        return cls(
            cycle_id=cycle_id,
            phase=result.phase,
            status=result.status,
            items_processed=result.items_processed,
            items_affected=result.items_affected,
            duration_ms=result.duration_ms,
            error=result.error,
            ordinal=ordinal,
        )

    def event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "cycle_id": self.cycle_id,
            "phase": self.phase,
            "status": self.status,
            "items_processed": self.items_processed,
            "items_affected": self.items_affected,
            "duration_ms": self.duration_ms,
            "lifecycleStage": "consolidate",
        }
        if self.ordinal is not None:
            payload["phaseOrdinal"] = self.ordinal
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class ConsolidationCycleLifecycleResult:
    """Stable lifecycle result for a completed consolidation cycle."""

    cycle_id: str
    status: str
    duration_ms: float
    phases: int
    trigger: str
    dry_run: bool
    error: str | None = None
    phase_issue: str | None = None
    finalization: dict[str, object] | None = None

    @classmethod
    def from_cycle(
        cls,
        cycle: ConsolidationCycle,
        *,
        finalization: dict[str, object] | None = None,
    ) -> ConsolidationCycleLifecycleResult:
        return cls(
            cycle_id=cycle.id,
            status=cycle.status,
            duration_ms=cycle.total_duration_ms,
            phases=len(cycle.phase_results),
            trigger=cycle.trigger,
            dry_run=cycle.dry_run,
            error=cycle.error,
            phase_issue=cycle_phase_issue_text(cycle),
            finalization=finalization,
        )

    def event_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "cycle_id": self.cycle_id,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "phases": self.phases,
            "trigger": self.trigger,
            "dry_run": self.dry_run,
            "lifecycleStage": "consolidate",
        }
        if self.error is not None:
            payload["error"] = self.error
        if self.phase_issue is not None:
            payload["phase_issue"] = self.phase_issue
        if self.finalization is not None:
            payload["finalization"] = self.finalization
        return payload


def build_cycle_plan(
    *,
    cycle: ConsolidationCycle,
    phases: list[Any],
    phase_names: set[str] | None = None,
) -> ConsolidationCyclePlan:
    """Build a selected phase plan while preserving engine phase order."""
    phase_plans = tuple(
        ConsolidationPhasePlan(
            name=phase.name,
            ordinal=index,
            selected=phase_names is None or phase.name in phase_names,
        )
        for index, phase in enumerate(phases)
    )
    return ConsolidationCyclePlan(
        cycle_id=cycle.id,
        group_id=cycle.group_id,
        trigger=cycle.trigger,
        dry_run=cycle.dry_run,
        phases=phase_plans,
        requested_phase_names=frozenset(phase_names) if phase_names is not None else None,
    )
