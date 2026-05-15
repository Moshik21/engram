"""Shared presentation helpers for consolidation cycle results."""

from __future__ import annotations

from typing import Any

from engram.models.consolidation import ConsolidationCycle, PhaseResult


def serialize_phase_result(phase_result: PhaseResult) -> dict[str, Any]:
    return {
        "phase": phase_result.phase,
        "status": phase_result.status,
        "items_processed": phase_result.items_processed,
        "items_affected": phase_result.items_affected,
        "duration_ms": phase_result.duration_ms,
        "error": phase_result.error,
    }


def serialize_cycle_summary(cycle: ConsolidationCycle) -> dict[str, Any]:
    phase_issue = cycle_phase_issue_text(cycle)
    return {
        "id": cycle.id,
        "status": cycle.status,
        "error": cycle.error,
        "phase_issue": phase_issue,
        "dry_run": cycle.dry_run,
        "trigger": cycle.trigger,
        "started_at": cycle.started_at,
        "completed_at": cycle.completed_at,
        "total_duration_ms": cycle.total_duration_ms,
        "phases": [serialize_phase_result(pr) for pr in cycle.phase_results],
        "summary": cycle_operator_summary(cycle),
    }


def cycle_totals(cycle: ConsolidationCycle) -> dict[str, int]:
    return {
        "total_processed": sum(pr.items_processed for pr in cycle.phase_results),
        "total_affected": sum(pr.items_affected for pr in cycle.phase_results),
    }


def cycle_operator_summary(cycle: ConsolidationCycle) -> dict[str, Any]:
    return {
        **cycle_totals(cycle),
        "description": cycle_description(cycle),
    }


def cycle_phase_issue_text(cycle: ConsolidationCycle) -> str | None:
    for phase in cycle.phase_results:
        phase_error = phase.error
        if phase.status != "error" and not phase_error:
            continue
        if phase_error:
            return f"{phase.phase}: {phase_error}"
        return f"{phase.phase}: phase error"
    return None


def cycle_description(cycle: ConsolidationCycle) -> str:
    totals = cycle_totals(cycle)
    phase_issue = cycle_phase_issue_text(cycle)
    cycle_label = (
        "cycle with warnings"
        if cycle.status == "completed" and phase_issue
        else "cycle"
        if cycle.status == "completed"
        else f"{cycle.status} cycle"
    )
    return (
        f"{'Dry run' if cycle.dry_run else 'Live'} {cycle_label}: "
        f"{totals['total_processed']} items processed, "
        f"{totals['total_affected']} affected across {len(cycle.phase_results)} phases"
    )
