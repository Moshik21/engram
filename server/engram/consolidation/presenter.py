"""Shared presentation helpers for consolidation cycle results."""

from __future__ import annotations

from typing import Any

from engram.consolidation.audit_reader import ConsolidationCycleDetail
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


def _record_fields(record: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    return {field_name: getattr(record, field_name, None) for field_name in field_names}


def serialize_cycle_detail(detail: ConsolidationCycleDetail) -> dict[str, Any]:
    """Serialize a full consolidation cycle detail surface."""
    return {
        **serialize_cycle_summary(detail.cycle),
        "merges": [
            _record_fields(
                record,
                (
                    "id",
                    "keep_id",
                    "remove_id",
                    "keep_name",
                    "remove_name",
                    "similarity",
                    "decision_confidence",
                    "decision_source",
                    "decision_reason",
                    "relationships_transferred",
                ),
            )
            for record in detail.merges
        ],
        "identifier_reviews": [
            _record_fields(
                record,
                (
                    "id",
                    "entity_a_id",
                    "entity_b_id",
                    "entity_a_name",
                    "entity_b_name",
                    "entity_a_type",
                    "entity_b_type",
                    "raw_similarity",
                    "adjusted_similarity",
                    "decision_source",
                    "decision_reason",
                    "entity_a_regime",
                    "entity_b_regime",
                    "canonical_identifier_a",
                    "canonical_identifier_b",
                    "review_status",
                    "metadata",
                ),
            )
            for record in detail.identifier_reviews
        ],
        "inferred_edges": [
            _record_fields(
                record,
                (
                    "id",
                    "source_id",
                    "target_id",
                    "source_name",
                    "target_name",
                    "co_occurrence_count",
                    "confidence",
                    "infer_type",
                    "pmi_score",
                    "llm_verdict",
                ),
            )
            for record in detail.inferred_edges
        ],
        "prunes": [
            _record_fields(
                record,
                ("id", "entity_id", "entity_name", "entity_type", "reason"),
            )
            for record in detail.prunes
        ],
        "reindexes": [
            _record_fields(
                record,
                ("id", "entity_id", "entity_name", "source_phase"),
            )
            for record in detail.reindexes
        ],
        "replays": [
            _record_fields(
                record,
                (
                    "id",
                    "episode_id",
                    "new_entities_found",
                    "new_relationships_found",
                    "entities_updated",
                    "skipped_reason",
                ),
            )
            for record in detail.replays
        ],
        "dreams": [
            _record_fields(
                record,
                (
                    "id",
                    "source_entity_id",
                    "target_entity_id",
                    "weight_delta",
                    "seed_entity_id",
                ),
            )
            for record in detail.dreams
        ],
        "decision_traces": [
            _record_fields(
                record,
                (
                    "id",
                    "phase",
                    "candidate_type",
                    "candidate_id",
                    "decision",
                    "decision_source",
                    "confidence",
                    "threshold_band",
                    "features",
                    "constraints_hit",
                    "policy_version",
                    "metadata",
                ),
            )
            for record in detail.decision_traces
        ],
        "decision_outcomes": [
            _record_fields(
                record,
                (
                    "id",
                    "phase",
                    "decision_trace_id",
                    "outcome_type",
                    "label",
                    "value",
                    "metadata",
                ),
            )
            for record in detail.decision_outcomes
        ],
        "distillation_examples": [
            _record_fields(
                record,
                (
                    "id",
                    "phase",
                    "candidate_type",
                    "candidate_id",
                    "decision_trace_id",
                    "teacher_label",
                    "teacher_source",
                    "student_decision",
                    "student_confidence",
                    "threshold_band",
                    "features",
                    "correct",
                    "metadata",
                ),
            )
            for record in detail.distillation_examples
        ],
        "calibration_snapshots": [
            _record_fields(
                record,
                (
                    "id",
                    "phase",
                    "window_cycles",
                    "total_traces",
                    "labeled_examples",
                    "oracle_examples",
                    "abstain_count",
                    "accuracy",
                    "mean_confidence",
                    "expected_calibration_error",
                    "summary",
                ),
            )
            for record in detail.calibration_snapshots
        ],
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
