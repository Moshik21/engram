from __future__ import annotations

from engram.consolidation.presenter import (
    cycle_description,
    cycle_operator_summary,
    cycle_phase_issue_text,
    cycle_totals,
    serialize_cycle_summary,
)
from engram.models.consolidation import ConsolidationCycle, PhaseResult


def test_serialize_cycle_summary_includes_cycle_and_phase_errors():
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="failed",
        error="cycle failed",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="error",
                items_processed=3,
                items_affected=1,
                duration_ms=12.5,
                error="phase failed",
            )
        ],
    )
    cycle.total_duration_ms = 20.0

    payload = serialize_cycle_summary(cycle)

    assert payload["id"] == cycle.id
    assert payload["status"] == "failed"
    assert payload["error"] == "cycle failed"
    assert payload["phase_issue"] == "triage: phase failed"
    assert payload["total_duration_ms"] == 20.0
    assert payload["summary"] == {
        "total_processed": 3,
        "total_affected": 1,
        "description": (
            "Dry run failed cycle: 3 items processed, 1 affected across 1 phases"
        ),
    }
    assert payload["phases"] == [
        {
            "phase": "triage",
            "status": "error",
            "items_processed": 3,
            "items_affected": 1,
            "duration_ms": 12.5,
            "error": "phase failed",
        }
    ]


def test_cycle_totals_and_description_use_failed_cycle_wording():
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=False,
        status="failed",
        phase_results=[
            PhaseResult(phase="triage", items_processed=3, items_affected=1),
            PhaseResult(phase="merge", items_processed=4, items_affected=2),
        ],
    )

    assert cycle_totals(cycle) == {"total_processed": 7, "total_affected": 3}
    assert cycle_operator_summary(cycle) == {
        "total_processed": 7,
        "total_affected": 3,
        "description": (
            "Live failed cycle: 7 items processed, 3 affected across 2 phases"
        ),
    }
    assert cycle_description(cycle) == (
        "Live failed cycle: 7 items processed, 3 affected across 2 phases"
    )


def test_cycle_description_reports_completed_cycle_with_phase_warning():
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="completed",
        phase_results=[
            PhaseResult(
                phase="graph_embed",
                status="error",
                items_processed=1,
                items_affected=0,
                error="optional vector index unavailable",
            )
        ],
    )

    assert cycle_phase_issue_text(cycle) == "graph_embed: optional vector index unavailable"
    assert cycle_description(cycle) == (
        "Dry run cycle with warnings: 1 items processed, "
        "0 affected across 1 phases"
    )
