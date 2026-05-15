from __future__ import annotations

import json

import pytest

from engram.consolidation import cli
from engram.models.consolidation import ConsolidationCycle, PhaseResult


def test_print_cycle_result_reports_completed_cycle(capsys):
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="completed",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="success",
                items_processed=3,
                items_affected=2,
            )
        ],
    )
    cycle.total_duration_ms = 12.5

    cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "completed"
    assert payload["error"] is None
    assert payload["phases"][0]["error"] is None
    assert "[DRY RUN] Consolidation complete: 3 items processed, 2 affected" in captured.out
    assert captured.err == ""


def test_print_cycle_result_warns_for_completed_cycle_with_phase_error(capsys):
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
    cycle.total_duration_ms = 8.0

    cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "completed"
    assert payload["phases"][0]["error"] == "optional vector index unavailable"
    assert (
        "[DRY RUN] Consolidation completed with warnings: "
        "1 items processed, 0 affected"
    ) in captured.out
    assert (
        "Consolidation warning: graph_embed: optional vector index unavailable"
        in captured.err
    )


def test_print_cycle_result_exits_nonzero_for_failed_cycle(capsys):
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="failed",
        error="Phase 'triage' requires graph_store methods: missing_method",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="error",
                error="missing_method",
            )
        ],
    )
    cycle.total_duration_ms = 4.2

    with pytest.raises(SystemExit) as exc_info:
        cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "failed"
    assert payload["error"] == "Phase 'triage' requires graph_store methods: missing_method"
    assert payload["phases"][0]["error"] == "missing_method"
    assert "[DRY RUN] Consolidation failed: 0 items processed, 0 affected" in captured.out
    assert "Consolidation complete" not in captured.out
    assert (
        "Consolidation failed: Phase 'triage' requires graph_store methods: missing_method"
        in captured.err
    )
