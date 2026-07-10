"""Continuity golden-path suite (product metric, not LongMemEval)."""

from __future__ import annotations

import pytest

from engram.evaluation.continuity import (
    STRATEGY_DECISIONS,
    format_continuity_report,
    run_continuity_golden_path_smoke,
)


@pytest.mark.asyncio
async def test_continuity_golden_path_promote_context_recall():
    result = await run_continuity_golden_path_smoke()
    assert result["passed"] is True, format_continuity_report(result)
    assert len(result["promoted"]) == len(STRATEGY_DECISIONS)
    assert result["recall_hits"], "recall must surface at least one strategy Decision"
    assert result["context_hits"] or result["identity_core"], (
        "get_context or identity-core protection must land for continuity"
    )


def test_continuity_report_formats():
    report = format_continuity_report(
        {
            "passed": True,
            "metric": "test",
            "duration_ms": 12.3,
            "identity_core": ["A"],
            "context_hits": ["A"],
            "recall_hits": ["A"],
            "promoted": [{"name": "A", "episode_id": "ep1"}],
        }
    )
    assert "PASS" in report
    assert "A" in report
