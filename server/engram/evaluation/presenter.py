"""Presentation helpers for evaluation samples across REST and MCP."""

from __future__ import annotations

from typing import Any, Literal

from engram.evaluation.store import (
    StoredRecallEvalSample,
    StoredSessionContinuitySample,
)

EvaluationSurface = Literal["rest", "mcp"]


def present_recall_sample(
    sample: StoredRecallEvalSample,
    *,
    surface: EvaluationSurface,
) -> dict[str, Any]:
    """Serialize one recall-quality sample for a public surface."""
    if surface == "rest":
        return {
            "id": sample.id,
            "recallTriggered": sample.recall_triggered,
            "recallHelped": sample.recall_helped,
            "recallNeeded": sample.recall_needed,
            "packetsSurfaced": sample.packets_surfaced,
            "packetsUsed": sample.packets_used,
            "falseRecalls": sample.false_recalls,
            "source": sample.source,
            "query": sample.query,
            "notes": sample.notes,
            "timestamp": sample.timestamp,
        }
    return {
        "id": sample.id,
        "recall_triggered": sample.recall_triggered,
        "recall_helped": sample.recall_helped,
        "recall_needed": sample.recall_needed,
        "packets_surfaced": sample.packets_surfaced,
        "packets_used": sample.packets_used,
        "false_recalls": sample.false_recalls,
        "source": sample.source,
        "query": sample.query,
        "notes": sample.notes,
        "timestamp": sample.timestamp,
    }


def present_session_sample(
    sample: StoredSessionContinuitySample,
    *,
    surface: EvaluationSurface,
) -> dict[str, Any]:
    """Serialize one session-continuity sample for a public surface."""
    if surface == "rest":
        return {
            "id": sample.id,
            "baselineScore": sample.baseline_score,
            "memoryScore": sample.memory_score,
            "openLoopExpected": sample.open_loop_expected,
            "openLoopRecovered": sample.open_loop_recovered,
            "temporalExpected": sample.temporal_expected,
            "temporalCorrect": sample.temporal_correct,
            "source": sample.source,
            "scenario": sample.scenario,
            "notes": sample.notes,
            "timestamp": sample.timestamp,
        }
    return {
        "id": sample.id,
        "baseline_score": sample.baseline_score,
        "memory_score": sample.memory_score,
        "open_loop_expected": sample.open_loop_expected,
        "open_loop_recovered": sample.open_loop_recovered,
        "temporal_expected": sample.temporal_expected,
        "temporal_correct": sample.temporal_correct,
        "source": sample.source,
        "scenario": sample.scenario,
        "notes": sample.notes,
        "timestamp": sample.timestamp,
    }


def present_recall_sample_write(
    sample: StoredRecallEvalSample,
    *,
    surface: EvaluationSurface,
) -> dict[str, Any]:
    """Return the shared recall-label write acknowledgement."""
    if surface == "rest":
        return {
            "status": "stored",
            "groupId": sample.group_id,
            "sample": present_recall_sample(sample, surface=surface),
        }
    return {
        "status": "stored",
        "operation": "record_recall_evaluation",
        "group_id": sample.group_id,
        "sample": present_recall_sample(sample, surface=surface),
    }


def present_session_sample_write(
    sample: StoredSessionContinuitySample,
    *,
    surface: EvaluationSurface,
) -> dict[str, Any]:
    """Return the shared session-continuity label write acknowledgement."""
    if surface == "rest":
        return {
            "status": "stored",
            "groupId": sample.group_id,
            "sample": present_session_sample(sample, surface=surface),
        }
    return {
        "status": "stored",
        "operation": "record_session_continuity_evaluation",
        "group_id": sample.group_id,
        "sample": present_session_sample(sample, surface=surface),
    }
