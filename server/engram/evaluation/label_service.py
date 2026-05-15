"""Shared persistence helpers for evaluation labels."""

from __future__ import annotations

from typing import Any

from engram.evaluation.presenter import (
    present_recall_sample_write,
    present_session_sample_write,
)
from engram.evaluation.store import (
    StoredRecallEvalSample,
    StoredSessionContinuitySample,
)


async def build_recall_evaluation_write_surface(
    evaluation_store: Any,
    *,
    group_id: str,
    surface: str,
    recall_triggered: bool,
    recall_helped: bool,
    recall_needed: bool | None = None,
    packets_surfaced: int = 0,
    packets_used: int = 0,
    false_recalls: int = 0,
    source: str = "manual",
    query: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist and present one recall-quality evaluation label."""
    sample = await persist_recall_eval_sample(
        evaluation_store,
        group_id=group_id,
        recall_triggered=recall_triggered,
        recall_helped=recall_helped,
        recall_needed=recall_needed,
        packets_surfaced=packets_surfaced,
        packets_used=packets_used,
        false_recalls=false_recalls,
        source=source,
        query=query,
        notes=notes,
    )
    return present_recall_sample_write(sample, surface=surface)


async def build_session_continuity_evaluation_write_surface(
    evaluation_store: Any,
    *,
    group_id: str,
    surface: str,
    baseline_score: float,
    memory_score: float,
    open_loop_expected: bool = False,
    open_loop_recovered: bool = False,
    temporal_expected: bool = False,
    temporal_correct: bool = False,
    source: str = "manual",
    scenario: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist and present one session-continuity evaluation label."""
    sample = await persist_session_continuity_sample(
        evaluation_store,
        group_id=group_id,
        baseline_score=baseline_score,
        memory_score=memory_score,
        open_loop_expected=open_loop_expected,
        open_loop_recovered=open_loop_recovered,
        temporal_expected=temporal_expected,
        temporal_correct=temporal_correct,
        source=source,
        scenario=scenario,
        notes=notes,
    )
    return present_session_sample_write(sample, surface=surface)


async def persist_recall_eval_sample(
    evaluation_store: Any,
    *,
    group_id: str,
    recall_triggered: bool,
    recall_helped: bool,
    recall_needed: bool | None = None,
    packets_surfaced: int = 0,
    packets_used: int = 0,
    false_recalls: int = 0,
    source: str = "manual",
    query: str | None = None,
    notes: str | None = None,
) -> StoredRecallEvalSample:
    """Build and persist one recall-quality label for a brain group."""
    sample = StoredRecallEvalSample(
        group_id=group_id,
        recall_triggered=recall_triggered,
        recall_helped=recall_helped,
        recall_needed=recall_needed,
        packets_surfaced=max(0, packets_surfaced),
        packets_used=max(0, packets_used),
        false_recalls=max(0, false_recalls),
        source=source,
        query=query,
        notes=notes,
    )
    await evaluation_store.save_recall_sample(sample)
    return sample


async def persist_session_continuity_sample(
    evaluation_store: Any,
    *,
    group_id: str,
    baseline_score: float,
    memory_score: float,
    open_loop_expected: bool = False,
    open_loop_recovered: bool = False,
    temporal_expected: bool = False,
    temporal_correct: bool = False,
    source: str = "manual",
    scenario: str | None = None,
    notes: str | None = None,
) -> StoredSessionContinuitySample:
    """Build and persist one session-continuity label for a brain group."""
    sample = StoredSessionContinuitySample(
        group_id=group_id,
        baseline_score=baseline_score,
        memory_score=memory_score,
        open_loop_expected=open_loop_expected,
        open_loop_recovered=open_loop_recovered,
        temporal_expected=temporal_expected,
        temporal_correct=temporal_correct,
        source=source,
        scenario=scenario,
        notes=notes,
    )
    await evaluation_store.save_session_sample(sample)
    return sample
