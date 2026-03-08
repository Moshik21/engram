"""Tests for IR metric computation correctness."""

import math

import pytest

from engram.benchmark.metrics import (
    RecallEvalSample,
    SessionContinuitySample,
    bootstrap_ci,
    false_recall_rate,
    memory_need_precision,
    ndcg_at_k,
    open_loop_recovery_rate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    session_continuity_lift,
    summarize_recall_evaluation,
    surfaced_to_used_ratio,
    temporal_correctness,
    useful_packet_rate,
)


def test_precision_at_k_perfect():
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a": 3, "b": 2, "c": 1, "d": 1, "e": 1}
    assert precision_at_k(ranked, relevant, k=5) == 1.0


def test_precision_at_k_none():
    ranked = ["x", "y", "z"]
    relevant = {"a": 3, "b": 2}
    assert precision_at_k(ranked, relevant, k=3) == 0.0


def test_recall_at_k():
    ranked = ["a", "b", "x", "y", "z"]
    relevant = {"a": 3, "b": 2, "c": 1, "d": 1}
    # Found 2 of 4 relevant in top 5
    assert recall_at_k(ranked, relevant, k=5) == 0.5


def test_reciprocal_rank():
    ranked = ["x", "y", "a", "b"]
    relevant = {"a": 2, "b": 1}
    assert reciprocal_rank(ranked, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_none():
    ranked = ["x", "y", "z"]
    relevant = {"a": 2}
    assert reciprocal_rank(ranked, relevant) == 0.0


def test_ndcg_at_k():
    # Known computation:
    # ranked = [a(3), x(0), b(2)]
    # DCG = 3/log2(2) + 0/log2(3) + 2/log2(4) = 3.0 + 0 + 1.0 = 4.0
    # Ideal = [3, 2] -> 3/log2(2) + 2/log2(3) = 3.0 + 1.2618... = 4.2618...
    # nDCG = 4.0 / 4.2618... ~ 0.9386
    ranked = ["a", "x", "b"]
    relevant = {"a": 3, "b": 2}
    result = ndcg_at_k(ranked, relevant, k=3)
    assert 0.93 < result < 0.95


def test_bootstrap_ci_identical():
    scores = [0.5, 0.6, 0.7, 0.8, 0.9]
    mean_diff, lo, hi = bootstrap_ci(scores, scores)
    assert abs(mean_diff) < 0.01
    assert lo <= 0.0 <= hi


def test_memory_need_precision():
    samples = [
        RecallEvalSample(recall_triggered=True, recall_helped=True),
        RecallEvalSample(recall_triggered=True, recall_helped=False),
        RecallEvalSample(recall_triggered=False, recall_helped=False),
    ]
    assert memory_need_precision(samples) == pytest.approx(0.5)


def test_useful_packet_rate():
    samples = [
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=True,
            packets_surfaced=4,
            packets_used=2,
        ),
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=False,
            packets_surfaced=2,
            packets_used=1,
        ),
    ]
    assert useful_packet_rate(samples) == pytest.approx(0.5)


def test_false_recall_rate():
    samples = [
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=False,
            packets_surfaced=5,
            false_recalls=2,
        ),
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=True,
            packets_surfaced=3,
            false_recalls=1,
        ),
    ]
    assert false_recall_rate(samples) == pytest.approx(3 / 8)


def test_surfaced_to_used_ratio_handles_zero_used():
    assert math.isinf(surfaced_to_used_ratio(4, 0))


def test_session_continuity_lift():
    samples = [
        SessionContinuitySample(baseline_score=0.4, memory_score=0.8),
        SessionContinuitySample(baseline_score=0.6, memory_score=0.7),
    ]
    assert session_continuity_lift(samples) == pytest.approx(0.25)


def test_open_loop_recovery_rate():
    samples = [
        SessionContinuitySample(
            baseline_score=0.5,
            memory_score=0.8,
            open_loop_expected=True,
            open_loop_recovered=True,
        ),
        SessionContinuitySample(
            baseline_score=0.5,
            memory_score=0.6,
            open_loop_expected=True,
            open_loop_recovered=False,
        ),
        SessionContinuitySample(baseline_score=0.5, memory_score=0.5),
    ]
    assert open_loop_recovery_rate(samples) == pytest.approx(0.5)


def test_temporal_correctness():
    samples = [
        SessionContinuitySample(
            baseline_score=0.3,
            memory_score=0.9,
            temporal_expected=True,
            temporal_correct=True,
        ),
        SessionContinuitySample(
            baseline_score=0.3,
            memory_score=0.4,
            temporal_expected=True,
            temporal_correct=False,
        ),
    ]
    assert temporal_correctness(samples) == pytest.approx(0.5)


def test_summarize_recall_evaluation():
    recall_samples = [
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=True,
            packets_surfaced=3,
            packets_used=2,
            false_recalls=1,
        ),
        RecallEvalSample(
            recall_triggered=True,
            recall_helped=False,
            packets_surfaced=1,
            packets_used=0,
            false_recalls=1,
        ),
    ]
    session_samples = [
        SessionContinuitySample(
            baseline_score=0.5,
            memory_score=0.8,
            open_loop_expected=True,
            open_loop_recovered=True,
            temporal_expected=True,
            temporal_correct=True,
        )
    ]

    summary = summarize_recall_evaluation(recall_samples, session_samples)

    assert summary.memory_need_precision == pytest.approx(0.5)
    assert summary.useful_packet_rate == pytest.approx(0.5)
    assert summary.false_recall_rate == pytest.approx(0.5)
    assert summary.surfaced_to_used_ratio == pytest.approx(2.0)
    assert summary.session_continuity_lift == pytest.approx(0.3)
    assert summary.open_loop_recovery_rate == pytest.approx(1.0)
    assert summary.temporal_correctness == pytest.approx(1.0)
