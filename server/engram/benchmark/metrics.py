"""IR evaluation metrics for A/B benchmark."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RecallEvalSample:
    """Single-turn recall evaluation outcome."""

    recall_triggered: bool
    recall_helped: bool
    packets_surfaced: int = 0
    packets_used: int = 0
    false_recalls: int = 0


@dataclass(frozen=True)
class SessionContinuitySample:
    """Outcome for a multi-turn continuity task."""

    baseline_score: float
    memory_score: float
    open_loop_expected: bool = False
    open_loop_recovered: bool = False
    temporal_expected: bool = False
    temporal_correct: bool = False


@dataclass(frozen=True)
class RecallEvaluationSummary:
    """Aggregate Phase 6 evaluation metrics."""

    memory_need_precision: float
    useful_packet_rate: float
    false_recall_rate: float
    surfaced_to_used_ratio: float
    session_continuity_lift: float
    open_loop_recovery_rate: float
    temporal_correctness: float


def precision_at_k(
    ranked_ids: list[str],
    relevant: dict[str, int],
    k: int = 5,
) -> float:
    """Fraction of top-k results that are relevant (grade >= 1)."""
    if k <= 0:
        return 0.0
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if relevant.get(rid, 0) >= 1)
    return hits / len(top_k)


def recall_at_k(
    ranked_ids: list[str],
    relevant: dict[str, int],
    k: int = 10,
) -> float:
    """Fraction of all relevant entities found in top-k."""
    total_relevant = sum(1 for g in relevant.values() if g >= 1)
    if total_relevant == 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    found = sum(1 for rid, g in relevant.items() if g >= 1 and rid in top_k)
    return found / total_relevant


def reciprocal_rank(
    ranked_ids: list[str],
    relevant: dict[str, int],
) -> float:
    """1/rank of first relevant result, 0.0 if none found."""
    for i, rid in enumerate(ranked_ids):
        if relevant.get(rid, 0) >= 1:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    ranked_ids: list[str],
    relevant: dict[str, int],
    k: int = 5,
) -> float:
    """Normalized discounted cumulative gain with graded relevance."""
    top_k = ranked_ids[:k]
    if not top_k:
        return 0.0

    # DCG of actual ranking
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = relevant.get(rid, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1

    # Ideal DCG: sort all relevant grades descending
    ideal_gains = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_gains):
        if rel > 0:
            idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def gini_coefficient(values: list[float]) -> float:
    """Compute Gini coefficient for a list of non-negative values.

    Returns 0.0 for perfect equality, approaches 1.0 for extreme inequality.
    Returns 0.0 for empty or all-zero inputs.
    """
    if not values:
        return 0.0
    n = len(values)
    total = sum(values)
    if total == 0 or n == 0:
        return 0.0
    sorted_vals = sorted(values)
    cumulative = 0.0
    numerator = 0.0
    for i, v in enumerate(sorted_vals):
        cumulative += v
        numerator += (2 * (i + 1) - n - 1) * v
    return numerator / (n * total)


def memory_need_precision(samples: list[RecallEvalSample]) -> float:
    """Precision of recall triggering against downstream benefit."""
    triggered = [sample for sample in samples if sample.recall_triggered]
    if not triggered:
        return 0.0
    helped = sum(1 for sample in triggered if sample.recall_helped)
    return helped / len(triggered)


def useful_packet_rate(samples: list[RecallEvalSample]) -> float:
    """Fraction of surfaced packets that were used downstream."""
    surfaced = sum(max(0, sample.packets_surfaced) for sample in samples)
    if surfaced == 0:
        return 0.0
    used = sum(
        min(max(0, sample.packets_used), max(0, sample.packets_surfaced))
        for sample in samples
    )
    return used / surfaced


def false_recall_rate(samples: list[RecallEvalSample]) -> float:
    """Fraction of surfaced packets judged irrelevant or misleading."""
    surfaced = sum(max(0, sample.packets_surfaced) for sample in samples)
    if surfaced == 0:
        return 0.0
    false_recalls = sum(
        min(max(0, sample.false_recalls), max(0, sample.packets_surfaced))
        for sample in samples
    )
    return false_recalls / surfaced


def surfaced_to_used_ratio(
    surfaced_count: int,
    used_count: int,
) -> float:
    """Ratio of surfaced memories to actually used memories."""
    if surfaced_count <= 0:
        return 0.0
    if used_count <= 0:
        return math.inf
    return surfaced_count / used_count


def session_continuity_lift(samples: list[SessionContinuitySample]) -> float:
    """Average score lift from memory-enabled runs over baseline runs."""
    if not samples:
        return 0.0
    total_delta = sum(sample.memory_score - sample.baseline_score for sample in samples)
    return total_delta / len(samples)


def open_loop_recovery_rate(samples: list[SessionContinuitySample]) -> float:
    """Recovery rate for tasks where an unresolved loop should be surfaced."""
    expected = [sample for sample in samples if sample.open_loop_expected]
    if not expected:
        return 0.0
    recovered = sum(1 for sample in expected if sample.open_loop_recovered)
    return recovered / len(expected)


def temporal_correctness(samples: list[SessionContinuitySample]) -> float:
    """Accuracy on tasks where newer facts should override stale ones."""
    expected = [sample for sample in samples if sample.temporal_expected]
    if not expected:
        return 0.0
    correct = sum(1 for sample in expected if sample.temporal_correct)
    return correct / len(expected)


def summarize_recall_evaluation(
    recall_samples: list[RecallEvalSample],
    session_samples: list[SessionContinuitySample],
) -> RecallEvaluationSummary:
    """Aggregate Phase 6 evaluation metrics into a single summary object."""
    surfaced = sum(max(0, sample.packets_surfaced) for sample in recall_samples)
    used = sum(
        min(max(0, sample.packets_used), max(0, sample.packets_surfaced))
        for sample in recall_samples
    )
    return RecallEvaluationSummary(
        memory_need_precision=memory_need_precision(recall_samples),
        useful_packet_rate=useful_packet_rate(recall_samples),
        false_recall_rate=false_recall_rate(recall_samples),
        surfaced_to_used_ratio=surfaced_to_used_ratio(surfaced, used),
        session_continuity_lift=session_continuity_lift(session_samples),
        open_loop_recovery_rate=open_loop_recovery_rate(session_samples),
        temporal_correctness=temporal_correctness(session_samples),
    )


def bootstrap_ci(
    results_a: list[float],
    results_b: list[float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for metric difference (A - B).

    Returns (mean_diff, ci_lower, ci_upper).
    """
    assert len(results_a) == len(results_b), "Must have same number of queries"
    n = len(results_a)
    if n == 0:
        return (0.0, 0.0, 0.0)

    rng = random.Random(seed)
    diffs: list[float] = []

    for _ in range(n_resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_diff = sum(results_a[i] - results_b[i] for i in indices) / n
        diffs.append(sample_diff)

    diffs.sort()
    mean_diff = sum(diffs) / len(diffs)
    alpha = 1.0 - confidence
    lo_idx = int(math.floor(alpha / 2 * len(diffs)))
    hi_idx = int(math.ceil((1 - alpha / 2) * len(diffs))) - 1
    lo_idx = max(0, min(lo_idx, len(diffs) - 1))
    hi_idx = max(0, min(hi_idx, len(diffs) - 1))

    return (mean_diff, diffs[lo_idx], diffs[hi_idx])
