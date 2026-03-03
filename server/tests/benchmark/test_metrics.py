"""Tests for IR metric computation correctness."""

import pytest

from engram.benchmark.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
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
