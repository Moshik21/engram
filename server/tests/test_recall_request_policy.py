"""Tests for Recall request policy helpers."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.retrieval.request_policy import (
    recall_fetch_limit,
    should_record_ranking_feedback,
    split_primary_and_near_miss_results,
)


def test_fetch_limit_includes_near_miss_window_only_with_context() -> None:
    cfg = ActivationConfig(conv_near_miss_enabled=True, conv_near_miss_window=3)

    assert recall_fetch_limit(cfg, 5, conv_context=object()) == 8
    assert recall_fetch_limit(cfg, 5, conv_context=None) == 5


def test_ranking_feedback_tracks_true_usage_not_passive_surfacing() -> None:
    assert (
        should_record_ranking_feedback(record_access=True, interaction_type="surfaced")
        is False
    )
    assert (
        should_record_ranking_feedback(record_access=True, interaction_type="dismissed")
        is False
    )
    assert (
        should_record_ranking_feedback(record_access=False, interaction_type="used")
        is True
    )
    assert (
        should_record_ranking_feedback(record_access=False, interaction_type="confirmed")
        is True
    )
    assert should_record_ranking_feedback(record_access=True, interaction_type=None) is True


def test_splits_primary_and_near_miss_results() -> None:
    results = ["one", "two", "near"]

    primary, near_miss = split_primary_and_near_miss_results(
        results,
        2,
        near_miss_enabled=True,
    )

    assert primary == ["one", "two"]
    assert near_miss == ["near"]
    assert split_primary_and_near_miss_results(
        results,
        2,
        near_miss_enabled=False,
    ) == (["one", "two"], [])
