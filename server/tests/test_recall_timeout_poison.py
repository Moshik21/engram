"""Primary search must not be poisoned to ~100ms after stats probe timeout."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.retrieval.candidate_pool import (
    _extract_entity_names_from_query,
    _primary_search_timeout_seconds,
)


def test_primary_search_timeout_not_poisoned_after_stats_timeout():
    cfg = ActivationConfig()
    stage = {"recall_stats_timeout": 26.0, "graph_expand_timeout": 91.0}
    seconds = _primary_search_timeout_seconds(cfg, stage)
    assert seconds is not None
    # Product floor: at least 1s (explicit search budget), never 0.1s poison.
    assert seconds >= 1.0


def test_primary_search_timeout_without_probe_uses_primary_default():
    cfg = ActivationConfig()
    seconds = _primary_search_timeout_seconds(cfg, {})
    assert seconds is not None
    assert abs(seconds - cfg.retrieval_primary_search_timeout_ms / 1000.0) < 1e-6


def test_extract_names_includes_full_query_for_decision_lookup():
    query = "Cold Decision hit requires healthy search index"
    names = _extract_entity_names_from_query(query)
    assert query in names
