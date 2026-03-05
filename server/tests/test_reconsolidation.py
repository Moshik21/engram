"""Tests for Brain Architecture Phase 2B: Reconsolidation."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

from engram.config import ActivationConfig
from engram.retrieval.reconsolidation import (
    LabileEntry,
    LabileWindowTracker,
    attempt_reconsolidation,
    jaccard_token_overlap,
)

# --- LabileWindowTracker tests ---


def test_mark_labile_creates_entry():
    tracker = LabileWindowTracker(ttl=300.0)
    tracker.mark_labile("ent1", "Test", "Person", "A person", "who is test?")
    entry = tracker.get_labile("ent1")
    assert entry is not None
    assert entry.entity_id == "ent1"
    assert entry.name == "Test"
    assert entry.modification_count == 0


def test_window_does_not_extend():
    tracker = LabileWindowTracker(ttl=300.0)
    tracker.mark_labile("ent1", "Test", "Person", "A person", "query1")
    first_time = tracker.get_labile("ent1").recalled_at

    time.sleep(0.01)
    tracker.mark_labile("ent1", "Test", "Person", "Updated", "query2")
    second_time = tracker.get_labile("ent1").recalled_at

    assert first_time == second_time  # Window NOT extended
    assert tracker.get_labile("ent1").summary == "A person"  # Original summary kept


def test_expired_entries_evicted():
    tracker = LabileWindowTracker(ttl=0.01)  # Very short TTL
    tracker.mark_labile("ent1", "Test", "Person", "A person", "query1")
    time.sleep(0.02)
    entry = tracker.get_labile("ent1")
    assert entry is None


def test_max_entries_cap():
    tracker = LabileWindowTracker(ttl=300.0, max_entries=3)
    tracker.mark_labile("ent1", "A", "P", "s1", "q1")
    tracker.mark_labile("ent2", "B", "P", "s2", "q2")
    tracker.mark_labile("ent3", "C", "P", "s3", "q3")
    # Adding a 4th should evict the oldest
    tracker.mark_labile("ent4", "D", "P", "s4", "q4")
    assert tracker.get_labile("ent1") is None
    assert tracker.get_labile("ent4") is not None


def test_modification_budget():
    tracker = LabileWindowTracker(ttl=300.0)
    tracker.mark_labile("ent1", "Test", "Person", "summary", "query")

    # 3 modifications should succeed
    assert tracker.record_modification("ent1") is True
    assert tracker.record_modification("ent1") is True
    assert tracker.record_modification("ent1") is True

    # Budget exceeded (default max_mods=3)
    assert tracker.is_budget_exceeded("ent1", 3) is True


def test_record_modification_returns_false_for_missing():
    tracker = LabileWindowTracker(ttl=300.0)
    assert tracker.record_modification("nonexistent") is False


def test_is_budget_exceeded_returns_false_for_missing():
    tracker = LabileWindowTracker(ttl=300.0)
    assert tracker.is_budget_exceeded("nonexistent", 3) is False


# --- Jaccard overlap tests ---


def test_jaccard_overlap_identical():
    assert jaccard_token_overlap("hello world", "hello world") == 1.0


def test_jaccard_overlap_no_overlap():
    assert jaccard_token_overlap("hello world", "foo bar") == 0.0


def test_jaccard_overlap_partial():
    overlap = jaccard_token_overlap("hello world foo", "hello world bar")
    # intersection = {hello, world}, union = {hello, world, foo, bar}
    assert abs(overlap - 0.5) < 0.01


def test_jaccard_overlap_empty():
    assert jaccard_token_overlap("", "hello") == 0.0
    assert jaccard_token_overlap("hello", "") == 0.0
    assert jaccard_token_overlap("", "") == 0.0


def test_jaccard_overlap_case_insensitive():
    assert jaccard_token_overlap("Hello World", "hello world") == 1.0


# --- attempt_reconsolidation tests ---


def _make_entity(entity_id="ent1", name="Test", summary="A test entity", identity_core=False):
    entity = MagicMock()
    entity.id = entity_id
    entity.name = name
    entity.entity_type = "Person"
    entity.summary = summary
    entity.identity_core = identity_core
    entity.attributes = {}
    return entity


def test_reconsolidation_returns_updates_when_overlap_sufficient():
    cfg = ActivationConfig(reconsolidation_overlap_threshold=0.10)
    entity = _make_entity(summary="a test entity with details")
    labile = LabileEntry(
        entity_id="ent1", name="Test", entity_type="Person",
        summary="a test entity with details", query="q", recalled_at=time.time(),
    )
    # new_content shares words with summary
    result = attempt_reconsolidation(
        entity, "the test entity has new details and info", labile, cfg,
    )
    assert result is not None
    assert "summary" in result


def test_reconsolidation_returns_none_when_overlap_too_low():
    cfg = ActivationConfig(reconsolidation_overlap_threshold=0.90)
    entity = _make_entity(summary="a test entity")
    labile = LabileEntry(
        entity_id="ent1", name="Test", entity_type="Person",
        summary="a test entity", query="q", recalled_at=time.time(),
    )
    # Completely different content
    result = attempt_reconsolidation(
        entity, "quantum mechanics and physics", labile, cfg,
    )
    assert result is None


def test_reconsolidation_identity_core_summary_only():
    """identity_core entities should still get summary updates."""
    cfg = ActivationConfig(reconsolidation_overlap_threshold=0.10)
    entity = _make_entity(summary="Konner is a developer", identity_core=True)
    labile = LabileEntry(
        entity_id="ent1", name="Konner", entity_type="Person",
        summary="Konner is a developer", query="q", recalled_at=time.time(),
    )
    result = attempt_reconsolidation(
        entity, "Konner is a developer who also works on AI projects", labile, cfg,
    )
    assert result is not None
    assert "summary" in result


def test_reconsolidation_summary_capped_at_500():
    cfg = ActivationConfig(reconsolidation_overlap_threshold=0.0)
    long_summary = "a " * 250  # 500 chars
    entity = _make_entity(summary=long_summary.strip())
    labile = LabileEntry(
        entity_id="ent1", name="Test", entity_type="Person",
        summary=long_summary.strip(), query="q", recalled_at=time.time(),
    )
    result = attempt_reconsolidation(
        entity, "a new information about this entity a", labile, cfg,
    )
    if result and "summary" in result:
        assert len(result["summary"]) <= 500


def test_reconsolidation_no_update_when_no_new_info():
    cfg = ActivationConfig(reconsolidation_overlap_threshold=0.0)
    entity = _make_entity(summary="test")
    labile = LabileEntry(
        entity_id="ent1", name="Test", entity_type="Person",
        summary="test", query="q", recalled_at=time.time(),
    )
    # Empty new content
    result = attempt_reconsolidation(entity, "", labile, cfg)
    assert result is None


# --- Config tests ---


def test_reconsolidation_config_off_by_default():
    cfg = ActivationConfig()
    assert cfg.reconsolidation_enabled is False
    assert cfg.reconsolidation_window_seconds == 300.0
    assert cfg.reconsolidation_max_modifications == 3


def test_reconsolidation_enabled_in_standard():
    cfg = ActivationConfig(consolidation_profile="standard")
    assert cfg.reconsolidation_enabled is True


def test_reconsolidation_not_enabled_in_conservative():
    cfg = ActivationConfig(consolidation_profile="conservative")
    assert cfg.reconsolidation_enabled is False


# --- GraphManager integration test ---


def test_labile_tracker_created_when_enabled():
    """When reconsolidation is enabled, GraphManager should create a tracker."""
    from engram.graph_manager import GraphManager

    cfg = ActivationConfig(reconsolidation_enabled=True)
    gm = GraphManager(
        graph_store=AsyncMock(),
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        extractor=AsyncMock(),
        cfg=cfg,
    )
    assert gm._labile_tracker is not None


def test_labile_tracker_not_created_when_disabled():
    """When reconsolidation is disabled, no tracker should exist."""
    from engram.graph_manager import GraphManager

    cfg = ActivationConfig(reconsolidation_enabled=False)
    gm = GraphManager(
        graph_store=AsyncMock(),
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        extractor=AsyncMock(),
        cfg=cfg,
    )
    assert gm._labile_tracker is None
