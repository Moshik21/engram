"""Tests for Recall result selection helpers."""

from __future__ import annotations

from engram.retrieval.result_selection import (
    filter_current_state_results,
    query_prefers_current_state,
)


def test_current_state_detection_uses_narrow_tokens() -> None:
    assert query_prefers_current_state("Which framework does Falcon use now?") is True
    assert query_prefers_current_state("What is the current framework?") is True
    assert query_prefers_current_state("Which framework did Falcon use before?") is False


def test_current_state_filter_prefers_entities_when_available() -> None:
    results = [
        {"result_type": "episode", "episode": {"id": "ep_old"}},
        {"result_type": "cue_episode", "episode": {"id": "ep_cue"}},
        {"result_type": "entity", "entity": {"id": "ent_current"}},
    ]

    filtered = filter_current_state_results("What does Falcon use currently?", results)

    assert filtered == [{"result_type": "entity", "entity": {"id": "ent_current"}}]


def test_current_state_filter_keeps_episodes_without_entity_state() -> None:
    results = [
        {"result_type": "episode", "episode": {"id": "ep_old"}},
        {"result_type": "cue_episode", "episode": {"id": "ep_cue"}},
    ]

    assert filter_current_state_results("What does Falcon use now?", results) == results
