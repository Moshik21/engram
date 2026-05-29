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


def test_current_state_filter_prefers_relevant_entity() -> None:
    """Episodes are suppressed when a surfaced entity matches the query subject."""
    results = [
        {"result_type": "episode", "episode": {"id": "ep_old"}},
        {"result_type": "cue_episode", "episode": {"id": "ep_cue"}},
        {"result_type": "entity", "entity": {"id": "ent_current", "name": "Falcon"}},
    ]

    filtered = filter_current_state_results("What does Falcon use currently?", results)

    assert filtered == [
        {"result_type": "entity", "entity": {"id": "ent_current", "name": "Falcon"}}
    ]


def test_current_state_filter_keeps_episodes_when_entity_is_irrelevant() -> None:
    """When the surfaced entities are not about the query subject, the entity
    layer cannot answer the current-state question, so episodes are kept (the
    answer would otherwise be discarded). Regression for graph-on current-value
    multi-hops where extraction surfaced unrelated/incomplete entities."""
    results = [
        {"result_type": "episode", "episode": {"id": "ep_answer"}},
        {"result_type": "entity", "entity": {"id": "ent_other", "name": "Atlas"}},
        {"result_type": "entity", "entity": {"id": "ent_other2", "name": "NeurIPS"}},
    ]

    # Query subject is "Priya"; surfaced entities (Atlas/NeurIPS) don't match.
    filtered = filter_current_state_results(
        "What job title does my collaborator Priya hold now?", results
    )

    assert filtered == results


def test_current_state_filter_keeps_episodes_without_entity_state() -> None:
    results = [
        {"result_type": "episode", "episode": {"id": "ep_old"}},
        {"result_type": "cue_episode", "episode": {"id": "ep_cue"}},
    ]

    assert filter_current_state_results("What does Falcon use now?", results) == results
