"""Tests for durable-entity rescue ranking and explicit cache short-circuit."""

from __future__ import annotations

import asyncio

from engram.retrieval.recall_surface import (
    _durable_entity_name_rescue,
    _is_decision_statement_noise,
    _name_query_overlap_score,
    _packets_satisfy_explicit_query,
    _rescue_query_tokens,
)


def test_decision_statement_noise_detected():
    assert _is_decision_statement_noise(
        "MachineShopScheduler:decision_statement:4. **Making a Decision**:"
    )
    assert _is_decision_statement_noise("Project:decision_statement:hook is on part_batches")
    assert not _is_decision_statement_noise("LongMemEval is not Engram north star")


def test_rescue_tokens_drop_generic_decision_words():
    tokens = _rescue_query_tokens(
        "what strategy decisions did we make about LongMemEval and ingestion?"
    )
    lowered = {t.casefold() for t in tokens}
    assert "decision" not in lowered
    assert "decisions" not in lowered
    assert "strategy" not in lowered
    assert any("longmemeval" in t.casefold() for t in tokens)


def test_name_overlap_requires_distinctive_content():
    good = _name_query_overlap_score(
        "LongMemEval is not Engram north star",
        "what strategy decisions about LongMemEval and ingestion?",
        "LongMemEval",
    )
    weak = _name_query_overlap_score(
        "MachineShopScheduler:decision_statement:Making a Decision",
        "what strategy decisions about LongMemEval and ingestion?",
        "decision",
    )
    assert good >= 0.35
    assert weak < 0.35


def test_project_file_packets_do_not_satisfy_explicit_query():
    packets = [
        {
            "packet_type": "project_home",
            "title": "Project File: docs/CURRENT_HANDOFF.md",
            "summary": "LongMemEval is not Engram north star. Prefer sparse agent promotion.",
            "provenance": ["file:docs/CURRENT_HANDOFF.md"],
            "entity_ids": [],
            "episode_ids": [],
            "_project_file_fallback_version": 2,
            "_cache_scope": "project_file_fallback",
        }
    ]
    assert (
        _packets_satisfy_explicit_query(
            packets,
            query="LongMemEval is not Engram north star sparse agent promotion",
        )
        is False
    )


def test_graph_entity_packets_can_satisfy_explicit_query():
    packets = [
        {
            "packet_type": "fact_packet",
            "title": "Fact: LongMemEval is not Engram north star",
            "summary": "Product metric is multi-agent continuity, not LME scores.",
            "provenance": ["entity:dec_abc"],
            "entity_ids": ["dec_abc"],
            "episode_ids": [],
        }
    ]
    assert (
        _packets_satisfy_explicit_query(
            packets,
            query="LongMemEval is not Engram north star",
        )
        is True
    )


class _Entity:
    def __init__(self, id: str, name: str, entity_type: str, summary: str = "") -> None:
        self.id = id
        self.name = name
        self.entity_type = entity_type
        self.summary = summary


class _Graph:
    def __init__(self, entities: list[_Entity]) -> None:
        self._entities = entities

    async def find_entity_candidates(self, name: str, group_id: str) -> list[_Entity]:
        needle = name.casefold()
        return [e for e in self._entities if needle in e.name.casefold()]


class _Manager:
    def __init__(self, entities: list[_Entity]) -> None:
        self._graph = _Graph(entities)


def test_durable_rescue_prefers_strategy_decision_over_statement_noise():
    entities = [
        _Entity(
            "dec_noise",
            "MachineShopScheduler:decision_statement:4. **Making a Decision**:",
            "Decision",
            "Cadence scrap",
        ),
        _Entity(
            "dec_good",
            "LongMemEval is not Engram north star",
            "Decision",
            "Product metric is multi-agent continuity",
        ),
        _Entity(
            "dec_sparse",
            "Prefer sparse agent promotion",
            "Decision",
            "Passive observe + sparse remember",
        ),
    ]
    hits = asyncio.run(
        _durable_entity_name_rescue(
            _Manager(entities),
            group_id="default",
            query="what strategy decisions did we make about LongMemEval and sparse promotion?",
            limit=5,
        )
    )
    names = [h["entity"]["name"] for h in hits]
    assert "LongMemEval is not Engram north star" in names
    assert all("decision_statement" not in n for n in names)
    assert hits[0]["entity"]["id"] in {"dec_good", "dec_sparse"}
