"""Tests for get_context durable Decision/Preference pack."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.budgets import RecallBudget
from engram.retrieval.context_builder import (
    DURABLE_CONTEXT_PACKET_SCOPE,
    _durable_context_payload_from_manager,
    _list_durable_entities_by_type,
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
        self._entities = entities
        self.cfg = SimpleNamespace(
            recall_packet_explicit_limit=3,
            context_fast_preflight_timeout_ms=400,
            recall_fast_preflight_timeout_ms=400,
        )

    async def search_entities(
        self,
        group_id: str = "default",
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        rows = []
        for e in self._entities:
            if entity_type and e.entity_type != entity_type:
                continue
            if name and name.casefold() not in e.name.casefold():
                continue
            rows.append(
                {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "summary": e.summary,
                    "activation_score": 0.4,
                }
            )
            if len(rows) >= limit:
                break
        return rows


def _budget() -> RecallBudget:
    return RecallBudget.start(
        profile="explicit",
        surface="mcp",
        mode="mcp_context",
        max_wall_ms=4000,
        max_search_ms=1500,
        max_graph_ms=900,
        max_packet_ms=250,
        max_results=5,
        max_packets=3,
        max_output_tokens=1200,
        allow_deep_recall=True,
        allow_embeddings=True,
        allow_graph_probe=False,
    )


@pytest.mark.asyncio
async def test_list_durable_entities_skips_decision_statement_scrap():
    entities = [
        _Entity(
            "dec_noise",
            "MachineShopScheduler:decision_statement:Making a Decision",
            "Decision",
        ),
        _Entity(
            "dec_good",
            "LongMemEval is not Engram north star",
            "Decision",
            "continuity metric",
        ),
    ]
    hits = await _list_durable_entities_by_type(
        _Manager(entities),
        group_id="default",
        limit=5,
    )
    names = [h["entity"]["name"] for h in hits]
    assert "LongMemEval is not Engram north star" in names
    assert all("decision_statement" not in n for n in names)


@pytest.mark.asyncio
async def test_durable_context_payload_surfaces_strategy_decisions():
    entities = [
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
        _Entity(
            "dec_noise",
            "MachineShopScheduler:decision_statement:Approve caching",
            "Decision",
        ),
    ]
    manager = _Manager(entities)
    payload = await _durable_context_payload_from_manager(
        manager,
        group_id="default",
        topic_hint="strategy decisions LongMemEval sparse promotion",
        project_path="/Users/konnermoshier/Engram",
        format="structured",
        budget=_budget(),
        started=0.0,
    )
    assert payload is not None
    assert payload["entity_count"] >= 1
    assert payload["fact_count"] >= 1
    blob = payload["context"] + json_packets(payload)
    assert "LongMemEval is not Engram north star" in blob or "Prefer sparse" in blob
    assert "decision_statement" not in blob
    scopes = {
        p.get("_cache_scope") for p in (payload.get("cached_packets") or [])
    }
    assert DURABLE_CONTEXT_PACKET_SCOPE in scopes


def json_packets(payload: dict) -> str:
    return " ".join(
        str(p.get("title") or "") + " " + str(p.get("summary") or "")
        for p in (payload.get("cached_packets") or [])
    )


@pytest.mark.asyncio
async def test_durable_context_type_list_works_without_topic():
    entities = [
        _Entity(
            "dec_good",
            "Prefer markdown handoffs until proven",
            "Decision",
            "Use markdown until dogfood works",
        ),
    ]
    payload = await _durable_context_payload_from_manager(
        _Manager(entities),
        group_id="default",
        topic_hint=None,
        project_path="/Users/konnermoshier/Engram",
        format="structured",
        budget=_budget(),
        started=0.0,
    )
    assert payload is not None
    assert "Prefer markdown handoffs until proven" in payload["context"]
