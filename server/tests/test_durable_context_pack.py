"""Tests for get_context durable Decision/Preference pack."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.retrieval import context_builder as context_builder_mod
from engram.retrieval.budgets import RecallBudget
from engram.retrieval.context_builder import (
    DURABLE_CONTEXT_PACKET_SCOPE,
    _durable_context_payload_from_manager,
    _list_durable_entities_by_type,
    invalidate_durable_context_cache,
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


def _triple_entities() -> list[_Entity]:
    """Two relationship triples ahead of the real prose Decisions in list order."""
    return [
        _Entity(
            "trip_name",
            "Engram:full_mode_default_behavior:rework",
            "Decision",
            "Engram -> full_mode_default_behavior -> rework",
        ),
        _Entity(
            "trip_summary",
            "Engram public launch path",
            "Decision",
            "Engram -> public_launch_path -> OpenClaw",
        ),
        _Entity(
            "dec_index",
            "Cold Decision hit requires healthy search index",
            "Decision",
            "Product continuity fails if get_context cannot surface graph Decisions",
        ),
        _Entity(
            "dec_sparse",
            "Prefer sparse agent promotion",
            "Decision",
            "Passive observe + sparse remember",
        ),
    ]


@pytest.mark.asyncio
async def test_list_durable_entities_drops_relationship_triples():
    """Graph-edge triples must never take a briefing slot from a prose Decision.

    Regression pin for the live session-start defect: two of three "key memories
    to carry forward" were relationship triples (09813eb / d85de36 removed the
    same squatters from the rescue and reserved lane; this is the briefing).
    """
    hits = await _list_durable_entities_by_type(
        _Manager(_triple_entities()),
        group_id="default",
        limit=2,
    )
    names = [h["entity"]["name"] for h in hits]
    summaries = [h["entity"]["summary"] for h in hits]
    assert "Engram:full_mode_default_behavior:rework" not in names
    assert all("->" not in s for s in summaries)
    # Freed slots go to real prose Decisions, not to a shorter briefing.
    assert "Cold Decision hit requires healthy search index" in names
    assert "Prefer sparse agent promotion" in names


@pytest.mark.asyncio
async def test_durable_briefing_excludes_relationship_triples():
    invalidate_durable_context_cache()
    payload = await _durable_context_payload_from_manager(
        _Manager(_triple_entities()),
        group_id="triple_group",
        topic_hint=None,
        project_path="/Users/konnermoshier/Engram",
        format="briefing",
        budget=_budget(),
        started=time.perf_counter(),
    )
    invalidate_durable_context_cache("triple_group")
    assert payload is not None
    blob = payload["context"] + json_packets(payload)
    assert "Key memor" in payload["context"]
    assert "->" not in blob
    assert "full_mode_default_behavior" not in blob
    assert "Cold Decision hit requires healthy search index" in blob


@pytest.mark.asyncio
async def test_briefing_triple_filter_has_kill_switch():
    from engram.config import ActivationConfig

    class _KeepTriplesManager(_Manager):
        def get_activation_config(self) -> ActivationConfig:
            return ActivationConfig(recall_rescue_drop_triple_entities=False)

    hits = await _list_durable_entities_by_type(
        _KeepTriplesManager(_triple_entities()),
        group_id="default",
        limit=4,
    )
    names = [h["entity"]["name"] for h in hits]
    assert "Engram:full_mode_default_behavior:rework" in names


@pytest.mark.asyncio
async def test_list_durable_entities_dedupes_repeated_names():
    """One fact must not occupy every briefing slot (live: same Decision x3)."""
    entities = [
        _Entity("dup_a", "Cold Decision hit requires healthy search index", "Decision", "gate"),
        _Entity("dup_b", "Cold Decision hit requires healthy search index", "Decision", "gate 2"),
        _Entity("dup_c", "cold decision hit requires healthy search index", "Decision", "gate 3"),
        _Entity("dec_sparse", "Prefer sparse agent promotion", "Decision", "sparse remember"),
    ]
    hits = await _list_durable_entities_by_type(
        _Manager(entities),
        group_id="default",
        limit=3,
    )
    names = [h["entity"]["name"].casefold() for h in hits]
    assert names.count("cold decision hit requires healthy search index") == 1
    assert "prefer sparse agent promotion" in names


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
    scopes = {p.get("_cache_scope") for p in (payload.get("cached_packets") or [])}
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


@pytest.mark.asyncio
async def test_durable_context_process_cache_hit_on_second_call():
    invalidate_durable_context_cache()
    entities = [
        _Entity(
            "dec_good",
            "LongMemEval is not Engram north star",
            "Decision",
            "continuity metric",
        ),
    ]
    manager = _Manager(entities)
    first = await _durable_context_payload_from_manager(
        manager,
        group_id="cache_group",
        topic_hint="strategy decisions LongMemEval",
        project_path="/Users/konnermoshier/Engram",
        format="structured",
        budget=_budget(),
        started=time.perf_counter(),
    )
    assert first is not None
    assert first["packet_cache"]["hit"] is False

    second = await _durable_context_payload_from_manager(
        manager,
        group_id="cache_group",
        topic_hint="strategy decisions LongMemEval",
        project_path="/Users/konnermoshier/Engram",
        format="structured",
        budget=_budget(),
        started=time.perf_counter(),
    )
    assert second is not None
    assert second["packet_cache"]["hit"] is True
    assert "LongMemEval" in second["context"]
    assert second["diagnostics"]["stage_timings_ms"]["durable_context_cache_hit"] == 1.0
    invalidate_durable_context_cache("cache_group")


@pytest.mark.asyncio
async def test_durable_context_hard_budget_timeout_returns_none():
    invalidate_durable_context_cache()

    async def _slow_list(*_args, **_kwargs):
        await asyncio.sleep(2.0)
        return []

    manager = _Manager([])
    with (
        patch.object(
            context_builder_mod,
            "_DURABLE_CONTEXT_HARD_BUDGET_SECONDS",
            0.05,
        ),
        patch(
            "engram.retrieval.recall_surface._durable_entity_name_rescue",
            new=AsyncMock(side_effect=_slow_list),
        ),
        patch.object(
            context_builder_mod,
            "_list_durable_entities_by_type",
            new=AsyncMock(side_effect=_slow_list),
        ),
    ):
        payload = await _durable_context_payload_from_manager(
            manager,
            group_id="timeout_group",
            topic_hint="strategy",
            project_path=None,
            format="structured",
            budget=_budget(),
            started=time.perf_counter(),
        )
    assert payload is None
