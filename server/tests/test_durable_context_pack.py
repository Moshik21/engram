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


# ---------------------------------------------------------------------------
# Prose-fragment squatters (task #17 / P12)
# ---------------------------------------------------------------------------

# Names observed live winning briefing slots, plus the census scrap classes.
FRAGMENT_NAMES_MUST_DROP = [
    'use X")',
    "I'll make sure",
    'X not Y")',
    "the user stated'",
    "it.",
    "If you",
    "Right now the benchmark",
    "MCP Contract\n\nOpenClaw should",
    "MACHINES/STATUS",
    "onnx/model_quantized.onnx",
    "N/A",
    "arm-B/G2",
    "6379/0",
    "7a04e113-4e1d-4b53-8a7e-9584b69f11ef/tasks",
]

# Real durable facts. A predicate that drops ANY of these is a failure even if
# it cleans up every fragment — a false positive silently deletes a memory.
DURABLE_NAMES_MUST_SURVIVE = [
    "Cold Decision hit requires healthy search index",
    "GOLDEN_DECISION_1783643390: LongMemEval is not product north star",
    "Konner Moshier",
    "decision: use lite for smoke tests",
    "LongMemEval is not Engram north star",
    "Prefer sparse agent promotion",
    "Prefer markdown handoffs until proven",
    "Engram public launch path",
    # Adversarial keeps: shapes a careless rule would eat.
    "The Engram Project",
    "The New York Times",
    "engram/server",
    "Right to repair",
    "I/O latency budget",
    "Konner's laptop",
    'Engram "native" mode',
    "Node2Vec (n=50)",
    "Anthropic Inc.",
    "Use the cold brain for hygiene",
]


@pytest.mark.parametrize("name", FRAGMENT_NAMES_MUST_DROP)
def test_prose_fragment_predicate_drops_observed_scrap(name: str):
    from engram.extraction.promotion import is_prose_fragment_entity

    assert is_prose_fragment_entity(name) is True


@pytest.mark.parametrize("name", DURABLE_NAMES_MUST_SURVIVE)
def test_prose_fragment_predicate_keeps_real_memories(name: str):
    from engram.extraction.promotion import is_prose_fragment_entity

    assert is_prose_fragment_entity(name) is False


def _fragment_entities() -> list[_Entity]:
    """Prose fragments ahead of the real prose Decisions in list order."""
    return [
        _Entity("frag_quote", 'use X")', "Decision", "agent report text"),
        _Entity("frag_first_person", "I'll make sure", "Decision", "agent report text"),
        _Entity("frag_lead", "the user stated'", "Decision", "agent report text"),
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
            "Passive observe plus sparse remember keeps the graph high signal",
        ),
    ]


@pytest.mark.asyncio
async def test_list_durable_entities_drops_prose_fragments():
    """Sentence scrap must never take a briefing slot from a real Decision.

    Live regression pin: after the relationship triples were removed, prose
    fragments extracted from agent report text became the dominant squatter and
    the briefing was still ~2/3 junk.
    """
    hits = await _list_durable_entities_by_type(
        _Manager(_fragment_entities()),
        group_id="default",
        limit=2,
    )
    names = [h["entity"]["name"] for h in hits]
    assert 'use X")' not in names
    assert "I'll make sure" not in names
    assert "the user stated'" not in names
    # Freed slots go to real prose Decisions, not to a shorter briefing.
    assert "Cold Decision hit requires healthy search index" in names
    assert "Prefer sparse agent promotion" in names


@pytest.mark.asyncio
async def test_durable_briefing_excludes_prose_fragments():
    invalidate_durable_context_cache()
    payload = await _durable_context_payload_from_manager(
        _Manager(_fragment_entities()),
        group_id="fragment_group",
        topic_hint=None,
        project_path="/Users/konnermoshier/Engram",
        format="briefing",
        budget=_budget(),
        started=time.perf_counter(),
    )
    invalidate_durable_context_cache("fragment_group")
    assert payload is not None
    blob = payload["context"] + json_packets(payload)
    assert "Key memor" in payload["context"]
    assert "I'll make sure" not in blob
    assert "the user stated" not in blob
    assert "Cold Decision hit requires healthy search index" in blob


@pytest.mark.asyncio
async def test_prose_fragment_filter_has_kill_switch(monkeypatch):
    monkeypatch.setenv("ENGRAM_DROP_PROSE_FRAGMENT_ENTITIES", "0")
    hits = await _list_durable_entities_by_type(
        _Manager(_fragment_entities()),
        group_id="default",
        limit=5,
    )
    names = [h["entity"]["name"] for h in hits]
    assert "I'll make sure" in names


@pytest.mark.asyncio
async def test_list_durable_entities_dedupes_identical_summaries_under_distinct_names():
    """One FACT must not fill every slot just because it wears several names.

    The name dedupe cannot see this: the rows have distinct names and an
    identical summary body. Observed live as three briefing slots restating the
    same decision.
    """
    shared = "Product continuity fails if get_context cannot surface graph Decisions"
    entities = [
        _Entity("dup_a", "Cold Decision hit requires healthy search index", "Decision", shared),
        _Entity("dup_b", "Search index health gates cold Decision hits", "Decision", shared),
        _Entity("dup_c", "Healthy index is required for Decision recall", "Decision", shared),
        _Entity(
            "dec_sparse",
            "Prefer sparse agent promotion",
            "Decision",
            "Passive observe plus sparse remember keeps the graph high signal",
        ),
    ]
    hits = await _list_durable_entities_by_type(
        _Manager(entities),
        group_id="default",
        limit=3,
    )
    summaries = [h["entity"]["summary"] for h in hits]
    assert summaries.count(shared) == 1
    names = [h["entity"]["name"] for h in hits]
    assert "Prefer sparse agent promotion" in names


@pytest.mark.asyncio
async def test_summary_dedupe_keeps_rows_with_short_shared_labels():
    """Guard the dedupe knob: short shared summaries are labels, not facts."""
    entities = [
        _Entity("a", "Prefer lite for smoke tests", "Decision", "gate"),
        _Entity("b", "Prefer native for the dogfood brain", "Decision", "gate"),
    ]
    hits = await _list_durable_entities_by_type(
        _Manager(entities),
        group_id="default",
        limit=5,
    )
    names = [h["entity"]["name"] for h in hits]
    assert "Prefer lite for smoke tests" in names
    assert "Prefer native for the dogfood brain" in names


def test_packet_summary_collapses_repeated_clauses():
    """merge_entity_attributes appends '; {new}' forever; packets repeated it.

    Observed live: "Cold Decision hit...; Cold Decision hit...; Cold Decision
    hit..." — one packet restating its own title three times.
    """
    from engram.retrieval.packets import _packet_summary

    clause = "Cold Decision hit requires a healthy search index"
    entity = {"name": "Cold Decision hit", "summary": "; ".join([clause] * 3)}
    summary = _packet_summary("fact_packet", entity, [])
    assert summary.count("Cold Decision hit") == 1

    # Distinct clauses are updates, not noise — they must all survive.
    entity_two = {
        "name": "Cold Decision hit",
        "summary": f"{clause}; index rebuilt 2026-07-24; breaker pre-armed",
    }
    kept = _packet_summary("fact_packet", entity_two, [])
    assert "index rebuilt 2026-07-24" in kept
    assert "breaker pre-armed" in kept


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
