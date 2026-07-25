"""The graph-expansion probe must not claim the STORE is slow when it is not.

``graph_expand_timeout`` is one of the two keys that arm the recall graph gate
(``recall_graph_gate.PROBE_TIMEOUT_KEYS``). Arming it refuses every secondary
graph read for the rest of the request: entity-query pool, graph pool,
spreading activation, entity attributes.

Measured live (2026-07-24, warm helix shell, quiet profile) on a case-only A/B
-- the same six words, one letter's case changed:

    tell me about helixdb storage limits   expand_timeout=76.5ms  gate ARMED
                                           recall_stats=220.9ms (no timeout)
                                           recall_primary_search=7.4ms
    summarise the engram brain schedule    expand=3.1ms   gate open   spread reach 197
    summarise the Engram brain schedule    expand_timeout=77.3ms  ARMED  spread reach 0
    what were the anthropic model choices  expand=0.04ms  gate open   spread reach 497
    what were the Anthropic model choices  expand_timeout=77.3ms  ARMED  spread reach 0

The first row is the dishonest one: ``_extract_query_terms`` returns ``[]`` for
that all-lowercase query, so the stage issued ZERO graph reads, yet it recorded
"the store is over budget" and cost the recall its whole graph half while the
store was demonstrably answering in single-digit milliseconds.

So: a timeout with zero completed reads is recorded as ``graph_expand_starved``
and does not arm the gate. A timeout with reads behind it still records
``graph_expand_timeout`` and arms it, unchanged.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve
from engram.retrieval.recall_graph_gate import graph_probe_timed_out

pytestmark = pytest.mark.asyncio


@dataclass
class _Entity:
    id: str
    name: str
    summary: str = ""


@dataclass
class _Rel:
    source_id: str
    target_id: str
    predicate: str


class _SearchIndex:
    async def search(self, query: str, group_id: str, limit: int = 50):
        return [("ent_a", 0.9)]

    async def search_episodes(self, query: str, group_id: str, limit: int = 10):
        return [("ep_1", 0.5)]

    async def compute_similarity(self, query, entity_ids, group_id=None):
        return {}


class _Store:
    """Fast store. Any probe that fires against it is a false positive."""

    def __init__(self):
        self.reads = 0

    async def get_stats(self, group_id: str):
        return {"entity_count": 10}

    async def find_entity_candidates(self, term: str, group_id: str):
        self.reads += 1
        return [_Entity(id="ent_a", name="A")]

    async def get_relationships(self, entity_id: str, group_id: str | None = None):
        self.reads += 1
        return [_Rel(source_id=entity_id, target_id="ent_b", predicate="RELATES_TO")]

    async def get_entity(self, entity_id: str, group_id: str):
        self.reads += 1
        return _Entity(id=entity_id, name="B")

    async def find_entities(self, **kwargs):
        return []

    async def get_active_neighbors_with_weights(self, *args, **kwargs):
        return []

    async def get_episode_by_id(self, episode_id: str, group_id: str):
        return None


class _SlowStore(_Store):
    """Store whose graph reads are genuinely slow -- the honest probe case."""

    async def find_entity_candidates(self, term: str, group_id: str):
        self.reads += 1
        await asyncio.sleep(0.2)
        return [_Entity(id="ent_a", name="A")]


def _cfg(**overrides) -> ActivationConfig:
    base = {
        "multi_pool_enabled": False,
        "graph_query_expansion_enabled": True,
        "graph_query_expansion_timeout_ms": 40,
        "template_reformulation_enabled": False,
        "query_decomposition_enabled": False,
        "recall_planner_enabled": False,
        "chunk_search_enabled": False,
        "cue_recall_enabled": False,
        "mmr_enabled": False,
        "reranker_enabled": False,
        "working_memory_enabled": False,
    }
    base.update(overrides)
    return ActivationConfig(**base)


async def _run(cfg, store, query: str) -> dict[str, float]:
    stages: dict[str, float] = {}
    await retrieve(
        query=query,
        group_id="default",
        graph_store=store,
        activation_store=_Activation(),
        search_index=_SearchIndex(),
        cfg=cfg,
        limit=5,
        stage_timings_ms=stages,
    )
    return stages


class _Activation:
    async def batch_get(self, entity_ids):
        return {}

    async def get_top_activated(self, **kwargs):
        return []


class TestProbeHonesty:
    async def test_a_starved_stage_does_not_arm_the_gate(self):
        """Zero completed reads -> not evidence about the store."""
        store = _Store()

        async def _never_scheduled(*args, **kwargs):
            await asyncio.sleep(1.0)
            return "unused"

        import engram.retrieval.graph_expansion as ge

        original = ge.expand_query_from_graph
        ge.expand_query_from_graph = _never_scheduled
        try:
            stages = await _run(_cfg(), store, "tell me about helixdb storage limits")
        finally:
            ge.expand_query_from_graph = original

        assert stages.get("graph_expand_starved") is not None
        assert "graph_expand_timeout" not in stages
        assert not graph_probe_timed_out(stages), "a zero-read stage armed the gate"

    async def test_a_genuinely_slow_store_still_arms_the_gate(self):
        """Control: the probe must still fire for the case it was built for."""
        store = _SlowStore()
        stages = await _run(_cfg(), store, "tell me about Helixdb storage limits")

        assert stages.get("graph_expand_timeout") is not None
        assert "graph_expand_starved" not in stages
        assert graph_probe_timed_out(stages)

    async def test_a_fast_store_neither_times_out_nor_starves(self):
        store = _Store()
        stages = await _run(_cfg(), store, "tell me about Helixdb storage limits")

        assert "graph_expand_timeout" not in stages
        assert "graph_expand_starved" not in stages
        assert stages.get("graph_expand") is not None
        assert not graph_probe_timed_out(stages)
        assert store.reads > 0, "fixture never exercised the expansion"
