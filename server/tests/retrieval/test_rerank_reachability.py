"""Ticket 20: the cross-encoder rerank must be REACHABLE from the early returns.

The defect: ``retrieve()`` has two early returns that surface episode/cue
candidates directly (``if not candidates:`` and the no-semantic-anchor guard).
Both sit ABOVE Step 5.5, so on the default ``passage_first`` tier — where
``passage_first_entity_budget=0`` means the episode/cue channel is the only
channel that reaches the caller — an empty entity pool made the reranker
unreachable, not merely ineffective. That is the third recorded independent
cause of the reranker measuring dead.

Every test here drives the REAL ``retrieve()``. The reranker stub inverts the
vector order, so "the rerank ran and its output survived" is observable as an
inverted result order, and "the rerank did not run" is observable as the
untouched vector order. A test that could not tell those apart would be
vacuous, which is the failure mode this file was written against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve

pytestmark = pytest.mark.asyncio


@dataclass
class _Episode:
    id: str
    content: str
    group_id: str = "default"


class _EmptyEntitySearchIndex:
    """Search index whose ENTITY lane is empty and whose episode lane is not.

    This is the live shape of the default tier when the entity pool comes back
    empty: episodes and cues are found, entities are not.
    """

    def __init__(self, episodes: list[tuple[str, float]]):
        self._episodes = episodes

    async def search(self, query: str, group_id: str, limit: int = 50):
        return []

    async def search_episodes(self, query: str, group_id: str, limit: int = 10):
        return self._episodes[:limit]

    async def compute_similarity(self, query, entity_ids, group_id=None):
        return {}


class _GraphStore:
    def __init__(self, episodes: dict[str, _Episode]):
        self._episodes = episodes
        self.episode_reads: list[str] = []

    async def get_stats(self, group_id: str) -> dict:
        return {"entity_count": 0}

    async def get_entity(self, entity_id: str, group_id: str):
        return None

    async def find_entities(self, **kwargs):
        return []

    async def find_entity_candidates(self, *args, **kwargs):
        return []

    async def get_relationships(self, *args, **kwargs):
        return []

    async def get_active_neighbors_with_weights(self, *args, **kwargs):
        return []

    async def get_episode_by_id(self, episode_id: str, group_id: str):
        self.episode_reads.append(episode_id)
        return self._episodes.get(episode_id)


class _ActivationStore:
    async def batch_get(self, entity_ids):
        return {}

    async def get_top_activated(self, **kwargs):
        return []


@dataclass
class _InvertingReranker:
    """Reranker that reverses the document order it is handed.

    Scores descend from ``len(docs)``, so the LAST document in becomes the
    highest-scoring one out. Any caller that applies the result produces an
    inverted order; any caller that discards it leaves the input order intact.
    """

    calls: list[list[str]] = field(default_factory=list)

    async def rerank(self, query: str, docs, top_n: int):
        self.calls.append([doc_id for doc_id, _ in docs])
        return [(doc_id, float(len(docs) - i)) for i, (doc_id, _) in enumerate(reversed(docs))]


def _cfg(**overrides) -> ActivationConfig:
    base = {
        "reranker_enabled": True,
        "reranker_rerank_episodes": True,
        "retrieval_strategy": "passage_first",
        "passage_first_entity_budget": 0,
        "multi_pool_enabled": False,
        "cue_recall_enabled": False,
        "chunk_search_enabled": False,
        "recall_planner_enabled": False,
        "mmr_enabled": False,
        "graph_query_expansion_enabled": False,
        "template_reformulation_enabled": False,
        "query_decomposition_enabled": False,
        "working_memory_enabled": False,
        "episode_retrieval_enabled": True,
        "episode_retrieval_max": 5,
        # No substage timeout: this file measures reachability, not latency.
        "retrieval_reranker_timeout_ms": 0,
        # This file pins one episode read per candidate; the length rule reads
        # candidates to weigh them and is not what is under test here.
        "recall_short_episode_floor_chars": 0,
    }
    base.update(overrides)
    return ActivationConfig(**base)


def _fixture(n: int = 4):
    """Episode-only corpus: descending vector scores, ascending ids."""
    episodes = {f"ep_{i}": _Episode(id=f"ep_{i}", content=f"body {i}") for i in range(n)}
    hits = [(f"ep_{i}", 0.9 - 0.1 * i) for i in range(n)]
    return _EmptyEntitySearchIndex(hits), _GraphStore(episodes)


async def _run(cfg, reranker, stage_timings=None):
    search_index, graph_store = _fixture()
    results = await retrieve(
        query="what is the answer",
        group_id="default",
        graph_store=graph_store,
        activation_store=_ActivationStore(),
        search_index=search_index,
        cfg=cfg,
        limit=10,
        reranker=reranker,
        stage_timings_ms=stage_timings if stage_timings is not None else {},
    )
    return results, graph_store


class TestRerankReachableFromEarlyReturn:
    async def test_empty_entity_pool_still_reaches_the_reranker(self):
        """THE TICKET. Entity pool empty -> episodes returned -> rerank must run."""
        reranker = _InvertingReranker()
        stages: dict[str, float] = {}
        results, _ = await _run(_cfg(), reranker, stages)

        assert results, "episode channel produced nothing; fixture is broken"
        assert reranker.calls, "reranker was never called on the early-return path"
        assert stages.get("recall_reranker_special_applied") == float(len(results))

    async def test_the_rerank_output_survives_to_the_caller(self):
        """Reachability is not enough: the new order must reach the result list.

        Without this assertion the test would pass on a rerank whose output is
        computed and discarded -- the bug class this repo is named for.
        """
        reranker = _InvertingReranker()
        results, _ = await _run(_cfg(), reranker)

        order = [r.node_id for r in results]
        assert order == ["ep_3", "ep_2", "ep_1", "ep_0"], order

    async def test_vector_order_is_the_control(self):
        """With no reranker the same fixture returns raw vector order."""
        results, _ = await _run(_cfg(reranker_enabled=False), None)
        order = [r.node_id for r in results]
        assert order == ["ep_0", "ep_1", "ep_2", "ep_3"], order

    async def test_documents_are_read_before_scoring(self):
        """Rerank must score real content, never empty strings (ticket 19)."""
        reranker = _InvertingReranker()
        _, graph_store = await _run(_cfg(), reranker)
        assert sorted(graph_store.episode_reads) == ["ep_0", "ep_1", "ep_2", "ep_3"]

    async def test_entity_only_rerank_records_that_it_did_nothing(self):
        """reranker_rerank_episodes=False has nothing to sort here -- say so."""
        reranker = _InvertingReranker()
        stages: dict[str, float] = {}
        results, _ = await _run(_cfg(reranker_rerank_episodes=False), reranker, stages)

        assert not reranker.calls
        assert stages.get("recall_reranker_special_entity_only") == 0.0
        assert [r.node_id for r in results] == ["ep_0", "ep_1", "ep_2", "ep_3"]

    async def test_noop_reranker_is_recorded_not_silent(self):
        class NoopReranker:
            async def rerank(self, query, docs, top_n):  # pragma: no cover - never called
                raise AssertionError("NoopReranker must not be invoked")

        stages: dict[str, float] = {}
        await _run(_cfg(), NoopReranker(), stages)
        assert stages.get("recall_reranker_skipped_noop") == 0.0

    async def test_gate_refusal_keeps_pre_rerank_order_and_records_it(self):
        """A refused document read must degrade, not reorder by empty strings."""
        from engram.retrieval.recall_graph_gate import GraphGateTimeoutError

        class _RefusingGraphStore(_GraphStore):
            async def get_episode_by_id(self, episode_id: str, group_id: str):
                raise GraphGateTimeoutError("get_episode_by_id", "graph_expand_timeout")

        search_index, _ = _fixture()
        graph_store = _RefusingGraphStore({})
        reranker = _InvertingReranker()
        stages: dict[str, float] = {}
        results = await retrieve(
            query="what is the answer",
            group_id="default",
            graph_store=graph_store,
            activation_store=_ActivationStore(),
            search_index=search_index,
            cfg=_cfg(),
            limit=10,
            reranker=reranker,
            stage_timings_ms=stages,
        )

        assert not reranker.calls
        assert stages.get("recall_reranker_skipped_probe_timeout") == 0.0
        assert [r.node_id for r in results] == ["ep_0", "ep_1", "ep_2", "ep_3"]

    async def test_empty_documents_are_not_scored(self):
        """Episodes the store cannot materialise must not be reranked as ''."""
        search_index, _ = _fixture()
        graph_store = _GraphStore({})  # every lookup misses
        reranker = _InvertingReranker()
        stages: dict[str, float] = {}
        results = await retrieve(
            query="what is the answer",
            group_id="default",
            graph_store=graph_store,
            activation_store=_ActivationStore(),
            search_index=search_index,
            cfg=_cfg(),
            limit=10,
            reranker=reranker,
            stage_timings_ms=stages,
        )

        assert not reranker.calls
        assert stages.get("recall_reranker_special_no_documents") == 0.0
        assert [r.node_id for r in results] == ["ep_0", "ep_1", "ep_2", "ep_3"]
