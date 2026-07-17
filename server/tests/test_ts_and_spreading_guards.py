"""Tests for Thompson sampling determinism/feedback guards + spreading degrade.

Covers M1.2 (seeded TS, no feedback flood at entity budget 0) and M1.5
(spreading re-score degrades with a marker instead of killing the recall).
"""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve
from engram.storage.helix.native_transport import NativeQueryError


class SpyActivationStore:
    """Activation store spy that counts writes."""

    def __init__(self):
        self.set_calls: list[tuple[str, object]] = []
        self.states: dict[str, object] = {}

    async def batch_get(self, node_ids):
        return {nid: self.states[nid] for nid in node_ids if nid in self.states}

    async def get_activation(self, node_id):
        return self.states.get(node_id)

    async def set_activation(self, node_id, state):
        self.set_calls.append((node_id, state))
        self.states[node_id] = state

    async def get_top_activated(self, group_id=None, limit=10, now=None):
        return []


class FakeSearchIndex:
    """Minimal search index: entity search + similarity, optional episodes."""

    _embeddings_enabled = False

    def __init__(self, results, episode_results=None, similarity_error=None):
        self._results = results
        self._episode_results = episode_results
        self._similarity_error = similarity_error
        self.similarity_calls = 0

    async def search(self, query, group_id, limit=10):
        return list(self._results)

    async def compute_similarity(self, query, entity_ids, group_id, query_embedding=None):
        self.similarity_calls += 1
        if self._similarity_error is not None:
            raise self._similarity_error
        return {}

    def __getattr__(self, name):
        # search_episodes is only present when episode results were provided,
        # so _optional_recall_capability treats it as a missing capability
        # (not a partial implementation) in entity-only tests.
        if name == "search_episodes" and self._episode_results is not None:

            async def search_episodes(query, group_id, limit=10):
                return list(self._episode_results)

            return search_episodes
        raise AttributeError(name)


class FakeGraphStore:
    """Minimal graph store; neighbors_map feeds spreading."""

    def __init__(self, neighbors_map=None):
        self._neighbors_map = neighbors_map or {}

    async def get_stats(self, group_id, exact=False):
        return {"entity_count": 2}

    async def get_active_neighbors_with_weights(self, entity_id, group_id=None, **_kwargs):
        return self._neighbors_map.get(entity_id, [])

    async def get_entity(self, entity_id, group_id=None):
        return None

    async def get_relationships(self, entity_id, group_id=None):
        return []

    async def get_episode_by_id(self, episode_id, group_id=None):
        return None


def _entity_scores(results):
    return {r.node_id: r.score for r in results if r.result_type == "entity"}


class TestThompsonSeededDeterminism:
    @pytest.mark.asyncio
    async def test_same_query_twice_identical_entity_scores(self):
        """M1.2b: seeded TS makes identical recalls score identically."""
        cfg = ActivationConfig(
            ts_enabled=True,
            multi_pool_enabled=False,
            episode_retrieval_enabled=False,
        )
        search_idx = FakeSearchIndex(results=[("e1", 0.9), ("e2", 0.7), ("e3", 0.5)])
        graph = FakeGraphStore()

        runs = []
        for _ in range(2):
            results = await retrieve(
                query="alpha beta project",
                group_id="default",
                graph_store=graph,
                activation_store=SpyActivationStore(),
                search_index=search_idx,
                cfg=cfg,
                record_feedback=False,
            )
            runs.append(_entity_scores(results))

        assert runs[0], "expected surfaced entity results"
        assert runs[0] == runs[1]


class TestFeedbackFloodGuard:
    @pytest.mark.asyncio
    async def test_budget_zero_recall_performs_no_activation_writes(self):
        """M1.2a: entity budget 0 → nothing surfaced → zero feedback writes."""
        cfg = ActivationConfig(
            ts_enabled=True,
            multi_pool_enabled=False,
            retrieval_strategy="passage_first",
            passage_first_entity_budget=0,
            episode_retrieval_enabled=True,
        )
        spy = SpyActivationStore()
        search_idx = FakeSearchIndex(
            results=[("e1", 0.9), ("e2", 0.7)],
            episode_results=[("ep1", 0.9)],
        )
        results = await retrieve(
            query="alpha beta project",
            group_id="default",
            graph_store=FakeGraphStore(),
            activation_store=spy,
            search_index=search_idx,
            cfg=cfg,
        )

        assert results, "episodes should still surface"
        assert all(r.result_type != "entity" for r in results)
        assert spy.set_calls == []

    @pytest.mark.asyncio
    async def test_surfaced_entity_recall_records_feedback(self):
        """Feedback still runs when entity results were actually surfaced."""
        cfg = ActivationConfig(
            ts_enabled=True,
            multi_pool_enabled=False,
            episode_retrieval_enabled=False,
        )
        spy = SpyActivationStore()
        search_idx = FakeSearchIndex(results=[("e1", 0.9), ("e2", 0.7)])
        results = await retrieve(
            query="alpha beta project",
            group_id="default",
            graph_store=FakeGraphStore(),
            activation_store=spy,
            search_index=search_idx,
            cfg=cfg,
            limit=1,
        )

        returned = {r.node_id for r in results if r.result_type == "entity"}
        assert returned == {"e1"}
        written = dict(spy.set_calls)
        # Positive feedback for the surfaced entity, negative for the dropped one.
        assert written["e1"].ts_alpha > 1.0
        assert written["e2"].ts_beta > 1.0


class TestSpreadingRescoreDegrade:
    @pytest.mark.asyncio
    async def test_native_failure_degrades_spreading_and_returns_results(self):
        """M1.5: NativeQueryError in the re-score degrades, recall survives."""
        cfg = ActivationConfig(
            ts_enabled=True,
            multi_pool_enabled=False,
            episode_retrieval_enabled=False,
        )
        search_idx = FakeSearchIndex(
            results=[("e1", 0.95)],
            similarity_error=NativeQueryError("compute_similarity", "boom", timeout=True),
        )
        graph = FakeGraphStore(neighbors_map={"e1": [("n1", 0.8, "RELATED_TO")]})
        stage_timings: dict[str, float] = {}

        results = await retrieve(
            query="alpha beta project",
            group_id="default",
            graph_store=graph,
            activation_store=SpyActivationStore(),
            search_index=search_idx,
            cfg=cfg,
            record_feedback=False,
            stage_timings_ms=stage_timings,
        )

        assert search_idx.similarity_calls >= 1
        returned = {r.node_id for r in results}
        assert "e1" in returned
        assert "n1" not in returned  # degraded stage contributed nothing
        assert "recall_spreading_rescore_degraded" in stage_timings

    @pytest.mark.asyncio
    async def test_non_native_failure_still_raises(self):
        """Only NativeQueryError is tolerated; other errors propagate."""
        cfg = ActivationConfig(
            ts_enabled=True,
            multi_pool_enabled=False,
            episode_retrieval_enabled=False,
        )
        search_idx = FakeSearchIndex(
            results=[("e1", 0.95)],
            similarity_error=ValueError("bug"),
        )
        graph = FakeGraphStore(neighbors_map={"e1": [("n1", 0.8, "RELATED_TO")]})

        with pytest.raises(ValueError):
            await retrieve(
                query="alpha beta project",
                group_id="default",
                graph_store=graph,
                activation_store=SpyActivationStore(),
                search_index=search_idx,
                cfg=cfg,
                record_feedback=False,
            )
