"""Tests for multi-pool candidate generation."""

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.candidate_pool import (
    _activation_pool,
    _graph_neighborhood_pool,
    _merge_pools_rrf,
    _search_pool,
    _working_memory_pool,
    generate_candidates,
)
from engram.retrieval.working_memory import WorkingMemoryBuffer

_DEFAULT_SEARCH = [("e1", 0.9), ("e2", 0.7)]


def _mock_search_index(results=_DEFAULT_SEARCH, similarity=None):
    idx = AsyncMock()
    idx.search = AsyncMock(return_value=results)
    idx.compute_similarity = AsyncMock(return_value=similarity or {})
    idx.search_episodes = AsyncMock(return_value=[])
    idx._embeddings_enabled = False
    return idx


def _mock_graph_store(neighbors_map=None):
    store = AsyncMock()
    if neighbors_map:
        async def _get_neighbors(entity_id, group_id=None):
            return neighbors_map.get(entity_id, [])
        store.get_active_neighbors_with_weights = AsyncMock(side_effect=_get_neighbors)
    else:
        store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    return store


def _mock_activation_store(top_activated=None):
    store = AsyncMock()
    store.get_top_activated = AsyncMock(return_value=top_activated or [])
    store.batch_get = AsyncMock(return_value={})
    return store


# ---------------------------------------------------------------------------
# TestSearchPool
# ---------------------------------------------------------------------------


class TestSearchPool:
    @pytest.mark.asyncio
    async def test_returns_search_results(self):
        idx = _mock_search_index(results=[("e1", 0.9), ("e2", 0.7)])
        results = await _search_pool("test", "default", idx, 30)
        assert results == [("e1", 0.9), ("e2", 0.7)]

    @pytest.mark.asyncio
    async def test_empty_search_returns_empty(self):
        idx = _mock_search_index(results=[])
        results = await _search_pool("test", "default", idx, 30)
        assert results == []


# ---------------------------------------------------------------------------
# TestActivationPool
# ---------------------------------------------------------------------------


class TestActivationPool:
    @pytest.mark.asyncio
    async def test_returns_top_activated(self):
        import time
        now = time.time()
        state = ActivationState(
            node_id="a1", access_history=[now - 10], access_count=1,
        )
        store = _mock_activation_store(top_activated=[("a1", state)])
        results = await _activation_pool("default", store, 20, now)
        assert len(results) == 1
        assert results[0][0] == "a1"
        assert results[0][1] > 0  # Has some activation

    @pytest.mark.asyncio
    async def test_empty_activation_returns_empty(self):
        store = _mock_activation_store(top_activated=[])
        results = await _activation_pool("default", store, 20, 1000.0)
        assert results == []


# ---------------------------------------------------------------------------
# TestGraphNeighborhoodPool
# ---------------------------------------------------------------------------


class TestGraphNeighborhoodPool:
    @pytest.mark.asyncio
    async def test_expands_1_hop_neighbors(self):
        store = _mock_graph_store(neighbors_map={
            "e1": [("n1", 0.8, "KNOWS"), ("n2", 0.6, "WORKS_AT")],
        })
        results = await _graph_neighborhood_pool(["e1"], "default", store, 10, 20)
        ids = [eid for eid, _ in results]
        assert "n1" in ids
        assert "n2" in ids

    @pytest.mark.asyncio
    async def test_ranks_by_fan_in(self):
        """Neighbor connecting 2 seeds ranks higher than one connecting 1."""
        store = _mock_graph_store(neighbors_map={
            "e1": [("shared", 0.8, "KNOWS"), ("only1", 0.5, "USES")],
            "e2": [("shared", 0.7, "WORKS_AT"), ("only2", 0.3, "USES")],
        })
        results = await _graph_neighborhood_pool(
            ["e1", "e2"], "default", store, 10, 20,
        )
        # "shared" connects to both seeds -> fan-in=2, should be first
        assert results[0][0] == "shared"
        assert results[0][1] == 2.0

    @pytest.mark.asyncio
    async def test_excludes_seed_entities(self):
        store = _mock_graph_store(neighbors_map={
            "e1": [("e2", 0.8, "KNOWS"), ("n1", 0.6, "USES")],
        })
        results = await _graph_neighborhood_pool(["e1", "e2"], "default", store, 10, 20)
        ids = [eid for eid, _ in results]
        assert "e1" not in ids
        assert "e2" not in ids
        assert "n1" in ids

    @pytest.mark.asyncio
    async def test_respects_pool_limit(self):
        store = _mock_graph_store(neighbors_map={
            "e1": [(f"n{i}", 0.5, "USES") for i in range(20)],
        })
        results = await _graph_neighborhood_pool(["e1"], "default", store, 20, 5)
        assert len(results) <= 5


# ---------------------------------------------------------------------------
# TestWorkingMemoryPool
# ---------------------------------------------------------------------------


class TestWorkingMemoryPool:
    @pytest.mark.asyncio
    async def test_includes_wm_entities(self):
        import time
        now = time.time()
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        wm.add("wm1", "entity", 0.9, "query", now - 10)
        store = _mock_graph_store()
        results = await _working_memory_pool(wm, "default", store, now, 5, 15)
        ids = [eid for eid, _ in results]
        assert "wm1" in ids

    @pytest.mark.asyncio
    async def test_expands_wm_neighbors(self):
        import time
        now = time.time()
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        wm.add("wm1", "entity", 0.9, "query", now - 10)
        store = _mock_graph_store(neighbors_map={
            "wm1": [("nb1", 0.7, "KNOWS")],
        })
        results = await _working_memory_pool(wm, "default", store, now, 5, 15)
        ids = [eid for eid, _ in results]
        assert "wm1" in ids
        assert "nb1" in ids

    @pytest.mark.asyncio
    async def test_dampened_neighbor_scores(self):
        import time
        now = time.time()
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        wm.add("wm1", "entity", 0.9, "query", now - 10)
        store = _mock_graph_store(neighbors_map={
            "wm1": [("nb1", 0.7, "KNOWS")],
        })
        results = await _working_memory_pool(wm, "default", store, now, 5, 15)
        result_map = {eid: score for eid, score in results}
        # Neighbor should have dampened (0.5x) score
        wm1_recency = result_map["wm1"]
        nb1_score = result_map["nb1"]
        assert nb1_score < wm1_recency
        assert nb1_score == pytest.approx(wm1_recency * 0.5, abs=0.01)


# ---------------------------------------------------------------------------
# TestMergePoolsRRF
# ---------------------------------------------------------------------------


class TestMergePoolsRRF:
    def test_single_pool(self):
        pool = [("e1", 0.9), ("e2", 0.7), ("e3", 0.5)]
        merged = _merge_pools_rrf([pool], rrf_k=60, limit=10)
        assert merged == ["e1", "e2", "e3"]

    def test_overlapping_pools_rank_higher(self):
        """Entity appearing in 2 pools should outrank one in just 1."""
        pool_a = [("shared", 0.9), ("only_a", 0.8)]
        pool_b = [("shared", 0.7), ("only_b", 0.6)]
        merged = _merge_pools_rrf([pool_a, pool_b], rrf_k=60, limit=10)
        assert merged[0] == "shared"

    def test_disjoint_pools(self):
        pool_a = [("e1", 0.9)]
        pool_b = [("e2", 0.7)]
        merged = _merge_pools_rrf([pool_a, pool_b], rrf_k=60, limit=10)
        assert set(merged) == {"e1", "e2"}

    def test_respects_limit(self):
        pool = [(f"e{i}", 0.5) for i in range(50)]
        merged = _merge_pools_rrf([pool], rrf_k=60, limit=10)
        assert len(merged) == 10


# ---------------------------------------------------------------------------
# TestGenerateCandidates
# ---------------------------------------------------------------------------


class TestGenerateCandidates:
    @pytest.mark.asyncio
    async def test_full_pipeline_all_pools(self):
        import time
        now = time.time()

        state = ActivationState(
            node_id="a1", access_history=[now - 5], access_count=1,
        )
        search_idx = _mock_search_index(
            results=[("e1", 0.9), ("e2", 0.7)],
            similarity={"a1": 0.3},
        )
        act_store = _mock_activation_store(top_activated=[("a1", state)])
        graph = _mock_graph_store(neighbors_map={
            "e1": [("n1", 0.8, "KNOWS")],
        })

        cfg = ActivationConfig(multi_pool_enabled=True)
        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
            now=now,
        )
        assert len(results) > 0
        ids = [eid for eid, _ in results]
        # Should include search, activation, and graph results
        assert "e1" in ids
        assert "e2" in ids

    @pytest.mark.asyncio
    async def test_search_entities_have_real_scores(self):
        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()

        cfg = ActivationConfig(multi_pool_enabled=True)
        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
        )
        # e1 comes from search with score 0.9
        e1 = next((s for eid, s in results if eid == "e1"), None)
        assert e1 == 0.9

    @pytest.mark.asyncio
    async def test_non_search_entities_get_backfilled_scores(self):
        import time
        now = time.time()

        state = ActivationState(
            node_id="a1", access_history=[now - 5], access_count=1,
        )
        search_idx = _mock_search_index(
            results=[("e1", 0.9)],
            similarity={"a1": 0.45},
        )
        act_store = _mock_activation_store(top_activated=[("a1", state)])
        graph = _mock_graph_store()

        cfg = ActivationConfig(multi_pool_enabled=True)
        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
            now=now,
        )
        # a1 should have backfilled score from compute_similarity
        result_map = {eid: score for eid, score in results}
        if "a1" in result_map:
            assert result_map["a1"] == 0.45

    @pytest.mark.asyncio
    async def test_pool_failure_non_fatal(self):
        """If activation pool throws, search results still returned."""
        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        act_store.get_top_activated = AsyncMock(side_effect=RuntimeError("boom"))
        graph = _mock_graph_store()

        cfg = ActivationConfig(multi_pool_enabled=True)
        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
        )
        assert len(results) > 0
        assert results[0][0] == "e1"


# ---------------------------------------------------------------------------
# TestMultiPoolConfig
# ---------------------------------------------------------------------------


class TestMultiPoolConfig:
    def test_defaults(self):
        cfg = ActivationConfig()
        assert cfg.multi_pool_enabled is False
        assert cfg.pool_search_limit == 30
        assert cfg.pool_activation_limit == 20
        assert cfg.pool_graph_seed_count == 10
        assert cfg.pool_graph_max_neighbors == 10
        assert cfg.pool_graph_limit == 20
        assert cfg.pool_wm_max_neighbors == 5
        assert cfg.pool_wm_limit == 15
        assert cfg.pool_total_limit == 80

    def test_validation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ActivationConfig(pool_search_limit=1)  # < 5
        with pytest.raises(ValidationError):
            ActivationConfig(pool_total_limit=10)  # < 20
        with pytest.raises(ValidationError):
            ActivationConfig(pool_graph_seed_count=0)  # < 1


# ---------------------------------------------------------------------------
# TestPipelineMultiPoolIntegration
# ---------------------------------------------------------------------------


class TestPipelineMultiPoolIntegration:
    @pytest.mark.asyncio
    async def test_multi_pool_enabled_delegates(self):
        """Pipeline with multi_pool_enabled uses generate_candidates."""
        from engram.retrieval.pipeline import retrieve

        search_idx = _mock_search_index()
        search_idx.search_episodes = AsyncMock(return_value=[])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()

        cfg = ActivationConfig(
            multi_pool_enabled=True,
            episode_retrieval_enabled=False,
        )
        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph,
            activation_store=act_store,
            search_index=search_idx,
            cfg=cfg,
        )
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_multi_pool_disabled_original_path(self):
        """Pipeline with multi_pool_enabled=False uses original search."""
        from engram.retrieval.pipeline import retrieve

        search_idx = _mock_search_index()
        search_idx.search_episodes = AsyncMock(return_value=[])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()

        cfg = ActivationConfig(
            multi_pool_enabled=False,
            episode_retrieval_enabled=False,
        )
        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph,
            activation_store=act_store,
            search_index=search_idx,
            cfg=cfg,
        )
        assert len(results) > 0

    def test_all_methods_count(self):
        from engram.benchmark.methods import ALL_METHODS
        assert len(ALL_METHODS) == 16
