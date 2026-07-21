"""Tests for multi-pool candidate generation."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
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

    @pytest.mark.asyncio
    async def test_search_timeout_records_stage_timing(self):
        async def slow_search(**_kwargs):
            await asyncio.sleep(0.2)
            return [("e1", 0.9)]

        idx = _mock_search_index(results=[])
        idx.search = AsyncMock(side_effect=slow_search)
        stage_timings = {}

        results = await _search_pool(
            "test",
            "default",
            idx,
            30,
            timeout_seconds=0.01,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_primary_search_timeout"] >= 10


# ---------------------------------------------------------------------------
# TestActivationPool
# ---------------------------------------------------------------------------


class TestActivationPool:
    @pytest.mark.asyncio
    async def test_returns_top_activated(self):
        import time

        now = time.time()
        state = ActivationState(
            node_id="a1",
            access_history=[now - 10],
            access_count=1,
        )
        store = _mock_activation_store(top_activated=[("a1", state)])
        results = await _activation_pool("default", store, 20, now, ActivationConfig())
        assert len(results) == 1
        assert results[0][0] == "a1"
        assert results[0][1] > 0  # Has some activation

    @pytest.mark.asyncio
    async def test_empty_activation_returns_empty(self):
        store = _mock_activation_store(top_activated=[])
        results = await _activation_pool("default", store, 20, 1000.0, ActivationConfig())
        assert results == []

    @pytest.mark.asyncio
    async def test_live_cfg_override_reaches_pool_ranking(self):
        """A cfg-level B_mid override must change pool activation scores.

        Guards against the fresh-ActivationConfig() clone that silently
        ignored live overrides (M0.4 / Judge-3 anti-vacuous-arm).
        """
        import time

        now = time.time()
        state = ActivationState(
            node_id="a1",
            access_history=[now - 10],
            access_count=1,
        )
        store = _mock_activation_store(top_activated=[("a1", state)])
        base = await _activation_pool("default", store, 20, now, ActivationConfig())
        shifted = await _activation_pool(
            "default",
            store,
            20,
            now,
            ActivationConfig(B_mid=5.0),
        )
        assert base[0][0] == shifted[0][0] == "a1"
        assert base[0][1] != shifted[0][1]

    @pytest.mark.asyncio
    async def test_consolidated_strength_reaches_pool_ranking(self):
        """A zero-access entity with consolidated_strength scores above floor."""
        state = ActivationState(
            node_id="cs1",
            access_history=[],
            access_count=0,
            consolidated_strength=2.0,
        )
        store = _mock_activation_store(top_activated=[("cs1", state)])
        cfg = ActivationConfig()
        results = await _activation_pool("default", store, 20, 1000.0, cfg)

        from engram.activation.engine import compute_activation

        floor = compute_activation([], 1000.0, cfg)
        expected = compute_activation([], 1000.0, cfg, 2.0)
        assert results[0][1] == pytest.approx(expected)
        assert results[0][1] > floor


# ---------------------------------------------------------------------------
# TestGraphNeighborhoodPool
# ---------------------------------------------------------------------------


class TestGraphNeighborhoodPool:
    @pytest.mark.asyncio
    async def test_expands_1_hop_neighbors(self):
        store = _mock_graph_store(
            neighbors_map={
                "e1": [("n1", 0.8, "KNOWS"), ("n2", 0.6, "WORKS_AT")],
            }
        )
        results = await _graph_neighborhood_pool(["e1"], "default", store, 10, 20)
        ids = [eid for eid, _ in results]
        assert "n1" in ids
        assert "n2" in ids

    @pytest.mark.asyncio
    async def test_ranks_by_fan_in(self):
        """Neighbor connecting 2 seeds ranks higher than one connecting 1."""
        store = _mock_graph_store(
            neighbors_map={
                "e1": [("shared", 0.8, "KNOWS"), ("only1", 0.5, "USES")],
                "e2": [("shared", 0.7, "WORKS_AT"), ("only2", 0.3, "USES")],
            }
        )
        results = await _graph_neighborhood_pool(
            ["e1", "e2"],
            "default",
            store,
            10,
            20,
        )
        # "shared" connects to both seeds -> fan-in=2, should be first
        assert results[0][0] == "shared"
        assert results[0][1] == 2.0

    @pytest.mark.asyncio
    async def test_excludes_seed_entities(self):
        store = _mock_graph_store(
            neighbors_map={
                "e1": [("e2", 0.8, "KNOWS"), ("n1", 0.6, "USES")],
            }
        )
        results = await _graph_neighborhood_pool(["e1", "e2"], "default", store, 10, 20)
        ids = [eid for eid, _ in results]
        assert "e1" not in ids
        assert "e2" not in ids
        assert "n1" in ids

    @pytest.mark.asyncio
    async def test_respects_pool_limit(self):
        store = _mock_graph_store(
            neighbors_map={
                "e1": [(f"n{i}", 0.5, "USES") for i in range(20)],
            }
        )
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
        store = _mock_graph_store(
            neighbors_map={
                "wm1": [("nb1", 0.7, "KNOWS")],
            }
        )
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
        store = _mock_graph_store(
            neighbors_map={
                "wm1": [("nb1", 0.7, "KNOWS")],
            }
        )
        results = await _working_memory_pool(wm, "default", store, now, 5, 15)
        result_map = {eid: score for eid, score in results}
        # Neighbor should have dampened (0.5x) score
        wm1_recency = result_map["wm1"]
        nb1_score = result_map["nb1"]
        assert nb1_score < wm1_recency
        assert nb1_score == pytest.approx(wm1_recency * 0.5, abs=0.01)

    @pytest.mark.asyncio
    async def test_episode_entries_never_enter_entity_pool(self):
        """Episode-typed WM entries stay in the buffer but are skipped at
        consumption (session-pollution fix, M0.1)."""
        import time

        now = time.time()
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        wm.add("ep1", "episode", 1.0, "query", now - 5)
        wm.add("wm1", "entity", 0.9, "query", now - 10)
        store = _mock_graph_store()
        results = await _working_memory_pool(wm, "default", store, now, 5, 15)
        ids = [eid for eid, _ in results]
        assert "ep1" not in ids
        assert "wm1" in ids
        # Buffer itself keeps the episode entry (only consumption filters)
        buffer_ids = {item_id for item_id, _, _ in wm.get_candidates(now)}
        assert "ep1" in buffer_ids
        # Episode entries are not neighbor-expansion sources either
        called_ids = {
            call.kwargs.get("entity_id", call.args[0] if call.args else None)
            for call in store.get_active_neighbors_with_weights.await_args_list
        }
        assert "ep1" not in called_ids


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
            node_id="a1",
            access_history=[now - 5],
            access_count=1,
        )
        search_idx = _mock_search_index(
            results=[("e1", 0.9), ("e2", 0.7)],
            similarity={"a1": 0.3},
        )
        act_store = _mock_activation_store(top_activated=[("a1", state)])
        graph = _mock_graph_store(
            neighbors_map={
                "e1": [("n1", 0.8, "KNOWS")],
            }
        )

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
            node_id="a1",
            access_history=[now - 5],
            access_count=1,
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

    @pytest.mark.asyncio
    async def test_search_pool_timeout_keeps_activation_candidates(self):
        async def slow_search(**_kwargs):
            await asyncio.sleep(0.2)
            return [("e1", 0.9)]

        state = ActivationState(
            node_id="a1",
            access_history=[0.0],
            access_count=1,
        )
        search_idx = _mock_search_index(results=[])
        search_idx.search = AsyncMock(side_effect=slow_search)
        search_idx.compute_similarity = AsyncMock(return_value={"a1": 0.4})
        act_store = _mock_activation_store(top_activated=[("a1", state)])
        graph = _mock_graph_store()
        stage_timings = {}

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=ActivationConfig(
                multi_pool_enabled=True,
                retrieval_primary_search_timeout_ms=10,
                retrieval_activation_only_primary_timeout_short_circuit=False,
                retrieval_skip_similarity_backfill_after_primary_timeout=False,
            ),
            now=10.0,
            stage_timings_ms=stage_timings,
        )

        assert results == [("a1", 0.4)]
        assert stage_timings["recall_primary_search_timeout"] >= 10

    @pytest.mark.asyncio
    async def test_primary_timeout_skips_similarity_backfill_by_default(self):
        async def slow_search(**_kwargs):
            await asyncio.sleep(0.2)
            return []

        async def slow_similarity(**_kwargs):
            await asyncio.sleep(0.2)
            return {"e_native": 0.4}

        search_idx = _mock_search_index(results=[], similarity={})
        search_idx.search = AsyncMock(side_effect=slow_search)
        search_idx.compute_similarity = AsyncMock(side_effect=slow_similarity)
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.find_entity_candidates = AsyncMock(
            return_value=[
                Entity(
                    id="e_native",
                    name="Native Latency",
                    entity_type="Topic",
                    group_id="default",
                )
            ]
        )
        stage_timings = {}

        results = await generate_candidates(
            query='"Native Latency"',
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=ActivationConfig(
                multi_pool_enabled=True,
                retrieval_primary_search_timeout_ms=10,
            ),
            now=10.0,
            stage_timings_ms=stage_timings,
        )

        assert results == [("e_native", 0.0)]
        assert stage_timings["recall_primary_search_timeout"] >= 10
        assert stage_timings["recall_similarity_backfill_skipped_primary_timeout"] == 1.0
        search_idx.compute_similarity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_primary_timeout_activation_only_short_circuits_by_default(self):
        async def slow_search(**_kwargs):
            await asyncio.sleep(0.2)
            return []

        async def slow_similarity(**_kwargs):
            await asyncio.sleep(0.2)
            return {"a1": 0.4}

        now = 10.0
        state = ActivationState(node_id="a1", access_history=[9.0], access_count=1)
        search_idx = _mock_search_index(results=[], similarity={})
        search_idx.search = AsyncMock(side_effect=slow_search)
        search_idx.compute_similarity = AsyncMock(side_effect=slow_similarity)
        act_store = _mock_activation_store(top_activated=[("a1", state)])
        graph = _mock_graph_store()
        stage_timings = {}

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=ActivationConfig(
                multi_pool_enabled=True,
                retrieval_primary_search_timeout_ms=10,
            ),
            now=now,
            stage_timings_ms=stage_timings,
        )

        assert results == []
        assert stage_timings["recall_primary_search_timeout"] >= 10
        assert stage_timings["recall_activation_candidate_count"] == 1.0
        assert stage_timings["recall_activation_only_primary_timeout_short_circuit"] == 1.0
        assert stage_timings["recall_candidate_count"] == 0.0
        search_idx.compute_similarity.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graph_pool_timeout_keeps_search_candidates(self):
        async def slow_neighbors(**_kwargs):
            await asyncio.sleep(0.2)
            return [("n1", 0.8, "RELATED_TO")]

        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=slow_neighbors)
        stage_timings = {}

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=ActivationConfig(
                multi_pool_enabled=True,
                retrieval_graph_pool_timeout_ms=10,
            ),
            stage_timings_ms=stage_timings,
        )

        assert results == [("e1", 0.9)]
        assert stage_timings["recall_graph_pool_timeout"] >= 10

    @pytest.mark.asyncio
    async def test_auto_budget_profile_completes_slow_graph_pool(self):
        """Auto profiles must use the relaxed graph timeout through generate_candidates."""
        graph_delay_seconds = 0.12

        async def slow_neighbors(**_kwargs):
            await asyncio.sleep(graph_delay_seconds)
            return [("n1", 0.8, "RELATED_TO")]

        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=slow_neighbors)
        stage_timings: dict[str, float] = {}
        cfg = ActivationConfig(
            multi_pool_enabled=True,
            retrieval_graph_pool_timeout_ms=75,
            retrieval_graph_pool_timeout_auto_ms=250,
        )

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
            stage_timings_ms=stage_timings,
            budget_profile="auto_deep",
        )

        result_ids = {entity_id for entity_id, _score in results}
        assert "n1" in result_ids
        assert "recall_graph_pool" in stage_timings
        assert "recall_graph_pool_timeout" not in stage_timings
        assert stage_timings["recall_graph_pool"] >= graph_delay_seconds * 1000

    @pytest.mark.asyncio
    async def test_explicit_budget_profile_times_out_slow_graph_pool(self):
        """Explicit/default profiles keep the tight graph timeout through generate_candidates."""
        graph_delay_seconds = 0.12

        async def slow_neighbors(**_kwargs):
            await asyncio.sleep(graph_delay_seconds)
            return [("n1", 0.8, "RELATED_TO")]

        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=slow_neighbors)
        stage_timings: dict[str, float] = {}
        cfg = ActivationConfig(
            multi_pool_enabled=True,
            retrieval_graph_pool_timeout_ms=75,
            retrieval_graph_pool_timeout_auto_ms=250,
        )

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=cfg,
            stage_timings_ms=stage_timings,
            budget_profile="explicit",
        )

        result_ids = {entity_id for entity_id, _score in results}
        assert "n1" not in result_ids
        assert "recall_graph_pool_timeout" in stage_timings
        assert stage_timings["recall_graph_pool_timeout"] < graph_delay_seconds * 1000

    @pytest.mark.asyncio
    async def test_graph_pool_skips_after_probe_timeout_when_search_has_candidates(self):
        async def slow_neighbors(**_kwargs):
            await asyncio.sleep(0.2)
            return [("n1", 0.8, "RELATED_TO")]

        search_idx = _mock_search_index(results=[("e1", 0.9)])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=slow_neighbors)
        stage_timings = {"recall_stats_timeout": 75.0}

        results = await generate_candidates(
            query="test",
            group_id="default",
            search_index=search_idx,
            activation_store=act_store,
            graph_store=graph,
            cfg=ActivationConfig(multi_pool_enabled=True),
            stage_timings_ms=stage_timings,
        )

        assert results == [("e1", 0.9)]
        assert stage_timings["recall_graph_pool_skipped_probe_timeout"] == 0.0
        assert "recall_graph_pool_timeout" not in stage_timings
        graph.get_active_neighbors_with_weights.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestMultiPoolConfig
# ---------------------------------------------------------------------------


class TestMultiPoolConfig:
    def test_defaults(self):
        cfg = ActivationConfig()
        assert cfg.multi_pool_enabled is True
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
    async def test_retrieve_auto_budget_profile_relaxes_graph_pool(self):
        """retrieve() must thread auto budget profiles into candidate graph pooling."""
        from engram.retrieval.pipeline import retrieve

        graph_delay_seconds = 0.12
        neighbor_calls = 0

        async def neighbors_with_first_slow(
            entity_id: str,
            group_id: str | None = None,
            **_kwargs,
        ):
            nonlocal neighbor_calls
            neighbor_calls += 1
            if neighbor_calls == 1:
                await asyncio.sleep(graph_delay_seconds)
                return [("n1", 0.8, "RELATED_TO")]
            return []

        search_idx = _mock_search_index(results=[("e1", 0.9)])
        search_idx.search_episodes = AsyncMock(return_value=[])
        act_store = _mock_activation_store()
        graph = _mock_graph_store()
        graph.get_active_neighbors_with_weights = AsyncMock(side_effect=neighbors_with_first_slow)
        graph.get_entity = AsyncMock(return_value=None)
        graph.get_relationships = AsyncMock(return_value=[])
        stage_timings: dict[str, float] = {}
        cfg = ActivationConfig(
            multi_pool_enabled=True,
            episode_retrieval_enabled=False,
            retrieval_graph_pool_timeout_ms=75,
            retrieval_graph_pool_timeout_auto_ms=250,
        )

        await retrieve(
            query="test",
            group_id="default",
            graph_store=graph,
            activation_store=act_store,
            search_index=search_idx,
            cfg=cfg,
            stage_timings_ms=stage_timings,
            budget_profile="auto_lite",
        )

        assert "recall_graph_pool" in stage_timings
        assert "recall_graph_pool_timeout" not in stage_timings
        assert stage_timings["recall_graph_pool"] >= graph_delay_seconds * 1000

    @pytest.mark.asyncio
    async def test_wm_episode_entries_never_reach_results(self):
        """The single-pool WM injection skips episode-typed entries while
        entity-typed priming stays live (session-pollution fix, M0.1)."""
        import time

        from engram.retrieval.pipeline import retrieve

        now = time.time()
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        wm.add("ep_phantom", "episode", 1.0, "query", now - 5)
        wm.add("wm_ent", "entity", 0.9, "query", now - 10)

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
            working_memory=wm,
        )
        node_ids = [r.node_id for r in results]
        assert "ep_phantom" not in node_ids
        assert "wm_ent" in node_ids

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

        # 17 -> 16 after M5.3 deleted METHOD_THOMPSON (F4 KILL)
        assert len(ALL_METHODS) == 16
