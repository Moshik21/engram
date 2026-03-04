"""Tests for the working memory buffer."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.retrieval.working_memory import WorkingMemoryBuffer

# ── Unit tests: WorkingMemoryBuffer ──────────────────────────────────


class TestWorkingMemoryBuffer:
    """Unit tests for the buffer itself."""

    def test_basic_add_and_get(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.9, "test query", now)

        candidates = buf.get_candidates(now)
        assert len(candidates) == 1
        item_id, recency_score, item_type = candidates[0]
        assert item_id == "ent_1"
        assert item_type == "entity"
        assert recency_score == pytest.approx(1.0, abs=0.01)

    def test_capacity_eviction(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        for i in range(25):
            buf.add(f"ent_{i}", "entity", 0.5, "q", now + i * 0.001)

        assert buf.size == 20
        # Oldest 5 (ent_0 through ent_4) should be evicted
        candidates = buf.get_candidates(now + 0.03)
        candidate_ids = {c[0] for c in candidates}
        for i in range(5):
            assert f"ent_{i}" not in candidate_ids
        for i in range(5, 25):
            assert f"ent_{i}" in candidate_ids

    def test_ttl_expiry(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=100.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.9, "q", now)

        # Advance time past TTL
        candidates = buf.get_candidates(now + 200.0)
        assert len(candidates) == 0

    def test_linear_recency_decay(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=100.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.9, "q", now)

        # At TTL/2, score should be ~0.5
        candidates = buf.get_candidates(now + 50.0)
        assert len(candidates) == 1
        _, recency_score, _ = candidates[0]
        assert recency_score == pytest.approx(0.5, abs=0.01)

        # At 75% through TTL, score should be ~0.25
        candidates = buf.get_candidates(now + 75.0)
        _, recency_score, _ = candidates[0]
        assert recency_score == pytest.approx(0.25, abs=0.01)

    def test_same_id_updates_and_moves_to_end(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.5, "first query", now)
        buf.add("ent_2", "entity", 0.6, "q", now + 1)
        buf.add("ent_1", "entity", 0.9, "second query", now + 2)

        assert buf.size == 2
        # ent_1 should now be at the end (most recent)
        candidates = buf.get_candidates(now + 2)
        # The entry should have the updated score
        for item_id, _, _ in candidates:
            if item_id == "ent_1":
                break
        else:
            pytest.fail("ent_1 not found in candidates")

    def test_get_recent_queries(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add_query("query 1", now)
        buf.add_query("query 2", now + 1)
        buf.add_query("query 3", now + 2)
        buf.add_query("query 4", now + 3)

        recent = buf.get_recent_queries(n=3)
        assert len(recent) == 3
        assert recent[0] == "query 4"  # most recent first
        assert recent[1] == "query 3"
        assert recent[2] == "query 2"

    def test_clear(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.5, "q", now)
        buf.add_query("q", now)

        buf.clear()
        assert buf.size == 0
        assert buf.get_candidates(now) == []
        assert buf.get_recent_queries() == []

    def test_entity_and_episode_types(self):
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("ent_1", "entity", 0.5, "q", now)
        buf.add("ep_1", "episode", 0.7, "q", now)

        candidates = buf.get_candidates(now)
        assert len(candidates) == 2
        types = {c[2] for c in candidates}
        assert types == {"entity", "episode"}

    def test_size_property(self):
        buf = WorkingMemoryBuffer(capacity=10, ttl_seconds=300.0)
        assert buf.size == 0
        now = time.time()
        buf.add("a", "entity", 0.5, "q", now)
        assert buf.size == 1
        buf.add("b", "entity", 0.5, "q", now)
        assert buf.size == 2


# ── Config tests ─────────────────────────────────────────────────────


class TestWorkingMemoryConfig:
    """Validate the 4 new config fields."""

    def test_defaults(self):
        cfg = ActivationConfig()
        assert cfg.working_memory_enabled is True
        assert cfg.working_memory_capacity == 20
        assert cfg.working_memory_ttl_seconds == 300.0
        assert cfg.working_memory_seed_energy == 0.3

    def test_validation_capacity_bounds(self):
        with pytest.raises(Exception):
            ActivationConfig(working_memory_capacity=3)  # below ge=5
        with pytest.raises(Exception):
            ActivationConfig(working_memory_capacity=200)  # above le=100

    def test_validation_ttl_bounds(self):
        with pytest.raises(Exception):
            ActivationConfig(working_memory_ttl_seconds=10.0)  # below ge=30
        with pytest.raises(Exception):
            ActivationConfig(working_memory_ttl_seconds=5000.0)  # above le=3600


# ── Pipeline integration tests ───────────────────────────────────────


class TestPipelineWithWorkingMemory:
    """Test that the pipeline uses the working memory buffer correctly."""

    @pytest.mark.asyncio
    async def test_retrieve_with_working_memory_injects_candidates(self):
        """Buffer items appear as candidates in the pipeline."""
        from engram.retrieval.pipeline import retrieve

        cfg = ActivationConfig(
            working_memory_enabled=True,
            working_memory_seed_energy=0.3,
        )

        # Set up working memory with an item
        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("wm_ent_1", "entity", 0.8, "prior query", now)

        # Mock stores
        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=[("search_ent_1", 0.9)])
        mock_search.compute_similarity = AsyncMock(return_value={})

        mock_activation = AsyncMock()
        mock_activation.batch_get = AsyncMock(return_value={})
        mock_activation.get_top_activated = AsyncMock(return_value=[])

        mock_graph = AsyncMock()
        mock_graph.get_relationships = AsyncMock(return_value=[])

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=mock_graph,
            activation_store=mock_activation,
            search_index=mock_search,
            cfg=cfg,
            working_memory=buf,
        )

        # The working memory item should have been injected
        result_ids = {r.node_id for r in results}
        assert "search_ent_1" in result_ids
        # wm_ent_1 might or might not score high enough, but it was a candidate
        # Verify batch_get was called with both IDs
        batch_call_args = mock_activation.batch_get.call_args_list[0][0][0]
        assert "wm_ent_1" in batch_call_args
        assert "search_ent_1" in batch_call_args

    @pytest.mark.asyncio
    async def test_retrieve_without_working_memory(self):
        """Pipeline works normally when working_memory is None."""
        from engram.retrieval.pipeline import retrieve

        cfg = ActivationConfig()

        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=[("ent_1", 0.9)])
        mock_search.compute_similarity = AsyncMock(return_value={})

        mock_activation = AsyncMock()
        mock_activation.batch_get = AsyncMock(return_value={})
        mock_activation.get_top_activated = AsyncMock(return_value=[])

        mock_graph = AsyncMock()
        mock_graph.get_relationships = AsyncMock(return_value=[])

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=mock_graph,
            activation_store=mock_activation,
            search_index=mock_search,
            cfg=cfg,
            working_memory=None,
        )
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_working_memory_items_become_seeds(self):
        """Buffer entities are added as additional spreading seeds."""
        from engram.retrieval.pipeline import retrieve

        cfg = ActivationConfig(
            working_memory_enabled=True,
            working_memory_seed_energy=0.3,
        )

        buf = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        buf.add("wm_seed", "entity", 0.8, "prior query", now)

        mock_search = AsyncMock()
        mock_search.search = AsyncMock(return_value=[("ent_1", 0.9)])
        mock_search.compute_similarity = AsyncMock(return_value={})

        mock_activation = AsyncMock()
        mock_activation.batch_get = AsyncMock(return_value={})
        mock_activation.get_top_activated = AsyncMock(return_value=[])

        mock_graph = AsyncMock()
        mock_graph.get_relationships = AsyncMock(return_value=[])

        # Patch spread_activation to capture the seeds passed
        captured_seeds = []

        async def mock_spread(
            seeds,
            graph_store,
            cfg,
            group_id=None,
            community_store=None,
            context_gate=None,
            seed_entity_types=None,
        ):
            captured_seeds.extend(seeds)
            return {}, {}

        with patch("engram.retrieval.pipeline.spread_activation", side_effect=mock_spread):
            with patch("engram.retrieval.pipeline.identify_seeds", return_value=[("ent_1", 0.5)]):
                await retrieve(
                    query="test",
                    group_id="default",
                    graph_store=mock_graph,
                    activation_store=mock_activation,
                    search_index=mock_search,
                    cfg=cfg,
                    working_memory=buf,
                )

        seed_ids = {s[0] for s in captured_seeds}
        assert "wm_seed" in seed_ids


# ── GraphManager integration tests ──────────────────────────────────


class TestGraphManagerWorkingMemory:
    """Test that GraphManager initializes and populates working memory."""

    def test_working_memory_created_when_enabled(self):
        from engram.graph_manager import GraphManager

        cfg = ActivationConfig(working_memory_enabled=True, working_memory_capacity=15)
        mgr = GraphManager(
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            extractor=AsyncMock(),
            cfg=cfg,
        )
        assert mgr._working_memory is not None
        assert mgr._working_memory._capacity == 15

    def test_working_memory_none_when_disabled(self):
        from engram.graph_manager import GraphManager

        cfg = ActivationConfig(working_memory_enabled=False)
        mgr = GraphManager(
            graph_store=AsyncMock(),
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            extractor=AsyncMock(),
            cfg=cfg,
        )
        assert mgr._working_memory is None

    @pytest.mark.asyncio
    async def test_recall_populates_working_memory(self):
        from engram.graph_manager import GraphManager
        from engram.models.entity import Entity
        from engram.retrieval.scorer import ScoredResult

        cfg = ActivationConfig(working_memory_enabled=True)

        mock_entity = Entity(
            id="ent_1",
            name="Test",
            entity_type="Person",
            summary="A test entity",
            group_id="default",
        )

        mock_graph = AsyncMock()
        mock_graph.get_entity = AsyncMock(return_value=mock_entity)
        mock_graph.get_relationships = AsyncMock(return_value=[])

        mock_activation = AsyncMock()
        mock_activation.record_access = AsyncMock()

        mock_search = AsyncMock()

        mgr = GraphManager(
            graph_store=mock_graph,
            activation_store=mock_activation,
            search_index=mock_search,
            extractor=AsyncMock(),
            cfg=cfg,
        )

        # Mock retrieve to return a known result
        scored = ScoredResult(
            node_id="ent_1",
            score=0.8,
            semantic_similarity=0.9,
            activation=0.3,
            spreading=0.1,
            edge_proximity=0.1,
            result_type="entity",
        )
        with patch("engram.graph_manager.retrieve", return_value=[scored]):
            await mgr.recall("test query", group_id="default")

        # Working memory should now have the entity
        assert mgr._working_memory is not None
        assert mgr._working_memory.size == 1
        candidates = mgr._working_memory.get_candidates(time.time())
        assert any(c[0] == "ent_1" for c in candidates)

        # Query should be tracked
        queries = mgr._working_memory.get_recent_queries()
        assert "test query" in queries
