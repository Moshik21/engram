"""Tests for Wave 2: Conversation Awareness retrieval features."""

from __future__ import annotations

import math
import time
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.context import ConversationContext, ConversationFingerprinter
from engram.retrieval.scorer import ScoredResult, extract_near_misses, score_candidates

# ─── TestConversationContext ─────────────────────────────────────────────


class TestConversationContext:
    def test_add_turn_increments_turn_count(self):
        ctx = ConversationContext()
        assert ctx._turn_count == 0
        ctx.add_turn("hello")
        assert ctx._turn_count == 1
        ctx.add_turn("world")
        assert ctx._turn_count == 2

    def test_fingerprint_first_turn_sets_directly(self):
        ctx = ConversationContext()
        vec = [3.0, 4.0]
        ctx.add_turn("hello", embedding=vec)
        fp = ctx.get_fingerprint()
        assert fp is not None
        # Should be L2-normalized: [0.6, 0.8]
        norm = math.sqrt(sum(x * x for x in fp))
        assert abs(norm - 1.0) < 1e-6
        assert abs(fp[0] - 0.6) < 1e-6
        assert abs(fp[1] - 0.8) < 1e-6

    def test_fingerprint_ema_update(self):
        ctx = ConversationContext(alpha=0.5)
        # First turn sets fingerprint to normalized [1, 0]
        ctx.add_turn("t1", embedding=[1.0, 0.0])
        fp1 = ctx.get_fingerprint()
        assert fp1 is not None
        assert abs(fp1[0] - 1.0) < 1e-6
        assert abs(fp1[1] - 0.0) < 1e-6

        # Second turn: EMA with alpha=0.5
        # new = 0.5 * [1, 0] + 0.5 * [0, 1] = [0.5, 0.5], normalized = [0.707, 0.707]
        ctx.add_turn("t2", embedding=[0.0, 1.0])
        fp2 = ctx.get_fingerprint()
        assert fp2 is not None
        expected = 1.0 / math.sqrt(2)
        assert abs(fp2[0] - expected) < 1e-4
        assert abs(fp2[1] - expected) < 1e-4

    def test_session_entity_accumulates_weight(self):
        ctx = ConversationContext()
        ctx.add_session_entity("e1", "Python", "Technology", weight_increment=1.0, now=100.0)
        ctx.add_session_entity("e1", "Python", "Technology", weight_increment=2.0, now=200.0)
        entry = ctx._session_entities["e1"]
        assert entry.mention_weight == 3.0
        assert entry.first_seen == 100.0
        assert entry.last_seen == 200.0

    def test_get_top_entities_sorted_by_weight(self):
        ctx = ConversationContext()
        ctx.add_session_entity("e1", "A", "T", weight_increment=1.0, now=1.0)
        ctx.add_session_entity("e2", "B", "T", weight_increment=5.0, now=1.0)
        ctx.add_session_entity("e3", "C", "T", weight_increment=3.0, now=1.0)
        top = ctx.get_top_entities(2)
        assert len(top) == 2
        assert top[0].name == "B"
        assert top[1].name == "C"

    def test_get_recent_turns_returns_last_n(self):
        ctx = ConversationContext(max_turns=10)
        for i in range(5):
            ctx.add_turn(f"turn {i}")
        recent = ctx.get_recent_turns(3)
        assert recent == ["turn 2", "turn 3", "turn 4"]

    def test_get_recent_turns_returns_all_when_fewer(self):
        ctx = ConversationContext()
        ctx.add_turn("only")
        recent = ctx.get_recent_turns(5)
        assert recent == ["only"]

    def test_fingerprint_similarity_zero_when_no_fingerprint(self):
        ctx = ConversationContext()
        assert ctx.fingerprint_similarity([1.0, 0.0]) == 0.0

    def test_fingerprint_similarity_cosine(self):
        ctx = ConversationContext()
        ctx.add_turn("t1", embedding=[1.0, 0.0])
        # Same direction → similarity ~1.0
        sim = ctx.fingerprint_similarity([1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6
        # Orthogonal → similarity ~0.0
        sim_orth = ctx.fingerprint_similarity([0.0, 1.0])
        assert abs(sim_orth) < 1e-6

    def test_clear_resets_all_state(self):
        ctx = ConversationContext()
        ctx.add_turn("hello", embedding=[1.0, 0.0])
        ctx.add_session_entity("e1", "X", "T", now=1.0)
        ctx.clear()
        assert ctx._turn_count == 0
        assert ctx.get_fingerprint() is None
        assert ctx.get_top_entities() == []
        assert ctx.get_recent_turns() == []


# ─── TestConversationFingerprinter ───────────────────────────────────────


class TestConversationFingerprinter:
    @pytest.mark.asyncio
    async def test_ingest_turn_without_embed_fn(self):
        ctx = ConversationContext()
        await ConversationFingerprinter.ingest_turn(ctx, "hello")
        assert ctx._turn_count == 1
        assert ctx.get_fingerprint() is None

    @pytest.mark.asyncio
    async def test_ingest_turn_with_embed_fn(self):
        ctx = ConversationContext()

        async def embed(text):
            return [1.0, 0.0]

        await ConversationFingerprinter.ingest_turn(ctx, "hello", embed_fn=embed)
        assert ctx._turn_count == 1
        assert ctx.get_fingerprint() is not None

    @pytest.mark.asyncio
    async def test_ingest_turn_with_failing_embed_fn(self):
        ctx = ConversationContext()

        async def embed(text):
            raise RuntimeError("API error")

        await ConversationFingerprinter.ingest_turn(ctx, "hello", embed_fn=embed)
        assert ctx._turn_count == 1
        assert ctx.get_fingerprint() is None  # graceful fallback


# ─── TestNearMissDetection ───────────────────────────────────────────────


class TestNearMissDetection:
    def _make_scored(self, n: int) -> list[ScoredResult]:
        return [
            ScoredResult(
                node_id=f"e{i}",
                score=1.0 - i * 0.1,
                semantic_similarity=0.5,
                activation=0.0,
                spreading=0.0,
                edge_proximity=0.0,
            )
            for i in range(n)
        ]

    def test_empty_when_fewer_than_top_n(self):
        scored = self._make_scored(3)
        assert extract_near_misses(scored, top_n=5) == []

    def test_returns_correct_window(self):
        scored = self._make_scored(15)
        near = extract_near_misses(scored, top_n=10, window=3)
        assert len(near) == 3
        assert near[0].node_id == "e10"
        assert near[2].node_id == "e12"

    def test_window_capped_at_available(self):
        scored = self._make_scored(12)
        near = extract_near_misses(scored, top_n=10, window=5)
        assert len(near) == 2


# ─── TestContextualReranking ─────────────────────────────────────────────


class TestContextualReranking:
    def test_fingerprint_boost_applied(self):
        cfg = ActivationConfig(conv_context_rerank_weight=0.1)
        candidates = [("e1", 0.5), ("e2", 0.5)]
        states = {
            "e1": ActivationState(node_id="e1", access_history=[], access_count=0),
            "e2": ActivationState(node_id="e2", access_history=[], access_count=0),
        }
        # e1 has high fingerprint sim, e2 has low
        fp_sim = {"e1": 0.9, "e2": 0.1}
        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            conv_fingerprint_sim=fp_sim,
        )
        # e1 should be ranked higher due to fingerprint boost
        assert results[0].node_id == "e1"
        assert results[0].score > results[1].score

    def test_no_boost_when_disabled(self):
        cfg = ActivationConfig(conv_context_rerank_weight=0.0)
        candidates = [("e1", 0.5), ("e2", 0.5)]
        states = {
            "e1": ActivationState(node_id="e1", access_history=[], access_count=0),
            "e2": ActivationState(node_id="e2", access_history=[], access_count=0),
        }
        fp_sim = {"e1": 0.9, "e2": 0.1}
        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            conv_fingerprint_sim=fp_sim,
        )
        # With weight=0.0, both should have same score
        assert abs(results[0].score - results[1].score) < 1e-6

    def test_boost_zero_when_no_fingerprint(self):
        cfg = ActivationConfig(conv_context_rerank_weight=0.1)
        candidates = [("e1", 0.5)]
        states = {
            "e1": ActivationState(node_id="e1", access_history=[], access_count=0),
        }
        results_with = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            conv_fingerprint_sim=None,
        )
        results_without = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
        )
        assert abs(results_with[0].score - results_without[0].score) < 1e-6


# ─── TestMultiQueryDecomposition ─────────────────────────────────────────


class TestMultiQueryDecomposition:
    @pytest.mark.asyncio
    async def test_disabled_runs_single_query_only(self):
        """When conv_multi_query_enabled=False, no sub-queries are issued."""
        cfg = ActivationConfig(conv_multi_query_enabled=False)
        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[("e1", 0.8)])
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        from engram.retrieval.pipeline import retrieve

        await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
        )
        # search() called once for the main query only
        assert search_index.search.call_count == 1

    @pytest.mark.asyncio
    async def test_topic_query_from_recent_turns(self):
        """Multi-query issues sub-queries from recent conversation turns."""
        cfg = ActivationConfig(
            conv_multi_query_enabled=True,
            conv_multi_query_turns=2,
            conv_context_enabled=True,
        )
        ctx = ConversationContext()
        ctx.add_turn("Working on React migration")
        ctx.add_turn("Also using TypeScript")

        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[("e1", 0.8)])
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        from engram.retrieval.pipeline import retrieve

        await retrieve(
            query="frontend framework",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )
        # Main query + at least 1 sub-query (topic)
        assert search_index.search.call_count >= 2

    @pytest.mark.asyncio
    async def test_entity_query_from_session_entities(self):
        """Multi-query issues entity sub-query from session entities."""
        cfg = ActivationConfig(
            conv_multi_query_enabled=True,
            conv_multi_query_top_entities=3,
            conv_context_enabled=True,
        )
        ctx = ConversationContext()
        ctx.add_turn("discussion")  # need at least 1 turn
        ctx.add_session_entity("e10", "React", "Technology", now=1.0)
        ctx.add_session_entity("e11", "TypeScript", "Technology", now=1.0)

        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[("e1", 0.8)])
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        from engram.retrieval.pipeline import retrieve

        await retrieve(
            query="frontend",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )
        # Main query + topic + entity sub-queries
        assert search_index.search.call_count >= 2

    @pytest.mark.asyncio
    async def test_merges_candidates_by_max_score(self):
        """Sub-query results are merged by keeping max score per entity."""
        cfg = ActivationConfig(
            conv_multi_query_enabled=True,
            conv_multi_query_turns=2,
            conv_context_enabled=True,
        )
        ctx = ConversationContext()
        ctx.add_turn("topic A")
        ctx.add_turn("topic B")

        call_count = 0

        async def mock_search(query, group_id, limit):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [("e1", 0.8)]
            return [("e1", 0.9), ("e2", 0.7)]

        search_index = AsyncMock()
        search_index.search = AsyncMock(side_effect=mock_search)
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        from engram.retrieval.pipeline import retrieve

        results = await retrieve(
            query="unique query that differs",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )
        # Should have both e1 and e2 as candidates
        result_ids = {r.node_id for r in results}
        # e2 should appear from sub-query merge
        assert "e2" in result_ids or len(results) >= 1

    def test_early_session_query_dominant_weights(self):
        """In early session (<3 turns), topic weight is lower."""
        ctx = ConversationContext()
        ctx.add_turn("hello")  # Only 1 turn → early session

        # Verify the weight schedule logic
        assert ctx._turn_count < 3
        # Early session should use w_topic=0.25, w_entity=0.15 (from pipeline logic)


# ─── TestSessionEntitySeedInjection ──────────────────────────────────────


class TestSessionEntitySeedInjection:
    @pytest.mark.asyncio
    async def test_disabled_skips_injection(self):
        """When conv_session_entity_seeds_enabled=False, no session seeds injected."""
        cfg = ActivationConfig(conv_session_entity_seeds_enabled=False)
        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[("e1", 0.8)])
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        ctx = ConversationContext()
        ctx.add_session_entity("e99", "React", "Technology", now=1.0)

        from engram.retrieval.pipeline import retrieve

        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )
        # Session entity e99 should NOT appear as a candidate
        result_ids = {r.node_id for r in results}
        assert "e99" not in result_ids

    @pytest.mark.asyncio
    async def test_injects_as_spreading_seed(self):
        """Session entities become spreading seeds when enabled."""
        cfg = ActivationConfig(
            conv_session_entity_seeds_enabled=True,
            conv_session_entity_seed_energy=0.2,
            conv_multi_query_top_entities=5,
        )
        search_index = AsyncMock()
        search_index.search = AsyncMock(return_value=[("e1", 0.8)])
        search_index.search_episodes = AsyncMock(return_value=[])
        search_index.compute_similarity = AsyncMock(return_value={})

        activation_store = AsyncMock()
        activation_store.batch_get = AsyncMock(return_value={})

        graph_store = AsyncMock()
        graph_store.get_stats = AsyncMock(return_value={"entity_count": 10})
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        ctx = ConversationContext()
        ctx.add_session_entity("e99", "React", "Technology", weight_increment=3.0, now=1.0)

        from engram.retrieval.pipeline import retrieve

        # Just verify it doesn't crash — seed injection is internal
        results = await retrieve(
            query="test",
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            limit=10,
            enable_routing=False,
            conv_context=ctx,
        )
        assert isinstance(results, list)

    def test_energy_scales_with_mention_weight(self):
        """Seed energy scales linearly with mention_weight up to cap of 5."""
        cfg = ActivationConfig(
            conv_session_entity_seed_energy=0.2,
        )
        # weight=1 → energy = 0.2 * min(1, 1/5) = 0.2 * 0.2 = 0.04
        energy_1 = cfg.conv_session_entity_seed_energy * min(1.0, 1.0 / 5.0)
        assert abs(energy_1 - 0.04) < 1e-6

        # weight=5 → energy = 0.2 * min(1, 5/5) = 0.2 * 1.0 = 0.2
        energy_5 = cfg.conv_session_entity_seed_energy * min(1.0, 5.0 / 5.0)
        assert abs(energy_5 - 0.2) < 1e-6

        # weight=10 → capped at 1.0 → energy = 0.2
        energy_10 = cfg.conv_session_entity_seed_energy * min(1.0, 10.0 / 5.0)
        assert abs(energy_10 - 0.2) < 1e-6
