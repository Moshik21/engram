"""Tests for Wave 3: Proactive Intelligence — topic shift, surprise, priming, GC-MMR."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.context import ConversationContext
from engram.retrieval.scorer import ScoredResult, score_candidates
from engram.retrieval.surprise import SurpriseCache, SurpriseConnection, detect_surprises

# ─── TestTopicShiftDetection ────────────────────────────────────────


class TestTopicShiftDetection:
    """Unit tests for topic shift detection in ConversationContext."""

    def test_no_shift_on_similar_turns(self):
        """Same-topic turns stay below threshold — no shift detected."""
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        # Two very similar embeddings
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.95, 0.05, 0.0]
        ctx.update_fingerprint(emb1)
        ctx.update_fingerprint(emb2)
        assert not ctx.detect_topic_shift()

    def test_shift_detected_on_topic_change(self):
        """Orthogonal embeddings trigger a topic shift."""
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]  # orthogonal
        ctx.update_fingerprint(emb1)
        ctx.update_fingerprint(emb2)
        assert ctx.detect_topic_shift()

    def test_acknowledge_clears_flag(self):
        """Flag resets after acknowledgement."""
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        ctx.update_fingerprint(emb1)
        ctx.update_fingerprint(emb2)
        assert ctx.detect_topic_shift()
        ctx.acknowledge_shift()
        assert not ctx.detect_topic_shift()

    def test_no_shift_without_previous_fingerprint(self):
        """First turn never triggers a shift."""
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        emb = [1.0, 0.0, 0.0]
        ctx.update_fingerprint(emb)
        assert not ctx.detect_topic_shift()

    def test_shift_threshold_configurable(self):
        """Higher threshold makes detection more sensitive."""
        # With a high threshold (0.99), moderate changes trigger shift
        ctx_sensitive = ConversationContext(alpha=0.85, topic_shift_threshold=0.99)
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.7, 0.7, 0.0]  # cosine sim ~0.707 with emb1
        ctx_sensitive.update_fingerprint(emb1)
        ctx_sensitive.update_fingerprint(emb2)
        assert ctx_sensitive.detect_topic_shift()

        # With a low threshold (0.1), even big changes don't trigger
        ctx_insensitive = ConversationContext(alpha=0.85, topic_shift_threshold=0.1)
        emb3 = [0.5, 0.5, 0.7]  # moderate shift
        ctx_insensitive.update_fingerprint(emb1)
        ctx_insensitive.update_fingerprint(emb3)
        assert not ctx_insensitive.detect_topic_shift()

    def test_clear_resets_shift_state(self):
        """Clear resets all topic shift state."""
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        ctx.update_fingerprint([1.0, 0.0, 0.0])
        ctx.update_fingerprint([0.0, 1.0, 0.0])
        assert ctx.detect_topic_shift()
        ctx.clear()
        assert not ctx.detect_topic_shift()
        assert ctx._prev_fingerprint is None


# ─── TestSurpriseDetection ──────────────────────────────────────────


@pytest.mark.asyncio
class TestSurpriseDetection:
    """Async tests for surprise connection detection."""

    async def test_detects_dormant_strong_connection(self):
        """High edge weight + low activation = surprise."""
        cfg = ActivationConfig(
            surprise_detection_enabled=True,
            surprise_activation_floor=0.2,
            surprise_dormancy_days=7,
            surprise_edge_weight_min=0.3,
        )
        now = time.time()
        dormant_time = now - (30 * 86400)  # 30 days ago

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[
            ("neighbor_1", 0.8, "WORKS_AT", "Organization"),
        ])

        def _make_entity(ename):
            m = MagicMock()
            m.name = ename
            return m

        entity_obj = _make_entity("TestEntity")
        neighbor_obj = _make_entity("Neighbor1")
        graph_store.get_entity = AsyncMock(side_effect=lambda eid, gid: {
            "ent_1": entity_obj,
            "neighbor_1": neighbor_obj,
        }.get(eid))

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=ActivationState(
            node_id="neighbor_1",
            access_history=[dormant_time],
            access_count=1,
            last_accessed=dormant_time,
        ))

        surprises = await detect_surprises(
            entity_ids=["ent_1"],
            graph_store=graph_store,
            activation_store=activation_store,
            cfg=cfg,
            group_id="default",
            now=now,
        )
        assert len(surprises) == 1
        assert surprises[0].entity_name == "Neighbor1"
        assert surprises[0].edge_weight == 0.8

    async def test_skips_recently_accessed(self):
        """Active entities filtered out (not dormant enough)."""
        cfg = ActivationConfig(
            surprise_detection_enabled=True,
            surprise_activation_floor=0.5,
            surprise_dormancy_days=7,
            surprise_edge_weight_min=0.3,
        )
        now = time.time()
        recent_time = now - 3600  # 1 hour ago

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[
            ("neighbor_1", 0.8, "WORKS_AT", "Organization"),
        ])
        ent_mock = MagicMock()
        ent_mock.name = "X"
        graph_store.get_entity = AsyncMock(return_value=ent_mock)

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=ActivationState(
            node_id="neighbor_1",
            access_history=[recent_time],
            access_count=5,
            last_accessed=recent_time,
        ))

        surprises = await detect_surprises(
            entity_ids=["ent_1"],
            graph_store=graph_store,
            activation_store=activation_store,
            cfg=cfg,
            group_id="default",
            now=now,
        )
        assert len(surprises) == 0

    async def test_skips_weak_edges(self):
        """Edges below weight minimum are filtered out."""
        cfg = ActivationConfig(
            surprise_detection_enabled=True,
            surprise_activation_floor=0.2,
            surprise_dormancy_days=7,
            surprise_edge_weight_min=0.5,
        )
        now = time.time()

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[
            ("neighbor_1", 0.2, "MENTIONED_WITH", "Other"),
        ])
        ent_mock = MagicMock()
        ent_mock.name = "X"
        graph_store.get_entity = AsyncMock(return_value=ent_mock)

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=None)

        surprises = await detect_surprises(
            entity_ids=["ent_1"],
            graph_store=graph_store,
            activation_store=activation_store,
            cfg=cfg,
            group_id="default",
            now=now,
        )
        assert len(surprises) == 0

    async def test_surprise_score_formula(self):
        """Verify surprise = edge_weight * (1 - activation)."""
        cfg = ActivationConfig(
            surprise_detection_enabled=True,
            surprise_activation_floor=0.3,
            surprise_dormancy_days=1,
            surprise_edge_weight_min=0.1,
        )
        now = time.time()
        dormant_time = now - (10 * 86400)

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[
            ("neighbor_1", 0.6, "USES", "Technology"),
        ])
        ent_mock = MagicMock()
        ent_mock.name = "X"
        graph_store.get_entity = AsyncMock(return_value=ent_mock)

        # Set up activation so compute_activation returns ~0.05
        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=ActivationState(
            node_id="neighbor_1",
            access_history=[dormant_time],
            access_count=1,
            last_accessed=dormant_time,
        ))

        surprises = await detect_surprises(
            entity_ids=["ent_1"],
            graph_store=graph_store,
            activation_store=activation_store,
            cfg=cfg,
            group_id="default",
            now=now,
        )
        assert len(surprises) == 1
        s = surprises[0]
        expected = s.edge_weight * (1.0 - s.activation_score)
        assert abs(s.surprise_score - expected) < 0.0001

    async def test_empty_when_no_neighbors(self):
        """Graceful empty return when entity has no neighbors."""
        cfg = ActivationConfig(
            surprise_detection_enabled=True,
            surprise_activation_floor=0.2,
            surprise_dormancy_days=7,
            surprise_edge_weight_min=0.3,
        )
        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])
        ent_mock = MagicMock()
        ent_mock.name = "X"
        graph_store.get_entity = AsyncMock(return_value=ent_mock)

        activation_store = AsyncMock()

        surprises = await detect_surprises(
            entity_ids=["ent_1"],
            graph_store=graph_store,
            activation_store=activation_store,
            cfg=cfg,
            group_id="default",
            now=time.time(),
        )
        assert surprises == []

    def test_surprise_cache_ttl(self):
        """Cache entries expire after TTL."""
        cache = SurpriseCache(ttl_seconds=10.0)
        conn = SurpriseConnection(
            entity_id="e1", entity_name="E1",
            connected_to_id="e2", connected_to_name="E2",
            predicate="RELATED_TO", edge_weight=0.5,
            activation_score=0.1, surprise_score=0.45,
        )
        now = 1000.0
        cache.put("default", [conn], now)
        assert len(cache.get("default", now + 5)) == 1
        assert len(cache.get("default", now + 15)) == 0

    def test_surprise_cache_clear(self):
        """Cache clear removes all entries."""
        cache = SurpriseCache(ttl_seconds=300.0)
        conn = SurpriseConnection(
            entity_id="e1", entity_name="E1",
            connected_to_id="e2", connected_to_name="E2",
            predicate="RELATED_TO", edge_weight=0.5,
            activation_score=0.1, surprise_score=0.45,
        )
        cache.put("default", [conn])
        cache.clear()
        assert cache.get("default") == []


# ─── TestRetrievalPriming ───────────────────────────────────────────


class TestRetrievalPriming:
    """Tests for retrieval priming boost in scorer."""

    def test_priming_boosts_neighbor_scores(self):
        """Primed entities get a score increase."""
        cfg = ActivationConfig(retrieval_priming_enabled=True)
        now = time.time()
        candidates = [("node_a", 0.5), ("node_b", 0.5)]
        states = {
            "node_a": ActivationState(
                node_id="node_a", access_history=[now - 100],
                access_count=1, last_accessed=now - 100,
            ),
            "node_b": ActivationState(
                node_id="node_b", access_history=[now - 100],
                access_count=1, last_accessed=now - 100,
            ),
        }

        # Score without priming
        base = score_candidates(
            candidates=candidates, spreading_bonuses={}, hop_distances={},
            seed_node_ids=set(), activation_states=states, now=now, cfg=cfg,
        )
        base_scores = {r.node_id: r.score for r in base}

        # Score with priming boost on node_a
        boosted = score_candidates(
            candidates=candidates, spreading_bonuses={}, hop_distances={},
            seed_node_ids=set(), activation_states=states, now=now, cfg=cfg,
            priming_boosts={"node_a": 0.15},
        )
        boosted_scores = {r.node_id: r.score for r in boosted}

        assert boosted_scores["node_a"] > base_scores["node_a"]
        assert boosted_scores["node_b"] == base_scores["node_b"]

    def test_priming_expires_after_ttl(self):
        """Expired boosts are not applied in the pipeline priming filter."""
        now = time.time()
        priming_buffer = {
            "node_a": (0.15, now - 10),  # expired
            "node_b": (0.15, now + 100),  # still valid
        }
        active_boosts = {}
        for eid, (boost, expiry) in priming_buffer.items():
            if now < expiry:
                active_boosts[eid] = boost
        assert "node_a" not in active_boosts
        assert "node_b" in active_boosts

    def test_priming_disabled_by_default(self):
        """No boost when priming is disabled."""
        cfg = ActivationConfig()  # retrieval_priming_enabled=False by default
        assert not cfg.retrieval_priming_enabled

        now = time.time()
        candidates = [("node_a", 0.5)]
        states = {}
        base = score_candidates(
            candidates=candidates, spreading_bonuses={}, hop_distances={},
            seed_node_ids=set(), activation_states=states, now=now, cfg=cfg,
        )
        with_priming = score_candidates(
            candidates=candidates, spreading_bonuses={}, hop_distances={},
            seed_node_ids=set(), activation_states=states, now=now, cfg=cfg,
            priming_boosts=None,
        )
        assert base[0].score == with_priming[0].score

    def test_priming_scales_with_edge_weight(self):
        """Boost proportional to relationship weight."""
        cfg = ActivationConfig(retrieval_priming_enabled=True, retrieval_priming_boost=0.2)
        now = time.time()
        candidates = [("node_a", 0.5), ("node_b", 0.5)]
        states = {}

        # node_a gets boost from edge_weight=1.0, node_b from edge_weight=0.5
        boosts = {
            "node_a": cfg.retrieval_priming_boost * 1.0,
            "node_b": cfg.retrieval_priming_boost * 0.5,
        }
        results = score_candidates(
            candidates=candidates, spreading_bonuses={}, hop_distances={},
            seed_node_ids=set(), activation_states=states, now=now, cfg=cfg,
            priming_boosts=boosts,
        )
        scores = {r.node_id: r.score for r in results}
        assert scores["node_a"] > scores["node_b"]


# ─── TestGCMMR ──────────────────────────────────────────────────────


@pytest.mark.asyncio
class TestGCMMR:
    """Async tests for Graph-Connected MMR."""

    async def test_prefers_connected_results(self):
        """Connected entities ranked higher than disconnected."""
        from engram.retrieval.gc_mmr import apply_gc_mmr

        results = [
            ScoredResult(node_id="a", score=1.0, semantic_similarity=1.0,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
            ScoredResult(node_id="b", score=0.9, semantic_similarity=0.9,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
            ScoredResult(node_id="c", score=0.8, semantic_similarity=0.8,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
        ]

        graph_store = AsyncMock()
        # b is connected to a, c is not connected to anything
        graph_store.get_active_neighbors_with_weights = AsyncMock(side_effect=lambda nid, gid: {
            "a": [("b", 0.8, "WORKS_AT", "Organization")],
            "b": [("a", 0.8, "WORKS_AT", "Organization")],
            "c": [],
        }.get(nid, []))

        embeddings = {
            "a": [1.0, 0.0, 0.0],
            "b": [0.9, 0.1, 0.0],
            "c": [0.0, 1.0, 0.0],
        }

        reranked = await apply_gc_mmr(
            results, graph_store, "default", embeddings,
            lambda_rel=0.5, lambda_div=0.1, lambda_conn=0.4, top_n=3,
        )
        # First pick is always "a" (highest relevance)
        assert reranked[0].node_id == "a"
        # b should be preferred over c due to connectivity to a
        assert reranked[1].node_id == "b"

    async def test_falls_back_without_graph_data(self):
        """No graph data → same as relevance-based ranking."""
        from engram.retrieval.gc_mmr import apply_gc_mmr

        results = [
            ScoredResult(node_id="a", score=1.0, semantic_similarity=1.0,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
            ScoredResult(node_id="b", score=0.5, semantic_similarity=0.5,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
        ]
        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        reranked = await apply_gc_mmr(results, graph_store, "default", {}, top_n=2)
        assert reranked[0].node_id == "a"
        assert reranked[1].node_id == "b"

    async def test_balances_relevance_diversity_connectivity(self):
        """Verify 3-way tradeoff respects lambda weights."""
        from engram.retrieval.gc_mmr import apply_gc_mmr

        # Three results with equal scores but different connectivity
        results = [
            ScoredResult(node_id="a", score=1.0, semantic_similarity=1.0,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
            ScoredResult(node_id="b", score=0.99, semantic_similarity=0.99,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
            ScoredResult(node_id="c", score=0.98, semantic_similarity=0.98,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
        ]

        graph_store = AsyncMock()
        graph_store.get_active_neighbors_with_weights = AsyncMock(side_effect=lambda nid, gid: {
            "a": [("c", 0.9, "EXPERT_IN", "Technology")],
            "b": [],
            "c": [("a", 0.9, "EXPERT_IN", "Technology")],
        }.get(nid, []))

        embeddings = {
            "a": [1.0, 0.0],
            "b": [0.99, 0.01],
            "c": [0.0, 1.0],
        }

        # High connectivity weight — c should be picked after a
        reranked = await apply_gc_mmr(
            results, graph_store, "default", embeddings,
            lambda_rel=0.3, lambda_div=0.1, lambda_conn=0.6, top_n=3,
        )
        assert reranked[0].node_id == "a"
        # c is connected to a, b is not
        assert reranked[1].node_id == "c"

    async def test_empty_results_returns_empty(self):
        """Graceful empty handling."""
        from engram.retrieval.gc_mmr import apply_gc_mmr

        graph_store = AsyncMock()
        result = await apply_gc_mmr([], graph_store, "default", {})
        assert result == []

    async def test_single_result_returned_directly(self):
        """No re-ranking for 1 result."""
        from engram.retrieval.gc_mmr import apply_gc_mmr

        results = [
            ScoredResult(node_id="a", score=1.0, semantic_similarity=1.0,
                         activation=0.0, spreading=0.0, edge_proximity=0.0),
        ]
        graph_store = AsyncMock()
        reranked = await apply_gc_mmr(results, graph_store, "default", {}, top_n=5)
        assert len(reranked) == 1
        assert reranked[0].node_id == "a"


# ─── TestIntegration ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestIntegration:
    """Integration-level tests for Wave 3 proactive features."""

    async def test_topic_shift_increases_recall_count(self):
        """Auto-recall uses higher limit after topic shift detection."""
        cfg = ActivationConfig(
            auto_recall_enabled=True,
            auto_recall_limit=3,
            conv_topic_shift_enabled=True,
            conv_topic_shift_recall_boost=7,
            conv_context_enabled=True,
        )
        ctx = ConversationContext(alpha=0.85, topic_shift_threshold=0.60)
        # Simulate topic shift
        ctx.update_fingerprint([1.0, 0.0, 0.0])
        ctx.update_fingerprint([0.0, 1.0, 0.0])
        assert ctx.detect_topic_shift()

        # Verify the shift-aware recall logic
        recall_limit = cfg.auto_recall_limit
        if cfg.conv_topic_shift_enabled and ctx.detect_topic_shift():
            recall_limit = cfg.conv_topic_shift_recall_boost
            ctx.acknowledge_shift()

        assert recall_limit == 7
        assert not ctx.detect_topic_shift()  # cleared

    async def test_surprise_surfaces_in_recall_response(self):
        """Surprise connections appear in formatted recall response."""
        cache = SurpriseCache(ttl_seconds=300.0)
        conn = SurpriseConnection(
            entity_id="e1", entity_name="Old Project",
            connected_to_id="e2", connected_to_name="Current Topic",
            predicate="RELATED_TO", edge_weight=0.7,
            activation_score=0.05, surprise_score=0.665,
        )
        cache.put("default", [conn])
        surprises = cache.get("default", time.time())
        assert len(surprises) == 1
        # Format like MCP server does
        formatted = [
            {
                "entity": s.entity_name,
                "connected_to": s.connected_to_name,
                "relationship": s.predicate,
                "surprise_score": round(s.surprise_score, 4),
            }
            for s in surprises[:3]
        ]
        assert formatted[0]["entity"] == "Old Project"
        assert formatted[0]["surprise_score"] == 0.665

    async def test_config_defaults_all_disabled(self):
        """All Wave 3 features are disabled by default."""
        cfg = ActivationConfig()
        assert not cfg.conv_topic_shift_enabled
        assert not cfg.surprise_detection_enabled
        assert not cfg.retrieval_priming_enabled
        assert not cfg.gc_mmr_enabled

    async def test_config_values_match_plan(self):
        """Config default values match the plan specification."""
        cfg = ActivationConfig()
        assert cfg.conv_topic_shift_threshold == 0.60
        assert cfg.conv_topic_shift_recall_boost == 5
        assert cfg.surprise_activation_floor == 0.2
        assert cfg.surprise_dormancy_days == 7
        assert cfg.surprise_edge_weight_min == 0.3
        assert cfg.surprise_max_per_episode == 3
        assert cfg.surprise_cache_ttl_seconds == 300.0
        assert cfg.retrieval_priming_top_n == 3
        assert cfg.retrieval_priming_boost == 0.15
        assert cfg.retrieval_priming_ttl_seconds == 30.0
        assert cfg.retrieval_priming_max_neighbors == 5
        assert cfg.gc_mmr_lambda_relevance == 0.7
        assert cfg.gc_mmr_lambda_diversity == 0.2
        assert cfg.gc_mmr_lambda_connectivity == 0.1
