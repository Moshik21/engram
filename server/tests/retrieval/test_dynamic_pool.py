"""Tests for dynamic pool sizing with sqrt(N/1000) scaling."""

from engram.config import ActivationConfig
from engram.retrieval.candidate_pool import compute_dynamic_limits
from engram.retrieval.router import QueryType


class TestComputeDynamicLimits:
    def test_at_1k_limits_match_defaults(self):
        """At 1k entities, scale factor ~1.0, limits should match cfg defaults."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(1000, cfg)

        assert limits["pool_search_limit"] == cfg.pool_search_limit
        assert limits["pool_activation_limit"] == cfg.pool_activation_limit
        assert limits["pool_graph_limit"] == cfg.pool_graph_limit

    def test_at_5k_limits_scale_up(self):
        """At 5k entities, scale factor ~2.24, limits should be ~2x defaults."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(5000, cfg)

        # sqrt(5000/1000) = 2.236
        assert limits["pool_search_limit"] >= cfg.pool_search_limit * 2
        assert limits["pool_activation_limit"] >= cfg.pool_activation_limit * 2
        assert limits["pool_graph_limit"] >= cfg.pool_graph_limit * 2
        assert limits["pool_total_limit"] > cfg.pool_total_limit

    def test_at_50k_limits_hit_caps(self):
        """At 50k entities, some limits should hit their upper bound caps."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(50000, cfg)

        # sqrt(50000/1000) = 7.07
        # pool_search_limit: 30 * 7.07 = 212 -> capped at 200
        assert limits["pool_search_limit"] == 200
        # pool_activation_limit: 20 * 7.07 = 141 -> capped at 100
        assert limits["pool_activation_limit"] == 100
        # pool_graph_limit: 20 * 7.07 = 141 -> capped at 100
        assert limits["pool_graph_limit"] == 100

    def test_temporal_boosts_activation_pool(self):
        """TEMPORAL query type should boost activation pool 3x."""
        cfg = ActivationConfig()
        default_limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)
        temporal_limits = compute_dynamic_limits(1000, cfg, QueryType.TEMPORAL)

        # Activation should be 3x for TEMPORAL
        assert temporal_limits["pool_activation_limit"] == min(
            cfg.pool_activation_limit * 3,
            100,
        )
        # Search should stay the same
        assert temporal_limits["pool_search_limit"] == default_limits["pool_search_limit"]

    def test_associative_boosts_graph_pool(self):
        """ASSOCIATIVE query type should boost graph pool 2x."""
        cfg = ActivationConfig()
        default_limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)
        assoc_limits = compute_dynamic_limits(1000, cfg, QueryType.ASSOCIATIVE)

        # Graph should be 2x for ASSOCIATIVE
        assert assoc_limits["pool_graph_limit"] == min(
            cfg.pool_graph_limit * 2,
            100,
        )
        # Search should stay the same
        assert assoc_limits["pool_search_limit"] == default_limits["pool_search_limit"]

    def test_direct_lookup_boosts_search_pool(self):
        """DIRECT_LOOKUP query type should boost search pool 2x."""
        cfg = ActivationConfig()
        default_limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)
        direct_limits = compute_dynamic_limits(1000, cfg, QueryType.DIRECT_LOOKUP)

        # Search should be 2x for DIRECT_LOOKUP
        assert direct_limits["pool_search_limit"] == min(
            cfg.pool_search_limit * 2,
            200,
        )
        # Activation should stay the same
        assert direct_limits["pool_activation_limit"] == default_limits["pool_activation_limit"]

    def test_default_no_boost(self):
        """DEFAULT query type should apply no multiplier."""
        cfg = ActivationConfig()
        no_type_limits = compute_dynamic_limits(1000, cfg)
        default_limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)

        assert no_type_limits == default_limits

    def test_creation_boosts_graph_pool(self):
        """CREATION query type should boost graph pool 2x (like ASSOCIATIVE)."""
        cfg = ActivationConfig()
        default_limits = compute_dynamic_limits(1000, cfg, QueryType.DEFAULT)
        creation_limits = compute_dynamic_limits(1000, cfg, QueryType.CREATION)

        # Graph should be 2x for CREATION
        assert creation_limits["pool_graph_limit"] == min(
            cfg.pool_graph_limit * 2,
            100,
        )
        # Search should stay the same
        assert creation_limits["pool_search_limit"] == default_limits["pool_search_limit"]


class TestDynamicLimitsEdgeCases:
    def test_zero_entities_uses_floor(self):
        """0 entities should use floor of 1000 (scale=1.0)."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(0, cfg)
        assert limits["pool_search_limit"] == cfg.pool_search_limit

    def test_small_corpus_uses_floor(self):
        """Corpus below 1000 should still use scale=1.0."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(100, cfg)
        assert limits["pool_search_limit"] == cfg.pool_search_limit

    def test_combined_scaling_and_boost(self):
        """5k TEMPORAL should combine 2.24x scale with 3x activation boost."""
        cfg = ActivationConfig()
        limits = compute_dynamic_limits(5000, cfg, QueryType.TEMPORAL)

        # 20 * 2.236 * 3 = 134 -> capped at 100
        assert limits["pool_activation_limit"] == 100
