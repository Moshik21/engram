"""Tests for new config fields added in roadmap implementation."""

import pytest
from pydantic import ValidationError

from engram.config import ActivationConfig


class TestRoadmapConfigDefaults:
    """Verify default values for all new config fields."""

    def test_fan_s_max_default(self):
        cfg = ActivationConfig()
        assert cfg.fan_s_max == 3.5

    def test_fan_s_min_default(self):
        cfg = ActivationConfig()
        assert cfg.fan_s_min == 0.3

    def test_firing_threshold_new_default(self):
        cfg = ActivationConfig()
        assert cfg.spread_firing_threshold == 0.01

    def test_rrf_defaults(self):
        cfg = ActivationConfig()
        assert cfg.rrf_k == 60
        assert cfg.use_rrf is True

    def test_reranker_defaults(self):
        cfg = ActivationConfig()
        assert cfg.reranker_enabled is False
        assert cfg.reranker_top_n == 10

    def test_mmr_defaults(self):
        cfg = ActivationConfig()
        assert cfg.mmr_enabled is False
        assert cfg.mmr_lambda == 0.7

    def test_feedback_defaults(self):
        cfg = ActivationConfig()
        assert cfg.feedback_enabled is False
        assert cfg.feedback_ttl_days == 90

    def test_structure_aware_embeddings_default(self):
        cfg = ActivationConfig()
        assert cfg.structure_aware_embeddings is True

    def test_community_defaults(self):
        cfg = ActivationConfig()
        assert cfg.community_spreading_enabled is False
        assert cfg.community_bridge_boost == 1.5
        assert cfg.community_intra_dampen == 0.7
        assert cfg.community_stale_seconds == 300.0
        assert cfg.community_max_iterations == 10

    def test_context_gating_defaults(self):
        cfg = ActivationConfig()
        assert cfg.context_gating_enabled is False
        assert cfg.context_gate_floor == 0.3

    def test_multi_pool_defaults(self):
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


class TestRoadmapConfigConstraints:
    """Verify field constraints reject invalid values."""

    def test_fan_s_max_must_be_positive(self):
        with pytest.raises(ValidationError):
            ActivationConfig(fan_s_max=0.0)

    def test_fan_s_max_max_5(self):
        with pytest.raises(ValidationError):
            ActivationConfig(fan_s_max=5.1)

    def test_fan_s_min_max_constraint(self):
        with pytest.raises(ValidationError):
            ActivationConfig(fan_s_min=2.1)

    def test_rrf_k_min_1(self):
        with pytest.raises(ValidationError):
            ActivationConfig(rrf_k=0)

    def test_rrf_k_max_200(self):
        with pytest.raises(ValidationError):
            ActivationConfig(rrf_k=201)

    def test_reranker_top_n_min_1(self):
        with pytest.raises(ValidationError):
            ActivationConfig(reranker_top_n=0)

    def test_mmr_lambda_min_0(self):
        with pytest.raises(ValidationError):
            ActivationConfig(mmr_lambda=-0.1)

    def test_mmr_lambda_max_1(self):
        with pytest.raises(ValidationError):
            ActivationConfig(mmr_lambda=1.1)

    def test_feedback_ttl_days_min_1(self):
        with pytest.raises(ValidationError):
            ActivationConfig(feedback_ttl_days=0)

    def test_feedback_ttl_days_max_365(self):
        with pytest.raises(ValidationError):
            ActivationConfig(feedback_ttl_days=366)

    def test_multi_pool_search_limit_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(pool_search_limit=1)

    def test_multi_pool_total_limit_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(pool_total_limit=10)

    def test_multi_pool_graph_seed_count_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(pool_graph_seed_count=0)

    def test_community_bridge_boost_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(community_bridge_boost=0.5)

    def test_community_bridge_boost_max(self):
        with pytest.raises(ValidationError):
            ActivationConfig(community_bridge_boost=3.5)

    def test_community_intra_dampen_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(community_intra_dampen=0.05)

    def test_community_stale_seconds_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(community_stale_seconds=5.0)

    def test_community_max_iterations_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(community_max_iterations=0)

    def test_context_gate_floor_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(context_gate_floor=-0.1)

    def test_context_gate_floor_max(self):
        with pytest.raises(ValidationError):
            ActivationConfig(context_gate_floor=1.1)

    def test_valid_custom_values(self):
        cfg = ActivationConfig(
            fan_s_max=2.5,
            rrf_k=100,
            use_rrf=False,
            reranker_enabled=True,
            reranker_top_n=20,
            mmr_enabled=True,
            mmr_lambda=0.5,
            feedback_enabled=True,
            feedback_ttl_days=180,
            structure_aware_embeddings=True,
        )
        assert cfg.fan_s_max == 2.5
        assert cfg.rrf_k == 100
        assert cfg.use_rrf is False
        assert cfg.reranker_enabled is True
        assert cfg.reranker_top_n == 20
        assert cfg.mmr_enabled is True
        assert cfg.mmr_lambda == 0.5
        assert cfg.feedback_enabled is True
        assert cfg.feedback_ttl_days == 180
        assert cfg.structure_aware_embeddings is True
