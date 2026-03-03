"""Tests for consolidation config fields."""

import pytest
from pydantic import ValidationError

from engram.config import ActivationConfig


class TestConsolidationConfigDefaults:
    """Verify default values for all consolidation config fields."""

    def test_consolidation_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_enabled is False

    def test_consolidation_dry_run_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dry_run is True

    def test_consolidation_interval_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_interval_seconds == 3600.0

    def test_merge_defaults(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_merge_threshold == 0.88
        assert cfg.consolidation_merge_max_per_cycle == 50
        assert cfg.consolidation_merge_require_same_type is True

    def test_prune_defaults(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_prune_activation_floor == 0.05
        assert cfg.consolidation_prune_min_age_days == 30
        assert cfg.consolidation_prune_min_access_count == 0
        assert cfg.consolidation_prune_max_per_cycle == 100

    def test_infer_defaults(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_cooccurrence_min == 3
        assert cfg.consolidation_infer_confidence_floor == 0.6
        assert cfg.consolidation_infer_max_per_cycle == 50

    def test_compaction_defaults(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_compaction_horizon_days == 90
        assert cfg.consolidation_compaction_keep_min == 10
        assert cfg.consolidation_compaction_logarithmic is True


class TestConsolidationConfigValidation:
    """Verify field constraints reject invalid values."""

    def test_merge_threshold_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_merge_threshold=0.4)

    def test_merge_threshold_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_merge_threshold=1.1)

    def test_merge_max_per_cycle_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_merge_max_per_cycle=0)

    def test_prune_min_age_days_min(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_prune_min_age_days=0)

    def test_prune_min_age_days_max(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_prune_min_age_days=366)

    def test_infer_cooccurrence_min_below(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_cooccurrence_min=1)

    def test_compaction_horizon_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_compaction_horizon_days=20)

    def test_compaction_keep_min_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_compaction_keep_min=3)

    def test_interval_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_interval_seconds=30.0)

    def test_interval_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_interval_seconds=100000.0)

    def test_valid_custom_values(self):
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_dry_run=False,
            consolidation_merge_threshold=0.90,
            consolidation_merge_max_per_cycle=100,
            consolidation_prune_min_age_days=60,
            consolidation_infer_cooccurrence_min=5,
            consolidation_compaction_horizon_days=180,
        )
        assert cfg.consolidation_enabled is True
        assert cfg.consolidation_merge_threshold == 0.90
        assert cfg.consolidation_prune_min_age_days == 60


class TestTransitivityConfigDefaults:
    """Verify defaults and validation for transitivity config fields."""

    def test_transitivity_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_transitivity_enabled is False

    def test_transitivity_predicates_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_transitive_predicates == ["LOCATED_IN", "PART_OF"]

    def test_transitivity_decay_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_transitivity_decay == 0.8

    def test_transitivity_decay_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_transitivity_decay=0.05)


class TestReplayConfigDefaults:
    """Verify defaults and validation for replay config fields."""

    def test_replay_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_replay_enabled is False

    def test_replay_max_per_cycle_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_replay_max_per_cycle == 50

    def test_replay_window_hours_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_replay_window_hours == 24.0

    def test_replay_min_age_hours_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_replay_min_age_hours == 1.0

    def test_replay_max_per_cycle_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_replay_max_per_cycle=0)

    def test_replay_max_per_cycle_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_replay_max_per_cycle=501)

    def test_replay_window_hours_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_replay_window_hours=0.5)

    def test_replay_min_age_hours_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_replay_min_age_hours=49.0)


class TestPressureConfigDefaults:
    """Verify defaults and validation for pressure config fields."""

    def test_pressure_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_pressure_enabled is False

    def test_pressure_threshold_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_pressure_threshold == 100.0

    def test_pressure_threshold_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_pressure_threshold=0.0)

    def test_pressure_cooldown_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_pressure_cooldown_seconds == 300.0

    def test_reindex_max_per_cycle_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_reindex_max_per_cycle == 200


class TestDreamConfigDefaults:
    """Verify defaults and validation for dream spreading config fields."""

    def test_dream_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_enabled is False

    def test_dream_max_seeds_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_max_seeds == 20

    def test_dream_activation_band_defaults(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_activation_floor == 0.15
        assert cfg.consolidation_dream_activation_ceiling == 0.75
        assert cfg.consolidation_dream_activation_midpoint == 0.40

    def test_dream_weight_increment_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_weight_increment == 0.05

    def test_dream_max_boost_per_edge_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_max_boost_per_edge == 0.15

    def test_dream_max_edge_weight_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_max_edge_weight == 3.0

    def test_dream_min_boost_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_dream_min_boost == 0.005

    def test_dream_max_seeds_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_max_seeds=0)

    def test_dream_max_seeds_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_max_seeds=201)

    def test_dream_weight_increment_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_weight_increment=0.0)

    def test_dream_weight_increment_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_weight_increment=0.6)

    def test_dream_max_edge_weight_too_low(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_max_edge_weight=0.5)

    def test_dream_max_edge_weight_too_high(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_dream_max_edge_weight=11.0)


class TestPMIAndLLMConfigDefaults:
    """Verify defaults and validation for PMI/tf-idf and LLM config fields."""

    def test_pmi_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_pmi_enabled is False

    def test_pmi_min_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_pmi_min == 1.0

    def test_tfidf_weight_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_tfidf_weight == 0.3

    def test_llm_disabled_by_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_llm_enabled is False

    def test_llm_confidence_threshold_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_llm_confidence_threshold == 0.7

    def test_llm_max_per_cycle_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_llm_max_per_cycle == 20

    def test_llm_model_default(self):
        cfg = ActivationConfig()
        assert cfg.consolidation_infer_llm_model == "claude-haiku-4-5-20251001"

    def test_pmi_min_bounds(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_pmi_min=-1.0)
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_pmi_min=11.0)

    def test_tfidf_weight_bounds(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_tfidf_weight=-0.1)
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_tfidf_weight=1.1)

    def test_llm_confidence_threshold_bounds(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_llm_confidence_threshold=0.05)
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_llm_confidence_threshold=1.1)

    def test_llm_max_per_cycle_bounds(self):
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_llm_max_per_cycle=0)
        with pytest.raises(ValidationError):
            ActivationConfig(consolidation_infer_llm_max_per_cycle=101)
