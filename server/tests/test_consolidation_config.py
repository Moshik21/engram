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
