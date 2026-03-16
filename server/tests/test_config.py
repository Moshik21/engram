"""Tests for configuration loading."""

from pathlib import Path

import pytest

from engram.config import DEFAULT_ENV_FILES, ActivationConfig, EngramConfig


class TestEngramConfig:
    def test_default_config(self, monkeypatch):
        config = EngramConfig(_env_file=None)
        assert config.default_group_id == "default"
        assert config.server.port == 8100

    def test_activation_defaults(self):
        config = EngramConfig()
        assert config.activation.decay_exponent == 0.5
        assert config.activation.spread_max_hops == 2
        assert config.activation.weight_semantic == 0.40
        assert config.activation.rediscovery_weight == 0.02
        assert config.activation.rediscovery_halflife_days == 30.0

    def test_helix_config_defaults(self):
        config = EngramConfig(_env_file=None)
        assert config.helix.host == "localhost"
        assert config.helix.port == 6969

    def test_default_env_file_order_includes_repo_root(self):
        repo_root_env = str(Path(__file__).resolve().parents[2] / ".env")
        assert DEFAULT_ENV_FILES[0].endswith(".engram/.env")
        assert DEFAULT_ENV_FILES[1] == repo_root_env
        assert DEFAULT_ENV_FILES[2] == ".env"


class TestActivationConfig:
    def test_decay_exponent_bounds(self):
        with pytest.raises(Exception):
            ActivationConfig(decay_exponent=0.0)
        with pytest.raises(Exception):
            ActivationConfig(decay_exponent=2.0)

    def test_valid_config(self):
        cfg = ActivationConfig(decay_exponent=0.3, spread_max_hops=3)
        assert cfg.decay_exponent == 0.3
        assert cfg.spread_max_hops == 3

    def test_progressive_projection_defaults(self):
        cfg = ActivationConfig()
        assert cfg.cue_layer_enabled is False
        assert cfg.cue_vector_index_enabled is True
        assert cfg.targeted_projection_enabled is True
        assert cfg.projector_v2_enabled is True
        assert cfg.projection_max_retries == 2
        assert cfg.cue_recall_enabled is False
        assert cfg.cue_recall_weight == 0.65
        assert cfg.cue_recall_max == 2
        assert cfg.cue_recall_hit_threshold == 2
