"""Tests for configuration loading."""

from pathlib import Path

import pytest

from engram.config import ActivationConfig, EngramConfig


class TestEngramConfig:
    def test_default_config(self, monkeypatch):
        monkeypatch.delenv("ENGRAM_MODE", raising=False)
        config = EngramConfig(_env_file=None)
        assert config.mode == "auto"
        assert config.default_group_id == "default"
        assert config.server.port == 8100

    def test_activation_defaults(self):
        config = EngramConfig()
        assert config.activation.decay_exponent == 0.5
        assert config.activation.spread_max_hops == 2
        assert config.activation.weight_semantic == 0.40
        assert config.activation.rediscovery_weight == 0.02
        assert config.activation.rediscovery_halflife_days == 30.0

    def test_sqlite_path_expansion(self, tmp_path: Path):
        config = EngramConfig(sqlite={"path": str(tmp_path / "test.db")})
        path = config.get_sqlite_path()
        assert path.parent.exists()

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("ENGRAM_MODE", "lite")
        config = EngramConfig()
        assert config.mode == "lite"


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
