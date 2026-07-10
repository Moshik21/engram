"""Continuity startup warmup is enabled and wired."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.main import _start_continuity_warmup


def test_continuity_warmup_config_defaults():
    cfg = ActivationConfig()
    assert cfg.continuity_startup_warmup_enabled is True
    assert cfg.continuity_startup_warmup_timeout_ms >= 5000


def test_start_continuity_warmup_is_callable():
    assert callable(_start_continuity_warmup)
