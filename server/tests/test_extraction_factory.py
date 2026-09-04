"""The internal extraction rung is narrow, always; retired rungs map to it loudly."""

from __future__ import annotations

import logging

import pytest

from engram.config import ActivationConfig, EngramConfig
from engram.extraction.factory import create_extractor
from engram.extraction.narrow_adapter import NarrowExtractorAdapter


def test_factory_returns_narrow_for_the_default_config():
    assert isinstance(create_extractor(EngramConfig()), NarrowExtractorAdapter)


def test_only_narrow_is_a_valid_provider():
    assert ActivationConfig(extraction_provider="narrow").extraction_provider == "narrow"
    with pytest.raises(Exception):
        ActivationConfig(extraction_provider="haiku")


@pytest.mark.parametrize("retired", ["auto", "anthropic", "ollama"])
def test_retired_providers_map_to_narrow_with_a_warning(retired, caplog):
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = ActivationConfig(extraction_provider=retired)
    assert cfg.extraction_provider == "narrow"
    assert any("retired" in r.getMessage() for r in caplog.records)


def test_retired_ollama_keys_are_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = ActivationConfig(ollama_model="x", ollama_base_url="http://h")
    assert not hasattr(cfg, "ollama_model")
    assert any("OLLAMA_MODEL" in r.getMessage() for r in caplog.records)


def test_env_leftovers_do_not_crash_startup(monkeypatch):
    monkeypatch.setenv("ENGRAM_ACTIVATION__EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("ENGRAM_ACTIVATION__OLLAMA_BASE_URL", "http://100.64.0.1:11434")
    monkeypatch.setenv("ENGRAM_ACTIVATION__OLLAMA_MODEL", "gemma")
    cfg = EngramConfig()
    assert cfg.activation.extraction_provider == "narrow"
