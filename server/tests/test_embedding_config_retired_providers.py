"""Local FastEmbed is the only embedding provider; retired ones map to it loudly."""

from __future__ import annotations

import logging

import pytest

from engram.config import EmbeddingConfig, EngramConfig

try:
    import fastembed  # noqa: F401

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False


def test_default_provider_is_local():
    assert EmbeddingConfig().provider == "local"
    assert EngramConfig().embedding.provider == "local"


def test_only_local_and_noop_are_valid_providers():
    assert EmbeddingConfig(provider="noop").provider == "noop"
    assert EmbeddingConfig(provider="Local").provider == "local"
    with pytest.raises(Exception):
        EmbeddingConfig(provider="openai")


@pytest.mark.parametrize("retired", ["auto", "gemini", "voyage"])
def test_retired_providers_map_to_local_with_a_warning(retired, caplog):
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = EmbeddingConfig(provider=retired)
    assert cfg.provider == "local"
    assert any("retired" in r.getMessage() for r in caplog.records)


def test_retired_keys_are_dropped_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = EmbeddingConfig(api_key="sk-stale", gemini_model="g", model="voyage-4-lite")
    assert not hasattr(cfg, "api_key")
    assert not hasattr(cfg, "gemini_model")
    assert not hasattr(cfg, "model")
    messages = [r.getMessage() for r in caplog.records]
    assert any("ENGRAM_EMBEDDING__API_KEY" in m for m in messages)
    assert any("ENGRAM_EMBEDDING__GEMINI_MODEL" in m for m in messages)
    assert any("ENGRAM_EMBEDDING__MODEL" in m for m in messages)


def test_stale_env_leftovers_do_not_crash_startup(monkeypatch, caplog):
    monkeypatch.setenv("ENGRAM_EMBEDDING__PROVIDER", "gemini")
    monkeypatch.setenv("ENGRAM_EMBEDDING__API_KEY", "sk-stale")
    monkeypatch.setenv("ENGRAM_EMBEDDING__GEMINI_MODEL", "gemini-embedding-2-preview")
    monkeypatch.setenv("ENGRAM_EMBEDDING__MODEL", "voyage-4-lite")
    with caplog.at_level(logging.WARNING, logger="engram.config"):
        cfg = EngramConfig()
    assert cfg.embedding.provider == "local"
    assert any("ENGRAM_EMBEDDING__API_KEY" in r.getMessage() for r in caplog.records)


@pytest.mark.skipif(not HAS_FASTEMBED, reason="fastembed not installed")
def test_factory_ignores_external_api_keys_in_the_environment(monkeypatch):
    """A leftover GEMINI/VOYAGE key must not change the resolved provider."""
    from engram.embeddings.provider import FastEmbedProvider
    from engram.storage.factory import _create_embedding_provider

    monkeypatch.setenv("GEMINI_API_KEY", "sk-gemini-stale")
    monkeypatch.setenv("VOYAGE_API_KEY", "sk-voyage-stale")
    provider = _create_embedding_provider(EngramConfig())
    assert isinstance(provider, FastEmbedProvider)
    assert provider.dimension() == 768
    assert not provider.is_materialized  # known model: lazy, no ONNX load in the test
