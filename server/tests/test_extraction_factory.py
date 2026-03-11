"""Tests for extraction provider factory."""

from __future__ import annotations

import os
from unittest.mock import patch

from engram.config import EngramConfig
from engram.extraction.extractor import EntityExtractor
from engram.extraction.narrow_adapter import NarrowExtractorAdapter


def _make_config(**overrides) -> EngramConfig:
    config = EngramConfig()
    for k, v in overrides.items():
        object.__setattr__(config.activation, k, v)
    return config


class TestAutoFallback:
    """Auto mode falls back through the chain."""

    def test_auto_no_api_key_no_ollama_uses_narrow(self):
        """Without API key or Ollama, auto should fall back to narrow."""
        config = _make_config(extraction_provider="auto")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            import httpx as real_httpx

            with patch.object(real_httpx, "get", side_effect=Exception("connection refused")):
                from engram.extraction.factory import create_extractor

                extractor = create_extractor(config)
                assert isinstance(extractor, NarrowExtractorAdapter)

    def test_auto_with_api_key_uses_anthropic(self):
        """With API key, auto should select Anthropic."""
        config = _make_config(extraction_provider="auto")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}, clear=False):
            from engram.extraction.factory import create_extractor

            extractor = create_extractor(config)
            assert isinstance(extractor, EntityExtractor)


class TestExplicitProviders:
    """Explicit provider selection."""

    def test_explicit_narrow(self):
        """Explicit narrow should always work."""
        config = _make_config(extraction_provider="narrow")
        from engram.extraction.factory import create_extractor

        extractor = create_extractor(config)
        assert isinstance(extractor, NarrowExtractorAdapter)

    def test_explicit_anthropic_without_key_falls_back(self):
        """Explicit anthropic without key should fall back to narrow with warning."""
        config = _make_config(extraction_provider="anthropic")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}, clear=False):
            from engram.extraction.factory import create_extractor

            extractor = create_extractor(config)
            assert isinstance(extractor, NarrowExtractorAdapter)

    def test_explicit_anthropic_with_key(self):
        """Explicit anthropic with key should use EntityExtractor."""
        config = _make_config(extraction_provider="anthropic")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test-123"}, clear=False):
            from engram.extraction.factory import create_extractor

            extractor = create_extractor(config)
            assert isinstance(extractor, EntityExtractor)

    def test_explicit_ollama_unreachable_falls_back(self):
        """Explicit ollama when unreachable should fall back to narrow."""
        config = _make_config(extraction_provider="ollama")
        import httpx as real_httpx

        with patch.object(real_httpx, "get", side_effect=Exception("connection refused")):
            from engram.extraction.factory import create_extractor

            extractor = create_extractor(config)
            assert isinstance(extractor, NarrowExtractorAdapter)


class TestConfigDefaults:
    """Config defaults for extraction provider."""

    def test_default_is_auto(self):
        config = EngramConfig()
        assert config.activation.extraction_provider == "auto"

    def test_ollama_defaults(self):
        config = EngramConfig()
        assert config.activation.ollama_model == "llama3.1:8b"
        assert config.activation.ollama_base_url == "http://localhost:11434"
