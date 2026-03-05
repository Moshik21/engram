"""Tests for local embedding provider (FastEmbedProvider) and factory fallback logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from engram.config import EmbeddingConfig, EngramConfig
from engram.embeddings.provider import NoopProvider

# ---------------------------------------------------------------------------
# Check if fastembed is available
# ---------------------------------------------------------------------------
try:
    import fastembed  # noqa: F401

    HAS_FASTEMBED = True
except ImportError:
    HAS_FASTEMBED = False

requires_local = pytest.mark.skipif(not HAS_FASTEMBED, reason="fastembed not installed")


# ---------------------------------------------------------------------------
# FastEmbedProvider unit tests (require fastembed installed)
# ---------------------------------------------------------------------------


@requires_local
class TestFastEmbedProvider:
    """Tests that exercise the real FastEmbedProvider with ONNX inference."""

    @pytest.fixture()
    def provider(self):
        from engram.embeddings.provider import FastEmbedProvider

        return FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")

    def test_dimension(self, provider):
        assert provider.dimension() == 768

    @pytest.mark.asyncio
    async def test_embed_single(self, provider):
        vecs = await provider.embed(["Hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 768

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider):
        vecs = await provider.embed(["Hello", "World", "Test"])
        assert len(vecs) == 3
        for vec in vecs:
            assert len(vec) == 768

    @pytest.mark.asyncio
    async def test_embed_empty(self, provider):
        vecs = await provider.embed([])
        assert vecs == []

    @pytest.mark.asyncio
    async def test_embed_query(self, provider):
        vec = await provider.embed_query("test query")
        assert len(vec) == 768

    @pytest.mark.asyncio
    async def test_embed_query_cached(self, provider):
        """Second call should return cached result (same object)."""
        vec1 = await provider.embed_query("cached query test")
        vec2 = await provider.embed_query("cached query test")
        assert vec1 is vec2  # same list object from cache

    @pytest.mark.asyncio
    async def test_embed_query_cache_eviction(self, provider):
        """Cache evicts oldest entry when full."""
        # Fill cache beyond max
        for i in range(provider.CACHE_MAX_SIZE + 5):
            await provider.embed_query(f"query {i}")
        assert len(provider._query_cache) == provider.CACHE_MAX_SIZE


# ---------------------------------------------------------------------------
# Factory tests (mock fastembed to avoid real model download in CI)
# ---------------------------------------------------------------------------


class TestFactoryProviderResolution:
    """Test _create_embedding_provider logic without real fastembed."""

    def test_voyage_with_key(self):
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="voyage", api_key="test-key"),
        )
        provider = _create_embedding_provider(config)
        from engram.embeddings.provider import VoyageProvider

        assert isinstance(provider, VoyageProvider)

    def test_noop_explicit(self):
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="noop"),
        )
        provider = _create_embedding_provider(config)
        assert isinstance(provider, NoopProvider)

    def test_voyage_no_key_no_fastembed(self):
        """No API key + no fastembed → NoopProvider."""
        import builtins
        import os

        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="voyage", api_key=""),
        )
        env_clean = {k: v for k, v in os.environ.items() if k != "VOYAGE_API_KEY"}

        # Block fastembed import inside the factory's local import
        real_import = builtins.__import__

        def _no_fastembed(name, *args, **kwargs):
            if name == "fastembed":
                raise ImportError("no fastembed")
            return real_import(name, *args, **kwargs)

        with patch.dict("os.environ", env_clean, clear=True), \
             patch("builtins.__import__", side_effect=_no_fastembed):
            provider = _create_embedding_provider(config)
        assert isinstance(provider, NoopProvider)

    @requires_local
    def test_local_explicit(self):
        """provider=local with fastembed installed → FastEmbedProvider."""
        from engram.embeddings.provider import FastEmbedProvider
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="local"),
        )
        provider = _create_embedding_provider(config)
        assert isinstance(provider, FastEmbedProvider)
        assert provider.dimension() == 768
        # Verify dimensions config was updated
        assert config.embedding.dimensions == 768

    @requires_local
    def test_voyage_fallback_to_local(self):
        """provider=voyage but no API key + fastembed installed → FastEmbedProvider."""
        from engram.embeddings.provider import FastEmbedProvider
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="voyage", api_key=""),
        )
        with patch.dict("os.environ", {k: v for k, v in __import__("os").environ.items()
                                        if k != "VOYAGE_API_KEY"}, clear=True):
            provider = _create_embedding_provider(config)
        assert isinstance(provider, FastEmbedProvider)

    def test_local_model_config(self):
        """local_model config field works."""
        config = EmbeddingConfig(local_model="my-custom/model")
        assert config.local_model == "my-custom/model"
