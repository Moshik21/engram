"""Tests for local embedding provider (FastEmbedProvider) and factory fallback logic."""

from __future__ import annotations

import sys
from types import SimpleNamespace
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


def _model_materializes() -> bool:
    """Whether the real ONNX model can actually load from the local cache.

    Gating on `import fastembed` alone made these tests fail on any machine
    with an incomplete model cache (the exact incident the stable-cache fix
    addressed) — the tests should skip, not fail, when the model is absent.
    """
    if not HAS_FASTEMBED:
        return False
    from engram.embeddings.provider import FastEmbedProvider

    probe = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
    return probe._ensure_model() is not None


HAS_LOCAL_MODEL = _model_materializes()

requires_local_model = pytest.mark.skipif(
    not HAS_LOCAL_MODEL,
    reason="fastembed model not materialized in local cache",
)


# ---------------------------------------------------------------------------
# FastEmbedProvider unit tests (require fastembed + a materialized model)
# ---------------------------------------------------------------------------


@requires_local_model
class TestFastEmbedProvider:
    """Tests that exercise the real FastEmbedProvider with ONNX inference."""

    @pytest.fixture()
    def provider(self):
        from engram.embeddings.provider import FastEmbedProvider

        return FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")

    def test_dimension(self, provider):
        assert provider.dimension() == 768
        assert provider.is_materialized is False

    @pytest.mark.asyncio
    async def test_embed_single(self, provider):
        vecs = await provider.embed(["Hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 768
        assert provider.is_materialized is True

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
# Broken-model failure contract (no real model needed)
# ---------------------------------------------------------------------------


class TestBrokenModelContract:
    @pytest.fixture()
    def broken_provider(self):
        from engram.embeddings.provider import FastEmbedProvider

        with patch("importlib.util.find_spec", return_value=object()):
            provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")
        return provider

    @pytest.mark.asyncio
    async def test_broken_model_returns_whole_batch_failure(self, broken_provider):
        """Never [[]]*n — present-but-empty vectors poisoned stores."""
        with patch.object(broken_provider, "_ensure_model", return_value=None):
            vecs = await broken_provider.embed(["a", "b", "c"])
        assert vecs == []

    @pytest.mark.asyncio
    async def test_broken_model_query_returns_empty(self, broken_provider):
        with patch.object(broken_provider, "_ensure_model", return_value=None):
            vec = await broken_provider.embed_query("q")
        assert vec == []

    def test_broken_latch_expires(self, broken_provider):
        import time as _time

        broken_provider._model_broken_until = _time.monotonic() + 300
        assert broken_provider._broken_now() is True
        broken_provider._model_broken_until = _time.monotonic() - 1
        assert broken_provider._broken_now() is False

    def test_load_failure_sets_backoff_not_permanent_latch(self, broken_provider):
        class _Boom:
            def __init__(self, *a, **k):
                raise RuntimeError("no model.onnx")

        fake_fastembed = SimpleNamespace(TextEmbedding=_Boom)
        with patch.dict(sys.modules, {"fastembed": fake_fastembed}):
            assert broken_provider._ensure_model() is None
        assert broken_provider._broken_now() is True
        # Repairing the cache heals after the backoff without a restart.
        broken_provider._model_broken_until = 0.0

        class _Ok:
            embedding_size = 768

            def __init__(self, *a, **k):
                pass

        with patch.dict(sys.modules, {"fastembed": SimpleNamespace(TextEmbedding=_Ok)}):
            assert broken_provider._ensure_model() is not None


# ---------------------------------------------------------------------------
# Factory tests (mock fastembed to avoid real model download in CI)
# ---------------------------------------------------------------------------


class TestFactoryProviderResolution:
    """Test _create_embedding_provider logic without real fastembed."""

    def test_noop_explicit(self):
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="noop"),
        )
        provider = _create_embedding_provider(config)
        assert isinstance(provider, NoopProvider)

    def test_local_without_fastembed_falls_back_to_noop(self):
        """provider=local + no fastembed → NoopProvider (logged, vectors off)."""
        from engram.storage.factory import _create_embedding_provider

        config = EngramConfig(
            embedding=EmbeddingConfig(provider="local"),
        )
        with patch("importlib.util.find_spec", return_value=None):
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

    def test_local_model_config(self):
        """local_model config field works."""
        config = EmbeddingConfig(local_model="my-custom/model")
        assert config.local_model == "my-custom/model"

    def test_fastembed_default_model_initializes_lazily(self):
        """Known local models expose dimensions without loading the ONNX model."""
        from engram.embeddings.provider import FastEmbedProvider

        real_import = __import__("builtins").__import__

        def _fail_fastembed_import(name, *args, **kwargs):
            if name == "fastembed":
                raise AssertionError("fastembed should not import during init")
            return real_import(name, *args, **kwargs)

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("builtins.__import__", side_effect=_fail_fastembed_import),
        ):
            provider = FastEmbedProvider(model="nomic-ai/nomic-embed-text-v1.5")

        assert provider.dimension() == 768
        assert provider.is_materialized is False

    def test_fastembed_unknown_model_materializes_for_dimension(self):
        """Unknown local models still load once to discover the true dimension."""
        from engram.embeddings.provider import FastEmbedProvider

        class FakeTextEmbedding:
            embedding_size = 42

            def __init__(self, model_name: str, cache_dir: str | None = None, **kwargs) -> None:
                self.model_name = model_name
                self.cache_dir = cache_dir

        fake_fastembed = SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch.dict(sys.modules, {"fastembed": fake_fastembed}),
        ):
            provider = FastEmbedProvider(model="custom/model")

        assert provider.dimension() == 42
        assert provider.is_materialized is True


# --- memory bound -----------------------------------------------------------
# One 40k-char episode through nomic-embed (8192-token window) measured 7.4 GB;
# the startup outbox drain took the shell to 17 GB on a 16 GB Mac and Jetsam
# killed it (2026-09-02). The provider is the single choke point every embed
# call passes through, so the bound lives here.


class _RecordingModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int | None]] = []

    def embed(self, texts, batch_size=None):
        import numpy as np

        self.calls.append((list(texts), batch_size))
        return [np.zeros(4) for _ in texts]


def _provider_with_recording_model(**kwargs):
    from engram.embeddings.provider import FastEmbedProvider

    provider = FastEmbedProvider(dimensions=4, **kwargs)
    model = _RecordingModel()
    provider._model = model
    return provider, model


@pytest.mark.asyncio
async def test_local_embed_caps_text_length_and_batch_size() -> None:
    provider, model = _provider_with_recording_model()
    long_text = "x" * 40_000
    vecs = await provider.embed([long_text, "short"])
    assert len(vecs) == 2
    (texts, batch_size), = model.calls
    assert max(len(t) for t in texts) == provider.DEFAULT_MAX_CHARS
    assert texts[1] == "short"
    assert batch_size == provider.DEFAULT_BATCH_SIZE


@pytest.mark.asyncio
async def test_local_embed_cap_is_configurable_and_zero_disables() -> None:
    provider, model = _provider_with_recording_model(max_chars=100, batch_size=4)
    await provider.embed(["y" * 500])
    assert len(model.calls[0][0][0]) == 100 and model.calls[0][1] == 4

    provider, model = _provider_with_recording_model(max_chars=0)
    await provider.embed(["y" * 500])
    assert len(model.calls[0][0][0]) == 500


def test_factory_wires_local_max_chars() -> None:
    from engram.config import EmbeddingConfig

    assert EmbeddingConfig().local_max_chars == 2000
