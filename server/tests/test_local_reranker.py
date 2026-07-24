"""Tests for FastEmbedReranker and updated reranker factory."""

from __future__ import annotations

import pytest

from engram.retrieval.reranker import NoopReranker, create_reranker


class TestCreateReranker:
    """Test the create_reranker factory."""

    def test_noop_by_default(self):
        r = create_reranker()
        assert isinstance(r, NoopReranker)

    def test_noop_when_no_api_key(self):
        r = create_reranker(provider="cohere")
        assert isinstance(r, NoopReranker)

    def test_cohere_with_api_key(self):
        from engram.retrieval.reranker import CohereReranker

        r = create_reranker(api_key="test-key", provider="cohere")
        assert isinstance(r, CohereReranker)

    @pytest.mark.asyncio
    async def test_local_without_fastembed(self, monkeypatch):
        """Model load is lazy, so a missing fastembed surfaces at rerank time.

        The failure is non-fatal: rerank degrades to the input order instead of
        breaking recall.
        """
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "fastembed" in name:
                raise ImportError("fastembed not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        r = create_reranker(provider="local")
        docs = [("e1", "hello world"), ("e2", "foo bar"), ("e3", "baz qux")]
        result = await r.rerank("hello", docs, top_n=2)
        assert [eid for eid, _ in result] == ["e1", "e2"]

    def test_noop_explicit(self):
        r = create_reranker(provider="noop")
        assert isinstance(r, NoopReranker)


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order():
    r = NoopReranker()
    docs = [("e1", "hello world"), ("e2", "foo bar"), ("e3", "baz qux")]
    result = await r.rerank("hello", docs, top_n=2)
    assert len(result) == 2
    assert result[0][0] == "e1"
    assert result[1][0] == "e2"


@pytest.mark.asyncio
async def test_fastembed_reranker_mock():
    """Test FastEmbedReranker with a mocked TextCrossEncoder."""
    from unittest.mock import MagicMock, patch

    mock_encoder = MagicMock()
    mock_encoder.rerank.return_value = [0.9, 0.1, 0.5]

    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", return_value=mock_encoder):
        from engram.retrieval.reranker import FastEmbedReranker

        reranker = FastEmbedReranker(model="test-model")

        docs = [("e1", "hello"), ("e2", "world"), ("e3", "foo")]
        result = await reranker.rerank("query", docs, top_n=2)

        assert len(result) == 2
        # Sorted by score descending: e1 (0.9), e3 (0.5)
        assert result[0][0] == "e1"
        assert result[0][1] == 0.9
        assert result[1][0] == "e3"
        assert result[1][1] == 0.5


@pytest.mark.asyncio
async def test_fastembed_reranker_empty_docs():
    """Test FastEmbedReranker with empty documents."""
    from unittest.mock import MagicMock, patch

    mock_encoder = MagicMock()
    with patch("fastembed.rerank.cross_encoder.TextCrossEncoder", return_value=mock_encoder):
        from engram.retrieval.reranker import FastEmbedReranker

        reranker = FastEmbedReranker(model="test-model")
        result = await reranker.rerank("query", [])
        assert result == []
