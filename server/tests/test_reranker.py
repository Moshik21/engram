"""Tests for cross-encoder re-ranker providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.retrieval.reranker import (
    CohereReranker,
    NoopReranker,
    create_reranker,
)


class TestNoopReranker:
    @pytest.mark.asyncio
    async def test_preserves_order(self):
        """NoopReranker preserves the input ordering."""
        reranker = NoopReranker()
        docs = [("e1", "text 1"), ("e2", "text 2"), ("e3", "text 3")]
        results = await reranker.rerank("query", docs, top_n=10)
        assert [eid for eid, _ in results] == ["e1", "e2", "e3"]

    @pytest.mark.asyncio
    async def test_scores_descending(self):
        """NoopReranker scores decrease with rank."""
        reranker = NoopReranker()
        docs = [("e1", "text 1"), ("e2", "text 2"), ("e3", "text 3")]
        results = await reranker.rerank("query", docs)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """NoopReranker handles empty input."""
        reranker = NoopReranker()
        results = await reranker.rerank("query", [])
        assert results == []

    @pytest.mark.asyncio
    async def test_single_item(self):
        """NoopReranker handles single item."""
        reranker = NoopReranker()
        results = await reranker.rerank("query", [("e1", "text")])
        assert len(results) == 1
        assert results[0][0] == "e1"
        assert results[0][1] == 1.0

    @pytest.mark.asyncio
    async def test_top_n_limits_results(self):
        """NoopReranker respects top_n limit."""
        reranker = NoopReranker()
        docs = [("e1", "t1"), ("e2", "t2"), ("e3", "t3"), ("e4", "t4")]
        results = await reranker.rerank("query", docs, top_n=2)
        assert len(results) == 2


class TestCohereReranker:
    @pytest.mark.asyncio
    async def test_mocked_api_call(self):
        """CohereReranker parses API response correctly."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "results": [
                {"index": 1, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.80},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            reranker = CohereReranker(api_key="test-key")
            reranker._client = mock_client

            docs = [("e1", "text 1"), ("e2", "text 2")]
            results = await reranker.rerank("query", docs, top_n=2)

            assert len(results) == 2
            assert results[0][0] == "e2"
            assert results[0][1] == 0.95
            assert results[1][0] == "e1"
            assert results[1][1] == 0.80

    @pytest.mark.asyncio
    async def test_api_error_returns_original_order(self):
        """CohereReranker falls back to original order on API error."""
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=RuntimeError("API down"),
            )
            mock_client_cls.return_value = mock_client

            reranker = CohereReranker(api_key="test-key")
            reranker._client = mock_client

            docs = [("e1", "t1"), ("e2", "t2")]
            results = await reranker.rerank("query", docs, top_n=2)

            assert len(results) == 2
            assert results[0][0] == "e1"


class TestRerankerFactory:
    def test_creates_noop_without_api_key(self):
        """Factory creates NoopReranker when no API key."""
        reranker = create_reranker()
        assert isinstance(reranker, NoopReranker)

    def test_creates_noop_with_empty_key(self):
        """Factory creates NoopReranker with empty string key."""
        reranker = create_reranker(api_key="")
        assert isinstance(reranker, NoopReranker)

    def test_creates_cohere_with_api_key(self):
        """Factory creates CohereReranker when API key provided."""
        reranker = create_reranker(api_key="test-key-123")
        assert isinstance(reranker, CohereReranker)
