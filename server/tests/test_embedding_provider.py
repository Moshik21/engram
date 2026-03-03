"""Tests for embedding provider abstraction."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.embeddings.provider import NoopProvider, VoyageProvider


class TestNoopProvider:
    @pytest.mark.asyncio
    async def test_embed_returns_empty_list(self):
        """NoopProvider.embed() always returns empty list."""
        provider = NoopProvider()
        result = await provider.embed(["hello", "world"])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_query_returns_empty_list(self):
        """NoopProvider.embed_query() always returns empty list."""
        provider = NoopProvider()
        result = await provider.embed_query("hello")
        assert result == []

    def test_dimension_is_zero(self):
        """NoopProvider signals disabled embeddings with dimension=0."""
        provider = NoopProvider()
        assert provider.dimension() == 0


class TestVoyageProvider:
    @pytest.mark.asyncio
    async def test_embed_calls_api_with_document_type(self):
        """VoyageProvider.embed() uses input_type='document'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]
        }

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            provider = VoyageProvider(api_key="test-key", dimensions=3)
            result = await provider.embed(["text1", "text2"])

            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3]
            assert result[1] == [0.4, 0.5, 0.6]

            # Verify API call
            call_kwargs = mock_post.call_args
            body = call_kwargs.kwargs["json"]
            assert body["input_type"] == "document"
            assert body["input"] == ["text1", "text2"]

    @pytest.mark.asyncio
    async def test_embed_query_uses_query_type(self):
        """VoyageProvider.embed_query() uses input_type='query'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [0.1, 0.2]}]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            provider = VoyageProvider(api_key="test-key", dimensions=2)
            result = await provider.embed_query("search query")

            assert result == [0.1, 0.2]
            body = mock_post.call_args.kwargs["json"]
            assert body["input_type"] == "query"

    @pytest.mark.asyncio
    async def test_batching_splits_large_input(self):
        """VoyageProvider splits inputs exceeding batch_size."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": [{"index": 0, "embedding": [1.0]}]}

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            provider = VoyageProvider(
                api_key="test-key",
                dimensions=1,
                batch_size=2,
            )
            texts = ["a", "b", "c", "d", "e"]
            await provider.embed(texts)

            # 5 texts with batch_size=2 → 3 API calls
            assert mock_post.call_count == 3

    def test_dimension_returns_configured_value(self):
        """VoyageProvider.dimension() returns configured dimensions."""
        provider = VoyageProvider(api_key="test", dimensions=512)
        assert provider.dimension() == 512

    @pytest.mark.asyncio
    async def test_embed_empty_list(self):
        """VoyageProvider.embed([]) returns empty without API call."""
        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            provider = VoyageProvider(api_key="test", dimensions=3)
            result = await provider.embed([])
            assert result == []
            mock_post.assert_not_called()
