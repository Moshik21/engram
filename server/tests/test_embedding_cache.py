"""Tests for VoyageProvider query embedding cache."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from engram.embeddings.provider import VoyageProvider


@pytest.fixture
def provider():
    """Create a VoyageProvider with mocked httpx client."""
    with patch("httpx.AsyncClient"):
        p = VoyageProvider(api_key="test-key")
    return p


class TestQueryEmbeddingCache:
    @pytest.mark.asyncio
    async def test_cache_miss_calls_api(self, provider):
        """First query for a text calls the API."""
        mock_vec = [0.1, 0.2, 0.3]
        provider._call_api = AsyncMock(return_value=[mock_vec])

        result = await provider.embed_query("hello world")

        assert result == mock_vec
        provider._call_api.assert_called_once_with(["hello world"], input_type="query")

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, provider):
        """Second query for same text uses cache, no API call."""
        mock_vec = [0.1, 0.2, 0.3]
        provider._call_api = AsyncMock(return_value=[mock_vec])

        # First call - API
        await provider.embed_query("hello world")
        # Second call - cache
        result = await provider.embed_query("hello world")

        assert result == mock_vec
        assert provider._call_api.call_count == 1

    @pytest.mark.asyncio
    async def test_cache_eviction(self, provider):
        """Cache evicts oldest entry when exceeding max size."""
        provider.CACHE_MAX_SIZE = 3
        call_count = 0

        async def mock_api(texts, input_type):
            nonlocal call_count
            call_count += 1
            return [[float(call_count)] * 3]

        provider._call_api = mock_api

        # Fill cache
        await provider.embed_query("text1")
        await provider.embed_query("text2")
        await provider.embed_query("text3")
        assert len(provider._query_cache) == 3

        # This should evict "text1"
        await provider.embed_query("text4")
        assert len(provider._query_cache) == 3
        assert "text1" not in provider._query_cache
        assert "text4" in provider._query_cache

    @pytest.mark.asyncio
    async def test_clear_cache(self, provider):
        """clear_cache() empties the cache."""
        mock_vec = [0.1, 0.2, 0.3]
        provider._call_api = AsyncMock(return_value=[mock_vec])

        await provider.embed_query("hello")
        assert len(provider._query_cache) == 1

        provider.clear_cache()
        assert len(provider._query_cache) == 0

    @pytest.mark.asyncio
    async def test_empty_result_not_cached(self, provider):
        """Empty API results are not cached."""
        provider._call_api = AsyncMock(return_value=[[]])

        result = await provider.embed_query("empty")
        assert result == []
        # Empty result should not be cached
        assert "empty" not in provider._query_cache
