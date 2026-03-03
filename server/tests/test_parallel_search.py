"""Tests for parallel search and embedding passthrough in HybridSearchIndex."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.storage.sqlite.hybrid_search import HybridSearchIndex


@pytest.fixture
def mock_fts():
    fts = AsyncMock()
    fts.search = AsyncMock(return_value=[("e1", 5.0), ("e2", 3.0)])
    return fts


@pytest.fixture
def mock_vectors():
    vectors = AsyncMock()
    vectors.has_embeddings = AsyncMock(return_value=True)
    vectors.search = AsyncMock(return_value=[("e1", 0.9), ("e3", 0.7)])

    mock_cursor = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=None)
    vectors.db = AsyncMock()
    vectors.db.execute = AsyncMock(return_value=mock_cursor)
    return vectors


@pytest.fixture
def mock_provider():
    provider = MagicMock()
    provider.dimension.return_value = 512
    provider.embed_query = AsyncMock(return_value=[0.1] * 512)
    provider.embed = AsyncMock(return_value=[[0.1] * 512])
    return provider


@pytest.fixture
def hybrid_index(mock_fts, mock_vectors, mock_provider):
    return HybridSearchIndex(
        fts=mock_fts,
        vector_store=mock_vectors,
        provider=mock_provider,
    )


class TestParallelSearch:
    @pytest.mark.asyncio
    async def test_parallel_search_produces_results(
        self, hybrid_index, mock_fts, mock_vectors,
    ):
        """Parallel search returns merged results from both sources."""
        results = await hybrid_index.search("test query", group_id="default")
        assert len(results) > 0
        mock_fts.search.assert_called_once()
        mock_vectors.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_search_caches_query_vec(
        self, hybrid_index, mock_provider,
    ):
        """search() caches the query embedding for later reuse."""
        assert hybrid_index._last_query_vec is None
        await hybrid_index.search("test query", group_id="default")
        assert hybrid_index._last_query_vec == [0.1] * 512

    @pytest.mark.asyncio
    async def test_fts_only_fallback_no_embeddings(self, mock_fts, mock_vectors):
        """Falls back to FTS5-only when embeddings are disabled."""
        provider = MagicMock()
        provider.dimension.return_value = 0
        idx = HybridSearchIndex(
            fts=mock_fts, vector_store=mock_vectors, provider=provider,
        )
        results = await idx.search("test query", group_id="default")
        assert results == [("e1", 5.0), ("e2", 3.0)]
        provider.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_embed_failure(
        self, hybrid_index, mock_provider, mock_fts,
    ):
        """Falls back to FTS5 when embedding fails."""
        mock_provider.embed_query = AsyncMock(
            side_effect=RuntimeError("API error"),
        )
        results = await hybrid_index.search("test query", group_id="default")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_no_embeddings_group(
        self, hybrid_index, mock_vectors, mock_fts,
    ):
        """Falls back to FTS5 when group has no embeddings."""
        mock_vectors.has_embeddings = AsyncMock(return_value=False)
        results = await hybrid_index.search("test query", group_id="mygroup")
        assert results == [("e1", 5.0), ("e2", 3.0)]

    @pytest.mark.asyncio
    async def test_empty_query_vec_fallback(self, hybrid_index, mock_provider):
        """Falls back to FTS5 when embed_query returns empty."""
        mock_provider.embed_query = AsyncMock(return_value=[])
        results = await hybrid_index.search("test query", group_id="default")
        assert len(results) > 0


class TestEmbeddingPassthrough:
    @pytest.mark.asyncio
    async def test_passthrough_skips_embed_call(
        self, hybrid_index, mock_provider,
    ):
        """When query_embedding is provided, embed_query is not called."""
        precomputed = [0.5] * 512
        await hybrid_index.compute_similarity(
            "test", ["e1"], group_id="default", query_embedding=precomputed,
        )
        mock_provider.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_uses_cached_vec_from_search(
        self, hybrid_index, mock_provider,
    ):
        """compute_similarity uses _last_query_vec from prior search()."""
        await hybrid_index.search("test query", group_id="default")
        mock_provider.embed_query.reset_mock()
        await hybrid_index.compute_similarity(
            "test query", ["e1"], group_id="default",
        )
        mock_provider.embed_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_to_embed_query(
        self, hybrid_index, mock_provider,
    ):
        """When no cached vec and no passthrough, calls embed_query."""
        hybrid_index._last_query_vec = None
        await hybrid_index.compute_similarity(
            "new query", ["e1"], group_id="default",
        )
        mock_provider.embed_query.assert_called_once_with("new query")

    @pytest.mark.asyncio
    async def test_disabled_embeddings_returns_empty(self):
        """Returns empty when embeddings disabled."""
        provider = MagicMock()
        provider.dimension.return_value = 0
        idx = HybridSearchIndex(
            fts=AsyncMock(), vector_store=AsyncMock(), provider=provider,
        )
        result = await idx.compute_similarity("q", ["e1"], group_id="default")
        assert result == {}
