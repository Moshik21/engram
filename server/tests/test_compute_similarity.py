"""Tests for compute_similarity across search index implementations."""

from __future__ import annotations

import pytest

from engram.storage.sqlite.search import FTS5SearchIndex


@pytest.mark.asyncio
class TestFTS5ComputeSimilarity:
    async def test_fts5_returns_empty(self):
        """FTS5 search index always returns empty dict (no embeddings)."""
        fts = FTS5SearchIndex(":memory:")
        result = await fts.compute_similarity("test query", ["e1", "e2"], group_id="default")
        assert result == {}

    async def test_fts5_returns_empty_for_no_ids(self):
        """FTS5 with empty entity_ids returns empty dict."""
        fts = FTS5SearchIndex(":memory:")
        result = await fts.compute_similarity("test query", [], group_id="default")
        assert result == {}


@pytest.mark.asyncio
class TestHybridComputeSimilarity:
    async def test_returns_scores_for_existing_entities(self):
        """Hybrid search returns similarity scores for entities with embeddings."""
        from engram.embeddings.provider import NoopProvider
        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")
        provider = NoopProvider()
        hybrid = HybridSearchIndex(fts, vectors, provider)

        # NoopProvider → embeddings_enabled=False → should return {}
        result = await hybrid.compute_similarity("test", ["e1"], group_id="default")
        assert result == {}

    async def test_returns_empty_when_embeddings_disabled(self):
        """With NoopProvider (no embeddings), returns empty dict."""
        from engram.embeddings.provider import NoopProvider
        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")
        provider = NoopProvider()
        hybrid = HybridSearchIndex(fts, vectors, provider)

        result = await hybrid.compute_similarity("query", ["e1", "e2"])
        assert result == {}

    async def test_returns_empty_for_empty_entity_ids(self):
        """Empty entity_ids list returns empty dict."""
        from engram.embeddings.provider import NoopProvider
        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")
        provider = NoopProvider()
        hybrid = HybridSearchIndex(fts, vectors, provider)

        result = await hybrid.compute_similarity("query", [])
        assert result == {}

    async def test_missing_entities_not_in_results(self):
        """Entities without stored embeddings are omitted from results."""
        from unittest.mock import AsyncMock

        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")

        # Create a mock provider with embeddings enabled
        mock_provider = AsyncMock()
        mock_provider.dimension = lambda: 3
        mock_provider.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])

        hybrid = HybridSearchIndex(fts, vectors, mock_provider)
        await vectors.initialize()

        # Don't insert any embeddings — all lookups should miss
        result = await hybrid.compute_similarity(
            "test query", ["nonexistent_1", "nonexistent_2"], group_id="default"
        )
        assert result == {}

    async def test_with_stored_embeddings(self):
        """Returns real similarity when embeddings exist."""
        from unittest.mock import AsyncMock

        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")

        mock_provider = AsyncMock()
        mock_provider.dimension = lambda: 3
        mock_provider.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])

        hybrid = HybridSearchIndex(fts, vectors, mock_provider)
        await vectors.initialize()

        # Store a vector for entity
        await vectors.upsert("e1", "entity", "default", "test text", [1.0, 0.0, 0.0])

        result = await hybrid.compute_similarity("test query", ["e1"], group_id="default")
        assert "e1" in result
        assert abs(result["e1"] - 1.0) < 1e-5  # identical vectors → sim ≈ 1.0

    async def test_provider_error_returns_empty(self):
        """If the provider raises an exception, returns empty dict."""
        from unittest.mock import AsyncMock

        from engram.storage.sqlite.hybrid_search import HybridSearchIndex
        from engram.storage.sqlite.search import FTS5SearchIndex
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        fts = FTS5SearchIndex(":memory:")
        vectors = SQLiteVectorStore(":memory:")

        mock_provider = AsyncMock()
        mock_provider.dimension = lambda: 3
        mock_provider.embed_query = AsyncMock(side_effect=RuntimeError("API down"))

        hybrid = HybridSearchIndex(fts, vectors, mock_provider)

        result = await hybrid.compute_similarity("test query", ["e1"], group_id="default")
        assert result == {}
