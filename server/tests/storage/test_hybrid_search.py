"""Tests for the hybrid FTS5 + vector search index."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from engram.embeddings.provider import NoopProvider
from engram.models.entity import Entity
from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore


@pytest_asyncio.fixture
async def fts(tmp_path):
    from engram.storage.sqlite.graph import SQLiteGraphStore

    graph = SQLiteGraphStore(str(tmp_path / "hybrid.db"))
    await graph.initialize()
    index = FTS5SearchIndex(graph._db_path)
    await index.initialize(db=graph._db)

    # Insert test entities (FTS5 is trigger-based)
    for ent in [
        Entity(id="e1", name="Python", entity_type="Technology",
               summary="Programming language", group_id="default"),
        Entity(id="e2", name="JavaScript", entity_type="Technology",
               summary="Web language", group_id="default"),
        Entity(id="e3", name="FastAPI", entity_type="Technology",
               summary="Python web framework", group_id="default"),
    ]:
        await graph.create_entity(ent)

    yield index, graph


class TestHybridWithNoop:
    @pytest.mark.asyncio
    async def test_noop_falls_back_to_fts5(self, fts):
        """With NoopProvider, hybrid search degrades to FTS5-only."""
        fts_index, graph = fts
        vectors = SQLiteVectorStore(graph._db_path)
        await vectors.initialize(db=graph._db)

        hybrid = HybridSearchIndex(
            fts=fts_index,
            vector_store=vectors,
            provider=NoopProvider(),
        )

        results = await hybrid.search("Python", group_id="default", limit=5)
        assert len(results) > 0
        # First result should be Python (exact match in FTS5)
        assert results[0][0] == "e1"


class TestHybridWithMockProvider:
    @pytest.mark.asyncio
    async def test_combines_fts_and_vector_scores(self, fts):
        """Hybrid merges FTS5 and vector scores with configured weights."""
        fts_index, graph = fts
        vectors = SQLiteVectorStore(graph._db_path)
        await vectors.initialize(db=graph._db)

        # Create a mock provider
        mock_provider = MagicMock()
        mock_provider.dimension.return_value = 3

        # Mock embed_query to return a vector
        mock_provider.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])
        # Mock embed to return vectors
        mock_provider.embed = AsyncMock(return_value=[[0.9, 0.1, 0.0]])

        hybrid = HybridSearchIndex(
            fts=fts_index,
            vector_store=vectors,
            provider=mock_provider,
            fts_weight=0.3,
            vec_weight=0.7,
        )

        # Manually add a vector for e1
        await vectors.upsert("e1", "entity", "default", "Python", [0.9, 0.1, 0.0])
        await vectors.upsert("e2", "entity", "default", "JavaScript", [0.1, 0.9, 0.0])

        results = await hybrid.search("Python", group_id="default", limit=5)
        assert len(results) > 0
        # Scores should be normalized to [0, 1]
        for _, score in results:
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_scores_normalized(self, fts):
        """All scores are in [0.0, 1.0] range."""
        fts_index, graph = fts
        vectors = SQLiteVectorStore(graph._db_path)
        await vectors.initialize(db=graph._db)

        mock_provider = MagicMock()
        mock_provider.dimension.return_value = 3
        mock_provider.embed_query = AsyncMock(return_value=[1.0, 0.0, 0.0])

        hybrid = HybridSearchIndex(
            fts=fts_index,
            vector_store=vectors,
            provider=mock_provider,
        )

        # Even with no vectors, FTS results should be normalized
        results = await hybrid.search("language", group_id="default", limit=10)
        for _, score in results:
            assert 0.0 <= score <= 1.0
