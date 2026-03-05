"""Tests for Matryoshka truncation, mixed-dim search, and batch embedding."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.embeddings.provider import truncate_vectors


def _mock_provider(dim: int, embed_return=None, query_return=None):
    """Create a mock EmbeddingProvider with sync dimension() and async embed()."""
    provider = MagicMock()
    provider.dimension.return_value = dim
    provider.embed = AsyncMock(return_value=embed_return or [])
    provider.embed_query = AsyncMock(return_value=query_return or [])
    return provider


class TestTruncateVectors:
    """Test the truncate_vectors utility."""

    def test_truncate_normal(self):
        vecs = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        result = truncate_vectors(vecs, 2)
        assert result == [[1.0, 2.0], [5.0, 6.0]]

    def test_no_truncation_when_target_ge_dim(self):
        vecs = [[1.0, 2.0, 3.0]]
        assert truncate_vectors(vecs, 3) is vecs  # same object
        assert truncate_vectors(vecs, 5) is vecs

    def test_no_truncation_when_target_zero(self):
        vecs = [[1.0, 2.0]]
        assert truncate_vectors(vecs, 0) is vecs

    def test_no_truncation_when_negative(self):
        vecs = [[1.0, 2.0]]
        assert truncate_vectors(vecs, -1) is vecs

    def test_empty_list(self):
        assert truncate_vectors([], 2) == []

    def test_truncate_to_one(self):
        vecs = [[1.0, 2.0, 3.0]]
        assert truncate_vectors(vecs, 1) == [[1.0]]


@pytest.mark.asyncio
async def test_hybrid_search_with_storage_dim(tmp_path):
    """Test that HybridSearchIndex applies Matryoshka truncation."""
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(
        dim=4,
        embed_return=[[1.0, 2.0, 3.0, 4.0]],
        query_return=[1.0, 2.0, 3.0, 4.0],
    )

    idx = HybridSearchIndex(
        fts=fts,
        vector_store=vectors,
        provider=provider,
        storage_dim=2,
        embed_provider="local",
        embed_model="test-model",
    )

    import aiosqlite

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    from engram.models.entity import Entity

    entity = Entity(id="e1", name="Test Entity", group_id="default", entity_type="Thing")
    await idx.index_entity(entity)

    # Verify stored dimension is 2
    cursor = await db.execute("SELECT dimensions FROM embeddings WHERE id = 'e1'")
    row = await cursor.fetchone()
    assert row["dimensions"] == 2

    # Verify versioning metadata
    cursor = await db.execute(
        "SELECT embed_provider, embed_model FROM embeddings WHERE id = 'e1'"
    )
    row = await cursor.fetchone()
    assert row["embed_provider"] == "local"
    assert row["embed_model"] == "test-model"

    await db.close()


@pytest.mark.asyncio
async def test_batch_index_entities(tmp_path):
    """Test batch_index_entities method."""
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(
        dim=3,
        embed_return=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    )

    idx = HybridSearchIndex(
        fts=fts, vector_store=vectors, provider=provider,
        embed_provider="local", embed_model="test",
    )

    import aiosqlite

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    from engram.models.entity import Entity

    entities = [
        Entity(id="e1", name="Alpha", group_id="default", entity_type="Thing"),
        Entity(
            id="e2", name="Beta", summary="A beta entity",
            group_id="default", entity_type="Thing",
        ),
    ]

    count = await idx.batch_index_entities(entities)
    assert count == 2

    # Verify both stored
    cursor = await db.execute("SELECT COUNT(*) as cnt FROM embeddings")
    row = await cursor.fetchone()
    assert row["cnt"] == 2

    # Provider.embed was called once with both texts
    provider.embed.assert_called_once()
    call_args = provider.embed.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0] == "Alpha"
    assert "Beta" in call_args[1]

    await db.close()


@pytest.mark.asyncio
async def test_batch_index_with_truncation(tmp_path):
    """Test batch_index_entities applies Matryoshka truncation."""
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(dim=4, embed_return=[[1.0, 2.0, 3.0, 4.0]])

    idx = HybridSearchIndex(
        fts=fts, vector_store=vectors, provider=provider,
        storage_dim=2,
    )

    import aiosqlite

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    from engram.models.entity import Entity

    entities = [Entity(id="e1", name="Gamma", group_id="default", entity_type="Thing")]
    count = await idx.batch_index_entities(entities)
    assert count == 1

    cursor = await db.execute("SELECT dimensions FROM embeddings WHERE id = 'e1'")
    row = await cursor.fetchone()
    assert row["dimensions"] == 2

    await db.close()


@pytest.mark.asyncio
async def test_mixed_dim_search(tmp_path):
    """Test search handles mixed old full-dim and new truncated vectors."""
    import aiosqlite

    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteVectorStore(db_path)
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await store.initialize(db=db)

    # Store a full 4d vector (old format)
    await store.upsert("old1", "entity", "default", "old entity", [1.0, 0.0, 0.0, 0.0])
    # Store a truncated 2d vector (new format)
    await store.upsert("new1", "entity", "default", "new entity", [1.0, 0.0])

    # Search with storage_dim=2 and a 2d query
    results = await store.search([1.0, 0.0], "default", storage_dim=2)

    # Both should be found — old vector gets truncated to 2d on the fly
    ids = [r[0] for r in results]
    assert "old1" in ids
    assert "new1" in ids

    await db.close()


@pytest.mark.asyncio
async def test_batch_index_empty_entities(tmp_path):
    """Test batch_index_entities with empty list returns 0."""
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(dim=3)

    idx = HybridSearchIndex(fts=fts, vector_store=vectors, provider=provider)

    import aiosqlite

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    count = await idx.batch_index_entities([])
    assert count == 0
    provider.embed.assert_not_called()

    await db.close()
