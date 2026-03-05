"""Tests for embedding versioning columns and mismatch detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


def _mock_provider(dim: int):
    """Create a mock EmbeddingProvider with sync dimension() and async embed()."""
    provider = MagicMock()
    provider.dimension.return_value = dim
    provider.embed = AsyncMock(return_value=[])
    provider.embed_query = AsyncMock(return_value=[])
    return provider


@pytest.mark.asyncio
async def test_versioning_columns_exist(tmp_path):
    """Test that embed_provider and embed_model columns are created."""
    import aiosqlite

    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteVectorStore(db_path)
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await store.initialize(db=db)

    await store.upsert(
        "e1", "entity", "default", "test",
        [1.0, 2.0, 3.0],
        embed_provider="local",
        embed_model="nomic-ai/nomic-embed-text-v1.5",
    )

    cursor = await db.execute(
        "SELECT embed_provider, embed_model FROM embeddings WHERE id = 'e1'"
    )
    row = await cursor.fetchone()
    assert row["embed_provider"] == "local"
    assert row["embed_model"] == "nomic-ai/nomic-embed-text-v1.5"

    await db.close()


@pytest.mark.asyncio
async def test_versioning_migration_on_existing_db(tmp_path):
    """Test that ALTER TABLE migration works on an existing database."""
    import aiosqlite

    db_path = str(tmp_path / "test.db")

    # Create old-style table without versioning columns
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            content_type TEXT NOT NULL DEFAULT 'entity',
            group_id TEXT NOT NULL,
            text_content TEXT,
            embedding BLOB NOT NULL,
            dimensions INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    await db.commit()

    # Now initialize the store — should add columns via ALTER TABLE
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    store = SQLiteVectorStore(db_path)
    await store.initialize(db=db)

    # Insert with versioning metadata
    await store.upsert(
        "e1", "entity", "default", "test",
        [1.0, 2.0],
        embed_provider="voyage",
        embed_model="voyage-4-lite",
    )

    cursor = await db.execute(
        "SELECT embed_provider, embed_model FROM embeddings WHERE id = 'e1'"
    )
    row = await cursor.fetchone()
    assert row["embed_provider"] == "voyage"
    assert row["embed_model"] == "voyage-4-lite"

    await db.close()


@pytest.mark.asyncio
async def test_batch_upsert_with_versioning(tmp_path):
    """Test batch_upsert stores versioning metadata."""
    import aiosqlite

    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteVectorStore(db_path)
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await store.initialize(db=db)

    items = [
        ("e1", "entity", "default", "alpha", [1.0, 2.0]),
        ("e2", "entity", "default", "beta", [3.0, 4.0]),
    ]
    await store.batch_upsert(items, embed_provider="local", embed_model="test-model")

    cursor = await db.execute(
        "SELECT embed_provider, embed_model FROM embeddings ORDER BY id"
    )
    rows = await cursor.fetchall()
    assert len(rows) == 2
    for row in rows:
        assert row["embed_provider"] == "local"
        assert row["embed_model"] == "test-model"

    await db.close()


@pytest.mark.asyncio
async def test_version_check_logs_mismatch(tmp_path, caplog):
    """Test that _check_embedding_version logs warnings for dimension mismatch."""
    import logging

    import aiosqlite

    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(dim=2)

    idx = HybridSearchIndex(
        fts=fts, vector_store=vectors, provider=provider,
        embed_provider="local", embed_model="test",
    )

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    # Store a vector with different dimension (4d vs current 2d)
    await vectors.upsert("e1", "entity", "default", "test", [1.0, 2.0, 3.0, 4.0])

    with caplog.at_level(logging.WARNING):
        await idx._check_embedding_version()

    assert any("dimension mismatch" in r.message for r in caplog.records)

    await db.close()


@pytest.mark.asyncio
async def test_version_check_logs_provider_mismatch(tmp_path, caplog):
    """Test that _check_embedding_version logs warnings for provider mismatch."""
    import logging

    import aiosqlite

    from engram.storage.sqlite.hybrid_search import HybridSearchIndex
    from engram.storage.sqlite.search import FTS5SearchIndex
    from engram.storage.sqlite.vectors import SQLiteVectorStore

    db_path = str(tmp_path / "test.db")
    fts = FTS5SearchIndex(db_path)
    vectors = SQLiteVectorStore(db_path)

    provider = _mock_provider(dim=2)

    idx = HybridSearchIndex(
        fts=fts, vector_store=vectors, provider=provider,
        embed_provider="local", embed_model="test",
    )

    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await fts.initialize(db=db)
    await vectors.initialize(db=db)

    # Store a vector with different provider
    await vectors.upsert(
        "e1", "entity", "default", "test", [1.0, 2.0],
        embed_provider="voyage", embed_model="voyage-4-lite",
    )

    with caplog.at_level(logging.WARNING):
        await idx._check_embedding_version()

    assert any("provider mismatch" in r.message for r in caplog.records)

    await db.close()
