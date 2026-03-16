"""Tests for GraphEmbeddingStore CRUD operations."""

from __future__ import annotations

import aiosqlite
import pytest

from engram.embeddings.graph.storage import GraphEmbeddingStore


@pytest.fixture
async def db(tmp_path):
    """Create an in-memory SQLite database."""
    async with aiosqlite.connect(str(tmp_path / "test.db")) as conn:
        conn.row_factory = aiosqlite.Row
        store = GraphEmbeddingStore()
        await store.initialize(conn)
        yield conn


@pytest.fixture
def store():
    return GraphEmbeddingStore()


@pytest.mark.asyncio
async def test_upsert_and_get(db, store):
    """Upsert embeddings, then retrieve them."""
    embeddings = {
        "entity_1": [0.1, 0.2, 0.3],
        "entity_2": [0.4, 0.5, 0.6],
    }
    count = await store.upsert_batch(db, embeddings, "node2vec", "default")
    assert count == 2

    result = await store.get_embeddings(db, ["entity_1", "entity_2"], "node2vec", "default")
    assert len(result) == 2
    assert pytest.approx(result["entity_1"], abs=1e-5) == [0.1, 0.2, 0.3]
    assert pytest.approx(result["entity_2"], abs=1e-5) == [0.4, 0.5, 0.6]


@pytest.mark.asyncio
async def test_upsert_overwrites(db, store):
    """Upsert should overwrite existing embeddings."""
    await store.upsert_batch(db, {"e1": [1.0, 2.0]}, "node2vec", "default")
    await store.upsert_batch(db, {"e1": [3.0, 4.0]}, "node2vec", "default")

    result = await store.get_embeddings(db, ["e1"], "node2vec", "default")
    assert pytest.approx(result["e1"], abs=1e-5) == [3.0, 4.0]


@pytest.mark.asyncio
async def test_different_methods_coexist(db, store):
    """Same entity can have embeddings for different methods."""
    await store.upsert_batch(db, {"e1": [1.0, 2.0]}, "node2vec", "default")
    await store.upsert_batch(db, {"e1": [3.0, 4.0, 5.0]}, "transe", "default")

    n2v = await store.get_embeddings(db, ["e1"], "node2vec", "default")
    transe = await store.get_embeddings(db, ["e1"], "transe", "default")

    assert len(n2v["e1"]) == 2
    assert len(transe["e1"]) == 3


@pytest.mark.asyncio
async def test_get_all_embeddings(db, store):
    """get_all_embeddings returns all for a method+group."""
    await store.upsert_batch(
        db,
        {"e1": [1.0], "e2": [2.0], "e3": [3.0]},
        "node2vec",
        "default",
    )

    result = await store.get_all_embeddings(db, "node2vec", "default")
    assert len(result) == 3


@pytest.mark.asyncio
async def test_delete_by_method(db, store):
    """delete_by_method removes all embeddings for a method+group."""
    await store.upsert_batch(db, {"e1": [1.0], "e2": [2.0]}, "node2vec", "default")
    await store.upsert_batch(db, {"e1": [3.0]}, "transe", "default")

    deleted = await store.delete_by_method(db, "node2vec", "default")
    assert deleted == 2

    remaining = await store.get_all_embeddings(db, "node2vec", "default")
    assert len(remaining) == 0

    # TransE should be untouched
    transe = await store.get_all_embeddings(db, "transe", "default")
    assert len(transe) == 1


@pytest.mark.asyncio
async def test_group_isolation(db, store):
    """Different groups don't see each other's embeddings."""
    await store.upsert_batch(db, {"e1": [1.0]}, "node2vec", "group_a")
    await store.upsert_batch(db, {"e1": [2.0]}, "node2vec", "group_b")

    a = await store.get_embeddings(db, ["e1"], "node2vec", "group_a")
    b = await store.get_embeddings(db, ["e1"], "node2vec", "group_b")

    assert pytest.approx(a["e1"], abs=1e-5) == [1.0]
    assert pytest.approx(b["e1"], abs=1e-5) == [2.0]


@pytest.mark.asyncio
async def test_empty_upsert(db, store):
    """Upserting empty dict returns 0."""
    count = await store.upsert_batch(db, {}, "node2vec", "default")
    assert count == 0


@pytest.mark.asyncio
async def test_get_nonexistent(db, store):
    """Getting embeddings for non-existent IDs returns empty dict."""
    result = await store.get_embeddings(db, ["nope"], "node2vec", "default")
    assert result == {}


@pytest.mark.asyncio
async def test_get_empty_ids(db, store):
    """Getting embeddings with empty ID list returns empty dict."""
    result = await store.get_embeddings(db, [], "node2vec", "default")
    assert result == {}
