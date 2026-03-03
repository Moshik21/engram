"""Tests for SQLite vector storage."""

from __future__ import annotations

import pytest
import pytest_asyncio

from engram.storage.sqlite.vectors import (
    SQLiteVectorStore,
    cosine_similarity,
    pack_vector,
    unpack_vector,
)


class TestPackUnpack:
    def test_roundtrip(self):
        """pack_vector/unpack_vector roundtrip preserves values."""
        original = [1.0, 2.5, -3.7, 0.0, 0.001]
        blob = pack_vector(original)
        restored = unpack_vector(blob, len(original))
        for a, b in zip(original, restored):
            assert abs(a - b) < 1e-6

    def test_blob_size(self):
        """Each float takes 4 bytes in the BLOB."""
        vec = [0.0] * 512
        blob = pack_vector(vec)
        assert len(blob) == 512 * 4


class TestCosineSimilarity:
    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_zero_vector(self):
        """Zero vector returns 0.0 similarity."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_mismatched_dimensions(self):
        """Mismatched dimensions returns 0.0."""
        a = [1.0, 2.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0

    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0

    def test_opposite_vectors(self):
        """Opposite vectors have similarity -1.0."""
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


@pytest_asyncio.fixture
async def vector_store(tmp_path):
    store = SQLiteVectorStore(str(tmp_path / "vec_test.db"))
    await store.initialize()
    yield store


class TestSQLiteVectorStore:
    @pytest.mark.asyncio
    async def test_upsert_and_search(self, vector_store: SQLiteVectorStore):
        """Upsert vectors then search returns correct top-K order."""
        # Insert 3 vectors: v1 is closest to query
        query = [1.0, 0.0, 0.0]
        await vector_store.upsert("e1", "entity", "grp", "close", [0.9, 0.1, 0.0])
        await vector_store.upsert("e2", "entity", "grp", "far", [0.0, 1.0, 0.0])
        await vector_store.upsert("e3", "entity", "grp", "mid", [0.5, 0.5, 0.0])

        results = await vector_store.search(query, "grp", limit=3)
        assert len(results) == 3
        ids = [r[0] for r in results]
        assert ids[0] == "e1"  # closest
        assert ids[1] == "e3"  # second closest

    @pytest.mark.asyncio
    async def test_group_isolation(self, vector_store: SQLiteVectorStore):
        """Search in group A does not return group B results."""
        vec = [1.0, 0.0]
        await vector_store.upsert("a1", "entity", "group_a", "a", vec)
        await vector_store.upsert("b1", "entity", "group_b", "b", vec)

        results_a = await vector_store.search(vec, "group_a")
        assert len(results_a) == 1
        assert results_a[0][0] == "a1"

        results_b = await vector_store.search(vec, "group_b")
        assert len(results_b) == 1
        assert results_b[0][0] == "b1"

    @pytest.mark.asyncio
    async def test_remove(self, vector_store: SQLiteVectorStore):
        """Remove deletes the vector."""
        await vector_store.upsert("e1", "entity", "grp", "text", [1.0, 0.0])
        await vector_store.remove("e1")
        results = await vector_store.search([1.0, 0.0], "grp")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_has_embeddings(self, vector_store: SQLiteVectorStore):
        """has_embeddings returns correct boolean."""
        assert not await vector_store.has_embeddings("grp")
        await vector_store.upsert("e1", "entity", "grp", "text", [1.0])
        assert await vector_store.has_embeddings("grp")

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, vector_store: SQLiteVectorStore):
        """Upserting same ID updates the embedding."""
        await vector_store.upsert("e1", "entity", "grp", "old", [1.0, 0.0])
        await vector_store.upsert("e1", "entity", "grp", "new", [0.0, 1.0])

        results = await vector_store.search([0.0, 1.0], "grp")
        assert len(results) == 1
        assert results[0][0] == "e1"
        assert results[0][1] > 0.9  # should be very similar now
