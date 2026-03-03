"""Tests for numpy-accelerated vector operations."""

import pytest

from engram.storage.sqlite.vectors import cosine_similarity


class TestCosineSimNumpy:
    def test_identical_vectors(self):
        """Identical vectors have similarity 1.0."""
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors have similarity 0.0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_zero_vector(self):
        """Zero vector returns 0.0."""
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_empty_vectors(self):
        """Empty vectors return 0.0."""
        assert cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self):
        """Mismatched vector lengths return 0.0."""
        assert cosine_similarity([1.0, 2.0], [1.0]) == 0.0


class TestBatchSearch:
    @pytest.mark.asyncio
    async def test_batch_ordering(self):
        """Batch search returns results in descending similarity order."""
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        store = SQLiteVectorStore(":memory:")
        await store.initialize()

        # Insert 3 vectors
        await store.upsert("close", "entity", "g1", "close", [0.9, 0.1, 0.0])
        await store.upsert("medium", "entity", "g1", "medium", [0.5, 0.5, 0.0])
        await store.upsert("far", "entity", "g1", "far", [0.0, 0.0, 1.0])

        query = [1.0, 0.0, 0.0]
        results = await store.search(query, "g1", limit=10)

        assert len(results) == 3
        # "close" should be first (most similar to [1,0,0])
        assert results[0][0] == "close"
        # Scores should be descending
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    @pytest.mark.asyncio
    async def test_empty_search(self):
        """Search with no vectors returns empty list."""
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        store = SQLiteVectorStore(":memory:")
        await store.initialize()

        results = await store.search([1.0, 0.0], "g1")
        assert results == []

    @pytest.mark.asyncio
    async def test_zero_query_vector(self):
        """Search with zero query vector returns empty list."""
        from engram.storage.sqlite.vectors import SQLiteVectorStore

        store = SQLiteVectorStore(":memory:")
        await store.initialize()

        await store.upsert("v1", "entity", "g1", "text", [1.0, 0.0])
        results = await store.search([0.0, 0.0], "g1")
        assert results == []
