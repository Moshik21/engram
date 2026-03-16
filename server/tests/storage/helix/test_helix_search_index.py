"""Integration tests for HelixDB SearchIndex (vector + BM25 hybrid search).

Verifies that HelixSearchIndex correctly indexes entities and episodes,
performs vector search, BM25 search, hybrid (RRF fusion) search,
similarity computation, and deletion.

All tests require a running HelixDB instance on localhost:6969 and are
marked with ``requires_helix``.  The conftest ``helix_search_index``
fixture uses a NoopProvider (dimension=0), which disables vector search
and exercises the BM25-only code paths.  Tests that need vector search
use a ``FakeEmbeddingProvider`` that returns deterministic mock vectors.
"""

from __future__ import annotations

import hashlib
import socket
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.storage.helix.search import HelixSearchIndex, _cosine_similarity, _rrf_fusion


def helix_available() -> bool:
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


_helix_skip = pytest.mark.skipif(not helix_available(), reason="HelixDB not available")


def _uid() -> str:
    return uuid4().hex[:12]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Fake embedding provider that returns deterministic vectors
# ---------------------------------------------------------------------------


class FakeEmbeddingProvider:
    """Returns deterministic 64-dimensional vectors for testing."""

    _dim: int

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    async def embed_query(self, text: str) -> list[float]:
        return self._hash_vec(text)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vec(t) for t in texts]

    def _hash_vec(self, text: str) -> list[float]:
        """Return a reproducible vector derived from the text hash."""
        h = hashlib.sha256(text.encode()).digest()
        vec = []
        for i in range(self._dim):
            byte = h[i % len(h)]
            vec.append(byte / 255.0)
        return vec


@pytest_asyncio.fixture
async def helix_search_with_embeddings(test_group_id):
    """HelixSearchIndex with FakeEmbeddingProvider for vector search tests."""
    helix_config = HelixDBConfig(host="localhost", port=6969, verbose=False)
    provider = FakeEmbeddingProvider(dim=64)
    embed_config = EmbeddingConfig()
    index = HelixSearchIndex(
        helix_config=helix_config,
        provider=provider,
        embed_config=embed_config,
        storage_dim=64,
        embed_provider="fake",
        embed_model="fake-64",
    )
    try:
        await index.initialize()
    except Exception:
        pytest.skip("HelixDB not available or helix package not installed")

    yield index

    try:
        await index.delete_group(test_group_id)
    except Exception:
        pass
    await index.close()


# ======================================================================
# Indexing (NoopProvider -- BM25 path only)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestIndexingNoop:
    """With NoopProvider (dim=0), vector indexing is a no-op; BM25 still works."""

    async def test_index_entity_noop(self, helix_search_index, test_group_id):
        entity = Entity(
            id=f"ent_{_uid()}",
            name="Python",
            entity_type="Technology",
            summary="A versatile programming language",
            group_id=test_group_id,
        )
        # Should not raise even with embeddings disabled
        await helix_search_index.index_entity(entity)

    async def test_index_episode_noop(self, helix_search_index, test_group_id):
        episode = Episode(
            id=f"ep_{_uid()}",
            content="User discussed Python async patterns",
            group_id=test_group_id,
        )
        await helix_search_index.index_episode(episode)

    async def test_batch_index_entities_noop(self, helix_search_index, test_group_id):
        entities = [
            Entity(
                id=f"ent_{_uid()}",
                name=f"Noop{i}",
                entity_type="Test",
                group_id=test_group_id,
            )
            for i in range(3)
        ]
        count = await helix_search_index.batch_index_entities(entities)
        assert count == 0  # NoopProvider returns empty embeddings


# ======================================================================
# Search with NoopProvider (BM25-only fallback)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestSearchNoop:
    async def test_search_entities_returns_list(self, helix_search_index, test_group_id):
        results = await helix_search_index.search("Python", group_id=test_group_id)
        assert isinstance(results, list)

    async def test_search_episodes_returns_list(self, helix_search_index, test_group_id):
        results = await helix_search_index.search_episodes(
            "Python", group_id=test_group_id
        )
        assert isinstance(results, list)

    async def test_compute_similarity_empty_without_embeddings(
        self, helix_search_index, test_group_id
    ):
        result = await helix_search_index.compute_similarity(
            "Python", [f"ent_{_uid()}"], group_id=test_group_id
        )
        assert result == {}


# ======================================================================
# Indexing with FakeEmbeddingProvider (vector + BM25)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestIndexingWithEmbeddings:
    async def test_index_entity_with_embedding(
        self, helix_search_with_embeddings, test_group_id
    ):
        entity = Entity(
            id=f"ent_{_uid()}",
            name="Python",
            entity_type="Technology",
            summary="A popular programming language",
            group_id=test_group_id,
        )
        await helix_search_with_embeddings.index_entity(entity)

    async def test_index_episode_with_embedding(
        self, helix_search_with_embeddings, test_group_id
    ):
        episode = Episode(
            id=f"ep_{_uid()}",
            content="Python is used in data science and AI",
            group_id=test_group_id,
        )
        await helix_search_with_embeddings.index_episode(episode)

    async def test_batch_index_entities_with_embedding(
        self, helix_search_with_embeddings, test_group_id
    ):
        entities = [
            Entity(
                id=f"ent_{_uid()}",
                name=f"BatchEnt{i}",
                entity_type="Test",
                summary=f"Summary for entity {i}",
                group_id=test_group_id,
            )
            for i in range(3)
        ]
        count = await helix_search_with_embeddings.batch_index_entities(entities)
        assert count == 3

    async def test_batch_index_empty_list(
        self, helix_search_with_embeddings, test_group_id
    ):
        count = await helix_search_with_embeddings.batch_index_entities([])
        assert count == 0


# ======================================================================
# Vector search (with embeddings)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestVectorSearch:
    async def test_search_entities_returns_scored_tuples(
        self, helix_search_with_embeddings, test_group_id
    ):
        index = helix_search_with_embeddings
        entity = Entity(
            id=f"ent_{_uid()}",
            name="FastAPI Framework",
            entity_type="Technology",
            summary="Modern web framework for building APIs with Python",
            group_id=test_group_id,
        )
        await index.index_entity(entity)

        results = await index.search("FastAPI", group_id=test_group_id, limit=10)
        assert isinstance(results, list)
        for eid, score in results:
            assert isinstance(eid, str)
            assert isinstance(score, float)

    async def test_search_episodes_returns_scored_tuples(
        self, helix_search_with_embeddings, test_group_id
    ):
        index = helix_search_with_embeddings
        episode = Episode(
            id=f"ep_{_uid()}",
            content="We discussed FastAPI and Python at the meeting",
            group_id=test_group_id,
        )
        await index.index_episode(episode)

        results = await index.search_episodes(
            "FastAPI meeting", group_id=test_group_id, limit=10
        )
        assert isinstance(results, list)
        for eid, score in results:
            assert isinstance(eid, str)
            assert isinstance(score, float)


# ======================================================================
# Hybrid search (combined vector + BM25 via RRF)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestHybridSearch:
    async def test_hybrid_search_scores_normalized(
        self, helix_search_with_embeddings, test_group_id
    ):
        index = helix_search_with_embeddings
        entity = Entity(
            id=f"ent_{_uid()}",
            name="React",
            entity_type="Technology",
            summary="JavaScript library for building user interfaces",
            group_id=test_group_id,
        )
        await index.index_entity(entity)

        results = await index.search("React", group_id=test_group_id)
        for _eid, score in results:
            assert 0.0 <= score <= 1.0

    async def test_hybrid_search_respects_limit(
        self, helix_search_with_embeddings, test_group_id
    ):
        index = helix_search_with_embeddings
        for i in range(5):
            await index.index_entity(
                Entity(
                    id=f"ent_{_uid()}",
                    name=f"LimitTestEntity{i}",
                    entity_type="Test",
                    summary=f"Entity {i} for limit testing",
                    group_id=test_group_id,
                )
            )
        results = await index.search("LimitTest", group_id=test_group_id, limit=2)
        assert len(results) <= 2


# ======================================================================
# Deletion (best-effort no-op in Helix)
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestDeletion:
    async def test_remove_entity_is_noop(self, helix_search_index):
        """remove() is a documented no-op in Helix, should not raise."""
        await helix_search_index.remove(f"ent_{_uid()}")

    async def test_delete_group_is_noop(self, helix_search_index):
        """delete_group() is a documented no-op in Helix, should not raise."""
        await helix_search_index.delete_group(f"grp_{_uid()}")


# ======================================================================
# Compute similarity
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestComputeSimilarity:
    async def test_compute_similarity_empty_ids(
        self, helix_search_with_embeddings, test_group_id
    ):
        result = await helix_search_with_embeddings.compute_similarity(
            "Python", [], group_id=test_group_id
        )
        assert result == {}

    async def test_compute_similarity_returns_dict(
        self, helix_search_with_embeddings, test_group_id
    ):
        index = helix_search_with_embeddings
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id=test_group_id,
        )
        await index.index_entity(entity)

        result = await index.compute_similarity(
            "Python language", [eid], group_id=test_group_id
        )
        assert isinstance(result, dict)
        # Entity may or may not be found via zero-vector sweep depending on
        # Helix index state; we only check the type contract.
        for found_id, score in result.items():
            assert isinstance(found_id, str)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0


# ======================================================================
# Lifecycle
# ======================================================================


@pytest.mark.asyncio
@pytest.mark.requires_helix
@_helix_skip
class TestLifecycle:
    async def test_ensure_client_raises_before_init(self):
        """_ensure_client before initialize() should raise RuntimeError."""
        from engram.embeddings.provider import NoopProvider

        helix_config = HelixDBConfig(host="localhost", port=6969, verbose=False)
        index = HelixSearchIndex(
            helix_config=helix_config,
            provider=NoopProvider(),
            embed_config=EmbeddingConfig(),
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            index._ensure_client()

    async def test_close_is_noop(self, helix_search_index):
        """close() should not raise."""
        await helix_search_index.close()

    async def test_initialize_is_idempotent(self, helix_search_index):
        """Calling initialize() twice should not raise."""
        await helix_search_index.initialize()


# ======================================================================
# Pure-Python helpers (no HelixDB needed, but marked for module coherence)
# ======================================================================


class TestCosine:
    def test_identical_vectors(self):
        vec = [1.0, 0.0, 0.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=0.001)

    def test_empty_vectors(self):
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_length(self):
        assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0


class TestRRFFusion:
    def test_rrf_basic(self):
        fts = [("a", 10.0), ("b", 5.0)]
        vec = [("b", 0.9), ("c", 0.8)]
        merged = _rrf_fusion(fts, vec, fts_weight=0.3, vec_weight=0.7)
        assert len(merged) == 3
        # "b" appears in both lists so should rank first
        assert merged[0][0] == "b"
        for _, score in merged:
            assert 0.0 <= score <= 1.0

    def test_rrf_empty_inputs(self):
        assert _rrf_fusion([], [], 0.3, 0.7) == []

    def test_rrf_single_source(self):
        merged = _rrf_fusion([("a", 1.0)], [], 0.3, 0.7)
        assert len(merged) == 1
        assert merged[0][0] == "a"
