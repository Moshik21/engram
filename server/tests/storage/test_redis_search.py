"""Mock-based tests for RedisSearchIndex (no Docker needed)."""

import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import EmbeddingConfig
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.storage.vector.redis_search import RedisSearchIndex, pack_vector


class MockEmbeddingProvider:
    """Mock embedding provider that returns fixed-dimension vectors."""

    def __init__(self, dim: int = 512):
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim


class NoopMockProvider:
    """Provider that signals embeddings are disabled."""

    def dimension(self) -> int:
        return 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    async def embed_query(self, text: str) -> list[float]:
        return []


def make_redis_mock() -> AsyncMock:
    """Create a mock Redis client with common methods."""
    redis = AsyncMock()
    redis.execute_command = AsyncMock()
    redis.hset = AsyncMock()
    redis.delete = AsyncMock()
    redis.pipeline = MagicMock()
    redis.aclose = AsyncMock()
    redis.scan_iter = MagicMock()
    return redis


def make_config(**overrides) -> EmbeddingConfig:
    defaults = dict(
        provider="voyage",
        model="voyage-4-lite",
        dimensions=512,
    )
    defaults.update(overrides)
    return EmbeddingConfig(**defaults)


@pytest.mark.asyncio
class TestRedisSearchIndex:
    async def test_ensure_index_creates_ft_index(self):
        """_ensure_index creates FT index with correct schema."""
        redis = make_redis_mock()
        # First call (FT.INFO) raises to indicate index doesn't exist
        # Second call (FT.CREATE) succeeds
        redis.execute_command.side_effect = [Exception("no such index"), None]

        idx = RedisSearchIndex(redis, MockEmbeddingProvider(), make_config())
        await idx.initialize()

        # FT.CREATE should be the second call
        calls = redis.execute_command.call_args_list
        assert len(calls) == 2
        create_call = calls[1]
        args = create_call[0]
        assert args[0] == "FT.CREATE"
        assert args[1] == "engram_vectors"
        assert "HNSW" in args

    async def test_ensure_index_handles_existing_index(self):
        """_ensure_index handles case when index already exists."""
        redis = make_redis_mock()
        # FT.INFO succeeds → index exists
        redis.execute_command.return_value = {"index_name": "engram_vectors"}

        idx = RedisSearchIndex(redis, MockEmbeddingProvider(), make_config())
        await idx.initialize()

        # Should only call FT.INFO, not FT.CREATE
        assert redis.execute_command.call_count == 1
        assert redis.execute_command.call_args[0][0] == "FT.INFO"

    async def test_upsert_stores_hash_and_vector(self):
        """upsert stores hash with vector via hset."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))

        entity = Entity(
            id="ent_1",
            name="Alice",
            entity_type="Person",
            summary="A test person",
            group_id="grp1",
        )
        await idx.index_entity(entity)

        redis.hset.assert_called_once()
        call_kwargs = redis.hset.call_args
        mapping = call_kwargs.kwargs.get("mapping") or call_kwargs[1].get("mapping")
        assert mapping["group_id"] == "grp1"
        assert mapping["content_type"] == "entity"
        assert mapping["source_id"] == "ent_1"
        assert mapping["entity_type"] == "Person"
        assert isinstance(mapping["embedding"], bytes)

    async def test_upsert_with_no_embeddings_skips(self):
        """When embeddings are disabled (NoopProvider), index_entity is a no-op."""
        redis = make_redis_mock()
        provider = NoopMockProvider()

        idx = RedisSearchIndex(redis, provider, make_config())

        entity = Entity(
            id="ent_1",
            name="Alice",
            entity_type="Person",
            group_id="grp1",
        )
        await idx.index_entity(entity)

        redis.hset.assert_not_called()

    async def test_remove_deletes_keys(self):
        """remove scans and deletes matching keys."""
        redis = make_redis_mock()

        # Mock scan_iter to return async iterator
        async def mock_scan(*args, **kwargs):
            pattern = kwargs.get("match", "")
            if "entity" in pattern:
                yield b"engram:grp1:vec:entity:ent_1"
            elif "episode" in pattern:
                return

        redis.scan_iter = mock_scan

        idx = RedisSearchIndex(redis, MockEmbeddingProvider(), make_config())
        await idx.remove("ent_1")

        redis.delete.assert_called_once_with(b"engram:grp1:vec:entity:ent_1")

    async def test_search_happy_path(self):
        """search returns scored results from FT.SEARCH response."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        # Mock FT.SEARCH response format:
        # [total_count, key1, [field, value, ...], key2, [...], ...]
        redis.execute_command.return_value = [
            1,
            b"engram:grp1:vec:entity:ent_1",
            [b"source_id", b"ent_1", b"score", b"0.2"],
        ]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx.search("Alice", group_id="grp1")

        assert len(results) == 1
        assert results[0][0] == "ent_1"
        # score 0.2 → similarity = 1 - (0.2/2) = 0.9
        assert abs(results[0][1] - 0.9) < 0.01

    async def test_search_with_group_id_filter(self):
        """search includes group_id TAG filter in query."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)
        redis.execute_command.return_value = [0]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        await idx.search("test", group_id="tenant_a")

        call_args = redis.execute_command.call_args[0]
        query_str = call_args[2]
        assert "@group_id:{tenant_a}" in query_str

    async def test_search_with_entity_types_filter(self):
        """search includes entity_type TAG filter in query (P4 fix)."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)
        redis.execute_command.return_value = [0]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        await idx.search("test", entity_types=["Person", "Organization"])

        call_args = redis.execute_command.call_args[0]
        query_str = call_args[2]
        assert "@entity_type:{Person|Organization}" in query_str

    async def test_search_no_results(self):
        """search returns empty list when no matches."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)
        redis.execute_command.return_value = [0]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx.search("nonexistent", group_id="grp1")

        assert results == []

    async def test_search_with_content_type_filter(self):
        """search includes content_type filter in query."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)
        redis.execute_command.return_value = [0]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        await idx.search("test")

        call_args = redis.execute_command.call_args[0]
        query_str = call_args[2]
        assert "@content_type:{entity}" in query_str

    async def test_search_handles_error_gracefully(self):
        """search returns empty list when FT.SEARCH raises."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)
        redis.execute_command.side_effect = Exception("Connection lost")

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx.search("test", group_id="grp1")

        assert results == []

    async def test_pack_vector_produces_correct_binary(self):
        """pack_vector produces correct float32 binary encoding."""
        vec = [1.0, 2.0, 3.0]
        packed = pack_vector(vec)

        assert len(packed) == 12  # 3 floats * 4 bytes each
        unpacked = struct.unpack("<3f", packed)
        assert abs(unpacked[0] - 1.0) < 1e-6
        assert abs(unpacked[1] - 2.0) < 1e-6
        assert abs(unpacked[2] - 3.0) < 1e-6

    async def test_index_episode_stores_hash(self):
        """index_episode stores episode hash with content."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))

        episode = Episode(
            id="ep_1",
            content="Alice met Bob at the conference",
            source="mcp",
            group_id="grp1",
        )
        await idx.index_episode(episode)

        redis.hset.assert_called_once()
        call_kwargs = redis.hset.call_args
        mapping = call_kwargs.kwargs.get("mapping") or call_kwargs[1].get("mapping")
        assert mapping["content_type"] == "episode"
        assert mapping["source_id"] == "ep_1"
        assert "Alice met Bob" in mapping["text"]

    async def test_text_search_returns_keyword_matches(self):
        """_text_search constructs correct FT.SEARCH query with text clause."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        # Mock FT.SEARCH response for text search
        redis.execute_command.return_value = [
            1,
            b"engram:grp1:vec:entity:ent_1",
            [b"source_id", b"ent_1", b"score", b"1.0"],
        ]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx._text_search("Konner books", group_id="grp1")

        assert len(results) == 1
        assert results[0][0] == "ent_1"
        assert results[0][1] == 0.5  # baseline score

        # Verify FT.SEARCH was called with text clause
        call_args = redis.execute_command.call_args[0]
        assert call_args[0] == "FT.SEARCH"
        query_str = call_args[2]
        assert "@text:" in query_str
        assert "Konner" in query_str

    async def test_search_supplements_with_text_when_knn_sparse(self):
        """search merges text results when KNN returns fewer than limit."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        # First call: KNN search returns 1 result
        # Second call: text search returns 1 additional result
        redis.execute_command.side_effect = [
            [
                1,
                b"engram:grp1:vec:entity:ent_1",
                [b"source_id", b"ent_1", b"score", b"0.2"],
            ],
            [
                1,
                b"engram:grp1:vec:entity:ent_2",
                [b"source_id", b"ent_2", b"score", b"1.0"],
            ],
        ]

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx.search("Konner books", group_id="grp1", limit=20)

        assert len(results) == 2
        ids = {r[0] for r in results}
        assert "ent_1" in ids  # from KNN
        assert "ent_2" in ids  # from text search

    async def test_text_search_handles_empty_query(self):
        """_text_search returns empty for queries with only short tokens."""
        redis = make_redis_mock()
        provider = MockEmbeddingProvider(dim=4)

        idx = RedisSearchIndex(redis, provider, make_config(dimensions=4))
        results = await idx._text_search("a b", group_id="grp1")

        assert results == []
        redis.execute_command.assert_not_called()

    async def test_search_disabled_falls_back_to_text(self):
        """search falls back to text search when embeddings disabled (NoopProvider)."""
        redis = make_redis_mock()
        provider = NoopMockProvider()

        # Text search returns a result
        redis.execute_command.return_value = [
            1,
            b"engram:grp1:vec:entity:ent_1",
            [b"source_id", b"ent_1", b"score", b"1.0"],
        ]

        idx = RedisSearchIndex(redis, provider, make_config())
        results = await idx.search("test query", group_id="grp1")

        # Should get text search result despite no embeddings
        assert len(results) == 1
        assert results[0][0] == "ent_1"

        # FT.SEARCH was called (text search fallback)
        call_args = redis.execute_command.call_args[0]
        assert call_args[0] == "FT.SEARCH"
        assert "@text:" in call_args[2]
