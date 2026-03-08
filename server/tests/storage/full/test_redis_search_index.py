"""Tests for Redis Search HNSW index."""

import pytest

from engram.config import EmbeddingConfig
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.storage.vector.redis_search import RedisSearchIndex

pytestmark = pytest.mark.requires_docker


class _TextFallbackProvider:
    """Deterministic local provider that forces text fallback at query time."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            token_count = float(max(len(text.split()), 1))
            char_count = float(max(len(text), 1))
            vowel_count = float(sum(ch.lower() in "aeiou" for ch in text) or 1)
            vectors.append([token_count, char_count, vowel_count, 1.0])
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        del text
        return []


async def _make_text_fallback_index(redis_search_runtime) -> RedisSearchIndex:
    index = RedisSearchIndex(
        redis_search_runtime["client"],
        provider=_TextFallbackProvider(),
        config=EmbeddingConfig(provider="test", model="test", dimensions=4),
        index_name=redis_search_runtime["index_name"],
        key_prefix=redis_search_runtime["key_prefix"],
    )
    await index.initialize()
    return index


@pytest.mark.asyncio
class TestRedisSearchIndex:
    async def test_initialize_is_idempotent(self, redis_search_index):
        # Call initialize again — should not raise
        await redis_search_index.initialize()

    async def test_search_with_noop_provider_returns_empty(self, redis_search_index):
        results = await redis_search_index.search("test query", group_id="default")
        assert results == []

    async def test_index_entity_with_noop_is_noop(self, redis_search_index):
        entity = Entity(
            id="ent_idx1",
            name="TestEntity",
            entity_type="Concept",
            summary="A test entity",
            group_id="default",
        )
        # Should not raise even with NoopProvider
        await redis_search_index.index_entity(entity)

    async def test_remove_is_safe_with_no_data(self, redis_search_index):
        # Removing non-existent entity should not raise
        await redis_search_index.remove("nonexistent_id")

    async def test_hash_key_format(self, redis_search_index):
        key = redis_search_index._hash_key("mygroup", "entity", "ent_123")
        assert key == f"{redis_search_index._key_prefix}mygroup:vec:entity:ent_123"

    async def test_search_episodes_returns_indexed_raw_episode(self, redis_search_runtime):
        index = await _make_text_fallback_index(redis_search_runtime)
        await index.index_episode(
            Episode(
                id="ep_raw",
                content="Conference planning notes with Alice and Bob",
                source="test",
                group_id="default",
            )
        )

        results = await index.search_episodes("conference planning", group_id="default")

        assert [episode_id for episode_id, _ in results] == ["ep_raw"]

    async def test_episode_and_cue_search_are_scoped_independently(self, redis_search_runtime):
        index = await _make_text_fallback_index(redis_search_runtime)
        await index.index_episode(
            Episode(
                id="ep_episode",
                content="Budget review for migration planning",
                source="test",
                group_id="default",
            )
        )
        await index.index_episode_cue(
            EpisodeCue(
                episode_id="ep_cue",
                group_id="default",
                cue_text="migration followup reminder",
            )
        )

        episode_results = await index.search_episodes("budget review", group_id="default")
        cue_results = await index.search_episode_cues(
            "followup reminder",
            group_id="default",
        )

        assert [episode_id for episode_id, _ in episode_results] == ["ep_episode"]
        assert [episode_id for episode_id, _ in cue_results] == ["ep_cue"]
