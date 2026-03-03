"""Tests for Redis Search HNSW index."""

import pytest

from engram.models.entity import Entity

pytestmark = pytest.mark.requires_docker


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
        assert key == "engram:mygroup:vec:entity:ent_123"
