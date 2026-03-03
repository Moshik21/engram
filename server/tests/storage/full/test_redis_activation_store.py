"""Tests for Redis ActivationStore implementation."""

import time

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState

pytestmark = pytest.mark.requires_docker


@pytest.mark.asyncio
class TestRedisActivationStore:
    async def test_get_set_roundtrip(self, redis_activation_store):
        state = ActivationState(
            node_id="ent_1",
            access_history=[100.0, 200.0],
            spreading_bonus=0.5,
            last_accessed=200.0,
            access_count=2,
        )
        await redis_activation_store.set_activation("ent_1", state)
        result = await redis_activation_store.get_activation("ent_1")
        assert result is not None
        assert result.node_id == "ent_1"
        assert result.access_history == [100.0, 200.0]
        assert result.spreading_bonus == 0.5
        assert result.access_count == 2

    async def test_get_missing_returns_none(self, redis_activation_store):
        result = await redis_activation_store.get_activation("nonexistent")
        assert result is None

    async def test_batch_get_and_set(self, redis_activation_store):
        states = {
            "ent_a": ActivationState(node_id="ent_a", access_history=[100.0], access_count=1),
            "ent_b": ActivationState(node_id="ent_b", access_history=[200.0], access_count=1),
        }
        await redis_activation_store.batch_set(states)
        result = await redis_activation_store.batch_get(["ent_a", "ent_b", "missing"])
        assert "ent_a" in result
        assert "ent_b" in result
        assert "missing" not in result

    async def test_record_access(self, redis_activation_store):
        now = time.time()
        await redis_activation_store.record_access("ent_rec", now)
        state = await redis_activation_store.get_activation("ent_rec")
        assert state is not None
        assert len(state.access_history) == 1
        assert state.access_count == 1

        # Record another access
        await redis_activation_store.record_access("ent_rec", now + 10)
        state2 = await redis_activation_store.get_activation("ent_rec")
        assert state2 is not None
        assert len(state2.access_history) == 2
        assert state2.access_count == 2

    async def test_clear_activation(self, redis_activation_store):
        state = ActivationState(node_id="ent_clr", access_history=[100.0], access_count=1)
        await redis_activation_store.set_activation("ent_clr", state)
        await redis_activation_store.clear_activation("ent_clr")
        result = await redis_activation_store.get_activation("ent_clr")
        assert result is None

    async def test_get_top_activated(self, redis_activation_store):
        now = time.time()
        # Entity with many recent accesses should rank higher
        state_hot = ActivationState(
            node_id="ent_hot",
            access_history=[now - 1, now - 2, now - 3],
            access_count=3,
            last_accessed=now - 1,
        )
        state_cold = ActivationState(
            node_id="ent_cold",
            access_history=[now - 86400],
            access_count=1,
            last_accessed=now - 86400,
        )
        await redis_activation_store.set_activation("ent_hot", state_hot)
        await redis_activation_store.set_activation("ent_cold", state_cold)

        top = await redis_activation_store.get_top_activated(limit=10)
        assert len(top) >= 2
        # Hot entity should be first
        ids = [eid for eid, _ in top]
        assert ids.index("ent_hot") < ids.index("ent_cold")

    async def test_ttl_is_set(self, redis_activation_store, redis_client):
        """TTL should be positive and close to 90 days (default)."""
        state = ActivationState(node_id="ent_ttl", access_history=[100.0], access_count=1)
        await redis_activation_store.set_activation("ent_ttl", state)
        ttl = await redis_client.ttl("act:ent_ttl")
        # TTL should be positive and close to 90 days
        assert ttl > 0
        assert ttl <= 90 * 24 * 3600

    async def test_ttl_configurable(self, redis_client):
        """Store with custom activation_ttl_days uses that TTL."""
        from engram.storage.redis.activation import RedisActivationStore

        cfg = ActivationConfig(activation_ttl_days=30)
        store = RedisActivationStore(redis_client, cfg=cfg)
        state = ActivationState(node_id="ent_ttl30", access_history=[100.0], access_count=1)
        await store.set_activation("ent_ttl30", state)
        ttl = await redis_client.ttl("act:ent_ttl30")
        assert ttl > 0
        assert ttl <= 30 * 24 * 3600

    async def test_record_access_uses_config_ttl(self, redis_client):
        """record_access respects config TTL."""
        from engram.storage.redis.activation import RedisActivationStore

        cfg = ActivationConfig(activation_ttl_days=14)
        store = RedisActivationStore(redis_client, cfg=cfg)
        now = time.time()
        await store.record_access("ent_ttl14", now)
        ttl = await redis_client.ttl("act:ent_ttl14")
        assert ttl > 0
        assert ttl <= 14 * 24 * 3600

    async def test_batch_get_empty_list(self, redis_activation_store):
        result = await redis_activation_store.batch_get([])
        assert result == {}
