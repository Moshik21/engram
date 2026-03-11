"""Tests for the Redis pub/sub event bridge."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.events.bus import EventBus
from engram.events.redis_bridge import (
    _BRIDGE_ORIGIN,
    RedisEventPublisher,
    RedisEventSubscriber,
    create_publisher,
    create_subscriber,
)

# ── EventBus hook mechanism ──────────────────────────────────────────


class TestEventBusHooks:
    """Test the on-publish hook mechanism added to EventBus."""

    def test_add_and_remove_hook(self):
        bus = EventBus()
        hook = AsyncMock()
        bus.add_on_publish_hook(hook)
        assert hook in bus._on_publish_hooks

        bus.remove_on_publish_hook(hook)
        assert hook not in bus._on_publish_hooks

    def test_remove_nonexistent_hook_no_error(self):
        bus = EventBus()
        bus.remove_on_publish_hook(AsyncMock())  # should not raise

    @pytest.mark.asyncio
    async def test_hook_fires_on_publish(self):
        bus = EventBus()
        hook = AsyncMock()
        bus.add_on_publish_hook(hook)

        bus.publish("g1", "entity.created", {"name": "Alice"})

        # Let the fire-and-forget task complete
        await asyncio.sleep(0.05)

        hook.assert_called_once()
        args = hook.call_args[0]
        assert args[0] == "g1"  # group_id
        assert args[1] == "entity.created"  # event_type
        assert args[2] == {"name": "Alice"}  # payload
        assert args[3]["seq"] == 1  # event dict

    @pytest.mark.asyncio
    async def test_origin_propagated_to_event(self):
        bus = EventBus()
        hook = AsyncMock()
        bus.add_on_publish_hook(hook)

        bus.publish("g1", "test", _origin="external")
        await asyncio.sleep(0.05)

        event = hook.call_args[0][3]
        assert event["_origin"] == "external"


# ── RedisEventPublisher ──────────────────────────────────────────────


class TestRedisEventPublisher:
    @pytest.mark.asyncio
    async def test_publishes_event_to_redis(self):
        mock_redis = AsyncMock()
        pub = RedisEventPublisher(mock_redis, "default")

        event = {
            "seq": 1,
            "type": "entity.created",
            "timestamp": 1000.0,
            "group_id": "default",
            "payload": {"name": "Alice"},
        }
        await pub("default", "entity.created", {"name": "Alice"}, event)

        mock_redis.publish.assert_called_once()
        channel, data = mock_redis.publish.call_args[0]
        assert channel == "engram:events:default"
        parsed = json.loads(data)
        assert parsed["type"] == "entity.created"
        assert parsed["payload"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_skips_bridge_origin_events(self):
        mock_redis = AsyncMock()
        pub = RedisEventPublisher(mock_redis, "default")

        event = {
            "seq": 1,
            "type": "entity.created",
            "timestamp": 1000.0,
            "group_id": "default",
            "payload": {},
            "_origin": _BRIDGE_ORIGIN,
        }
        await pub("default", "entity.created", {}, event)

        mock_redis.publish.assert_not_called()

    @pytest.mark.asyncio
    async def test_strips_internal_keys(self):
        mock_redis = AsyncMock()
        pub = RedisEventPublisher(mock_redis, "default")

        event = {
            "seq": 1,
            "type": "test",
            "timestamp": 1000.0,
            "group_id": "default",
            "payload": {},
            "_origin": "local",
            "_internal": "secret",
        }
        await pub("default", "test", {}, event)

        data = json.loads(mock_redis.publish.call_args[0][1])
        assert "_origin" not in data
        assert "_internal" not in data
        assert "seq" in data

    @pytest.mark.asyncio
    async def test_handles_redis_failure_gracefully(self):
        mock_redis = AsyncMock()
        mock_redis.publish.side_effect = ConnectionError("lost connection")
        pub = RedisEventPublisher(mock_redis, "default")

        event = {
            "seq": 1,
            "type": "test",
            "timestamp": 1000.0,
            "group_id": "default",
            "payload": {},
        }
        # Should not raise
        await pub("default", "test", {}, event)


# ── RedisEventSubscriber ─────────────────────────────────────────────


class TestRedisEventSubscriber:
    @pytest.mark.asyncio
    async def test_injects_event_into_event_bus(self):
        bus = EventBus()
        queue = bus.subscribe("default")

        # redis.asyncio .pubsub() is sync (returns PubSub, not coroutine)
        mock_redis = MagicMock()
        mock_redis.aclose = AsyncMock()
        sub = RedisEventSubscriber(mock_redis, "default", bus)

        event_data = json.dumps(
            {
                "seq": 42,
                "type": "entity.created",
                "group_id": "default",
                "payload": {"name": "Bob"},
            }
        )

        # Simulate a single message then stop
        async def mock_listen():
            yield {"type": "subscribe", "data": None}
            yield {"type": "message", "data": event_data.encode("utf-8")}
            sub._running = False

        mock_pubsub = AsyncMock()
        mock_pubsub.listen = mock_listen
        mock_redis.pubsub.return_value = mock_pubsub

        sub._running = True
        task = asyncio.create_task(sub._listen())
        await asyncio.sleep(0.1)
        sub._running = False
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The event bus should have received the event (with new seq)
        assert not queue.empty()
        event = queue.get_nowait()
        assert event["type"] == "entity.created"
        assert event["payload"]["name"] == "Bob"
        assert event["_origin"] == _BRIDGE_ORIGIN

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        mock_redis = AsyncMock()
        bus = EventBus()
        sub = RedisEventSubscriber(mock_redis, "default", bus)

        # Mock _listen to just wait
        async def fake_listen():
            while sub._running:
                await asyncio.sleep(0.01)

        with patch.object(sub, "_listen", fake_listen):
            await sub.start()
            assert sub._task is not None
            assert not sub._task.done()

            await sub.stop()
            assert not sub._running
            mock_redis.aclose.assert_called_once()


# ── Factory functions ────────────────────────────────────────────────


class TestFactories:
    @pytest.mark.asyncio
    async def test_create_publisher_returns_none_without_redis(self):
        result = await create_publisher("default", redis_url="redis://localhost:19999")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_subscriber_returns_none_without_redis(self):
        bus = EventBus()
        result = await create_subscriber("default", bus, redis_url="redis://localhost:19999")
        assert result is None


# ── Integration (requires_docker) ────────────────────────────────────


@pytest.mark.requires_docker
class TestRedisBridgeIntegration:
    @pytest.mark.asyncio
    async def test_end_to_end_bridge(self):
        """Publisher → Redis → Subscriber → EventBus round-trip."""
        url = "redis://localhost:6381/0"
        bus = EventBus()
        queue = bus.subscribe("default")

        publisher = await create_publisher("default", redis_url=url)
        subscriber = await create_subscriber("default", bus, redis_url=url)

        assert publisher is not None
        assert subscriber is not None

        await subscriber.start()
        await asyncio.sleep(0.1)

        # Simulate an event from the MCP side
        event = {
            "seq": 1,
            "type": "entity.created",
            "timestamp": 1000.0,
            "group_id": "default",
            "payload": {"name": "Integration Test"},
        }
        await publisher("default", "entity.created", {"name": "Integration Test"}, event)

        # Wait for the event to arrive
        try:
            received = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert received["type"] == "entity.created"
            assert received["payload"]["name"] == "Integration Test"
            assert received["_origin"] == _BRIDGE_ORIGIN
        finally:
            await subscriber.stop()
            await publisher.close()
