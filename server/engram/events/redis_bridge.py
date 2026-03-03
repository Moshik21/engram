"""Redis pub/sub bridge for cross-process EventBus events.

Publisher side (MCP server): hooks into EventBus.publish() and forwards
events to a Redis channel so the REST API server can pick them up.

Subscriber side (REST API server): listens on the Redis channel and
injects received events into the local EventBus for WebSocket delivery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from engram.events.bus import EventBus

logger = logging.getLogger(__name__)

_BRIDGE_ORIGIN = "__redis_bridge__"
_CHANNEL_PREFIX = "engram:events:"


class RedisEventPublisher:
    """Async hook that publishes EventBus events to a Redis channel."""

    def __init__(self, redis_client, group_id: str) -> None:
        self._redis = redis_client
        self._channel = f"{_CHANNEL_PREFIX}{group_id}"

    async def __call__(
        self, group_id: str, event_type: str, payload: dict, event: dict
    ) -> None:
        """EventBus on-publish hook signature."""
        # Loop prevention: don't re-publish events that came from Redis
        if event.get("_origin") == _BRIDGE_ORIGIN:
            return

        # Strip internal keys (prefixed with _) before serializing
        cleaned = {k: v for k, v in event.items() if not k.startswith("_")}
        try:
            await self._redis.publish(self._channel, json.dumps(cleaned))
        except Exception:
            logger.warning("Failed to publish event to Redis", exc_info=True)

    async def close(self) -> None:
        """Close the Redis client."""
        await self._redis.aclose()


class RedisEventSubscriber:
    """Background task that subscribes to a Redis channel and injects events."""

    def __init__(self, redis_client, group_id: str, event_bus: EventBus) -> None:
        self._redis = redis_client
        self._channel = f"{_CHANNEL_PREFIX}{group_id}"
        self._event_bus = event_bus
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the subscriber background task."""
        self._running = True
        self._task = asyncio.create_task(self._listen())
        logger.info("Redis event subscriber started on channel %s", self._channel)

    async def stop(self) -> None:
        """Stop the subscriber gracefully."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self._redis.aclose()
        logger.info("Redis event subscriber stopped")

    async def _listen(self) -> None:
        """Subscribe and process messages with exponential backoff reconnection."""
        backoff = 1.0
        max_backoff = 30.0

        while self._running:
            try:
                pubsub = self._redis.pubsub()
                await pubsub.subscribe(self._channel)
                backoff = 1.0  # Reset on successful connect

                async for message in pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] != "message":
                        continue

                    try:
                        data = message["data"]
                        if isinstance(data, bytes):
                            data = data.decode("utf-8")
                        event = json.loads(data)

                        self._event_bus.publish(
                            group_id=event.get("group_id", "default"),
                            event_type=event.get("type", "unknown"),
                            payload=event.get("payload", {}),
                            _origin=_BRIDGE_ORIGIN,
                        )
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("Malformed event from Redis", exc_info=True)

                await pubsub.unsubscribe(self._channel)
                await pubsub.aclose()

            except asyncio.CancelledError:
                raise
            except Exception:
                if not self._running:
                    break
                logger.warning(
                    "Redis subscriber disconnected, reconnecting in %.1fs",
                    backoff,
                    exc_info=True,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)


async def create_publisher(
    group_id: str, redis_url: str | None = None
) -> RedisEventPublisher | None:
    """Create a RedisEventPublisher, returning None if Redis is unavailable."""
    url = redis_url or os.environ.get("ENGRAM_REDIS__URL", "redis://localhost:6381/0")
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        await client.ping()
        logger.info("Redis event publisher connected")
        return RedisEventPublisher(client, group_id)
    except Exception:
        logger.info("Redis not available — event bridge disabled")
        return None


async def create_subscriber(
    group_id: str, event_bus: EventBus, redis_url: str | None = None
) -> RedisEventSubscriber | None:
    """Create a RedisEventSubscriber, returning None if Redis is unavailable."""
    url = redis_url or os.environ.get("ENGRAM_REDIS__URL", "redis://localhost:6381/0")
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(url, decode_responses=True)
        await client.ping()
        logger.info("Redis event subscriber connected")
        return RedisEventSubscriber(client, group_id, event_bus)
    except Exception:
        logger.info("Redis not available — event bridge disabled")
        return None
