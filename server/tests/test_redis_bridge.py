from __future__ import annotations

import json

import pytest

from engram.events.bus import EventBus
from engram.events.redis_bridge import RedisEventPublisher, RedisEventSubscriber


class FakeRedis:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.closed = False

    async def publish(self, channel: str, payload: str) -> None:
        self.published.append((channel, payload))

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_redis_publisher_keeps_events_in_configured_group_channel():
    redis = FakeRedis()
    publisher = RedisEventPublisher(redis, "brain_a")

    await publisher(
        "brain_b",
        "recall.triggered",
        {"query": "wrong brain"},
        {
            "seq": 1,
            "type": "recall.triggered",
            "group_id": "brain_b",
            "payload": {"query": "wrong brain"},
        },
    )

    assert redis.published == []

    await publisher(
        "brain_a",
        "recall.triggered",
        {"query": "active brain"},
        {
            "seq": 2,
            "type": "recall.triggered",
            "group_id": "brain_a",
            "payload": {"query": "active brain"},
        },
    )

    assert len(redis.published) == 1
    channel, payload = redis.published[0]
    assert channel == "engram:events:brain_a"
    assert json.loads(payload)["group_id"] == "brain_a"


def test_redis_subscriber_uses_channel_group_when_event_group_missing():
    bus = EventBus()
    queue = bus.subscribe("brain_a")
    subscriber = RedisEventSubscriber(FakeRedis(), "brain_a", bus)

    subscriber._publish_received_event(
        {
            "seq": 1,
            "type": "consolidation.completed",
            "payload": {"cycle_id": "cyc_1"},
        }
    )

    event = queue.get_nowait()
    assert event["group_id"] == "brain_a"
    assert event["type"] == "consolidation.completed"
    assert event["payload"] == {"cycle_id": "cyc_1"}


def test_redis_subscriber_drops_mismatched_event_group():
    bus = EventBus()
    brain_a_queue = bus.subscribe("brain_a")
    brain_b_queue = bus.subscribe("brain_b")
    subscriber = RedisEventSubscriber(FakeRedis(), "brain_a", bus)

    subscriber._publish_received_event(
        {
            "seq": 1,
            "type": "consolidation.completed",
            "group_id": "brain_b",
            "payload": {"cycle_id": "cyc_wrong"},
        }
    )

    assert brain_a_queue.empty()
    assert brain_b_queue.empty()
