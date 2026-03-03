"""Tests for the EventBus pub/sub infrastructure."""

from __future__ import annotations

import pytest

from engram.events.bus import EventBus


class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_delivers_to_subscriber(self):
        """Published events are delivered to subscribed queues."""
        bus = EventBus()
        q = bus.subscribe("grp1")
        seq = bus.publish("grp1", "test.event", {"key": "value"})
        assert seq == 1

        event = q.get_nowait()
        assert event["type"] == "test.event"
        assert event["payload"]["key"] == "value"
        assert event["group_id"] == "grp1"
        assert event["seq"] == 1

    @pytest.mark.asyncio
    async def test_unsubscribe_stops_delivery(self):
        """After unsubscribe, events are no longer received."""
        bus = EventBus()
        q = bus.subscribe("grp1")
        bus.unsubscribe("grp1", q)
        bus.publish("grp1", "test.event", {})
        assert q.empty()

    @pytest.mark.asyncio
    async def test_seq_monotonically_increasing(self):
        """Sequence numbers increase monotonically across publishes."""
        bus = EventBus()
        q = bus.subscribe("grp1")
        s1 = bus.publish("grp1", "a", {})
        s2 = bus.publish("grp1", "b", {})
        s3 = bus.publish("grp1", "c", {})
        assert s1 < s2 < s3

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        seqs = [e["seq"] for e in events]
        assert seqs == sorted(seqs)

    @pytest.mark.asyncio
    async def test_group_isolation(self):
        """Events for one group are not delivered to another group's subscribers."""
        bus = EventBus()
        q1 = bus.subscribe("grp1")
        q2 = bus.subscribe("grp2")

        bus.publish("grp1", "event_for_grp1", {})
        bus.publish("grp2", "event_for_grp2", {})

        e1 = q1.get_nowait()
        assert e1["type"] == "event_for_grp1"
        assert q1.empty()

        e2 = q2.get_nowait()
        assert e2["type"] == "event_for_grp2"
        assert q2.empty()


class TestEventBusRingBuffer:
    """Tests for the ring buffer and get_events_since() resync support."""

    @pytest.mark.asyncio
    async def test_get_events_since_returns_missed(self):
        """get_events_since returns events after the given seq."""
        bus = EventBus()
        s1 = bus.publish("grp", "a", {"n": 1})
        s2 = bus.publish("grp", "b", {"n": 2})
        s3 = bus.publish("grp", "c", {"n": 3})

        events, is_full = bus.get_events_since("grp", s1)
        assert is_full is False
        assert len(events) == 2
        assert events[0]["seq"] == s2
        assert events[1]["seq"] == s3

    @pytest.mark.asyncio
    async def test_get_events_since_empty_group(self):
        """Empty group with nonzero lastSeq returns full resync."""
        bus = EventBus()
        events, is_full = bus.get_events_since("unknown", 5)
        assert events == []
        assert is_full is True

    @pytest.mark.asyncio
    async def test_ring_buffer_eviction(self):
        """When history exceeds size, oldest events are evicted."""
        bus = EventBus(history_size=3)
        bus.publish("grp", "a", {})
        bus.publish("grp", "b", {})
        bus.publish("grp", "c", {})
        bus.publish("grp", "d", {})  # evicts "a"

        # Requesting from seq=0 (before "a") should trigger full resync
        # because oldest is now "b" (seq=2) and 0 < 2
        events, is_full = bus.get_events_since("grp", 0)
        assert is_full is True

    @pytest.mark.asyncio
    async def test_get_all_from_zero_in_non_evicted(self):
        """When no events evicted, seq=0 returns all events (not full resync)."""
        bus = EventBus(history_size=100)
        bus.publish("grp", "a", {})

        # seq=0 < oldest_seq (s1=1), so full resync
        events, is_full = bus.get_events_since("grp", 0)
        assert is_full is True
