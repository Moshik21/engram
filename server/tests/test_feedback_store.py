"""Tests for SQLite feedback store."""

import time

import pytest

from engram.models.feedback import FeedbackEvent
from engram.storage.sqlite.feedback import SQLiteFeedbackStore


@pytest.fixture
async def store(tmp_path):
    """Create and initialize a feedback store with a temp database."""
    db_path = str(tmp_path / "test_feedback.db")
    s = SQLiteFeedbackStore(db_path)
    await s.initialize()
    yield s
    await s.close()


class TestFeedbackStore:
    @pytest.mark.asyncio
    async def test_record_and_retrieve(self, store):
        """Record an event and retrieve it."""
        event = FeedbackEvent(
            entity_id="e1", event_type="returned",
            query="test query", group_id="default",
        )
        await store.record_event(event)
        events = await store.get_entity_feedback("e1", "default")
        assert len(events) == 1
        assert events[0].entity_id == "e1"
        assert events[0].event_type == "returned"
        assert events[0].query == "test query"

    @pytest.mark.asyncio
    async def test_stats_aggregation(self, store):
        """Stats correctly aggregate across event types."""
        events = [
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q1", group_id="g1",
            ),
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q2", group_id="g1",
            ),
            FeedbackEvent(
                entity_id="e1", event_type="ignored",
                query="q3", group_id="g1",
            ),
            FeedbackEvent(
                entity_id="e2", event_type="re_accessed",
                query="q4", group_id="g1",
            ),
        ]
        for e in events:
            await store.record_event(e)

        stats = await store.get_feedback_stats("g1")
        assert "e1" in stats
        assert stats["e1"].returned_count == 2
        assert stats["e1"].ignored_count == 1
        assert stats["e1"].total_events == 3
        assert "e2" in stats
        assert stats["e2"].re_accessed_count == 1

    @pytest.mark.asyncio
    async def test_group_id_filtering(self, store):
        """Events are filtered by group_id."""
        await store.record_event(
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q1", group_id="g1",
            ),
        )
        await store.record_event(
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q2", group_id="g2",
            ),
        )
        events_g1 = await store.get_entity_feedback("e1", "g1")
        events_g2 = await store.get_entity_feedback("e1", "g2")
        assert len(events_g1) == 1
        assert len(events_g2) == 1

    @pytest.mark.asyncio
    async def test_cleanup_ttl(self, store):
        """Cleanup removes events older than TTL."""
        old_event = FeedbackEvent(
            entity_id="e1", event_type="returned", query="q1", group_id="g1",
            timestamp=time.time() - (100 * 86400),
        )
        new_event = FeedbackEvent(
            entity_id="e2", event_type="returned", query="q2", group_id="g1",
            timestamp=time.time(),
        )
        await store.record_event(old_event)
        await store.record_event(new_event)
        deleted = await store.cleanup(ttl_days=90)
        assert deleted == 1
        events = await store.get_entity_feedback("e1", "g1")
        assert len(events) == 0
        events = await store.get_entity_feedback("e2", "g1")
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_multiple_event_types(self, store):
        """All event types are stored and aggregated correctly."""
        types = [
            "returned", "re_accessed", "mentioned_in_remember", "ignored",
        ]
        for t in types:
            await store.record_event(
                FeedbackEvent(
                    entity_id="e1", event_type=t, query="q", group_id="g1",
                ),
            )
        stats = await store.get_feedback_stats("g1")
        s = stats["e1"]
        assert s.returned_count == 1
        assert s.re_accessed_count == 1
        assert s.mentioned_count == 1
        assert s.ignored_count == 1
        assert s.total_events == 4

    @pytest.mark.asyncio
    async def test_empty_store(self, store):
        """Empty store returns empty results."""
        events = await store.get_entity_feedback("nonexistent", "g1")
        assert events == []
        stats = await store.get_feedback_stats("g1")
        assert stats == {}

    @pytest.mark.asyncio
    async def test_stats_per_group(self, store):
        """Stats are scoped to group_id."""
        await store.record_event(
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q", group_id="g1",
            ),
        )
        await store.record_event(
            FeedbackEvent(
                entity_id="e1", event_type="returned",
                query="q", group_id="g2",
            ),
        )
        stats_g1 = await store.get_feedback_stats("g1")
        stats_g2 = await store.get_feedback_stats("g2")
        assert stats_g1["e1"].returned_count == 1
        assert stats_g2["e1"].returned_count == 1
