"""Tests for proactive memory notification surfacing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.notifications.models import (
    NOTIFICATION_TYPES,
    PRIORITY_LEVELS,
    MemoryNotification,
    notification_to_dict,
)
from engram.notifications.store import NotificationStore

# ─── Helpers ──────────────────────────────────────────────────────


def _make_notification(
    group_id: str = "default",
    notification_type: str = "dream_association",
    priority: str = "normal",
    title: str = "Test",
    body: str = "Test body",
    entity_ids: list[str] | None = None,
    **kwargs,
) -> MemoryNotification:
    return MemoryNotification(
        group_id=group_id,
        notification_type=notification_type,
        priority=priority,
        title=title,
        body=body,
        entity_ids=entity_ids or [],
        metadata={},
        created_at=kwargs.pop("created_at", time.time()),
        **kwargs,
    )


# ─── Model tests ─────────────────────────────────────────────────


class TestMemoryNotification:
    def test_construction(self):
        n = _make_notification()
        assert n.id.startswith("ntf_")
        assert n.dismissed_at is None
        assert n.surfaced_count == 0

    def test_id_auto_generated(self):
        a = _make_notification()
        b = _make_notification()
        assert a.id != b.id

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Unknown notification type"):
            _make_notification(notification_type="invalid_type")

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError, match="Unknown priority"):
            _make_notification(priority="critical")

    def test_all_types_valid(self):
        for t in NOTIFICATION_TYPES:
            n = _make_notification(notification_type=t)
            assert n.notification_type == t

    def test_all_priorities_valid(self):
        for p in PRIORITY_LEVELS:
            n = _make_notification(priority=p)
            assert n.priority == p

    def test_serialization(self):
        n = _make_notification(
            title="Dream link",
            entity_ids=["e1", "e2"],
            source_cycle_id="cyc_123",
        )
        d = notification_to_dict(n)
        assert d["id"] == n.id
        assert d["title"] == "Dream link"
        assert d["entity_ids"] == ["e1", "e2"]
        assert d["source_cycle_id"] == "cyc_123"
        assert d["dismissed_at"] is None
        assert d["surfaced_count"] == 0

    def test_serialization_roundtrip_keys(self):
        n = _make_notification()
        d = notification_to_dict(n)
        expected_keys = {
            "id",
            "group_id",
            "notification_type",
            "priority",
            "title",
            "body",
            "entity_ids",
            "metadata",
            "source_cycle_id",
            "created_at",
            "dismissed_at",
            "surfaced_count",
        }
        assert set(d.keys()) == expected_keys


# ─── Store tests ──────────────────────────────────────────────────


class TestNotificationStore:
    def test_add_and_get_pending(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].id == n.id

    def test_get_pending_excludes_dismissed(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        store.dismiss(n.id)
        pending = store.get_pending("default")
        assert len(pending) == 0

    def test_get_pending_newest_first(self):
        store = NotificationStore()
        n1 = _make_notification(title="First", created_at=1.0)
        n2 = _make_notification(title="Second", created_at=2.0)
        store.add(n1)
        store.add(n2)
        pending = store.get_pending("default")
        assert pending[0].title == "Second"
        assert pending[1].title == "First"

    def test_get_pending_respects_limit(self):
        store = NotificationStore()
        for i in range(10):
            store.add(_make_notification(title=f"N{i}"))
        pending = store.get_pending("default", limit=3)
        assert len(pending) == 3

    def test_get_since(self):
        store = NotificationStore()
        n1 = _make_notification(created_at=100.0)
        n2 = _make_notification(created_at=200.0)
        n3 = _make_notification(created_at=300.0)
        store.add(n1)
        store.add(n2)
        store.add(n3)
        result = store.get_since("default", 150.0)
        assert len(result) == 2
        assert result[0].created_at == 200.0

    def test_get_since_returns_oldest_first(self):
        store = NotificationStore()
        store.add(_make_notification(created_at=100.0))
        store.add(_make_notification(created_at=200.0))
        result = store.get_since("default", 0.0)
        assert result[0].created_at == 100.0

    def test_dismiss(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        assert store.dismiss(n.id) is True
        assert n.dismissed_at is not None

    def test_dismiss_nonexistent(self):
        store = NotificationStore()
        assert store.dismiss("ntf_nonexistent") is False

    def test_dismiss_batch(self):
        store = NotificationStore()
        n1 = _make_notification()
        n2 = _make_notification()
        n3 = _make_notification()
        store.add(n1)
        store.add(n2)
        store.add(n3)
        count = store.dismiss_batch([n1.id, n3.id])
        assert count == 2
        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].id == n2.id

    def test_dismiss_batch_ignores_already_dismissed(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        store.dismiss(n.id)
        count = store.dismiss_batch([n.id])
        assert count == 0

    def test_ring_buffer_eviction(self):
        store = NotificationStore(max_per_group=5)
        ids = []
        for i in range(8):
            n = _make_notification(title=f"N{i}")
            store.add(n)
            ids.append(n.id)
        pending = store.get_pending("default")
        assert len(pending) == 5
        # Oldest 3 should be evicted
        pending_ids = {p.id for p in pending}
        for old_id in ids[:3]:
            assert old_id not in pending_ids

    def test_group_isolation(self):
        store = NotificationStore()
        store.add(_make_notification(group_id="group_a"))
        store.add(_make_notification(group_id="group_b"))
        assert len(store.get_pending("group_a")) == 1
        assert len(store.get_pending("group_b")) == 1
        assert len(store.get_pending("group_c")) == 0

    def test_get_for_mcp(self):
        store = NotificationStore()
        for i in range(5):
            store.add(_make_notification(title=f"N{i}"))
        result = store.get_for_mcp("default", limit=3)
        assert len(result) == 3
        # surfaced_count should be incremented
        for n in result:
            assert n.surfaced_count == 1

    def test_get_for_mcp_excludes_dismissed(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        store.dismiss(n.id)
        assert len(store.get_for_mcp("default")) == 0

    def test_get_for_mcp_max_surfaces(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        # First surface
        result1 = store.get_for_mcp("default", limit=3, max_surfaces=2)
        assert len(result1) == 1
        # Second surface
        result2 = store.get_for_mcp("default", limit=3, max_surfaces=2)
        # After first call surfaced_count=1. Second call: 1<2 → included, bumps to 2.
        # Third call would see 2 >= 2 → excluded.
        result3 = store.get_for_mcp("default", limit=3, max_surfaces=2)
        assert len(result2) == 1
        assert len(result3) == 0

    def test_get_for_mcp_surface_count_increments(self):
        store = NotificationStore()
        n = _make_notification()
        store.add(n)
        store.get_for_mcp("default", limit=3, max_surfaces=3)
        assert n.surfaced_count == 1
        store.get_for_mcp("default", limit=3, max_surfaces=3)
        assert n.surfaced_count == 2
        store.get_for_mcp("default", limit=3, max_surfaces=3)
        assert n.surfaced_count == 3
        # Now at max, should not be returned
        result = store.get_for_mcp("default", limit=3, max_surfaces=3)
        assert len(result) == 0


# ─── Collector tests ──────────────────────────────────────────────


class TestNotificationCollector:
    def _make_cfg(self, **overrides):
        cfg = MagicMock()
        cfg.notification_dream_enabled = True
        cfg.notification_schema_enabled = True
        cfg.notification_maturation_enabled = True
        cfg.notification_merge_enabled = True
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.mark.asyncio
    async def test_ignores_non_consolidation_events(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        collector = NotificationCollector(store, self._make_cfg())
        await collector.on_event("default", "episode.queued", {}, {})
        assert len(store.get_pending("default")) == 0

    @pytest.mark.asyncio
    async def test_ignores_non_completed_events(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        collector = NotificationCollector(store, self._make_cfg())
        await collector.on_event(
            "default",
            "consolidation.phase.dream.started",
            {"cycle_id": "c1", "phase": "dream"},
            {},
        )
        assert len(store.get_pending("default")) == 0

    @pytest.mark.asyncio
    async def test_ignores_zero_items_affected(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()
        collector = NotificationCollector(store, self._make_cfg(), cs)
        await collector.on_event(
            "default",
            "consolidation.phase.dream.completed",
            {"cycle_id": "c1", "phase": "dream", "items_affected": 0},
            {},
        )
        assert len(store.get_pending("default")) == 0

    @pytest.mark.asyncio
    async def test_dream_creates_notification(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeDreamAssoc:
            source_entity_id: str = "e1"
            target_entity_id: str = "e2"
            source_entity_name: str = "Python"
            target_entity_name: str = "Music"
            source_domain: str = "tech"
            target_domain: str = "art"
            surprise_score: float = 0.85

        cs.get_dream_association_records = AsyncMock(return_value=[FakeDreamAssoc()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            await collector.on_event(
                "default",
                "consolidation.phase.dream.completed",
                {"cycle_id": "c1", "phase": "dream", "items_affected": 1},
                {},
            )

        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].notification_type == "dream_association"
        assert "Python" in pending[0].body

    @pytest.mark.asyncio
    async def test_schema_creates_notification(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeSchema:
            schema_entity_id: str = "s1"
            schema_name: str = "Person-WORKS_AT-Organization"
            instance_count: int = 7
            predicate_count: int = 2
            action: str = "created"

        cs.get_schema_records = AsyncMock(return_value=[FakeSchema()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            await collector.on_event(
                "default",
                "consolidation.phase.schema.completed",
                {"cycle_id": "c2", "phase": "schema", "items_affected": 1},
                {},
            )

        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].notification_type == "schema_discovery"

    @pytest.mark.asyncio
    async def test_maturation_creates_notification(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeMat:
            entity_id: str = "e1"
            entity_name: str = "Konnor"
            old_tier: str = "episodic"
            new_tier: str = "transitional"

        cs.get_maturation_records = AsyncMock(return_value=[FakeMat()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            await collector.on_event(
                "default",
                "consolidation.phase.mature.completed",
                {"cycle_id": "c3", "phase": "mature", "items_affected": 1},
                {},
            )

        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].notification_type == "entity_maturation"
        assert "Konnor" in pending[0].body

    @pytest.mark.asyncio
    async def test_merge_creates_notification(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeMerge:
            keep_id: str = "e1"
            remove_id: str = "e2"
            keep_name: str = "Konnor Moshier"
            remove_name: str = "Konner"
            decision_reason: str | None = None

        cs.get_merge_records = AsyncMock(return_value=[FakeMerge()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            await collector.on_event(
                "default",
                "consolidation.phase.merge.completed",
                {"cycle_id": "c4", "phase": "merge", "items_affected": 1},
                {},
            )

        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].notification_type == "entity_merge"
        assert pending[0].priority == "normal"

    @pytest.mark.asyncio
    async def test_merge_identity_core_is_high_priority(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeMerge:
            keep_id: str = "e1"
            remove_id: str = "e2"
            keep_name: str = "Konnor"
            remove_name: str = "Konner"
            decision_reason: str = "identity_core match"

        cs.get_merge_records = AsyncMock(return_value=[FakeMerge()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            await collector.on_event(
                "default",
                "consolidation.phase.merge.completed",
                {"cycle_id": "c5", "phase": "merge", "items_affected": 1},
                {},
            )

        pending = store.get_pending("default")
        assert pending[0].priority == "high"

    @pytest.mark.asyncio
    async def test_idempotency_prevents_duplicates(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()

        @dataclass
        class FakeDreamAssoc:
            source_entity_id: str = "e1"
            target_entity_id: str = "e2"
            source_entity_name: str = "A"
            target_entity_name: str = "B"
            source_domain: str = "x"
            target_domain: str = "y"
            surprise_score: float = 0.5

        cs.get_dream_association_records = AsyncMock(return_value=[FakeDreamAssoc()])
        collector = NotificationCollector(store, self._make_cfg(), cs)

        with patch("engram.events.bus.get_event_bus"):
            for _ in range(3):
                await collector.on_event(
                    "default",
                    "consolidation.phase.dream.completed",
                    {"cycle_id": "c1", "phase": "dream", "items_affected": 1},
                    {},
                )

        assert len(store.get_pending("default")) == 1

    @pytest.mark.asyncio
    async def test_disabled_type_skipped(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        cs = AsyncMock()
        cfg = self._make_cfg(notification_dream_enabled=False)
        collector = NotificationCollector(store, cfg, cs)

        await collector.on_event(
            "default",
            "consolidation.phase.dream.completed",
            {"cycle_id": "c1", "phase": "dream", "items_affected": 1},
            {},
        )
        assert len(store.get_pending("default")) == 0
        cs.get_dream_association_records.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_consolidation_store_no_crash(self):
        from engram.notifications.collector import NotificationCollector

        store = NotificationStore()
        collector = NotificationCollector(store, self._make_cfg(), consolidation_store=None)

        # Should not raise
        await collector.on_event(
            "default",
            "consolidation.phase.dream.completed",
            {"cycle_id": "c1", "phase": "dream", "items_affected": 1},
            {},
        )
        assert len(store.get_pending("default")) == 0


# ─── Temporal scanner tests ──────────────────────────────────────


class TestTemporalIntentionScanner:
    def _make_cfg(self, **overrides):
        cfg = MagicMock()
        cfg.notification_temporal_enabled = True
        cfg.notification_temporal_horizon_seconds = 3600.0
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.mark.asyncio
    async def test_approaching_deadline_creates_notification(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        now = time.time()
        expires = datetime.fromtimestamp(now + 1800, tz=timezone.utc).isoformat()
        entity = MagicMock()
        entity.id = "intent_1"
        entity.attributes = {
            "trigger_text": "meeting with Sarah",
            "action_text": "Remind about agenda",
            "expires_at": expires,
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])

        count = await scanner.scan("default", graph_store)
        assert count == 1

        pending = store.get_pending("default")
        assert len(pending) == 1
        assert pending[0].notification_type == "temporal_intention"
        assert pending[0].priority == "high"
        assert "meeting with Sarah" in pending[0].body

    @pytest.mark.asyncio
    async def test_expired_intention_skipped(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        entity = MagicMock()
        entity.id = "intent_2"
        entity.attributes = {
            "trigger_text": "old task",
            "action_text": "Do something",
            "expires_at": datetime.fromtimestamp(
                time.time() - 100, tz=timezone.utc
            ).isoformat(),
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])

        count = await scanner.scan("default", graph_store)
        assert count == 0

    @pytest.mark.asyncio
    async def test_far_future_intention_skipped(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        entity = MagicMock()
        entity.id = "intent_3"
        entity.attributes = {
            "trigger_text": "future task",
            "action_text": "Do something",
            "expires_at": datetime.fromtimestamp(
                time.time() + 86400, tz=timezone.utc
            ).isoformat(),
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])

        count = await scanner.scan("default", graph_store)
        assert count == 0

    @pytest.mark.asyncio
    async def test_already_fired_not_duplicated(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        entity = MagicMock()
        entity.id = "intent_4"
        entity.attributes = {
            "trigger_text": "task",
            "action_text": "action",
            "expires_at": datetime.fromtimestamp(
                time.time() + 600, tz=timezone.utc
            ).isoformat(),
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])

        count1 = await scanner.scan("default", graph_store)
        count2 = await scanner.scan("default", graph_store)
        assert count1 == 1
        assert count2 == 0
        assert len(store.get_pending("default")) == 1

    @pytest.mark.asyncio
    async def test_disabled_returns_zero(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        cfg = self._make_cfg(notification_temporal_enabled=False)
        scanner = TemporalIntentionScanner(store, cfg)

        graph_store = AsyncMock()
        count = await scanner.scan("default", graph_store)
        assert count == 0
        graph_store.list_intentions.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_expires_at_skipped(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        entity = MagicMock()
        entity.id = "intent_5"
        entity.attributes = {
            "trigger_text": "no deadline",
            "action_text": "action",
            # no expires_at
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])

        count = await scanner.scan("default", graph_store)
        assert count == 0

    @pytest.mark.asyncio
    async def test_time_display_hours(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        scanner = TemporalIntentionScanner(store, self._make_cfg())

        entity = MagicMock()
        entity.id = "intent_6"
        entity.attributes = {
            "trigger_text": "meeting",
            "action_text": "prepare",
            "expires_at": datetime.fromtimestamp(
                time.time() + 2700, tz=timezone.utc
            ).isoformat(),
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])
        await scanner.scan("default", graph_store)

        pending = store.get_pending("default")
        # Should show minutes since < 1 hour
        assert "minutes" in pending[0].title

    @pytest.mark.asyncio
    async def test_time_display_large_hours(self):
        from engram.notifications.temporal import TemporalIntentionScanner

        store = NotificationStore()
        cfg = self._make_cfg(notification_temporal_horizon_seconds=86400.0)
        scanner = TemporalIntentionScanner(store, cfg)

        entity = MagicMock()
        entity.id = "intent_7"
        entity.attributes = {
            "trigger_text": "deadline",
            "action_text": "submit",
            "expires_at": datetime.fromtimestamp(
                time.time() + 7200, tz=timezone.utc
            ).isoformat(),
        }

        graph_store = AsyncMock()
        graph_store.list_intentions = AsyncMock(return_value=[entity])
        await scanner.scan("default", graph_store)

        pending = store.get_pending("default")
        assert "hours" in pending[0].title
