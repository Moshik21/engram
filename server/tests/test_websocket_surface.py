"""Tests for dashboard WebSocket presentation helpers."""

from __future__ import annotations

from engram.api.websocket_surface import (
    build_dashboard_activation_snapshot_message,
    build_dashboard_pong_surface,
    build_dashboard_resync_surface,
    dismiss_dashboard_notification_command,
    flatten_dashboard_event,
)
from engram.notifications.models import MemoryNotification
from engram.notifications.store import NotificationStore
from engram.notifications.surface import NotificationSurfaceService


class FakeBus:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def get_events_since(self, group_id: str, last_seq: int) -> tuple[list[dict], bool]:
        self.calls.append((group_id, last_seq))
        return ([{"seq": last_seq + 1, "group_id": group_id}], False)


def test_flatten_dashboard_event_lifts_payload_fields() -> None:
    event = {
        "seq": 7,
        "type": "episode.updated",
        "group_id": "brain",
        "payload": {"episodeId": "ep1", "status": "projected"},
    }

    assert flatten_dashboard_event(event) == {
        "seq": 7,
        "type": "episode.updated",
        "group_id": "brain",
        "episodeId": "ep1",
        "status": "projected",
    }


def test_build_dashboard_pong_surface_uses_clock() -> None:
    assert build_dashboard_pong_surface(now=lambda: 12.5) == {
        "type": "pong",
        "timestamp": 12.5,
    }


def test_build_dashboard_resync_surface_scopes_group_and_normalizes_seq() -> None:
    bus = FakeBus()

    assert build_dashboard_resync_surface(
        bus,
        group_id="brain",
        last_seq="bad",
    ) == {
        "type": "resync",
        "events": [{"seq": 1, "group_id": "brain"}],
        "isFull": False,
    }
    assert bus.calls == [("brain", 0)]


def test_build_dashboard_activation_snapshot_message_uses_dashboard_contract() -> None:
    snapshot = {
        "topActivated": [{"id": "e1", "activation": 0.7}],
        "ignored": True,
    }

    assert build_dashboard_activation_snapshot_message(snapshot) == {
        "type": "activation.snapshot",
        "payload": {"topActivated": [{"id": "e1", "activation": 0.7}]},
    }


def test_dismiss_dashboard_notification_command_scopes_to_connected_group() -> None:
    store = NotificationStore()
    service = NotificationSurfaceService(store)
    active = MemoryNotification(
        group_id="brain",
        notification_type="entity_merge",
        priority="normal",
        title="Active",
        body="Dismiss me.",
        entity_ids=[],
        metadata={},
        created_at=1.0,
    )
    other = MemoryNotification(
        group_id="other",
        notification_type="entity_merge",
        priority="normal",
        title="Other",
        body="Do not dismiss me.",
        entity_ids=[],
        metadata={},
        created_at=2.0,
    )
    store.add(active)
    store.add(other)

    assert dismiss_dashboard_notification_command(
        service,
        group_id="brain",
        notification_id=other.id,
    ) == {"dismissed": 0}
    assert other.dismissed_at is None

    assert dismiss_dashboard_notification_command(
        service,
        group_id="brain",
        notification_id=active.id,
    ) == {"dismissed": 1}
    assert active.dismissed_at is not None
