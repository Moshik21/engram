"""Dashboard WebSocket command and event presentation helpers."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from engram.notifications.surface import (
    NotificationSurfaceService,
    build_api_notification_dismiss_surface,
)


def flatten_dashboard_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten EventBus payload fields into the dashboard WebSocket envelope."""
    flat = {k: v for k, v in event.items() if k != "payload"}
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        flat.update(payload)
    return flat


def build_dashboard_pong_surface(*, now: Callable[[], float] = time.time) -> dict[str, Any]:
    """Return the dashboard WebSocket pong payload."""
    return {
        "type": "pong",
        "timestamp": now(),
    }


def build_dashboard_resync_surface(
    bus: Any,
    *,
    group_id: str,
    last_seq: Any,
) -> dict[str, Any]:
    """Return the dashboard WebSocket missed-event replay payload."""
    try:
        normalized_last_seq = int(last_seq)
    except (TypeError, ValueError):
        normalized_last_seq = 0

    events, is_full = bus.get_events_since(group_id, normalized_last_seq)
    return {
        "type": "resync",
        "events": events,
        "isFull": is_full,
    }


def build_dashboard_activation_snapshot_message(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the dashboard WebSocket activation snapshot event payload."""
    return {
        "type": "activation.snapshot",
        "payload": {"topActivated": snapshot.get("topActivated", [])},
    }


def dismiss_dashboard_notification_command(
    service: NotificationSurfaceService | None,
    *,
    group_id: str,
    notification_id: Any,
) -> dict[str, int]:
    """Dismiss one notification for the connected dashboard brain."""
    if notification_id is None or notification_id == "":
        return {"dismissed": 0}
    return build_api_notification_dismiss_surface(
        service,
        group_id=group_id,
        ids=[str(notification_id)],
    )
