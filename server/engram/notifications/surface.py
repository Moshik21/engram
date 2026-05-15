"""Notification read/dismiss surface shared by REST and MCP."""

from __future__ import annotations

from typing import Any

from engram.config import ActivationConfig
from engram.notifications.models import notification_to_dict
from engram.notifications.store import NotificationStore


class NotificationSurfaceService:
    """Present proactive memory notifications for public transports."""

    def __init__(self, store: NotificationStore) -> None:
        self._store = store

    def list_notifications(
        self,
        *,
        group_id: str,
        limit: int = 20,
        since: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return REST notification payloads for one brain."""
        if since > 0:
            items = self._store.get_since(group_id, since)
        else:
            items = self._store.get_pending(group_id, limit=limit)
        return [notification_to_dict(notification) for notification in items]

    def dismiss_notifications(self, *, group_id: str, ids: list[str]) -> int:
        """Dismiss notification IDs scoped to one brain."""
        return self._store.dismiss_batch(ids, group_id=group_id)

    def mcp_notifications(
        self,
        *,
        cfg: ActivationConfig,
        group_id: str,
    ) -> list[dict[str, Any]] | None:
        """Return MCP piggyback notification payloads when enabled."""
        if not cfg.notification_surfacing_enabled:
            return None

        notifications = self._store.get_for_mcp(
            group_id,
            limit=cfg.notification_mcp_max_per_response,
            max_surfaces=cfg.notification_mcp_max_surfaces,
        )
        if not notifications:
            return None
        return [
            {
                "type": notification.notification_type,
                "title": notification.title,
                "body": notification.body,
                "priority": notification.priority,
            }
            for notification in notifications
        ]


def build_api_notifications_surface(
    service: NotificationSurfaceService | None,
    *,
    group_id: str,
    limit: int = 20,
    since: float = 0.0,
) -> dict[str, Any]:
    """Return the REST notification list payload."""
    if service is None:
        return {"notifications": []}
    return {
        "notifications": service.list_notifications(
            group_id=group_id,
            limit=limit,
            since=since,
        )
    }


def build_api_notification_dismiss_surface(
    service: NotificationSurfaceService | None,
    *,
    group_id: str,
    ids: list[str],
) -> dict[str, int]:
    """Return the REST notification dismissal payload."""
    if service is None:
        return {"dismissed": 0}
    return {"dismissed": service.dismiss_notifications(group_id=group_id, ids=ids)}


def get_notification_surface_service_from_state() -> NotificationSurfaceService | None:
    """Return the app notification surface service if notification storage exists."""
    from engram.main import _app_state

    existing = _app_state.get("notification_surface_service")
    if isinstance(existing, NotificationSurfaceService):
        return existing

    store = _app_state.get("notification_store")
    if store is None:
        return None
    return NotificationSurfaceService(store)
