"""In-memory ring buffer store for notifications."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from engram.notifications.models import MemoryNotification


class NotificationStore:
    """Per-group ring buffer of notifications.

    Thread-safe via a single lock. Notifications are stored in insertion
    order and evicted FIFO when the per-group cap is reached.
    """

    def __init__(self, max_per_group: int = 200) -> None:
        self._max = max_per_group
        self._lock = threading.Lock()
        # group_id -> OrderedDict[notification_id -> MemoryNotification]
        self._groups: dict[str, OrderedDict[str, MemoryNotification]] = {}

    def add(self, notification: MemoryNotification) -> None:
        """Add a notification, evicting oldest if at capacity."""
        with self._lock:
            ring = self._groups.setdefault(notification.group_id, OrderedDict())
            ring[notification.id] = notification
            while len(ring) > self._max:
                ring.popitem(last=False)

    def get_pending(self, group_id: str, limit: int = 20) -> list[MemoryNotification]:
        """Return undismissed notifications, newest first."""
        with self._lock:
            ring = self._groups.get(group_id, OrderedDict())
            result = [n for n in reversed(ring.values()) if n.dismissed_at is None]
        return result[:limit]

    def get_since(self, group_id: str, since_ts: float) -> list[MemoryNotification]:
        """Return notifications created after *since_ts*, oldest first."""
        with self._lock:
            ring = self._groups.get(group_id, OrderedDict())
            return [n for n in ring.values() if n.created_at > since_ts]

    def dismiss(self, notification_id: str) -> bool:
        """Mark a single notification as dismissed. Returns True if found."""
        now = time.time()
        with self._lock:
            for ring in self._groups.values():
                n = ring.get(notification_id)
                if n is not None:
                    n.dismissed_at = now
                    return True
        return False

    def dismiss_batch(self, notification_ids: list[str]) -> int:
        """Dismiss multiple notifications. Returns count dismissed."""
        now = time.time()
        count = 0
        with self._lock:
            for ring in self._groups.values():
                for nid in notification_ids:
                    n = ring.get(nid)
                    if n is not None and n.dismissed_at is None:
                        n.dismissed_at = now
                        count += 1
        return count

    def get_for_mcp(
        self,
        group_id: str,
        limit: int = 3,
        max_surfaces: int = 2,
    ) -> list[MemoryNotification]:
        """Return undismissed notifications for MCP piggyback.

        Increments ``surfaced_count``. Notifications already surfaced
        ``max_surfaces`` times are excluded to prevent stale clutter.
        """
        with self._lock:
            ring = self._groups.get(group_id, OrderedDict())
            result: list[MemoryNotification] = []
            for n in reversed(ring.values()):
                if n.dismissed_at is not None:
                    continue
                if n.surfaced_count >= max_surfaces:
                    continue
                result.append(n)
                if len(result) >= limit:
                    break
            for n in result:
                n.surfaced_count += 1
        return result
