"""Event bus for real-time dashboard notifications."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

_EVENT_BUS: EventBus | None = None


class EventBus:
    """In-process pub/sub with per-group subscriber queues.

    Events are dicts: { seq, type, timestamp, group_id, payload }
    Subscribers receive events for their group via an asyncio.Queue.
    Maintains a ring buffer of recent events for resync support.
    """

    def __init__(self, history_size: int = 1000) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._seq_counter: int = 0
        self._event_history: dict[str, deque] = {}
        self._history_size = history_size

    def publish(self, group_id: str, event_type: str, payload: dict | None = None) -> int:
        """Publish an event to all subscribers of a group. Returns sequence number."""
        self._seq_counter += 1
        seq = self._seq_counter

        event = {
            "seq": seq,
            "type": event_type,
            "timestamp": time.time(),
            "group_id": group_id,
            "payload": payload or {},
        }

        # Store in ring buffer
        if group_id not in self._event_history:
            self._event_history[group_id] = deque(maxlen=self._history_size)
        self._event_history[group_id].append(event)

        queues = self._subscribers.get(group_id, [])
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "Event queue full for group %s, dropping event seq=%d",
                    group_id,
                    seq,
                )

        return seq

    def subscribe(self, group_id: str) -> asyncio.Queue:
        """Create and return a new subscription queue for a group."""
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(group_id, []).append(q)
        return q

    def unsubscribe(self, group_id: str, queue: asyncio.Queue) -> None:
        """Remove a subscription queue."""
        queues = self._subscribers.get(group_id, [])
        try:
            queues.remove(queue)
        except ValueError:
            pass
        if not queues and group_id in self._subscribers:
            del self._subscribers[group_id]

    def get_events_since(self, group_id: str, last_seq: int) -> tuple[list[dict], bool]:
        """Return (events, is_full_resync).

        is_full_resync=True if the gap is too large (client needs full refresh).
        """
        history = self._event_history.get(group_id, deque())
        if not history:
            return [], last_seq > 0  # full resync if client had events but we have none

        oldest_seq = history[0]["seq"]
        if last_seq < oldest_seq:
            # Gap too large — events were evicted from ring buffer
            return [], True

        missed = [e for e in history if e["seq"] > last_seq]
        return missed, False


def get_event_bus() -> EventBus:
    """Return the module-level singleton EventBus, creating it if needed."""
    global _EVENT_BUS
    if _EVENT_BUS is None:
        _EVENT_BUS = EventBus()
    return _EVENT_BUS
