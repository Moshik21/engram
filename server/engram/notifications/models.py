"""Notification data model for proactive memory surfacing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

NOTIFICATION_TYPES = frozenset(
    {
        "temporal_intention",
        "dream_association",
        "schema_discovery",
        "entity_maturation",
        "entity_merge",
        "activation_anomaly",
    }
)

PRIORITY_LEVELS = frozenset({"low", "normal", "high"})


@dataclass
class MemoryNotification:
    group_id: str
    notification_type: str
    priority: str
    title: str
    body: str
    entity_ids: list[str]
    metadata: dict[str, Any]
    created_at: float
    source_cycle_id: str | None = None
    id: str = field(default_factory=lambda: f"ntf_{uuid.uuid4().hex[:12]}")
    dismissed_at: float | None = None
    surfaced_count: int = 0

    def __post_init__(self) -> None:
        if self.notification_type not in NOTIFICATION_TYPES:
            raise ValueError(f"Unknown notification type: {self.notification_type}")
        if self.priority not in PRIORITY_LEVELS:
            raise ValueError(f"Unknown priority: {self.priority}")


def notification_to_dict(n: MemoryNotification) -> dict[str, Any]:
    """Serialize a notification to a plain dict."""
    return {
        "id": n.id,
        "group_id": n.group_id,
        "notification_type": n.notification_type,
        "priority": n.priority,
        "title": n.title,
        "body": n.body,
        "entity_ids": n.entity_ids,
        "metadata": n.metadata,
        "source_cycle_id": n.source_cycle_id,
        "created_at": n.created_at,
        "dismissed_at": n.dismissed_at,
        "surfaced_count": n.surfaced_count,
    }
