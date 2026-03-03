"""Implicit feedback event model for learn-to-rank infrastructure."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class FeedbackEvent:
    """A single implicit feedback signal."""

    entity_id: str
    event_type: str  # "returned", "re_accessed", "mentioned_in_remember", "ignored"
    query: str
    group_id: str
    id: str = field(default_factory=lambda: f"fb_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class FeedbackStats:
    """Aggregated feedback statistics for an entity."""

    entity_id: str
    total_events: int = 0
    returned_count: int = 0
    re_accessed_count: int = 0
    mentioned_count: int = 0
    ignored_count: int = 0
