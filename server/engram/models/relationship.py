"""Relationship model for knowledge graph edges."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from engram.utils.dates import utc_now


class Relationship(BaseModel):
    """An edge in the knowledge graph connecting two entities with a predicate."""

    id: str
    source_id: str
    target_id: str
    predicate: str
    weight: float = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    confidence: float = 1.0
    polarity: str = "positive"  # positive | negative | uncertain
    source_episode: str | None = None
    group_id: str = "default"
