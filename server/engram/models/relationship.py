"""Relationship model for knowledge graph edges."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Relationship(BaseModel):
    """An edge in the knowledge graph connecting two entities with a predicate."""

    id: str
    source_id: str
    target_id: str
    predicate: str
    weight: float = 1.0
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 1.0
    source_episode: str | None = None
    group_id: str = "default"
