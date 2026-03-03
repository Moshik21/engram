"""Entity model for knowledge graph nodes."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A node in the knowledge graph representing a person, concept, project, etc."""

    id: str
    name: str
    entity_type: str
    summary: str | None = None
    attributes: dict | None = None
    group_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: datetime | None = None
    activation_current: float = 0.0
    access_count: int = 0
    last_accessed: datetime | None = None
    pii_detected: bool = False
    pii_categories: list[str] | None = None
