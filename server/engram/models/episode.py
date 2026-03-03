"""Episode model for raw memory ingestion."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EpisodeStatus(str, Enum):
    # Granular pipeline states
    QUEUED = "queued"
    EXTRACTING = "extracting"
    RESOLVING = "resolving"
    WRITING = "writing"
    EMBEDDING = "embedding"
    ACTIVATING = "activating"
    COMPLETED = "completed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"

    # Legacy aliases (kept for backwards compat)
    PENDING = "pending"
    PROCESSING = "processing"
    FAILED = "failed"


class Episode(BaseModel):
    """A raw memory episode — the input text from a conversation or event."""

    id: str
    content: str
    source: str | None = None
    status: EpisodeStatus = EpisodeStatus.PENDING
    group_id: str = "default"
    session_id: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0
    processing_duration_ms: int | None = None
