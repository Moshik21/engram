"""Episode model for raw memory ingestion."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from engram.utils.dates import utc_now


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


class EpisodeProjectionState(str, Enum):
    QUEUED = "queued"
    CUED = "cued"
    CUE_ONLY = "cue_only"
    SCHEDULED = "scheduled"
    PROJECTING = "projecting"
    PROJECTED = "projected"
    MERGED = "merged"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class Attachment(BaseModel):
    """A multimodal attachment on an episode (image, audio, video, PDF)."""

    mime_type: str  # "image/png", "audio/mp3", "video/mp4", "application/pdf"
    data_url: str  # base64 data URI or file path
    description: str = ""  # optional text description of the content


class Episode(BaseModel):
    """A raw memory episode — the input text from a conversation or event."""

    id: str
    content: str
    source: str | None = None
    status: EpisodeStatus = EpisodeStatus.PENDING
    group_id: str = "default"
    session_id: str | None = None
    conversation_date: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None
    error: str | None = None
    retry_count: int = 0
    processing_duration_ms: int | None = None
    encoding_context: str | None = None
    memory_tier: str = "episodic"  # "episodic" | "transitional" | "semantic"
    consolidation_cycles: int = 0
    entity_coverage: float = 0.0
    projection_state: EpisodeProjectionState = EpisodeProjectionState.QUEUED
    last_projection_reason: str | None = None
    last_projected_at: datetime | None = None
    attachments: list[Attachment] = Field(default_factory=list)
