"""Cue-oriented latent memory representation for an episode."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from engram.models.episode import EpisodeProjectionState
from engram.utils.dates import utc_now


class EpisodeCue(BaseModel):
    """Deterministic, retrieval-oriented representation of an episode."""

    episode_id: str
    group_id: str = "default"
    cue_version: int = 1
    discourse_class: str = "world"
    projection_state: EpisodeProjectionState = EpisodeProjectionState.CUED
    cue_score: float = 0.0
    salience_score: float = 0.0
    projection_priority: float = 0.0
    route_reason: str | None = None
    cue_text: str = ""
    entity_mentions: list[dict] = Field(default_factory=list)
    temporal_markers: list[str] = Field(default_factory=list)
    quote_spans: list[str] = Field(default_factory=list)
    contradiction_keys: list[str] = Field(default_factory=list)
    first_spans: list[str] = Field(default_factory=list)
    hit_count: int = 0
    surfaced_count: int = 0
    selected_count: int = 0
    used_count: int = 0
    near_miss_count: int = 0
    # M5.1 (RF goal): ranking-side episode usage via the cue substrate.
    # usage_used_count is the TIER-WEIGHTED float sum (echo-guarded citation
    # scan, w_used etc. from ActivationConfig.usage_tier_weights); the legacy
    # int used_count above keeps its hygiene/telemetry semantics untouched.
    # Read only under usage_ranking_enabled (u_episode = f*r', compute_u_values).
    usage_used_count: float = 0.0
    usage_last_used_at: datetime | None = None
    policy_score: float = 0.0
    projection_attempts: int = 0
    last_hit_at: datetime | None = None
    last_feedback_at: datetime | None = None
    last_projected_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime | None = None
