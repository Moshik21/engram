"""Models for prospective memory (Wave 4): intentions and trigger matching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, Field


class Intention(BaseModel):
    """A stored intention with a semantic or entity-mention trigger condition."""

    id: str
    trigger_text: str
    action_text: str
    trigger_type: str = "semantic"  # "semantic" | "entity_mention"
    entity_name: str | None = None
    threshold: float = 0.7
    max_fires: int = 5
    fire_count: int = 0
    enabled: bool = True
    group_id: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime | None = None


class IntentionMeta(BaseModel):
    """Metadata stored in Entity.attributes for graph-embedded Intention entities."""

    trigger_text: str
    action_text: str
    trigger_type: str = "activation"  # "activation" | "entity_mention"
    activation_threshold: float = 0.5
    max_fires: int = 5
    fire_count: int = 0
    enabled: bool = True
    expires_at: str | None = None  # ISO 8601
    trigger_entity_ids: list[str] = Field(default_factory=list)
    cooldown_seconds: float = 300.0
    last_fired: str | None = None  # ISO 8601
    priority: str = "normal"  # "low" | "normal" | "high" | "critical"
    origin: str = "explicit"  # "explicit" | "inferred"
    context: str | None = None  # Rich background for agent at fire time
    see_also: list[str] | None = None  # Breadcrumb topic hints ("cliffhangers")


@dataclass
class IntentionMatch:
    """Result of a trigger match against an intention."""

    intention_id: str
    trigger_text: str
    action_text: str
    similarity: float
    matched_via: str  # "semantic" | "entity_mention" | "activation"
    context: str | None = None
    see_also: list[str] | None = None
