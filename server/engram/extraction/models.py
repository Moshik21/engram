"""Typed intermediates for projection planning, extraction, and apply."""

from __future__ import annotations

from dataclasses import dataclass, field

from engram.models.consolidation import RelationshipApplyResult


@dataclass
class ProjectedSpan:
    """A scored text span selected for extraction."""

    span_id: str
    start_char: int
    end_char: int
    text: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)


@dataclass
class ProjectionPlan:
    """Deterministic plan describing what text to send to the extractor."""

    episode_id: str
    strategy: str
    spans: list[ProjectedSpan]
    selected_text: str
    selected_chars: int
    total_chars: int
    was_truncated: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class EntityCandidate:
    """Typed entity candidate with span provenance."""

    name: str
    entity_type: str
    summary: str | None = None
    attributes: dict | None = None
    pii_detected: bool = False
    pii_categories: list[str] | None = None
    epistemic_mode: str | None = None
    source_span_ids: list[str] = field(default_factory=list)
    raw_payload: dict = field(default_factory=dict)


@dataclass
class ClaimCandidate:
    """Typed relationship / claim candidate with span provenance."""

    subject_text: str
    predicate: str
    object_text: str | None = None
    object_value: dict | None = None
    polarity: str = "positive"
    temporal_hint: str | None = None
    confidence: float = 1.0
    source_span_ids: list[str] = field(default_factory=list)
    raw_payload: dict = field(default_factory=dict)


@dataclass
class ProjectionBundle:
    """Extractor output enriched with planned spans and typed candidates."""

    episode_id: str
    plan: ProjectionPlan
    entities: list[EntityCandidate]
    claims: list[ClaimCandidate]
    warnings: list[str] = field(default_factory=list)
    extractor_status: str = "ok"
    extractor_error: str | None = None
    retryable: bool = False

    @property
    def is_error(self) -> bool:
        return self.extractor_status in {"parse_error", "api_error", "truncated"}


@dataclass
class ApplyOutcome:
    """Summary of bundle application against graph storage."""

    entity_map: dict[str, str] = field(default_factory=dict)
    new_entity_names: list[str] = field(default_factory=list)
    meta_entity_names: set[str] = field(default_factory=set)
    relationship_results: list[RelationshipApplyResult] = field(default_factory=list)
