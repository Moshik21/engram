"""Evidence-based extraction data model for v2 extractor pipeline."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

from engram.utils.dates import utc_now


@dataclass
class EvidenceCandidate:
    """A single piece of evidence extracted from an episode."""

    evidence_id: str = field(default_factory=lambda: f"evi_{uuid.uuid4().hex[:12]}")
    episode_id: str = ""
    group_id: str = "default"
    fact_class: str = ""  # "entity" | "relationship" | "attribute" | "temporal"
    confidence: float = 0.0
    # "narrow_extractor" | "client_proposal" | "llm_extraction" | "consolidation"
    source_type: str = ""
    extractor_name: str = ""
    payload: dict = field(default_factory=dict)
    source_span: str | None = None
    corroborating_signals: list[str] = field(default_factory=list)
    ambiguity_tags: list[str] = field(default_factory=list)
    ambiguity_score: float = 0.0
    adjudication_request_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)


@dataclass
class EvidenceBundle:
    """Collection of evidence from an extraction pipeline run."""

    episode_id: str = ""
    group_id: str = "default"
    candidates: list[EvidenceCandidate] = field(default_factory=list)
    extractor_stats: dict = field(default_factory=dict)
    total_ms: float = 0.0


@dataclass
class CommitDecision:
    """Decision about what to do with a piece of evidence."""

    evidence_id: str = ""
    action: str = ""  # "commit" | "defer" | "reject"
    reason: str = ""
    effective_confidence: float = 0.0
    committed_id: str | None = None  # entity_id or relationship_id if committed


def evidence_candidate_from_dict(data: dict) -> EvidenceCandidate:
    """Hydrate a stored evidence row back into an EvidenceCandidate."""
    created_at = data.get("created_at")
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at)
        except ValueError:
            created_at = utc_now()
    elif not isinstance(created_at, datetime):
        created_at = utc_now()
    return EvidenceCandidate(
        evidence_id=data.get("evidence_id", ""),
        episode_id=data.get("episode_id", ""),
        group_id=data.get("group_id", "default"),
        fact_class=data.get("fact_class", ""),
        confidence=float(data.get("confidence", 0.0) or 0.0),
        source_type=data.get("source_type", ""),
        extractor_name=data.get("extractor_name", ""),
        payload=dict(data.get("payload", {})),
        source_span=data.get("source_span"),
        corroborating_signals=list(data.get("corroborating_signals", [])),
        ambiguity_tags=list(data.get("ambiguity_tags", [])),
        ambiguity_score=float(data.get("ambiguity_score", 0.0) or 0.0),
        adjudication_request_id=data.get("adjudication_request_id"),
        created_at=created_at,
    )
