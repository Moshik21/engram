"""Recall planning and telemetry models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class MemoryNeed:
    """Decision output from the memory-need analyzer."""

    need_type: str
    should_recall: bool
    confidence: float
    reasons: list[str] = field(default_factory=list)
    query_hint: str | None = None
    urgency: float = 0.0
    packet_budget: int = 1
    entity_budget: int = 3
    # Signal metadata (Phase 0 scaffolding)
    signal_scores: dict[str, float] | None = None
    trigger_family: str | None = None
    trigger_kind: str | None = None
    detected_entities: list[str] | None = None
    detected_referents: list[str] | None = None
    resonance_score: float = 0.0
    decision_path: str | None = None
    thresholds: dict[str, float] | None = None
    analyzer_latency_ms: float = 0.0
    probe_triggered: bool = False
    probe_latency_ms: float = 0.0
    graph_override_used: bool = False

    def to_payload(
        self,
        *,
        source: str,
        mode: str,
        turn_preview: str,
    ) -> dict:
        """Serialize for telemetry events."""
        payload: dict = {
            "needType": self.need_type,
            "shouldRecall": self.should_recall,
            "confidence": round(self.confidence, 4),
            "urgency": round(self.urgency, 4),
            "packetBudget": self.packet_budget,
            "entityBudget": self.entity_budget,
            "queryHint": self.query_hint,
            "reasons": self.reasons,
            "source": source,
            "mode": mode,
            "turnPreview": turn_preview,
        }
        if self.signal_scores:
            payload["signalScores"] = {
                k: round(v, 4) for k, v in self.signal_scores.items()
            }
        if self.trigger_family:
            payload["triggerFamily"] = self.trigger_family
        if self.trigger_kind:
            payload["triggerKind"] = self.trigger_kind
        if self.detected_entities:
            payload["detectedEntities"] = self.detected_entities
        if self.detected_referents:
            payload["detectedReferents"] = self.detected_referents
        if self.resonance_score > 0:
            payload["resonanceScore"] = round(self.resonance_score, 4)
        if self.decision_path:
            payload["decisionPath"] = self.decision_path
        if self.thresholds:
            payload["thresholds"] = {
                key: round(value, 4) for key, value in self.thresholds.items()
            }
        if self.analyzer_latency_ms > 0:
            payload["analyzerLatencyMs"] = round(self.analyzer_latency_ms, 4)
        if self.probe_triggered:
            payload["probeTriggered"] = True
        if self.probe_latency_ms > 0:
            payload["probeLatencyMs"] = round(self.probe_latency_ms, 4)
        if self.graph_override_used:
            payload["graphOverrideUsed"] = True
        return payload


@dataclass
class RecallIntent:
    """A single planner-generated retrieval intent."""

    intent_type: str
    query_text: str
    weight: float
    candidate_budget: int
    packet_types: list[str] = field(default_factory=list)


@dataclass
class RecallPlan:
    """A bounded bundle of intents for a recall request."""

    query: str
    mode: str
    intents: list[RecallIntent] = field(default_factory=list)
    seed_entity_ids: list[str] = field(default_factory=list)


@dataclass
class RecallTrace:
    """Execution trace showing which intents supported each candidate."""

    plan: RecallPlan
    merged_candidates: list[tuple[str, float]] = field(default_factory=list)
    support_scores: dict[str, float] = field(default_factory=dict)
    intent_types: dict[str, list[str]] = field(default_factory=dict)
    support_details: dict[str, list[dict]] = field(default_factory=dict)


@dataclass
class MemoryPacket:
    """A compact, action-oriented memory unit."""

    packet_type: str
    title: str
    summary: str
    why_now: str
    confidence: float
    entity_ids: list[str] = field(default_factory=list)
    relationship_ids: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    evidence_lines: list[str] = field(default_factory=list)
    provenance: list[str] = field(default_factory=list)
    supporting_intents: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize for APIs and MCP."""
        return {
            "packet_type": self.packet_type,
            "title": self.title,
            "summary": self.summary,
            "why_now": self.why_now,
            "confidence": round(self.confidence, 4),
            "entity_ids": self.entity_ids,
            "relationship_ids": self.relationship_ids,
            "episode_ids": self.episode_ids,
            "evidence_lines": self.evidence_lines,
            "provenance": self.provenance,
            "supporting_intents": self.supporting_intents,
        }


@dataclass
class MemoryInteractionEvent:
    """Structured interaction event for recall telemetry and feedback."""

    group_id: str
    entity_id: str
    interaction_type: str
    source: str
    query: str
    entity_name: str | None = None
    entity_type: str | None = None
    score: float | None = None
    recorded_access: bool = False
    id: str = field(default_factory=lambda: f"ri_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)

    def to_payload(self) -> dict:
        """Serialize for recall.interaction events."""
        payload = {
            "id": self.id,
            "timestamp": self.timestamp,
            "entityId": self.entity_id,
            "interactionType": self.interaction_type,
            "source": self.source,
            "query": self.query,
            "recordedAccess": self.recorded_access,
        }
        if self.entity_name is not None:
            payload["name"] = self.entity_name
        if self.entity_type is not None:
            payload["entityType"] = self.entity_type
        if self.score is not None:
            payload["score"] = round(self.score, 4)
        return payload
