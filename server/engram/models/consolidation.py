"""Data models for memory consolidation cycles."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class ConsolidationStatus(str, Enum):
    """Lifecycle states for a consolidation cycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PhaseResult:
    """Outcome of a single consolidation phase."""

    phase: str
    status: str = "success"  # "success", "skipped", "error"
    items_processed: int = 0
    items_affected: int = 0
    duration_ms: float = 0.0
    error: str | None = None


@dataclass
class ConsolidationCycle:
    """A complete consolidation run with audit trail."""

    group_id: str
    trigger: str = "manual"  # "manual", "scheduled", "pressure", "shutdown"
    dry_run: bool = True
    status: str = "pending"
    phase_results: list[PhaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    total_duration_ms: float = 0.0
    id: str = field(default_factory=lambda: f"cyc_{uuid.uuid4().hex[:12]}")
    error: str | None = None


@dataclass
class MergeRecord:
    """Audit entry for an entity merge operation."""

    cycle_id: str
    group_id: str
    keep_id: str
    remove_id: str
    keep_name: str
    remove_name: str
    similarity: float
    decision_confidence: float | None = None
    decision_source: str | None = None
    decision_reason: str | None = None
    relationships_transferred: int = 0
    id: str = field(default_factory=lambda: f"mrg_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class IdentifierReviewRecord:
    """Audit entry for a suspicious identifier-like merge candidate kept separate."""

    cycle_id: str
    group_id: str
    entity_a_id: str
    entity_b_id: str
    entity_a_name: str
    entity_b_name: str
    entity_a_type: str
    entity_b_type: str
    raw_similarity: float
    adjusted_similarity: float | None = None
    decision_source: str | None = None
    decision_reason: str | None = None
    entity_a_regime: str | None = None
    entity_b_regime: str | None = None
    canonical_identifier_a: str | None = None
    canonical_identifier_b: str | None = None
    review_status: str = "quarantined"
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"idr_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class InferredEdge:
    """Audit entry for an inferred co-occurrence edge."""

    cycle_id: str
    group_id: str
    source_id: str
    target_id: str
    source_name: str
    target_name: str
    co_occurrence_count: int
    confidence: float
    predicate: str = "MENTIONED_WITH"
    infer_type: str = "co_occurrence"
    pmi_score: float | None = None
    llm_verdict: str | None = None
    escalation_verdict: str | None = None
    validation_score: float | None = None
    validation_signals: dict = field(default_factory=dict)
    materialization_action: str | None = None
    relationship_id: str | None = None
    id: str = field(default_factory=lambda: f"inf_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class PruneRecord:
    """Audit entry for a pruned entity."""

    cycle_id: str
    group_id: str
    entity_id: str
    entity_name: str
    entity_type: str
    reason: str  # "dead_entity", "no_access", etc.
    id: str = field(default_factory=lambda: f"prn_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriageRecord:
    """Audit entry for a triaged episode."""

    cycle_id: str
    group_id: str
    episode_id: str
    score: float
    decision: str  # "extract" | "skip"
    score_breakdown: dict = field(default_factory=dict)
    llm_reason: str | None = None
    llm_tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: f"tri_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleContext:
    """Mutable shared state passed through all phases in a single cycle."""

    trigger: str = "manual"
    affected_entity_ids: set[str] = field(default_factory=set)
    merge_survivor_ids: set[str] = field(default_factory=set)
    inferred_edge_entity_ids: set[str] = field(default_factory=set)
    pruned_entity_ids: set[str] = field(default_factory=set)
    replay_new_entity_ids: set[str] = field(default_factory=set)
    dream_seed_ids: set[str] = field(default_factory=set)
    dream_association_ids: set[str] = field(default_factory=set)
    triage_promoted_ids: set[str] = field(default_factory=set)
    matured_entity_ids: set[str] = field(default_factory=set)
    transitioned_episode_ids: set[str] = field(default_factory=set)
    schema_entity_ids: set[str] = field(default_factory=set)
    microglia_demoted_edge_ids: set[str] = field(default_factory=set)
    microglia_repaired_entity_ids: set[str] = field(default_factory=set)
    maturity_feature_cache: dict[str, dict] = field(default_factory=dict)
    decision_traces: list[DecisionTrace] = field(default_factory=list)
    decision_outcome_labels: list[DecisionOutcomeLabel] = field(default_factory=list)

    def add_decision_trace(self, trace: DecisionTrace) -> None:
        self.decision_traces.append(trace)

    def add_decision_outcome_label(self, label: DecisionOutcomeLabel) -> None:
        self.decision_outcome_labels.append(label)


@dataclass
class ReindexRecord:
    """Audit entry for a re-indexed entity."""

    cycle_id: str
    group_id: str
    entity_id: str
    entity_name: str
    source_phase: str  # "merge" | "infer"
    id: str = field(default_factory=lambda: f"rix_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReplayRecord:
    """Audit entry for a replayed episode."""

    cycle_id: str
    group_id: str
    episode_id: str
    new_entities_found: int = 0
    new_relationships_found: int = 0
    entities_updated: int = 0
    skipped_reason: str | None = None  # "extraction_failed" | "no_new_info" | None
    id: str = field(default_factory=lambda: f"rpl_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class DreamRecord:
    """Audit entry for a dream-boosted edge."""

    cycle_id: str
    group_id: str
    source_entity_id: str
    target_entity_id: str
    weight_delta: float
    seed_entity_id: str = ""
    id: str = field(default_factory=lambda: f"drm_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraphEmbedRecord:
    """Audit entry for a graph embedding training run."""

    cycle_id: str
    group_id: str
    method: str
    entities_trained: int
    dimensions: int
    training_duration_ms: float
    full_retrain: bool
    id: str = field(default_factory=lambda: f"gemb_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class DreamAssociationRecord:
    """Audit entry for a dream-discovered cross-domain association."""

    cycle_id: str
    group_id: str
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    source_domain: str
    target_domain: str
    surprise_score: float
    embedding_similarity: float
    structural_proximity: float
    relationship_id: str | None = None
    id: str = field(default_factory=lambda: f"dra_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class MaturationRecord:
    """Audit entry for a matured entity."""

    cycle_id: str
    group_id: str
    entity_id: str
    entity_name: str
    old_tier: str
    new_tier: str
    maturity_score: float
    source_diversity: int
    temporal_span_days: float
    relationship_richness: int
    access_regularity: float
    id: str = field(default_factory=lambda: f"mat_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class SchemaRecord:
    """Audit entry for a discovered or reinforced schema."""

    cycle_id: str
    group_id: str
    schema_entity_id: str
    schema_name: str
    instance_count: int
    predicate_count: int
    action: str  # "created" | "reinforced"
    id: str = field(default_factory=lambda: f"sch_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class SemanticTransitionRecord:
    """Audit entry for an episode tier transition."""

    cycle_id: str
    group_id: str
    episode_id: str
    old_tier: str
    new_tier: str
    entity_coverage: float
    consolidation_cycles: int
    id: str = field(default_factory=lambda: f"sem_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class EvidenceAdjudicationRecord:
    """Audit record for evidence adjudication phase."""

    cycle_id: str
    group_id: str
    evidence_id: str
    # "approved"|"materialized"|"materialization_failed"|"deferred"|"expired"|"corroborated"
    action: str
    new_confidence: float
    reason: str
    id: str = field(default_factory=lambda: f"evadj_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionTrace:
    """Standard trace for a scored or constrained phase decision."""

    cycle_id: str
    group_id: str
    phase: str
    candidate_type: str
    candidate_id: str
    decision: str
    decision_source: str
    confidence: float | None = None
    threshold_band: str | None = None
    features: dict = field(default_factory=dict)
    constraints_hit: list[str] = field(default_factory=list)
    policy_version: str = "v1"
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"dtr_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionOutcomeLabel:
    """Outcome label attached to a previously recorded decision trace."""

    cycle_id: str
    group_id: str
    phase: str
    decision_trace_id: str
    outcome_type: str
    label: str
    value: float | None = None
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"dol_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class DistillationExample:
    """Training-ready example derived from traces, outcomes, or oracle decisions."""

    cycle_id: str
    group_id: str
    phase: str
    candidate_type: str
    candidate_id: str
    decision_trace_id: str
    teacher_label: str
    teacher_source: str
    student_decision: str
    student_confidence: float | None = None
    threshold_band: str | None = None
    features: dict = field(default_factory=dict)
    correct: bool | None = None
    metadata: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"dex_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class CalibrationSnapshot:
    """Rolling calibration summary for a consolidation phase."""

    cycle_id: str
    group_id: str
    phase: str
    window_cycles: int
    total_traces: int
    labeled_examples: int
    oracle_examples: int
    abstain_count: int
    accuracy: float | None = None
    mean_confidence: float | None = None
    expected_calibration_error: float | None = None
    summary: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"cal_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class RelationshipApplyResult:
    """Structured result for the shared relationship apply path."""

    source_id: str | None = None
    target_id: str | None = None
    predicate: str | None = None
    polarity: str = "positive"
    confidence: float | None = None
    weight: float | None = None
    action: str = "skipped"
    created: bool = False
    constraints_hit: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class MicrogliaRecord:
    """Audit entry for a microglia phase action."""

    cycle_id: str
    group_id: str
    target_type: str        # "edge" | "entity_summary"
    target_id: str
    action: str             # "tagged" | "confirmed" | "demoted" | "cleared" | "repaired"
    tag_type: str           # "c1q_domain" | "c1q_embedding" | "c3_summary" | "c4_orphan"
    score: float
    detail: str             # Human-readable reason
    id: str = field(default_factory=lambda: f"mcg_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)
