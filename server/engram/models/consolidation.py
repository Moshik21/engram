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
    relationships_transferred: int = 0
    id: str = field(default_factory=lambda: f"mrg_{uuid.uuid4().hex[:12]}")
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
    infer_type: str = "co_occurrence"
    pmi_score: float | None = None
    llm_verdict: str | None = None
    escalation_verdict: str | None = None
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
