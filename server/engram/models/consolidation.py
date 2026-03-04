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
    id: str = field(default_factory=lambda: f"tri_{uuid.uuid4().hex[:12]}")
    timestamp: float = field(default_factory=time.time)


@dataclass
class CycleContext:
    """Mutable shared state passed through all phases in a single cycle."""

    affected_entity_ids: set[str] = field(default_factory=set)
    merge_survivor_ids: set[str] = field(default_factory=set)
    inferred_edge_entity_ids: set[str] = field(default_factory=set)
    pruned_entity_ids: set[str] = field(default_factory=set)
    replay_new_entity_ids: set[str] = field(default_factory=set)
    dream_seed_ids: set[str] = field(default_factory=set)
    triage_promoted_ids: set[str] = field(default_factory=set)


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
