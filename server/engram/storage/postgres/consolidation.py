"""PostgreSQL-backed consolidation cycle audit store."""

from __future__ import annotations

import json
import time

import asyncpg

from engram.models.consolidation import (
    CalibrationSnapshot,
    ConsolidationCycle,
    DecisionOutcomeLabel,
    DecisionTrace,
    DistillationExample,
    DreamAssociationRecord,
    DreamRecord,
    IdentifierReviewRecord,
    InferredEdge,
    MergeRecord,
    PhaseResult,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    TriageRecord,
)


class PostgresConsolidationStore:
    """Stores consolidation cycle history and audit records in PostgreSQL."""

    def __init__(
        self,
        dsn: str,
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ) -> None:
        self._dsn = dsn
        self._min_pool_size = min_pool_size
        self._max_pool_size = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        """Create the connection pool and all tables."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._min_pool_size,
            max_size=self._max_pool_size,
        )

        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_cycles (
                    id TEXT PRIMARY KEY,
                    group_id TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    dry_run BOOLEAN NOT NULL DEFAULT TRUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    phase_results JSONB,
                    started_at DOUBLE PRECISION NOT NULL,
                    completed_at DOUBLE PRECISION,
                    total_duration_ms DOUBLE PRECISION DEFAULT 0.0,
                    error TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_merges (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    keep_id TEXT NOT NULL,
                    remove_id TEXT NOT NULL,
                    keep_name TEXT NOT NULL,
                    remove_name TEXT NOT NULL,
                    similarity DOUBLE PRECISION NOT NULL,
                    decision_confidence DOUBLE PRECISION,
                    decision_source TEXT,
                    decision_reason TEXT,
                    relationships_transferred INTEGER DEFAULT 0,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute(
                "ALTER TABLE consolidation_merges "
                "ADD COLUMN IF NOT EXISTS decision_confidence DOUBLE PRECISION"
            )
            await conn.execute(
                "ALTER TABLE consolidation_merges "
                "ADD COLUMN IF NOT EXISTS decision_source TEXT"
            )
            await conn.execute(
                "ALTER TABLE consolidation_merges "
                "ADD COLUMN IF NOT EXISTS decision_reason TEXT"
            )
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_inferred_edges (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    co_occurrence_count INTEGER NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    infer_type TEXT DEFAULT 'co_occurrence',
                    pmi_score DOUBLE PRECISION,
                    llm_verdict TEXT,
                    relationship_id TEXT,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_identifier_reviews (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    entity_a_id TEXT NOT NULL,
                    entity_b_id TEXT NOT NULL,
                    entity_a_name TEXT NOT NULL,
                    entity_b_name TEXT NOT NULL,
                    entity_a_type TEXT NOT NULL,
                    entity_b_type TEXT NOT NULL,
                    raw_similarity DOUBLE PRECISION NOT NULL,
                    adjusted_similarity DOUBLE PRECISION,
                    decision_source TEXT,
                    decision_reason TEXT,
                    entity_a_regime TEXT,
                    entity_b_regime TEXT,
                    canonical_identifier_a TEXT,
                    canonical_identifier_b TEXT,
                    review_status TEXT NOT NULL DEFAULT 'quarantined',
                    metadata JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_prunes (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_reindexes (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL,
                    source_phase TEXT NOT NULL,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_replays (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    new_entities_found INTEGER DEFAULT 0,
                    new_relationships_found INTEGER DEFAULT 0,
                    entities_updated INTEGER DEFAULT 0,
                    skipped_reason TEXT,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_dreams (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    weight_delta DOUBLE PRECISION NOT NULL,
                    seed_entity_id TEXT,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_triage (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    score DOUBLE PRECISION NOT NULL,
                    decision TEXT NOT NULL,
                    score_breakdown JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_dream_associations (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    source_entity_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    source_entity_name TEXT NOT NULL,
                    target_entity_name TEXT NOT NULL,
                    source_domain TEXT NOT NULL,
                    target_domain TEXT NOT NULL,
                    surprise_score DOUBLE PRECISION NOT NULL,
                    embedding_similarity DOUBLE PRECISION NOT NULL,
                    structural_proximity DOUBLE PRECISION NOT NULL,
                    relationship_id TEXT,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_decision_traces (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    decision_source TEXT NOT NULL,
                    confidence DOUBLE PRECISION,
                    threshold_band TEXT,
                    features JSONB,
                    constraints_json JSONB,
                    policy_version TEXT NOT NULL,
                    metadata JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_decision_outcomes (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    decision_trace_id TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    value DOUBLE PRECISION,
                    metadata JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_distillation_examples (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    candidate_type TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    decision_trace_id TEXT NOT NULL,
                    teacher_label TEXT NOT NULL,
                    teacher_source TEXT NOT NULL,
                    student_decision TEXT NOT NULL,
                    student_confidence DOUBLE PRECISION,
                    threshold_band TEXT,
                    features JSONB,
                    correct BOOLEAN,
                    metadata JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidation_calibration_snapshots (
                    id TEXT PRIMARY KEY,
                    cycle_id TEXT NOT NULL,
                    group_id TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    window_cycles INTEGER NOT NULL,
                    total_traces INTEGER NOT NULL,
                    labeled_examples INTEGER NOT NULL,
                    oracle_examples INTEGER NOT NULL,
                    abstain_count INTEGER NOT NULL,
                    accuracy DOUBLE PRECISION,
                    mean_confidence DOUBLE PRECISION,
                    expected_calibration_error DOUBLE PRECISION,
                    summary JSONB,
                    timestamp DOUBLE PRECISION NOT NULL
                )
            """)

            # Indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_cycles_group "
                "ON consolidation_cycles(group_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_merges_cycle "
                "ON consolidation_merges(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_identifier_reviews_cycle "
                "ON consolidation_identifier_reviews(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_edges_cycle "
                "ON consolidation_inferred_edges(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_prunes_cycle "
                "ON consolidation_prunes(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_reindexes_cycle "
                "ON consolidation_reindexes(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_replays_cycle "
                "ON consolidation_replays(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_dreams_cycle "
                "ON consolidation_dreams(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_triage_cycle "
                "ON consolidation_triage(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_dream_assoc_cycle "
                "ON consolidation_dream_associations(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_decision_traces_cycle "
                "ON consolidation_decision_traces(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_decision_outcomes_cycle "
                "ON consolidation_decision_outcomes(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_distill_cycle "
                "ON consolidation_distillation_examples(cycle_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_consol_calibration_cycle "
                "ON consolidation_calibration_snapshots(cycle_id)"
            )

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresConsolidationStore not initialized.")
        return self._pool

    # ── Cycle CRUD ──────────────────────────────────────────────────────

    async def save_cycle(self, cycle: ConsolidationCycle) -> None:
        """Insert a new consolidation cycle."""
        phase_json = json.dumps(
            [
                {
                    "phase": pr.phase,
                    "status": pr.status,
                    "items_processed": pr.items_processed,
                    "items_affected": pr.items_affected,
                    "duration_ms": pr.duration_ms,
                    "error": pr.error,
                }
                for pr in cycle.phase_results
            ]
        )
        await self.pool.execute(
            "INSERT INTO consolidation_cycles "
            "(id, group_id, trigger, dry_run, status, phase_results, "
            "started_at, completed_at, total_duration_ms, error) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10) "
            "ON CONFLICT (id) DO NOTHING",
            cycle.id,
            cycle.group_id,
            cycle.trigger,
            cycle.dry_run,
            cycle.status,
            phase_json,
            cycle.started_at,
            cycle.completed_at,
            cycle.total_duration_ms,
            cycle.error,
        )

    async def update_cycle(self, cycle: ConsolidationCycle) -> None:
        """Update an existing consolidation cycle."""
        phase_json = json.dumps(
            [
                {
                    "phase": pr.phase,
                    "status": pr.status,
                    "items_processed": pr.items_processed,
                    "items_affected": pr.items_affected,
                    "duration_ms": pr.duration_ms,
                    "error": pr.error,
                }
                for pr in cycle.phase_results
            ]
        )
        await self.pool.execute(
            "UPDATE consolidation_cycles "
            "SET status = $1, phase_results = $2::jsonb, completed_at = $3, "
            "total_duration_ms = $4, error = $5 "
            "WHERE id = $6 AND group_id = $7",
            cycle.status,
            phase_json,
            cycle.completed_at,
            cycle.total_duration_ms,
            cycle.error,
            cycle.id,
            cycle.group_id,
        )

    async def get_cycle(self, cycle_id: str, group_id: str) -> ConsolidationCycle | None:
        """Fetch a single cycle by ID."""
        row = await self.pool.fetchrow(
            "SELECT * FROM consolidation_cycles WHERE id = $1 AND group_id = $2",
            cycle_id,
            group_id,
        )
        if not row:
            return None
        return self._row_to_cycle(row)

    async def get_recent_cycles(
        self,
        group_id: str,
        limit: int = 10,
    ) -> list[ConsolidationCycle]:
        """Fetch recent cycles for a group, newest first."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_cycles WHERE group_id = $1 "
            "ORDER BY started_at DESC LIMIT $2",
            group_id,
            limit,
        )
        return [self._row_to_cycle(r) for r in rows]

    # ── Merge records ───────────────────────────────────────────────────

    async def save_merge_record(self, record: MergeRecord) -> None:
        """Insert a merge audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_merges "
            "(id, cycle_id, group_id, keep_id, remove_id, keep_name, "
            "remove_name, similarity, decision_confidence, decision_source, "
            "decision_reason, relationships_transferred, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.keep_id,
            record.remove_id,
            record.keep_name,
            record.remove_name,
            record.similarity,
            record.decision_confidence,
            record.decision_source,
            record.decision_reason,
            record.relationships_transferred,
            record.timestamp,
        )

    async def save_identifier_review_record(self, record: IdentifierReviewRecord) -> None:
        """Insert a quarantined identifier review record."""
        await self.pool.execute(
            "INSERT INTO consolidation_identifier_reviews "
            "(id, cycle_id, group_id, entity_a_id, entity_b_id, entity_a_name, "
            "entity_b_name, entity_a_type, entity_b_type, raw_similarity, "
            "adjusted_similarity, decision_source, decision_reason, entity_a_regime, "
            "entity_b_regime, canonical_identifier_a, canonical_identifier_b, "
            "review_status, metadata, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, "
            "$14, $15, $16, $17, $18, $19, $20) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.entity_a_id,
            record.entity_b_id,
            record.entity_a_name,
            record.entity_b_name,
            record.entity_a_type,
            record.entity_b_type,
            record.raw_similarity,
            record.adjusted_similarity,
            record.decision_source,
            record.decision_reason,
            record.entity_a_regime,
            record.entity_b_regime,
            record.canonical_identifier_a,
            record.canonical_identifier_b,
            record.review_status,
            json.dumps(record.metadata or {}),
            record.timestamp,
        )

    async def get_merge_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[MergeRecord]:
        """Fetch merge records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_merges "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            MergeRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                keep_id=r["keep_id"],
                remove_id=r["remove_id"],
                keep_name=r["keep_name"],
                remove_name=r["remove_name"],
                similarity=r["similarity"],
                decision_confidence=(
                    r["decision_confidence"] if "decision_confidence" in r else None
                ),
                decision_source=r["decision_source"] if "decision_source" in r else None,
                decision_reason=r["decision_reason"] if "decision_reason" in r else None,
                relationships_transferred=r["relationships_transferred"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def get_identifier_review_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[IdentifierReviewRecord]:
        """Fetch quarantined identifier review records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_identifier_reviews "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            IdentifierReviewRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                entity_a_id=r["entity_a_id"],
                entity_b_id=r["entity_b_id"],
                entity_a_name=r["entity_a_name"],
                entity_b_name=r["entity_b_name"],
                entity_a_type=r["entity_a_type"],
                entity_b_type=r["entity_b_type"],
                raw_similarity=r["raw_similarity"],
                adjusted_similarity=r["adjusted_similarity"],
                decision_source=r["decision_source"],
                decision_reason=r["decision_reason"],
                entity_a_regime=r["entity_a_regime"],
                entity_b_regime=r["entity_b_regime"],
                canonical_identifier_a=r["canonical_identifier_a"],
                canonical_identifier_b=r["canonical_identifier_b"],
                review_status=r["review_status"],
                metadata=(
                    json.loads(r["metadata"]) if isinstance(r["metadata"], str) else r["metadata"]
                ) or {},
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Inferred edge records ───────────────────────────────────────────

    async def save_inferred_edge(self, edge: InferredEdge) -> None:
        """Insert an inferred edge audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_inferred_edges "
            "(id, cycle_id, group_id, source_id, target_id, source_name, "
            "target_name, co_occurrence_count, confidence, infer_type, "
            "pmi_score, llm_verdict, relationship_id, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
            "ON CONFLICT (id) DO NOTHING",
            edge.id,
            edge.cycle_id,
            edge.group_id,
            edge.source_id,
            edge.target_id,
            edge.source_name,
            edge.target_name,
            edge.co_occurrence_count,
            edge.confidence,
            edge.infer_type,
            edge.pmi_score,
            edge.llm_verdict,
            edge.relationship_id,
            edge.timestamp,
        )

    async def get_inferred_edges(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[InferredEdge]:
        """Fetch inferred edge records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_inferred_edges "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            InferredEdge(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                source_id=r["source_id"],
                target_id=r["target_id"],
                source_name=r["source_name"],
                target_name=r["target_name"],
                co_occurrence_count=r["co_occurrence_count"],
                confidence=r["confidence"],
                infer_type=r["infer_type"],
                pmi_score=r["pmi_score"],
                llm_verdict=r["llm_verdict"],
                relationship_id=r["relationship_id"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Prune records ───────────────────────────────────────────────────

    async def save_prune_record(self, record: PruneRecord) -> None:
        """Insert a prune audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_prunes "
            "(id, cycle_id, group_id, entity_id, entity_name, "
            "entity_type, reason, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.entity_id,
            record.entity_name,
            record.entity_type,
            record.reason,
            record.timestamp,
        )

    async def get_prune_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[PruneRecord]:
        """Fetch prune records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_prunes "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            PruneRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                entity_id=r["entity_id"],
                entity_name=r["entity_name"],
                entity_type=r["entity_type"],
                reason=r["reason"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Reindex records ─────────────────────────────────────────────────

    async def save_reindex_record(self, record: ReindexRecord) -> None:
        """Insert a reindex audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_reindexes "
            "(id, cycle_id, group_id, entity_id, entity_name, "
            "source_phase, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.entity_id,
            record.entity_name,
            record.source_phase,
            record.timestamp,
        )

    async def get_reindex_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReindexRecord]:
        """Fetch reindex records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_reindexes "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            ReindexRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                entity_id=r["entity_id"],
                entity_name=r["entity_name"],
                source_phase=r["source_phase"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Replay records ──────────────────────────────────────────────────

    async def save_replay_record(self, record: ReplayRecord) -> None:
        """Insert a replay audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_replays "
            "(id, cycle_id, group_id, episode_id, new_entities_found, "
            "new_relationships_found, entities_updated, skipped_reason, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.episode_id,
            record.new_entities_found,
            record.new_relationships_found,
            record.entities_updated,
            record.skipped_reason,
            record.timestamp,
        )

    async def get_replay_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReplayRecord]:
        """Fetch replay records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_replays "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            ReplayRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                episode_id=r["episode_id"],
                new_entities_found=r["new_entities_found"],
                new_relationships_found=r["new_relationships_found"],
                entities_updated=r["entities_updated"],
                skipped_reason=r["skipped_reason"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Dream records ───────────────────────────────────────────────────

    async def save_dream_record(self, record: DreamRecord) -> None:
        """Insert a dream spreading audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_dreams "
            "(id, cycle_id, group_id, source_entity_id, target_entity_id, "
            "weight_delta, seed_entity_id, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.source_entity_id,
            record.target_entity_id,
            record.weight_delta,
            record.seed_entity_id,
            record.timestamp,
        )

    async def get_dream_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamRecord]:
        """Fetch dream records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_dreams "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            DreamRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                source_entity_id=r["source_entity_id"],
                target_entity_id=r["target_entity_id"],
                weight_delta=r["weight_delta"],
                seed_entity_id=r["seed_entity_id"] or "",
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Dream association records ────────────────────────────────────────

    async def save_dream_association_record(self, record: DreamAssociationRecord) -> None:
        """Insert a dream association audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_dream_associations "
            "(id, cycle_id, group_id, source_entity_id, target_entity_id, "
            "source_entity_name, target_entity_name, source_domain, target_domain, "
            "surprise_score, embedding_similarity, structural_proximity, "
            "relationship_id, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.source_entity_id,
            record.target_entity_id,
            record.source_entity_name,
            record.target_entity_name,
            record.source_domain,
            record.target_domain,
            record.surprise_score,
            record.embedding_similarity,
            record.structural_proximity,
            record.relationship_id,
            record.timestamp,
        )

    async def get_dream_association_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamAssociationRecord]:
        """Fetch dream association records for a cycle, ordered by surprise score."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_dream_associations "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY surprise_score DESC",
            cycle_id,
            group_id,
        )
        return [
            DreamAssociationRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                source_entity_id=r["source_entity_id"],
                target_entity_id=r["target_entity_id"],
                source_entity_name=r["source_entity_name"],
                target_entity_name=r["target_entity_name"],
                source_domain=r["source_domain"],
                target_domain=r["target_domain"],
                surprise_score=r["surprise_score"],
                embedding_similarity=r["embedding_similarity"],
                structural_proximity=r["structural_proximity"],
                relationship_id=r["relationship_id"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Triage records ──────────────────────────────────────────────────

    async def save_triage_record(self, record: TriageRecord) -> None:
        """Insert a triage audit record."""
        await self.pool.execute(
            "INSERT INTO consolidation_triage "
            "(id, cycle_id, group_id, episode_id, score, decision, "
            "score_breakdown, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.episode_id,
            record.score,
            record.decision,
            json.dumps(record.score_breakdown),
            record.timestamp,
        )

    async def get_triage_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[TriageRecord]:
        """Fetch triage records for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_triage "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY score DESC",
            cycle_id,
            group_id,
        )
        return [
            TriageRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                episode_id=r["episode_id"],
                score=r["score"],
                decision=r["decision"],
                score_breakdown=json.loads(r["score_breakdown"]) if r["score_breakdown"] else {},
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def save_decision_trace(self, record: DecisionTrace) -> None:
        """Insert a structured decision trace."""
        await self.pool.execute(
            "INSERT INTO consolidation_decision_traces "
            "(id, cycle_id, group_id, phase, candidate_type, candidate_id, "
            "decision, decision_source, confidence, threshold_band, features, "
            "constraints_json, policy_version, metadata, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, "
            "$12::jsonb, $13, $14::jsonb, $15) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.phase,
            record.candidate_type,
            record.candidate_id,
            record.decision,
            record.decision_source,
            record.confidence,
            record.threshold_band,
            json.dumps(record.features),
            json.dumps(record.constraints_hit),
            record.policy_version,
            json.dumps(record.metadata),
            record.timestamp,
        )

    async def get_decision_traces(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DecisionTrace]:
        """Fetch decision traces for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_decision_traces "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            DecisionTrace(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                phase=r["phase"],
                candidate_type=r["candidate_type"],
                candidate_id=r["candidate_id"],
                decision=r["decision"],
                decision_source=r["decision_source"],
                confidence=r["confidence"],
                threshold_band=r["threshold_band"],
                features=(
                    json.loads(r["features"]) if isinstance(r["features"], str)
                    else (r["features"] or {})
                ),
                constraints_hit=(
                    json.loads(r["constraints_json"]) if isinstance(r["constraints_json"], str)
                    else (r["constraints_json"] or [])
                ),
                policy_version=r["policy_version"],
                metadata=(
                    json.loads(r["metadata"]) if isinstance(r["metadata"], str)
                    else (r["metadata"] or {})
                ),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def save_decision_outcome_label(self, record: DecisionOutcomeLabel) -> None:
        """Insert a decision outcome label."""
        await self.pool.execute(
            "INSERT INTO consolidation_decision_outcomes "
            "(id, cycle_id, group_id, phase, decision_trace_id, outcome_type, "
            "label, value, metadata, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.phase,
            record.decision_trace_id,
            record.outcome_type,
            record.label,
            record.value,
            json.dumps(record.metadata),
            record.timestamp,
        )

    async def get_decision_outcome_labels(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DecisionOutcomeLabel]:
        """Fetch decision outcome labels for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_decision_outcomes "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            DecisionOutcomeLabel(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                phase=r["phase"],
                decision_trace_id=r["decision_trace_id"],
                outcome_type=r["outcome_type"],
                label=r["label"],
                value=r["value"],
                metadata=(
                    json.loads(r["metadata"]) if isinstance(r["metadata"], str)
                    else (r["metadata"] or {})
                ),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def save_distillation_example(self, record: DistillationExample) -> None:
        """Insert a distillation-ready example derived from decision history."""
        await self.pool.execute(
            "INSERT INTO consolidation_distillation_examples "
            "(id, cycle_id, group_id, phase, candidate_type, candidate_id, "
            "decision_trace_id, teacher_label, teacher_source, student_decision, "
            "student_confidence, threshold_band, features, correct, metadata, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
            "$13::jsonb, $14, $15::jsonb, $16) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.phase,
            record.candidate_type,
            record.candidate_id,
            record.decision_trace_id,
            record.teacher_label,
            record.teacher_source,
            record.student_decision,
            record.student_confidence,
            record.threshold_band,
            json.dumps(record.features),
            record.correct,
            json.dumps(record.metadata),
            record.timestamp,
        )

    async def get_distillation_examples(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DistillationExample]:
        """Fetch persisted distillation examples for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_distillation_examples "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY timestamp",
            cycle_id,
            group_id,
        )
        return [
            DistillationExample(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                phase=r["phase"],
                candidate_type=r["candidate_type"],
                candidate_id=r["candidate_id"],
                decision_trace_id=r["decision_trace_id"],
                teacher_label=r["teacher_label"],
                teacher_source=r["teacher_source"],
                student_decision=r["student_decision"],
                student_confidence=r["student_confidence"],
                threshold_band=r["threshold_band"],
                features=(
                    json.loads(r["features"]) if isinstance(r["features"], str)
                    else (r["features"] or {})
                ),
                correct=r["correct"],
                metadata=(
                    json.loads(r["metadata"]) if isinstance(r["metadata"], str)
                    else (r["metadata"] or {})
                ),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def save_calibration_snapshot(self, record: CalibrationSnapshot) -> None:
        """Insert a rolling calibration snapshot for a phase."""
        await self.pool.execute(
            "INSERT INTO consolidation_calibration_snapshots "
            "(id, cycle_id, group_id, phase, window_cycles, total_traces, "
            "labeled_examples, oracle_examples, abstain_count, accuracy, "
            "mean_confidence, expected_calibration_error, summary, timestamp) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, "
            "$13::jsonb, $14) "
            "ON CONFLICT (id) DO NOTHING",
            record.id,
            record.cycle_id,
            record.group_id,
            record.phase,
            record.window_cycles,
            record.total_traces,
            record.labeled_examples,
            record.oracle_examples,
            record.abstain_count,
            record.accuracy,
            record.mean_confidence,
            record.expected_calibration_error,
            json.dumps(record.summary),
            record.timestamp,
        )

    async def get_calibration_snapshots(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[CalibrationSnapshot]:
        """Fetch calibration snapshots for a cycle."""
        rows = await self.pool.fetch(
            "SELECT * FROM consolidation_calibration_snapshots "
            "WHERE cycle_id = $1 AND group_id = $2 ORDER BY phase, timestamp",
            cycle_id,
            group_id,
        )
        return [
            CalibrationSnapshot(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                phase=r["phase"],
                window_cycles=r["window_cycles"],
                total_traces=r["total_traces"],
                labeled_examples=r["labeled_examples"],
                oracle_examples=r["oracle_examples"],
                abstain_count=r["abstain_count"],
                accuracy=r["accuracy"],
                mean_confidence=r["mean_confidence"],
                expected_calibration_error=r["expected_calibration_error"],
                summary=(
                    json.loads(r["summary"]) if isinstance(r["summary"], str)
                    else (r["summary"] or {})
                ),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    # ── Cleanup ─────────────────────────────────────────────────────────

    async def cleanup(self, ttl_days: int = 90) -> int:
        """Delete cycle records older than ttl_days. Returns count deleted."""
        cutoff = time.time() - (ttl_days * 86400)

        # Get cycle IDs to delete
        rows = await self.pool.fetch(
            "SELECT id FROM consolidation_cycles WHERE started_at < $1",
            cutoff,
        )
        cycle_ids = [r["id"] for r in rows]
        if not cycle_ids:
            return 0

        # Delete child records using ANY($1::text[]) for all tables
        for table in (
            "consolidation_merges",
            "consolidation_identifier_reviews",
            "consolidation_inferred_edges",
            "consolidation_prunes",
            "consolidation_reindexes",
            "consolidation_replays",
            "consolidation_dreams",
            "consolidation_triage",
            "consolidation_dream_associations",
            "consolidation_decision_traces",
            "consolidation_decision_outcomes",
            "consolidation_distillation_examples",
            "consolidation_calibration_snapshots",
        ):
            await self.pool.execute(
                f"DELETE FROM {table} WHERE cycle_id = ANY($1::text[])",  # noqa: S608
                cycle_ids,
            )

        # Delete the cycles themselves
        result = await self.pool.execute(
            "DELETE FROM consolidation_cycles WHERE started_at < $1",
            cutoff,
        )
        # asyncpg returns a status string like "DELETE 5"
        count = int(result.split()[-1]) if result else 0
        return count

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_cycle(row: asyncpg.Record) -> ConsolidationCycle:
        phase_results: list[PhaseResult] = []
        raw_phases = row["phase_results"]
        if raw_phases:
            # asyncpg returns JSONB as native Python objects (list/dict),
            # but handle string case defensively
            parsed = json.loads(raw_phases) if isinstance(raw_phases, str) else raw_phases
            for pr in parsed:
                phase_results.append(
                    PhaseResult(
                        phase=pr["phase"],
                        status=pr["status"],
                        items_processed=pr["items_processed"],
                        items_affected=pr["items_affected"],
                        duration_ms=pr["duration_ms"],
                        error=pr.get("error"),
                    )
                )
        return ConsolidationCycle(
            id=row["id"],
            group_id=row["group_id"],
            trigger=row["trigger"],
            dry_run=bool(row["dry_run"]),
            status=row["status"],
            phase_results=phase_results,
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            total_duration_ms=row["total_duration_ms"] or 0.0,
            error=row["error"],
        )
