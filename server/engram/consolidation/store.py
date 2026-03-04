"""SQLite-backed consolidation cycle audit store."""

from __future__ import annotations

import json
import time

import aiosqlite

from engram.models.consolidation import (
    ConsolidationCycle,
    DreamAssociationRecord,
    DreamRecord,
    InferredEdge,
    MergeRecord,
    PhaseResult,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    TriageRecord,
)


class SQLiteConsolidationStore:
    """Stores consolidation cycle history and audit records in SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        """Create tables if they don't exist."""
        if db:
            self._db = db
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_cycles (
                id TEXT PRIMARY KEY,
                group_id TEXT NOT NULL,
                trigger TEXT NOT NULL,
                dry_run INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'pending',
                phase_results TEXT,
                started_at REAL NOT NULL,
                completed_at REAL,
                total_duration_ms REAL DEFAULT 0.0,
                error TEXT
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_merges (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                keep_id TEXT NOT NULL,
                remove_id TEXT NOT NULL,
                keep_name TEXT NOT NULL,
                remove_name TEXT NOT NULL,
                similarity REAL NOT NULL,
                relationships_transferred INTEGER DEFAULT 0,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_inferred_edges (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                target_name TEXT NOT NULL,
                co_occurrence_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                infer_type TEXT DEFAULT 'co_occurrence',
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_prunes (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                reason TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_reindexes (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                source_phase TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_cycles_group ON consolidation_cycles(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_merges_cycle ON consolidation_merges(cycle_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_edges_cycle "
            "ON consolidation_inferred_edges(cycle_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_prunes_cycle ON consolidation_prunes(cycle_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_reindexes_cycle "
            "ON consolidation_reindexes(cycle_id)"
        )
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_replays (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                new_entities_found INTEGER DEFAULT 0,
                new_relationships_found INTEGER DEFAULT 0,
                entities_updated INTEGER DEFAULT 0,
                skipped_reason TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_replays_cycle ON consolidation_replays(cycle_id)"
        )
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_dreams (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                weight_delta REAL NOT NULL,
                seed_entity_id TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_dreams_cycle ON consolidation_dreams(cycle_id)"
        )
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_triage (
                id TEXT PRIMARY KEY,
                cycle_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                episode_id TEXT NOT NULL,
                score REAL NOT NULL,
                decision TEXT NOT NULL,
                score_breakdown_json TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_triage_cycle ON consolidation_triage(cycle_id)"
        )
        await self.db.execute("""
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
                surprise_score REAL NOT NULL,
                embedding_similarity REAL NOT NULL,
                structural_proximity REAL NOT NULL,
                relationship_id TEXT,
                timestamp REAL NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_consol_dream_assoc_cycle "
            "ON consolidation_dream_associations(cycle_id)"
        )
        # Migrations: add columns for existing databases
        for migration_sql in [
            "ALTER TABLE consolidation_inferred_edges "
            "ADD COLUMN infer_type TEXT DEFAULT 'co_occurrence'",
            "ALTER TABLE consolidation_inferred_edges ADD COLUMN pmi_score REAL",
            "ALTER TABLE consolidation_inferred_edges ADD COLUMN llm_verdict TEXT",
            "ALTER TABLE consolidation_inferred_edges ADD COLUMN relationship_id TEXT",
        ]:
            try:
                await self.db.execute(migration_sql)
            except Exception:
                pass
        await self.db.commit()

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteConsolidationStore not initialized.")
        return self._db

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
        await self.db.execute(
            "INSERT INTO consolidation_cycles "
            "(id, group_id, trigger, dry_run, status, phase_results, "
            "started_at, completed_at, total_duration_ms, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cycle.id,
                cycle.group_id,
                cycle.trigger,
                1 if cycle.dry_run else 0,
                cycle.status,
                phase_json,
                cycle.started_at,
                cycle.completed_at,
                cycle.total_duration_ms,
                cycle.error,
            ),
        )
        await self.db.commit()

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
        await self.db.execute(
            "UPDATE consolidation_cycles "
            "SET status = ?, phase_results = ?, completed_at = ?, "
            "total_duration_ms = ?, error = ? "
            "WHERE id = ? AND group_id = ?",
            (
                cycle.status,
                phase_json,
                cycle.completed_at,
                cycle.total_duration_ms,
                cycle.error,
                cycle.id,
                cycle.group_id,
            ),
        )
        await self.db.commit()

    async def get_cycle(self, cycle_id: str, group_id: str) -> ConsolidationCycle | None:
        """Fetch a single cycle by ID."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_cycles WHERE id = ? AND group_id = ?",
            (cycle_id, group_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_cycle(row)

    async def get_recent_cycles(
        self,
        group_id: str,
        limit: int = 10,
    ) -> list[ConsolidationCycle]:
        """Fetch recent cycles for a group, newest first."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_cycles WHERE group_id = ? "
            "ORDER BY started_at DESC LIMIT ?",
            (group_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_cycle(r) for r in rows]

    async def save_merge_record(self, record: MergeRecord) -> None:
        """Insert a merge audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_merges "
            "(id, cycle_id, group_id, keep_id, remove_id, keep_name, "
            "remove_name, similarity, relationships_transferred, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.keep_id,
                record.remove_id,
                record.keep_name,
                record.remove_name,
                record.similarity,
                record.relationships_transferred,
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def save_inferred_edge(self, edge: InferredEdge) -> None:
        """Insert an inferred edge audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_inferred_edges "
            "(id, cycle_id, group_id, source_id, target_id, source_name, "
            "target_name, co_occurrence_count, confidence, infer_type, "
            "pmi_score, llm_verdict, relationship_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
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
            ),
        )
        await self.db.commit()

    async def save_prune_record(self, record: PruneRecord) -> None:
        """Insert a prune audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_prunes "
            "(id, cycle_id, group_id, entity_id, entity_name, "
            "entity_type, reason, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.entity_id,
                record.entity_name,
                record.entity_type,
                record.reason,
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def get_merge_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[MergeRecord]:
        """Fetch merge records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_merges "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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
                relationships_transferred=r["relationships_transferred"],
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def get_inferred_edges(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[InferredEdge]:
        """Fetch inferred edge records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_inferred_edges "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
        keys = set()
        if rows:
            keys = set(rows[0].keys())
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
                infer_type=r["infer_type"] if "infer_type" in keys else "co_occurrence",
                pmi_score=r["pmi_score"] if "pmi_score" in keys else None,
                llm_verdict=r["llm_verdict"] if "llm_verdict" in keys else None,
                relationship_id=r["relationship_id"] if "relationship_id" in keys else None,
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def save_reindex_record(self, record: ReindexRecord) -> None:
        """Insert a reindex audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_reindexes "
            "(id, cycle_id, group_id, entity_id, entity_name, "
            "source_phase, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.entity_id,
                record.entity_name,
                record.source_phase,
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def get_reindex_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReindexRecord]:
        """Fetch reindex records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_reindexes "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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

    async def get_prune_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[PruneRecord]:
        """Fetch prune records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_prunes "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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

    async def save_replay_record(self, record: ReplayRecord) -> None:
        """Insert a replay audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_replays "
            "(id, cycle_id, group_id, episode_id, new_entities_found, "
            "new_relationships_found, entities_updated, skipped_reason, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.episode_id,
                record.new_entities_found,
                record.new_relationships_found,
                record.entities_updated,
                record.skipped_reason,
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def get_replay_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReplayRecord]:
        """Fetch replay records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_replays "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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

    async def save_dream_record(self, record: DreamRecord) -> None:
        """Insert a dream spreading audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_dreams "
            "(id, cycle_id, group_id, source_entity_id, target_entity_id, "
            "weight_delta, seed_entity_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.source_entity_id,
                record.target_entity_id,
                record.weight_delta,
                record.seed_entity_id,
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def get_dream_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamRecord]:
        """Fetch dream records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_dreams "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY timestamp",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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

    async def save_dream_association_record(self, record: DreamAssociationRecord) -> None:
        """Insert a dream association audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_dream_associations "
            "(id, cycle_id, group_id, source_entity_id, target_entity_id, "
            "source_entity_name, target_entity_name, source_domain, target_domain, "
            "surprise_score, embedding_similarity, structural_proximity, "
            "relationship_id, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
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
            ),
        )
        await self.db.commit()

    async def get_dream_association_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamAssociationRecord]:
        """Fetch dream association records for a cycle, ordered by surprise score."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_dream_associations "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY surprise_score DESC",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
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

    async def save_triage_record(self, record: TriageRecord) -> None:
        """Insert a triage audit record."""
        await self.db.execute(
            "INSERT INTO consolidation_triage "
            "(id, cycle_id, group_id, episode_id, score, decision, "
            "score_breakdown_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.cycle_id,
                record.group_id,
                record.episode_id,
                record.score,
                record.decision,
                json.dumps(record.score_breakdown),
                record.timestamp,
            ),
        )
        await self.db.commit()

    async def get_triage_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[TriageRecord]:
        """Fetch triage records for a cycle."""
        cursor = await self.db.execute(
            "SELECT * FROM consolidation_triage "
            "WHERE cycle_id = ? AND group_id = ? ORDER BY score DESC",
            (cycle_id, group_id),
        )
        rows = await cursor.fetchall()
        return [
            TriageRecord(
                id=r["id"],
                cycle_id=r["cycle_id"],
                group_id=r["group_id"],
                episode_id=r["episode_id"],
                score=r["score"],
                decision=r["decision"],
                score_breakdown=json.loads(r["score_breakdown_json"] or "{}"),
                timestamp=r["timestamp"],
            )
            for r in rows
        ]

    async def cleanup(self, ttl_days: int = 90) -> int:
        """Delete cycle records older than ttl_days. Returns count deleted."""
        cutoff = time.time() - (ttl_days * 86400)
        # Get cycle IDs to delete
        cursor = await self.db.execute(
            "SELECT id FROM consolidation_cycles WHERE started_at < ?",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        cycle_ids = [r["id"] for r in rows]
        if not cycle_ids:
            return 0

        placeholders = ",".join("?" * len(cycle_ids))
        await self.db.execute(
            f"DELETE FROM consolidation_merges WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_inferred_edges WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_prunes WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_reindexes WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_replays WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_dreams WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_triage WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        await self.db.execute(
            f"DELETE FROM consolidation_dream_associations WHERE cycle_id IN ({placeholders})",
            cycle_ids,
        )
        del_cursor = await self.db.execute(
            "DELETE FROM consolidation_cycles WHERE started_at < ?",
            (cutoff,),
        )
        await self.db.commit()
        return del_cursor.rowcount

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @staticmethod
    def _row_to_cycle(row) -> ConsolidationCycle:
        phase_results = []
        if row["phase_results"]:
            for pr in json.loads(row["phase_results"]):
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
