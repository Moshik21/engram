"""HelixDB-backed consolidation cycle audit store."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import aiosqlite

from engram.config import HelixDBConfig
from engram.models.consolidation import (
    CalibrationSnapshot,
    ConsolidationCycle,
    DecisionOutcomeLabel,
    DecisionTrace,
    DistillationExample,
    DreamAssociationRecord,
    DreamRecord,
    EvidenceAdjudicationRecord,
    GraphEmbedRecord,
    IdentifierReviewRecord,
    InferredEdge,
    MergeRecord,
    MicrogliaRecord,
    PhaseResult,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    TriageRecord,
)

logger = logging.getLogger(__name__)


def _safe_get(d: dict, key: str, default: Any = None) -> Any:
    """Safely get a value from a dict returned by Helix."""
    v = d.get(key, default)
    return v if v is not None else default


class HelixConsolidationStore:
    """Stores consolidation cycle history and audit records in HelixDB."""

    def __init__(
        self,
        config: HelixDBConfig,
        client=None,
        owns_client: bool | None = None,
    ) -> None:
        self._config = config
        self._client: Any | None = None
        self._helix_client = client  # Shared HelixClient (async httpx)
        self._owns_helix_client = client is None if owns_client is None else owns_client
        # cycle_id -> Helix internal node ID when known to be unique
        self._cycle_id_cache: dict[str, Any] = {}
        # (group_id, cycle_id) -> Helix internal node ID for brain-safe lookups
        self._cycle_group_id_cache: dict[tuple[str, str], Any] = {}
        # complement tag id (int) -> Helix internal node ID
        self._tag_id_cache: dict[int, Any] = {}
        # Auto-incrementing counter for complement tag IDs
        self._next_tag_id: int = 1
        self._scheduler_sidecar_db: aiosqlite.Connection | None = None
        self._scheduler_sidecar_ready = False

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a Helix query.

        Fast path: shared async HelixClient (httpx connection pool).
        Legacy fallback: synchronous helix-py SDK via thread pool.
        """
        # Fast path: shared async client
        if self._helix_client is not None:
            return await self._helix_client.query(endpoint, payload)

        # Legacy fallback: synchronous helix-py SDK
        client = self._client
        if client is None:
            raise RuntimeError("HelixConsolidationStore not initialized")
        try:
            result = await asyncio.to_thread(client.query, endpoint, payload)
            if result is None:
                return []
            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(result)
        except Exception as exc:
            exc_name = type(exc).__name__
            if "NoValue" in exc_name or "NotFound" in exc_name:
                return []
            raise

    @staticmethod
    def _extract_helix_id(item: dict):
        """Extract the Helix-assigned internal ID from a response dict."""
        for key in ("id", "_id", "node_id", "edge_id"):
            if key in item and item[key] is not None:
                return item[key]
        return None

    @staticmethod
    def _remember_unique_id_cache(cache: dict[str, Any], external_id: str, helix_id: Any) -> None:
        """Cache a bare external id only while it maps to one Helix id."""
        if helix_id is None or not external_id:
            return
        if external_id not in cache:
            cache[external_id] = helix_id
            return
        if str(cache[external_id]) != str(helix_id):
            cache[external_id] = None

    def _cache_tag_id(self, tag_id: int, helix_id: Any) -> None:
        """Cache a tag's Helix node id and advance the id counter past it.

        Repopulates _tag_id_cache from list-fetch queries so confirm/clear/demote
        work after a process restart, and seeds _next_tag_id beyond loaded ids to
        avoid collisions when creating new tags.
        """
        if helix_id is not None:
            self._tag_id_cache[tag_id] = helix_id
        if isinstance(tag_id, int) and tag_id >= self._next_tag_id:
            self._next_tag_id = tag_id + 1

    def _cache_cycle(
        self,
        helix_id: Any,
        cycle_id: str,
        group_id: str | None,
    ) -> None:
        if helix_id is None or not cycle_id:
            return
        if group_id:
            self._cycle_group_id_cache[(group_id, cycle_id)] = helix_id
        self._remember_unique_id_cache(self._cycle_id_cache, cycle_id, helix_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to HelixDB."""
        if self._helix_client is None:
            from engram.storage.helix.client import HelixClient

            self._helix_client = HelixClient(self._config)
        if not self._helix_client.is_connected:
            await self._helix_client.initialize()

        transport = getattr(self._config, "transport", "http")
        if transport == "native":
            await self._ensure_scheduler_sidecar()
            logger.info("HelixDB consolidation store initialized (native transport)")
            return

        from helix import Client  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "port": self._config.port,
            "verbose": self._config.verbose,
        }
        if self._config.api_endpoint:
            kwargs["url"] = self._config.api_endpoint
            kwargs["local"] = False
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key
        else:
            kwargs["local"] = True

        self._client = await asyncio.to_thread(Client, **kwargs)
        logger.info(
            "HelixDB consolidation store initialized (port=%d)",
            self._config.port,
        )

    async def close(self) -> None:
        """Close owned clients."""
        self._client = None
        if self._scheduler_sidecar_db is not None:
            await self._scheduler_sidecar_db.close()
            self._scheduler_sidecar_db = None
            self._scheduler_sidecar_ready = False
        if self._owns_helix_client and self._helix_client is not None:
            await self._helix_client.close()
            self._helix_client = None

    def _scheduler_sidecar_path(self) -> Path:
        base = (
            Path(self._config.data_dir).expanduser()
            if self._config.data_dir
            else Path.home() / ".helix" / "engram-native"
        )
        path = base / "scheduler-state.sqlite3"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    async def _ensure_scheduler_sidecar(self) -> aiosqlite.Connection:
        if self._scheduler_sidecar_db is not None:
            if not self._scheduler_sidecar_ready:
                await self._scheduler_sidecar_db.execute("""
                    CREATE TABLE IF NOT EXISTS consolidation_scheduler_tiers (
                        group_id TEXT NOT NULL,
                        tier TEXT NOT NULL,
                        last_run_at REAL NOT NULL,
                        PRIMARY KEY (group_id, tier)
                    )
                """)
                # Loop Steward active set-point (Helix companion; dual-write
                # with the ~/.engram file)
                await self._scheduler_sidecar_db.execute("""
                    CREATE TABLE IF NOT EXISTS loop_adjustments (
                        group_id TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        expires_at TEXT,
                        updated_at TEXT NOT NULL
                    )
                """)
                await self._scheduler_sidecar_db.commit()
                self._scheduler_sidecar_ready = True
            return self._scheduler_sidecar_db

        db = await aiosqlite.connect(self._scheduler_sidecar_path())
        db.row_factory = aiosqlite.Row
        self._scheduler_sidecar_db = db
        self._scheduler_sidecar_ready = False
        return await self._ensure_scheduler_sidecar()

    async def get_scheduler_tier_last_runs(self, group_id: str) -> dict[str, float]:
        """Return persisted per-tier last-run timestamps for a group."""
        db = await self._ensure_scheduler_sidecar()
        cursor = await db.execute(
            "SELECT tier, last_run_at FROM consolidation_scheduler_tiers WHERE group_id = ?",
            (group_id,),
        )
        rows = await cursor.fetchall()
        return {str(row["tier"]): float(row["last_run_at"]) for row in rows}

    async def save_scheduler_tier_last_runs(
        self,
        group_id: str,
        tier_times: dict[str, float],
    ) -> None:
        """Persist per-tier last-run timestamps for a group."""
        if not tier_times:
            return
        db = await self._ensure_scheduler_sidecar()
        await db.executemany(
            "INSERT INTO consolidation_scheduler_tiers "
            "(group_id, tier, last_run_at) VALUES (?, ?, ?) "
            "ON CONFLICT(group_id, tier) DO UPDATE SET last_run_at = excluded.last_run_at",
            [(group_id, tier, last_run_at) for tier, last_run_at in tier_times.items()],
        )
        await db.commit()

    async def save_loop_adjustment(self, group_id: str, payload: dict) -> None:
        """Persist Loop Steward active adjustment (Helix native sidecar)."""
        import json

        from engram.utils.dates import utc_now_iso

        db = await self._ensure_scheduler_sidecar()
        await db.execute(
            "INSERT INTO loop_adjustments (group_id, payload_json, expires_at, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(group_id) DO UPDATE SET "
            "payload_json = excluded.payload_json, "
            "expires_at = excluded.expires_at, "
            "updated_at = excluded.updated_at",
            (
                group_id,
                json.dumps(payload, default=str),
                str(payload.get("expires_at") or ""),
                utc_now_iso(),
            ),
        )
        await db.commit()

    async def get_loop_adjustment(self, group_id: str) -> dict | None:
        """Load Loop Steward active adjustment from Helix native sidecar."""
        import json

        db = await self._ensure_scheduler_sidecar()
        cursor = await db.execute(
            "SELECT payload_json FROM loop_adjustments WHERE group_id = ?",
            (group_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            # silent-ok: malformed sidecar payload → treat as no active adjustment
            return None
        return data if isinstance(data, dict) else None

    async def clear_loop_adjustment(self, group_id: str) -> bool:
        """Clear Loop Steward active adjustment from Helix native sidecar."""
        db = await self._ensure_scheduler_sidecar()
        cursor = await db.execute(
            "DELETE FROM loop_adjustments WHERE group_id = ?",
            (group_id,),
        )
        await db.commit()
        return bool(cursor.rowcount)

    # ------------------------------------------------------------------
    # Cycle CRUD
    # ------------------------------------------------------------------

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
        result = await self._query(
            "create_consol_cycle",
            {
                "cycle_id": cycle.id,
                "group_id": cycle.group_id,
                "trigger": cycle.trigger,
                "dry_run": cycle.dry_run,
                "status": cycle.status,
                "phase_results_json": phase_json,
                "started_at": cycle.started_at,
                "completed_at": cycle.completed_at or 0.0,
                "total_duration_ms": cycle.total_duration_ms,
                "error": cycle.error or "",
            },
        )
        if result:
            hid = self._extract_helix_id(result[0])
            self._cache_cycle(hid, cycle.id, cycle.group_id)

    async def update_cycle(self, cycle: ConsolidationCycle) -> None:
        """Update an existing consolidation cycle."""
        helix_id = self._cycle_group_id_cache.get((cycle.group_id, cycle.id))
        if helix_id is None:
            results = await self._query(
                "find_consol_cycles_by_group",
                {"gid": cycle.group_id},
            )
            for item in results:
                hid = self._extract_helix_id(item)
                self._cache_cycle(hid, item.get("cycle_id", ""), item.get("group_id"))
                if item.get("cycle_id") == cycle.id and item.get("group_id") == cycle.group_id:
                    helix_id = hid
                    break

        if helix_id is None:
            logger.warning("Cannot update cycle %s: not found in Helix", cycle.id)
            return

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
        await self._query(
            "update_consol_cycle",
            {
                "id": helix_id,
                "status": cycle.status,
                "phase_results_json": phase_json,
                "completed_at": cycle.completed_at or 0.0,
                "total_duration_ms": cycle.total_duration_ms,
                "error": cycle.error or "",
            },
        )

    async def get_cycle(self, cycle_id: str, group_id: str) -> ConsolidationCycle | None:
        """Fetch a single cycle by ID."""
        results = await self._query(
            "find_consol_cycles_by_group",
            {"gid": group_id},
        )
        for item in results:
            if item.get("cycle_id") == cycle_id:
                hid = self._extract_helix_id(item)
                self._cache_cycle(hid, cycle_id, item.get("group_id"))
                return self._dict_to_cycle(item)
        return None

    async def get_recent_cycles(
        self,
        group_id: str,
        limit: int = 10,
    ) -> list[ConsolidationCycle]:
        """Fetch recent cycles for a group, newest first."""
        results = await self._query(
            "find_consol_cycles_by_group",
            {"gid": group_id},
        )
        # Cache helix IDs
        for item in results:
            cid = item.get("cycle_id", "")
            hid = self._extract_helix_id(item)
            self._cache_cycle(hid, cid, item.get("group_id"))
        # Results should already be ordered by started_at DESC from the query
        cycles = [self._dict_to_cycle(item) for item in results]
        # Sort by started_at DESC in Python as safety measure
        cycles.sort(key=lambda c: c.started_at, reverse=True)
        return cycles[:limit]

    # ------------------------------------------------------------------
    # Merge records
    # ------------------------------------------------------------------

    async def save_merge_record(self, record: MergeRecord) -> None:
        """Insert a merge audit record."""
        await self._query(
            "create_consol_merge",
            {
                "merge_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "keep_id": record.keep_id,
                "remove_id": record.remove_id,
                "keep_name": record.keep_name,
                "remove_name": record.remove_name,
                "similarity": record.similarity,
                "decision_confidence": record.decision_confidence or 0.0,
                "decision_source": record.decision_source or "",
                "decision_reason": record.decision_reason or "",
                "relationships_transferred": record.relationships_transferred,
                "timestamp": record.timestamp,
            },
        )

    async def get_merge_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[MergeRecord]:
        """Fetch merge records for a cycle."""
        results = await self._query(
            "find_consol_merges_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            MergeRecord(
                id=_safe_get(r, "merge_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                keep_id=_safe_get(r, "keep_id", ""),
                remove_id=_safe_get(r, "remove_id", ""),
                keep_name=_safe_get(r, "keep_name", ""),
                remove_name=_safe_get(r, "remove_name", ""),
                similarity=_safe_get(r, "similarity", 0.0),
                decision_confidence=_safe_get(r, "decision_confidence", None),
                decision_source=_safe_get(r, "decision_source", None),
                decision_reason=_safe_get(r, "decision_reason", None),
                relationships_transferred=_safe_get(r, "relationships_transferred", 0),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Identifier review records
    # ------------------------------------------------------------------

    async def save_identifier_review_record(self, record: IdentifierReviewRecord) -> None:
        """Insert a quarantined identifier review record."""
        await self._query(
            "create_consol_identifier_review",
            {
                "review_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "entity_a_id": record.entity_a_id,
                "entity_b_id": record.entity_b_id,
                "entity_a_name": record.entity_a_name,
                "entity_b_name": record.entity_b_name,
                "entity_a_type": record.entity_a_type,
                "entity_b_type": record.entity_b_type,
                "raw_similarity": record.raw_similarity,
                "adjusted_similarity": record.adjusted_similarity or 0.0,
                "decision_source": record.decision_source or "",
                "decision_reason": record.decision_reason or "",
                "entity_a_regime": record.entity_a_regime or "",
                "entity_b_regime": record.entity_b_regime or "",
                "canonical_identifier_a": record.canonical_identifier_a or "",
                "canonical_identifier_b": record.canonical_identifier_b or "",
                "review_status": record.review_status,
                "metadata_json": json.dumps(record.metadata or {}),
                "timestamp": record.timestamp,
            },
        )

    async def get_identifier_review_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[IdentifierReviewRecord]:
        """Fetch quarantined identifier review records for a cycle."""
        results = await self._query(
            "find_consol_identifier_reviews_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            IdentifierReviewRecord(
                id=_safe_get(r, "review_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                entity_a_id=_safe_get(r, "entity_a_id", ""),
                entity_b_id=_safe_get(r, "entity_b_id", ""),
                entity_a_name=_safe_get(r, "entity_a_name", ""),
                entity_b_name=_safe_get(r, "entity_b_name", ""),
                entity_a_type=_safe_get(r, "entity_a_type", ""),
                entity_b_type=_safe_get(r, "entity_b_type", ""),
                raw_similarity=_safe_get(r, "raw_similarity", 0.0),
                adjusted_similarity=_safe_get(r, "adjusted_similarity", None),
                decision_source=_safe_get(r, "decision_source", None),
                decision_reason=_safe_get(r, "decision_reason", None),
                entity_a_regime=_safe_get(r, "entity_a_regime", None),
                entity_b_regime=_safe_get(r, "entity_b_regime", None),
                canonical_identifier_a=_safe_get(r, "canonical_identifier_a", None),
                canonical_identifier_b=_safe_get(r, "canonical_identifier_b", None),
                review_status=_safe_get(r, "review_status", "quarantined"),
                metadata=json.loads(_safe_get(r, "metadata_json", "{}") or "{}"),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Inferred edges
    # ------------------------------------------------------------------

    async def save_inferred_edge(self, edge: InferredEdge) -> None:
        """Insert an inferred edge audit record."""
        await self._query(
            "create_consol_inferred_edge",
            {
                "edge_id": edge.id,
                "cycle_id": edge.cycle_id,
                "group_id": edge.group_id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "source_name": edge.source_name,
                "target_name": edge.target_name,
                "co_occurrence_count": edge.co_occurrence_count,
                "confidence": edge.confidence,
                "infer_type": edge.infer_type,
                "pmi_score": edge.pmi_score or 0.0,
                "llm_verdict": edge.llm_verdict or "",
                "relationship_id": edge.relationship_id or "",
                "timestamp": edge.timestamp,
            },
        )

    async def get_inferred_edges(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[InferredEdge]:
        """Fetch inferred edge records for a cycle."""
        results = await self._query(
            "find_consol_inferred_edges_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            InferredEdge(
                id=_safe_get(r, "edge_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                source_id=_safe_get(r, "source_id", ""),
                target_id=_safe_get(r, "target_id", ""),
                source_name=_safe_get(r, "source_name", ""),
                target_name=_safe_get(r, "target_name", ""),
                co_occurrence_count=_safe_get(r, "co_occurrence_count", 0),
                confidence=_safe_get(r, "confidence", 0.0),
                infer_type=_safe_get(r, "infer_type", "co_occurrence"),
                pmi_score=_safe_get(r, "pmi_score", None),
                llm_verdict=_safe_get(r, "llm_verdict", None),
                relationship_id=_safe_get(r, "relationship_id", None),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Prune records
    # ------------------------------------------------------------------

    async def save_prune_record(self, record: PruneRecord) -> None:
        """Insert a prune audit record."""
        await self._query(
            "create_consol_prune",
            {
                "prune_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "entity_id": record.entity_id,
                "entity_name": record.entity_name,
                "entity_type": record.entity_type,
                "reason": record.reason,
                "timestamp": record.timestamp,
            },
        )

    async def get_prune_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[PruneRecord]:
        """Fetch prune records for a cycle."""
        results = await self._query(
            "find_consol_prunes_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            PruneRecord(
                id=_safe_get(r, "prune_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                entity_id=_safe_get(r, "entity_id", ""),
                entity_name=_safe_get(r, "entity_name", ""),
                entity_type=_safe_get(r, "entity_type", ""),
                reason=_safe_get(r, "reason", ""),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Reindex records
    # ------------------------------------------------------------------

    async def save_reindex_record(self, record: ReindexRecord) -> None:
        """Insert a reindex audit record."""
        await self._query(
            "create_consol_reindex",
            {
                "reindex_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "entity_id": record.entity_id,
                "entity_name": record.entity_name,
                "source_phase": record.source_phase,
                "timestamp": record.timestamp,
            },
        )

    async def get_reindex_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReindexRecord]:
        """Fetch reindex records for a cycle."""
        results = await self._query(
            "find_consol_reindexes_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            ReindexRecord(
                id=_safe_get(r, "reindex_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                entity_id=_safe_get(r, "entity_id", ""),
                entity_name=_safe_get(r, "entity_name", ""),
                source_phase=_safe_get(r, "source_phase", ""),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Replay records
    # ------------------------------------------------------------------

    async def save_replay_record(self, record: ReplayRecord) -> None:
        """Insert a replay audit record."""
        await self._query(
            "create_consol_replay",
            {
                "replay_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "episode_id": record.episode_id,
                "new_entities_found": record.new_entities_found,
                "new_relationships_found": record.new_relationships_found,
                "entities_updated": record.entities_updated,
                "skipped_reason": record.skipped_reason or "",
                "timestamp": record.timestamp,
            },
        )

    async def get_replay_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[ReplayRecord]:
        """Fetch replay records for a cycle."""
        results = await self._query(
            "find_consol_replays_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            ReplayRecord(
                id=_safe_get(r, "replay_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                episode_id=_safe_get(r, "episode_id", ""),
                new_entities_found=_safe_get(r, "new_entities_found", 0),
                new_relationships_found=_safe_get(r, "new_relationships_found", 0),
                entities_updated=_safe_get(r, "entities_updated", 0),
                skipped_reason=_safe_get(r, "skipped_reason", None),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Dream records
    # ------------------------------------------------------------------

    async def save_dream_record(self, record: DreamRecord) -> None:
        """Insert a dream spreading audit record."""
        await self._query(
            "create_consol_dream",
            {
                "dream_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "source_entity_id": record.source_entity_id,
                "target_entity_id": record.target_entity_id,
                "weight_delta": record.weight_delta,
                "seed_entity_id": record.seed_entity_id or "",
                "timestamp": record.timestamp,
            },
        )

    async def get_dream_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamRecord]:
        """Fetch dream records for a cycle."""
        results = await self._query(
            "find_consol_dreams_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            DreamRecord(
                id=_safe_get(r, "dream_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                source_entity_id=_safe_get(r, "source_entity_id", ""),
                target_entity_id=_safe_get(r, "target_entity_id", ""),
                weight_delta=_safe_get(r, "weight_delta", 0.0),
                seed_entity_id=_safe_get(r, "seed_entity_id", ""),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Dream association records
    # ------------------------------------------------------------------

    async def save_dream_association_record(self, record: DreamAssociationRecord) -> None:
        """Insert a dream association audit record."""
        await self._query(
            "create_consol_dream_association",
            {
                "assoc_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "source_entity_id": record.source_entity_id,
                "target_entity_id": record.target_entity_id,
                "source_entity_name": record.source_entity_name,
                "target_entity_name": record.target_entity_name,
                "source_domain": record.source_domain,
                "target_domain": record.target_domain,
                "surprise_score": record.surprise_score,
                "embedding_similarity": record.embedding_similarity,
                "structural_proximity": record.structural_proximity,
                "relationship_id": record.relationship_id or "",
                "timestamp": record.timestamp,
            },
        )

    async def get_dream_association_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DreamAssociationRecord]:
        """Fetch dream association records for a cycle, ordered by surprise score."""
        results = await self._query(
            "find_consol_dream_associations_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        records = [
            DreamAssociationRecord(
                id=_safe_get(r, "assoc_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                source_entity_id=_safe_get(r, "source_entity_id", ""),
                target_entity_id=_safe_get(r, "target_entity_id", ""),
                source_entity_name=_safe_get(r, "source_entity_name", ""),
                target_entity_name=_safe_get(r, "target_entity_name", ""),
                source_domain=_safe_get(r, "source_domain", ""),
                target_domain=_safe_get(r, "target_domain", ""),
                surprise_score=_safe_get(r, "surprise_score", 0.0),
                embedding_similarity=_safe_get(r, "embedding_similarity", 0.0),
                structural_proximity=_safe_get(r, "structural_proximity", 0.0),
                relationship_id=_safe_get(r, "relationship_id", None),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]
        records.sort(key=lambda rec: rec.surprise_score, reverse=True)
        return records

    # ------------------------------------------------------------------
    # Triage records
    # ------------------------------------------------------------------

    async def save_triage_record(self, record: TriageRecord) -> None:
        """Insert a triage audit record."""
        await self._query(
            "create_consol_triage",
            {
                "triage_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "episode_id": record.episode_id,
                "score": record.score,
                "decision": record.decision,
                "score_breakdown_json": json.dumps(record.score_breakdown),
                "timestamp": record.timestamp,
            },
        )

    async def get_triage_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[TriageRecord]:
        """Fetch triage records for a cycle, ordered by score DESC."""
        results = await self._query(
            "find_consol_triages_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        records = [
            TriageRecord(
                id=_safe_get(r, "triage_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                episode_id=_safe_get(r, "episode_id", ""),
                score=_safe_get(r, "score", 0.0),
                decision=_safe_get(r, "decision", ""),
                score_breakdown=json.loads(_safe_get(r, "score_breakdown_json", "{}") or "{}"),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]
        records.sort(key=lambda rec: rec.score, reverse=True)
        return records

    # ------------------------------------------------------------------
    # Graph embed records
    # ------------------------------------------------------------------

    async def save_graph_embed_record(self, record: GraphEmbedRecord) -> None:
        """Insert a graph embed audit record."""
        await self._query(
            "create_consol_graph_embed",
            {
                "embed_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "method": record.method,
                "entities_trained": record.entities_trained,
                "dimensions": record.dimensions,
                "training_duration_ms": record.training_duration_ms,
                "full_retrain": record.full_retrain,
                "timestamp": record.timestamp,
            },
        )

    async def get_graph_embed_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[GraphEmbedRecord]:
        """Fetch graph embed records for a cycle."""
        results = await self._query(
            "find_consol_graph_embeds_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            GraphEmbedRecord(
                id=_safe_get(r, "embed_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                method=_safe_get(r, "method", ""),
                entities_trained=_safe_get(r, "entities_trained", 0),
                dimensions=_safe_get(r, "dimensions", 0),
                training_duration_ms=_safe_get(r, "training_duration_ms", 0.0),
                full_retrain=bool(_safe_get(r, "full_retrain", False)),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Evidence adjudication records
    # ------------------------------------------------------------------

    async def save_evidence_adjudication_record(
        self,
        record: EvidenceAdjudicationRecord,
    ) -> None:
        """Insert an evidence adjudication audit record."""
        await self._query(
            "create_consol_evidence_adj",
            {
                "adj_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "evidence_id": record.evidence_id,
                "action": record.action,
                "new_confidence": record.new_confidence,
                "reason": record.reason,
                "timestamp": record.timestamp,
            },
        )

    async def get_evidence_adjudication_records(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[EvidenceAdjudicationRecord]:
        """Fetch evidence adjudication records for a cycle."""
        results = await self._query(
            "find_consol_evidence_adjs_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            EvidenceAdjudicationRecord(
                id=_safe_get(r, "adj_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                evidence_id=_safe_get(r, "evidence_id", ""),
                action=_safe_get(r, "action", ""),
                new_confidence=_safe_get(r, "new_confidence", 0.0),
                reason=_safe_get(r, "reason", ""),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Decision traces
    # ------------------------------------------------------------------

    async def save_decision_trace(self, record: DecisionTrace) -> None:
        """Insert a structured decision trace."""
        await self._query(
            "create_consol_decision_trace",
            {
                "trace_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "phase": record.phase,
                "candidate_type": record.candidate_type,
                "candidate_id": record.candidate_id,
                "decision": record.decision,
                "decision_source": record.decision_source,
                "confidence": record.confidence or 0.0,
                "threshold_band": record.threshold_band or "",
                "features_json": json.dumps(record.features),
                "constraints_json": json.dumps(record.constraints_hit),
                "policy_version": record.policy_version,
                "metadata_json": json.dumps(record.metadata),
                "timestamp": record.timestamp,
            },
        )

    async def get_decision_traces(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DecisionTrace]:
        """Fetch decision traces for a cycle."""
        results = await self._query(
            "find_consol_decision_traces_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            DecisionTrace(
                id=_safe_get(r, "trace_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                phase=_safe_get(r, "phase", ""),
                candidate_type=_safe_get(r, "candidate_type", ""),
                candidate_id=_safe_get(r, "candidate_id", ""),
                decision=_safe_get(r, "decision", ""),
                decision_source=_safe_get(r, "decision_source", ""),
                confidence=_safe_get(r, "confidence", None),
                threshold_band=_safe_get(r, "threshold_band", None),
                features=json.loads(_safe_get(r, "features_json", "{}") or "{}"),
                constraints_hit=json.loads(_safe_get(r, "constraints_json", "[]") or "[]"),
                policy_version=_safe_get(r, "policy_version", "v1"),
                metadata=json.loads(_safe_get(r, "metadata_json", "{}") or "{}"),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Decision outcome labels
    # ------------------------------------------------------------------

    async def save_decision_outcome_label(self, record: DecisionOutcomeLabel) -> None:
        """Insert a decision outcome label."""
        await self._query(
            "create_consol_decision_outcome",
            {
                "outcome_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "phase": record.phase,
                "decision_trace_id": record.decision_trace_id,
                "outcome_type": record.outcome_type,
                "outcome_label": record.label,
                "outcome_value": record.value or 0.0,
                "metadata_json": json.dumps(record.metadata),
                "timestamp": record.timestamp,
            },
        )

    async def get_decision_outcome_labels(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DecisionOutcomeLabel]:
        """Fetch decision outcome labels for a cycle."""
        results = await self._query(
            "find_consol_decision_outcomes_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        return [
            DecisionOutcomeLabel(
                id=_safe_get(r, "outcome_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                phase=_safe_get(r, "phase", ""),
                decision_trace_id=_safe_get(r, "decision_trace_id", ""),
                outcome_type=_safe_get(r, "outcome_type", ""),
                label=_safe_get(r, "label", ""),
                value=_safe_get(r, "value", None),
                metadata=json.loads(_safe_get(r, "metadata_json", "{}") or "{}"),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]

    # ------------------------------------------------------------------
    # Distillation examples
    # ------------------------------------------------------------------

    async def save_distillation_example(self, record: DistillationExample) -> None:
        """Insert a distillation-ready example derived from decision history."""
        correct_val: Any = False  # HelixDB Boolean field, can't be "" or null
        if record.correct is not None:
            correct_val = bool(record.correct)
        await self._query(
            "create_consol_distillation",
            {
                "distill_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "phase": record.phase,
                "candidate_type": record.candidate_type,
                "candidate_id": record.candidate_id,
                "decision_trace_id": record.decision_trace_id,
                "teacher_label": record.teacher_label,
                "teacher_source": record.teacher_source,
                "student_decision": record.student_decision,
                "student_confidence": record.student_confidence or 0.0,
                "threshold_band": record.threshold_band or "",
                "features_json": json.dumps(record.features),
                "correct": correct_val,
                "metadata_json": json.dumps(record.metadata),
                "timestamp": record.timestamp,
            },
        )

    async def get_distillation_examples(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[DistillationExample]:
        """Fetch persisted distillation examples for a cycle."""
        results = await self._query(
            "find_consol_distillations_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        records = []
        for r in results:
            raw_correct = _safe_get(r, "correct", None)
            # Handle correct as None/bool: empty string or missing means None
            if raw_correct is None or raw_correct == "":
                correct_val_parsed: bool | None = None
            else:
                correct_val_parsed = bool(raw_correct)
            records.append(
                DistillationExample(
                    id=_safe_get(r, "distill_id", ""),
                    cycle_id=_safe_get(r, "cycle_id", ""),
                    group_id=_safe_get(r, "group_id", ""),
                    phase=_safe_get(r, "phase", ""),
                    candidate_type=_safe_get(r, "candidate_type", ""),
                    candidate_id=_safe_get(r, "candidate_id", ""),
                    decision_trace_id=_safe_get(r, "decision_trace_id", ""),
                    teacher_label=_safe_get(r, "teacher_label", ""),
                    teacher_source=_safe_get(r, "teacher_source", ""),
                    student_decision=_safe_get(r, "student_decision", ""),
                    student_confidence=_safe_get(r, "student_confidence", None),
                    threshold_band=_safe_get(r, "threshold_band", None),
                    features=json.loads(_safe_get(r, "features_json", "{}") or "{}"),
                    correct=correct_val_parsed,
                    metadata=json.loads(_safe_get(r, "metadata_json", "{}") or "{}"),
                    timestamp=_safe_get(r, "timestamp", 0.0),
                )
            )
        return records

    # ------------------------------------------------------------------
    # Calibration snapshots
    # ------------------------------------------------------------------

    async def save_calibration_snapshot(self, record: CalibrationSnapshot) -> None:
        """Insert a rolling calibration snapshot for a phase."""
        await self._query(
            "create_consol_calibration",
            {
                "calibration_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "phase": record.phase,
                "window_cycles": record.window_cycles,
                "total_traces": record.total_traces,
                "labeled_examples": record.labeled_examples,
                "oracle_examples": record.oracle_examples,
                "abstain_count": record.abstain_count,
                "accuracy": record.accuracy or 0.0,
                "mean_confidence": record.mean_confidence or 0.0,
                "expected_calibration_error": record.expected_calibration_error or 0.0,
                "summary_json": json.dumps(record.summary),
                "timestamp": record.timestamp,
            },
        )

    async def get_calibration_snapshots(
        self,
        cycle_id: str,
        group_id: str,
    ) -> list[CalibrationSnapshot]:
        """Fetch calibration snapshots for a cycle, ordered by phase then timestamp."""
        results = await self._query(
            "find_consol_calibrations_by_cycle",
            {"cycle_id": cycle_id, "gid": group_id},
        )
        records = [
            CalibrationSnapshot(
                id=_safe_get(r, "calibration_id", ""),
                cycle_id=_safe_get(r, "cycle_id", ""),
                group_id=_safe_get(r, "group_id", ""),
                phase=_safe_get(r, "phase", ""),
                window_cycles=_safe_get(r, "window_cycles", 0),
                total_traces=_safe_get(r, "total_traces", 0),
                labeled_examples=_safe_get(r, "labeled_examples", 0),
                oracle_examples=_safe_get(r, "oracle_examples", 0),
                abstain_count=_safe_get(r, "abstain_count", 0),
                accuracy=_safe_get(r, "accuracy", None),
                mean_confidence=_safe_get(r, "mean_confidence", None),
                expected_calibration_error=_safe_get(r, "expected_calibration_error", None),
                summary=json.loads(_safe_get(r, "summary_json", "{}") or "{}"),
                timestamp=_safe_get(r, "timestamp", 0.0),
            )
            for r in results
        ]
        records.sort(key=lambda rec: (rec.phase, rec.timestamp))
        return records

    # ------------------------------------------------------------------
    # Complement tags (Microglia phase)
    # ------------------------------------------------------------------

    async def create_complement_tag(
        self,
        target_type: str,
        target_id: str,
        tag_type: str,
        score: float,
        cycle_tagged: int,
        group_id: str = "default",
    ) -> int:
        """Create or update a complement tag. Returns an integer tag ID.

        Implements UPSERT semantics: if a tag already exists for the given
        (target_id, tag_type) pair, update it instead of creating a new one.
        """
        # Check if tag already exists
        existing = await self.get_complement_tag(target_id, tag_type)
        if existing is not None:
            # Update existing tag
            tag_id = existing["id"]
            helix_id = self._tag_id_cache.get(tag_id)
            if helix_id is not None:
                await self._query(
                    "update_complement_tag",
                    {
                        "id": helix_id,
                        "score": score,
                        "cycle_tagged": cycle_tagged,
                    },
                )
            return tag_id

        # Assign a new integer ID
        tag_id = self._next_tag_id
        self._next_tag_id += 1

        result = await self._query(
            "create_complement_tag",
            {
                "tag_id": tag_id,
                "target_type": target_type,
                "target_id": target_id,
                "tag_type": tag_type,
                "score": score,
                "cycle_tagged": cycle_tagged,
                "cycle_confirmed": -1,  # sentinel for NULL
                "confirmed": False,
                "cleared": False,
                "group_id": group_id,
            },
        )
        if result:
            hid = self._extract_helix_id(result[0])
            if hid is not None:
                self._tag_id_cache[tag_id] = hid
        return tag_id

    async def get_active_complement_tags(
        self,
        group_id: str = "default",
    ) -> list[dict]:
        """Return all non-cleared complement tags."""
        results = await self._query(
            "find_active_complement_tags",
            {"gid": group_id},
        )
        tags = []
        for r in results:
            tag = self._tag_dict(r)
            self._cache_tag_id(tag["id"], self._extract_helix_id(r))
            tags.append(tag)
        return tags

    async def get_confirmed_tags(
        self,
        min_age_cycles: int,
        current_cycle: int,
        group_id: str = "default",
    ) -> list[dict]:
        """Return confirmed tags old enough for demotion.

        Filters in Python: tags where cycle_confirmed is set,
        (current_cycle - cycle_confirmed) >= min_age_cycles, and not cleared.
        """
        results = await self._query(
            "find_confirmed_complement_tags",
            {"gid": group_id},
        )
        tags = []
        for r in results:
            tag = self._tag_dict(r)
            self._cache_tag_id(tag["id"], self._extract_helix_id(r))
            cc = tag.get("cycle_confirmed")
            if cc is not None and (current_cycle - cc) >= min_age_cycles:
                tags.append(tag)
        return tags

    async def get_unconfirmed_tags(
        self,
        max_cycle: int,
        group_id: str = "default",
    ) -> list[dict]:
        """Return tags from previous cycles that haven't been confirmed.

        Filters in Python: tags where cycle_confirmed is None,
        cycle_tagged < max_cycle, and not cleared.
        """
        results = await self._query(
            "find_unconfirmed_complement_tags",
            {"gid": group_id},
        )
        tags = []
        for r in results:
            tag = self._tag_dict(r)
            self._cache_tag_id(tag["id"], self._extract_helix_id(r))
            if tag.get("cycle_tagged", 0) < max_cycle:
                tags.append(tag)
        return tags

    async def get_complement_tag(
        self,
        target_id: str,
        tag_type: str,
    ) -> dict | None:
        """Look up a single complement tag by target_id and tag_type."""
        results = await self._query(
            "find_complement_tags_by_target",
            {"target_id": target_id},
        )
        for r in results:
            if _safe_get(r, "tag_type", "") == tag_type:
                tag = self._tag_dict(r)
                # Cache helix ID
                hid = self._extract_helix_id(r)
                if hid is not None:
                    self._tag_id_cache[tag["id"]] = hid
                return tag
        return None

    async def confirm_complement_tag(
        self,
        tag_id: int,
        cycle_number: int,
    ) -> None:
        """Mark a complement tag as confirmed."""
        helix_id = self._tag_id_cache.get(tag_id)
        if helix_id is None:
            logger.warning("Cannot confirm tag %d: not found in cache", tag_id)
            return
        await self._query(
            "update_complement_tag",
            {
                "id": helix_id,
                "cycle_confirmed": cycle_number,
                "confirmed": True,
            },
        )

    async def clear_complement_tag(self, tag_id: int) -> None:
        """Clear (soft-delete) a complement tag."""
        helix_id = self._tag_id_cache.get(tag_id)
        if helix_id is None:
            logger.warning("Cannot clear tag %d: not found in cache", tag_id)
            return
        await self._query(
            "update_complement_tag",
            {
                "id": helix_id,
                "cleared": True,
            },
        )

    # ------------------------------------------------------------------
    # Microglia records
    # ------------------------------------------------------------------

    async def save_microglia_record(self, record: MicrogliaRecord) -> None:
        """Persist a microglia audit record."""
        await self._query(
            "create_consol_microglia",
            {
                "microglia_id": record.id,
                "cycle_id": record.cycle_id,
                "group_id": record.group_id,
                "target_type": record.target_type,
                "target_id": record.target_id,
                "action": record.action,
                "tag_type": record.tag_type,
                "score": record.score,
                "detail": record.detail,
                "timestamp": record.timestamp,
            },
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def cleanup(self, ttl_days: int = 90) -> int:
        """Delete cycle records older than ttl_days. Returns count deleted.

        Since Helix doesn't have cascading deletes, we query for old cycles
        and then delete all associated records for each cycle.
        """
        cutoff = time.time() - (ttl_days * 86400)

        # Find all cycles to determine which are expired
        # We scan all groups — cleanup is global
        # First, try to get cycles from cache
        expired_cycles: list[tuple[str, str, Any]] = []

        # We need to scan cycles. Since we don't know all group IDs globally,
        # cleanup works from cycles this process has cached while preserving
        # their group scope.
        for (group_id, cycle_id), helix_id in list(self._cycle_group_id_cache.items()):
            try:
                results = await self._query(
                    "get_consol_cycle",
                    {"id": helix_id},
                )
                if results:
                    started_at = _safe_get(results[0], "started_at", 0.0)
                    if started_at < cutoff:
                        expired_cycles.append((group_id, cycle_id, helix_id))
            except Exception:
                # silent-ok: best-effort TTL expiry scan; an un-scannable cycle
                # is simply reconsidered on the next cleanup pass.
                pass

        if not expired_cycles:
            return 0

        # Record types to delete for each expired cycle
        record_query_names = [
            "find_consol_merges_by_cycle",
            "find_consol_identifier_reviews_by_cycle",
            "find_consol_inferred_edges_by_cycle",
            "find_consol_prunes_by_cycle",
            "find_consol_reindexes_by_cycle",
            "find_consol_replays_by_cycle",
            "find_consol_dreams_by_cycle",
            "find_consol_triages_by_cycle",
            "find_consol_dream_associations_by_cycle",
            "find_consol_graph_embeds_by_cycle",
            # mature/semanticize/schema phases are deleted; entries stay so
            # retention drains their legacy audit rows from existing brains.
            "find_consol_maturations_by_cycle",
            "find_consol_semantic_transitions_by_cycle",
            "find_consol_schemas_by_cycle",
            "find_consol_decision_traces_by_cycle",
            "find_consol_decision_outcomes_by_cycle",
            "find_consol_distillations_by_cycle",
            "find_consol_calibrations_by_cycle",
            "find_consol_evidence_adjs_by_cycle",
            "find_consol_microglias_by_cycle",
        ]

        deleted_count = 0
        for group_id, cycle_id, helix_id in expired_cycles:
            # Delete associated records
            for query_name in record_query_names:
                try:
                    records = await self._query(
                        query_name,
                        {"cycle_id": cycle_id, "gid": group_id},
                    )
                    for rec in records:
                        rec_helix_id = self._extract_helix_id(rec)
                        if rec_helix_id is not None:
                            try:
                                await self._query(
                                    "delete_node",
                                    {"id": rec_helix_id},
                                )
                            except Exception:
                                # silent-ok: best-effort GC of an expired child
                                # record; retried next cleanup pass if it fails.
                                pass
                except Exception:
                    # silent-ok: best-effort GC record scan for one record type;
                    # the remaining types and cycles still get processed.
                    pass

            # Delete the cycle node itself
            if helix_id is not None:
                try:
                    await self._query(
                        "delete_node",
                        {"id": helix_id},
                    )
                    deleted_count += 1
                    self._cycle_group_id_cache.pop((group_id, cycle_id), None)
                    if str(self._cycle_id_cache.get(cycle_id)) == str(helix_id):
                        self._cycle_id_cache.pop(cycle_id, None)
                except Exception:
                    # silent-ok: best-effort GC of an expired cycle node; the
                    # returned deleted_count omits it, so the failure is not
                    # metered as a success, and the next pass retries it.
                    pass

        return deleted_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_cycle(d: dict) -> ConsolidationCycle:
        """Convert a Helix dict to a ConsolidationCycle model."""
        phase_results: list[PhaseResult] = []
        raw_phases = _safe_get(d, "phase_results_json", "[]")
        if raw_phases:
            try:
                for pr in json.loads(raw_phases):
                    phase_results.append(
                        PhaseResult(
                            phase=pr.get("phase", ""),
                            status=pr.get("status", "success"),
                            items_processed=pr.get("items_processed", 0),
                            items_affected=pr.get("items_affected", 0),
                            duration_ms=pr.get("duration_ms", 0.0),
                            error=pr.get("error"),
                        )
                    )
            except (json.JSONDecodeError, TypeError):
                # silent-ok: malformed phase_results_json → cycle is still
                # returned, just without its per-phase breakdown.
                pass

        completed_at = _safe_get(d, "completed_at", None)
        # Treat 0.0 as None (sentinel for no completion)
        if completed_at == 0.0:
            completed_at = None

        error = _safe_get(d, "error", None)
        if error == "":
            error = None

        return ConsolidationCycle(
            id=_safe_get(d, "cycle_id", ""),
            group_id=_safe_get(d, "group_id", ""),
            trigger=_safe_get(d, "trigger", "manual"),
            dry_run=bool(_safe_get(d, "dry_run", True)),
            status=_safe_get(d, "status", "pending"),
            phase_results=phase_results,
            started_at=_safe_get(d, "started_at", 0.0),
            completed_at=completed_at,
            total_duration_ms=_safe_get(d, "total_duration_ms", 0.0),
            error=error,
        )

    @staticmethod
    def _tag_dict(r: dict) -> dict:
        """Convert a Helix complement tag dict to the standard dict shape."""
        cycle_confirmed_raw = _safe_get(r, "cycle_confirmed", -1)
        # Treat sentinel -1 as None
        cycle_confirmed = None if cycle_confirmed_raw == -1 else cycle_confirmed_raw

        return {
            "id": _safe_get(r, "tag_id", 0),
            "target_type": _safe_get(r, "target_type", ""),
            "target_id": _safe_get(r, "target_id", ""),
            "tag_type": _safe_get(r, "tag_type", ""),
            "score": _safe_get(r, "score", 0.0),
            "cycle_tagged": _safe_get(r, "cycle_tagged", 0),
            "cycle_confirmed": cycle_confirmed,
            "group_id": _safe_get(r, "group_id", "default"),
        }
