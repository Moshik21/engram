"""SQLite implementation of GraphStore protocol (Lite mode)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import aiosqlite

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.protocols import ENTITY_UPDATABLE_FIELDS, EPISODE_UPDATABLE_FIELDS
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)


class SQLiteGraphStore:
    """Graph store backed by a single SQLite file with WAL mode."""

    def __init__(self, db_path: str, encryptor=None) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._encryptor = encryptor

    async def initialize(self) -> None:
        """Create tables and indexes. Idempotent."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")

        schema_path = Path(__file__).parent / "schema.sql"
        schema_sql = schema_path.read_text()
        await self._db.executescript(schema_sql)
        await self._db.commit()

        # Migrations — add new columns idempotently
        migrations = [
            # Week 2
            "ALTER TABLE entities ADD COLUMN pii_detected INTEGER DEFAULT 0",
            "ALTER TABLE entities ADD COLUMN pii_categories TEXT",
            "ALTER TABLE relationships ADD COLUMN confidence REAL DEFAULT 1.0",
            # Week 6 — episode pipeline fields
            "ALTER TABLE episodes ADD COLUMN updated_at TEXT",
            "ALTER TABLE episodes ADD COLUMN error TEXT",
            "ALTER TABLE episodes ADD COLUMN retry_count INTEGER DEFAULT 0",
            "ALTER TABLE episodes ADD COLUMN processing_duration_ms INTEGER",
            # Identity core
            "ALTER TABLE entities ADD COLUMN identity_core INTEGER DEFAULT 0",
            # Negation/uncertainty polarity
            "ALTER TABLE relationships ADD COLUMN polarity TEXT DEFAULT 'positive'",
        ]
        for sql in migrations:
            try:
                await self._db.execute(sql)
            except Exception:
                pass  # Column already exists
        await self._db.commit()

        logger.info("SQLite graph store initialized at %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("GraphStore not initialized. Call initialize() first.")
        return self._db

    # --- Encryption helpers ---

    def _encrypt(self, group_id: str, plaintext: str | None) -> str | None:
        if not plaintext or not self._encryptor:
            return plaintext
        return self._encryptor.encrypt(group_id, plaintext)

    def _decrypt(self, group_id: str, data: str | None) -> str | None:
        if not data or not self._encryptor:
            return data
        return self._encryptor.decrypt(group_id, data)

    # --- Entities ---

    async def create_entity(self, entity: Entity) -> str:
        now = datetime.utcnow().isoformat()
        summary = self._encrypt(entity.group_id, entity.summary)
        await self.db.execute(
            """INSERT INTO entities
               (id, name, entity_type, summary, attributes, group_id,
                created_at, updated_at,
                access_count, last_accessed, pii_detected, pii_categories)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.id,
                entity.name,
                entity.entity_type,
                summary,
                json.dumps(entity.attributes) if entity.attributes else None,
                entity.group_id,
                entity.created_at.isoformat() if entity.created_at else now,
                now,
                entity.access_count,
                entity.last_accessed.isoformat() if entity.last_accessed else None,
                1 if entity.pii_detected else 0,
                json.dumps(entity.pii_categories) if entity.pii_categories else None,
            ),
        )
        await self.db.commit()
        return entity.id

    async def get_entity(self, entity_id: str, group_id: str) -> Entity | None:
        cursor = await self.db.execute(
            "SELECT * FROM entities WHERE id = ? AND group_id = ? AND deleted_at IS NULL",
            (entity_id, group_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_entity(row, group_id)

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        if not updates:
            return
        updates["updated_at"] = datetime.utcnow().isoformat()
        invalid = set(updates.keys()) - ENTITY_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed entity update fields: {invalid}")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [entity_id, group_id]
        await self.db.execute(
            f"UPDATE entities SET {set_clause} WHERE id = ? AND group_id = ?",
            values,
        )
        await self.db.commit()

    async def delete_entity(self, entity_id: str, soft: bool = True, *, group_id: str) -> None:
        if soft:
            await self.db.execute(
                "UPDATE entities SET deleted_at = ? WHERE id = ? AND group_id = ?",
                (datetime.utcnow().isoformat(), entity_id, group_id),
            )
        else:
            await self.db.execute("DELETE FROM episode_entities WHERE entity_id = ?", (entity_id,))
            await self.db.execute(
                "DELETE FROM relationships WHERE (source_id = ? OR target_id = ?) AND group_id = ?",
                (entity_id, entity_id, group_id),
            )
            await self.db.execute(
                "DELETE FROM entities WHERE id = ? AND group_id = ?",
                (entity_id, group_id),
            )
        await self.db.commit()

    async def find_entities(
        self,
        name: str | None = None,
        entity_type: str | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[Entity]:
        conditions = ["deleted_at IS NULL"]
        params: list = []
        if name:
            conditions.append("LOWER(name) LIKE LOWER(?)")
            params.append(f"%{name}%")
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if group_id:
            conditions.append("group_id = ?")
            params.append(group_id)
        where = " AND ".join(conditions)
        params.append(limit)
        cursor = await self.db.execute(
            f"SELECT * FROM entities WHERE {where} ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_entity(r, group_id) for r in rows]

    async def find_entity_candidates(
        self, name: str, group_id: str, limit: int = 30,
    ) -> list[Entity]:
        """Retrieve candidate entities for fuzzy resolution via exact + FTS5 match."""
        seen_ids: set[str] = set()
        results: list[Entity] = []

        # Phase 1: Exact normalized name match
        normalized = name.strip().lower().replace("_", " ").replace("-", " ")
        cursor = await self.db.execute(
            """SELECT * FROM entities
               WHERE LOWER(REPLACE(REPLACE(TRIM(name), '_', ' '), '-', ' ')) = ?
                 AND group_id = ? AND deleted_at IS NULL
               LIMIT ?""",
            (normalized, group_id, limit),
        )
        for row in await cursor.fetchall():
            entity = self._row_to_entity(row, group_id)
            if entity.id not in seen_ids:
                seen_ids.add(entity.id)
                results.append(entity)

        if len(results) >= limit:
            return results[:limit]

        # Phase 2: FTS5 token search
        tokens = [t for t in name.strip().split() if len(t) >= 2]
        if tokens:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)
            try:
                remaining = limit - len(results)
                fts_cursor = await self.db.execute(
                    """SELECT e.* FROM entities_fts fts
                       JOIN entities e ON e.rowid = fts.rowid
                       WHERE entities_fts MATCH ?
                         AND e.group_id = ? AND e.deleted_at IS NULL
                       ORDER BY bm25(entities_fts)
                       LIMIT ?""",
                    (fts_query, group_id, remaining),
                )
                for row in await fts_cursor.fetchall():
                    entity = self._row_to_entity(row, group_id)
                    if entity.id not in seen_ids:
                        seen_ids.add(entity.id)
                        results.append(entity)
            except Exception:
                pass  # Malformed FTS5 query — fall through gracefully

        return results[:limit]

    # --- Relationships ---

    async def create_relationship(self, rel: Relationship) -> str:
        await self.db.execute(
            """INSERT INTO relationships
               (id, source_id, target_id, predicate, weight,
                valid_from, valid_to, created_at, source_episode, group_id, confidence,
                polarity)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rel.id,
                rel.source_id,
                rel.target_id,
                rel.predicate,
                rel.weight,
                rel.valid_from.isoformat() if rel.valid_from else None,
                rel.valid_to.isoformat() if rel.valid_to else None,
                rel.created_at.isoformat() if rel.created_at else datetime.utcnow().isoformat(),
                rel.source_episode,
                rel.group_id,
                rel.confidence,
                rel.polarity,
            ),
        )
        await self.db.commit()
        return rel.id

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "both",
        predicate: str | None = None,
        active_only: bool = True,
        group_id: str = "default",
    ) -> list[Relationship]:
        conditions: list[str] = []
        params: list = []
        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entity_id)
        else:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entity_id, entity_id])
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if active_only:
            conditions.append("(valid_to IS NULL OR datetime(valid_to) > datetime('now'))")
        conditions.append("group_id = ?")
        params.append(group_id)
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT * FROM relationships WHERE {where}",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def invalidate_relationship(self, rel_id: str, valid_to: datetime, group_id: str) -> None:
        await self.db.execute(
            "UPDATE relationships SET valid_to = ? WHERE id = ? AND group_id = ?",
            (valid_to.isoformat(), rel_id, group_id),
        )
        await self.db.commit()

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        weight_delta: float,
        max_weight: float = 3.0,
        group_id: str = "default",
    ) -> float | None:
        """Atomically increment edge weight, capped at max_weight."""
        cursor = await self.db.execute(
            """UPDATE relationships
               SET weight = MIN(?, weight + ?)
               WHERE group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))
                 AND ((source_id = ? AND target_id = ?)
                   OR (source_id = ? AND target_id = ?))
               RETURNING weight""",
            (max_weight, weight_delta, group_id, source_id, target_id, target_id, source_id),
        )
        row = await cursor.fetchone()
        await self.db.commit()
        return row[0] if row else None

    async def find_conflicting_relationships(
        self,
        source_id: str,
        predicate: str,
        group_id: str,
    ) -> list[Relationship]:
        """Find active relationships with same source + predicate (for exclusive predicates)."""
        cursor = await self.db.execute(
            """SELECT * FROM relationships
               WHERE source_id = ? AND predicate = ? AND group_id = ? AND valid_to IS NULL""",
            (source_id, predicate, group_id),
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def find_existing_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        group_id: str,
    ) -> Relationship | None:
        """Find an active relationship matching (source, target, predicate)."""
        cursor = await self.db.execute(
            """SELECT * FROM relationships
               WHERE source_id = ? AND target_id = ? AND predicate = ?
                 AND group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))
               LIMIT 1""",
            (source_id, target_id, predicate, group_id),
        )
        row = await cursor.fetchone()
        return self._row_to_relationship(row) if row else None

    async def get_relationships_at(
        self,
        entity_id: str,
        at_time: datetime,
        direction: str = "both",
        group_id: str = "default",
    ) -> list[Relationship]:
        """Return relationships active at a specific point in time."""
        conditions: list[str] = []
        params: list = []
        if direction == "outgoing":
            conditions.append("source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_id = ?")
            params.append(entity_id)
        else:
            conditions.append("(source_id = ? OR target_id = ?)")
            params.extend([entity_id, entity_id])

        t = at_time.isoformat()
        conditions.append("(valid_from IS NULL OR valid_from <= ?)")
        params.append(t)
        conditions.append("(valid_to IS NULL OR valid_to > ?)")
        params.append(t)
        conditions.append("group_id = ?")
        params.append(group_id)

        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT * FROM relationships WHERE {where}",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        group_id: str | None = None,
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops using recursive CTE."""
        group_filter = "AND e.group_id = ?" if group_id else ""
        params: list = [entity_id, hops]
        if group_id:
            params.append(group_id)

        sql = f"""
        WITH RECURSIVE neighbors(entity_id, depth) AS (
            SELECT ?, 0
            UNION ALL
            SELECT
                CASE WHEN r.source_id = n.entity_id THEN r.target_id ELSE r.source_id END,
                n.depth + 1
            FROM neighbors n
            JOIN relationships r ON (r.source_id = n.entity_id OR r.target_id = n.entity_id)
            WHERE n.depth < ?
              AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
              AND r.source_id != r.target_id
        )
        SELECT DISTINCT e.*, r.id as rel_id, r.source_id, r.target_id,
               r.predicate, r.weight, r.valid_from, r.valid_to,
               r.created_at as rel_created_at, r.source_episode, r.group_id as rel_group_id,
               r.confidence as rel_confidence, r.polarity as rel_polarity
        FROM neighbors n
        JOIN entities e ON e.id = n.entity_id
        JOIN relationships r ON (
            (r.source_id = n.entity_id OR r.target_id = n.entity_id)
            AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
        )
        WHERE e.deleted_at IS NULL AND n.depth > 0
        {group_filter}
        """
        cursor = await self.db.execute(sql, params)
        rows = await cursor.fetchall()

        results = []
        for row in rows:
            entity = self._row_to_entity(row, group_id)
            rel_polarity = "positive"
            if "rel_polarity" in row.keys() and row["rel_polarity"]:
                rel_polarity = row["rel_polarity"]
            rel = Relationship(
                id=row["rel_id"],
                source_id=row["source_id"],
                target_id=row["target_id"],
                predicate=row["predicate"],
                weight=row["weight"],
                valid_from=_parse_dt(row["valid_from"]),
                valid_to=_parse_dt(row["valid_to"]),
                created_at=_parse_dt(row["rel_created_at"]) or datetime.utcnow(),
                confidence=row["rel_confidence"] if "rel_confidence" in row.keys() else 1.0,
                polarity=rel_polarity,
                source_episode=row["source_episode"],
                group_id=row["rel_group_id"],
            )
            results.append((entity, rel))
        return results

    async def get_all_edges(
        self,
        group_id: str,
        entity_ids: set[str] | None = None,
        limit: int = 10000,
    ) -> list[Relationship]:
        """Return all active edges for a group, optionally filtered to a set of entity IDs."""
        conditions = [
            "(valid_to IS NULL OR datetime(valid_to) > datetime('now'))",
            "group_id = ?",
        ]
        params: list = [group_id]

        if entity_ids is not None:
            placeholders = ",".join("?" for _ in entity_ids)
            conditions.append(f"source_id IN ({placeholders})")
            conditions.append(f"target_id IN ({placeholders})")
            params.extend(entity_ids)
            params.extend(entity_ids)

        params.append(limit)
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"""SELECT id, source_id, target_id, predicate, weight,
                       valid_from, valid_to, created_at, source_episode,
                       group_id, confidence, polarity
                FROM relationships
                WHERE {where}
                LIMIT ?""",
            params,
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            polarity = "positive"
            if "polarity" in row.keys() and row["polarity"]:
                polarity = row["polarity"]
            results.append(
                Relationship(
                    id=row["id"],
                    source_id=row["source_id"],
                    target_id=row["target_id"],
                    predicate=row["predicate"],
                    weight=row["weight"],
                    valid_from=_parse_dt(row["valid_from"]),
                    valid_to=_parse_dt(row["valid_to"]),
                    created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
                    confidence=row["confidence"] if "confidence" in row.keys() else 1.0,
                    polarity=polarity,
                    source_episode=row["source_episode"],
                    group_id=row["group_id"],
                )
            )
        return results

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str, str]]:
        """Return (neighbor_id, edge_weight, predicate, entity_type) for active relationships.

        Negative-polarity edges are excluded. Uncertain edges have weight halved.
        """
        conditions = [
            "(r.source_id = ? OR r.target_id = ?)",
            "(r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))",
            "r.source_id != r.target_id",
            "COALESCE(r.polarity, 'positive') != 'negative'",
        ]
        params: list = [entity_id, entity_id]
        if group_id:
            conditions.append("r.group_id = ?")
            params.append(group_id)
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"""SELECT
                    CASE WHEN r.source_id = ?
                         THEN r.target_id
                         ELSE r.source_id END AS neighbor_id,
                    CASE WHEN COALESCE(r.polarity, 'positive') = 'uncertain'
                         THEN r.weight * 0.5
                         ELSE r.weight END AS weight,
                    r.predicate,
                    e.entity_type
                FROM relationships r
                JOIN entities e ON e.id = (
                    CASE WHEN r.source_id = ?
                         THEN r.target_id
                         ELSE r.source_id END)
                WHERE {where}""",
            [entity_id, entity_id] + params,
        )
        rows = await cursor.fetchall()
        return [
            (row["neighbor_id"], row["weight"], row["predicate"], row["entity_type"])
            for row in rows
        ]

    # --- Episodes ---

    async def create_episode(self, episode: Episode) -> str:
        content = self._encrypt(episode.group_id, episode.content)
        now_iso = datetime.utcnow().isoformat()
        await self.db.execute(
            """INSERT INTO episodes
               (id, content, source, status, group_id, session_id, created_at,
                updated_at, error, retry_count, processing_duration_ms,
                encoding_context, memory_tier, consolidation_cycles,
                entity_coverage)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id,
                content,
                episode.source,
                episode.status.value if hasattr(episode.status, "value") else episode.status,
                episode.group_id,
                episode.session_id,
                episode.created_at.isoformat() if episode.created_at else now_iso,
                episode.updated_at.isoformat() if episode.updated_at else now_iso,
                episode.error,
                episode.retry_count,
                episode.processing_duration_ms,
                episode.encoding_context,
                episode.memory_tier,
                episode.consolidation_cycles,
                episode.entity_coverage,
            ),
        )
        await self.db.commit()
        return episode.id

    async def update_episode(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        """Update episode fields (status, error, retry_count, etc.)."""
        if not updates:
            return
        updates["updated_at"] = datetime.utcnow().isoformat()
        invalid = set(updates.keys()) - EPISODE_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed episode update fields: {invalid}")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [episode_id, group_id]
        await self.db.execute(
            f"UPDATE episodes SET {set_clause} WHERE id = ? AND group_id = ?",
            values,
        )
        await self.db.commit()

    async def get_episodes(
        self,
        group_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Episode]:
        if group_id:
            cursor = await self.db.execute(
                "SELECT * FROM episodes WHERE group_id = ?"
                " ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (group_id, limit, offset),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM episodes ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r, group_id) for r in rows]

    async def link_episode_entity(self, episode_id: str, entity_id: str) -> None:
        await self.db.execute(
            "INSERT OR IGNORE INTO episode_entities (episode_id, entity_id) VALUES (?, ?)",
            (episode_id, entity_id),
        )
        await self.db.commit()

    async def get_episode_by_id(self, episode_id: str, group_id: str) -> Episode | None:
        """Fetch a single episode by ID and group."""
        cursor = await self.db.execute(
            "SELECT * FROM episodes WHERE id = ? AND group_id = ?",
            (episode_id, group_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_episode(row, group_id)

    async def get_episode_entities(self, episode_id: str) -> list[str]:
        """Return entity IDs linked to an episode."""
        cursor = await self.db.execute(
            "SELECT entity_id FROM episode_entities WHERE episode_id = ?",
            (episode_id,),
        )
        rows = await cursor.fetchall()
        return [row["entity_id"] for row in rows]

    async def get_episodes_paginated(
        self,
        group_id: str | None = None,
        cursor: str | None = None,
        limit: int = 50,
        source: str | None = None,
        status: str | None = None,
    ) -> tuple[list[Episode], str | None]:
        """Cursor-paginated episode listing. Returns (episodes, next_cursor)."""
        conditions: list[str] = []
        params: list = []

        if group_id:
            conditions.append("group_id = ?")
            params.append(group_id)
        if source:
            conditions.append("source = ?")
            params.append(source)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if cursor:
            conditions.append("created_at < ?")
            params.append(cursor)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit + 1)  # Fetch one extra to detect next page

        sql = f"SELECT * FROM episodes WHERE {where} ORDER BY created_at DESC LIMIT ?"
        rows_cursor = await self.db.execute(sql, params)
        rows = await rows_cursor.fetchall()

        episodes = [self._row_to_episode(r, group_id) for r in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            next_cursor = episodes[-1].created_at.isoformat()

        return episodes, next_cursor

    async def get_top_connected(self, group_id: str | None = None, limit: int = 10) -> list[dict]:
        """Return entities sorted by number of active edges (degree)."""
        group_filter = "AND r.group_id = ?" if group_id else ""
        entity_group_filter = "AND e.group_id = ?" if group_id else ""
        params: list = []
        if group_id:
            params.append(group_id)
        if group_id:
            params.append(group_id)
        params.append(limit)

        sql = f"""
        SELECT e.id, e.name, e.entity_type,
               COUNT(r.id) AS edge_count
        FROM entities e
        LEFT JOIN relationships r
            ON (r.source_id = e.id OR r.target_id = e.id)
            AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
            {group_filter}
        WHERE e.deleted_at IS NULL
            {entity_group_filter}
        GROUP BY e.id
        ORDER BY edge_count DESC
        LIMIT ?
        """
        cursor = await self.db.execute(sql, params)
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "entityType": row["entity_type"],
                "edgeCount": row["edge_count"],
            }
            for row in rows
        ]

    async def get_growth_timeline(self, group_id: str | None = None, days: int = 30) -> list[dict]:
        """Return daily entity and episode counts for the last N days."""
        group_filter = "WHERE group_id = ?" if group_id else ""
        entity_group_filter = (
            "WHERE deleted_at IS NULL AND group_id = ?" if group_id else "WHERE deleted_at IS NULL"
        )

        params_ep: list = []
        params_ent: list = []
        if group_id:
            params_ep.append(group_id)
            params_ent.append(group_id)
        params_ep.append(days)
        params_ent.append(days)

        # Episodes per day
        ep_sql = f"""
        SELECT DATE(created_at) AS day, COUNT(*) AS count
        FROM episodes {group_filter}
        GROUP BY DATE(created_at)
        ORDER BY day DESC
        LIMIT ?
        """
        ep_cursor = await self.db.execute(ep_sql, params_ep)
        ep_rows = await ep_cursor.fetchall()
        ep_map = {row[0]: row[1] for row in ep_rows}

        # Entities per day
        ent_sql = f"""
        SELECT DATE(created_at) AS day, COUNT(*) AS count
        FROM entities {entity_group_filter}
        GROUP BY DATE(created_at)
        ORDER BY day DESC
        LIMIT ?
        """
        ent_cursor = await self.db.execute(ent_sql, params_ent)
        ent_rows = await ent_cursor.fetchall()
        ent_map = {row[0]: row[1] for row in ent_rows}

        all_days = sorted(set(list(ep_map.keys()) + list(ent_map.keys())), reverse=True)
        return [
            {
                "date": day,
                "episodes": ep_map.get(day, 0),
                "entities": ent_map.get(day, 0),
            }
            for day in all_days[:days]
        ]

    async def get_entity_type_counts(self, group_id: str | None = None) -> dict[str, int]:
        """Return entity counts grouped by entity_type."""
        if group_id:
            cursor = await self.db.execute(
                "SELECT entity_type, COUNT(*) FROM entities"
                " WHERE group_id = ? AND deleted_at IS NULL GROUP BY entity_type",
                (group_id,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT entity_type, COUNT(*) FROM entities"
                " WHERE deleted_at IS NULL GROUP BY entity_type",
            )
        return {row[0]: row[1] for row in await cursor.fetchall()}

    # --- Consolidation methods ---

    async def get_co_occurring_entity_pairs(
        self,
        group_id: str,
        since: datetime | None = None,
        min_co_occurrence: int = 3,
        limit: int = 100,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs that co-occur in multiple episodes.

        Returns (entity_id_a, entity_id_b, count) tuples, excluding pairs
        that already have a direct relationship.
        """
        conditions = ["e1.group_id = ?"]
        params: list = [group_id]
        if since:
            conditions.append("ep.created_at >= ?")
            params.append(since.isoformat())
        where = " AND ".join(conditions)
        params.extend([min_co_occurrence, limit])

        sql = f"""
        SELECT ee1.entity_id AS a, ee2.entity_id AS b,
               COUNT(DISTINCT ee1.episode_id) AS cnt
        FROM episode_entities ee1
        JOIN episode_entities ee2
            ON ee1.episode_id = ee2.episode_id
            AND ee1.entity_id < ee2.entity_id
        JOIN entities e1 ON e1.id = ee1.entity_id AND e1.deleted_at IS NULL
        JOIN entities e2 ON e2.id = ee2.entity_id AND e2.deleted_at IS NULL
        JOIN episodes ep ON ep.id = ee1.episode_id
        WHERE {where}
        GROUP BY ee1.entity_id, ee2.entity_id
        HAVING cnt >= ?
        AND NOT EXISTS (
            SELECT 1 FROM relationships r
            WHERE ((r.source_id = ee1.entity_id AND r.target_id = ee2.entity_id)
                OR (r.source_id = ee2.entity_id AND r.target_id = ee1.entity_id))
            AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
            AND r.group_id = e1.group_id
        )
        ORDER BY cnt DESC
        LIMIT ?
        """
        cursor = await self.db.execute(sql, params)
        rows = await cursor.fetchall()
        return [(row[0], row[1], row[2]) for row in rows]

    async def get_entity_episode_counts(
        self,
        group_id: str,
        entity_ids: list[str],
    ) -> dict[str, int]:
        """Return how many episodes each entity appears in."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        sql = f"""
        SELECT ee.entity_id, COUNT(DISTINCT ee.episode_id) AS ep_count
        FROM episode_entities ee
        JOIN episodes ep ON ep.id = ee.episode_id
        WHERE ep.group_id = ? AND ee.entity_id IN ({placeholders})
        GROUP BY ee.entity_id
        """
        cursor = await self.db.execute(sql, [group_id, *entity_ids])
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    async def get_dead_entities(
        self,
        group_id: str,
        min_age_days: int = 30,
        limit: int = 100,
        max_access_count: int = 0,
    ) -> list[Entity]:
        """Find entities with no relationships and low access."""
        cutoff = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta

        cutoff = (cutoff - timedelta(days=min_age_days)).isoformat()

        sql = """
        SELECT e.*
        FROM entities e
        LEFT JOIN relationships r ON (r.source_id = e.id OR r.target_id = e.id)
            AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now')) AND r.group_id = ?
        WHERE e.group_id = ?
          AND e.deleted_at IS NULL
          AND e.access_count <= ?
          AND e.created_at < ?
          AND COALESCE(e.identity_core, 0) = 0
          AND r.id IS NULL
        ORDER BY e.access_count ASC, e.created_at ASC
        LIMIT ?
        """
        cursor = await self.db.execute(
            sql, (group_id, group_id, max_access_count, cutoff, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entity(r, group_id) for r in rows]

    async def get_identity_core_entities(self, group_id: str) -> list[Entity]:
        """Return all entities marked as identity_core for a group."""
        cursor = await self.db.execute(
            """SELECT * FROM entities
               WHERE group_id = ? AND deleted_at IS NULL
                 AND identity_core = 1""",
            (group_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entity(r, group_id) for r in rows]

    async def get_entity_episode_count(self, entity_id: str, group_id: str) -> int:
        """Count episodes that mention this entity."""
        cursor = await self.db.execute(
            """SELECT COUNT(*) FROM episode_entities ee
               JOIN episodes e ON e.id = ee.episode_id
               WHERE ee.entity_id = ? AND e.group_id = ?""",
            (entity_id, group_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_entity_temporal_span(
        self, entity_id: str, group_id: str,
    ) -> tuple[str | None, str | None]:
        """Return (min_created_at, max_created_at) for episodes mentioning this entity."""
        cursor = await self.db.execute(
            """SELECT MIN(e.created_at), MAX(e.created_at)
               FROM episode_entities ee
               JOIN episodes e ON e.id = ee.episode_id
               WHERE ee.entity_id = ? AND e.group_id = ?""",
            (entity_id, group_id),
        )
        row = await cursor.fetchone()
        if row:
            return (row[0], row[1])
        return (None, None)

    async def get_entity_relationship_types(
        self, entity_id: str, group_id: str,
    ) -> list[str]:
        """Return distinct predicates connected to this entity."""
        cursor = await self.db.execute(
            """SELECT DISTINCT predicate FROM relationships
               WHERE (source_id = ? OR target_id = ?) AND group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))""",
            (entity_id, entity_id, group_id),
        )
        rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def merge_entities(
        self,
        keep_id: str,
        remove_id: str,
        group_id: str,
    ) -> int:
        """Merge remove_id into keep_id: re-point relationships, merge summaries, soft-delete loser.

        Returns count of relationships transferred.
        """
        # Re-point outgoing relationships
        await self.db.execute(
            "UPDATE relationships SET source_id = ? "
            "WHERE source_id = ? AND group_id = ? AND target_id != ?",
            (keep_id, remove_id, group_id, keep_id),
        )
        # Re-point incoming relationships
        await self.db.execute(
            "UPDATE relationships SET target_id = ? "
            "WHERE target_id = ? AND group_id = ? AND source_id != ?",
            (keep_id, remove_id, group_id, keep_id),
        )
        # Delete self-loops that may have been created
        await self.db.execute(
            "DELETE FROM relationships WHERE source_id = ? AND target_id = ? AND group_id = ?",
            (keep_id, keep_id, group_id),
        )
        # Collapse duplicate (source_id, target_id, predicate) triples — keep oldest
        await self.db.execute(
            """DELETE FROM relationships WHERE id NOT IN (
                   SELECT MIN(id) FROM relationships
                   WHERE (source_id = ? OR target_id = ?) AND group_id = ?
                     AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))
                   GROUP BY source_id, target_id, predicate
               ) AND (source_id = ? OR target_id = ?) AND group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))""",
            (keep_id, keep_id, group_id, keep_id, keep_id, group_id),
        )
        # Count transferred relationships
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM relationships "
            "WHERE (source_id = ? OR target_id = ?) AND group_id = ?",
            (keep_id, keep_id, group_id),
        )
        total_count = (await cursor.fetchone())[0]

        # Re-point episode_entities (ignore conflicts for already-linked episodes)
        await self.db.execute(
            "UPDATE OR IGNORE episode_entities SET entity_id = ? WHERE entity_id = ?",
            (keep_id, remove_id),
        )
        # Delete any remaining episode_entities for the loser
        await self.db.execute(
            "DELETE FROM episode_entities WHERE entity_id = ?",
            (remove_id,),
        )

        # Merge summaries: append loser's summary to keeper's
        keep_row = await self.db.execute(
            "SELECT summary, access_count FROM entities WHERE id = ? AND group_id = ?",
            (keep_id, group_id),
        )
        keep_data = await keep_row.fetchone()
        remove_row = await self.db.execute(
            "SELECT summary, access_count FROM entities WHERE id = ? AND group_id = ?",
            (remove_id, group_id),
        )
        remove_data = await remove_row.fetchone()

        if keep_data and remove_data:
            keep_summary = keep_data["summary"] or ""
            remove_summary = remove_data["summary"] or ""
            merged_count = keep_data["access_count"] + remove_data["access_count"]
            if remove_summary and remove_summary not in keep_summary:
                if is_meta_summary(remove_summary):
                    logger.warning(
                        "Rejected meta-contaminated summary during merge into %s: %s",
                        keep_id, remove_summary[:80],
                    )
                else:
                    merged = f"{keep_summary} {remove_summary}".strip()
                    if len(merged) > 500:
                        merged = merged[:497] + "..."
                    keep_summary = merged
            await self.db.execute(
                "UPDATE entities SET summary = ?, access_count = ?, updated_at = ? "
                "WHERE id = ? AND group_id = ?",
                (keep_summary, merged_count, datetime.utcnow().isoformat(), keep_id, group_id),
            )

        # Soft-delete the loser
        await self.db.execute(
            "UPDATE entities SET deleted_at = ? WHERE id = ? AND group_id = ?",
            (datetime.utcnow().isoformat(), remove_id, group_id),
        )
        await self.db.commit()
        return total_count

    async def path_exists_within_hops(
        self,
        source_id: str,
        target_id: str,
        max_hops: int,
        group_id: str,
    ) -> bool:
        """Check if a path exists between two entities within N hops."""
        sql = """
        WITH RECURSIVE reachable(entity_id, depth) AS (
            SELECT ?, 0
            UNION ALL
            SELECT
                CASE WHEN r.source_id = rc.entity_id THEN r.target_id ELSE r.source_id END,
                rc.depth + 1
            FROM reachable rc
            JOIN relationships r ON (r.source_id = rc.entity_id OR r.target_id = rc.entity_id)
            WHERE rc.depth < ?
              AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
              AND r.group_id = ?
              AND r.source_id != r.target_id
        )
        SELECT 1 FROM reachable WHERE entity_id = ? LIMIT 1
        """
        cursor = await self.db.execute(sql, (source_id, max_hops, group_id, target_id))
        row = await cursor.fetchone()
        return row is not None

    async def get_expired_relationships(
        self,
        group_id: str,
        predicate: str | None = None,
        limit: int = 100,
    ) -> list[Relationship]:
        """Return relationships whose valid_to has passed."""
        conditions = [
            "group_id = ?",
            "valid_to IS NOT NULL",
            "datetime(valid_to) <= datetime('now')",
        ]
        params: list = [group_id]
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        params.append(limit)
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT * FROM relationships WHERE {where} LIMIT ?",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def get_relationships_by_predicate(
        self,
        group_id: str,
        predicate: str,
        active_only: bool = True,
        limit: int = 10000,
    ) -> list[Relationship]:
        """Fetch relationships matching a specific predicate."""
        conditions = ["group_id = ?", "predicate = ?"]
        params: list = [group_id, predicate]
        if active_only:
            conditions.append("(valid_to IS NULL OR datetime(valid_to) > datetime('now'))")
        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT * FROM relationships WHERE {where} LIMIT ?",
            (*params, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def get_stats(self, group_id: str | None = None) -> dict:
        if group_id:
            entities = await self.db.execute(
                "SELECT COUNT(*) FROM entities WHERE group_id = ? AND deleted_at IS NULL",
                (group_id,),
            )
            rels = await self.db.execute(
                "SELECT COUNT(*) FROM relationships WHERE group_id = ?", (group_id,)
            )
            eps = await self.db.execute(
                "SELECT COUNT(*) FROM episodes WHERE group_id = ?", (group_id,)
            )
        else:
            entities = await self.db.execute(
                "SELECT COUNT(*) FROM entities WHERE deleted_at IS NULL"
            )
            rels = await self.db.execute("SELECT COUNT(*) FROM relationships")
            eps = await self.db.execute("SELECT COUNT(*) FROM episodes")

        return {
            "entities": (await entities.fetchone())[0],
            "relationships": (await rels.fetchone())[0],
            "episodes": (await eps.fetchone())[0],
        }

    # --- Helpers ---

    def _row_to_entity(self, row, group_id: str | None = None) -> Entity:
        row_group = row["group_id"]
        decrypt_group = group_id or row_group
        summary = self._decrypt(decrypt_group, row["summary"])

        pii_detected = False
        pii_categories = None
        if "pii_detected" in row.keys():
            pii_detected = bool(row["pii_detected"])
        if "pii_categories" in row.keys() and row["pii_categories"]:
            pii_categories = json.loads(row["pii_categories"])

        identity_core = False
        if "identity_core" in row.keys():
            identity_core = bool(row["identity_core"])

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            summary=summary,
            attributes=json.loads(row["attributes"]) if row["attributes"] else None,
            group_id=row_group,
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
            deleted_at=_parse_dt(row["deleted_at"]),
            activation_current=(
                row["activation_current"] if "activation_current" in row.keys() else 0.0
            ),
            access_count=row["access_count"],
            last_accessed=_parse_dt(row["last_accessed"]),
            pii_detected=pii_detected,
            pii_categories=pii_categories,
            identity_core=identity_core,
        )

    @staticmethod
    def _row_to_relationship(row) -> Relationship:
        confidence = 1.0
        if "confidence" in row.keys():
            confidence = row["confidence"] if row["confidence"] is not None else 1.0

        polarity = "positive"
        if "polarity" in row.keys() and row["polarity"]:
            polarity = row["polarity"]

        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            predicate=row["predicate"],
            weight=row["weight"],
            valid_from=_parse_dt(row["valid_from"]),
            valid_to=_parse_dt(row["valid_to"]),
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            confidence=confidence,
            polarity=polarity,
            source_episode=row["source_episode"],
            group_id=row["group_id"],
        )

    # --- Schema Formation (Brain Architecture Phase 3) ---

    async def get_schema_members(
        self, schema_entity_id: str, group_id: str,
    ) -> list[dict]:
        """Fetch schema member definitions for a schema entity."""
        cursor = await self.db.execute(
            """SELECT role_label, member_type, member_predicate
               FROM schema_members
               WHERE schema_entity_id = ? AND group_id = ?""",
            (schema_entity_id, group_id),
        )
        rows = await cursor.fetchall()
        return [
            {
                "role_label": r["role_label"],
                "member_type": r["member_type"],
                "member_predicate": r["member_predicate"],
            }
            for r in rows
        ]

    async def save_schema_members(
        self, schema_entity_id: str, members: list[dict], group_id: str,
    ) -> None:
        """Insert or replace schema member rows."""
        for m in members:
            await self.db.execute(
                """INSERT OR REPLACE INTO schema_members
                   (schema_entity_id, role_label, member_type, member_predicate, group_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    schema_entity_id,
                    m["role_label"],
                    m["member_type"],
                    m["member_predicate"],
                    group_id,
                ),
            )
        await self.db.commit()

    async def find_entities_by_type(
        self, entity_type: str, group_id: str, limit: int = 100,
    ) -> list[Entity]:
        """Return non-deleted entities of a specific type."""
        cursor = await self.db.execute(
            """SELECT * FROM entities
               WHERE entity_type = ? AND group_id = ? AND deleted_at IS NULL
               LIMIT ?""",
            (entity_type, group_id, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_entity(r, group_id) for r in rows]

    # --- Prospective memory (Wave 4) ---

    async def create_intention(self, intention: object) -> str:
        """Store a new intention."""
        from engram.models.prospective import Intention

        i: Intention = intention  # type: ignore[assignment]
        await self.db.execute(
            """INSERT INTO intentions
               (id, trigger_text, action_text, trigger_type, entity_name,
                threshold, max_fires, fire_count, enabled, group_id,
                created_at, updated_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                i.id, i.trigger_text, i.action_text, i.trigger_type,
                i.entity_name, i.threshold, i.max_fires, i.fire_count,
                1 if i.enabled else 0, i.group_id,
                i.created_at.isoformat(), i.updated_at.isoformat(),
                i.expires_at.isoformat() if i.expires_at else None,
            ),
        )
        await self.db.commit()
        return i.id

    async def get_intention(self, id: str, group_id: str) -> object | None:
        """Get a single intention by ID and group."""
        cursor = await self.db.execute(
            "SELECT * FROM intentions WHERE id = ? AND group_id = ?",
            (id, group_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_intention(row)

    async def list_intentions(
        self, group_id: str, enabled_only: bool = True,
    ) -> list:
        """List intentions for a group, filtering expired and optionally disabled."""
        if enabled_only:
            cursor = await self.db.execute(
                """SELECT * FROM intentions
                   WHERE group_id = ? AND enabled = 1
                     AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
                   ORDER BY created_at DESC""",
                (group_id,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM intentions WHERE group_id = ? ORDER BY created_at DESC",
                (group_id,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_intention(r) for r in rows]

    async def update_intention(
        self, id: str, updates: dict, group_id: str,
    ) -> None:
        """Update intention fields."""
        allowed = {"trigger_text", "action_text", "threshold", "max_fires", "enabled", "expires_at"}
        parts = []
        values = []
        for key, val in updates.items():
            if key not in allowed:
                continue
            if key == "enabled":
                val = 1 if val else 0
            parts.append(f"{key} = ?")
            values.append(val)
        if not parts:
            return
        parts.append("updated_at = ?")
        values.append(datetime.utcnow().isoformat())
        values.extend([id, group_id])
        sql = f"UPDATE intentions SET {', '.join(parts)} WHERE id = ? AND group_id = ?"
        await self.db.execute(sql, values)
        await self.db.commit()

    async def delete_intention(
        self, id: str, group_id: str, soft: bool = True,
    ) -> None:
        """Delete an intention (soft = disable, hard = remove)."""
        if soft:
            await self.db.execute(
                "UPDATE intentions SET enabled = 0, updated_at = ? WHERE id = ? AND group_id = ?",
                (datetime.utcnow().isoformat(), id, group_id),
            )
        else:
            await self.db.execute(
                "DELETE FROM intentions WHERE id = ? AND group_id = ?",
                (id, group_id),
            )
        await self.db.commit()

    async def increment_intention_fire_count(
        self, id: str, group_id: str,
    ) -> None:
        """Increment fire_count by 1."""
        await self.db.execute(
            """UPDATE intentions
               SET fire_count = fire_count + 1, updated_at = ?
               WHERE id = ? AND group_id = ?""",
            (datetime.utcnow().isoformat(), id, group_id),
        )
        await self.db.commit()

    def _row_to_intention(self, row) -> object:
        """Convert a database row to an Intention model."""
        from engram.models.prospective import Intention

        expires_at = None
        if row["expires_at"]:
            expires_at = _parse_dt(row["expires_at"])

        return Intention(
            id=row["id"],
            trigger_text=row["trigger_text"],
            action_text=row["action_text"],
            trigger_type=row["trigger_type"],
            entity_name=row["entity_name"],
            threshold=row["threshold"],
            max_fires=row["max_fires"],
            fire_count=row["fire_count"],
            enabled=bool(row["enabled"]),
            group_id=row["group_id"],
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=_parse_dt(row["updated_at"]) or datetime.utcnow(),
            expires_at=expires_at,
        )

    def _row_to_episode(self, row, group_id: str | None = None) -> Episode:
        row_group = row["group_id"]
        decrypt_group = group_id or row_group
        content = self._decrypt(decrypt_group, row["content"])
        keys = row.keys()

        return Episode(
            id=row["id"],
            content=content,
            source=row["source"],
            status=row["status"],
            group_id=row_group,
            session_id=row["session_id"] if "session_id" in keys else None,
            created_at=_parse_dt(row["created_at"]) or datetime.utcnow(),
            updated_at=_parse_dt(row["updated_at"]) if "updated_at" in keys else None,
            error=row["error"] if "error" in keys else None,
            retry_count=(
                row["retry_count"]
                if "retry_count" in keys and row["retry_count"] is not None
                else 0
            ),
            processing_duration_ms=(
                row["processing_duration_ms"] if "processing_duration_ms" in keys else None
            ),
            encoding_context=(
                row["encoding_context"] if "encoding_context" in keys else None
            ),
            memory_tier=(
                row["memory_tier"] if "memory_tier" in keys and row["memory_tier"] else "episodic"
            ),
            consolidation_cycles=(
                row["consolidation_cycles"]
                if "consolidation_cycles" in keys and row["consolidation_cycles"] is not None
                else 0
            ),
            entity_coverage=(
                row["entity_coverage"]
                if "entity_coverage" in keys and row["entity_coverage"] is not None
                else 0.0
            ),
        )


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
