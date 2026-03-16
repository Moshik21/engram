"""SQLite implementation of GraphStore protocol (Lite mode)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import aiosqlite

from engram.entity_dedup_policy import NameRegime, analyze_name, entity_identifier_facets
from engram.models.entity import Entity
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship
from engram.storage.protocols import ENTITY_UPDATABLE_FIELDS, EPISODE_UPDATABLE_FIELDS
from engram.utils.dates import utc_now, utc_now_iso
from engram.utils.text_guards import is_meta_summary

logger = logging.getLogger(__name__)


def _row_value(row: aiosqlite.Row | None, key: str | int, default: Any = None) -> Any:
    if row is None:
        return default
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _evidence_row_to_dict(row: aiosqlite.Row | tuple[Any, ...]) -> dict:
    """Convert a raw evidence row tuple to a dictionary."""
    return {
        "evidence_id": row[0],
        "episode_id": row[1],
        "group_id": row[2],
        "fact_class": row[3],
        "confidence": row[4],
        "source_type": row[5],
        "extractor_name": row[6],
        "payload": json.loads(row[7]) if row[7] else {},
        "source_span": row[8],
        "corroborating_signals": json.loads(row[9]) if row[9] else [],
        "ambiguity_tags": json.loads(row[10]) if row[10] else [],
        "ambiguity_score": row[11] or 0.0,
        "adjudication_request_id": row[12],
        "status": row[13],
        "commit_reason": row[14],
        "committed_id": row[15],
        "deferred_cycles": row[16],
        "created_at": row[17],
        "resolved_at": row[18],
    }


def _adjudication_row_to_dict(row: aiosqlite.Row | tuple[Any, ...]) -> dict:
    """Convert a raw adjudication request row tuple to a dictionary."""
    return {
        "request_id": row[0],
        "episode_id": row[1],
        "group_id": row[2],
        "status": row[3],
        "ambiguity_tags": json.loads(row[4]) if row[4] else [],
        "evidence_ids": json.loads(row[5]) if row[5] else [],
        "selected_text": row[6] or "",
        "request_reason": row[7] or "",
        "resolution_source": row[8],
        "resolution_payload": json.loads(row[9]) if row[9] else None,
        "attempt_count": row[10] or 0,
        "created_at": row[11],
        "resolved_at": row[12],
    }


def _dedup_summaries(existing: str, incoming: str, max_len: int = 500) -> str:
    """Merge two summaries, removing duplicate sentences via token-set Jaccard."""
    import re

    # Split into sentences
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip() and len(s.strip()) > 5]

    existing_sents = _sentences(existing)
    incoming_sents = _sentences(incoming)

    if not incoming_sents:
        return existing.strip()

    # Build token sets for existing sentences
    def _tokens(sent: str) -> set[str]:
        return {w.lower() for w in re.findall(r"\b\w{3,}\b", sent)}

    existing_token_sets = [_tokens(s) for s in existing_sents]

    # Only add incoming sentences that are sufficiently novel
    novel = []
    for sent in incoming_sents:
        sent_tokens = _tokens(sent)
        if not sent_tokens:
            continue
        is_dup = False
        for etokens in existing_token_sets:
            if not etokens:
                continue
            intersection = sent_tokens & etokens
            union = sent_tokens | etokens
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= 0.6:  # 60% overlap = duplicate
                is_dup = True
                break
        if not is_dup:
            novel.append(sent)
            existing_token_sets.append(sent_tokens)

    if not novel:
        result = existing.strip()
    else:
        novel_text = ". ".join(novel)
        result = f"{existing.strip()} {novel_text}".strip()

    if len(result) > max_len:
        result = result[: max_len - 3] + "..."
    return result


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
            "ALTER TABLE episodes ADD COLUMN projection_state TEXT DEFAULT 'queued'",
            "ALTER TABLE episodes ADD COLUMN last_projection_reason TEXT",
            "ALTER TABLE episodes ADD COLUMN last_projected_at TEXT",
            # Identity core
            "ALTER TABLE entities ADD COLUMN identity_core INTEGER DEFAULT 0",
            # Negation/uncertainty polarity
            "ALTER TABLE relationships ADD COLUMN polarity TEXT DEFAULT 'positive'",
            "ALTER TABLE episode_cues ADD COLUMN surfaced_count INTEGER DEFAULT 0",
            "ALTER TABLE episode_cues ADD COLUMN selected_count INTEGER DEFAULT 0",
            "ALTER TABLE episode_cues ADD COLUMN used_count INTEGER DEFAULT 0",
            "ALTER TABLE episode_cues ADD COLUMN near_miss_count INTEGER DEFAULT 0",
            "ALTER TABLE episode_cues ADD COLUMN policy_score REAL DEFAULT 0.0",
            "ALTER TABLE episode_cues ADD COLUMN last_feedback_at TEXT",
            # Identifier-aware entity facets
            "ALTER TABLE entities ADD COLUMN lexical_regime TEXT",
            "ALTER TABLE entities ADD COLUMN canonical_identifier TEXT",
            "ALTER TABLE entities ADD COLUMN identifier_label INTEGER DEFAULT 0",
            # Evidence v3 ambiguity metadata
            "ALTER TABLE episode_evidence ADD COLUMN ambiguity_tags_json "
            "TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE episode_evidence ADD COLUMN ambiguity_score REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE episode_evidence ADD COLUMN adjudication_request_id TEXT",
            "ALTER TABLE episodes ADD COLUMN conversation_date TEXT",
            "ALTER TABLE episodes ADD COLUMN attachments_json TEXT DEFAULT '[]'",
        ]
        for sql in migrations:
            try:
                await self._db.execute(sql)
            except Exception:
                pass  # Column already exists
        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_entities_lexical_regime ON entities(lexical_regime)",
            "CREATE INDEX IF NOT EXISTS idx_entities_canonical_identifier "
            "ON entities(canonical_identifier)",
            "CREATE INDEX IF NOT EXISTS idx_evidence_adjudication_request "
            "ON episode_evidence(adjudication_request_id)",
            """CREATE TABLE IF NOT EXISTS episode_adjudications (
                   request_id TEXT PRIMARY KEY,
                   episode_id TEXT NOT NULL REFERENCES episodes(id),
                   group_id TEXT NOT NULL DEFAULT 'default',
                   status TEXT NOT NULL DEFAULT 'pending',
                   ambiguity_tags_json TEXT NOT NULL DEFAULT '[]',
                   evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                   selected_text TEXT NOT NULL DEFAULT '',
                   request_reason TEXT NOT NULL DEFAULT '',
                   resolution_source TEXT,
                   resolution_payload_json TEXT,
                   attempt_count INTEGER NOT NULL DEFAULT 0,
                   created_at TEXT NOT NULL,
                   resolved_at TEXT
               )""",
            "CREATE INDEX IF NOT EXISTS idx_episode_adjudications_episode "
            "ON episode_adjudications(episode_id)",
            "CREATE INDEX IF NOT EXISTS idx_episode_adjudications_status "
            "ON episode_adjudications(status)",
        ]:
            try:
                await self._db.execute(sql)
            except Exception:
                pass
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
        return cast(str | None, self._encryptor.encrypt(group_id, plaintext))

    def _decrypt(self, group_id: str, data: str | None) -> str | None:
        if not data or not self._encryptor:
            return data
        return cast(str | None, self._encryptor.decrypt(group_id, data))

    # --- Entities ---

    async def create_entity(self, entity: Entity) -> str:
        now = utc_now_iso()
        summary = self._encrypt(entity.group_id, entity.summary)
        await self.db.execute(
            """INSERT INTO entities
               (id, name, entity_type, summary, attributes, group_id,
                created_at, updated_at,
                access_count, last_accessed, pii_detected, pii_categories,
                lexical_regime, canonical_identifier, identifier_label,
                source_episode_ids, evidence_count,
                evidence_span_start, evidence_span_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                entity.lexical_regime,
                entity.canonical_identifier,
                1 if entity.identifier_label else 0,
                json.dumps(entity.source_episode_ids),
                entity.evidence_count,
                entity.evidence_span_start.isoformat() if entity.evidence_span_start else None,
                entity.evidence_span_end.isoformat() if entity.evidence_span_end else None,
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

    async def batch_get_entities(
        self,
        entity_ids: list[str],
        group_id: str,
    ) -> dict[str, Entity]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        cursor = await self.db.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders}) "
            f"AND group_id = ? AND deleted_at IS NULL",
            [*entity_ids, group_id],
        )
        rows = await cursor.fetchall()
        return {row["id"]: self._row_to_entity(row, group_id) for row in rows}

    async def update_entity(self, entity_id: str, updates: dict, group_id: str) -> None:
        if not updates:
            return
        if "name" in updates or "entity_type" in updates:
            current = await self.get_entity(entity_id, group_id)
            if current is not None:
                next_name = updates.get("name", current.name)
                facets = entity_identifier_facets(next_name)
                updates["lexical_regime"] = facets["lexical_regime"]
                updates["canonical_identifier"] = facets["canonical_identifier"]
                updates["identifier_label"] = 1 if facets["identifier_label"] else 0
        updates["updated_at"] = utc_now_iso()
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
                (utc_now_iso(), entity_id, group_id),
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

    async def delete_group(self, group_id: str) -> None:
        """Delete all data belonging to *group_id*.

        Deletion order respects foreign-key constraints:
        episode_entities → episode_cues → episodes → relationships →
        schema_members → graph_embeddings → episode_evidence →
        episode_adjudications → intentions → entities.
        """
        # Junction tables first (reference episodes / entities)
        await self.db.execute(
            """DELETE FROM episode_entities
               WHERE episode_id IN (SELECT id FROM episodes WHERE group_id = ?)""",
            (group_id,),
        )
        await self.db.execute(
            "DELETE FROM episode_cues WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM episodes WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM relationships WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM schema_members WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM graph_embeddings WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM episode_evidence WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM episode_adjudications WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM intentions WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM complement_tags WHERE group_id = ?", (group_id,)
        )
        await self.db.execute(
            "DELETE FROM entities WHERE group_id = ?", (group_id,)
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
        self,
        name: str,
        group_id: str,
        limit: int = 30,
    ) -> list[Entity]:
        """Retrieve candidate entities for fuzzy resolution via exact + FTS5 match."""
        seen_ids: set[str] = set()
        results: list[Entity] = []
        form = analyze_name(name)
        regime = form.regime

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

        # Phase 1.5: Exact canonical identifier match
        if form.canonical_code:
            cursor = await self.db.execute(
                """SELECT * FROM entities
                   WHERE canonical_identifier = ?
                     AND group_id = ? AND deleted_at IS NULL
                   LIMIT ?""",
                (form.canonical_code, group_id, limit - len(results)),
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

        # Phase 3: Prefix LIKE fallback (catches typos that FTS5 stemming misses)
        if (
            len(results) < limit
            and len(name.strip()) >= 3
            and regime == NameRegime.NATURAL_LANGUAGE
        ):
            prefix = name.strip()[:3].lower()
            remaining = limit - len(results)
            try:
                prefix_cursor = await self.db.execute(
                    """SELECT * FROM entities
                       WHERE LOWER(name) LIKE ? || '%'
                         AND group_id = ? AND deleted_at IS NULL
                       LIMIT ?""",
                    (prefix, group_id, remaining),
                )
                for row in await prefix_cursor.fetchall():
                    entity = self._row_to_entity(row, group_id)
                    if entity.id not in seen_ids:
                        seen_ids.add(entity.id)
                        results.append(entity)
            except Exception:
                pass

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
                rel.created_at.isoformat() if rel.created_at else utc_now_iso(),
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
        # Filter out edges to soft-deleted entities
        conditions.append("source_id NOT IN (SELECT id FROM entities WHERE deleted_at IS NOT NULL)")
        conditions.append("target_id NOT IN (SELECT id FROM entities WHERE deleted_at IS NOT NULL)")
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
        predicate: str | None = None,
    ) -> float | None:
        """Atomically increment edge weight, capped at max_weight."""
        predicate_clause = ""
        params: list[object] = [
            max_weight,
            weight_delta,
            group_id,
            source_id,
            target_id,
            target_id,
            source_id,
        ]
        if predicate is not None:
            predicate_clause = " AND predicate = ?"
            params.append(predicate)
        cursor = await self.db.execute(
            """UPDATE relationships
               SET weight = MIN(?, weight + ?)
               WHERE group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))
                 AND ((source_id = ? AND target_id = ?)
                  OR (source_id = ? AND target_id = ?))"""
            + predicate_clause
            + """
               RETURNING weight""",
            params,
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
            conditions.append("r.source_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("r.target_id = ?")
            params.append(entity_id)
        else:
            conditions.append("(r.source_id = ? OR r.target_id = ?)")
            params.extend([entity_id, entity_id])

        t = at_time.isoformat()
        conditions.append("(r.valid_from IS NULL OR r.valid_from <= ?)")
        params.append(t)
        conditions.append("(r.valid_to IS NULL OR r.valid_to > ?)")
        params.append(t)
        conditions.append("r.group_id = ?")
        params.append(group_id)

        where = " AND ".join(conditions)
        cursor = await self.db.execute(
            f"SELECT r.* FROM relationships r "
            f"JOIN entities es ON es.id = r.source_id AND es.deleted_at IS NULL "
            f"JOIN entities et ON et.id = r.target_id AND et.deleted_at IS NULL "
            f"WHERE {where}",
            params,
        )
        rows = await cursor.fetchall()
        return [self._row_to_relationship(r) for r in rows]

    async def get_neighbors(
        self,
        entity_id: str,
        hops: int = 1,
        group_id: str | None = None,
        max_results: int = 5000,
    ) -> list[tuple[Entity, Relationship]]:
        """Return entities within N hops using recursive CTE."""
        group_filter = "AND e.group_id = ?" if group_id else ""
        params: list = [entity_id, hops]
        if group_id:
            params.append(group_id)
        params.append(max_results)

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
        LIMIT ?
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
                created_at=_parse_dt(row["rel_created_at"]) or utc_now(),
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
                    created_at=_parse_dt(row["created_at"]) or utc_now(),
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
            "e.deleted_at IS NULL",
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
        now_iso = utc_now_iso()
        await self.db.execute(
            """INSERT INTO episodes
               (id, content, source, status, group_id, session_id,
                conversation_date, created_at,
                updated_at, error, retry_count, processing_duration_ms,
                encoding_context, memory_tier, consolidation_cycles,
                entity_coverage, projection_state, last_projection_reason,
                last_projected_at, attachments_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                episode.id,
                content,
                episode.source,
                episode.status.value if hasattr(episode.status, "value") else episode.status,
                episode.group_id,
                episode.session_id,
                episode.conversation_date.isoformat() if episode.conversation_date else None,
                episode.created_at.isoformat() if episode.created_at else now_iso,
                episode.updated_at.isoformat() if episode.updated_at else now_iso,
                episode.error,
                episode.retry_count,
                episode.processing_duration_ms,
                episode.encoding_context,
                episode.memory_tier,
                episode.consolidation_cycles,
                episode.entity_coverage,
                (
                    episode.projection_state.value
                    if hasattr(episode.projection_state, "value")
                    else episode.projection_state
                ),
                episode.last_projection_reason,
                episode.last_projected_at.isoformat() if episode.last_projected_at else None,
                (
                    json.dumps([a.model_dump() for a in episode.attachments])
                    if episode.attachments
                    else "[]"
                ),
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
        updates["updated_at"] = utc_now_iso()
        invalid = set(updates.keys()) - EPISODE_UPDATABLE_FIELDS
        if invalid:
            raise ValueError(f"Disallowed episode update fields: {invalid}")
        transformed: dict[str, object] = {}
        for key, value in updates.items():
            if key == "projection_state" and hasattr(value, "value"):
                transformed[key] = value.value
            elif key == "last_projected_at" and value is not None:
                transformed[key] = value.isoformat() if hasattr(value, "isoformat") else value
            else:
                transformed[key] = value
        set_clause = ", ".join(f"{k} = ?" for k in transformed)
        values = list(transformed.values()) + [episode_id, group_id]
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

    async def get_episodes_for_entity(
        self,
        entity_id: str,
        group_id: str = "default",
        limit: int = 20,
    ) -> list[str]:
        """Return episode IDs linked to an entity, newest first."""
        cursor = await self.db.execute(
            """SELECT ee.episode_id FROM episode_entities ee
               JOIN episodes ep ON ep.id = ee.episode_id
               WHERE ee.entity_id = ? AND ep.group_id = ?
               ORDER BY ep.created_at DESC
               LIMIT ?""",
            (entity_id, group_id, limit),
        )
        rows = await cursor.fetchall()
        return [row["episode_id"] for row in rows]

    async def get_adjacent_episodes(
        self,
        episode_id: str,
        group_id: str,
        limit: int = 3,
    ) -> list[Episode]:
        """Get temporally adjacent episodes from the same session."""
        cursor = await self.db.execute(
            "SELECT session_id, created_at FROM episodes WHERE id = ? AND group_id = ?",
            (episode_id, group_id),
        )
        ref = await cursor.fetchone()
        if not ref or not ref["session_id"]:
            return []

        session_id = ref["session_id"]
        ref_created = ref["created_at"]

        cursor = await self.db.execute(
            """SELECT * FROM episodes
               WHERE session_id = ? AND group_id = ? AND id != ?
               ORDER BY ABS(julianday(created_at) - julianday(?))
               LIMIT ?""",
            (session_id, group_id, episode_id, ref_created, limit),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(row, group_id) for row in rows]

    async def upsert_episode_cue(self, cue: EpisodeCue) -> None:
        """Insert or update the cue record for an episode."""
        await self.db.execute(
            """INSERT INTO episode_cues
               (episode_id, group_id, cue_version, discourse_class, projection_state,
                cue_score, salience_score, projection_priority, route_reason, cue_text,
                entity_mentions_json, temporal_markers_json, quote_spans_json,
                contradiction_keys_json, first_spans_json, hit_count, surfaced_count,
                selected_count, used_count, near_miss_count, policy_score,
                projection_attempts, last_hit_at, last_feedback_at, last_projected_at,
                created_at, updated_at)
               VALUES (
                   ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
               )
               ON CONFLICT(episode_id) DO UPDATE SET
                   group_id=excluded.group_id,
                   cue_version=excluded.cue_version,
                   discourse_class=excluded.discourse_class,
                   projection_state=excluded.projection_state,
                   cue_score=excluded.cue_score,
                   salience_score=excluded.salience_score,
                   projection_priority=excluded.projection_priority,
                   route_reason=excluded.route_reason,
                   cue_text=excluded.cue_text,
                   entity_mentions_json=excluded.entity_mentions_json,
                   temporal_markers_json=excluded.temporal_markers_json,
                   quote_spans_json=excluded.quote_spans_json,
                   contradiction_keys_json=excluded.contradiction_keys_json,
                   first_spans_json=excluded.first_spans_json,
                   hit_count=excluded.hit_count,
                   surfaced_count=excluded.surfaced_count,
                   selected_count=excluded.selected_count,
                   used_count=excluded.used_count,
                   near_miss_count=excluded.near_miss_count,
                   policy_score=excluded.policy_score,
                   projection_attempts=excluded.projection_attempts,
                   last_hit_at=excluded.last_hit_at,
                   last_feedback_at=excluded.last_feedback_at,
                   last_projected_at=excluded.last_projected_at,
                   updated_at=excluded.updated_at""",
            (
                cue.episode_id,
                cue.group_id,
                cue.cue_version,
                cue.discourse_class,
                (
                    cue.projection_state.value
                    if hasattr(cue.projection_state, "value")
                    else cue.projection_state
                ),
                cue.cue_score,
                cue.salience_score,
                cue.projection_priority,
                cue.route_reason,
                cue.cue_text,
                json.dumps(cue.entity_mentions),
                json.dumps(cue.temporal_markers),
                json.dumps(cue.quote_spans),
                json.dumps(cue.contradiction_keys),
                json.dumps(cue.first_spans),
                cue.hit_count,
                cue.surfaced_count,
                cue.selected_count,
                cue.used_count,
                cue.near_miss_count,
                cue.policy_score,
                cue.projection_attempts,
                cue.last_hit_at.isoformat() if cue.last_hit_at else None,
                cue.last_feedback_at.isoformat() if cue.last_feedback_at else None,
                cue.last_projected_at.isoformat() if cue.last_projected_at else None,
                cue.created_at.isoformat() if cue.created_at else utc_now_iso(),
                cue.updated_at.isoformat() if cue.updated_at else utc_now_iso(),
            ),
        )
        await self.db.commit()

    async def get_episode_cue(self, episode_id: str, group_id: str) -> EpisodeCue | None:
        cursor = await self.db.execute(
            "SELECT * FROM episode_cues WHERE episode_id = ? AND group_id = ?",
            (episode_id, group_id),
        )
        row = await cursor.fetchone()
        if not row or not row["cue_text"]:
            return None
        return self._row_to_episode_cue(row)

    async def update_episode_cue(
        self,
        episode_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        if not updates:
            return
        updates = dict(updates)
        updates["updated_at"] = utc_now_iso()
        json_fields = {
            "entity_mentions",
            "temporal_markers",
            "quote_spans",
            "contradiction_keys",
            "first_spans",
        }
        remap = {
            "entity_mentions": "entity_mentions_json",
            "temporal_markers": "temporal_markers_json",
            "quote_spans": "quote_spans_json",
            "contradiction_keys": "contradiction_keys_json",
            "first_spans": "first_spans_json",
        }
        transformed: dict = {}
        for key, value in updates.items():
            dest = remap.get(key, key)
            if key in json_fields:
                transformed[dest] = json.dumps(value)
            elif key == "projection_state" and hasattr(value, "value"):
                transformed[dest] = value.value
            elif (
                key in {"last_hit_at", "last_feedback_at", "last_projected_at"}
                and value is not None
            ):
                transformed[dest] = value.isoformat() if hasattr(value, "isoformat") else value
            else:
                transformed[dest] = value

        set_clause = ", ".join(f"{k} = ?" for k in transformed)
        values = list(transformed.values()) + [episode_id, group_id]
        await self.db.execute(
            f"UPDATE episode_cues SET {set_clause} WHERE episode_id = ? AND group_id = ?",
            values,
        )
        await self.db.commit()

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
        rows = list(await rows_cursor.fetchall())

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

    async def find_structural_merge_candidates(
        self,
        group_id: str,
        min_shared_neighbors: int = 3,
        limit: int = 200,
    ) -> list[tuple[str, str, int]]:
        """Find entity pairs sharing many neighbors (structural equivalence).

        Uses an inverted neighbor index: for each neighbor, track which entities
        connect to it. Pairs sharing >=min_shared_neighbors become merge candidates.
        This catches semantic duplicates like "Fourth Son" <-> "Benjamin" that share
        parent/sibling relationships but have zero name overlap.

        Returns (entity_id_a, entity_id_b, shared_count) tuples.
        """
        # Get all active relationships
        cursor = await self.db.execute(
            """SELECT source_id, target_id FROM relationships
               WHERE group_id = ?
                 AND (valid_to IS NULL OR datetime(valid_to) > datetime('now'))
                 AND source_id != target_id""",
            (group_id,),
        )
        rows = await cursor.fetchall()

        # Get active entity IDs to filter out deleted entities
        ent_cursor = await self.db.execute(
            "SELECT id FROM entities WHERE group_id = ? AND deleted_at IS NULL",
            (group_id,),
        )
        active_ids = {r[0] for r in await ent_cursor.fetchall()}

        # Build neighbor sets: entity_id -> set of neighbor IDs
        from collections import defaultdict

        neighbors: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            src, tgt = row[0], row[1]
            if src in active_ids and tgt in active_ids:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

        # Build inverted index: neighbor_id -> set of entities connected to it
        inv_index: dict[str, set[str]] = defaultdict(set)
        for eid, nbrs in neighbors.items():
            for nbr in nbrs:
                inv_index[nbr].add(eid)

        # Find pairs sharing >= min_shared_neighbors via inverted index
        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for nbr_id, connected in inv_index.items():
            connected_list = sorted(connected)
            for i in range(len(connected_list)):
                for j in range(i + 1, len(connected_list)):
                    pair = (connected_list[i], connected_list[j])
                    pair_counts[pair] += 1

        # Filter and sort
        results = [
            (a, b, count) for (a, b), count in pair_counts.items() if count >= min_shared_neighbors
        ]
        results.sort(key=lambda x: -x[2])
        return results[:limit]

    async def get_episode_cooccurrence_count(
        self,
        entity_id_a: str,
        entity_id_b: str,
        group_id: str,
    ) -> int:
        """Count episodes where both entities appear together.

        Returns 0 if they never co-occur (referential exclusivity signal).
        """
        cursor = await self.db.execute(
            """SELECT COUNT(*) FROM episode_entities ee1
               JOIN episode_entities ee2
                   ON ee1.episode_id = ee2.episode_id
               JOIN episodes e ON e.id = ee1.episode_id
               WHERE ee1.entity_id = ? AND ee2.entity_id = ?
                 AND e.group_id = ?""",
            (entity_id_a, entity_id_b, group_id),
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_dead_entities(
        self,
        group_id: str,
        min_age_days: int = 30,
        limit: int = 100,
        max_access_count: int = 0,
    ) -> list[Entity]:
        """Find entities with no relationships and low access."""
        cutoff_dt = utc_now().replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta

        cutoff = (cutoff_dt - timedelta(days=min_age_days)).isoformat()

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
            sql,
            (group_id, group_id, max_access_count, cutoff, limit),
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
        self,
        entity_id: str,
        group_id: str,
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
        self,
        entity_id: str,
        group_id: str,
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
        total_row = await cursor.fetchone()
        total_count = int(_row_value(total_row, 0, 0))

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
                        keep_id,
                        remove_summary[:80],
                    )
                else:
                    # Deduplicate sentences before appending
                    keep_summary = _dedup_summaries(keep_summary, remove_summary)
            await self.db.execute(
                "UPDATE entities SET summary = ?, access_count = ?, updated_at = ? "
                "WHERE id = ? AND group_id = ?",
                (keep_summary, merged_count, utc_now_iso(), keep_id, group_id),
            )

        # Soft-delete the loser
        await self.db.execute(
            "UPDATE entities SET deleted_at = ? WHERE id = ? AND group_id = ?",
            (utc_now_iso(), remove_id, group_id),
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

    async def sample_edges(
        self,
        group_id: str,
        limit: int = 500,
        exclude_ids: set[str] | None = None,
    ) -> list[Relationship]:
        """Return a random sample of active relationships."""
        params: list = [group_id]
        exclude_clause = ""
        if exclude_ids:
            placeholders = ",".join("?" for _ in exclude_ids)
            exclude_clause = f" AND r.id NOT IN ({placeholders})"
            params.extend(sorted(exclude_ids))
        params.append(limit)
        cursor = await self.db.execute(
            f"""SELECT * FROM relationships r
                WHERE r.group_id = ?
                  AND (r.valid_to IS NULL OR datetime(r.valid_to) > datetime('now'))
                  {exclude_clause}
                ORDER BY RANDOM()
                LIMIT ?""",
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
        entity_where = "WHERE deleted_at IS NULL"
        rel_where = ""
        episode_where = ""
        params: tuple[str, ...] = ()
        if group_id:
            entity_where = "WHERE group_id = ? AND deleted_at IS NULL"
            rel_where = "WHERE group_id = ?"
            episode_where = "WHERE group_id = ?"
            params = (group_id,)

        entities = await self.db.execute(
            f"SELECT COUNT(*) FROM entities {entity_where}",
            params,
        )
        rels = await self.db.execute(
            f"SELECT COUNT(*) FROM relationships {rel_where}",
            params,
        )
        episode_stats = await self.db.execute(
            f"""SELECT
                    COUNT(*) AS episodes,
                    SUM(CASE WHEN projection_state = 'queued' THEN 1 ELSE 0 END)
                        AS projection_queued_count,
                    SUM(CASE WHEN projection_state = 'cued' THEN 1 ELSE 0 END)
                        AS projection_cued_count,
                    SUM(CASE WHEN projection_state = 'cue_only' THEN 1 ELSE 0 END)
                        AS projection_cue_only_count,
                    SUM(CASE WHEN projection_state = 'scheduled' THEN 1 ELSE 0 END)
                        AS projection_scheduled_count,
                    SUM(CASE WHEN projection_state = 'projecting' THEN 1 ELSE 0 END)
                        AS projection_projecting_count,
                    SUM(CASE WHEN projection_state = 'projected' THEN 1 ELSE 0 END)
                        AS projection_projected_count,
                    SUM(CASE WHEN projection_state = 'failed' THEN 1 ELSE 0 END)
                        AS projection_failed_count,
                    SUM(CASE WHEN projection_state = 'dead_letter' THEN 1 ELSE 0 END)
                        AS projection_dead_letter_count,
                    AVG(
                        CASE
                            WHEN projection_state = 'projected'
                                 AND processing_duration_ms IS NOT NULL
                            THEN processing_duration_ms
                        END
                    ) AS avg_processing_duration_ms,
                    AVG(
                        CASE
                            WHEN projection_state = 'projected'
                                 AND last_projected_at IS NOT NULL
                            THEN (julianday(last_projected_at) - julianday(created_at)) * 86400000.0
                        END
                    ) AS avg_time_to_projection_ms
                FROM episodes
                {episode_where}""",
            params,
        )
        cue_stats = await self.db.execute(
            f"""SELECT
                    SUM(CASE WHEN cue_text IS NOT NULL AND cue_text <> '' THEN 1 ELSE 0 END)
                        AS cue_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> '' AND hit_count > 0
                            THEN 1
                            ELSE 0
                        END
                    ) AS cue_hit_episode_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN hit_count
                            ELSE 0
                        END
                    ) AS cue_hit_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN surfaced_count
                            ELSE 0
                        END
                    ) AS cue_surfaced_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN selected_count
                            ELSE 0
                        END
                    ) AS cue_selected_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN used_count
                            ELSE 0
                        END
                    ) AS cue_used_count,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN near_miss_count
                            ELSE 0
                        END
                    ) AS cue_near_miss_count,
                    AVG(CASE WHEN cue_text IS NOT NULL AND cue_text <> '' THEN policy_score END)
                        AS avg_policy_score,
                    AVG(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN projection_attempts
                        END
                    ) AS avg_projection_attempts,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL AND cue_text <> ''
                            THEN projection_attempts
                            ELSE 0
                        END
                    ) AS projection_attempt_total,
                    SUM(
                        CASE
                            WHEN cue_text IS NOT NULL
                                 AND cue_text <> ''
                                 AND projection_state = 'projected'
                            THEN 1
                            ELSE 0
                        END
                    )
                        AS projected_cue_count
                FROM episode_cues
                {episode_where}""",
            params,
        )
        yield_stats = await self.db.execute(
            f"""SELECT
                    (
                        SELECT COUNT(*)
                        FROM episode_entities ee
                        JOIN episodes ep ON ep.id = ee.episode_id
                        WHERE ep.projection_state = 'projected'
                        {"AND ep.group_id = ?" if group_id else ""}
                    ) AS linked_entity_count,
                    (
                        SELECT COUNT(*)
                        FROM relationships r
                        JOIN episodes ep ON ep.id = r.source_episode
                        WHERE ep.projection_state = 'projected'
                        {"AND ep.group_id = ?" if group_id else ""}
                    ) AS relationship_count""",
            params * 2,
        )

        entity_count = int(_row_value(await entities.fetchone(), 0, 0))
        relationship_count = int(_row_value(await rels.fetchone(), 0, 0))
        episode_row = await episode_stats.fetchone()
        cue_row = await cue_stats.fetchone()
        yield_row = await yield_stats.fetchone()

        episode_count = int(_row_value(episode_row, "episodes", 0) or 0)
        projection_counts = {
            "queued": int(_row_value(episode_row, "projection_queued_count", 0) or 0),
            "cued": int(_row_value(episode_row, "projection_cued_count", 0) or 0),
            "cue_only": int(_row_value(episode_row, "projection_cue_only_count", 0) or 0),
            "scheduled": int(_row_value(episode_row, "projection_scheduled_count", 0) or 0),
            "projecting": int(_row_value(episode_row, "projection_projecting_count", 0) or 0),
            "projected": int(_row_value(episode_row, "projection_projected_count", 0) or 0),
            "failed": int(_row_value(episode_row, "projection_failed_count", 0) or 0),
            "dead_letter": int(_row_value(episode_row, "projection_dead_letter_count", 0) or 0),
        }
        cue_count = int(_row_value(cue_row, "cue_count", 0) or 0)
        projected_cue_count = int(_row_value(cue_row, "projected_cue_count", 0) or 0)
        attempted_episode_count = (
            projection_counts["projected"]
            + projection_counts["failed"]
            + projection_counts["dead_letter"]
        )
        linked_entity_count = int(_row_value(yield_row, "linked_entity_count", 0) or 0)
        projected_relationship_count = int(_row_value(yield_row, "relationship_count", 0) or 0)

        cue_metrics = {
            "cue_count": cue_count,
            "episodes_without_cues": max(episode_count - cue_count, 0),
            "cue_coverage": round(cue_count / episode_count, 4) if episode_count else 0.0,
            "cue_hit_count": int(_row_value(cue_row, "cue_hit_count", 0) or 0),
            "cue_hit_episode_count": int(_row_value(cue_row, "cue_hit_episode_count", 0) or 0),
            "cue_hit_episode_rate": (
                round((int(_row_value(cue_row, "cue_hit_episode_count", 0) or 0)) / cue_count, 4)
                if cue_count
                else 0.0
            ),
            "cue_surfaced_count": int(_row_value(cue_row, "cue_surfaced_count", 0) or 0),
            "cue_selected_count": int(_row_value(cue_row, "cue_selected_count", 0) or 0),
            "cue_used_count": int(_row_value(cue_row, "cue_used_count", 0) or 0),
            "cue_near_miss_count": int(_row_value(cue_row, "cue_near_miss_count", 0) or 0),
            "avg_policy_score": round(
                float(_row_value(cue_row, "avg_policy_score", 0.0) or 0.0), 4
            ),
            "avg_projection_attempts": round(
                float(_row_value(cue_row, "avg_projection_attempts", 0.0) or 0.0), 4
            ),
            "projected_cue_count": projected_cue_count,
            "cue_to_projection_conversion_rate": (
                round(projected_cue_count / cue_count, 4) if cue_count else 0.0
            ),
        }
        projection_metrics = {
            "state_counts": projection_counts,
            "attempted_episode_count": attempted_episode_count,
            "total_attempts": int(_row_value(cue_row, "projection_attempt_total", 0) or 0),
            "failure_count": projection_counts["failed"],
            "dead_letter_count": projection_counts["dead_letter"],
            "failure_rate": (
                round(
                    (projection_counts["failed"] + projection_counts["dead_letter"])
                    / attempted_episode_count,
                    4,
                )
                if attempted_episode_count
                else 0.0
            ),
            "avg_processing_duration_ms": round(
                float(_row_value(episode_row, "avg_processing_duration_ms", 0.0) or 0.0), 2
            ),
            "avg_time_to_projection_ms": round(
                float(_row_value(episode_row, "avg_time_to_projection_ms", 0.0) or 0.0), 2
            ),
            "yield": {
                "linked_entity_count": linked_entity_count,
                "relationship_count": projected_relationship_count,
                "avg_linked_entities_per_projected_episode": (
                    round(linked_entity_count / projection_counts["projected"], 4)
                    if projection_counts["projected"]
                    else 0.0
                ),
                "avg_relationships_per_projected_episode": (
                    round(
                        projected_relationship_count / projection_counts["projected"],
                        4,
                    )
                    if projection_counts["projected"]
                    else 0.0
                ),
            },
        }

        return {
            "entities": entity_count,
            "relationships": relationship_count,
            "episodes": episode_count,
            "cue_metrics": cue_metrics,
            "projection_metrics": projection_metrics,
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
        lexical_regime = row["lexical_regime"] if "lexical_regime" in row.keys() else None
        canonical_identifier = (
            row["canonical_identifier"] if "canonical_identifier" in row.keys() else None
        )
        identifier_label = False
        if "identifier_label" in row.keys():
            identifier_label = bool(row["identifier_label"])

        source_episode_ids: list[str] = []
        if "source_episode_ids" in row.keys() and row["source_episode_ids"]:
            try:
                source_episode_ids = json.loads(row["source_episode_ids"])
            except (json.JSONDecodeError, TypeError):
                pass
        evidence_count = row["evidence_count"] if "evidence_count" in row.keys() else 0

        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            summary=summary,
            attributes=json.loads(row["attributes"]) if row["attributes"] else None,
            group_id=row_group,
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
            deleted_at=_parse_dt(row["deleted_at"]),
            activation_current=(
                row["activation_current"] if "activation_current" in row.keys() else 0.0
            ),
            access_count=row["access_count"],
            last_accessed=_parse_dt(row["last_accessed"]),
            pii_detected=pii_detected,
            pii_categories=pii_categories,
            identity_core=identity_core,
            lexical_regime=lexical_regime,
            canonical_identifier=canonical_identifier,
            identifier_label=identifier_label,
            source_episode_ids=source_episode_ids,
            evidence_count=evidence_count or 0,
            evidence_span_start=(
                _parse_dt(row["evidence_span_start"])
                if "evidence_span_start" in row.keys()
                else None
            ),
            evidence_span_end=(
                _parse_dt(row["evidence_span_end"])
                if "evidence_span_end" in row.keys()
                else None
            ),
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
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            confidence=confidence,
            polarity=polarity,
            source_episode=row["source_episode"],
            group_id=row["group_id"],
        )

    # --- Schema Formation (Brain Architecture Phase 3) ---

    async def get_schema_members(
        self,
        schema_entity_id: str,
        group_id: str,
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
        self,
        schema_entity_id: str,
        members: list[dict],
        group_id: str,
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
        self,
        entity_type: str,
        group_id: str,
        limit: int = 100,
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
                i.id,
                i.trigger_text,
                i.action_text,
                i.trigger_type,
                i.entity_name,
                i.threshold,
                i.max_fires,
                i.fire_count,
                1 if i.enabled else 0,
                i.group_id,
                i.created_at.isoformat(),
                i.updated_at.isoformat(),
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
        self,
        group_id: str,
        enabled_only: bool = True,
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
        self,
        id: str,
        updates: dict,
        group_id: str,
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
        values.append(utc_now_iso())
        values.extend([id, group_id])
        sql = f"UPDATE intentions SET {', '.join(parts)} WHERE id = ? AND group_id = ?"
        await self.db.execute(sql, values)
        await self.db.commit()

    async def delete_intention(
        self,
        id: str,
        group_id: str,
        soft: bool = True,
    ) -> None:
        """Delete an intention (soft = disable, hard = remove)."""
        if soft:
            await self.db.execute(
                "UPDATE intentions SET enabled = 0, updated_at = ? WHERE id = ? AND group_id = ?",
                (utc_now_iso(), id, group_id),
            )
        else:
            await self.db.execute(
                "DELETE FROM intentions WHERE id = ? AND group_id = ?",
                (id, group_id),
            )
        await self.db.commit()

    async def increment_intention_fire_count(
        self,
        id: str,
        group_id: str,
    ) -> None:
        """Increment fire_count by 1."""
        await self.db.execute(
            """UPDATE intentions
               SET fire_count = fire_count + 1, updated_at = ?
               WHERE id = ? AND group_id = ?""",
            (utc_now_iso(), id, group_id),
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
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]) or utc_now(),
            expires_at=expires_at,
        )

    # --- Evidence storage (v2) ---

    async def store_evidence(
        self,
        evidence: list[dict],
        group_id: str = "default",
        *,
        default_status: str = "pending",
    ) -> None:
        """Persist evidence candidates from the extraction pipeline."""
        if not evidence:
            return
        sql = (
            "INSERT OR IGNORE INTO episode_evidence "
            "(evidence_id, episode_id, group_id, fact_class, confidence, "
            "source_type, extractor_name, payload_json, source_span, "
            "signals_json, ambiguity_tags_json, ambiguity_score, adjudication_request_id, "
            "status, commit_reason, committed_id, deferred_cycles, created_at, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        rows = [
            (
                ev["evidence_id"],
                ev["episode_id"],
                group_id,
                ev["fact_class"],
                ev["confidence"],
                ev["source_type"],
                ev.get("extractor_name", ""),
                json.dumps(ev.get("payload", {})),
                ev.get("source_span"),
                json.dumps(ev.get("corroborating_signals", [])),
                json.dumps(ev.get("ambiguity_tags", [])),
                ev.get("ambiguity_score", 0.0),
                ev.get("adjudication_request_id"),
                ev.get("status", default_status),
                ev.get("commit_reason"),
                ev.get("committed_id"),
                ev.get("deferred_cycles", 0),
                ev.get("created_at", utc_now_iso()),
                ev.get("resolved_at")
                or (
                    utc_now_iso()
                    if ev.get("status", default_status)
                    in {"committed", "rejected", "expired", "superseded"}
                    else None
                ),
            )
            for ev in evidence
        ]
        await self.db.executemany(sql, rows)
        await self.db.commit()

    async def get_pending_evidence(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Get unresolved evidence candidates for adjudication."""
        sql = (
            "SELECT evidence_id, episode_id, group_id, fact_class, confidence, "
            "source_type, extractor_name, payload_json, source_span, "
            "signals_json, ambiguity_tags_json, ambiguity_score, adjudication_request_id, "
            "status, commit_reason, committed_id, deferred_cycles, created_at, resolved_at "
            "FROM episode_evidence "
            "WHERE group_id = ? AND status IN ('pending', 'deferred', 'approved') "
            "ORDER BY confidence DESC LIMIT ?"
        )
        cursor = await self.db.execute(sql, (group_id, limit))
        rows = await cursor.fetchall()
        return [_evidence_row_to_dict(r) for r in rows]

    async def get_episode_evidence(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Get all evidence for a specific episode."""
        sql = (
            "SELECT evidence_id, episode_id, group_id, fact_class, confidence, "
            "source_type, extractor_name, payload_json, source_span, "
            "signals_json, ambiguity_tags_json, ambiguity_score, adjudication_request_id, "
            "status, commit_reason, committed_id, deferred_cycles, created_at, resolved_at "
            "FROM episode_evidence "
            "WHERE episode_id = ? AND group_id = ? "
            "ORDER BY confidence DESC"
        )
        cursor = await self.db.execute(sql, (episode_id, group_id))
        rows = await cursor.fetchall()
        return [_evidence_row_to_dict(r) for r in rows]

    async def update_evidence_status(
        self,
        evidence_id: str,
        status: str,
        updates: dict | None = None,
        group_id: str = "default",
    ) -> None:
        """Update evidence status and optional fields."""
        updates = updates or {}
        sets = ["status = ?"]
        params: list = [status]
        if "commit_reason" in updates:
            sets.append("commit_reason = ?")
            params.append(updates["commit_reason"])
        if "committed_id" in updates:
            sets.append("committed_id = ?")
            params.append(updates["committed_id"])
        if "confidence" in updates:
            sets.append("confidence = ?")
            params.append(updates["confidence"])
        if "deferred_cycles" in updates:
            sets.append("deferred_cycles = ?")
            params.append(updates["deferred_cycles"])
        if "ambiguity_tags" in updates:
            sets.append("ambiguity_tags_json = ?")
            params.append(json.dumps(updates["ambiguity_tags"]))
        if "ambiguity_score" in updates:
            sets.append("ambiguity_score = ?")
            params.append(updates["ambiguity_score"])
        if "adjudication_request_id" in updates:
            sets.append("adjudication_request_id = ?")
            params.append(updates["adjudication_request_id"])
        if status in ("committed", "rejected", "expired", "superseded"):
            sets.append("resolved_at = ?")
            params.append(utc_now_iso())
        params.extend([evidence_id, group_id])
        sql = (
            f"UPDATE episode_evidence SET {', '.join(sets)} WHERE evidence_id = ? AND group_id = ?"
        )
        await self.db.execute(sql, params)
        await self.db.commit()

    async def get_entity_count(self, group_id: str = "default") -> int:
        """Count non-deleted entities in a group."""
        sql = "SELECT COUNT(*) FROM entities WHERE group_id = ? AND deleted_at IS NULL"
        cursor = await self.db.execute(sql, (group_id,))
        row = await cursor.fetchone()
        return int(_row_value(row, 0, 0))

    async def store_adjudication_requests(
        self,
        requests: list[dict],
        group_id: str = "default",
    ) -> None:
        """Persist edge adjudication requests."""
        if not requests:
            return
        sql = (
            "INSERT OR IGNORE INTO episode_adjudications "
            "(request_id, episode_id, group_id, status, ambiguity_tags_json, "
            "evidence_ids_json, selected_text, request_reason, resolution_source, "
            "resolution_payload_json, attempt_count, created_at, resolved_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
        rows = [
            (
                req["request_id"],
                req["episode_id"],
                group_id,
                req.get("status", "pending"),
                json.dumps(req.get("ambiguity_tags", [])),
                json.dumps(req.get("evidence_ids", [])),
                req.get("selected_text", ""),
                req.get("request_reason", ""),
                req.get("resolution_source"),
                (
                    json.dumps(req.get("resolution_payload"))
                    if req.get("resolution_payload") is not None
                    else None
                ),
                req.get("attempt_count", 0),
                req.get("created_at", utc_now_iso()),
                req.get("resolved_at"),
            )
            for req in requests
        ]
        await self.db.executemany(sql, rows)
        await self.db.commit()

    async def get_episode_adjudications(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Get adjudication requests for an episode."""
        cursor = await self.db.execute(
            "SELECT request_id, episode_id, group_id, status, ambiguity_tags_json, "
            "evidence_ids_json, selected_text, request_reason, resolution_source, "
            "resolution_payload_json, attempt_count, created_at, resolved_at "
            "FROM episode_adjudications WHERE episode_id = ? AND group_id = ? "
            "ORDER BY created_at ASC",
            (episode_id, group_id),
        )
        rows = await cursor.fetchall()
        return [_adjudication_row_to_dict(row) for row in rows]

    async def get_adjudication_request(
        self,
        request_id: str,
        group_id: str = "default",
    ) -> dict | None:
        """Get a single adjudication request by ID."""
        cursor = await self.db.execute(
            "SELECT request_id, episode_id, group_id, status, ambiguity_tags_json, "
            "evidence_ids_json, selected_text, request_reason, resolution_source, "
            "resolution_payload_json, attempt_count, created_at, resolved_at "
            "FROM episode_adjudications WHERE request_id = ? AND group_id = ?",
            (request_id, group_id),
        )
        row = await cursor.fetchone()
        return _adjudication_row_to_dict(row) if row else None

    async def get_pending_adjudication_requests(
        self,
        group_id: str = "default",
        limit: int = 100,
    ) -> list[dict]:
        """Get unresolved adjudication requests for consolidation."""
        cursor = await self.db.execute(
            "SELECT request_id, episode_id, group_id, status, ambiguity_tags_json, "
            "evidence_ids_json, selected_text, request_reason, resolution_source, "
            "resolution_payload_json, attempt_count, created_at, resolved_at "
            "FROM episode_adjudications "
            "WHERE group_id = ? AND status IN ('pending', 'deferred', 'error') "
            "ORDER BY created_at ASC LIMIT ?",
            (group_id, limit),
        )
        rows = await cursor.fetchall()
        return [_adjudication_row_to_dict(row) for row in rows]

    async def update_adjudication_request(
        self,
        request_id: str,
        updates: dict,
        group_id: str = "default",
    ) -> None:
        """Update adjudication request status and metadata."""
        if not updates:
            return
        sets: list[str] = []
        params: list = []
        for field, column in (
            ("status", "status"),
            ("ambiguity_tags", "ambiguity_tags_json"),
            ("evidence_ids", "evidence_ids_json"),
            ("selected_text", "selected_text"),
            ("request_reason", "request_reason"),
            ("resolution_source", "resolution_source"),
            ("attempt_count", "attempt_count"),
            ("resolved_at", "resolved_at"),
        ):
            if field not in updates:
                continue
            sets.append(f"{column} = ?")
            value = updates[field]
            if field in {"ambiguity_tags", "evidence_ids"}:
                value = json.dumps(value)
            params.append(value)
        if "resolution_payload" in updates:
            sets.append("resolution_payload_json = ?")
            params.append(
                json.dumps(updates["resolution_payload"])
                if updates["resolution_payload"] is not None
                else None,
            )
        status = updates.get("status")
        if status in {"materialized", "rejected", "expired"} and "resolved_at" not in updates:
            sets.append("resolved_at = ?")
            params.append(utc_now_iso())
        if not sets:
            return
        params.extend([request_id, group_id])
        await self.db.execute(
            f"UPDATE episode_adjudications SET {', '.join(sets)} "
            "WHERE request_id = ? AND group_id = ?",
            params,
        )
        await self.db.commit()

    def _row_to_episode(self, row, group_id: str | None = None) -> Episode:
        row_group = row["group_id"]
        decrypt_group = group_id or row_group
        content = self._decrypt(decrypt_group, row["content"]) or ""
        keys = row.keys()
        raw_status = row["status"]
        status = (
            raw_status if isinstance(raw_status, EpisodeStatus) else EpisodeStatus(str(raw_status))
        )
        raw_projection_state = (
            row["projection_state"]
            if "projection_state" in keys and row["projection_state"]
            else EpisodeProjectionState.QUEUED.value
        )
        projection_state = (
            raw_projection_state
            if isinstance(raw_projection_state, EpisodeProjectionState)
            else EpisodeProjectionState(str(raw_projection_state))
        )

        return Episode(
            id=row["id"],
            content=content,
            source=row["source"],
            status=status,
            group_id=row_group,
            session_id=row["session_id"] if "session_id" in keys else None,
            conversation_date=(
                _parse_dt(row["conversation_date"]) if "conversation_date" in keys else None
            ),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
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
            encoding_context=(row["encoding_context"] if "encoding_context" in keys else None),
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
            projection_state=projection_state,
            last_projection_reason=(
                row["last_projection_reason"] if "last_projection_reason" in keys else None
            ),
            last_projected_at=(
                _parse_dt(row["last_projected_at"]) if "last_projected_at" in keys else None
            ),
            attachments=[
                Attachment(**a)
                for a in json.loads(
                    row["attachments_json"]
                    if "attachments_json" in keys and row["attachments_json"]
                    else "[]"
                )
            ],
        )

    def _row_to_episode_cue(self, row) -> EpisodeCue:
        return EpisodeCue(
            episode_id=row["episode_id"],
            group_id=row["group_id"],
            cue_version=row["cue_version"],
            discourse_class=row["discourse_class"],
            projection_state=row["projection_state"],
            cue_score=row["cue_score"],
            salience_score=row["salience_score"],
            projection_priority=row["projection_priority"],
            route_reason=row["route_reason"],
            cue_text=row["cue_text"],
            entity_mentions=json.loads(row["entity_mentions_json"] or "[]"),
            temporal_markers=json.loads(row["temporal_markers_json"] or "[]"),
            quote_spans=json.loads(row["quote_spans_json"] or "[]"),
            contradiction_keys=json.loads(row["contradiction_keys_json"] or "[]"),
            first_spans=json.loads(row["first_spans_json"] or "[]"),
            hit_count=row["hit_count"] or 0,
            surfaced_count=row["surfaced_count"] or 0,
            selected_count=row["selected_count"] or 0,
            used_count=row["used_count"] or 0,
            near_miss_count=row["near_miss_count"] or 0,
            policy_score=row["policy_score"] or 0.0,
            projection_attempts=row["projection_attempts"] or 0,
            last_hit_at=_parse_dt(row["last_hit_at"]),
            last_feedback_at=_parse_dt(row["last_feedback_at"]),
            last_projected_at=_parse_dt(row["last_projected_at"]),
            created_at=_parse_dt(row["created_at"]) or utc_now(),
            updated_at=_parse_dt(row["updated_at"]),
        )


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
