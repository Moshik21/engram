"""SQLite BLOB-based vector storage with brute-force cosine similarity."""

from __future__ import annotations

import logging
import struct

import aiosqlite
import numpy as np
from numpy.typing import NDArray

from engram.utils.dates import utc_now_iso

logger = logging.getLogger(__name__)


def pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary BLOB (4 bytes per dimension)."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a binary BLOB back into a float vector."""
    return list(struct.unpack(f"{dim}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class SQLiteVectorStore:
    """BLOB-stored float32 vectors with brute-force cosine similarity.

    Designed for lite mode — no external dependencies, ~20ms for 10K vectors.
    """

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._owns_db = False

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        """Initialize, optionally sharing a db connection."""
        if db:
            self._db = db
            self._owns_db = False
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            self._owns_db = True

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'entity',
                group_id TEXT NOT NULL,
                text_content TEXT,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embed_provider TEXT NOT NULL DEFAULT '',
                embed_model TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (id, content_type)
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_group ON embeddings(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(content_type)"
        )
        # Migration: add versioning columns for existing databases
        for col in ("embed_provider", "embed_model"):
            try:
                await self.db.execute(
                    f"ALTER TABLE embeddings ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )
            except Exception:
                pass  # column already exists

        # Migration: upgrade legacy single-column PK (id) to composite (id, content_type).
        # Older DBs collided episode and cue vectors on the same id; rebuild the table so
        # both rows can coexist, preserving existing data.
        await self._migrate_composite_pk()

        # Graph embeddings table (structural embeddings from Node2Vec/TransE/GNN)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS graph_embeddings (
                id         TEXT NOT NULL,
                group_id   TEXT NOT NULL,
                embedding  BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                method     TEXT NOT NULL,
                model_version TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (id, method, group_id)
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_group ON graph_embeddings(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_method ON graph_embeddings(method)"
        )
        await self.db.commit()

    async def _migrate_composite_pk(self) -> None:
        """Rebuild legacy embeddings tables whose PK is (id) only.

        Old DBs declared `id TEXT PRIMARY KEY`, so an episode vector and its cue
        vector (both keyed on episode.id) collided — the second write overwrote
        the first. This rebuilds the table with PRIMARY KEY (id, content_type),
        deduplicating any pre-existing collided rows (keeps the most recent).
        Idempotent: a no-op once the composite PK is present.
        """
        cursor = await self.db.execute("PRAGMA table_info(embeddings)")
        cols = await cursor.fetchall()
        if not cols:
            return  # table does not exist yet (e.g. fresh create raced)
        # `pk` is column index 5; >0 means the column participates in the PK.
        pk_cols = {row[1] for row in cols if row[5]}
        if pk_cols == {"id", "content_type"}:
            return  # already migrated

        await self.db.execute("ALTER TABLE embeddings RENAME TO embeddings_legacy")
        await self.db.execute("""
            CREATE TABLE embeddings (
                id TEXT NOT NULL,
                content_type TEXT NOT NULL DEFAULT 'entity',
                group_id TEXT NOT NULL,
                text_content TEXT,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embed_provider TEXT NOT NULL DEFAULT '',
                embed_model TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (id, content_type)
            )
        """)
        # Copy rows; on the (id, content_type) collision that legacy DBs cannot
        # have had (PK was id alone) this is a straight copy. INSERT OR REPLACE
        # guards against any unexpected duplicate just in case.
        await self.db.execute("""
            INSERT OR REPLACE INTO embeddings
                (id, content_type, group_id, text_content, embedding,
                 dimensions, created_at, updated_at, embed_provider, embed_model)
            SELECT id, content_type, group_id, text_content, embedding,
                   dimensions, created_at, updated_at, embed_provider, embed_model
            FROM embeddings_legacy
        """)
        await self.db.execute("DROP TABLE embeddings_legacy")
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_group ON embeddings(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(content_type)"
        )
        await self.db.commit()
        logger.info("Migrated embeddings table to composite PK (id, content_type)")

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteVectorStore not initialized.")
        return self._db

    async def upsert(
        self,
        item_id: str,
        content_type: str,
        group_id: str,
        text_content: str | None,
        embedding: list[float],
        embed_provider: str = "",
        embed_model: str = "",
    ) -> None:
        """Insert or update a vector embedding."""
        now = utc_now_iso()
        blob = pack_vector(embedding)
        await self.db.execute(
            """INSERT INTO embeddings
                (id, content_type, group_id, text_content,
                 embedding, dimensions, created_at, updated_at,
                 embed_provider, embed_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, content_type) DO UPDATE SET
                text_content = excluded.text_content,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                updated_at = excluded.updated_at,
                embed_provider = excluded.embed_provider,
                embed_model = excluded.embed_model
            """,
            (
                item_id,
                content_type,
                group_id,
                text_content,
                blob,
                len(embedding),
                now,
                now,
                embed_provider,
                embed_model,
            ),
        )
        await self.db.commit()

    async def batch_upsert(
        self,
        items: list[tuple[str, str, str, str | None, list[float]]],
        embed_provider: str = "",
        embed_model: str = "",
    ) -> None:
        """Batch upsert vectors.

        Each tuple: (id, content_type, group_id, text, embedding).
        """
        now = utc_now_iso()
        rows = []
        for item_id, content_type, group_id, text_content, embedding in items:
            blob = pack_vector(embedding)
            rows.append(
                (
                    item_id,
                    content_type,
                    group_id,
                    text_content,
                    blob,
                    len(embedding),
                    now,
                    now,
                    embed_provider,
                    embed_model,
                )
            )

        await self.db.executemany(
            """INSERT INTO embeddings
                (id, content_type, group_id, text_content,
                 embedding, dimensions, created_at, updated_at,
                 embed_provider, embed_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, content_type) DO UPDATE SET
                text_content = excluded.text_content,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                updated_at = excluded.updated_at,
                embed_provider = excluded.embed_provider,
                embed_model = excluded.embed_model
            """,
            rows,
        )
        await self.db.commit()

    async def search(
        self,
        query_vector: list[float],
        group_id: str | None,
        content_type: str = "entity",
        limit: int = 20,
        storage_dim: int = 0,
        entity_types: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Search for nearest vectors by cosine similarity.

        Returns (item_id, similarity_score) pairs sorted by score descending.
        When storage_dim > 0, truncates old full-dim vectors to match.

        When *entity_types* is provided, the candidate set is restricted to
        embeddings whose backing entity has a matching entity_type (joined from
        the entities table). This keeps the vector branch consistent with the
        FTS branch so an entity_types-scoped query (e.g. search_artifacts) never
        surfaces off-type entities. The filter is silently ignored for
        non-entity content types.
        """
        params: list[str] = [content_type]
        if entity_types and content_type == "entity":
            placeholders = ",".join("?" * len(entity_types))
            sql = (
                "SELECT em.id AS id, em.embedding AS embedding, "
                "em.dimensions AS dimensions FROM embeddings em "
                "JOIN entities en ON en.id = em.id "
                f"WHERE em.content_type = ? AND en.entity_type IN ({placeholders})"
            )
            params.extend(entity_types)
            if group_id is not None:
                sql += " AND em.group_id = ?"
                params.append(group_id)
        else:
            sql = "SELECT id, embedding, dimensions FROM embeddings WHERE content_type = ?"
            if group_id is not None:
                sql += " AND group_id = ?"
                params.append(group_id)

        cursor = await self.db.execute(
            sql,
            params,
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        ids = []
        vecs = []
        for row in rows:
            vec = unpack_vector(row["embedding"], row["dimensions"])
            # Truncate old full-dim vectors to storage_dim for consistent comparison
            if storage_dim > 0 and len(vec) > storage_dim:
                vec = vec[:storage_dim]
            ids.append(row["id"])
            vecs.append(vec)

        query_np: NDArray[np.float64] = np.asarray(query_vector, dtype=np.float64)
        qn = np.linalg.norm(query_np)
        if qn == 0.0:
            return []
        query_np = query_np / qn

        mat: NDArray[np.float64] = np.array(vecs, dtype=np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        mat = mat / norms
        sims = mat @ query_np

        scored = sorted(zip(ids, sims.tolist()), key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def remove(self, item_id: str, content_type: str | None = None) -> None:
        """Remove a vector by ID, optionally constrained to a content type."""
        if content_type is None:
            await self.db.execute("DELETE FROM embeddings WHERE id = ?", (item_id,))
        else:
            await self.db.execute(
                "DELETE FROM embeddings WHERE id = ? AND content_type = ?",
                (item_id, content_type),
            )
        await self.db.commit()

    async def delete_group(self, group_id: str) -> None:
        """Remove all embeddings for *group_id*."""
        await self.db.execute("DELETE FROM embeddings WHERE group_id = ?", (group_id,))
        await self.db.execute("DELETE FROM graph_embeddings WHERE group_id = ?", (group_id,))
        await self.db.commit()

    async def has_embeddings(self, group_id: str) -> bool:
        """Check if any embeddings exist for a group."""
        cursor = await self.db.execute(
            "SELECT 1 FROM embeddings WHERE group_id = ? LIMIT 1",
            (group_id,),
        )
        row = await cursor.fetchone()
        return row is not None

    async def close(self) -> None:
        """Close owned connections while leaving borrowed graph DBs open."""
        if self._db and self._owns_db:
            await self._db.close()
        self._db = None
        self._owns_db = False
