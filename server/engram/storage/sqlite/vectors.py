"""SQLite BLOB-based vector storage with brute-force cosine similarity."""

from __future__ import annotations

import logging
import struct

import aiosqlite
import numpy as np

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

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        """Initialize, optionally sharing a db connection."""
        if db:
            self._db = db
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL DEFAULT 'entity',
                group_id TEXT NOT NULL,
                text_content TEXT,
                embedding BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                embed_provider TEXT NOT NULL DEFAULT '',
                embed_model TEXT NOT NULL DEFAULT ''
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
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_group "
            "ON graph_embeddings(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_method "
            "ON graph_embeddings(method)"
        )
        await self.db.commit()

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
            ON CONFLICT(id) DO UPDATE SET
                text_content = excluded.text_content,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                updated_at = excluded.updated_at,
                embed_provider = excluded.embed_provider,
                embed_model = excluded.embed_model
            """,
            (item_id, content_type, group_id, text_content, blob, len(embedding),
             now, now, embed_provider, embed_model),
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
            ON CONFLICT(id) DO UPDATE SET
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
        group_id: str,
        content_type: str = "entity",
        limit: int = 20,
        storage_dim: int = 0,
    ) -> list[tuple[str, float]]:
        """Search for nearest vectors by cosine similarity.

        Returns (item_id, similarity_score) pairs sorted by score descending.
        When storage_dim > 0, truncates old full-dim vectors to match.
        """
        cursor = await self.db.execute(
            "SELECT id, embedding, dimensions FROM embeddings "
            "WHERE group_id = ? AND content_type = ?",
            (group_id, content_type),
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

        query_np = np.asarray(query_vector, dtype=np.float32)
        qn = np.linalg.norm(query_np)
        if qn == 0.0:
            return []
        query_np = query_np / qn

        mat = np.array(vecs, dtype=np.float32)
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

    async def has_embeddings(self, group_id: str) -> bool:
        """Check if any embeddings exist for a group."""
        cursor = await self.db.execute(
            "SELECT 1 FROM embeddings WHERE group_id = ? LIMIT 1",
            (group_id,),
        )
        row = await cursor.fetchone()
        return row is not None
