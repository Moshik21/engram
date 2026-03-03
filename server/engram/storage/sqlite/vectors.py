"""SQLite BLOB-based vector storage with brute-force cosine similarity."""

from __future__ import annotations

import logging
import struct
from datetime import datetime

import aiosqlite
import numpy as np

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
                updated_at TEXT NOT NULL
            )
        """)
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_group ON embeddings(group_id)"
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(content_type)"
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
    ) -> None:
        """Insert or update a vector embedding."""
        now = datetime.utcnow().isoformat()
        blob = pack_vector(embedding)
        await self.db.execute(
            """INSERT INTO embeddings
                (id, content_type, group_id, text_content,
                 embedding, dimensions, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text_content = excluded.text_content,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                updated_at = excluded.updated_at
            """,
            (item_id, content_type, group_id, text_content, blob, len(embedding), now, now),
        )
        await self.db.commit()

    async def batch_upsert(
        self,
        items: list[tuple[str, str, str, str | None, list[float]]],
    ) -> None:
        """Batch upsert vectors.

        Each tuple: (id, content_type, group_id, text, embedding).
        """
        now = datetime.utcnow().isoformat()
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
                )
            )

        await self.db.executemany(
            """INSERT INTO embeddings
                (id, content_type, group_id, text_content,
                 embedding, dimensions, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                text_content = excluded.text_content,
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                updated_at = excluded.updated_at
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
    ) -> list[tuple[str, float]]:
        """Search for nearest vectors by cosine similarity.

        Returns (item_id, similarity_score) pairs sorted by score descending.
        """
        cursor = await self.db.execute(
            "SELECT id, embedding, dimensions FROM embeddings "
            "WHERE group_id = ? AND content_type = ?",
            (group_id, content_type),
        )
        rows = await cursor.fetchall()

        if not rows:
            return []

        ids = [row["id"] for row in rows]
        query_np = np.asarray(query_vector, dtype=np.float32)
        qn = np.linalg.norm(query_np)
        if qn == 0.0:
            return []
        query_np = query_np / qn

        mat = np.array(
            [unpack_vector(row["embedding"], row["dimensions"]) for row in rows],
            dtype=np.float32,
        )
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        mat = mat / norms
        sims = mat @ query_np

        scored = sorted(zip(ids, sims.tolist()), key=lambda x: x[1], reverse=True)
        return scored[:limit]

    async def remove(self, item_id: str) -> None:
        """Remove a vector by ID."""
        await self.db.execute("DELETE FROM embeddings WHERE id = ?", (item_id,))
        await self.db.commit()

    async def has_embeddings(self, group_id: str) -> bool:
        """Check if any embeddings exist for a group."""
        cursor = await self.db.execute(
            "SELECT 1 FROM embeddings WHERE group_id = ? LIMIT 1",
            (group_id,),
        )
        row = await cursor.fetchone()
        return row is not None
