"""Storage layer for graph embeddings — separate from text embeddings."""

from __future__ import annotations

import logging
from datetime import datetime

import aiosqlite

from engram.storage.sqlite.vectors import pack_vector, unpack_vector

logger = logging.getLogger(__name__)


class GraphEmbeddingStore:
    """CRUD for the graph_embeddings table.

    Stores structural embeddings (Node2Vec, TransE, GNN) separately from
    text embeddings. Uses same pack/unpack as SQLiteVectorStore.
    """

    async def initialize(self, db: aiosqlite.Connection) -> None:
        """Create the graph_embeddings table if it doesn't exist."""
        await db.execute("""
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
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_group "
            "ON graph_embeddings(group_id)"
        )
        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_graph_emb_method "
            "ON graph_embeddings(method)"
        )
        await db.commit()

    async def upsert_batch(
        self,
        db: aiosqlite.Connection,
        embeddings: dict[str, list[float]],
        method: str,
        group_id: str,
        model_version: str = "",
    ) -> int:
        """Insert or update a batch of graph embeddings.

        Returns number of embeddings upserted.
        """
        if not embeddings:
            return 0

        now = datetime.utcnow().isoformat()
        rows = []
        for entity_id, vec in embeddings.items():
            blob = pack_vector(vec)
            rows.append((
                entity_id, group_id, blob, len(vec), method,
                model_version, now, now,
            ))

        await db.executemany(
            """INSERT INTO graph_embeddings
                (id, group_id, embedding, dimensions, method,
                 model_version, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id, method, group_id) DO UPDATE SET
                embedding = excluded.embedding,
                dimensions = excluded.dimensions,
                model_version = excluded.model_version,
                updated_at = excluded.updated_at
            """,
            rows,
        )
        await db.commit()
        return len(rows)

    async def get_embeddings(
        self,
        db: aiosqlite.Connection,
        entity_ids: list[str],
        method: str,
        group_id: str,
    ) -> dict[str, list[float]]:
        """Retrieve graph embeddings for specific entity IDs."""
        if not entity_ids:
            return {}

        placeholders = ",".join("?" for _ in entity_ids)
        cursor = await db.execute(
            f"SELECT id, embedding, dimensions FROM graph_embeddings "
            f"WHERE id IN ({placeholders}) AND method = ? AND group_id = ?",
            [*entity_ids, method, group_id],
        )
        rows = await cursor.fetchall()

        result = {}
        for row in rows:
            result[row[0]] = unpack_vector(row[1], row[2])
        return result

    async def get_all_embeddings(
        self,
        db: aiosqlite.Connection,
        method: str,
        group_id: str,
    ) -> dict[str, list[float]]:
        """Retrieve all graph embeddings for a method and group."""
        cursor = await db.execute(
            "SELECT id, embedding, dimensions FROM graph_embeddings "
            "WHERE method = ? AND group_id = ?",
            (method, group_id),
        )
        rows = await cursor.fetchall()

        result = {}
        for row in rows:
            result[row[0]] = unpack_vector(row[1], row[2])
        return result

    async def delete_by_method(
        self,
        db: aiosqlite.Connection,
        method: str,
        group_id: str,
    ) -> int:
        """Delete all graph embeddings for a method and group.

        Returns number of rows deleted.
        """
        cursor = await db.execute(
            "DELETE FROM graph_embeddings WHERE method = ? AND group_id = ?",
            (method, group_id),
        )
        await db.commit()
        return cursor.rowcount
