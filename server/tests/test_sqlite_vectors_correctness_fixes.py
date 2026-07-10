"""Regression tests for SQLite vector store correctness (cluster A3).

B2: episode and cue vectors must not collide on a single-column primary key.
    The embeddings PK is composite (id, content_type), so an episode vector and
    its cue vector keyed on the same episode.id coexist as two rows and both
    vector searches return the right vector.

B5: an entity_types-scoped query through the embeddings path must not return
    off-type entities. The vector search branch honors entity_types via a JOIN
    on the entities table, matching the FTS branch.
"""

from __future__ import annotations

import aiosqlite
import pytest

from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore


class TinyEmbeddingProvider:
    """2-D provider: 'alpha' -> [1,0], everything else -> [0,1]."""

    def dimension(self) -> int:
        return 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


class EmptyFTS:
    """FTS stub that returns nothing, isolating the vector branch."""

    async def initialize(self, **_kwargs) -> None:
        return None

    async def search(self, **_kwargs) -> list[tuple[str, float]]:
        return []

    async def search_episodes(self, **_kwargs) -> list[tuple[str, float]]:
        return []

    async def search_episode_cues(self, **_kwargs) -> list[tuple[str, float]]:
        return []

    async def remove(self, _entity_id: str) -> None:
        return None

    async def delete_group(self, _group_id: str) -> None:
        return None


# --------------------------------------------------------------------------- #
# B2: composite primary key (id, content_type)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_episode_and_cue_same_id_produce_two_rows() -> None:
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        ep_id = "episode-1"
        # Episode vector and its cue vector share the same id.
        await vectors.upsert(ep_id, "episode", "g", "episode body", [1.0, 0.0])
        await vectors.upsert(ep_id, "episode_cue", "g", "alpha cue", [0.0, 1.0])

        cursor = await vectors.db.execute(
            "SELECT content_type, embedding, dimensions FROM embeddings "
            "WHERE id = ? ORDER BY content_type",
            (ep_id,),
        )
        rows = await cursor.fetchall()
        # Two distinct rows survive — no overwrite.
        assert len(rows) == 2
        assert {row["content_type"] for row in rows} == {"episode", "episode_cue"}
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_episode_and_cue_vectors_searchable_independently() -> None:
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        ep_id = "episode-1"
        await vectors.upsert(ep_id, "episode", "g", "episode body", [1.0, 0.0])
        await vectors.upsert(ep_id, "episode_cue", "g", "alpha cue", [0.0, 1.0])

        # Episode vector points to [1,0]; cue vector points to [0,1].
        ep_hits = await vectors.search([1.0, 0.0], "g", content_type="episode")
        cue_hits = await vectors.search([0.0, 1.0], "g", content_type="episode_cue")

        assert ep_hits and ep_hits[0][0] == ep_id
        assert ep_hits[0][1] == pytest.approx(1.0, abs=1e-5)
        assert cue_hits and cue_hits[0][0] == ep_id
        assert cue_hits[0][1] == pytest.approx(1.0, abs=1e-5)
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_hybrid_search_episode_cues_returns_cue_not_corrupted() -> None:
    """End-to-end via HybridSearchIndex.search_episode_cues (was dead, [])."""
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        ep_id = "episode-1"
        # Episode embedding intentionally differs from the cue embedding so that
        # an overwrite (the old bug) would corrupt the cue search.
        await vectors.upsert(ep_id, "episode", "g", "beta episode", [0.0, 1.0])
        await vectors.upsert(ep_id, "episode_cue", "g", "alpha cue", [1.0, 0.0])

        index = HybridSearchIndex(
            EmptyFTS(),
            vectors,
            TinyEmbeddingProvider(),
            storage_dim=2,
            embed_provider="tiny",
            embed_model="tiny-2",
        )

        cue_results = await index.search_episode_cues("alpha", group_id="g", limit=5)
        assert cue_results, "cue search must not be dead"
        assert cue_results[0][0] == ep_id
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_legacy_single_column_pk_migrated_in_place(tmp_path) -> None:
    """A DB created with the old `id PRIMARY KEY` schema migrates and keeps data."""
    db_path = str(tmp_path / "legacy.db")
    # Build a legacy-schema embeddings table by hand (single-column PK).
    legacy = await aiosqlite.connect(db_path)
    legacy.row_factory = aiosqlite.Row
    await legacy.execute(
        """
        CREATE TABLE embeddings (
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
        """
    )
    import struct

    from engram.utils.dates import utc_now_iso

    now = utc_now_iso()
    await legacy.execute(
        "INSERT INTO embeddings (id, content_type, group_id, text_content, "
        "embedding, dimensions, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        ("ent-1", "entity", "g", "kept", struct.pack("2f", 1.0, 0.0), 2, now, now),
    )
    await legacy.commit()
    await legacy.close()

    # initialize() must migrate to composite PK and preserve the existing row.
    vectors = SQLiteVectorStore(db_path)
    await vectors.initialize()
    try:
        cursor = await vectors.db.execute("PRAGMA table_info(embeddings)")
        cols = await cursor.fetchall()
        pk_cols = {row[1] for row in cols if row[5]}
        assert pk_cols == {"id", "content_type"}

        # Pre-existing row preserved.
        cursor = await vectors.db.execute(
            "SELECT text_content FROM embeddings WHERE id = ? AND content_type = ?",
            ("ent-1", "entity"),
        )
        row = await cursor.fetchone()
        assert row is not None and row["text_content"] == "kept"

        # Migrated DB now supports same-id episode + cue rows.
        await vectors.upsert("ep-1", "episode", "g", "body", [1.0, 0.0])
        await vectors.upsert("ep-1", "episode_cue", "g", "cue", [0.0, 1.0])
        cursor = await vectors.db.execute(
            "SELECT COUNT(*) AS c FROM embeddings WHERE id = ?", ("ep-1",)
        )
        assert (await cursor.fetchone())["c"] == 2
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_initialize_idempotent_after_migration(tmp_path) -> None:
    """Re-running initialize on an already-composite DB is a no-op (no data loss)."""
    db_path = str(tmp_path / "idem.db")
    vectors = SQLiteVectorStore(db_path)
    await vectors.initialize()
    await vectors.upsert("ent-1", "entity", "g", "kept", [1.0, 0.0])
    await vectors.db.close()

    vectors2 = SQLiteVectorStore(db_path)
    await vectors2.initialize()
    try:
        cursor = await vectors2.db.execute(
            "SELECT COUNT(*) AS c FROM embeddings WHERE id = ?", ("ent-1",)
        )
        assert (await cursor.fetchone())["c"] == 1
    finally:
        await vectors2.db.close()


# --------------------------------------------------------------------------- #
# B5: entity_types filtering in the vector search path
# --------------------------------------------------------------------------- #


async def _create_entities_table(db: aiosqlite.Connection) -> None:
    """Minimal entities table sufficient for the entity_type JOIN."""
    await db.execute(
        """
        CREATE TABLE IF NOT EXISTS entities (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            group_id TEXT NOT NULL DEFAULT 'default'
        )
        """
    )


@pytest.mark.asyncio
async def test_vector_search_entity_types_filters_off_type_entities() -> None:
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        await _create_entities_table(vectors.db)
        # Two entities with near-identical embeddings; only one is an Artifact.
        await vectors.db.execute(
            "INSERT INTO entities (id, name, entity_type, group_id) VALUES (?,?,?,?)",
            ("art-1", "README.md", "Artifact", "g"),
        )
        await vectors.db.execute(
            "INSERT INTO entities (id, name, entity_type, group_id) VALUES (?,?,?,?)",
            ("person-1", "Alpha Person", "Person", "g"),
        )
        await vectors.db.commit()
        await vectors.upsert("art-1", "entity", "g", "alpha artifact", [1.0, 0.0])
        await vectors.upsert("person-1", "entity", "g", "alpha person", [1.0, 0.0])

        # Without the filter both come back (proves the query vector matches both).
        unfiltered = await vectors.search([1.0, 0.0], "g", content_type="entity")
        assert {hit[0] for hit in unfiltered} == {"art-1", "person-1"}

        # With entity_types=['Artifact'] only the Artifact survives.
        filtered = await vectors.search(
            [1.0, 0.0], "g", content_type="entity", entity_types=["Artifact"]
        )
        assert [hit[0] for hit in filtered] == ["art-1"]
        assert "person-1" not in {hit[0] for hit in filtered}
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_hybrid_search_entity_types_excludes_non_matching_via_embeddings() -> None:
    """search() path (used by search_artifacts) must not blend off-type vectors."""
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        await _create_entities_table(vectors.db)
        await vectors.db.execute(
            "INSERT INTO entities (id, name, entity_type, group_id) VALUES (?,?,?,?)",
            ("art-1", "alpha.txt", "Artifact", "g"),
        )
        await vectors.db.execute(
            "INSERT INTO entities (id, name, entity_type, group_id) VALUES (?,?,?,?)",
            ("person-1", "Alpha Person", "Person", "g"),
        )
        await vectors.db.commit()
        await vectors.upsert("art-1", "entity", "g", "alpha artifact", [1.0, 0.0])
        await vectors.upsert("person-1", "entity", "g", "alpha person", [1.0, 0.0])

        index = HybridSearchIndex(
            EmptyFTS(),  # FTS returns nothing, so any hit comes from the vector branch
            vectors,
            TinyEmbeddingProvider(),
            storage_dim=2,
            embed_provider="tiny",
            embed_model="tiny-2",
        )

        results = await index.search("alpha", entity_types=["Artifact"], group_id="g", limit=10)
        result_ids = {eid for eid, _score in results}
        assert "person-1" not in result_ids
        assert result_ids == {"art-1"}
    finally:
        await vectors.db.close()
