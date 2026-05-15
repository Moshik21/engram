from __future__ import annotations

import aiosqlite
import pytest

from engram.embeddings.graph.storage import GraphEmbeddingStore
from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore


class TinyEmbeddingProvider:
    def dimension(self) -> int:
        return 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


class EmptyFTS:
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


@pytest.mark.asyncio
async def test_hybrid_search_close_does_not_close_borrowed_db_connection(tmp_path) -> None:
    db = await aiosqlite.connect(tmp_path / "search.db")
    db.row_factory = aiosqlite.Row
    index = HybridSearchIndex(
        FTS5SearchIndex(str(tmp_path / "search.db")),
        SQLiteVectorStore(str(tmp_path / "search.db")),
        TinyEmbeddingProvider(),
        storage_dim=2,
        embed_provider="tiny",
        embed_model="tiny-2",
    )
    await index.initialize(db=db)
    try:
        await index.close()

        row = await (await db.execute("SELECT COUNT(*) FROM embeddings")).fetchone()
        assert row[0] == 0
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_hybrid_search_close_clears_owned_component_connections(tmp_path) -> None:
    index = HybridSearchIndex(
        FTS5SearchIndex(str(tmp_path / "owned-search.db")),
        SQLiteVectorStore(str(tmp_path / "owned-search.db")),
        TinyEmbeddingProvider(),
        storage_dim=2,
        embed_provider="tiny",
        embed_model="tiny-2",
    )
    await index.initialize()

    await index.close()

    with pytest.raises(RuntimeError, match="FTS5SearchIndex not initialized"):
        _ = index._fts.db
    with pytest.raises(RuntimeError, match="SQLiteVectorStore not initialized"):
        _ = index._vectors.db


@pytest.mark.asyncio
async def test_sqlite_vector_search_omitted_group_reads_all_groups() -> None:
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        await vectors.upsert("ent-alpha", "entity", "brain-a", "alpha", [1.0, 0.0])
        await vectors.upsert("ent-beta", "entity", "brain-b", "beta", [0.0, 1.0])

        all_group_results = await vectors.search(
            [1.0, 0.0],
            group_id=None,
            content_type="entity",
            limit=10,
        )
        scoped_results = await vectors.search(
            [1.0, 0.0],
            group_id="brain-a",
            content_type="entity",
            limit=10,
        )

        assert {item_id for item_id, _score in all_group_results} == {
            "ent-alpha",
            "ent-beta",
        }
        assert [item_id for item_id, _score in scoped_results] == ["ent-alpha"]
    finally:
        await vectors.db.close()


@pytest.mark.asyncio
async def test_sqlite_hybrid_omitted_group_uses_all_group_vectors() -> None:
    vectors = SQLiteVectorStore(":memory:")
    await vectors.initialize()
    try:
        await vectors.upsert("ent-alpha", "entity", "brain-a", "alpha", [1.0, 0.0])
        await vectors.upsert("ep-alpha", "episode", "brain-a", "alpha", [1.0, 0.0])
        await vectors.upsert(
            "cue-alpha",
            "episode_cue",
            "brain-b",
            "alpha cue",
            [1.0, 0.0],
        )

        graph_store = GraphEmbeddingStore()
        await graph_store.initialize(vectors.db)
        await graph_store.upsert_batch(
            vectors.db,
            {"ent-alpha": [0.25, 0.75]},
            method="node2vec",
            group_id="brain-a",
        )

        index = HybridSearchIndex(
            EmptyFTS(),
            vectors,
            TinyEmbeddingProvider(),
            storage_dim=2,
            embed_provider="tiny",
            embed_model="tiny-2",
        )

        entity_results = await index.search("alpha", group_id=None, limit=5)
        episode_results = await index.search_episodes(
            "alpha",
            group_id=None,
            limit=5,
        )
        cue_results = await index.search_episode_cues(
            "alpha",
            group_id=None,
            limit=5,
        )
        entity_embeddings = await index.get_entity_embeddings(
            ["ent-alpha"],
            group_id=None,
        )
        graph_embeddings = await index.get_graph_embeddings(
            ["ent-alpha"],
            group_id=None,
        )
        similarities = await index.compute_similarity(
            "alpha",
            ["ent-alpha"],
            group_id=None,
        )

        assert entity_results[0][0] == "ent-alpha"
        assert episode_results[0][0] == "ep-alpha"
        assert cue_results[0][0] == "cue-alpha"
        assert entity_embeddings["ent-alpha"] == pytest.approx([1.0, 0.0])
        assert graph_embeddings["ent-alpha"] == pytest.approx([0.25, 0.75])
        assert similarities["ent-alpha"] == pytest.approx(1.0)
    finally:
        await vectors.db.close()
