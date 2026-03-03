"""Tests for FTS5SearchIndex."""

import pytest

from engram.models.entity import Entity
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest.mark.asyncio
class TestFTS5SearchIndex:
    async def test_search_by_name(
        self,
        graph_store: SQLiteGraphStore,
        search_index: FTS5SearchIndex,
    ):
        await graph_store.create_entity(
            Entity(
                id="ent_py",
                name="Python",
                entity_type="Technology",
                summary="A versatile programming language",
                group_id="default",
            )
        )
        results = await search_index.search("Python", group_id="default")
        assert len(results) >= 1
        assert results[0][0] == "ent_py"

    async def test_search_by_summary(
        self,
        graph_store: SQLiteGraphStore,
        search_index: FTS5SearchIndex,
    ):
        await graph_store.create_entity(
            Entity(
                id="ent_fa",
                name="FastAPI",
                entity_type="Technology",
                summary="Modern web framework for building APIs",
                group_id="default",
            )
        )
        results = await search_index.search("web framework", group_id="default")
        assert len(results) >= 1

    async def test_search_empty_query(self, search_index: FTS5SearchIndex):
        results = await search_index.search("")
        assert results == []

    async def test_search_no_results(self, search_index: FTS5SearchIndex):
        results = await search_index.search("xyznonexistent")
        assert results == []

    async def test_search_with_type_filter(
        self,
        graph_store: SQLiteGraphStore,
        search_index: FTS5SearchIndex,
    ):
        await graph_store.create_entity(
            Entity(
                id="ent_person",
                name="Alice",
                entity_type="Person",
                group_id="default",
            )
        )
        await graph_store.create_entity(
            Entity(
                id="ent_tech",
                name="Alice Framework",
                entity_type="Technology",
                group_id="default",
            )
        )
        results = await search_index.search("Alice", entity_types=["Person"], group_id="default")
        assert all(r[0] == "ent_person" for r in results)

    async def test_scores_normalized(
        self,
        graph_store: SQLiteGraphStore,
        search_index: FTS5SearchIndex,
    ):
        await graph_store.create_entity(
            Entity(id="ent_norm", name="Test", entity_type="Test", group_id="default")
        )
        results = await search_index.search("Test", group_id="default")
        if results:
            for _, score in results:
                assert 0.0 <= score <= 1.0
