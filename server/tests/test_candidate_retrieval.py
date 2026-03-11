"""Tests for find_entity_candidates in SQLite graph store."""

import pytest

from engram.models.entity import Entity
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest.fixture
async def graph():
    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


async def _create_entity(graph, name, entity_type="Technology", group_id="default"):
    import uuid

    entity = Entity(
        id=f"ent_{uuid.uuid4().hex[:12]}",
        name=name,
        entity_type=entity_type,
        summary=f"A {entity_type.lower()} called {name}",
        group_id=group_id,
    )
    await graph.create_entity(entity)
    return entity


@pytest.mark.asyncio
class TestFindEntityCandidates:
    async def test_exact_match(self, graph):
        await _create_entity(graph, "Python")
        candidates = await graph.find_entity_candidates("Python", "default")
        assert len(candidates) == 1
        assert candidates[0].name == "Python"

    async def test_case_insensitive(self, graph):
        await _create_entity(graph, "Python")
        candidates = await graph.find_entity_candidates("python", "default")
        assert len(candidates) >= 1
        assert any(c.name == "Python" for c in candidates)

    async def test_hyphen_underscore_normalization(self, graph):
        await _create_entity(graph, "ACT-R")
        candidates = await graph.find_entity_candidates("ACT_R", "default")
        assert len(candidates) >= 1
        assert any(c.name == "ACT-R" for c in candidates)

    async def test_fts5_token_match(self, graph):
        await _create_entity(graph, "React Framework")
        candidates = await graph.find_entity_candidates("React", "default")
        assert len(candidates) >= 1
        assert any("React" in c.name for c in candidates)

    async def test_group_id_isolation(self, graph):
        await _create_entity(graph, "Python", group_id="group_a")
        await _create_entity(graph, "Python", group_id="group_b")
        candidates_a = await graph.find_entity_candidates("Python", "group_a")
        candidates_b = await graph.find_entity_candidates("Python", "group_b")
        assert len(candidates_a) == 1
        assert len(candidates_b) == 1
        assert candidates_a[0].id != candidates_b[0].id

    async def test_excludes_deleted(self, graph):
        entity = await _create_entity(graph, "OldTech")
        await graph.delete_entity(entity.id, soft=True, group_id="default")
        candidates = await graph.find_entity_candidates("OldTech", "default")
        assert len(candidates) == 0

    async def test_respects_limit(self, graph):
        for i in range(10):
            await _create_entity(graph, f"Item{i}")
        candidates = await graph.find_entity_candidates("Item", "default", limit=3)
        assert len(candidates) <= 3

    async def test_no_match(self, graph):
        await _create_entity(graph, "Python")
        candidates = await graph.find_entity_candidates("Xylophone", "default")
        assert len(candidates) == 0

    async def test_multiple_token_match(self, graph):
        await _create_entity(graph, "Spreading Activation")
        candidates = await graph.find_entity_candidates(
            "ACT-R Spreading Activation",
            "default",
        )
        assert len(candidates) >= 1
        assert any("Spreading" in c.name for c in candidates)
