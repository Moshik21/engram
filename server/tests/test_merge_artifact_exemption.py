"""Tests for Artifact/Schema exemption from the merge phase (M2.4)."""

import uuid

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.merge import EntityMergePhase
from engram.models.entity import Entity
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


@pytest.fixture
def gid():
    return f"test_{uuid.uuid4().hex[:8]}"


def _entity(name, entity_type="Concept", group_id="test", attributes=None):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
        attributes=attributes or {},
    )


async def _run_merge(store, activation, search, gid):
    cfg = ActivationConfig(consolidation_merge_threshold=0.85)
    phase = EntityMergePhase()
    return await phase.execute(
        group_id=gid,
        graph_store=store,
        activation_store=activation,
        search_index=search,
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )


@pytest.mark.asyncio
async def test_same_named_artifacts_from_two_projects_stay_unmerged(store, activation, search, gid):
    a = _entity(
        "README.md",
        entity_type="Artifact",
        group_id=gid,
        attributes={
            "project_path": "/proj/a",
            "rel_path": "README.md",
            "content_hash": "hash_a",
        },
    )
    b = _entity(
        "README.md",
        entity_type="Artifact",
        group_id=gid,
        attributes={
            "project_path": "/proj/b",
            "rel_path": "README.md",
            "content_hash": "hash_b",
        },
    )
    await store.create_entity(a)
    await store.create_entity(b)

    result, records = await _run_merge(store, activation, search, gid)

    assert result.items_affected == 0
    assert records == []
    kept_a = await store.get_entity(a.id, gid)
    kept_b = await store.get_entity(b.id, gid)
    assert kept_a is not None and kept_b is not None
    assert (kept_a.attributes or {}).get("project_path") == "/proj/a"
    assert (kept_b.attributes or {}).get("project_path") == "/proj/b"
    assert (kept_a.attributes or {}).get("content_hash") == "hash_a"
    assert (kept_b.attributes or {}).get("content_hash") == "hash_b"


@pytest.mark.asyncio
async def test_same_named_schemas_stay_unmerged(store, activation, search, gid):
    a = _entity("Schema: project-decision", entity_type="Schema", group_id=gid)
    b = _entity("Schema: project-decision", entity_type="Schema", group_id=gid)
    await store.create_entity(a)
    await store.create_entity(b)

    result, _records = await _run_merge(store, activation, search, gid)

    assert result.items_affected == 0
    assert await store.get_entity(a.id, gid) is not None
    assert await store.get_entity(b.id, gid) is not None


@pytest.mark.asyncio
async def test_non_exempt_types_still_merge(store, activation, search, gid):
    """Control: the exemption must not disable merging for other types."""
    a = _entity("John Smith", entity_type="Person", group_id=gid)
    b = _entity("john smith", entity_type="Person", group_id=gid)
    await store.create_entity(a)
    await store.create_entity(b)

    result, records = await _run_merge(store, activation, search, gid)

    assert result.items_affected == 1
    assert len(records) == 1
