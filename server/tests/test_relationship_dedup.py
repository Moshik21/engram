"""Tests for relationship dedup and meta-contamination guards."""

from __future__ import annotations

import pytest

from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.utils.text_guards import is_meta_summary

GROUP = "default"


@pytest.fixture
async def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    s = SQLiteGraphStore(db_path)
    await s.initialize()
    yield s
    await s.close()


def _entity(eid: str, name: str, entity_type: str = "Person", summary: str = "") -> Entity:
    return Entity(id=eid, name=name, entity_type=entity_type, summary=summary, group_id=GROUP)


def _rel(
    rid: str,
    source: str,
    target: str,
    predicate: str = "PARENT_OF",
    weight: float = 1.0,
) -> Relationship:
    return Relationship(
        id=rid,
        source_id=source,
        target_id=target,
        predicate=predicate,
        weight=weight,
        group_id=GROUP,
    )


# --- find_existing_relationship ---


@pytest.mark.asyncio
async def test_find_existing_relationship_found(store):
    await store.create_entity(_entity("e1", "A"))
    await store.create_entity(_entity("e2", "B"))
    await store.create_relationship(_rel("r1", "e1", "e2", "PARENT_OF"))

    found = await store.find_existing_relationship("e1", "e2", "PARENT_OF", GROUP)
    assert found is not None
    assert found.id == "r1"


@pytest.mark.asyncio
async def test_find_existing_relationship_not_found(store):
    await store.create_entity(_entity("e1", "A"))
    await store.create_entity(_entity("e2", "B"))
    await store.create_relationship(_rel("r1", "e1", "e2", "PARENT_OF"))

    # Different predicate
    found = await store.find_existing_relationship("e1", "e2", "CHILD_OF", GROUP)
    assert found is None


@pytest.mark.asyncio
async def test_find_existing_relationship_ignores_expired(store):
    from datetime import datetime, timedelta

    await store.create_entity(_entity("e1", "A"))
    await store.create_entity(_entity("e2", "B"))
    await store.create_relationship(_rel("r1", "e1", "e2", "PARENT_OF"))

    # Invalidate it
    await store.invalidate_relationship(
        "r1", datetime.utcnow() - timedelta(seconds=1), GROUP
    )

    found = await store.find_existing_relationship("e1", "e2", "PARENT_OF", GROUP)
    assert found is None


# --- merge_entities dedup ---


@pytest.mark.asyncio
async def test_merge_entities_deduplicates_edges(store):
    """When merging B into A, if both A→X and B→X exist with same predicate, keep one."""
    await store.create_entity(_entity("a", "A"))
    await store.create_entity(_entity("b", "B"))
    await store.create_entity(_entity("x", "X"))
    await store.create_relationship(_rel("r1", "a", "x", "KNOWS"))
    await store.create_relationship(_rel("r2", "b", "x", "KNOWS"))

    await store.merge_entities("a", "b", GROUP)

    rels = await store.get_relationships("a", direction="outgoing", group_id=GROUP)
    knows_rels = [r for r in rels if r.predicate == "KNOWS"]
    assert len(knows_rels) == 1


@pytest.mark.asyncio
async def test_merge_entities_deduplicates_incoming(store):
    """When merging B into A, incoming edges get deduped too."""
    await store.create_entity(_entity("a", "A"))
    await store.create_entity(_entity("b", "B"))
    await store.create_entity(_entity("x", "X"))
    await store.create_relationship(_rel("r1", "x", "a", "FRIEND_OF"))
    await store.create_relationship(_rel("r2", "x", "b", "FRIEND_OF"))

    await store.merge_entities("a", "b", GROUP)

    rels = await store.get_relationships("a", direction="incoming", group_id=GROUP)
    friend_rels = [r for r in rels if r.predicate == "FRIEND_OF"]
    assert len(friend_rels) == 1


# --- Meta-contamination guards in merge ---


@pytest.mark.asyncio
async def test_meta_summary_rejected_in_merge_entities(store):
    """Merge entity with meta-contaminated summary — keeper summary unchanged."""
    await store.create_entity(
        _entity("a", "Konner", summary="Father of four")
    )
    await store.create_entity(
        _entity("b", "Konner2", summary="activation score 0.85 knowledge graph entity")
    )

    await store.merge_entities("a", "b", GROUP)

    keeper = await store.get_entity("a", GROUP)
    assert keeper is not None
    assert keeper.summary == "Father of four"
    assert "activation" not in keeper.summary


@pytest.mark.asyncio
async def test_merge_summary_500_char_cap(store):
    """Long summaries get truncated to 500 chars."""
    long_summary = "x" * 300
    await store.create_entity(
        _entity("a", "A", summary="y" * 250)
    )
    await store.create_entity(
        _entity("b", "B", summary=long_summary)
    )

    await store.merge_entities("a", "b", GROUP)

    keeper = await store.get_entity("a", GROUP)
    assert keeper is not None
    assert len(keeper.summary) <= 500
    assert keeper.summary.endswith("...")


# --- Universal meta-guard (Step 7) ---


@pytest.mark.asyncio
async def test_meta_summary_rejected_for_all_types(store):
    """Technology/Software entities also reject meta-summaries."""
    from engram.graph_manager import GraphManager

    tech_entity = _entity(
        "t1", "Python", entity_type="Technology", summary="A programming language",
    )
    meta_summary = "activation score 0.5 knowledge graph entity"

    updates = GraphManager._merge_entity_attributes(tech_entity, meta_summary)
    assert "summary" not in updates  # Should have been rejected


# --- is_meta_summary utility ---


def test_is_meta_summary_detects_patterns():
    assert is_meta_summary("activation score 0.85") is True
    assert is_meta_summary("knowledge graph entity") is True
    assert is_meta_summary("episode worker processing") is True
    assert is_meta_summary("entity ent_abc123def") is True


def test_is_meta_summary_allows_normal():
    assert is_meta_summary("Father of four sons") is False
    assert is_meta_summary("Software engineer at Google") is False
    assert is_meta_summary("Lives in Portland") is False


# --- project_episode dedup (integration-style) ---


@pytest.mark.asyncio
async def test_project_episode_skips_duplicate_relationship(store):
    """Ingesting same relationship twice should result in 1 active relationship."""
    await store.create_entity(_entity("e1", "Konner"))
    await store.create_entity(_entity("e2", "Kallon"))

    # Create first relationship
    await store.create_relationship(_rel("r1", "e1", "e2", "PARENT_OF"))

    # Simulate what project_episode now does: check before creating
    existing = await store.find_existing_relationship("e1", "e2", "PARENT_OF", GROUP)
    assert existing is not None  # Should find it

    # So project_episode would skip creation. Verify only 1 exists.
    rels = await store.get_relationships("e1", direction="outgoing", group_id=GROUP)
    parent_rels = [r for r in rels if r.predicate == "PARENT_OF"]
    assert len(parent_rels) == 1
