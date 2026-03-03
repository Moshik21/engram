"""Tests for FalkorDB consolidation methods (requires Docker)."""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]


def _make_store():
    """Create a FalkorDB store pointing at local Docker instance."""
    from engram.config import FalkorDBConfig
    from engram.storage.falkordb.graph import FalkorDBGraphStore

    cfg = FalkorDBConfig(
        host="localhost",
        port=6380,
        graph_name=f"test_{uuid.uuid4().hex[:8]}",
    )
    return FalkorDBGraphStore(cfg)


def _entity(name: str, entity_type: str = "Person", group_id: str = "test") -> Entity:
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
    )


def _episode(group_id: str = "test") -> Episode:
    return Episode(
        id=f"ep_{uuid.uuid4().hex[:8]}",
        content="test content",
        source="test",
        status="completed",
        group_id=group_id,
    )


@pytest_asyncio.fixture
async def store():
    s = _make_store()
    await s.initialize()
    yield s
    await s.close()


class TestFalkorDBCoOccurrence:
    async def test_basic_co_occurrence(self, store):
        """Finds pair with 3+ shared episodes."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        for _ in range(3):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        pairs = await store.get_co_occurring_entity_pairs(
            group_id="test",
            min_co_occurrence=3,
        )
        assert len(pairs) >= 1
        ids = {(p[0], p[1]) for p in pairs}
        # Canonical ordering: smaller id first
        canonical = tuple(sorted([e1.id, e2.id]))
        assert canonical in ids

    async def test_excludes_existing_rel(self, store):
        """Pair with existing RELATES_TO is excluded."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Create existing relationship
        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e1.id,
            target_id=e2.id,
            predicate="KNOWS",
            group_id="test",
        )
        await store.create_relationship(rel)

        for _ in range(5):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        pairs = await store.get_co_occurring_entity_pairs(
            group_id="test",
            min_co_occurrence=3,
        )
        pair_ids = {(p[0], p[1]) for p in pairs}
        canonical = tuple(sorted([e1.id, e2.id]))
        assert canonical not in pair_ids


class TestFalkorDBDeadEntities:
    async def test_finds_dead_entity(self, store):
        """Old, unconnected, 0-access entity found."""
        from datetime import datetime, timedelta

        e = _entity("Forgotten")
        # Override created_at to be old
        e.created_at = datetime.utcnow() - timedelta(days=60)
        e.access_count = 0
        await store.create_entity(e)

        dead = await store.get_dead_entities(group_id="test", min_age_days=30)
        dead_ids = {d.id for d in dead}
        assert e.id in dead_ids

    async def test_connected_not_dead(self, store):
        """Entity with active relationship excluded."""
        from datetime import datetime, timedelta

        e1 = _entity("Connected")
        e1.created_at = datetime.utcnow() - timedelta(days=60)
        e1.access_count = 0
        e2 = _entity("Other")
        await store.create_entity(e1)
        await store.create_entity(e2)

        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e1.id,
            target_id=e2.id,
            predicate="KNOWS",
            group_id="test",
        )
        await store.create_relationship(rel)

        dead = await store.get_dead_entities(group_id="test", min_age_days=30)
        dead_ids = {d.id for d in dead}
        assert e1.id not in dead_ids


class TestFalkorDBMergeEntities:
    async def test_merge_repoints_and_deletes(self, store):
        """Rels transferred, loser soft-deleted."""
        keeper = _entity("Alice")
        loser = _entity("alice")
        other = _entity("Charlie")
        await store.create_entity(keeper)
        await store.create_entity(loser)
        await store.create_entity(other)

        # Loser has an outgoing relationship to other
        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=loser.id,
            target_id=other.id,
            predicate="KNOWS",
            group_id="test",
        )
        await store.create_relationship(rel)

        # Loser linked to an episode
        ep = _episode()
        await store.create_episode(ep)
        await store.link_episode_entity(ep.id, loser.id)

        count = await store.merge_entities(keeper.id, loser.id, "test")
        assert count >= 1

        # Keeper should now have the relationship
        rels = await store.get_relationships(keeper.id, group_id="test")
        assert any(r.predicate == "KNOWS" for r in rels)

        # Loser should be soft-deleted
        loser_entity = await store.get_entity(loser.id, "test")
        assert loser_entity is not None
        assert loser_entity.deleted_at is not None
