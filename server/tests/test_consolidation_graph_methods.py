"""Tests for new GraphStore methods used by consolidation."""

import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


def _entity(name, entity_type="Person", group_id="test", **kwargs):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
        **kwargs,
    )


def _episode(group_id="test"):
    return Episode(
        id=f"ep_{uuid.uuid4().hex[:8]}",
        content="test episode",
        source="test",
        status="completed",
        group_id=group_id,
    )


def _rel(source_id, target_id, predicate="KNOWS", group_id="test"):
    return Relationship(
        id=f"rel_{uuid.uuid4().hex[:8]}",
        source_id=source_id,
        target_id=target_id,
        predicate=predicate,
        group_id=group_id,
    )


class TestCoOccurringEntityPairs:
    """Tests for get_co_occurring_entity_pairs."""

    @pytest.mark.asyncio
    async def test_finds_co_occurring_pairs(self, store):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Create 3 episodes linking both entities
        for _ in range(3):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        pairs = await store.get_co_occurring_entity_pairs("test", min_co_occurrence=3)
        assert len(pairs) == 1
        ids = {pairs[0][0], pairs[0][1]}
        assert ids == {e1.id, e2.id}
        assert pairs[0][2] == 3

    @pytest.mark.asyncio
    async def test_excludes_existing_relationships(self, store):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Create relationship between them
        await store.create_relationship(_rel(e1.id, e2.id, group_id="test"))

        # Create 5 co-occurring episodes
        for _ in range(5):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        pairs = await store.get_co_occurring_entity_pairs("test", min_co_occurrence=3)
        assert len(pairs) == 0  # Excluded because relationship exists

    @pytest.mark.asyncio
    async def test_below_threshold_excluded(self, store):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Only 2 co-occurrences (below min_co_occurrence=3)
        for _ in range(2):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        pairs = await store.get_co_occurring_entity_pairs("test", min_co_occurrence=3)
        assert len(pairs) == 0


class TestDeadEntities:
    """Tests for get_dead_entities."""

    @pytest.mark.asyncio
    async def test_finds_dead_entities(self, store):
        old_time = (datetime.utcnow() - timedelta(days=60)).isoformat()
        e = _entity("Ghost", access_count=0)
        e.created_at = datetime.fromisoformat(old_time)
        await store.create_entity(e)
        # Manually update created_at to be old
        await store.db.execute(
            "UPDATE entities SET created_at = ? WHERE id = ?",
            (old_time, e.id),
        )
        await store.db.commit()

        dead = await store.get_dead_entities("test", min_age_days=30)
        assert len(dead) == 1
        assert dead[0].id == e.id

    @pytest.mark.asyncio
    async def test_entity_with_relationships_not_dead(self, store):
        old_time = (datetime.utcnow() - timedelta(days=60)).isoformat()
        e1 = _entity("Connected")
        e2 = _entity("Other")
        await store.create_entity(e1)
        await store.create_entity(e2)
        await store.create_relationship(_rel(e1.id, e2.id, group_id="test"))

        await store.db.execute(
            "UPDATE entities SET created_at = ?, access_count = 0 WHERE id = ?",
            (old_time, e1.id),
        )
        await store.db.commit()

        dead = await store.get_dead_entities("test", min_age_days=30)
        # e1 has relationships, should not be returned
        dead_ids = {d.id for d in dead}
        assert e1.id not in dead_ids

    @pytest.mark.asyncio
    async def test_recent_entity_not_dead(self, store):
        e = _entity("Recent", access_count=0)
        await store.create_entity(e)

        dead = await store.get_dead_entities("test", min_age_days=30)
        assert len(dead) == 0  # Too recent


class TestMergeEntities:
    """Tests for merge_entities."""

    @pytest.mark.asyncio
    async def test_repoints_relationships(self, store):
        e1 = _entity("Alice")
        e2 = _entity("alice")
        e3 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)
        await store.create_entity(e3)

        # e2 → e3 relationship
        await store.create_relationship(_rel(e2.id, e3.id, group_id="test"))

        count = await store.merge_entities(e1.id, e2.id, "test")
        assert count >= 1  # At least one relationship re-pointed

        # e2 should be soft-deleted
        deleted = await store.get_entity(e2.id, "test")
        assert deleted is None  # Soft-deleted, not returned

        # e1 should have the relationship now
        rels = await store.get_relationships(e1.id, group_id="test")
        assert len(rels) >= 1

    @pytest.mark.asyncio
    async def test_merges_summaries(self, store):
        e1 = _entity("Alice", summary="Works at Acme")
        e2 = _entity("alice", summary="Lives in NYC")
        await store.create_entity(e1)
        await store.create_entity(e2)

        await store.merge_entities(e1.id, e2.id, "test")

        merged = await store.get_entity(e1.id, "test")
        assert "Acme" in merged.summary
        assert "NYC" in merged.summary

    @pytest.mark.asyncio
    async def test_removes_self_loops(self, store):
        e1 = _entity("Alice")
        e2 = _entity("alice")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Create e1 → e2 relationship (would become self-loop after merge)
        await store.create_relationship(_rel(e1.id, e2.id, group_id="test"))

        await store.merge_entities(e1.id, e2.id, "test")

        # Should not have self-loops
        rels = await store.get_relationships(e1.id, group_id="test")
        for r in rels:
            assert not (r.source_id == r.target_id)
