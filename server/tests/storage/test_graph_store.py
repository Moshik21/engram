"""Tests for SQLite GraphStore implementation."""

from datetime import datetime

import pytest

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest.mark.asyncio
class TestSQLiteGraphStore:
    async def test_create_and_get_entity(self, graph_store: SQLiteGraphStore):
        entity = Entity(
            id="ent_test1",
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id="default",
        )
        await graph_store.create_entity(entity)
        result = await graph_store.get_entity("ent_test1", "default")
        assert result is not None
        assert result.name == "Python"
        assert result.entity_type == "Technology"

    async def test_tenant_isolation(self, graph_store: SQLiteGraphStore):
        entity = Entity(
            id="ent_iso1",
            name="Secret",
            entity_type="Concept",
            group_id="tenant_a",
        )
        await graph_store.create_entity(entity)

        # Cannot see entity from different group
        result = await graph_store.get_entity("ent_iso1", "tenant_b")
        assert result is None

        # Can see from correct group
        result = await graph_store.get_entity("ent_iso1", "tenant_a")
        assert result is not None

    async def test_find_entities(self, graph_store: SQLiteGraphStore):
        for i in range(3):
            await graph_store.create_entity(
                Entity(
                    id=f"ent_find{i}",
                    name=f"Entity{i}",
                    entity_type="Test",
                    group_id="default",
                )
            )
        results = await graph_store.find_entities(group_id="default")
        assert len(results) == 3

    async def test_find_entities_by_name(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_named", name="JavaScript", entity_type="Technology", group_id="default")
        )
        results = await graph_store.find_entities(name="JavaScript", group_id="default")
        assert len(results) == 1
        assert results[0].name == "JavaScript"

    async def test_update_entity(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_upd", name="Old", entity_type="Test", group_id="default")
        )
        await graph_store.update_entity("ent_upd", {"name": "New"}, group_id="default")
        result = await graph_store.get_entity("ent_upd", "default")
        assert result.name == "New"

    async def test_soft_delete(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_del", name="Doomed", entity_type="Test", group_id="default")
        )
        await graph_store.delete_entity("ent_del", soft=True, group_id="default")
        result = await graph_store.get_entity("ent_del", "default")
        assert result is None  # Soft deleted entities are filtered out

    async def test_create_and_get_relationship(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_src", name="A", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_tgt", name="B", entity_type="Test", group_id="default")
        )
        rel = Relationship(
            id="rel_test1",
            source_id="ent_src",
            target_id="ent_tgt",
            predicate="CONNECTS",
            group_id="default",
        )
        await graph_store.create_relationship(rel)
        rels = await graph_store.get_relationships("ent_src", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].predicate == "CONNECTS"

    async def test_invalidate_relationship(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_inv1", name="X", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_inv2", name="Y", entity_type="Test", group_id="default")
        )
        rel = Relationship(
            id="rel_inv1",
            source_id="ent_inv1",
            target_id="ent_inv2",
            predicate="OLD_REL",
            group_id="default",
        )
        await graph_store.create_relationship(rel)
        await graph_store.invalidate_relationship("rel_inv1", datetime.utcnow(), group_id="default")
        # active_only=False shows the invalidated rel
        rels = await graph_store.get_relationships("ent_inv1", active_only=False)
        assert rels[0].valid_to is not None

    async def test_get_relationships_filters_invalidated(self, graph_store: SQLiteGraphStore):
        """active_only=True (default) should filter out invalidated relationships."""
        await graph_store.create_entity(
            Entity(id="ent_af1", name="A", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_af2", name="B", entity_type="Test", group_id="default")
        )
        rel = Relationship(
            id="rel_af1",
            source_id="ent_af1",
            target_id="ent_af2",
            predicate="TEST_REL",
            group_id="default",
        )
        await graph_store.create_relationship(rel)
        await graph_store.invalidate_relationship("rel_af1", datetime.utcnow(), group_id="default")

        # Default (active_only=True) should return empty
        active = await graph_store.get_relationships("ent_af1")
        assert len(active) == 0

        # active_only=False should return the invalidated rel
        all_rels = await graph_store.get_relationships("ent_af1", active_only=False)
        assert len(all_rels) == 1

    async def test_create_entity_with_pii(self, graph_store: SQLiteGraphStore):
        entity = Entity(
            id="ent_pii_test",
            name="Jake",
            entity_type="Person",
            group_id="default",
            pii_detected=True,
            pii_categories=["phone", "email"],
        )
        await graph_store.create_entity(entity)
        result = await graph_store.get_entity("ent_pii_test", "default")
        assert result is not None
        assert result.pii_detected is True
        assert "phone" in result.pii_categories

    async def test_create_relationship_with_confidence(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_conf1", name="A", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_conf2", name="B", entity_type="Test", group_id="default")
        )
        rel = Relationship(
            id="rel_conf_test",
            source_id="ent_conf1",
            target_id="ent_conf2",
            predicate="KNOWS",
            confidence=0.8,
            group_id="default",
        )
        await graph_store.create_relationship(rel)
        rels = await graph_store.get_relationships("ent_conf1")
        assert len(rels) == 1
        assert rels[0].confidence == 0.8

    async def test_create_and_get_episode(self, graph_store: SQLiteGraphStore):
        ep = Episode(id="ep_test1", content="Test episode content", group_id="default")
        await graph_store.create_episode(ep)
        episodes = await graph_store.get_episodes(group_id="default")
        assert len(episodes) == 1
        assert episodes[0].content == "Test episode content"

    async def test_link_episode_entity(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_link1", name="Linked", entity_type="Test", group_id="default")
        )
        await graph_store.create_episode(
            Episode(id="ep_link1", content="Linked episode", group_id="default")
        )
        await graph_store.link_episode_entity("ep_link1", "ent_link1")
        # No assertion error means the link was created

    async def test_get_stats(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_stat1", name="Stat", entity_type="Test", group_id="default")
        )
        stats = await graph_store.get_stats(group_id="default")
        assert stats["entities"] >= 1
        assert "relationships" in stats
        assert "episodes" in stats

    async def test_update_entity_rejects_invalid_column(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_val", name="X", entity_type="Test", group_id="default")
        )
        with pytest.raises(ValueError, match="Disallowed"):
            await graph_store.update_entity("ent_val", {"DROP TABLE": "bad"}, group_id="default")

    async def test_update_entity_allows_valid_columns(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_val2", name="Y", entity_type="Test", group_id="default")
        )
        await graph_store.update_entity(
            "ent_val2", {"name": "Z", "summary": "new"}, group_id="default",
        )
        result = await graph_store.get_entity("ent_val2", "default")
        assert result.name == "Z"
        assert result.summary == "new"

    async def test_update_episode_rejects_invalid_column(self, graph_store: SQLiteGraphStore):
        ep = Episode(id="ep_val", content="test", group_id="default")
        await graph_store.create_episode(ep)
        with pytest.raises(ValueError, match="Disallowed"):
            await graph_store.update_episode("ep_val", {"DROP TABLE": "bad"})

    async def test_update_episode_wrong_group_id_no_effect(self, graph_store: SQLiteGraphStore):
        """update_episode with wrong group_id should not modify the row."""
        ep = Episode(id="ep_gid1", content="original", group_id="tenant_a")
        await graph_store.create_episode(ep)
        # Try to update with wrong group_id
        await graph_store.update_episode("ep_gid1", {"status": "completed"}, group_id="tenant_b")
        # Verify original status unchanged
        episodes = await graph_store.get_episodes(group_id="tenant_a")
        match = [e for e in episodes if e.id == "ep_gid1"]
        assert len(match) == 1
        assert match[0].status != "completed"

    async def test_update_episode_correct_group_id(self, graph_store: SQLiteGraphStore):
        """update_episode with correct group_id should update the row."""
        ep = Episode(id="ep_gid2", content="test", group_id="tenant_a")
        await graph_store.create_episode(ep)
        await graph_store.update_episode("ep_gid2", {"status": "completed"}, group_id="tenant_a")
        episodes = await graph_store.get_episodes(group_id="tenant_a")
        match = [e for e in episodes if e.id == "ep_gid2"]
        assert len(match) == 1
        assert match[0].status == "completed"

    async def test_get_relationships_always_filters_by_group(
        self, graph_store: SQLiteGraphStore,
    ):
        """get_relationships must always filter by group_id (tenant isolation)."""
        # Create entities in two groups
        await graph_store.create_entity(
            Entity(id="ent_g1", name="A", entity_type="Test", group_id="group1")
        )
        await graph_store.create_entity(
            Entity(id="ent_g2", name="B", entity_type="Test", group_id="group1")
        )
        await graph_store.create_entity(
            Entity(id="ent_g3", name="C", entity_type="Test", group_id="group2")
        )
        await graph_store.create_entity(
            Entity(id="ent_g4", name="D", entity_type="Test", group_id="group2")
        )
        await graph_store.create_relationship(
            Relationship(id="rel_g1", source_id="ent_g1", target_id="ent_g2",
                         predicate="KNOWS", group_id="group1")
        )
        await graph_store.create_relationship(
            Relationship(id="rel_g2", source_id="ent_g3", target_id="ent_g4",
                         predicate="KNOWS", group_id="group2")
        )
        # group1 should only see its own relationship
        rels1 = await graph_store.get_relationships("ent_g1", group_id="group1")
        assert len(rels1) == 1
        assert rels1[0].group_id == "group1"
        # group2 should not see group1's relationships
        rels2 = await graph_store.get_relationships("ent_g1", group_id="group2")
        assert len(rels2) == 0

    async def test_get_relationships_at_always_filters_by_group(
        self, graph_store: SQLiteGraphStore,
    ):
        """get_relationships_at must always filter by group_id."""
        await graph_store.create_entity(
            Entity(id="ent_at1", name="E", entity_type="Test", group_id="grpA")
        )
        await graph_store.create_entity(
            Entity(id="ent_at2", name="F", entity_type="Test", group_id="grpA")
        )
        now = datetime.utcnow()
        await graph_store.create_relationship(
            Relationship(id="rel_at1", source_id="ent_at1", target_id="ent_at2",
                         predicate="LINKS", group_id="grpA", valid_from=now)
        )
        # Correct group sees the relationship
        rels = await graph_store.get_relationships_at("ent_at1", now, group_id="grpA")
        assert len(rels) == 1
        # Wrong group sees nothing
        rels_wrong = await graph_store.get_relationships_at("ent_at1", now, group_id="grpB")
        assert len(rels_wrong) == 0

    async def test_get_neighbors(self, graph_store: SQLiteGraphStore):
        await graph_store.create_entity(
            Entity(id="ent_n1", name="Center", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_n2", name="Neighbor", entity_type="Test", group_id="default")
        )
        await graph_store.create_relationship(
            Relationship(
                id="rel_n1",
                source_id="ent_n1",
                target_id="ent_n2",
                predicate="KNOWS",
                group_id="default",
            )
        )
        neighbors = await graph_store.get_neighbors("ent_n1", hops=1, group_id="default")
        assert len(neighbors) >= 1
        entity_names = [e.name for e, _ in neighbors]
        assert "Neighbor" in entity_names
