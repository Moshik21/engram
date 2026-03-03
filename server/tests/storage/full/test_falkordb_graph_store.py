"""Tests for FalkorDB GraphStore implementation."""

from datetime import datetime

import pytest

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship

pytestmark = pytest.mark.requires_docker


@pytest.mark.asyncio
class TestFalkorDBGraphStore:
    async def test_create_and_get_entity(self, falkordb_graph_store):
        entity = Entity(
            id="ent_fdb1",
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id="default",
        )
        await falkordb_graph_store.create_entity(entity)
        result = await falkordb_graph_store.get_entity("ent_fdb1", "default")
        assert result is not None
        assert result.name == "Python"
        assert result.entity_type == "Technology"
        assert result.summary == "Programming language"

    async def test_tenant_isolation(self, falkordb_graph_store):
        entity = Entity(
            id="ent_iso1",
            name="Secret",
            entity_type="Concept",
            group_id="tenant_a",
        )
        await falkordb_graph_store.create_entity(entity)

        # Cannot see entity from different group
        result = await falkordb_graph_store.get_entity("ent_iso1", "tenant_b")
        assert result is None

        # Can see from correct group
        result = await falkordb_graph_store.get_entity("ent_iso1", "tenant_a")
        assert result is not None

    async def test_update_entity(self, falkordb_graph_store):
        entity = Entity(
            id="ent_upd1",
            name="OldName",
            entity_type="Person",
            group_id="default",
        )
        await falkordb_graph_store.create_entity(entity)
        await falkordb_graph_store.update_entity(
            "ent_upd1", {"name": "NewName"}, group_id="default"
        )
        result = await falkordb_graph_store.get_entity("ent_upd1", "default")
        assert result is not None
        assert result.name == "NewName"

    async def test_soft_delete_entity(self, falkordb_graph_store):
        entity = Entity(
            id="ent_del1",
            name="ToDelete",
            entity_type="Concept",
            group_id="default",
        )
        await falkordb_graph_store.create_entity(entity)
        await falkordb_graph_store.delete_entity("ent_del1", soft=True, group_id="default")
        result = await falkordb_graph_store.get_entity("ent_del1", "default")
        assert result is None  # Soft-deleted, not visible

    async def test_hard_delete_entity(self, falkordb_graph_store):
        entity = Entity(
            id="ent_hdel1",
            name="HardDelete",
            entity_type="Concept",
            group_id="default",
        )
        await falkordb_graph_store.create_entity(entity)
        await falkordb_graph_store.delete_entity("ent_hdel1", soft=False, group_id="default")
        # Verify it's completely gone (not just soft-deleted)
        result = await falkordb_graph_store._query(
            "MATCH (n:Entity {id: $id}) RETURN n", {"id": "ent_hdel1"}
        )
        assert len(result.result_set) == 0

    async def test_find_entities(self, falkordb_graph_store):
        for i in range(3):
            await falkordb_graph_store.create_entity(
                Entity(
                    id=f"ent_find{i}",
                    name=f"Item{i}",
                    entity_type="Technology",
                    group_id="default",
                )
            )
        results = await falkordb_graph_store.find_entities(
            entity_type="Technology", group_id="default"
        )
        assert len(results) == 3

    async def test_find_entities_by_name(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_fn1", name="React", entity_type="Technology", group_id="default")
        )
        # Case-insensitive match
        results = await falkordb_graph_store.find_entities(name="react", group_id="default")
        assert len(results) == 1
        assert results[0].name == "React"

    async def test_create_and_get_relationship(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_r1", name="A", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_r2", name="B", entity_type="Company", group_id="default")
        )
        rel = Relationship(
            id="rel_1",
            source_id="ent_r1",
            target_id="ent_r2",
            predicate="WORKS_AT",
            weight=1.0,
            group_id="default",
        )
        await falkordb_graph_store.create_relationship(rel)
        rels = await falkordb_graph_store.get_relationships("ent_r1", direction="outgoing")
        assert len(rels) == 1
        assert rels[0].predicate == "WORKS_AT"

    async def test_invalidate_relationship(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_inv1", name="C", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_inv2", name="D", entity_type="City", group_id="default")
        )
        rel = Relationship(
            id="rel_inv1",
            source_id="ent_inv1",
            target_id="ent_inv2",
            predicate="LIVES_IN",
            group_id="default",
        )
        await falkordb_graph_store.create_relationship(rel)
        now = datetime.utcnow()
        await falkordb_graph_store.invalidate_relationship("rel_inv1", now, group_id="default")
        # Active-only should return empty
        active_rels = await falkordb_graph_store.get_relationships("ent_inv1", active_only=True)
        assert len(active_rels) == 0
        # All rels should still return it
        all_rels = await falkordb_graph_store.get_relationships("ent_inv1", active_only=False)
        assert len(all_rels) == 1

    async def test_find_conflicting_relationships(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_cf1", name="E", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_cf2", name="F", entity_type="City", group_id="default")
        )
        rel = Relationship(
            id="rel_cf1",
            source_id="ent_cf1",
            target_id="ent_cf2",
            predicate="LIVES_IN",
            group_id="default",
        )
        await falkordb_graph_store.create_relationship(rel)
        conflicts = await falkordb_graph_store.find_conflicting_relationships(
            "ent_cf1", "LIVES_IN", "default"
        )
        assert len(conflicts) == 1

    async def test_get_neighbors(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_n1", name="Center", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_n2", name="Neighbor1", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_n3", name="Neighbor2", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_relationship(
            Relationship(
                id="rel_n1",
                source_id="ent_n1",
                target_id="ent_n2",
                predicate="KNOWS",
                group_id="default",
            )
        )
        await falkordb_graph_store.create_relationship(
            Relationship(
                id="rel_n2",
                source_id="ent_n2",
                target_id="ent_n3",
                predicate="KNOWS",
                group_id="default",
            )
        )
        # 1-hop: should find ent_n2
        neighbors_1 = await falkordb_graph_store.get_neighbors("ent_n1", hops=1)
        neighbor_ids_1 = {e.id for e, _ in neighbors_1}
        assert "ent_n2" in neighbor_ids_1

        # 2-hop: should find ent_n2 and ent_n3
        neighbors_2 = await falkordb_graph_store.get_neighbors("ent_n1", hops=2)
        neighbor_ids_2 = {e.id for e, _ in neighbors_2}
        assert "ent_n2" in neighbor_ids_2
        assert "ent_n3" in neighbor_ids_2

    async def test_episode_crud(self, falkordb_graph_store):
        ep = Episode(
            id="ep_1",
            content="User discussed Python programming",
            source="chat",
            group_id="default",
        )
        await falkordb_graph_store.create_episode(ep)
        result = await falkordb_graph_store.get_episode_by_id("ep_1", "default")
        assert result is not None
        assert "Python" in result.content

    async def test_link_episode_entity(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_le1", name="Go", entity_type="Technology", group_id="default")
        )
        await falkordb_graph_store.create_episode(
            Episode(id="ep_le1", content="Discussed Go", source="chat", group_id="default")
        )
        await falkordb_graph_store.link_episode_entity("ep_le1", "ent_le1")
        entities = await falkordb_graph_store.get_episode_entities("ep_le1")
        assert "ent_le1" in entities

    async def test_get_stats(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_st1", name="Stats", entity_type="Concept", group_id="default")
        )
        await falkordb_graph_store.create_episode(
            Episode(id="ep_st1", content="Stats ep", source="chat", group_id="default")
        )
        stats = await falkordb_graph_store.get_stats(group_id="default")
        assert stats["entities"] >= 1
        assert stats["episodes"] >= 1

    async def test_get_top_connected(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_tc1", name="Hub", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_entity(
            Entity(id="ent_tc2", name="Spoke", entity_type="Person", group_id="default")
        )
        await falkordb_graph_store.create_relationship(
            Relationship(
                id="rel_tc1",
                source_id="ent_tc1",
                target_id="ent_tc2",
                predicate="KNOWS",
                group_id="default",
            )
        )
        top = await falkordb_graph_store.get_top_connected(group_id="default", limit=5)
        assert len(top) >= 1
        # Hub should have at least 1 edge
        hub = next((e for e in top if e["id"] == "ent_tc1"), None)
        assert hub is not None
        assert hub["edgeCount"] >= 1

    async def test_initialize_is_idempotent(self, falkordb_graph_store):
        # Calling initialize again should not raise
        await falkordb_graph_store.initialize()

    async def test_get_growth_timeline(self, falkordb_graph_store):
        await falkordb_graph_store.create_entity(
            Entity(id="ent_gt1", name="Timeline", entity_type="Concept", group_id="default")
        )
        await falkordb_graph_store.create_episode(
            Episode(id="ep_gt1", content="Timeline ep", source="chat", group_id="default")
        )
        timeline = await falkordb_graph_store.get_growth_timeline(group_id="default", days=7)
        assert isinstance(timeline, list)
