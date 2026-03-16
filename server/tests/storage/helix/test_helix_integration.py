"""End-to-end integration tests for the Helix backend.

Exercises the full lifecycle: entity creation, episode creation, linking,
entity search, and data round-trip verification.  Also runs the shared
``GraphStoreContractTests`` suite against HelixGraphStore.

All tests require a running HelixDB instance on localhost:6969 and are
marked with ``requires_helix``.
"""

from __future__ import annotations

import socket
from uuid import uuid4

import pytest

from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship
from tests.storage.contract import GraphStoreContractTests


def helix_available() -> bool:
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not helix_available(), reason="HelixDB not available"),
]


def _uid() -> str:
    return uuid4().hex[:12]


# ======================================================================
# Contract suite -- reuse shared mixin
# ======================================================================


@pytest.mark.asyncio
class TestHelixContractSuite(GraphStoreContractTests):
    """Run the full GraphStoreContractTests against HelixGraphStore."""

    @property
    def store(self):
        return self._store

    @property
    def group_id(self) -> str:
        return self._group_id

    @pytest.fixture(autouse=True)
    def _inject(self, helix_graph_store, test_group_id):
        self._store = helix_graph_store
        self._group_id = test_group_id


# ======================================================================
# End-to-end integration flow
# ======================================================================


@pytest.mark.asyncio
class TestHelixEndToEnd:
    """Integration tests that exercise a full observe-recall lifecycle."""

    async def test_entity_episode_link_round_trip(self, helix_graph_store, test_group_id):
        """Create entity + episode, link them, verify the link persists."""
        ent_id = f"ent_{_uid()}"
        ep_id = f"ep_{_uid()}"

        await helix_graph_store.create_entity(
            Entity(
                id=ent_id,
                name="Engram Project",
                entity_type="Project",
                summary="A persistent memory layer for AI agents",
                group_id=test_group_id,
            )
        )
        await helix_graph_store.create_episode(
            Episode(
                id=ep_id,
                content="Discussed the Engram project architecture today",
                group_id=test_group_id,
            )
        )
        await helix_graph_store.link_episode_entity(ep_id, ent_id)

        entity_ids = await helix_graph_store.get_episode_entities(ep_id)
        assert ent_id in entity_ids

        episode = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert episode is not None
        assert episode.content == "Discussed the Engram project architecture today"

    async def test_relationship_connects_entities(self, helix_graph_store, test_group_id):
        """Create two entities and a relationship, verify traversal both ways."""
        src_id = f"ent_{_uid()}"
        tgt_id = f"ent_{_uid()}"

        await helix_graph_store.create_entity(
            Entity(id=src_id, name="Alice", entity_type="Person", group_id=test_group_id)
        )
        await helix_graph_store.create_entity(
            Entity(id=tgt_id, name="Engram", entity_type="Project", group_id=test_group_id)
        )
        rel = Relationship(
            id=f"rel_{_uid()}",
            source_id=src_id,
            target_id=tgt_id,
            predicate="WORKS_ON",
            group_id=test_group_id,
        )
        await helix_graph_store.create_relationship(rel)

        outgoing = await helix_graph_store.get_relationships(
            src_id, direction="outgoing", group_id=test_group_id
        )
        assert any(r.predicate == "WORKS_ON" and r.target_id == tgt_id for r in outgoing)

        incoming = await helix_graph_store.get_relationships(
            tgt_id, direction="incoming", group_id=test_group_id
        )
        assert any(r.predicate == "WORKS_ON" and r.source_id == src_id for r in incoming)

    async def test_find_entity_after_creation(self, helix_graph_store, test_group_id):
        """Entity should be discoverable via find_entity_candidates after creation."""
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(
                id=eid,
                name="HelixDB Integration",
                entity_type="Technology",
                summary="Graph database backend",
                group_id=test_group_id,
            )
        )

        candidates = await helix_graph_store.find_entity_candidates(
            "HelixDB Integration", test_group_id
        )
        candidate_ids = [c.id for c in candidates]
        assert eid in candidate_ids

    async def test_update_entity_preserves_other_fields(self, helix_graph_store, test_group_id):
        """Updating one field should not clobber unrelated fields."""
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(
                id=eid,
                name="Original",
                entity_type="Concept",
                summary="Original summary",
                group_id=test_group_id,
            )
        )
        await helix_graph_store.update_entity(
            eid, {"summary": "Updated summary"}, group_id=test_group_id
        )

        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is not None
        assert result.summary == "Updated summary"
        assert result.name == "Original"  # Unchanged
        assert result.entity_type == "Concept"  # Unchanged

    async def test_episode_status_lifecycle(self, helix_graph_store, test_group_id):
        """Episode should progress through status transitions."""
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Status lifecycle test", group_id=test_group_id)
        )

        # Default status
        ep = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert ep is not None
        assert ep.status == EpisodeStatus.PENDING

        # Transition to COMPLETED
        await helix_graph_store.update_episode(
            ep_id, {"status": EpisodeStatus.COMPLETED}, group_id=test_group_id
        )
        ep = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert ep is not None
        assert ep.status == EpisodeStatus.COMPLETED

    async def test_multi_entity_episode_links(self, helix_graph_store, test_group_id):
        """An episode can link to multiple entities."""
        ep_id = f"ep_{_uid()}"
        ent_ids = [f"ent_{_uid()}" for _ in range(3)]

        await helix_graph_store.create_episode(
            Episode(
                id=ep_id,
                content="Meeting about Python, FastAPI, and Docker",
                group_id=test_group_id,
            )
        )
        for i, eid in enumerate(ent_ids):
            await helix_graph_store.create_entity(
                Entity(
                    id=eid,
                    name=["Python", "FastAPI", "Docker"][i],
                    entity_type="Technology",
                    group_id=test_group_id,
                )
            )
            await helix_graph_store.link_episode_entity(ep_id, eid)

        linked = await helix_graph_store.get_episode_entities(ep_id)
        for eid in ent_ids:
            assert eid in linked

    async def test_stats_reflect_created_data(self, helix_graph_store, test_group_id):
        """Stats should account for entities, relationships, and episodes we created."""
        ent1 = f"ent_{_uid()}"
        ent2 = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=ent1, name="StatA", entity_type="Test", group_id=test_group_id)
        )
        await helix_graph_store.create_entity(
            Entity(id=ent2, name="StatB", entity_type="Test", group_id=test_group_id)
        )
        await helix_graph_store.create_relationship(
            Relationship(
                id=f"rel_{_uid()}",
                source_id=ent1,
                target_id=ent2,
                predicate="RELATED_TO",
                group_id=test_group_id,
            )
        )
        await helix_graph_store.create_episode(
            Episode(id=f"ep_{_uid()}", content="Stats episode", group_id=test_group_id)
        )

        stats = await helix_graph_store.get_stats(group_id=test_group_id)
        assert stats["entities"] >= 2
        assert stats["relationships"] >= 1
        assert stats["episodes"] >= 1

    async def test_soft_delete_hides_from_search(self, helix_graph_store, test_group_id):
        """Soft-deleted entity should not appear in find_entities results."""
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="Ephemeral", entity_type="Test", group_id=test_group_id)
        )

        # Verify it exists
        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is not None

        # Soft-delete
        await helix_graph_store.delete_entity(eid, soft=True, group_id=test_group_id)

        # Should be hidden
        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is None
