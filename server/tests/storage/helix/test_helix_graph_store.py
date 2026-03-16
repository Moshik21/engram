"""Integration tests for HelixDB GraphStore protocol conformance.

Verifies that HelixGraphStore correctly implements the GraphStore protocol
for entity, relationship, episode, intention, and evidence CRUD operations.

All tests require a running HelixDB instance on localhost:6969 and are
marked with ``requires_helix``.
"""

from __future__ import annotations

import socket
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship


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


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# ======================================================================
# Entity CRUD
# ======================================================================


@pytest.mark.asyncio
class TestEntityCRUD:
    async def test_create_and_get_entity(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="Python",
            entity_type="Technology",
            summary="Programming language",
            group_id=test_group_id,
        )
        result_id = await helix_graph_store.create_entity(entity)
        assert result_id == eid

        fetched = await helix_graph_store.get_entity(eid, test_group_id)
        assert fetched is not None
        assert fetched.name == "Python"
        assert fetched.entity_type == "Technology"
        assert fetched.summary == "Programming language"

    async def test_get_entity_returns_none_for_missing(self, helix_graph_store, test_group_id):
        result = await helix_graph_store.get_entity(f"ent_{_uid()}", test_group_id)
        assert result is None

    async def test_tenant_isolation(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="Secret",
            entity_type="Concept",
            group_id=test_group_id,
        )
        await helix_graph_store.create_entity(entity)

        # Different group should not see the entity
        result = await helix_graph_store.get_entity(eid, "other_group_xyz")
        assert result is None

        # Correct group should see it
        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is not None

    async def test_update_entity(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="OldName", entity_type="Person", group_id=test_group_id)
        )
        await helix_graph_store.update_entity(eid, {"name": "NewName"}, group_id=test_group_id)

        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is not None
        assert result.name == "NewName"

    async def test_update_entity_rejects_invalid_field(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="Test", entity_type="Test", group_id=test_group_id)
        )
        with pytest.raises(ValueError, match="Disallowed"):
            await helix_graph_store.update_entity(
                eid, {"id": "hacked"}, group_id=test_group_id
            )

    async def test_soft_delete_entity(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="ToDelete", entity_type="Test", group_id=test_group_id)
        )
        await helix_graph_store.delete_entity(eid, soft=True, group_id=test_group_id)

        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is None  # Soft-deleted entities are hidden

    async def test_find_entities_by_group(self, helix_graph_store, test_group_id):
        ids = []
        for i in range(3):
            eid = f"ent_find_{_uid()}"
            ids.append(eid)
            await helix_graph_store.create_entity(
                Entity(id=eid, name=f"Entity{i}", entity_type="Test", group_id=test_group_id)
            )
        results = await helix_graph_store.find_entities(group_id=test_group_id)
        assert len(results) >= 3

    async def test_find_entities_by_name(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="JavaScript", entity_type="Technology", group_id=test_group_id)
        )
        results = await helix_graph_store.find_entities(name="JavaScript", group_id=test_group_id)
        assert len(results) >= 1
        assert any(r.name == "JavaScript" for r in results)

    async def test_batch_get_entities(self, helix_graph_store, test_group_id):
        ids = []
        for i in range(3):
            eid = f"ent_batch_{_uid()}"
            ids.append(eid)
            await helix_graph_store.create_entity(
                Entity(id=eid, name=f"Batch{i}", entity_type="Test", group_id=test_group_id)
            )
        result = await helix_graph_store.batch_get_entities(ids, test_group_id)
        assert len(result) == 3
        for eid in ids:
            assert eid in result

    async def test_find_entity_candidates(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(
                id=eid,
                name="FastAPI Framework",
                entity_type="Technology",
                group_id=test_group_id,
            )
        )
        candidates = await helix_graph_store.find_entity_candidates(
            "FastAPI Framework", test_group_id
        )
        candidate_ids = [c.id for c in candidates]
        assert eid in candidate_ids

    async def test_identifier_facets_persist_on_create(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        entity = Entity(
            id=eid,
            name="SKU 1712061",
            entity_type="Identifier",
            group_id=test_group_id,
        )
        await helix_graph_store.create_entity(entity)
        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is not None
        assert result.lexical_regime == "identifier"
        assert result.canonical_identifier == "1712061"
        assert result.identifier_label is True

    async def test_delete_group_removes_entities(self, helix_graph_store, test_group_id):
        eid = f"ent_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=eid, name="GroupEntity", entity_type="Test", group_id=test_group_id)
        )
        await helix_graph_store.delete_group(test_group_id)
        result = await helix_graph_store.get_entity(eid, test_group_id)
        assert result is None


# ======================================================================
# Relationship CRUD
# ======================================================================


@pytest.mark.asyncio
class TestRelationshipCRUD:
    async def _create_pair(self, store, group_id):
        """Helper: create source + target entities, return their IDs."""
        src = f"ent_src_{_uid()}"
        tgt = f"ent_tgt_{_uid()}"
        await store.create_entity(
            Entity(id=src, name="Source", entity_type="Test", group_id=group_id)
        )
        await store.create_entity(
            Entity(id=tgt, name="Target", entity_type="Test", group_id=group_id)
        )
        return src, tgt

    async def test_create_and_get_relationship(self, helix_graph_store, test_group_id):
        src, tgt = await self._create_pair(helix_graph_store, test_group_id)
        rel = Relationship(
            id=f"rel_{_uid()}",
            source_id=src,
            target_id=tgt,
            predicate="CONNECTS",
            group_id=test_group_id,
        )
        await helix_graph_store.create_relationship(rel)

        rels = await helix_graph_store.get_relationships(
            src, direction="outgoing", group_id=test_group_id
        )
        assert len(rels) >= 1
        assert any(r.predicate == "CONNECTS" for r in rels)

    async def test_get_entity_relationships_both_directions(
        self, helix_graph_store, test_group_id
    ):
        src, tgt = await self._create_pair(helix_graph_store, test_group_id)
        rel = Relationship(
            id=f"rel_{_uid()}",
            source_id=src,
            target_id=tgt,
            predicate="LINKS",
            group_id=test_group_id,
        )
        await helix_graph_store.create_relationship(rel)

        # Target entity should see the relationship via incoming direction
        incoming = await helix_graph_store.get_relationships(
            tgt, direction="incoming", group_id=test_group_id
        )
        assert len(incoming) >= 1

        # Both directions from source should also include it
        both = await helix_graph_store.get_relationships(
            src, direction="both", group_id=test_group_id
        )
        assert len(both) >= 1

    async def test_update_relationship_weight(self, helix_graph_store, test_group_id):
        src, tgt = await self._create_pair(helix_graph_store, test_group_id)
        rel = Relationship(
            id=f"rel_{_uid()}",
            source_id=src,
            target_id=tgt,
            predicate="USES",
            weight=1.0,
            group_id=test_group_id,
        )
        await helix_graph_store.create_relationship(rel)

        new_weight = await helix_graph_store.update_relationship_weight(
            src, tgt, 0.5, group_id=test_group_id, predicate="USES"
        )
        assert new_weight is not None
        assert new_weight == pytest.approx(1.5, abs=0.01)

    async def test_invalidate_relationship(self, helix_graph_store, test_group_id):
        src, tgt = await self._create_pair(helix_graph_store, test_group_id)
        rid = f"rel_{_uid()}"
        rel = Relationship(
            id=rid,
            source_id=src,
            target_id=tgt,
            predicate="OLD_REL",
            group_id=test_group_id,
        )
        await helix_graph_store.create_relationship(rel)
        await helix_graph_store.invalidate_relationship(
            rid, datetime.now(tz=timezone.utc), group_id=test_group_id
        )

        # active_only=True should hide invalidated relationship
        active = await helix_graph_store.get_relationships(
            src, active_only=True, group_id=test_group_id
        )
        assert all(r.id != rid for r in active)

        # active_only=False should show it with valid_to set
        all_rels = await helix_graph_store.get_relationships(
            src, active_only=False, group_id=test_group_id
        )
        invalidated = [r for r in all_rels if r.id == rid]
        assert len(invalidated) == 1
        assert invalidated[0].valid_to is not None

    async def test_find_existing_relationship(self, helix_graph_store, test_group_id):
        src, tgt = await self._create_pair(helix_graph_store, test_group_id)
        await helix_graph_store.create_relationship(
            Relationship(
                id=f"rel_{_uid()}",
                source_id=src,
                target_id=tgt,
                predicate="KNOWS",
                group_id=test_group_id,
            )
        )

        found = await helix_graph_store.find_existing_relationship(
            src, tgt, "KNOWS", test_group_id
        )
        assert found is not None
        assert found.predicate == "KNOWS"

        not_found = await helix_graph_store.find_existing_relationship(
            src, tgt, "HATES", test_group_id
        )
        assert not_found is None


# ======================================================================
# Episode CRUD
# ======================================================================


@pytest.mark.asyncio
class TestEpisodeCRUD:
    async def test_create_and_list_episodes(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        ep = Episode(
            id=ep_id,
            content="Test episode content for HelixDB",
            group_id=test_group_id,
        )
        result_id = await helix_graph_store.create_episode(ep)
        assert result_id == ep_id

        episodes = await helix_graph_store.get_episodes(group_id=test_group_id)
        assert len(episodes) >= 1
        assert any(e.content == "Test episode content for HelixDB" for e in episodes)

    async def test_get_episode_by_id(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Specific episode", group_id=test_group_id)
        )
        result = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert result is not None
        assert result.id == ep_id
        assert result.content == "Specific episode"

    async def test_get_episode_returns_none_for_missing(self, helix_graph_store, test_group_id):
        result = await helix_graph_store.get_episode_by_id(f"ep_{_uid()}", test_group_id)
        assert result is None

    async def test_update_episode_status(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Status test", group_id=test_group_id)
        )
        await helix_graph_store.update_episode(
            ep_id,
            {"status": EpisodeStatus.COMPLETED},
            group_id=test_group_id,
        )
        result = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert result is not None
        assert result.status == EpisodeStatus.COMPLETED

    async def test_update_episode_content(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Original content", group_id=test_group_id)
        )
        await helix_graph_store.update_episode(
            ep_id, {"content": "Updated content"}, group_id=test_group_id
        )
        result = await helix_graph_store.get_episode_by_id(ep_id, test_group_id)
        assert result is not None
        assert result.content == "Updated content"

    async def test_update_episode_rejects_invalid_field(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Content", group_id=test_group_id)
        )
        with pytest.raises(ValueError, match="Disallowed"):
            await helix_graph_store.update_episode(
                ep_id, {"id": "hacked"}, group_id=test_group_id
            )

    async def test_link_episode_entity(self, helix_graph_store, test_group_id):
        ent_id = f"ent_{_uid()}"
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_entity(
            Entity(id=ent_id, name="Linked", entity_type="Test", group_id=test_group_id)
        )
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Episode with link", group_id=test_group_id)
        )
        await helix_graph_store.link_episode_entity(ep_id, ent_id)

        entity_ids = await helix_graph_store.get_episode_entities(ep_id)
        assert ent_id in entity_ids


# ======================================================================
# Episode Cues
# ======================================================================


@pytest.mark.asyncio
class TestEpisodeCues:
    async def test_upsert_and_get_cue(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Cue episode", group_id=test_group_id)
        )
        cue = EpisodeCue(
            episode_id=ep_id,
            group_id=test_group_id,
            cue_text="important context about Python",
        )
        await helix_graph_store.upsert_episode_cue(cue)

        result = await helix_graph_store.get_episode_cue(ep_id, test_group_id)
        assert result is not None
        assert result.cue_text == "important context about Python"


# ======================================================================
# Stats & Counts
# ======================================================================


@pytest.mark.asyncio
class TestStats:
    async def test_get_stats(self, helix_graph_store, test_group_id):
        await helix_graph_store.create_entity(
            Entity(
                id=f"ent_{_uid()}", name="Stat", entity_type="Test", group_id=test_group_id
            )
        )
        stats = await helix_graph_store.get_stats(group_id=test_group_id)
        assert stats["entities"] >= 1
        assert "relationships" in stats
        assert "episodes" in stats

    async def test_get_entity_count(self, helix_graph_store, test_group_id):
        await helix_graph_store.create_entity(
            Entity(
                id=f"ent_{_uid()}", name="Counter", entity_type="Test", group_id=test_group_id
            )
        )
        count = await helix_graph_store.get_entity_count(group_id=test_group_id)
        assert count >= 1

    async def test_get_entity_type_counts(self, helix_graph_store, test_group_id):
        await helix_graph_store.create_entity(
            Entity(
                id=f"ent_{_uid()}", name="A", entity_type="Person", group_id=test_group_id
            )
        )
        await helix_graph_store.create_entity(
            Entity(
                id=f"ent_{_uid()}", name="B", entity_type="Person", group_id=test_group_id
            )
        )
        await helix_graph_store.create_entity(
            Entity(
                id=f"ent_{_uid()}", name="C", entity_type="Technology", group_id=test_group_id
            )
        )
        counts = await helix_graph_store.get_entity_type_counts(group_id=test_group_id)
        assert counts.get("Person", 0) >= 2
        assert counts.get("Technology", 0) >= 1


# ======================================================================
# Intentions (Prospective memory)
# ======================================================================


@pytest.mark.asyncio
class TestIntentions:
    async def test_create_and_get_intention(self, helix_graph_store, test_group_id):
        from engram.models.prospective import Intention

        iid = f"int_{_uid()}"
        intention = Intention(
            id=iid,
            trigger_text="when Python is mentioned",
            action_text="remind about FastAPI",
            group_id=test_group_id,
        )
        result_id = await helix_graph_store.create_intention(intention)
        assert result_id == iid

        fetched = await helix_graph_store.get_intention(iid, test_group_id)
        assert fetched is not None
        assert fetched.trigger_text == "when Python is mentioned"
        assert fetched.action_text == "remind about FastAPI"

    async def test_list_intentions(self, helix_graph_store, test_group_id):
        from engram.models.prospective import Intention

        for i in range(3):
            await helix_graph_store.create_intention(
                Intention(
                    id=f"int_list_{_uid()}",
                    trigger_text=f"trigger {i}",
                    action_text=f"action {i}",
                    group_id=test_group_id,
                    enabled=True,
                )
            )
        intentions = await helix_graph_store.list_intentions(test_group_id, enabled_only=True)
        assert len(intentions) >= 3


# ======================================================================
# Evidence storage
# ======================================================================


@pytest.mark.asyncio
class TestEvidence:
    async def test_store_and_get_pending_evidence(self, helix_graph_store, test_group_id):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Evidence episode", group_id=test_group_id)
        )
        ev_id = f"ev_{_uid()}"
        evidence = [
            {
                "evidence_id": ev_id,
                "episode_id": ep_id,
                "fact_class": "entity",
                "confidence": 0.9,
                "source_type": "extraction",
                "extractor_name": "test",
                "payload": {"name": "Python"},
            },
        ]
        await helix_graph_store.store_evidence(evidence, group_id=test_group_id)

        pending = await helix_graph_store.get_pending_evidence(group_id=test_group_id)
        ev_ids = [e["evidence_id"] for e in pending]
        assert ev_id in ev_ids

    async def test_update_evidence_status_removes_from_pending(
        self, helix_graph_store, test_group_id
    ):
        ep_id = f"ep_{_uid()}"
        await helix_graph_store.create_episode(
            Episode(id=ep_id, content="Evidence ep", group_id=test_group_id)
        )
        ev_id = f"ev_{_uid()}"
        evidence = [
            {
                "evidence_id": ev_id,
                "episode_id": ep_id,
                "fact_class": "entity",
                "confidence": 0.7,
                "source_type": "extraction",
                "payload": {},
            },
        ]
        await helix_graph_store.store_evidence(evidence, group_id=test_group_id)

        await helix_graph_store.update_evidence_status(
            ev_id,
            "committed",
            updates={"commit_reason": "high_confidence"},
            group_id=test_group_id,
        )

        pending = await helix_graph_store.get_pending_evidence(group_id=test_group_id)
        pending_ids = [e["evidence_id"] for e in pending]
        assert ev_id not in pending_ids
