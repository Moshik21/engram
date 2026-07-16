"""Tests for conflict detection (exclusive predicates)."""

from datetime import datetime
from uuid import uuid4

import pytest

from engram.extraction.conflicts import get_contradictory_predicates, is_exclusive_predicate
from engram.models.entity import Entity
from engram.models.relationship import Relationship


class TestExclusivePredicates:
    def test_lives_in_is_exclusive(self):
        assert is_exclusive_predicate("LIVES_IN")

    def test_works_at_is_exclusive(self):
        assert is_exclusive_predicate("WORKS_AT")

    def test_married_to_is_exclusive(self):
        assert is_exclusive_predicate("MARRIED_TO")

    def test_uses_is_not_exclusive(self):
        assert not is_exclusive_predicate("USES")

    def test_builds_is_not_exclusive(self):
        assert not is_exclusive_predicate("BUILDS")

    def test_case_insensitive(self):
        assert is_exclusive_predicate("lives_in")

    def test_likes_contradicts_dislikes(self):
        assert "DISLIKES" in get_contradictory_predicates("LIKES")
        assert "LIKES" in get_contradictory_predicates("DISLIKES")

    def test_no_contradictions_for_works_at(self):
        assert get_contradictory_predicates("WORKS_AT") == set()

    def test_contradictory_case_insensitive(self):
        assert "DISLIKES" in get_contradictory_predicates("likes")


@pytest.mark.asyncio
class TestConflictDetection:
    async def test_find_conflicting_relationships(self, graph_store):
        """find_conflicting_relationships returns active rels with same source+predicate."""
        # Unique ids + group so a persistent native store cannot accrue
        # cross-run residue on shared "default"/fixed ids.
        uid = uuid4().hex[:8]
        gid = f"default_{uid}"
        person, city1, city2 = f"ent_person_{uid}", f"ent_city1_{uid}", f"ent_city2_{uid}"
        await graph_store.create_entity(
            Entity(id=person, name="Alice", entity_type="Person", group_id=gid)
        )
        await graph_store.create_entity(
            Entity(id=city1, name="Mesa", entity_type="Location", group_id=gid)
        )
        await graph_store.create_entity(
            Entity(id=city2, name="Denver", entity_type="Location", group_id=gid)
        )

        # Create first LIVES_IN
        await graph_store.create_relationship(
            Relationship(
                id=f"rel_lives1_{uid}",
                source_id=person,
                target_id=city1,
                predicate="LIVES_IN",
                group_id=gid,
            )
        )

        conflicts = await graph_store.find_conflicting_relationships(person, "LIVES_IN", gid)
        assert len(conflicts) == 1
        assert conflicts[0].target_id == city1

    async def test_find_conflicting_ignores_invalidated(self, graph_store):
        """Already-invalidated relationships should not be returned as conflicts."""
        await graph_store.create_entity(
            Entity(id="ent_p", name="Bob", entity_type="Person", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_c1", name="City1", entity_type="Location", group_id="default")
        )

        await graph_store.create_relationship(
            Relationship(
                id="rel_old",
                source_id="ent_p",
                target_id="ent_c1",
                predicate="LIVES_IN",
                group_id="default",
            )
        )
        # Invalidate
        await graph_store.invalidate_relationship(
            "rel_old", datetime(2026, 1, 1), group_id="default"
        )

        conflicts = await graph_store.find_conflicting_relationships("ent_p", "LIVES_IN", "default")
        assert len(conflicts) == 0

    async def test_non_exclusive_no_invalidation(self, graph_store):
        """Non-exclusive predicates should not trigger conflict detection."""
        assert not is_exclusive_predicate("USES")

    async def test_same_target_no_conflict(self, graph_store):
        """Same source+predicate+target is not a real conflict."""
        uid = uuid4().hex[:8]
        gid = f"default_{uid}"
        source, target = f"ent_s_{uid}", f"ent_t_{uid}"
        await graph_store.create_entity(
            Entity(id=source, name="S", entity_type="Person", group_id=gid)
        )
        await graph_store.create_entity(
            Entity(id=target, name="T", entity_type="Location", group_id=gid)
        )

        await graph_store.create_relationship(
            Relationship(
                id=f"rel_same_{uid}",
                source_id=source,
                target_id=target,
                predicate="LIVES_IN",
                group_id=gid,
            )
        )

        conflicts = await graph_store.find_conflicting_relationships(source, "LIVES_IN", gid)
        # The existing rel is returned; the graph_manager would skip same-target
        assert len(conflicts) == 1

    async def test_get_relationships_at_active(self, graph_store):
        """get_relationships_at returns rels active at a specific time."""
        await graph_store.create_entity(
            Entity(id="ent_at1", name="P1", entity_type="Person", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_at2", name="C1", entity_type="Location", group_id="default")
        )

        await graph_store.create_relationship(
            Relationship(
                id="rel_at1",
                source_id="ent_at1",
                target_id="ent_at2",
                predicate="LIVES_IN",
                valid_from=datetime(2024, 1, 1),
                valid_to=datetime(2025, 6, 1),
                group_id="default",
            )
        )

        # Active in 2024
        rels = await graph_store.get_relationships_at("ent_at1", datetime(2024, 6, 1))
        assert len(rels) == 1

        # Not active after valid_to
        rels = await graph_store.get_relationships_at("ent_at1", datetime(2026, 1, 1))
        assert len(rels) == 0

    async def test_get_relationships_at_no_valid_from(self, graph_store):
        """Rel with no valid_from is always active from the beginning."""
        await graph_store.create_entity(
            Entity(id="ent_nf1", name="X", entity_type="Test", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_nf2", name="Y", entity_type="Test", group_id="default")
        )

        await graph_store.create_relationship(
            Relationship(
                id="rel_nf1",
                source_id="ent_nf1",
                target_id="ent_nf2",
                predicate="KNOWS",
                group_id="default",
            )
        )

        rels = await graph_store.get_relationships_at("ent_nf1", datetime(2020, 1, 1))
        assert len(rels) == 1
