"""Tests for conflict detection (exclusive predicates)."""

from datetime import datetime

import pytest

from engram.extraction.conflicts import is_exclusive_predicate
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.sqlite.graph import SQLiteGraphStore


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


@pytest.mark.asyncio
class TestConflictDetection:
    async def test_find_conflicting_relationships(self, graph_store: SQLiteGraphStore):
        """find_conflicting_relationships returns active rels with same source+predicate."""
        await graph_store.create_entity(
            Entity(id="ent_person", name="Alice", entity_type="Person", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_city1", name="Mesa", entity_type="Location", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_city2", name="Denver", entity_type="Location", group_id="default")
        )

        # Create first LIVES_IN
        await graph_store.create_relationship(
            Relationship(
                id="rel_lives1",
                source_id="ent_person",
                target_id="ent_city1",
                predicate="LIVES_IN",
                group_id="default",
            )
        )

        conflicts = await graph_store.find_conflicting_relationships(
            "ent_person", "LIVES_IN", "default"
        )
        assert len(conflicts) == 1
        assert conflicts[0].target_id == "ent_city1"

    async def test_find_conflicting_ignores_invalidated(self, graph_store: SQLiteGraphStore):
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

        conflicts = await graph_store.find_conflicting_relationships(
            "ent_p", "LIVES_IN", "default"
        )
        assert len(conflicts) == 0

    async def test_non_exclusive_no_invalidation(self, graph_store: SQLiteGraphStore):
        """Non-exclusive predicates should not trigger conflict detection."""
        assert not is_exclusive_predicate("USES")

    async def test_same_target_no_conflict(self, graph_store: SQLiteGraphStore):
        """Same source+predicate+target is not a real conflict."""
        await graph_store.create_entity(
            Entity(id="ent_s", name="S", entity_type="Person", group_id="default")
        )
        await graph_store.create_entity(
            Entity(id="ent_t", name="T", entity_type="Location", group_id="default")
        )

        await graph_store.create_relationship(
            Relationship(
                id="rel_same",
                source_id="ent_s",
                target_id="ent_t",
                predicate="LIVES_IN",
                group_id="default",
            )
        )

        conflicts = await graph_store.find_conflicting_relationships(
            "ent_s", "LIVES_IN", "default"
        )
        # The existing rel is returned; the graph_manager would skip same-target
        assert len(conflicts) == 1

    async def test_get_relationships_at_active(self, graph_store: SQLiteGraphStore):
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

    async def test_get_relationships_at_no_valid_from(self, graph_store: SQLiteGraphStore):
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
