"""Tests for GraphManager.record_explicit_feedback()."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engram.config import ActivationConfig

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class MockEntity:
    id: str
    name: str
    entity_type: str
    summary: str = ""
    group_id: str = "default"
    attributes: dict = field(default_factory=dict)


@dataclass
class MockRelationship:
    id: str
    source_id: str
    target_id: str
    predicate: str
    weight: float = 1.0
    group_id: str = "default"


class MockGraphStore:
    """Minimal graph store mock for feedback tests."""

    def __init__(self, entities: list[MockEntity] | None = None):
        self._entities = {e.id: e for e in (entities or [])}
        self._rels: list[MockRelationship] = []
        self._created_entities: list[MockEntity] = []
        self._created_rels: list[MockRelationship] = []
        self._weight_updates: list[tuple] = []

    async def get_entity(self, entity_id: str, group_id: str = "default") -> MockEntity | None:
        return self._entities.get(entity_id)

    async def find_entities(
        self, name: str = "", entity_type: str = "", group_id: str = "default", limit: int = 100
    ) -> list[MockEntity]:
        results = []
        for e in self._entities.values():
            match = True
            if name and e.name != name:
                match = False
            if entity_type and e.entity_type != entity_type:
                match = False
            if match:
                results.append(e)
        return results[:limit]

    async def get_relationships(
        self,
        entity_id: str,
        direction: str = "outgoing",
        predicate: str | None = None,
        group_id: str = "default",
    ) -> list[MockRelationship]:
        results = []
        for r in self._rels:
            if direction == "outgoing" and r.source_id == entity_id:
                if predicate is None or r.predicate == predicate:
                    results.append(r)
        return results

    async def create_entity(self, entity) -> None:
        self._entities[entity.id] = entity
        self._created_entities.append(entity)

    async def create_relationship(self, rel) -> None:
        self._rels.append(rel)
        self._created_rels.append(rel)

    async def update_relationship_weight(
        self, rel_id: str, new_weight: float, group_id: str = "default"
    ) -> None:
        self._weight_updates.append((rel_id, new_weight, group_id))
        for r in self._rels:
            if r.id == rel_id:
                r.weight = new_weight


class MockEventBus:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def publish(self, event_type: str, data: dict) -> None:
        self.events.append((event_type, data))


class MockSyncGroupEventBus:
    def __init__(self):
        self.events: list[tuple[str, str, dict]] = []

    def publish(self, group_id: str, event_type: str, data: dict) -> int:
        self.events.append((group_id, event_type, data))
        return len(self.events)


class FakeGraphManager:
    """Lightweight stand-in that only has the fields record_explicit_feedback needs."""

    def __init__(self, graph_store: MockGraphStore, cfg: ActivationConfig, event_bus=None):
        self._graph = graph_store
        self._cfg = cfg
        self._event_bus = event_bus

    # Bind the real method
    from engram.graph_manager import GraphManager

    record_explicit_feedback = GraphManager.record_explicit_feedback


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEntityNotFound:
    """Raises ValueError when entity does not exist."""

    @pytest.mark.asyncio
    async def test_entity_not_found(self):
        store = MockGraphStore()
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)
        with pytest.raises(ValueError, match="not found"):
            await mgr.record_explicit_feedback(
                group_id="default",
                entity_id="nonexistent",
                rating=5,
            )


class TestPositiveFeedback:
    """Positive rating creates PREFERS edge."""

    @pytest.mark.asyncio
    async def test_creates_prefers_edge(self):
        entity = MockEntity(id="ent_1", name="Python", entity_type="Technology")
        store = MockGraphStore(entities=[entity])
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=5,
        )

        assert result["status"] == "recorded"
        assert result["edge_type"] == "PREFERS"
        assert result["domain"] == "technical"
        assert result["edge_weight"] == 1.0  # abs(5-3)/2 = 1.0

        # Verify entity and relationship were created
        assert len(store._created_entities) == 1  # UserPreference singleton
        assert store._created_entities[0].name == "UserPreference"
        assert len(store._created_rels) == 1
        assert store._created_rels[0].predicate == "PREFERS"
        assert store._created_rels[0].target_id == "ent_1"


class TestNegativeFeedback:
    """Negative rating creates AVOIDS edge."""

    @pytest.mark.asyncio
    async def test_creates_avoids_edge(self):
        entity = MockEntity(id="ent_1", name="COBOL", entity_type="Technology")
        store = MockGraphStore(entities=[entity])
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=1,
        )

        assert result["status"] == "recorded"
        assert result["edge_type"] == "AVOIDS"
        assert result["edge_weight"] == 1.0  # abs(1-3)/2 = 1.0
        assert len(store._created_rels) == 1
        assert store._created_rels[0].predicate == "AVOIDS"


class TestNeutralRating:
    """Neutral rating (3) creates no edge."""

    @pytest.mark.asyncio
    async def test_neutral_no_edge(self):
        entity = MockEntity(id="ent_1", name="Go", entity_type="Technology")
        store = MockGraphStore(entities=[entity])
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=3,
        )

        assert result["status"] == "neutral"
        assert result["edge_type"] is None
        assert result["edge_weight"] == 0.0
        assert len(store._created_rels) == 0


class TestExistingEdgeStrengthened:
    """Existing edge gets weight increased."""

    @pytest.mark.asyncio
    async def test_strengthens_existing_edge(self):
        entity = MockEntity(id="ent_1", name="Python", entity_type="Technology")
        pref = MockEntity(
            id="pref_existing",
            name="UserPreference",
            entity_type="PreferenceProfile",
        )
        store = MockGraphStore(entities=[entity, pref])
        # Add existing PREFERS edge
        existing_rel = MockRelationship(
            id="rel_existing",
            source_id="pref_existing",
            target_id="ent_1",
            predicate="PREFERS",
            weight=0.5,
        )
        store._rels.append(existing_rel)

        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=4,
        )

        assert result["status"] == "recorded"
        assert result["edge_type"] == "PREFERS"
        # Weight update should have been called
        assert len(store._weight_updates) == 1
        rel_id, new_weight, _ = store._weight_updates[0]
        assert rel_id == "rel_existing"
        # new_weight = min(1.0, 0.5 + (0.5 * 0.5)) = 0.75
        assert new_weight == pytest.approx(0.75)
        # No new relationship created
        assert len(store._created_rels) == 0


class TestEventPublished:
    """feedback.recorded event is published."""

    @pytest.mark.asyncio
    async def test_event_published(self):
        entity = MockEntity(id="ent_1", name="Python", entity_type="Technology")
        store = MockGraphStore(entities=[entity])
        bus = MockEventBus()
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg, event_bus=bus)

        await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=5,
        )

        assert len(bus.events) == 1
        event_type, data = bus.events[0]
        assert event_type == "feedback.recorded"
        assert data["entity_id"] == "ent_1"
        assert data["rating"] == 5
        assert data["edge_type"] == "PREFERS"

    @pytest.mark.asyncio
    async def test_group_scoped_sync_event_bus_published(self):
        entity = MockEntity(id="ent_1", name="Python", entity_type="Technology")
        store = MockGraphStore(entities=[entity])
        bus = MockSyncGroupEventBus()
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg, event_bus=bus)

        await mgr.record_explicit_feedback(
            group_id="native_brain",
            entity_id="ent_1",
            rating=5,
        )

        assert len(bus.events) == 1
        group_id, event_type, data = bus.events[0]
        assert group_id == "native_brain"
        assert event_type == "feedback.recorded"
        assert data["entity_id"] == "ent_1"
        assert data["rating"] == 5
        assert data["edge_type"] == "PREFERS"


class TestDomainDetection:
    """Entity type maps to correct domain."""

    @pytest.mark.asyncio
    async def test_person_maps_to_personal(self):
        entity = MockEntity(id="ent_1", name="Alice", entity_type="Person")
        store = MockGraphStore(entities=[entity])
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=5,
        )
        assert result["domain"] == "personal"

    @pytest.mark.asyncio
    async def test_unknown_type_maps_to_general(self):
        entity = MockEntity(id="ent_1", name="Something", entity_type="UnknownType")
        store = MockGraphStore(entities=[entity])
        cfg = ActivationConfig()
        mgr = FakeGraphManager(store, cfg)

        result = await mgr.record_explicit_feedback(
            group_id="default",
            entity_id="ent_1",
            rating=5,
        )
        assert result["domain"] == "general"
