"""Tests for the edge inference consolidation phase."""

import uuid

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


@pytest_asyncio.fixture
async def store(tmp_path):
    s = SQLiteGraphStore(str(tmp_path / "test.db"))
    await s.initialize()
    yield s
    await s.close()


@pytest_asyncio.fixture
async def search(store):
    idx = FTS5SearchIndex(store._db_path)
    await idx.initialize(db=store._db)
    return idx


@pytest_asyncio.fixture
async def activation():
    return MemoryActivationStore(cfg=ActivationConfig())


def _entity(name, entity_type="Person", group_id="test"):
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name, entity_type=entity_type, group_id=group_id,
    )


def _episode(group_id="test"):
    return Episode(
        id=f"ep_{uuid.uuid4().hex[:8]}",
        content="test", source="test", status="completed", group_id=group_id,
    )


class TestEdgeInferencePhase:
    """Tests for EdgeInferencePhase."""

    @pytest.mark.asyncio
    async def test_creates_mentioned_with_edges(self, store, activation, search):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # 3 co-occurrences
        for _ in range(3):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected == 1
        assert len(records) == 1
        assert records[0].co_occurrence_count == 3

        # Verify the relationship was created
        rels = await store.get_relationships(e1.id, group_id="test")
        mentioned = [r for r in rels if r.predicate == "MENTIONED_WITH"]
        assert len(mentioned) == 1
        assert mentioned[0].source_episode.startswith("consolidation:")

    @pytest.mark.asyncio
    async def test_confidence_scaling(self, store, activation, search):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # 10 co-occurrences (high count)
        for _ in range(10):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.6,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert len(records) == 1
        # Higher co-occurrence → higher confidence (up to 0.9 cap)
        assert records[0].confidence > 0.6

    @pytest.mark.asyncio
    async def test_limit_enforcement(self, store, activation, search):
        # Create 4 entity pairs with 3+ co-occurrences each
        entities = []
        for i in range(8):
            e = _entity(f"Entity_{i}")
            await store.create_entity(e)
            entities.append(e)

        # Link pairs: (0,1), (2,3), (4,5), (6,7)
        for pair_idx in range(4):
            a, b = entities[pair_idx * 2], entities[pair_idx * 2 + 1]
            for _ in range(3):
                ep = _episode()
                await store.create_episode(ep)
                await store.link_episode_entity(ep.id, a.id)
                await store.link_episode_entity(ep.id, b.id)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_max_per_cycle=2,
        )
        phase = EdgeInferencePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        assert result.items_affected <= 2

    @pytest.mark.asyncio
    async def test_no_duplicate_inference(self, store, activation, search):
        """Should not infer an edge if one already exists."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        # Add existing relationship
        rel = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e1.id, target_id=e2.id,
            predicate="KNOWS", group_id="test",
        )
        await store.create_relationship(rel)

        # Add co-occurrences
        for _ in range(5):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )

        # Existing relationship should prevent inference
        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_dry_run(self, store, activation, search):
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        await store.create_entity(e1)
        await store.create_entity(e2)

        for _ in range(3):
            ep = _episode()
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()
        result, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=True,
        )

        assert result.items_affected == 1
        # But no actual relationship created
        rels = await store.get_relationships(e1.id, group_id="test")
        mentioned = [r for r in rels if r.predicate == "MENTIONED_WITH"]
        assert len(mentioned) == 0

    @pytest.mark.asyncio
    async def test_group_isolation(self, store, activation, search):
        e1 = _entity("Alice", group_id="group_a")
        e2 = _entity("Bob", group_id="group_a")
        await store.create_entity(e1)
        await store.create_entity(e2)

        for _ in range(3):
            ep = _episode(group_id="group_a")
            await store.create_episode(ep)
            await store.link_episode_entity(ep.id, e1.id)
            await store.link_episode_entity(ep.id, e2.id)

        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()

        # Query group_b — should find nothing
        result, records = await phase.execute(
            group_id="group_b", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_test", dry_run=False,
        )
        assert result.items_affected == 0


class TestTransitivityPass:
    """Tests for the transitivity inference pass."""

    @pytest.mark.asyncio
    async def test_transitivity_disabled_by_default(self, store, activation, search):
        """Default config produces no transitive edges."""
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        await store.create_entity(e_a)
        await store.create_entity(e_b)
        await store.create_entity(e_c)

        # A LOCATED_IN B, B LOCATED_IN C
        rel_ab = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_b.id,
            predicate="LOCATED_IN", group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id, target_id=e_c.id,
            predicate="LOCATED_IN", group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,  # No co-occurrence edges
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_trans", dry_run=False,
        )
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_transitivity_basic(self, store, activation, search):
        """A LOCATED_IN B + B LOCATED_IN C → A LOCATED_IN C."""
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        await store.create_entity(e_a)
        await store.create_entity(e_b)
        await store.create_entity(e_c)

        rel_ab = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_b.id,
            predicate="LOCATED_IN", confidence=0.9, group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id, target_id=e_c.id,
            predicate="LOCATED_IN", confidence=0.8, group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,
            consolidation_infer_transitivity_enabled=True,
            consolidation_infer_transitive_predicates=["LOCATED_IN"],
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_trans", dry_run=False,
        )

        trans = [r for r in records if r.infer_type == "transitivity"]
        assert len(trans) == 1
        assert trans[0].source_id == e_a.id
        assert trans[0].target_id == e_c.id

        # Verify the relationship was actually created
        rels = await store.get_relationships(e_a.id, group_id="test")
        loc_rels = [r for r in rels if r.predicate == "LOCATED_IN"]
        targets = {r.target_id for r in loc_rels}
        assert e_c.id in targets

    @pytest.mark.asyncio
    async def test_transitivity_no_duplicate(self, store, activation, search):
        """A→C already exists → not re-created."""
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        await store.create_entity(e_a)
        await store.create_entity(e_b)
        await store.create_entity(e_c)

        rel_ab = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_b.id,
            predicate="LOCATED_IN", group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id, target_id=e_c.id,
            predicate="LOCATED_IN", group_id="test",
        )
        # Direct A→C already exists
        rel_ac = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_c.id,
            predicate="LOCATED_IN", group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)
        await store.create_relationship(rel_ac)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,
            consolidation_infer_transitivity_enabled=True,
            consolidation_infer_transitive_predicates=["LOCATED_IN"],
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_trans", dry_run=False,
        )

        trans = [r for r in records if r.infer_type == "transitivity"]
        assert len(trans) == 0

    @pytest.mark.asyncio
    async def test_transitivity_confidence_decay(self, store, activation, search):
        """Confidence = min(conf_AB, conf_BC) * decay."""
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        await store.create_entity(e_a)
        await store.create_entity(e_b)
        await store.create_entity(e_c)

        rel_ab = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_b.id,
            predicate="LOCATED_IN", confidence=0.9, group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id, target_id=e_c.id,
            predicate="LOCATED_IN", confidence=0.7, group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,
            consolidation_infer_transitivity_enabled=True,
            consolidation_infer_transitive_predicates=["LOCATED_IN"],
            consolidation_infer_transitivity_decay=0.8,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_trans", dry_run=False,
        )

        trans = [r for r in records if r.infer_type == "transitivity"]
        assert len(trans) == 1
        # min(0.9, 0.7) * 0.8 = 0.56
        assert trans[0].confidence == round(min(0.9, 0.7) * 0.8, 4)

    @pytest.mark.asyncio
    async def test_transitivity_dry_run(self, store, activation, search):
        """Records generated but no DB writes in dry_run."""
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        await store.create_entity(e_a)
        await store.create_entity(e_b)
        await store.create_entity(e_c)

        rel_ab = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id, target_id=e_b.id,
            predicate="LOCATED_IN", group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id, target_id=e_c.id,
            predicate="LOCATED_IN", group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,
            consolidation_infer_transitivity_enabled=True,
            consolidation_infer_transitive_predicates=["LOCATED_IN"],
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test", graph_store=store, activation_store=activation,
            search_index=search, cfg=cfg, cycle_id="cyc_trans", dry_run=True,
        )

        trans = [r for r in records if r.infer_type == "transitivity"]
        assert len(trans) == 1
        # But no actual relationship to C should exist
        rels = await store.get_relationships(e_a.id, group_id="test")
        targets = {r.target_id for r in rels if r.predicate == "LOCATED_IN"}
        assert e_c.id not in targets
