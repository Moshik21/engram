"""Tests for the edge inference consolidation phase."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.infer import (
    EdgeInferencePhase,
    _compute_pmi,
    _compute_tfidf_importance,
    _pmi_confidence,
)
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
        name=name,
        entity_type=entity_type,
        group_id=group_id,
    )


def _episode(group_id="test"):
    return Episode(
        id=f"ep_{uuid.uuid4().hex[:8]}",
        content="test",
        source="test",
        status="completed",
        group_id=group_id,
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
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
            source_id=e1.id,
            target_id=e2.id,
            predicate="KNOWS",
            group_id="test",
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
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
            group_id="group_b",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
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
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            group_id="test",
        )
        await store.create_relationship(rel_ab)
        await store.create_relationship(rel_bc)

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=20,  # No co-occurrence edges
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_trans",
            dry_run=False,
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
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            confidence=0.9,
            group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            confidence=0.8,
            group_id="test",
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_trans",
            dry_run=False,
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
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            group_id="test",
        )
        # Direct A→C already exists
        rel_ac = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_a.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            group_id="test",
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_trans",
            dry_run=False,
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
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            confidence=0.9,
            group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            confidence=0.7,
            group_id="test",
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_trans",
            dry_run=False,
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
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            group_id="test",
        )
        rel_bc = Relationship(
            id=f"rel_{uuid.uuid4().hex[:8]}",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            group_id="test",
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
            group_id="test",
            graph_store=store,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_trans",
            dry_run=True,
        )

        trans = [r for r in records if r.infer_type == "transitivity"]
        assert len(trans) == 1
        # But no actual relationship to C should exist
        rels = await store.get_relationships(e_a.id, group_id="test")
        targets = {r.target_id for r in rels if r.predicate == "LOCATED_IN"}
        assert e_c.id not in targets


# ---------------------------------------------------------------------------
# PMI Scoring Tests (Tier 2)
# ---------------------------------------------------------------------------


def _mock_graph_store(pairs, entities, ep_counts, total_episodes):
    """Build an AsyncMock graph_store for PMI tests."""
    gs = AsyncMock()
    gs.get_co_occurring_entity_pairs.return_value = pairs
    gs.get_entity_episode_counts.return_value = ep_counts
    gs.get_stats.return_value = {"total_episodes": total_episodes}
    gs.get_relationships_by_predicate.return_value = []

    async def _get_entity(eid, gid):
        return entities.get(eid)

    gs.get_entity.side_effect = _get_entity
    gs.create_relationship.return_value = "rel_mock"
    return gs


class TestPMIScoring:
    """Tests for Tier 2 PMI statistical scoring."""

    @pytest.mark.asyncio
    async def test_pmi_disabled_by_default(self):
        """Default config uses linear confidence, pmi_score is None."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0].pmi_score is None
        assert records[0].infer_type == "co_occurrence"

    @pytest.mark.asyncio
    async def test_pmi_basic_calculation(self):
        """PMI correctly computed, infer_type='co_occurrence_pmi'."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={e1.id: 10, e2.id: 10},
            total_episodes=100,
        )
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=0.0,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0].infer_type == "co_occurrence_pmi"
        assert records[0].pmi_score is not None
        # PMI = log2(0.05 / (0.1 * 0.1)) = log2(5) ≈ 2.322
        assert abs(records[0].pmi_score - 2.3219) < 0.01

    @pytest.mark.asyncio
    async def test_pmi_filters_below_min(self):
        """Pairs with PMI < pmi_min excluded."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        # Very common entities → low PMI
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 3)],
            entities=entities,
            ep_counts={e1.id: 50, e2.id: 50},
            total_episodes=100,
        )
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=5.0,  # Very high threshold
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 0

    @pytest.mark.asyncio
    async def test_pmi_rare_entities_higher_confidence(self):
        """tf-idf boosts rare entity pairs."""
        e1 = _entity("RareAlice")
        e2 = _entity("RareBob")
        e3 = _entity("CommonCharlie")
        e4 = _entity("CommonDana")
        entities = {e1.id: e1, e2.id: e2, e3.id: e3, e4.id: e4}
        # Rare: appear in 5/100, common: 20/100. Both co-occur 5 times.
        # Rare PMI ≈ 4.32, Common PMI ≈ 0.32 — both positive but rare is much higher
        gs = _mock_graph_store(
            pairs=[
                (e1.id, e2.id, 5),  # Rare pair
                (e3.id, e4.id, 5),  # Common pair
            ],
            entities=entities,
            ep_counts={e1.id: 5, e2.id: 5, e3.id: 20, e4.id: 20},
            total_episodes=100,
        )
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=0.0,
            consolidation_infer_tfidf_weight=0.3,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 2
        rare_rec = next(r for r in records if r.source_name == "RareAlice")
        common_rec = next(r for r in records if r.source_name == "CommonCharlie")
        # Rare entities should have higher confidence due to tf-idf
        assert rare_rec.confidence >= common_rec.confidence

    @pytest.mark.asyncio
    async def test_pmi_confidence_floor_respected(self):
        """Confidence never below floor."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        # ep_counts 10/10 with 3 co-occurrences → PMI = log2(3) ≈ 1.58 (positive)
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 3)],
            entities=entities,
            ep_counts={e1.id: 10, e2.id: 10},
            total_episodes=100,
        )
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.6,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=0.0,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0].confidence >= 0.6

    @pytest.mark.asyncio
    async def test_pmi_tfidf_weight_zero(self):
        """With tfidf_weight=0, only PMI matters."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={e1.id: 10, e2.id: 10},
            total_episodes=100,
        )
        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=0.0,
            consolidation_infer_tfidf_weight=0.0,
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        # With tfidf_weight=0, confidence is purely from PMI sigmoid
        assert records[0].confidence > 0.0

    @pytest.mark.asyncio
    async def test_pmi_with_transitivity(self):
        """Both PMI and transitivity passes run in same cycle."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        e_a = _entity("CityA", entity_type="Location")
        e_b = _entity("CityB", entity_type="Location")
        e_c = _entity("CityC", entity_type="Location")
        entities = {e.id: e for e in [e1, e2, e_a, e_b, e_c]}

        rel_ab = Relationship(
            id="rel_ab",
            source_id=e_a.id,
            target_id=e_b.id,
            predicate="LOCATED_IN",
            confidence=0.9,
            group_id="test",
        )
        rel_bc = Relationship(
            id="rel_bc",
            source_id=e_b.id,
            target_id=e_c.id,
            predicate="LOCATED_IN",
            confidence=0.8,
            group_id="test",
        )

        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={e1.id: 10, e2.id: 10},
            total_episodes=100,
        )
        gs.get_relationships_by_predicate.return_value = [rel_ab, rel_bc]

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_pmi_enabled=True,
            consolidation_infer_pmi_min=0.0,
            consolidation_infer_transitivity_enabled=True,
            consolidation_infer_transitive_predicates=["LOCATED_IN"],
        )
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        pmi_records = [r for r in records if r.infer_type == "co_occurrence_pmi"]
        trans_records = [r for r in records if r.infer_type == "transitivity"]
        assert len(pmi_records) == 1
        assert len(trans_records) == 1


# ---------------------------------------------------------------------------
# LLM Validation Tests (Tier 3)
# ---------------------------------------------------------------------------


def _make_llm_response(verdict: str, reason: str = "test"):
    """Create a mock Anthropic API response."""
    content_block = MagicMock()
    content_block.text = json.dumps({"verdict": verdict, "reason": reason})
    response = MagicMock()
    response.content = [content_block]
    return response


class TestLLMValidation:
    """Tests for Tier 3 LLM validation."""

    @pytest.mark.asyncio
    async def test_llm_disabled_by_default(self):
        """No llm_verdict set when LLM disabled."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        cfg = ActivationConfig(consolidation_infer_cooccurrence_min=3)
        phase = EdgeInferencePhase()
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0].llm_verdict is None

    @pytest.mark.asyncio
    async def test_llm_approves_edge(self):
        """Approved edge becomes llm_validated."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert len(records) == 1
        assert records[0].infer_type == "llm_validated"
        assert records[0].llm_verdict == "approved"
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_rejects_edge(self):
        """Rejected edge becomes llm_rejected and relationship invalidated."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("rejected")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert len(records) == 1
        assert records[0].infer_type == "llm_rejected"
        assert records[0].llm_verdict == "rejected"
        gs.invalidate_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_uncertain_leaves_edge(self):
        """Uncertain verdict leaves edge unchanged."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("uncertain")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        assert len(records) == 1
        assert records[0].infer_type == "co_occurrence"  # Unchanged
        assert records[0].llm_verdict == "uncertain"

    @pytest.mark.asyncio
    async def test_llm_failure_nonfatal(self):
        """API error caught, phase completes."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API failure")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        result, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        # Phase should complete despite error
        assert result.status == "success"
        assert len(records) == 1
        assert records[0].llm_verdict == "error"

    @pytest.mark.asyncio
    async def test_llm_respects_confidence_threshold(self):
        """Only edges above threshold validated."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 3)],  # Low count → low confidence
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.5,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.95,  # Higher than possible
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        # LLM should not be called (confidence below threshold)
        mock_client.messages.create.assert_not_called()
        assert records[0].llm_verdict is None

    @pytest.mark.asyncio
    async def test_llm_dry_run_skips_api(self):
        """Dry run sets llm_verdict='dry_run_skipped', no client calls."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )
        assert len(records) == 1
        assert records[0].llm_verdict == "dry_run_skipped"
        mock_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_uses_cached_system_prompt(self):
        """LLM validation uses cached system prompt."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store(
            pairs=[(e1.id, e2.id, 5)],
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_llm_max_per_cycle(self):
        """At most N edges validated."""
        entities = {}
        pairs = []
        for i in range(5):
            e1 = _entity(f"Entity{i}A")
            e2 = _entity(f"Entity{i}B")
            entities[e1.id] = e1
            entities[e2.id] = e2
            pairs.append((e1.id, e2.id, 5))

        gs = _mock_graph_store(
            pairs=pairs,
            entities=entities,
            ep_counts={},
            total_episodes=0,
        )
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_llm_max_per_cycle=2,
        )
        phase = EdgeInferencePhase(llm_client=mock_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )
        # Only 2 should be LLM-validated
        validated = [r for r in records if r.llm_verdict == "approved"]
        assert len(validated) == 2
        assert mock_client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# PMI Helper Unit Tests
# ---------------------------------------------------------------------------


class TestPMIHelpers:
    """Unit tests for PMI/tf-idf helper functions."""

    def test_compute_pmi_basic(self):
        # P(a,b) = 5/100 = 0.05, P(a) = 10/100 = 0.1, P(b) = 10/100 = 0.1
        # PMI = log2(0.05 / (0.1*0.1)) = log2(5) ≈ 2.322
        result = _compute_pmi(5, 10, 10, 100)
        assert abs(result - 2.3219) < 0.01

    def test_compute_pmi_zero_input(self):
        assert _compute_pmi(0, 10, 10, 100) == 0.0
        assert _compute_pmi(5, 0, 10, 100) == 0.0
        assert _compute_pmi(5, 10, 10, 0) == 0.0

    def test_compute_tfidf_importance(self):
        # Rare entity (appears in 1 of 100 episodes) should have high importance
        rare = _compute_tfidf_importance(1, 100)
        common = _compute_tfidf_importance(90, 100)
        assert rare > common
        assert 0.0 <= rare <= 1.0
        assert 0.0 <= common <= 1.0

    def test_compute_tfidf_importance_zero(self):
        assert _compute_tfidf_importance(0, 100) == 0.0
        assert _compute_tfidf_importance(5, 0) == 0.0

    def test_pmi_confidence_respects_floor(self):
        result = _pmi_confidence(0.0, 0.0, 0.0, 0.3, 0.6)
        assert result >= 0.6
