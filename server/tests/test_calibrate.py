"""Tests for CalibratePhase — preference calibration consolidation phase."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.calibrate import CalibratePhase
from engram.models.consolidation import CalibrationRecord

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
    """Minimal mock graph store for CalibratePhase tests."""

    def __init__(
        self,
        entities: list[MockEntity] | None = None,
        rels: list[MockRelationship] | None = None,
    ):
        self._entities = {e.id: e for e in (entities or [])}
        self._rels = rels or []
        self._updates: list[tuple] = []

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
            if group_id and e.group_id != group_id:
                match = False
            if match:
                results.append(e)
        return results[:limit]

    async def get_entity(self, entity_id: str, group_id: str = "default") -> MockEntity | None:
        return self._entities.get(entity_id)

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
            elif direction == "incoming" and r.target_id == entity_id:
                if predicate is None or r.predicate == predicate:
                    results.append(r)
        return results

    async def update_entity(self, entity_id: str, updates: dict, group_id: str = "default") -> None:
        self._updates.append((entity_id, updates, group_id))
        # Also apply in-memory for follow-up checks
        if entity_id in self._entities:
            for k, v in updates.items():
                setattr(self._entities[entity_id], k, v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalibratePhaseSkipped:
    """Phase is skipped when config disables it."""

    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self):
        cfg = ActivationConfig(preference_calibrate_enabled=False)
        phase = CalibratePhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraphStore(),
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
        )
        assert result.status == "skipped"
        assert records == []

    @pytest.mark.asyncio
    async def test_skipped_when_no_preference_entity(self):
        cfg = ActivationConfig(preference_calibrate_enabled=True)
        phase = CalibratePhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraphStore(),
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
        )
        assert result.items_processed == 0
        assert records == []


class TestCalibratePhaseDomainAggregation:
    """Phase aggregates PREFERS/AVOIDS edges into domain scores."""

    def _build_store(self) -> MockGraphStore:
        pref = MockEntity(
            id="pref_001",
            name="UserPreference",
            entity_type="PreferenceProfile",
            group_id="default",
            attributes={},
        )
        tech_entity = MockEntity(
            id="ent_tech",
            name="Python",
            entity_type="Technology",
            group_id="default",
        )
        person_entity = MockEntity(
            id="ent_person",
            name="Alice",
            entity_type="Person",
            group_id="default",
        )
        rels = [
            MockRelationship(
                id="rel_1",
                source_id="pref_001",
                target_id="ent_tech",
                predicate="PREFERS",
                weight=0.8,
            ),
            MockRelationship(
                id="rel_2",
                source_id="pref_001",
                target_id="ent_person",
                predicate="AVOIDS",
                weight=0.5,
            ),
        ]
        return MockGraphStore(
            entities=[pref, tech_entity, person_entity],
            rels=rels,
        )

    @pytest.mark.asyncio
    async def test_domain_scores_computed(self):
        cfg = ActivationConfig(preference_calibrate_enabled=True)
        phase = CalibratePhase()
        store = self._build_store()
        result, records = await phase.execute(
            group_id="default",
            graph_store=store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
        )
        assert result.items_processed == 2
        assert result.items_affected == 2
        assert len(records) == 2

        # Check domain scores in records
        domain_map = {r.domain: r for r in records}
        assert "technical" in domain_map
        assert "personal" in domain_map
        # Technical should be positive (PREFERS)
        assert domain_map["technical"].preference_score > 0
        # Personal should be negative (AVOIDS)
        assert domain_map["personal"].preference_score < 0

    @pytest.mark.asyncio
    async def test_dry_run_does_not_persist(self):
        cfg = ActivationConfig(preference_calibrate_enabled=True)
        phase = CalibratePhase()
        store = self._build_store()
        result, records = await phase.execute(
            group_id="default",
            graph_store=store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
            dry_run=True,
        )
        assert result.items_affected == 2
        # No update_entity calls in dry_run
        assert len(store._updates) == 0

    @pytest.mark.asyncio
    async def test_persists_when_not_dry_run(self):
        cfg = ActivationConfig(preference_calibrate_enabled=True)
        phase = CalibratePhase()
        store = self._build_store()
        result, records = await phase.execute(
            group_id="default",
            graph_store=store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
            dry_run=False,
        )
        assert len(store._updates) == 1
        update_args = store._updates[0]
        assert update_args[0] == "pref_001"
        attrs = update_args[1]["attributes"]
        assert "domain_preference_scores" in attrs
        assert "domain_entity_counts" in attrs


class TestCalibratePhaseDecay:
    """Decay is applied to previous domain scores not seen this cycle."""

    @pytest.mark.asyncio
    async def test_decay_applied_to_unseen_domains(self):
        pref = MockEntity(
            id="pref_001",
            name="UserPreference",
            entity_type="PreferenceProfile",
            group_id="default",
            attributes={
                "domain_preference_scores": {
                    "technical": 0.8,
                    "creative": 0.5,  # Not in current edges
                },
            },
        )
        tech_entity = MockEntity(
            id="ent_tech",
            name="Python",
            entity_type="Technology",
            group_id="default",
        )
        rels = [
            MockRelationship(
                id="rel_1",
                source_id="pref_001",
                target_id="ent_tech",
                predicate="PREFERS",
                weight=0.8,
            ),
        ]
        store = MockGraphStore(entities=[pref, tech_entity], rels=rels)
        cfg = ActivationConfig(
            preference_calibrate_enabled=True,
            preference_decay_rate=0.01,
        )
        phase = CalibratePhase()
        result, records = await phase.execute(
            group_id="default",
            graph_store=store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-1",
        )
        assert result.items_processed == 1
        # Records should contain technical domain
        domain_map = {r.domain: r for r in records}
        assert "technical" in domain_map
        assert domain_map["technical"].decay_applied == 0.01


class TestCalibratePhaseRecords:
    """Audit records are generated correctly."""

    @pytest.mark.asyncio
    async def test_records_have_correct_fields(self):
        pref = MockEntity(
            id="pref_001",
            name="UserPreference",
            entity_type="PreferenceProfile",
            group_id="default",
            attributes={},
        )
        tech_entity = MockEntity(
            id="ent_tech",
            name="React",
            entity_type="Software",
            group_id="default",
        )
        rels = [
            MockRelationship(
                id="rel_1",
                source_id="pref_001",
                target_id="ent_tech",
                predicate="PREFERS",
                weight=1.0,
            ),
        ]
        store = MockGraphStore(entities=[pref, tech_entity], rels=rels)
        cfg = ActivationConfig(preference_calibrate_enabled=True)
        phase = CalibratePhase()
        _, records = await phase.execute(
            group_id="default",
            graph_store=store,
            activation_store=None,
            search_index=None,
            cfg=cfg,
            cycle_id="cycle-42",
        )
        assert len(records) == 1
        rec = records[0]
        assert isinstance(rec, CalibrationRecord)
        assert rec.cycle_id == "cycle-42"
        assert rec.group_id == "default"
        assert rec.domain == "technical"
        assert rec.entity_count == 1
        assert rec.id.startswith("pcal_")
