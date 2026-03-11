"""Tests for Brain Architecture Phase 3: Schema Formation."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.schema_formation import (
    SchemaFormationPhase,
    _fingerprint_to_members,
    _generate_schema_name,
    _schema_matches_fingerprint,
    compute_fingerprint,
)
from engram.models.consolidation import CycleContext, SchemaRecord
from engram.models.entity import Entity
from engram.models.relationship import Relationship

# --- Helper factories ---


def _entity(
    eid: str,
    etype: str = "Person",
    name: str = "Test",
    attrs: dict | None = None,
) -> Entity:
    return Entity(
        id=eid, name=name, entity_type=etype, group_id="default",
        attributes=attrs or {},
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    )


def _rel(src: str, tgt: str, predicate: str = "KNOWS") -> Relationship:
    return Relationship(
        id=f"rel_{src}_{tgt}",
        source_id=src, target_id=tgt, predicate=predicate,
        weight=1.0, group_id="default",
    )


# --- Unit tests: compute_fingerprint ---


def test_compute_fingerprint_outgoing():
    entity_cache = {"b": _entity("b", "Technology")}
    rels = [_rel("a", "b", "EXPERT_IN")]
    fp = compute_fingerprint("Person", rels, entity_cache, "a")
    assert fp == frozenset({("Person", "EXPERT_IN", "Technology")})


def test_compute_fingerprint_incoming():
    entity_cache = {"a": _entity("a", "Person")}
    rels = [_rel("a", "b", "MEMBER_OF")]
    fp = compute_fingerprint("Organization", rels, entity_cache, "b")
    assert fp == frozenset({("Person", "MEMBER_OF", "Organization")})


def test_compute_fingerprint_both_directions():
    entity_cache = {
        "a": _entity("a", "Person"),
        "b": _entity("b", "Technology"),
        "c": _entity("c", "Organization"),
    }
    rels = [
        _rel("a", "b", "EXPERT_IN"),
        _rel("c", "a", "EMPLOYS"),
    ]
    fp = compute_fingerprint("Person", rels, entity_cache, "a")
    assert ("Person", "EXPERT_IN", "Technology") in fp
    assert ("Organization", "EMPLOYS", "Person") in fp
    assert len(fp) == 2


def test_compute_fingerprint_no_relationships():
    fp = compute_fingerprint("Person", [], {}, "a")
    assert fp == frozenset()


def test_compute_fingerprint_missing_cache_entry():
    rels = [_rel("a", "b", "KNOWS")]
    fp = compute_fingerprint("Person", rels, {}, "a")
    assert fp == frozenset()


# --- Unit tests: _generate_schema_name ---


def test_generate_schema_name():
    fp = frozenset({
        ("Person", "EXPERT_IN", "Technology"),
        ("Person", "MEMBER_OF", "Organization"),
    })
    name = _generate_schema_name(fp)
    assert "Person" in name
    assert "EXPERT_IN" in name
    assert "MEMBER_OF" in name


def test_generate_schema_name_empty():
    assert _generate_schema_name(frozenset()) == "EmptySchema"


# --- Unit tests: _fingerprint_to_members ---


def test_fingerprint_to_members():
    fp = frozenset({("Person", "EXPERT_IN", "Technology")})
    members = _fingerprint_to_members(fp)
    assert len(members) == 1
    assert members[0]["member_type"] == "Technology"
    assert members[0]["member_predicate"] == "EXPERT_IN"
    assert "role_label" in members[0]


def test_fingerprint_to_members_multi():
    fp = frozenset({
        ("Person", "EXPERT_IN", "Technology"),
        ("Person", "MEMBER_OF", "Organization"),
    })
    members = _fingerprint_to_members(fp)
    assert len(members) == 2


# --- Unit tests: _schema_matches_fingerprint ---


def test_schema_matches_fingerprint_match():
    fp = frozenset({("Person", "EXPERT_IN", "Technology")})
    members = _fingerprint_to_members(fp)
    assert _schema_matches_fingerprint(members, fp) is True


def test_schema_matches_fingerprint_no_match():
    fp1 = frozenset({("Person", "EXPERT_IN", "Technology")})
    fp2 = frozenset({("Person", "MEMBER_OF", "Organization")})
    members = _fingerprint_to_members(fp1)
    assert _schema_matches_fingerprint(members, fp2) is False


def test_schema_matches_fingerprint_different_length():
    fp1 = frozenset({("Person", "EXPERT_IN", "Technology")})
    fp2 = frozenset({
        ("Person", "EXPERT_IN", "Technology"),
        ("Person", "MEMBER_OF", "Organization"),
    })
    members = _fingerprint_to_members(fp1)
    assert _schema_matches_fingerprint(members, fp2) is False


# --- Config tests ---


def test_config_schema_off_by_default():
    cfg = ActivationConfig()
    assert cfg.schema_formation_enabled is False


def test_config_standard_enables_schema():
    cfg = ActivationConfig(consolidation_profile="standard")
    assert cfg.schema_formation_enabled is True


def test_config_conservative_does_not_enable_schema():
    cfg = ActivationConfig(consolidation_profile="conservative")
    assert cfg.schema_formation_enabled is False


def test_config_defaults():
    cfg = ActivationConfig()
    assert cfg.schema_min_instances == 5
    assert cfg.schema_min_edges == 2
    assert cfg.schema_max_per_cycle == 5
    assert cfg.schema_max_entities_scan == 500


# --- Phase tests ---


def _make_phase_deps(entities=None, rels_by_entity=None, existing_schemas=None):
    """Create mock dependencies for SchemaFormationPhase."""
    graph_store = AsyncMock()
    graph_store.find_entities = AsyncMock(return_value=entities or [])
    graph_store.find_entities_by_type = AsyncMock(return_value=existing_schemas or [])
    graph_store.get_schema_members = AsyncMock(return_value=[])
    graph_store.save_schema_members = AsyncMock()
    graph_store.create_entity = AsyncMock()
    graph_store.create_relationship = AsyncMock()
    graph_store.update_entity = AsyncMock()

    rels_map = rels_by_entity or {}

    async def mock_get_rels(entity_id, direction="both", group_id="default"):
        return rels_map.get(entity_id, [])

    graph_store.get_relationships = mock_get_rels

    activation_store = AsyncMock()
    search_index = AsyncMock()

    return graph_store, activation_store, search_index


async def _run_phase(graph_store, activation_store, search_index, cfg, context=None, dry_run=False):
    phase = SchemaFormationPhase()
    return await phase.execute(
        group_id="test", graph_store=graph_store,
        activation_store=activation_store, search_index=search_index,
        cfg=cfg, cycle_id="cyc1", context=context, dry_run=dry_run,
    )


@pytest.mark.asyncio
async def test_phase_skipped_when_disabled():
    graph_store, activation_store, search_index = _make_phase_deps()
    cfg = ActivationConfig()
    assert cfg.schema_formation_enabled is False

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.status == "skipped"
    assert records == []


@pytest.mark.asyncio
async def test_phase_skipped_no_entities():
    graph_store, activation_store, search_index = _make_phase_deps(entities=[])
    cfg = ActivationConfig(schema_formation_enabled=True)

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.status == "skipped"
    assert records == []


@pytest.mark.asyncio
async def test_phase_creates_schema_when_enough_instances():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {}
    for i in range(5):
        rels_map[f"p{i}"] = [_rel(f"p{i}", "tech1", "EXPERT_IN")]
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )
    context = CycleContext()

    result, records = await _run_phase(
        graph_store, activation_store, search_index, cfg, context,
    )

    assert result.items_affected >= 1
    assert len(records) >= 1
    assert records[0].action == "created"
    assert records[0].instance_count >= 5
    assert "schema_" in records[0].schema_entity_id

    graph_store.create_entity.assert_called()
    created_entity = graph_store.create_entity.call_args[0][0]
    assert created_entity.entity_type == "Schema"

    assert graph_store.create_relationship.call_count >= 5
    assert len(context.schema_entity_ids) >= 1
    assert len(context.affected_entity_ids) >= 1
    graph_store.save_schema_members.assert_called()


@pytest.mark.asyncio
async def test_phase_does_not_create_when_too_few_instances():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(3)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(3)}
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.items_affected == 0
    assert records == []
    graph_store.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_phase_does_not_create_when_too_few_edges():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=3,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.items_affected == 0


@pytest.mark.asyncio
async def test_phase_skips_noisy_motif_when_support_is_weak():
    noisy_attrs = {
        "maturity_features_v1": {
            "episode_count": 1,
            "support_windows": 1,
            "maturity_score": 0.15,
        },
    }
    entities = [_entity(f"p{i}", "Person", f"Person{i}", attrs=noisy_attrs) for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.items_affected == 0
    assert records == []
    graph_store.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_phase_prefers_stable_supported_motif():
    stable_attrs = {
        "mat_tier": "semantic",
        "maturity_features_v1": {
            "episode_count": 4,
            "support_windows": 2,
            "maturity_score": 0.92,
        },
    }
    noisy_attrs = {
        "maturity_features_v1": {
            "episode_count": 1,
            "support_windows": 1,
            "maturity_score": 0.15,
        },
    }

    stable_people = [_entity(f"p{i}", "Person", f"Person{i}", attrs=stable_attrs) for i in range(5)]
    noisy_people = [_entity(f"q{i}", "Person", f"Other{i}", attrs=noisy_attrs) for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    org = _entity("org1", "Organization", "Acme")
    entities = stable_people + noisy_people + [tech, org]

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map.update({f"q{i}": [_rel(f"q{i}", "org1", "MEMBER_OF")] for i in range(5)})
    rels_map["tech1"] = []
    rels_map["org1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
        schema_max_per_cycle=1,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)
    assert result.items_affected == 1
    assert len(records) == 1
    assert "EXPERT_IN" in records[0].schema_name


@pytest.mark.asyncio
async def test_phase_reinforces_existing_schema():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map["tech1"] = []

    existing_schema = _entity("schema_existing", "Schema", "Existing Schema")
    fp = frozenset({("Person", "EXPERT_IN", "Technology")})
    expected_members = _fingerprint_to_members(fp)

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
        existing_schemas=[existing_schema],
    )
    graph_store.get_schema_members = AsyncMock(return_value=expected_members)

    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)

    assert len(records) == 1
    assert records[0].action == "reinforced"
    assert records[0].schema_entity_id == "schema_existing"
    activation_store.record_access.assert_called_once()
    graph_store.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_phase_respects_max_per_cycle():
    entities = []
    rels_map = {}
    for pattern_idx in range(20):
        pred = f"REL_{pattern_idx}"
        tgt_type = f"Type_{pattern_idx}"
        tgt = _entity(f"tgt_{pattern_idx}", tgt_type, f"Target{pattern_idx}")
        entities.append(tgt)
        for i in range(5):
            eid = f"p{pattern_idx}_{i}"
            entities.append(_entity(eid, "Person", f"Person{pattern_idx}_{i}"))
            rels_map[eid] = [_rel(eid, f"tgt_{pattern_idx}", pred)]
        rels_map[f"tgt_{pattern_idx}"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
        schema_max_per_cycle=3,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)

    created_records = [r for r in records if r.action == "created"]
    assert len(created_records) <= 3


@pytest.mark.asyncio
async def test_phase_respects_dry_run():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )

    result, records = await _run_phase(
        graph_store, activation_store, search_index, cfg, dry_run=True,
    )

    assert len(records) >= 1
    assert records[0].action == "created"
    graph_store.create_entity.assert_not_called()
    graph_store.create_relationship.assert_not_called()
    graph_store.save_schema_members.assert_not_called()


@pytest.mark.asyncio
async def test_phase_schema_record_fields():
    entities = [_entity(f"p{i}", "Person", f"Person{i}") for i in range(5)]
    tech = _entity("tech1", "Technology", "Python")
    entities.append(tech)

    rels_map = {f"p{i}": [_rel(f"p{i}", "tech1", "EXPERT_IN")] for i in range(5)}
    rels_map["tech1"] = []

    graph_store, activation_store, search_index = _make_phase_deps(
        entities=entities, rels_by_entity=rels_map,
    )
    cfg = ActivationConfig(
        schema_formation_enabled=True,
        schema_min_instances=5,
        schema_min_edges=1,
    )

    result, records = await _run_phase(graph_store, activation_store, search_index, cfg)

    rec = records[0]
    assert isinstance(rec, SchemaRecord)
    assert rec.cycle_id == "cyc1"
    assert rec.group_id == "test"
    assert rec.instance_count >= 5
    assert rec.predicate_count >= 1
    assert rec.action == "created"
    assert rec.schema_name != ""


# --- Integration tests ---


def test_engine_has_15_phases():
    from engram.consolidation.engine import ConsolidationEngine

    cfg = ActivationConfig()
    e = ConsolidationEngine(AsyncMock(), AsyncMock(), AsyncMock(), cfg)
    phases = [p.name for p in e._phases]
    assert len(phases) == 15
    assert phases == [
        "triage", "merge", "infer", "evidence_adjudication", "edge_adjudication", "replay",
        "prune", "compact", "mature", "semanticize", "schema", "reindex",
        "graph_embed", "microglia", "dream",
    ]


def test_phase_tiers_has_schema():
    from engram.consolidation.scheduler import PHASE_TIERS

    assert "schema" in PHASE_TIERS
    assert PHASE_TIERS["schema"] == "cold"


def test_cycle_context_has_schema_entity_ids():
    ctx = CycleContext()
    assert hasattr(ctx, "schema_entity_ids")
    assert ctx.schema_entity_ids == set()


def test_schema_record_dataclass():
    rec = SchemaRecord(
        cycle_id="cyc1",
        group_id="test",
        schema_entity_id="schema_abc",
        schema_name="Test Schema",
        instance_count=5,
        predicate_count=2,
        action="created",
    )
    assert rec.id.startswith("sch_")
    assert rec.timestamp > 0
    assert rec.action == "created"
