"""Tests for bi-temporal supersession (M3.4, cfg.supersession_enabled).

The always-on EXCLUSIVE_PREDICATES set already supersedes the location/
employment/role classes (LIVES_IN -> LOCATED_IN etc.). The flag extends the
same commit-time invalidation to the additional high-value classes
(USES_VERSION, NAMED/IS_NAMED, PREFERS) and records superseded-edge
provenance on the apply result. Flag off = today's behavior: the new classes
accumulate, the legacy exclusive set still invalidates, no provenance keys.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from engram.config import ActivationConfig
from engram.extraction.apply import apply_relationship_fact
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.models.entity import Entity
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest.fixture
async def graph():
    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


_ENTITIES = {
    "Konner": ("Person", "ent_konner"),
    "Seattle": ("Location", "ent_seattle"),
    "Denver": ("Location", "ent_denver"),
    "Python 3.11": ("Technology", "ent_py311"),
    "Python 3.12": ("Technology", "ent_py312"),
    "Alice": ("Person", "ent_alice"),
    "Bob": ("Person", "ent_bob"),
}


async def _seed_entities(graph) -> dict[str, str]:
    entity_map: dict[str, str] = {}
    for name, (etype, eid) in _ENTITIES.items():
        await graph.create_entity(Entity(id=eid, name=name, entity_type=etype, group_id="default"))
        entity_map[name] = eid
    return entity_map


async def _apply(graph, cfg, entity_map, source, predicate, target):
    return await apply_relationship_fact(
        graph_store=graph,
        canonicalizer=PredicateCanonicalizer(),
        cfg=cfg,
        rel_data={"source": source, "target": target, "predicate": predicate},
        entity_map=entity_map,
        group_id="default",
        source_episode=f"ep_{uuid.uuid4().hex[:12]}",
    )


async def _edges(graph, source_id, predicate, active_only=True):
    return await graph.get_relationships(
        source_id,
        direction="outgoing",
        predicate=predicate,
        active_only=active_only,
        group_id="default",
    )


@pytest.mark.asyncio
async def test_denver_supersedes_seattle_flag_on(graph):
    """End-to-end: "moved to Denver" ends the Seattle edge; only Denver stays live."""
    cfg = ActivationConfig(supersession_enabled=True)
    entity_map = await _seed_entities(graph)

    first = await _apply(graph, cfg, entity_map, "Konner", "LIVES_IN", "Seattle")
    assert first.action == "created"
    # LIVES_IN canonicalizes to LOCATED_IN before commit.
    assert first.predicate == "LOCATED_IN"

    second = await _apply(graph, cfg, entity_map, "Konner", "LIVES_IN", "Denver")
    assert second.action == "created"
    assert "superseded_prior" in second.constraints_hit

    active = await _edges(graph, "ent_konner", "LOCATED_IN")
    assert [r.target_id for r in active] == ["ent_denver"]

    all_edges = await _edges(graph, "ent_konner", "LOCATED_IN", active_only=False)
    superseded = [r for r in all_edges if r.target_id == "ent_seattle"]
    assert len(superseded) == 1
    assert superseded[0].valid_to is not None


@pytest.mark.asyncio
async def test_new_class_supersedes_flag_on(graph):
    """Flag on: a USES_VERSION re-assertion with a new target closes the old edge."""
    cfg = ActivationConfig(supersession_enabled=True)
    entity_map = await _seed_entities(graph)

    first = await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.11")
    old_id = first.metadata["relationship_id"]

    second = await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.12")
    assert second.action == "created"
    assert second.metadata["superseded_edge_ids"] == [old_id]

    active = await _edges(graph, "ent_konner", "USES_VERSION")
    assert [r.target_id for r in active] == ["ent_py312"]


@pytest.mark.asyncio
async def test_flag_off_new_class_both_live(graph):
    """Flag off (default): the new classes accumulate — today's behavior."""
    cfg = ActivationConfig()
    assert cfg.supersession_enabled is False
    entity_map = await _seed_entities(graph)

    await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.11")
    second = await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.12")
    assert "superseded_prior" not in second.constraints_hit
    assert "superseded_edge_ids" not in second.metadata

    active = await _edges(graph, "ent_konner", "USES_VERSION")
    assert {r.target_id for r in active} == {"ent_py311", "ent_py312"}


@pytest.mark.asyncio
async def test_flag_off_legacy_exclusive_still_supersedes(graph):
    """Flag off must not regress the pre-existing exclusive-predicate invalidation."""
    cfg = ActivationConfig()
    entity_map = await _seed_entities(graph)

    await _apply(graph, cfg, entity_map, "Konner", "LIVES_IN", "Seattle")
    second = await _apply(graph, cfg, entity_map, "Konner", "LIVES_IN", "Denver")
    assert second.action == "created"
    # Pre-existing behavior: invalidates, but carries no supersession provenance.
    assert "exclusive_predicate" in second.constraints_hit
    assert "superseded_prior" not in second.constraints_hit
    assert "superseded_edge_ids" not in second.metadata

    active = await _edges(graph, "ent_konner", "LOCATED_IN")
    assert [r.target_id for r in active] == ["ent_denver"]


@pytest.mark.asyncio
async def test_same_target_reassertion_reinforces(graph):
    """Re-asserting the same target never supersedes — it hits the duplicate path."""
    cfg = ActivationConfig(supersession_enabled=True)
    entity_map = await _seed_entities(graph)

    await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.12")
    second = await _apply(graph, cfg, entity_map, "Konner", "USES_VERSION", "Python 3.12")
    assert second.action in {"duplicate_skipped", "updated_existing"}
    assert "superseded_prior" not in second.constraints_hit

    active = await _edges(graph, "ent_konner", "USES_VERSION")
    assert len(active) == 1
    assert active[0].valid_to is None


@pytest.mark.asyncio
async def test_non_exclusive_predicate_never_supersedes(graph):
    """KNOWS is not exclusive: multiple targets stay live even with the flag on."""
    cfg = ActivationConfig(supersession_enabled=True)
    entity_map = await _seed_entities(graph)

    await _apply(graph, cfg, entity_map, "Konner", "KNOWS", "Alice")
    second = await _apply(graph, cfg, entity_map, "Konner", "KNOWS", "Bob")
    assert "exclusive_predicate" not in second.constraints_hit
    assert "superseded_edge_ids" not in second.metadata

    active = await _edges(graph, "ent_konner", "KNOWS")
    assert {r.target_id for r in active} == {"ent_alice", "ent_bob"}


@pytest.mark.asyncio
async def test_supersession_provenance_attributes(graph):
    """The superseding result carries superseded_edge_ids + a parseable superseded_at."""
    cfg = ActivationConfig(supersession_enabled=True)
    entity_map = await _seed_entities(graph)

    first = await _apply(graph, cfg, entity_map, "Konner", "PREFERS", "Python 3.12")
    old_id = first.metadata["relationship_id"]
    second = await _apply(graph, cfg, entity_map, "Konner", "PREFERS", "Python 3.11")

    assert second.metadata["superseded_edge_ids"] == [old_id]
    superseded_at = datetime.fromisoformat(second.metadata["superseded_at"])
    all_edges = await _edges(graph, "ent_konner", "PREFERS", active_only=False)
    old = next(r for r in all_edges if r.id == old_id)
    # The closed edge's valid_to matches the recorded supersession instant.
    assert old.valid_to == superseded_at
