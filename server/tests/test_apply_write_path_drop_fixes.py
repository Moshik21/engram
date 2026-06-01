"""Regression tests for graph WRITE-path data loss in the apply engine.

Reproduces and guards four confirmed bugs on the SQLite (in-memory) backend:

  A. Entity drop — a validly-named extracted entity (e.g. a Project named
     "Helix") never reaches the store.
  B. Attribute loss on recurrence — when an entity recurs carrying new
     attributes (e.g. {title: "Director"}), the new attributes are lost.
  C. Duplicate self-entity — a recurring canonical "User" creates a second
     node instead of resolving to one.
  D. Garbage self-loop — a relationship whose resolved source == target is
     persisted instead of dropped.
"""

from __future__ import annotations

import uuid

import pytest

from engram.config import ActivationConfig
from engram.extraction.apply import ApplyEngine, apply_relationship_fact
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.models import EntityCandidate
from engram.models.episode import Episode
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest.fixture
async def graph():
    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def activation() -> MemoryActivationStore:
    return MemoryActivationStore()


def _engine(graph, activation) -> ApplyEngine:
    return ApplyEngine(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )


async def _episode(graph, content: str, group_id: str = "default") -> Episode:
    episode = Episode(
        id=f"ep_{uuid.uuid4().hex[:12]}",
        content=content,
        group_id=group_id,
    )
    await graph.create_episode(episode)
    return episode


async def _all_entities(graph, group_id: str = "default"):
    return await graph.find_entities(group_id=group_id, limit=1000)


@pytest.mark.asyncio
async def test_bug_a_project_entity_reaches_store(graph, activation):
    """BUG A: a Project-typed extracted entity must commit, not be dropped."""
    engine = _engine(graph, activation)
    episode = await _episode(graph, "Started a new job on the Helix project.")

    candidates = [
        EntityCandidate(name="Helix", entity_type="Project"),
        EntityCandidate(name="Priya", entity_type="Person"),
    ]
    outcome = await engine.apply_entities(candidates, episode, "default")

    stored = await _all_entities(graph)
    names = {e.name for e in stored}
    types = {e.entity_type for e in stored}

    assert "Helix" in names, f"Project entity dropped; stored={names}"
    assert "Project" in types, f"No Project-typed entity stored; types={types}"
    assert "Helix" in outcome.entity_map


@pytest.mark.asyncio
async def test_bug_b_recurring_attributes_merge(graph, activation):
    """BUG B: new attributes on a recurring entity must merge into the node."""
    engine = _engine(graph, activation)

    ep1 = await _episode(graph, "Priya is a team lead.")
    await engine.apply_entities(
        [EntityCandidate(name="Priya", entity_type="Person", attributes={"role": "team lead"})],
        ep1,
        "default",
    )

    ep2 = await _episode(graph, "Priya was promoted to Director.")
    await engine.apply_entities(
        [EntityCandidate(name="Priya", entity_type="Person", attributes={"title": "Director"})],
        ep2,
        "default",
    )

    stored = await _all_entities(graph)
    priyas = [e for e in stored if e.name == "Priya"]
    assert len(priyas) == 1, f"Expected single Priya, got {len(priyas)}"
    attrs = priyas[0].attributes or {}
    # Role/title attribute keys collapse to a canonical "role", and the newer
    # assertion (Director) SUPERSEDES the stale one (team lead) instead of both
    # coexisting -- the current-value extraction-quality fix. The new attribute
    # still merges into the node (BUG B); it now also overwrites the prior role.
    assert attrs.get("role") == "Director", f"Current role not surfaced; attrs={attrs}"
    assert "team lead" not in str(attrs), f"Stale role not superseded; attrs={attrs}"


@pytest.mark.asyncio
async def test_bug_c_recurring_user_dedups(graph, activation):
    """BUG C: a recurring canonical 'User' must resolve to one node."""
    engine = _engine(graph, activation)

    ep1 = await _episode(graph, "User prefers tea.")
    await engine.apply_entities(
        [EntityCandidate(name="User", entity_type="Concept")], ep1, "default"
    )
    ep2 = await _episode(graph, "User prefers coffee.")
    await engine.apply_entities(
        [EntityCandidate(name="User", entity_type="Concept")], ep2, "default"
    )

    stored = await _all_entities(graph)
    users = [e for e in stored if e.name == "User"]
    assert len(users) == 1, f"Duplicate User entities created: {len(users)}"


@pytest.mark.asyncio
async def test_bug_d_self_loop_dropped(graph, activation):
    """BUG D: a relationship with resolved source == target must be dropped."""
    engine = _engine(graph, activation)
    episode = await _episode(graph, "I play guitar.")
    await engine.apply_entities(
        [EntityCandidate(name="Guitar", entity_type="Concept")], episode, "default"
    )
    entity_map = {"Guitar": (await _all_entities(graph))[0].id}

    result = await apply_relationship_fact(
        graph_store=graph,
        canonicalizer=PredicateCanonicalizer(),
        cfg=ActivationConfig(),
        rel_data={"source": "Guitar", "target": "Guitar", "predicate": "FOCUSES_ON"},
        entity_map=entity_map,
        group_id="default",
        source_episode=episode.id,
    )

    assert result.created is False, "Self-loop was persisted"
    gid = entity_map["Guitar"]
    rels = await graph.get_relationships(gid, direction="outgoing", group_id="default")
    assert all(r.target_id != gid for r in rels), "Self-loop relationship stored"
