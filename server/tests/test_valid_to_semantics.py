"""Tests for valid_to temporal semantics: TTL edges should be visible until expiry."""

from __future__ import annotations

import socket
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from engram.models.entity import Entity
from engram.models.relationship import Relationship


def _helix_available() -> bool:
    try:
        socket.create_connection(("localhost", 6969), timeout=2)
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_helix,
    pytest.mark.skipif(not _helix_available(), reason="HelixDB not available"),
]


@pytest.fixture
async def graph():
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    await store.initialize()
    yield store
    await store.close()


async def _create_entity(graph, eid: str, name: str, etype: str = "Concept", group_id: str = "default"):
    entity = Entity(
        id=eid,
        name=name,
        entity_type=etype,
        summary=f"Summary for {name}",
        group_id=group_id,
    )
    await graph.create_entity(entity)
    return entity


async def _create_ttl_edge(
    graph,
    src: str,
    tgt: str,
    days_from_now: int,
    rel_id: str = "rel_ttl1",
    group_id: str = "default",
):
    valid_to = datetime.utcnow() + timedelta(days=days_from_now)
    rel = Relationship(
        id=rel_id,
        source_id=src,
        target_id=tgt,
        predicate="DREAM_ASSOCIATED",
        weight=0.1,
        valid_to=valid_to,
        group_id=group_id,
        confidence=0.5,
    )
    await graph.create_relationship(rel)
    return rel


@pytest.mark.asyncio
async def test_ttl_edge_visible_in_get_relationships(graph):
    """A TTL edge with future valid_to appears in get_relationships(active_only=True)."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_ttl_edge(graph, "e1", "e2", days_from_now=30)

    rels = await graph.get_relationships("e1", active_only=True, group_id="default")
    assert len(rels) == 1
    assert rels[0].predicate == "DREAM_ASSOCIATED"


@pytest.mark.asyncio
async def test_ttl_edge_visible_in_get_active_neighbors_with_weights(graph):
    """A TTL edge appears in get_active_neighbors_with_weights."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_ttl_edge(graph, "e1", "e2", days_from_now=30)

    neighbors = await graph.get_active_neighbors_with_weights("e1", group_id="default")
    assert len(neighbors) == 1
    assert neighbors[0][0] == "e2"  # neighbor_id


@pytest.mark.asyncio
async def test_ttl_edge_visible_in_get_neighbors(graph):
    """A TTL edge appears in get_neighbors()."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_ttl_edge(graph, "e1", "e2", days_from_now=30)

    neighbors = await graph.get_neighbors("e1", hops=1, group_id="default")
    assert len(neighbors) == 1
    entity, rel = neighbors[0]
    assert entity.id == "e2"


@pytest.mark.asyncio
async def test_entity_with_ttl_edge_not_dead(graph):
    """An entity with only a TTL edge is NOT returned by get_dead_entities."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_ttl_edge(graph, "e1", "e2", days_from_now=30)

    # Set entity to look old enough — requires direct DB access (SQLite only)
    if not hasattr(graph, "db"):
        pytest.skip("SQLite-only test (direct SQL update)")
    old_date = (datetime.utcnow() - timedelta(days=60)).isoformat()
    await graph.db.execute("UPDATE entities SET created_at = ? WHERE id = ?", (old_date, "e1"))
    await graph.db.commit()

    dead = await graph.get_dead_entities("default", min_age_days=30)
    dead_ids = {e.id for e in dead}
    assert "e1" not in dead_ids


@pytest.mark.asyncio
async def test_update_relationship_weight_on_ttl_edge(graph):
    """update_relationship_weight works on TTL edges."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_ttl_edge(graph, "e1", "e2", days_from_now=30)

    new_weight = await graph.update_relationship_weight(
        "e1", "e2", weight_delta=0.5, group_id="default"
    )
    assert new_weight is not None
    assert new_weight == pytest.approx(0.6, abs=0.01)


@pytest.mark.asyncio
async def test_expired_edge_invisible_in_active_queries(graph):
    """An edge with valid_to in the past does NOT appear in active queries."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")

    # Create edge that expired yesterday
    past = datetime.utcnow() - timedelta(days=1)
    rel = Relationship(
        id="rel_expired1",
        source_id="e1",
        target_id="e2",
        predicate="DREAM_ASSOCIATED",
        weight=0.1,
        valid_to=past,
        group_id="default",
    )
    await graph.create_relationship(rel)

    # Should NOT appear in active queries
    rels = await graph.get_relationships("e1", active_only=True, group_id="default")
    assert len(rels) == 0

    neighbors = await graph.get_active_neighbors_with_weights("e1", group_id="default")
    assert len(neighbors) == 0

    neighbor_pairs = await graph.get_neighbors("e1", hops=1, group_id="default")
    assert len(neighbor_pairs) == 0


@pytest.mark.asyncio
async def test_path_exists_within_hops(graph):
    """path_exists_within_hops finds direct and 2-hop connections."""
    await _create_entity(graph, "e1", "Alpha")
    await _create_entity(graph, "e2", "Beta")
    await _create_entity(graph, "e3", "Gamma")

    # Direct e1->e2
    rel1 = Relationship(
        id="rel_1",
        source_id="e1",
        target_id="e2",
        predicate="RELATED_TO",
        weight=1.0,
        group_id="default",
    )
    await graph.create_relationship(rel1)

    # e2->e3
    rel2 = Relationship(
        id="rel_2",
        source_id="e2",
        target_id="e3",
        predicate="RELATED_TO",
        weight=1.0,
        group_id="default",
    )
    await graph.create_relationship(rel2)

    # e1 to e2: 1 hop
    assert await graph.path_exists_within_hops("e1", "e2", 1, "default") is True
    # e1 to e3: 2 hops
    assert await graph.path_exists_within_hops("e1", "e3", 2, "default") is True
    # e1 to e3: only 1 hop — should fail
    assert await graph.path_exists_within_hops("e1", "e3", 1, "default") is False


@pytest.mark.asyncio
async def test_get_expired_relationships(graph):
    """get_expired_relationships returns only expired edges."""
    gid = f"test_{uuid4().hex[:8]}"
    await _create_entity(graph, f"e1_{gid}", "Alpha", group_id=gid)
    await _create_entity(graph, f"e2_{gid}", "Beta", group_id=gid)

    # Active TTL edge (future)
    await _create_ttl_edge(
        graph, f"e1_{gid}", f"e2_{gid}", days_from_now=30,
        rel_id=f"rel_future_{gid}", group_id=gid,
    )

    # Expired edge
    past = datetime.utcnow() - timedelta(days=1)
    expired_rel = Relationship(
        id=f"rel_past_{gid}",
        source_id=f"e1_{gid}",
        target_id=f"e2_{gid}",
        predicate="DREAM_ASSOCIATED",
        weight=0.1,
        valid_to=past,
        group_id=gid,
    )
    await graph.create_relationship(expired_rel)

    expired = await graph.get_expired_relationships(gid)
    assert len(expired) == 1
    assert expired[0].id == f"rel_past_{gid}"

    # Filter by predicate
    expired_filtered = await graph.get_expired_relationships(gid, predicate="NONEXISTENT")
    assert len(expired_filtered) == 0
