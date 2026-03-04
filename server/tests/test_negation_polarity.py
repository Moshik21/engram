"""Tests for negation/uncertainty polarity handling."""

from __future__ import annotations

from datetime import datetime

import pytest
import pytest_asyncio

from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.sqlite.graph import SQLiteGraphStore


@pytest_asyncio.fixture
async def graph_store(tmp_path):
    """Create and initialize a SQLite graph store."""
    store = SQLiteGraphStore(str(tmp_path / "test_polarity.db"))
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def seed_entities(graph_store):
    """Create test entities."""
    e1 = Entity(
        id="ent_alice", name="Alice", entity_type="Person",
        summary="Engineer", group_id="default",
    )
    e2 = Entity(
        id="ent_python", name="Python", entity_type="Technology",
        summary="Programming language", group_id="default",
    )
    e3 = Entity(
        id="ent_acme", name="Acme", entity_type="Organization",
        summary="Tech company", group_id="default",
    )
    await graph_store.create_entity(e1)
    await graph_store.create_entity(e2)
    await graph_store.create_entity(e3)
    return {"alice": e1, "python": e2, "acme": e3}


# --- Relationship model ---


def test_relationship_polarity_default():
    """Polarity defaults to 'positive'."""
    rel = Relationship(
        id="rel_1", source_id="a", target_id="b", predicate="USES",
    )
    assert rel.polarity == "positive"


def test_relationship_polarity_negative():
    """Polarity can be set to 'negative'."""
    rel = Relationship(
        id="rel_1", source_id="a", target_id="b", predicate="USES",
        polarity="negative",
    )
    assert rel.polarity == "negative"


def test_relationship_polarity_uncertain():
    """Polarity can be set to 'uncertain'."""
    rel = Relationship(
        id="rel_1", source_id="a", target_id="b", predicate="USES",
        polarity="uncertain",
    )
    assert rel.polarity == "uncertain"


# --- SQLite schema migration ---


@pytest.mark.asyncio
async def test_polarity_column_exists(graph_store):
    """Migration adds polarity column to relationships table."""
    cursor = await graph_store.db.execute("PRAGMA table_info(relationships)")
    columns = {row[1] for row in await cursor.fetchall()}
    assert "polarity" in columns


# --- CRUD with polarity ---


@pytest.mark.asyncio
async def test_create_relationship_with_polarity(graph_store, seed_entities):
    """Creating a relationship stores polarity."""
    rel = Relationship(
        id="rel_pos", source_id="ent_alice", target_id="ent_python",
        predicate="USES", polarity="positive", group_id="default",
    )
    await graph_store.create_relationship(rel)

    rels = await graph_store.get_relationships("ent_alice", group_id="default")
    assert len(rels) == 1
    assert rels[0].polarity == "positive"


@pytest.mark.asyncio
async def test_create_negative_relationship(graph_store, seed_entities):
    """Negative polarity is stored and retrievable."""
    rel = Relationship(
        id="rel_neg", source_id="ent_alice", target_id="ent_python",
        predicate="STOPPED_USING", polarity="negative", group_id="default",
    )
    await graph_store.create_relationship(rel)

    rels = await graph_store.get_relationships("ent_alice", group_id="default")
    assert len(rels) == 1
    assert rels[0].polarity == "negative"


@pytest.mark.asyncio
async def test_create_uncertain_relationship(graph_store, seed_entities):
    """Uncertain polarity is stored and retrievable."""
    rel = Relationship(
        id="rel_unc", source_id="ent_alice", target_id="ent_python",
        predicate="CONSIDERING", polarity="uncertain", group_id="default",
    )
    await graph_store.create_relationship(rel)

    rels = await graph_store.get_relationships("ent_alice", group_id="default")
    assert len(rels) == 1
    assert rels[0].polarity == "uncertain"


# --- Backward compatibility ---


@pytest.mark.asyncio
async def test_existing_relationships_default_positive(graph_store, seed_entities):
    """Relationships without explicit polarity default to 'positive'."""
    # Insert directly without polarity column to simulate legacy data
    await graph_store.db.execute(
        """INSERT INTO relationships
           (id, source_id, target_id, predicate, weight,
            created_at, group_id, confidence)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("rel_legacy", "ent_alice", "ent_python", "USES", 1.0,
         datetime.utcnow().isoformat(), "default", 1.0),
    )
    await graph_store.db.commit()

    rels = await graph_store.get_relationships("ent_alice", group_id="default")
    assert len(rels) == 1
    assert rels[0].polarity == "positive"


# --- Spreading activation filtering ---


@pytest.mark.asyncio
async def test_negative_edges_excluded_from_neighbors(graph_store, seed_entities):
    """Negative-polarity edges are excluded from get_active_neighbors_with_weights."""
    # Create positive edge
    rel_pos = Relationship(
        id="rel_pos", source_id="ent_alice", target_id="ent_python",
        predicate="USES", polarity="positive", group_id="default",
    )
    await graph_store.create_relationship(rel_pos)

    # Create negative edge
    rel_neg = Relationship(
        id="rel_neg", source_id="ent_alice", target_id="ent_acme",
        predicate="WORKS_AT", polarity="negative", group_id="default",
    )
    await graph_store.create_relationship(rel_neg)

    neighbors = await graph_store.get_active_neighbors_with_weights(
        "ent_alice", group_id="default"
    )
    neighbor_ids = [n[0] for n in neighbors]
    assert "ent_python" in neighbor_ids
    assert "ent_acme" not in neighbor_ids


@pytest.mark.asyncio
async def test_uncertain_edges_weight_halved(graph_store, seed_entities):
    """Uncertain-polarity edges have weight halved in get_active_neighbors_with_weights."""
    rel = Relationship(
        id="rel_unc", source_id="ent_alice", target_id="ent_python",
        predicate="CONSIDERING", weight=1.0, polarity="uncertain",
        group_id="default",
    )
    await graph_store.create_relationship(rel)

    neighbors = await graph_store.get_active_neighbors_with_weights(
        "ent_alice", group_id="default"
    )
    assert len(neighbors) == 1
    assert neighbors[0][0] == "ent_python"
    assert neighbors[0][1] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_positive_edges_weight_unchanged(graph_store, seed_entities):
    """Positive-polarity edges keep their full weight."""
    rel = Relationship(
        id="rel_pos", source_id="ent_alice", target_id="ent_python",
        predicate="USES", weight=2.0, polarity="positive",
        group_id="default",
    )
    await graph_store.create_relationship(rel)

    neighbors = await graph_store.get_active_neighbors_with_weights(
        "ent_alice", group_id="default"
    )
    assert len(neighbors) == 1
    assert neighbors[0][1] == pytest.approx(2.0)


# --- Graph manager negation invalidation ---


@pytest.mark.asyncio
async def test_negation_invalidates_existing_edge(graph_store, seed_entities):
    """When a negative assertion arrives, existing positive edges are invalidated."""
    # First: create a positive "USES" relationship
    rel_pos = Relationship(
        id="rel_uses", source_id="ent_alice", target_id="ent_python",
        predicate="USES", polarity="positive", group_id="default",
    )
    await graph_store.create_relationship(rel_pos)

    # Verify it's active
    active = await graph_store.get_relationships(
        "ent_alice", direction="outgoing", predicate="USES",
        active_only=True, group_id="default",
    )
    assert len(active) == 1

    # Simulate what graph_manager does for negative polarity:
    # invalidate existing positive edges, then store the negative assertion
    dt_now = datetime.utcnow()
    existing_rels = await graph_store.get_relationships(
        "ent_alice", direction="outgoing", predicate="USES",
        active_only=True, group_id="default",
    )
    for existing_rel in existing_rels:
        if existing_rel.target_id == "ent_python":
            await graph_store.invalidate_relationship(
                existing_rel.id, dt_now, group_id="default",
            )

    rel_neg = Relationship(
        id="rel_stopped", source_id="ent_alice", target_id="ent_python",
        predicate="USES", polarity="negative", group_id="default",
    )
    await graph_store.create_relationship(rel_neg)

    # The positive edge should be invalidated
    active_after = await graph_store.get_relationships(
        "ent_alice", direction="outgoing", predicate="USES",
        active_only=True, group_id="default",
    )
    # Only the negative assertion remains active
    assert len(active_after) == 1
    assert active_after[0].polarity == "negative"

    # Negative edge excluded from spreading neighbors
    neighbors = await graph_store.get_active_neighbors_with_weights(
        "ent_alice", group_id="default"
    )
    python_neighbors = [n for n in neighbors if n[0] == "ent_python"]
    assert len(python_neighbors) == 0


# --- get_neighbors includes polarity ---


@pytest.mark.asyncio
async def test_get_neighbors_includes_polarity(graph_store, seed_entities):
    """get_neighbors returns Relationship objects with correct polarity."""
    rel = Relationship(
        id="rel_unc", source_id="ent_alice", target_id="ent_python",
        predicate="CONSIDERING", polarity="uncertain", group_id="default",
    )
    await graph_store.create_relationship(rel)

    neighbors = await graph_store.get_neighbors("ent_alice", hops=1, group_id="default")
    assert len(neighbors) >= 1
    found = [r for _, r in neighbors if r.id == "rel_unc"]
    assert len(found) == 1
    assert found[0].polarity == "uncertain"
