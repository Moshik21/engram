"""Tests for typed edge weighting in spreading activation."""

from __future__ import annotations

import pytest

from engram.activation.spreading import spread_activation
from engram.config import ActivationConfig


class MockNeighborProvider2Tuple:
    """Mock that returns legacy 2-tuple (neighbor_id, weight)."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float]]]):
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float]]:
        return self._adj.get(entity_id, [])


class MockNeighborProvider3Tuple:
    """Mock that returns 3-tuple (neighbor_id, weight, predicate)."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float, str]]]):
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str]]:
        return self._adj.get(entity_id, [])


@pytest.mark.asyncio
async def test_works_at_stronger_than_mentioned_with():
    """WORKS_AT (0.8) should propagate more than MENTIONED_WITH (0.3)."""
    # A -> B via WORKS_AT, A -> C via MENTIONED_WITH
    provider = MockNeighborProvider3Tuple(
        {
            "A": [
                ("B", 1.0, "WORKS_AT"),
                ("C", 1.0, "MENTIONED_WITH"),
            ],
        }
    )
    cfg = ActivationConfig(
        spread_max_hops=1,
        spread_decay_per_hop=1.0,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
    )
    seeds = [("A", 1.0)]
    bonuses, _ = await spread_activation(seeds, provider, cfg)
    assert bonuses.get("B", 0) > bonuses.get("C", 0)
    # WORKS_AT=0.8, MENTIONED_WITH=0.3
    assert bonuses["B"] / bonuses["C"] == pytest.approx(0.8 / 0.3, rel=0.01)


@pytest.mark.asyncio
async def test_unknown_predicate_gets_default():
    """A predicate not in the weight map should use predicate_weight_default."""
    provider = MockNeighborProvider3Tuple(
        {
            "A": [("B", 1.0, "SOME_UNKNOWN_PREDICATE")],
        }
    )
    cfg = ActivationConfig(
        spread_max_hops=1,
        spread_decay_per_hop=1.0,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
        predicate_weight_default=0.42,
    )
    seeds = [("A", 1.0)]
    bonuses, _ = await spread_activation(seeds, provider, cfg)
    # fan_factor = max(0, 1.5 - ln(2)) ≈ 0.807
    # spread = 1.0 * 1.0 * 0.42 * 0.807 * 1.0 ≈ 0.339
    import math

    fan_factor = max(0.0, cfg.fan_s_max - math.log(2))
    assert bonuses["B"] == pytest.approx(0.42 * fan_factor, rel=0.01)


@pytest.mark.asyncio
async def test_backward_compat_2tuple():
    """Legacy 2-tuple neighbor providers should still work."""
    provider = MockNeighborProvider2Tuple(
        {
            "A": [("B", 0.8)],
        }
    )
    cfg = ActivationConfig(
        spread_max_hops=1,
        spread_decay_per_hop=1.0,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
        predicate_weight_default=0.5,
    )
    seeds = [("A", 1.0)]
    bonuses, _ = await spread_activation(seeds, provider, cfg)
    # fan_factor = max(0, 1.5 - ln(2)) ≈ 0.807
    # spread = 1.0 * 0.8 * 0.5 * 0.807 * 1.0 ≈ 0.323
    import math

    fan_factor = max(0.0, cfg.fan_s_max - math.log(2))
    assert bonuses["B"] == pytest.approx(0.8 * 0.5 * fan_factor, rel=0.01)


@pytest.mark.asyncio
async def test_config_override_predicate_weights():
    """Custom predicate_weights in config should override defaults."""
    provider = MockNeighborProvider3Tuple(
        {
            "A": [("B", 1.0, "WORKS_AT")],
        }
    )
    cfg = ActivationConfig(
        spread_max_hops=1,
        spread_decay_per_hop=1.0,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
        predicate_weights={"WORKS_AT": 0.1},
    )
    seeds = [("A", 1.0)]
    bonuses, _ = await spread_activation(seeds, provider, cfg)
    # fan_factor = max(0, 1.5 - ln(2)) ≈ 0.807
    # spread = 1.0 * 1.0 * 0.1 * 0.807 * 1.0 ≈ 0.081
    import math

    fan_factor = max(0.0, cfg.fan_s_max - math.log(2))
    assert bonuses["B"] == pytest.approx(0.1 * fan_factor, rel=0.01)


@pytest.mark.asyncio
async def test_zero_weight_blocks_propagation():
    """A predicate with weight 0.0 should block spreading."""
    provider = MockNeighborProvider3Tuple(
        {
            "A": [("B", 1.0, "BLOCKED_EDGE")],
        }
    )
    cfg = ActivationConfig(
        spread_max_hops=2,
        spread_decay_per_hop=1.0,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
        predicate_weights={"BLOCKED_EDGE": 0.0},
    )
    seeds = [("A", 1.0)]
    bonuses, _ = await spread_activation(seeds, provider, cfg)
    # Zero predicate weight → spread_amount = 0 → below firing threshold
    assert bonuses.get("B", 0.0) == 0.0


@pytest.mark.asyncio
async def test_sqlite_integration_returns_3tuple(tmp_path):
    """SQLiteGraphStore.get_active_neighbors_with_weights returns 3-tuple."""
    from engram.models.entity import Entity
    from engram.models.relationship import Relationship
    from engram.storage.sqlite.graph import SQLiteGraphStore

    db_path = str(tmp_path / "test.db")
    store = SQLiteGraphStore(db_path)
    await store.initialize()

    # Create two entities and a relationship
    e1 = Entity(id="e1", name="Alice", entity_type="person", group_id="test")
    e2 = Entity(id="e2", name="Acme", entity_type="org", group_id="test")
    await store.create_entity(e1)
    await store.create_entity(e2)

    rel = Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        predicate="WORKS_AT",
        weight=0.9,
        group_id="test",
    )
    await store.create_relationship(rel)

    neighbors = await store.get_active_neighbors_with_weights("e1", group_id="test")
    assert len(neighbors) == 1
    assert len(neighbors[0]) == 4
    neighbor_id, weight, predicate, entity_type = neighbors[0]
    assert neighbor_id == "e2"
    assert weight == pytest.approx(0.9)
    assert predicate == "WORKS_AT"

    await store.close()
