"""Tests for Personalized PageRank spreading activation strategy."""

from __future__ import annotations

import pytest

from engram.activation.ppr import PPRStrategy
from engram.activation.strategy import create_strategy
from engram.config import ActivationConfig


class MockNeighborProvider:
    """Mock that returns 3-tuple (neighbor_id, weight, predicate)."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float, str]]]):
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float, str]]:
        return self._adj.get(entity_id, [])


class MockNeighborProvider2Tuple:
    """Mock that returns legacy 2-tuple."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float]]]):
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float]]:
        return self._adj.get(entity_id, [])


def _ppr_cfg(**overrides) -> ActivationConfig:
    """Build PPR config with sensible test defaults."""
    defaults = dict(
        spreading_strategy="ppr",
        ppr_alpha=0.15,
        ppr_max_iterations=50,
        ppr_epsilon=1e-8,
        ppr_expansion_hops=3,
        spread_energy_budget=100.0,
        spread_firing_threshold=0.0,
        spread_max_hops=2,
        spread_decay_per_hop=0.5,
        fan_s_max=1.5,  # keep low for PPR row-stochastic convergence
    )
    defaults.update(overrides)
    return ActivationConfig(**defaults)


@pytest.mark.asyncio
async def test_empty_seeds():
    """Empty seed list should return empty results."""
    ppr = PPRStrategy()
    provider = MockNeighborProvider({})
    cfg = _ppr_cfg()
    bonuses, hops = await ppr.spread([], provider, cfg)
    assert bonuses == {}
    assert hops == {}


@pytest.mark.asyncio
async def test_single_seed_no_neighbors():
    """A single seed with no neighbors: no bonuses, seed in hop_distances."""
    ppr = PPRStrategy()
    provider = MockNeighborProvider({"A": []})
    cfg = _ppr_cfg()
    bonuses, hops = await ppr.spread([("A", 1.0)], provider, cfg)
    assert bonuses == {}
    assert hops == {"A": 0}


@pytest.mark.asyncio
async def test_3hop_chain_reachable():
    """PPR can reach nodes 3 hops away (no hard BFS cutoff)."""
    provider = MockNeighborProvider({
        "A": [("B", 1.0, "KNOWS")],
        "B": [("A", 1.0, "KNOWS"), ("C", 1.0, "KNOWS")],
        "C": [("B", 1.0, "KNOWS"), ("D", 1.0, "KNOWS")],
        "D": [("C", 1.0, "KNOWS")],
    })
    cfg = _ppr_cfg(ppr_expansion_hops=4)
    ppr = PPRStrategy()
    bonuses, hops = await ppr.spread([("A", 1.0)], provider, cfg)
    # PPR should give some bonus to B, C, and D
    assert "B" in bonuses
    assert "C" in bonuses
    assert "D" in bonuses
    # Closer nodes get higher bonuses
    assert bonuses["B"] > bonuses["C"]
    assert bonuses["C"] > bonuses["D"]


@pytest.mark.asyncio
async def test_hub_dampening():
    """Hub nodes (high degree) shouldn't monopolize all activation."""
    # A -> hub (connects to B, C, D, E, F) and A -> leaf
    hub_neighbors = [
        ("B", 1.0, "KNOWS"), ("C", 1.0, "KNOWS"),
        ("D", 1.0, "KNOWS"), ("E", 1.0, "KNOWS"),
        ("F", 1.0, "KNOWS"), ("A", 1.0, "KNOWS"),
    ]
    provider = MockNeighborProvider({
        "A": [("hub", 1.0, "KNOWS"), ("leaf", 1.0, "KNOWS")],
        "hub": hub_neighbors,
        "leaf": [("A", 1.0, "KNOWS")],
    })
    cfg = _ppr_cfg()
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread([("A", 1.0)], provider, cfg)
    # Leaf should get a meaningful share despite hub having more edges
    assert bonuses.get("leaf", 0) > 0
    # Individual hub-connected nodes should each get less than leaf
    assert bonuses.get("B", 0) < bonuses.get("hub", 0)


@pytest.mark.asyncio
async def test_convergence_under_max_iterations():
    """PPR should converge well before max_iterations on a small graph."""
    provider = MockNeighborProvider({
        "A": [("B", 1.0, "KNOWS")],
        "B": [("A", 1.0, "KNOWS"), ("C", 1.0, "KNOWS")],
        "C": [("B", 1.0, "KNOWS")],
    })
    cfg = _ppr_cfg(ppr_max_iterations=100, ppr_epsilon=1e-10)
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread([("A", 1.0)], provider, cfg)
    # Should produce stable results
    assert bonuses.get("B", 0) > 0
    assert bonuses.get("C", 0) > 0


@pytest.mark.asyncio
async def test_cycle_stability():
    """PPR should handle cycles without diverging."""
    # A -> B -> C -> A (cycle)
    provider = MockNeighborProvider({
        "A": [("B", 1.0, "KNOWS")],
        "B": [("C", 1.0, "KNOWS")],
        "C": [("A", 1.0, "KNOWS")],
    })
    cfg = _ppr_cfg()
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread([("A", 1.0)], provider, cfg)
    # All non-seed nodes should have bounded bonuses
    for val in bonuses.values():
        assert 0 < val < 1.0


@pytest.mark.asyncio
async def test_dense_clique():
    """PPR on fully connected 4-node clique: non-seed nodes get similar scores."""
    nodes = ["A", "B", "C", "D"]
    adj: dict[str, list[tuple[str, float, str]]] = {}
    for n in nodes:
        adj[n] = [
            (m, 1.0, "KNOWS") for m in nodes if m != n
        ]
    provider = MockNeighborProvider(adj)
    cfg = _ppr_cfg()
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread([("A", 1.0)], provider, cfg)
    # B, C, D should all get similar scores
    vals = [bonuses.get(n, 0) for n in ["B", "C", "D"]]
    assert all(v > 0 for v in vals)
    assert max(vals) / min(vals) < 1.5  # roughly equal


@pytest.mark.asyncio
async def test_typed_edge_influence():
    """EXPERT_IN (0.9) edges should propagate more than MENTIONED_WITH (0.3)."""
    provider = MockNeighborProvider({
        "A": [
            ("B", 1.0, "EXPERT_IN"),
            ("C", 1.0, "MENTIONED_WITH"),
        ],
        "B": [("A", 1.0, "EXPERT_IN")],
        "C": [("A", 1.0, "MENTIONED_WITH")],
    })
    cfg = _ppr_cfg()
    ppr = PPRStrategy()
    bonuses, _ = await ppr.spread([("A", 1.0)], provider, cfg)
    assert bonuses.get("B", 0) > bonuses.get("C", 0)


@pytest.mark.asyncio
async def test_alpha_sensitivity():
    """Different alpha values produce different bonus distributions."""
    provider = MockNeighborProvider({
        "A": [("B", 1.0, "KNOWS")],
        "B": [("A", 1.0, "KNOWS"), ("C", 1.0, "KNOWS")],
        "C": [("B", 1.0, "KNOWS")],
    })
    ppr = PPRStrategy()

    cfg_low_alpha = _ppr_cfg(ppr_alpha=0.05)
    bonuses_low, _ = await ppr.spread(
        [("A", 1.0)], provider, cfg_low_alpha
    )

    cfg_high_alpha = _ppr_cfg(ppr_alpha=0.45)
    bonuses_high, _ = await ppr.spread(
        [("A", 1.0)], provider, cfg_high_alpha
    )

    # Both should produce bonuses for B and C
    assert bonuses_low.get("B", 0) > 0
    assert bonuses_high.get("B", 0) > 0
    assert bonuses_low.get("C", 0) > 0
    assert bonuses_high.get("C", 0) > 0
    # Different alpha values should produce different distributions
    assert bonuses_low.get("B", 0) != bonuses_high.get("B", 0)


@pytest.mark.asyncio
async def test_backward_compat_2tuple():
    """PPR should work with legacy 2-tuple neighbor providers."""
    provider = MockNeighborProvider2Tuple({
        "A": [("B", 0.8)],
        "B": [("A", 0.8)],
    })
    cfg = _ppr_cfg()
    ppr = PPRStrategy()
    bonuses, hops = await ppr.spread([("A", 1.0)], provider, cfg)
    assert bonuses.get("B", 0) > 0
    assert hops["B"] == 1


@pytest.mark.asyncio
async def test_ppr_vs_bfs_same_graph():
    """PPR and BFS should both produce bonuses on the same graph."""
    from engram.activation.bfs import BFSStrategy

    provider = MockNeighborProvider({
        "A": [("B", 1.0, "KNOWS"), ("C", 1.0, "KNOWS")],
        "B": [("A", 1.0, "KNOWS"), ("D", 1.0, "KNOWS")],
        "C": [("A", 1.0, "KNOWS")],
        "D": [("B", 1.0, "KNOWS")],
    })
    seeds = [("A", 1.0)]

    bfs_cfg = ActivationConfig(
        spreading_strategy="bfs",
        spread_max_hops=3,
        spread_decay_per_hop=0.5,
        spread_firing_threshold=0.0,
        spread_energy_budget=100.0,
    )
    bfs = BFSStrategy()
    bfs_bonuses, _ = await bfs.spread(seeds, provider, bfs_cfg)

    ppr_cfg = _ppr_cfg()
    ppr = PPRStrategy()
    ppr_bonuses, _ = await ppr.spread(seeds, provider, ppr_cfg)

    # Both should discover B, C, D
    for nid in ["B", "C", "D"]:
        assert bfs_bonuses.get(nid, 0) > 0, f"BFS missing {nid}"
        assert ppr_bonuses.get(nid, 0) > 0, f"PPR missing {nid}"


def test_strategy_factory():
    """create_strategy should return correct strategy types."""
    bfs = create_strategy("bfs")
    ppr = create_strategy("ppr")
    assert type(bfs).__name__ == "BFSStrategy"
    assert type(ppr).__name__ == "PPRStrategy"

    with pytest.raises(ValueError, match="Unknown spreading strategy"):
        create_strategy("invalid")
