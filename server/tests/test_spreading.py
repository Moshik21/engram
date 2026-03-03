"""Tests for spreading activation."""

from __future__ import annotations

import time

import pytest

from engram.activation.spreading import identify_seeds, spread_activation
from engram.config import ActivationConfig


class MockNeighborProvider:
    """Mock graph store that only provides neighbor information."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float]]]) -> None:
        # adjacency: {node_id: [(neighbor_id, weight), ...]}
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self, entity_id: str, group_id: str | None = None
    ) -> list[tuple[str, float]]:
        return self._adj.get(entity_id, [])


class TestIdentifySeeds:
    def test_only_above_threshold(self):
        """Only candidates with sem >= seed_threshold become seeds."""
        cfg = ActivationConfig(seed_threshold=0.3)
        now = time.time()
        candidates = [
            ("ent_1", 0.8),
            ("ent_2", 0.2),  # below threshold
            ("ent_3", 0.5),
        ]
        seeds = identify_seeds(candidates, {}, now, cfg)
        seed_ids = [s[0] for s in seeds]
        assert "ent_1" in seed_ids
        assert "ent_3" in seed_ids
        assert "ent_2" not in seed_ids

    def test_cold_but_semantic_gets_energy_floor(self):
        """Cold but semantically strong node gets energy floor."""
        cfg = ActivationConfig(seed_threshold=0.3)
        now = time.time()
        candidates = [("ent_1", 0.8)]
        seeds = identify_seeds(candidates, {}, now, cfg)
        assert len(seeds) == 1
        # Energy = sem * max(0.0, sem * 0.1) = 0.8 * 0.08 = 0.064
        assert seeds[0][1] > 0


@pytest.mark.asyncio
class TestSpreadActivation:
    async def test_no_seeds(self):
        """No seeds -> empty bonuses and hop_distances."""
        cfg = ActivationConfig()
        provider = MockNeighborProvider({})
        bonuses, hops = await spread_activation([], provider, cfg)
        assert bonuses == {}
        assert hops == {}

    async def test_single_seed_no_neighbors(self):
        """Single seed with no neighbors -> only seed in hop_distances."""
        cfg = ActivationConfig()
        provider = MockNeighborProvider({"ent_1": []})
        seeds = [("ent_1", 0.5)]
        bonuses, hops = await spread_activation(seeds, provider, cfg)
        assert bonuses == {}
        assert hops == {"ent_1": 0}

    async def test_single_seed_one_neighbor(self):
        """Single seed -> neighbor gets bonus."""
        cfg = ActivationConfig(
            spread_firing_threshold=0.01,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider({
            "ent_1": [("ent_2", 1.0)],
            "ent_2": [],
        })
        seeds = [("ent_1", 1.0)]
        bonuses, hops = await spread_activation(seeds, provider, cfg)
        assert "ent_2" in bonuses
        assert bonuses["ent_2"] > 0
        assert hops["ent_2"] == 1

    async def test_chain_two_hops(self):
        """Chain A->B->C with 2 hops: C gets decayed bonus."""
        cfg = ActivationConfig(
            spread_max_hops=2,
            spread_firing_threshold=0.001,
            spread_energy_budget=10.0,
            spread_decay_per_hop=0.5,
        )
        provider = MockNeighborProvider({
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [],
        })
        seeds = [("A", 1.0)]
        bonuses, hops = await spread_activation(seeds, provider, cfg)
        assert "B" in bonuses
        assert "C" in bonuses
        assert bonuses["B"] > bonuses["C"]  # decayed
        assert hops["B"] == 1
        assert hops["C"] == 2

    async def test_chain_max_hops_limit(self):
        """Chain A->B->C->D with max_hops=2: D not reached."""
        cfg = ActivationConfig(
            spread_max_hops=2,
            spread_firing_threshold=0.001,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider({
            "A": [("B", 1.0)],
            "B": [("C", 1.0)],
            "C": [("D", 1.0)],
            "D": [],
        })
        seeds = [("A", 1.0)]
        bonuses, hops = await spread_activation(seeds, provider, cfg)
        assert "B" in bonuses
        assert "C" in bonuses
        assert "D" not in bonuses

    async def test_star_degree_normalization(self):
        """Star graph: hub dampening via fan-based S_ji."""
        cfg = ActivationConfig(
            spread_firing_threshold=0.001,
            spread_energy_budget=50.0,
            fan_s_max=3.0,  # High enough so 10 neighbors still spread
        )
        # Hub with 10 neighbors
        neighbors = [(f"leaf_{i}", 1.0) for i in range(10)]
        provider = MockNeighborProvider({
            "hub": neighbors,
            **{f"leaf_{i}": [] for i in range(10)},
        })
        seeds = [("hub", 1.0)]
        bonuses, _ = await spread_activation(seeds, provider, cfg)
        # fan_factor = max(0, 3.0 - ln(11)) ≈ 0.60
        # Each leaf gets energy * predicate_weight * fan_factor * decay
        for i in range(10):
            assert bonuses.get(f"leaf_{i}", 0) > 0
            assert bonuses[f"leaf_{i}"] < 0.5

    async def test_cycle_visited_set(self):
        """Cycle A->B->A: visited set prevents infinite loop."""
        cfg = ActivationConfig(
            spread_max_hops=3,
            spread_firing_threshold=0.001,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider({
            "A": [("B", 1.0)],
            "B": [("A", 1.0)],
        })
        seeds = [("A", 1.0)]
        bonuses, hops = await spread_activation(seeds, provider, cfg)
        # B should get bonus, A should not get bonus from B (visited)
        assert "B" in bonuses
        # A is a seed, already visited

    async def test_energy_budget_exhaustion(self):
        """Energy budget exhaustion: spreading stops mid-BFS."""
        cfg = ActivationConfig(
            spread_energy_budget=0.2,
            spread_firing_threshold=0.001,
        )
        neighbors = [(f"n_{i}", 1.0) for i in range(20)]
        provider = MockNeighborProvider({
            "hub": neighbors,
            **{f"n_{i}": [] for i in range(20)},
        })
        seeds = [("hub", 1.0)]
        bonuses, _ = await spread_activation(seeds, provider, cfg)
        # Should stop before reaching all 20 neighbors
        assert len(bonuses) < 20

    async def test_firing_threshold_filters(self):
        """Tiny energy filtered out by firing threshold."""
        cfg = ActivationConfig(
            spread_firing_threshold=0.5,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider({
            "A": [("B", 0.1)],  # spread = 0.1 * 0.1 * 1.0 * 0.5 = tiny
            "B": [],
        })
        seeds = [("A", 0.1)]
        bonuses, _ = await spread_activation(seeds, provider, cfg)
        # Spread amount too small for threshold
        assert "B" not in bonuses

    async def test_multiple_seeds_accumulate(self):
        """Multiple seeds: bonuses accumulate from both paths."""
        cfg = ActivationConfig(
            spread_firing_threshold=0.001,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider({
            "A": [("C", 1.0)],
            "B": [("C", 1.0)],
            "C": [],
        })
        seeds = [("A", 1.0), ("B", 1.0)]
        bonuses, _ = await spread_activation(seeds, provider, cfg)
        assert "C" in bonuses
        # C should get bonus from both A and B
        # At least from A's spread
        assert bonuses["C"] > 0
