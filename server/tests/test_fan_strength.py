"""Tests for fan-based associative strength (S_ji) in BFS and PPR."""

import pytest

from engram.activation.bfs import BFSStrategy
from engram.activation.ppr import PPRStrategy
from engram.config import ActivationConfig


class MockNeighborProvider:
    """Mock provider that returns configurable neighbors."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float, str]]]):
        self._adj = adjacency

    async def get_active_neighbors_with_weights(
        self,
        node_id: str,
        group_id: str | None = None,
    ) -> list[tuple[str, float, str]]:
        return self._adj.get(node_id, [])


class TestFanStrengthBFS:
    @pytest.mark.asyncio
    async def test_fan_1_gives_near_s_max(self):
        """A node with 1 neighbor should have fan_factor close to S_max."""
        cfg = ActivationConfig(
            fan_s_max=1.5,
            spread_max_hops=1,
            spread_energy_budget=10.0,
        )
        provider = MockNeighborProvider(
            {
                "seed": [("n1", 1.0, "KNOWS")],
            }
        )
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        assert "n1" in bonuses
        assert bonuses["n1"] > 0

    @pytest.mark.asyncio
    async def test_high_fan_hub_dampening(self):
        """A hub node with many neighbors should spread less per neighbor."""
        cfg = ActivationConfig(
            fan_s_max=1.5,
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )
        provider_low = MockNeighborProvider(
            {
                "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
            }
        )
        high_neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(20)]
        provider_high = MockNeighborProvider({"seed": high_neighbors})

        strategy = BFSStrategy()
        bonuses_low, _ = await strategy.spread(
            [("seed", 1.0)],
            provider_low,
            cfg,
        )
        bonuses_high, _ = await strategy.spread(
            [("seed", 1.0)],
            provider_high,
            cfg,
        )

        avg_low = sum(bonuses_low.values()) / max(len(bonuses_low), 1)
        avg_high = sum(bonuses_high.values()) / max(len(bonuses_high), 1)
        assert avg_low > avg_high

    @pytest.mark.asyncio
    async def test_fan_100_gives_near_zero(self):
        """A node with 100 neighbors: fan_factor clamped to fan_s_min=0."""
        cfg = ActivationConfig(
            fan_s_max=1.5,
            fan_s_min=0.0,
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )
        neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(100)]
        provider = MockNeighborProvider({"seed": neighbors})
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        assert len(bonuses) == 0

    @pytest.mark.asyncio
    async def test_fan_s_min_floor(self):
        """fan_s_min ensures high-degree nodes still propagate energy."""
        cfg = ActivationConfig(
            fan_s_max=1.0,
            fan_s_min=0.5,
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )
        # Degree 50: ln(51) ≈ 3.93, so fan_s_max - ln(51) < 0, but floor is 0.5
        neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(50)]
        provider = MockNeighborProvider({"seed": neighbors})
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        assert len(bonuses) > 0
        assert all(v > 0 for v in bonuses.values())

    @pytest.mark.asyncio
    async def test_config_s_max_override(self):
        """Higher S_max allows more spread from high-fan nodes."""
        neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(10)]
        provider = MockNeighborProvider({"seed": neighbors})

        cfg_low = ActivationConfig(
            fan_s_max=1.0,
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )
        cfg_high = ActivationConfig(
            fan_s_max=3.0,
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )

        strategy = BFSStrategy()
        bonuses_low, _ = await strategy.spread(
            [("seed", 1.0)],
            provider,
            cfg_low,
        )
        bonuses_high, _ = await strategy.spread(
            [("seed", 1.0)],
            provider,
            cfg_high,
        )
        total_low = sum(bonuses_low.values())
        total_high = sum(bonuses_high.values())
        assert total_high > total_low

    @pytest.mark.asyncio
    async def test_isolated_node(self):
        """A seed with no neighbors produces no bonuses."""
        cfg = ActivationConfig(fan_s_max=1.5)
        provider = MockNeighborProvider({"seed": []})
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        assert bonuses == {}


class TestFanStrengthPPR:
    @pytest.mark.asyncio
    async def test_ppr_with_fan_attenuation(self):
        """PPR produces bonuses with fan-based weights."""
        cfg = ActivationConfig(
            spreading_strategy="ppr",
            fan_s_max=1.5,
            ppr_expansion_hops=1,
            ppr_max_iterations=20,
        )
        provider = MockNeighborProvider(
            {
                "seed": [("n1", 1.0, "KNOWS"), ("n2", 1.0, "KNOWS")],
                "n1": [("n3", 1.0, "WORKS_AT")],
                "n2": [],
            }
        )
        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        assert len(bonuses) > 0

    @pytest.mark.asyncio
    async def test_ppr_high_fan_reduces_spread(self):
        """PPR applies fan dampening to hub nodes."""
        provider_low = MockNeighborProvider(
            {
                "seed": [("n1", 1.0, "KNOWS")],
                "n1": [("n2", 1.0, "WORKS_AT")],
                "n2": [],
            }
        )
        hub_neighbors = [(f"h{i}", 1.0, "KNOWS") for i in range(15)]
        provider_high_adj: dict[str, list[tuple[str, float, str]]] = {
            "seed": hub_neighbors,
        }
        for i in range(15):
            provider_high_adj[f"h{i}"] = []
        provider_high = MockNeighborProvider(provider_high_adj)

        cfg = ActivationConfig(
            spreading_strategy="ppr",
            fan_s_max=1.5,
            ppr_expansion_hops=1,
            ppr_max_iterations=20,
        )
        strategy = PPRStrategy()
        bonuses_low, _ = await strategy.spread(
            [("seed", 1.0)],
            provider_low,
            cfg,
        )
        bonuses_high, _ = await strategy.spread(
            [("seed", 1.0)],
            provider_high,
            cfg,
        )

        if bonuses_low and bonuses_high:
            avg_low = sum(bonuses_low.values()) / len(bonuses_low)
            avg_high = sum(bonuses_high.values()) / len(bonuses_high)
            assert avg_low > avg_high

    @pytest.mark.asyncio
    async def test_ppr_empty_seeds(self):
        """PPR with no seeds returns empty."""
        cfg = ActivationConfig(spreading_strategy="ppr", fan_s_max=1.5)
        provider = MockNeighborProvider({})
        strategy = PPRStrategy()
        bonuses, hops = await strategy.spread([], provider, cfg)
        assert bonuses == {}
        assert hops == {}


class TestDefaultFanConfig:
    @pytest.mark.asyncio
    async def test_default_fan_allows_medium_degree(self):
        """Default fan_s_max=3.5 allows spreading from degree-9 nodes."""
        cfg = ActivationConfig(
            spread_max_hops=1,
            spread_energy_budget=100.0,
            spread_firing_threshold=0.001,
        )
        # Default fan_s_max should be 3.5
        assert cfg.fan_s_max == 3.5
        # Degree-9 node: fan_factor = max(0, 3.5 - ln(10)) ≈ 3.5 - 2.30 = 1.20
        neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(9)]
        provider = MockNeighborProvider({"seed": neighbors})
        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread([("seed", 1.0)], provider, cfg)
        # Should spread to all 9 neighbors (fan_factor > 0)
        assert len(bonuses) == 9
        assert all(v > 0 for v in bonuses.values())
