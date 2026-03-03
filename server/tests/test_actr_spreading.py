"""Tests for ACT-R spreading activation strategy."""

from __future__ import annotations

import math
import time

import pytest

from engram.activation.actr import ACTRStrategy
from engram.activation.spreading import identify_actr_seeds, spread_activation
from engram.activation.strategy import create_strategy
from engram.config import ActivationConfig
from engram.retrieval.working_memory import WorkingMemoryBuffer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    """Config accepts 'actr' as a spreading strategy."""

    def test_actr_strategy_accepted(self):
        cfg = ActivationConfig(spreading_strategy="actr")
        assert cfg.spreading_strategy == "actr"

    def test_actr_total_w_default(self):
        cfg = ActivationConfig()
        assert cfg.actr_total_w == 1.0

    def test_actr_max_sources_default(self):
        cfg = ActivationConfig()
        assert cfg.actr_max_sources == 7

    def test_actr_total_w_bounds(self):
        cfg = ActivationConfig(actr_total_w=2.5)
        assert cfg.actr_total_w == 2.5
        with pytest.raises(Exception):
            ActivationConfig(actr_total_w=0.0)
        with pytest.raises(Exception):
            ActivationConfig(actr_total_w=4.0)

    def test_actr_max_sources_bounds(self):
        cfg = ActivationConfig(actr_max_sources=1)
        assert cfg.actr_max_sources == 1
        with pytest.raises(Exception):
            ActivationConfig(actr_max_sources=0)
        with pytest.raises(Exception):
            ActivationConfig(actr_max_sources=25)


# ---------------------------------------------------------------------------
# Strategy factory
# ---------------------------------------------------------------------------


class TestStrategyFactory:
    """create_strategy('actr') returns ACTRStrategy."""

    def test_creates_actr(self):
        strategy = create_strategy("actr")
        assert isinstance(strategy, ACTRStrategy)

    def test_bfs_still_works(self):
        from engram.activation.bfs import BFSStrategy

        assert isinstance(create_strategy("bfs"), BFSStrategy)

    def test_ppr_still_works(self):
        from engram.activation.ppr import PPRStrategy

        assert isinstance(create_strategy("ppr"), PPRStrategy)


# ---------------------------------------------------------------------------
# identify_actr_seeds
# ---------------------------------------------------------------------------


class TestIdentifyACTRSeeds:
    """identify_actr_seeds filters WM to entities, caps, and sorts."""

    def test_empty_wm(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        cfg = ActivationConfig()
        now = time.time()
        seeds = identify_actr_seeds(wm, now, cfg)
        assert seeds == []

    def test_entity_only_filter(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        wm.add("ent_1", "entity", 0.9, "q1", now)
        wm.add("ep_1", "episode", 0.8, "q1", now)
        wm.add("ent_2", "entity", 0.7, "q1", now)

        cfg = ActivationConfig()
        seeds = identify_actr_seeds(wm, now, cfg)
        seed_ids = [s[0] for s in seeds]
        assert "ent_1" in seed_ids
        assert "ent_2" in seed_ids
        assert "ep_1" not in seed_ids

    def test_max_sources_cap(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        for i in range(15):
            wm.add(f"ent_{i}", "entity", 0.5, "q", now - i)

        cfg = ActivationConfig(actr_max_sources=5)
        seeds = identify_actr_seeds(wm, now, cfg)
        assert len(seeds) == 5

    def test_recency_ordering(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        wm.add("old", "entity", 0.5, "q", now - 100)
        wm.add("new", "entity", 0.5, "q", now - 1)
        wm.add("mid", "entity", 0.5, "q", now - 50)

        cfg = ActivationConfig(actr_max_sources=7)
        seeds = identify_actr_seeds(wm, now, cfg)
        seed_ids = [s[0] for s in seeds]
        assert seed_ids[0] == "new"
        assert seed_ids[1] == "mid"
        assert seed_ids[2] == "old"

    def test_energy_is_placeholder(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        wm.add("ent_1", "entity", 0.9, "q", now)
        cfg = ActivationConfig()
        seeds = identify_actr_seeds(wm, now, cfg)
        assert seeds[0][1] == 1.0

    def test_expired_entries_excluded(self):
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=10.0)
        now = time.time()
        wm.add("expired", "entity", 0.9, "q", now - 20)
        wm.add("fresh", "entity", 0.9, "q", now)
        cfg = ActivationConfig()
        seeds = identify_actr_seeds(wm, now, cfg)
        assert len(seeds) == 1
        assert seeds[0][0] == "fresh"


# ---------------------------------------------------------------------------
# ACTRStrategy.spread
# ---------------------------------------------------------------------------


class MockNeighborProvider:
    """Mock graph store for testing neighbor retrieval."""

    def __init__(self, adjacency: dict[str, list[tuple[str, float, str]]]):
        self._adjacency = adjacency

    async def get_active_neighbors_with_weights(
        self,
        node_id: str,
        group_id: str | None = None,
    ) -> list[tuple[str, float, str]]:
        return self._adjacency.get(node_id, [])


class TestACTRStrategy:
    """ACTRStrategy spread behavior."""

    @pytest.mark.asyncio
    async def test_no_seeds(self):
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr")
        provider = MockNeighborProvider({})
        bonuses, hops = await strategy.spread([], provider, cfg)
        assert bonuses == {}
        assert hops == {}

    @pytest.mark.asyncio
    async def test_single_source(self):
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        provider = MockNeighborProvider(
            {
                "A": [("B", 1.0, "KNOWS")],
            }
        )
        bonuses, hops = await strategy.spread(
            [("A", 1.0)],
            provider,
            cfg,
        )
        # W_j = 1.0/1 = 1.0
        # fan = 1, S_ji = max(0.3, 3.5 - ln(2)) = max(0.3, 2.807) = 2.807
        # predicate_weight for KNOWS = 0.5
        # bonus = 1.0 * 2.807 * 0.5 * 1.0 = 1.4035
        assert "B" in bonuses
        expected = 1.0 * (3.5 - math.log(2)) * 0.5 * 1.0
        assert abs(bonuses["B"] - expected) < 1e-6
        assert hops["B"] == 1
        assert hops["A"] == 0

    @pytest.mark.asyncio
    async def test_multi_source_accumulation(self):
        """Entity connected to multiple WM items accumulates bonuses."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        provider = MockNeighborProvider(
            {
                "A": [("C", 1.0, "KNOWS")],
                "B": [("C", 1.0, "KNOWS")],
            }
        )
        bonuses, hops = await strategy.spread(
            [("A", 1.0), ("B", 1.0)],
            provider,
            cfg,
        )
        # W_j = 1.0/2 = 0.5
        # Each source contributes: 0.5 * (3.5-ln(2)) * 0.5 * 1.0
        single = 0.5 * (3.5 - math.log(2)) * 0.5 * 1.0
        assert abs(bonuses["C"] - 2 * single) < 1e-6

    @pytest.mark.asyncio
    async def test_fan_dampening(self):
        """High-fan nodes get lower S_ji."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        # Node with 10 neighbors
        neighbors = [(f"n{i}", 1.0, "KNOWS") for i in range(10)]
        provider = MockNeighborProvider({"A": neighbors})
        bonuses, _ = await strategy.spread(
            [("A", 1.0)],
            provider,
            cfg,
        )
        # fan = 10, S_ji = max(0.3, 3.5 - ln(11)) = max(0.3, 1.103)
        s_ji = max(0.3, 3.5 - math.log(11))
        expected_each = 1.0 * s_ji * 0.5 * 1.0
        for i in range(10):
            assert abs(bonuses[f"n{i}"] - expected_each) < 1e-6

    @pytest.mark.asyncio
    async def test_one_hop_only(self):
        """ACT-R does NOT do multi-hop spreading."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr")
        provider = MockNeighborProvider(
            {
                "A": [("B", 1.0, "KNOWS")],
                "B": [("C", 1.0, "KNOWS")],
            }
        )
        bonuses, hops = await strategy.spread(
            [("A", 1.0)],
            provider,
            cfg,
        )
        assert "B" in bonuses
        assert "C" not in bonuses  # no 2nd hop

    @pytest.mark.asyncio
    async def test_meaningful_magnitude(self):
        """Bonuses should be in a meaningful range (>> 0.034 BFS median)."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(
            spreading_strategy="actr",
            actr_total_w=1.0,
        )
        # 5 sources, target connected to all 5, each source has degree 5
        adjacency = {}
        for i in range(5):
            neighbors = [(f"other_{i}_{j}", 1.0, "EXPERT_IN") for j in range(4)]
            neighbors.append(("target", 1.0, "EXPERT_IN"))
            adjacency[f"src_{i}"] = neighbors
        provider = MockNeighborProvider(adjacency)
        seeds = [(f"src_{i}", 1.0) for i in range(5)]
        bonuses, _ = await strategy.spread(seeds, provider, cfg)
        # W_j = 1.0/5 = 0.2
        # fan = 5, S_ji = max(0.3, 3.5 - ln(6)) ≈ 1.71
        # pred_weight for EXPERT_IN = 0.9
        # Per source: 0.2 * 1.71 * 0.9 * 1.0 = 0.308
        # Total: 5 * 0.308 = 1.539
        assert bonuses["target"] > 0.5  # much higher than BFS median 0.034

    @pytest.mark.asyncio
    async def test_predicate_weight_applied(self):
        """Different predicates produce different bonuses."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        provider = MockNeighborProvider(
            {
                "A": [("B", 1.0, "EXPERT_IN"), ("C", 1.0, "MENTIONED_WITH")],
            }
        )
        bonuses, _ = await strategy.spread([("A", 1.0)], provider, cfg)
        # EXPERT_IN weight = 0.9, MENTIONED_WITH weight = 0.3
        assert bonuses["B"] > bonuses["C"]
        ratio = bonuses["B"] / bonuses["C"]
        assert abs(ratio - 3.0) < 0.01  # 0.9/0.3 = 3.0

    @pytest.mark.asyncio
    async def test_edge_weight_scales_bonus(self):
        """Edge weight multiplies into the bonus."""
        strategy = ACTRStrategy()
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        provider = MockNeighborProvider(
            {
                "A": [("B", 2.0, "KNOWS"), ("C", 0.5, "KNOWS")],
            }
        )
        bonuses, _ = await strategy.spread([("A", 1.0)], provider, cfg)
        assert bonuses["B"] / bonuses["C"] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Via spread_activation dispatcher
# ---------------------------------------------------------------------------


class TestSpreadActivationDispatcher:
    """spread_activation dispatches to ACTRStrategy when configured."""

    @pytest.mark.asyncio
    async def test_dispatches_to_actr(self):
        cfg = ActivationConfig(spreading_strategy="actr", actr_total_w=1.0)
        provider = MockNeighborProvider(
            {
                "A": [("B", 1.0, "KNOWS")],
            }
        )
        bonuses, hops = await spread_activation(
            [("A", 1.0)],
            provider,
            cfg,
        )
        assert "B" in bonuses
        assert hops["B"] == 1


# ---------------------------------------------------------------------------
# Pipeline integration (lightweight)
# ---------------------------------------------------------------------------


class TestPipelineACTRSeeds:
    """ACT-R path in pipeline uses WM seeds, not search seeds."""

    def test_actr_seeds_from_wm(self):
        """Verify identify_actr_seeds uses WM items as seeds."""
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        now = time.time()
        wm.add("ent_A", "entity", 0.9, "q", now)
        wm.add("ent_B", "entity", 0.8, "q", now - 5)
        cfg = ActivationConfig(spreading_strategy="actr")
        seeds = identify_actr_seeds(wm, now, cfg)
        assert len(seeds) == 2
        assert seeds[0][0] == "ent_A"  # most recent first

    def test_empty_wm_produces_no_seeds(self):
        """No WM = no spreading in ACT-R mode."""
        wm = WorkingMemoryBuffer(capacity=20, ttl_seconds=300.0)
        cfg = ActivationConfig(spreading_strategy="actr")
        seeds = identify_actr_seeds(wm, time.time(), cfg)
        assert seeds == []
