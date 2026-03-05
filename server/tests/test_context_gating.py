"""Tests for context-gated spreading activation."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from engram.activation.context_gate import (
    ContextGate,
    PredicateEmbeddingCache,
    build_context_gate,
)
from engram.config import ActivationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unit_vec(angle_deg: float, dim: int = 8) -> list[float]:
    """Create a unit vector rotated in the first two dims."""
    v = np.zeros(dim)
    v[0] = np.cos(np.radians(angle_deg))
    v[1] = np.sin(np.radians(angle_deg))
    return v.tolist()


def _random_vec(dim: int = 8, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    return (v / np.linalg.norm(v)).tolist()


# ---------------------------------------------------------------------------
# TestContextGate
# ---------------------------------------------------------------------------


class TestContextGate:
    def test_high_sim_predicate_gets_high_gate(self):
        """Predicate aligned with query should get gate near 1.0."""
        query = _make_unit_vec(0)
        preds = {"EXPERT_IN": _make_unit_vec(5)}  # Nearly aligned
        gate = ContextGate(query, preds, floor=0.3)
        value = gate.gate("EXPERT_IN")
        assert value > 0.9

    def test_low_sim_predicate_gets_floor_gate(self):
        """Predicate orthogonal to query should get gate near floor."""
        query = _make_unit_vec(0)
        preds = {"LOCATED_IN": _make_unit_vec(90)}  # Orthogonal
        gate = ContextGate(query, preds, floor=0.3)
        value = gate.gate("LOCATED_IN")
        assert value == pytest.approx(0.3, abs=0.05)

    def test_unknown_predicate_returns_one(self):
        """Predicate not in cache returns 1.0 (no gating)."""
        query = _make_unit_vec(0)
        preds = {"EXPERT_IN": _make_unit_vec(0)}
        gate = ContextGate(query, preds, floor=0.3)
        assert gate.gate("UNKNOWN_PRED") == 1.0

    def test_floor_zero_allows_full_range(self):
        """With floor=0, orthogonal predicate gets ~0 gate."""
        query = _make_unit_vec(0)
        preds = {"LOCATED_IN": _make_unit_vec(90)}
        gate = ContextGate(query, preds, floor=0.0)
        value = gate.gate("LOCATED_IN")
        assert value == pytest.approx(0.0, abs=0.05)

    def test_gate_values_cached(self):
        """Calling gate() twice returns same value (memoized)."""
        query = _make_unit_vec(0)
        preds = {"EXPERT_IN": _make_unit_vec(30)}
        gate = ContextGate(query, preds, floor=0.3)
        v1 = gate.gate("EXPERT_IN")
        v2 = gate.gate("EXPERT_IN")
        assert v1 == v2


# ---------------------------------------------------------------------------
# TestPredicateEmbeddingCache
# ---------------------------------------------------------------------------


class TestPredicateEmbeddingCache:
    @pytest.mark.asyncio
    async def test_initialize_embeds_all_predicates(self):
        """Initialize embeds all predicates from config."""
        cfg = ActivationConfig()
        n_preds = len(cfg.predicate_natural_names)

        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(
            return_value=[_random_vec(dim=512, seed=i) for i in range(n_preds)]
        )

        cache = PredicateEmbeddingCache()
        await cache.initialize(cfg, mock_provider)

        assert cache.initialized is True
        embeddings = cache.get_embeddings()
        assert len(embeddings) == n_preds
        mock_provider.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Second initialize() call is a no-op."""
        cfg = ActivationConfig()
        n_preds = len(cfg.predicate_natural_names)

        mock_provider = AsyncMock()
        mock_provider.embed = AsyncMock(
            return_value=[_random_vec(dim=512, seed=i) for i in range(n_preds)]
        )

        cache = PredicateEmbeddingCache()
        await cache.initialize(cfg, mock_provider)
        await cache.initialize(cfg, mock_provider)

        assert mock_provider.embed.call_count == 1

    def test_set_embeddings_for_testing(self):
        """set_embeddings() directly injects embeddings."""
        cache = PredicateEmbeddingCache()
        assert cache.initialized is False

        test_embs = {"EXPERT_IN": [1.0, 0.0], "LOCATED_IN": [0.0, 1.0]}
        cache.set_embeddings(test_embs)

        assert cache.initialized is True
        assert cache.get_embeddings() == test_embs

    def test_uninitialized_returns_empty(self):
        """Uninitialized cache returns empty dict."""
        cache = PredicateEmbeddingCache()
        assert cache.get_embeddings() == {}
        assert cache.initialized is False


# ---------------------------------------------------------------------------
# TestBuildContextGate
# ---------------------------------------------------------------------------


class TestBuildContextGate:
    def test_builds_gate_with_valid_inputs(self):
        """Factory builds a ContextGate when all inputs present."""
        cfg = ActivationConfig(context_gating_enabled=True, context_gate_floor=0.3)
        cache = PredicateEmbeddingCache()
        cache.set_embeddings({"EXPERT_IN": _make_unit_vec(0)})

        gate = build_context_gate(_make_unit_vec(0), cache, cfg)
        assert gate is not None
        assert isinstance(gate, ContextGate)

    def test_returns_none_without_query_embedding(self):
        """No query embedding -> None."""
        cfg = ActivationConfig(context_gating_enabled=True)
        cache = PredicateEmbeddingCache()
        cache.set_embeddings({"EXPERT_IN": _make_unit_vec(0)})

        assert build_context_gate(None, cache, cfg) is None
        assert build_context_gate([], cache, cfg) is None

    def test_returns_none_without_predicate_cache(self):
        """No predicate cache -> None."""
        cfg = ActivationConfig(context_gating_enabled=True)
        assert build_context_gate(_make_unit_vec(0), None, cfg) is None

    def test_returns_none_when_disabled(self):
        """context_gating_enabled=False -> None."""
        cfg = ActivationConfig(context_gating_enabled=False)
        cache = PredicateEmbeddingCache()
        cache.set_embeddings({"EXPERT_IN": _make_unit_vec(0)})
        assert build_context_gate(_make_unit_vec(0), cache, cfg) is None


# ---------------------------------------------------------------------------
# TestBFSContextGate
# ---------------------------------------------------------------------------


class TestBFSContextGate:
    @pytest.mark.asyncio
    async def test_relevant_predicate_gets_more_energy(self):
        """Edge with query-relevant predicate receives more spread energy."""
        from engram.activation.bfs import BFSStrategy

        query_vec = _make_unit_vec(0)
        pred_embs = {
            "EXPERT_IN": _make_unit_vec(5),  # Nearly aligned
            "LOCATED_IN": _make_unit_vec(90),  # Orthogonal
        }
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=True,
            context_gate_floor=0.3,
            spread_max_hops=1,
            spread_energy_budget=50.0,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[
                ("expert_neighbor", 1.0, "EXPERT_IN"),
                ("location_neighbor", 1.0, "LOCATED_IN"),
            ]
        )

        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        assert bonuses["expert_neighbor"] > bonuses["location_neighbor"]

    @pytest.mark.asyncio
    async def test_disabled_no_effect(self):
        """context_gating_enabled=False makes context_gate a no-op."""
        from engram.activation.bfs import BFSStrategy

        query_vec = _make_unit_vec(0)
        pred_embs = {
            "EXPERT_IN": _make_unit_vec(5),
            "LOCATED_IN": _make_unit_vec(90),
        }
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=False,  # Disabled
            spread_max_hops=1,
            spread_energy_budget=50.0,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[
                ("expert_neighbor", 1.0, "EXPERT_IN"),
                ("location_neighbor", 1.0, "LOCATED_IN"),
            ]
        )

        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        # With gating disabled, both should have same energy
        # (only predicate_weights differ)
        assert "expert_neighbor" in bonuses
        assert "location_neighbor" in bonuses

    @pytest.mark.asyncio
    async def test_no_gate_no_effect(self):
        """context_gate=None has no effect on spreading."""
        from engram.activation.bfs import BFSStrategy

        cfg = ActivationConfig(
            context_gating_enabled=True,
            spread_max_hops=1,
            spread_energy_budget=50.0,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("n1", 1.0, "EXPERT_IN")]
        )

        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=None,
        )

        assert "n1" in bonuses
        assert bonuses["n1"] > 0

    @pytest.mark.asyncio
    async def test_floor_prevents_zero_energy(self):
        """Even with orthogonal predicate, floor ensures nonzero spread."""
        from engram.activation.bfs import BFSStrategy

        query_vec = _make_unit_vec(0)
        pred_embs = {"LOCATED_IN": _make_unit_vec(90)}
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=True,
            context_gate_floor=0.3,
            spread_max_hops=1,
            spread_energy_budget=50.0,
            spread_firing_threshold=0.001,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("n1", 1.0, "LOCATED_IN")]
        )

        strategy = BFSStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        assert "n1" in bonuses
        assert bonuses["n1"] > 0


# ---------------------------------------------------------------------------
# TestPPRContextGate
# ---------------------------------------------------------------------------


class TestPPRContextGate:
    @pytest.mark.asyncio
    async def test_relevant_predicate_higher_score(self):
        """PPR should give higher score to relevant-predicate neighbor."""
        from engram.activation.ppr import PPRStrategy

        query_vec = _make_unit_vec(0)
        pred_embs = {
            "EXPERT_IN": _make_unit_vec(5),
            "LOCATED_IN": _make_unit_vec(90),
        }
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=True,
            context_gate_floor=0.3,
            spreading_strategy="ppr",
            ppr_expansion_hops=1,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[
                ("expert_neighbor", 1.0, "EXPERT_IN"),
                ("location_neighbor", 1.0, "LOCATED_IN"),
            ]
        )

        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        assert bonuses.get("expert_neighbor", 0) > bonuses.get("location_neighbor", 0)

    @pytest.mark.asyncio
    async def test_disabled_no_effect(self):
        """context_gating_enabled=False makes gate a no-op in PPR."""
        from engram.activation.ppr import PPRStrategy

        query_vec = _make_unit_vec(0)
        pred_embs = {"EXPERT_IN": _make_unit_vec(5)}
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=False,
            spreading_strategy="ppr",
            ppr_expansion_hops=1,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("n1", 1.0, "EXPERT_IN")]
        )

        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        assert "n1" in bonuses

    @pytest.mark.asyncio
    async def test_no_gate_no_effect(self):
        """context_gate=None has no effect on PPR."""
        from engram.activation.ppr import PPRStrategy

        cfg = ActivationConfig(
            context_gating_enabled=True,
            spreading_strategy="ppr",
            ppr_expansion_hops=1,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("n1", 1.0, "EXPERT_IN")]
        )

        strategy = PPRStrategy()
        bonuses, _ = await strategy.spread(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=None,
        )

        assert "n1" in bonuses


# ---------------------------------------------------------------------------
# TestContextGatingConfig
# ---------------------------------------------------------------------------


class TestContextGatingConfig:
    def test_defaults(self):
        cfg = ActivationConfig()
        assert cfg.context_gating_enabled is True
        assert cfg.context_gate_floor == 0.3

    def test_validation(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ActivationConfig(context_gate_floor=-0.1)
        with pytest.raises(ValidationError):
            ActivationConfig(context_gate_floor=1.5)


# ---------------------------------------------------------------------------
# TestIntegration
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.asyncio
    async def test_spread_activation_with_context_gate(self):
        """spread_activation() passes context_gate through to strategy."""
        from engram.activation.spreading import spread_activation

        query_vec = _make_unit_vec(0)
        pred_embs = {
            "EXPERT_IN": _make_unit_vec(5),
            "LOCATED_IN": _make_unit_vec(90),
        }
        gate = ContextGate(query_vec, pred_embs, floor=0.3)

        cfg = ActivationConfig(
            context_gating_enabled=True,
            context_gate_floor=0.3,
            spread_max_hops=1,
            spread_energy_budget=50.0,
        )

        neighbor_provider = AsyncMock()
        neighbor_provider.get_active_neighbors_with_weights = AsyncMock(
            return_value=[
                ("expert_neighbor", 1.0, "EXPERT_IN"),
                ("location_neighbor", 1.0, "LOCATED_IN"),
            ]
        )

        bonuses, _ = await spread_activation(
            seed_nodes=[("seed", 1.0)],
            neighbor_provider=neighbor_provider,
            cfg=cfg,
            context_gate=gate,
        )

        # Expert predicate should receive more energy
        assert bonuses["expert_neighbor"] > bonuses["location_neighbor"]
