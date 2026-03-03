"""Tests for Thompson Sampling exploration."""

from __future__ import annotations

import time

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import score_candidates, score_candidates_thompson
from engram.storage.memory.activation import MemoryActivationStore


def _make_state(node_id, ts_alpha=1.0, ts_beta=1.0, access_count=0):
    """Helper to create an ActivationState with TS params."""
    return ActivationState(
        node_id=node_id,
        ts_alpha=ts_alpha,
        ts_beta=ts_beta,
        access_count=access_count,
    )


class TestThompsonSamplingScorer:
    def test_default_prior(self):
        """Default Beta(1,1) should give moderate exploration bonus."""
        cfg = ActivationConfig(ts_enabled=True, ts_weight=0.1)
        states = {"A": _make_state("A")}
        results = score_candidates_thompson(
            candidates=[("A", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            rng_seed=42,
        )
        assert len(results) == 1
        assert results[0].exploration_bonus > 0

    def test_high_alpha_boosts_exploration(self):
        """High ts_alpha (many successes) → higher expected exploration."""
        cfg = ActivationConfig(
            ts_enabled=True,
            ts_weight=0.5,
            rediscovery_weight=0.0,
        )
        states_high = {"A": _make_state("A", ts_alpha=50.0, ts_beta=1.0)}
        states_low = {"A": _make_state("A", ts_alpha=1.0, ts_beta=50.0)}

        # Run many samples and compare averages
        high_total = 0.0
        low_total = 0.0
        n = 100
        for seed in range(n):
            r_high = score_candidates_thompson(
                candidates=[("A", 0.8)],
                spreading_bonuses={},
                hop_distances={},
                seed_node_ids=set(),
                activation_states=states_high,
                now=time.time(),
                cfg=cfg,
                rng_seed=seed,
            )
            r_low = score_candidates_thompson(
                candidates=[("A", 0.8)],
                spreading_bonuses={},
                hop_distances={},
                seed_node_ids=set(),
                activation_states=states_low,
                now=time.time(),
                cfg=cfg,
                rng_seed=seed,
            )
            high_total += r_high[0].exploration_bonus
            low_total += r_low[0].exploration_bonus

        assert high_total / n > low_total / n

    def test_high_beta_suppresses_exploration(self):
        """High ts_beta (many failures) → lower expected exploration."""
        cfg = ActivationConfig(
            ts_enabled=True,
            ts_weight=0.5,
            rediscovery_weight=0.0,
        )
        states = {"A": _make_state("A", ts_alpha=1.0, ts_beta=100.0)}
        total = 0.0
        n = 50
        for seed in range(n):
            results = score_candidates_thompson(
                candidates=[("A", 0.8)],
                spreading_bonuses={},
                hop_distances={},
                seed_node_ids=set(),
                activation_states=states,
                now=time.time(),
                cfg=cfg,
                rng_seed=seed,
            )
            total += results[0].exploration_bonus
        avg = total / n
        # With Beta(1, 100), expected value ~0.01, scaled by 0.5 * 0.8 = max 0.4
        assert avg < 0.05

    def test_deterministic_with_seed(self):
        """Same rng_seed should give identical results."""
        cfg = ActivationConfig(
            ts_enabled=True,
            ts_weight=0.1,
            rediscovery_weight=0.0,
        )
        states = {"A": _make_state("A")}
        r1 = score_candidates_thompson(
            candidates=[("A", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=1000000.0,
            cfg=cfg,
            rng_seed=123,
        )
        r2 = score_candidates_thompson(
            candidates=[("A", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=1000000.0,
            cfg=cfg,
            rng_seed=123,
        )
        assert r1[0].score == r2[0].score

    def test_disabled_fallback(self):
        """When ts_enabled=False, score_candidates should be used instead."""
        cfg = ActivationConfig(ts_enabled=False)
        states = {"A": _make_state("A", ts_alpha=100.0)}
        results = score_candidates(
            candidates=[("A", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
        )
        # Should use deterministic exploration, not TS
        assert results[0].exploration_bonus >= 0

    def test_zero_weight_no_ts_bonus(self):
        """ts_weight=0 should eliminate TS exploration bonus."""
        cfg = ActivationConfig(
            ts_enabled=True,
            ts_weight=0.0,
            rediscovery_weight=0.0,
        )
        states = {"A": _make_state("A", ts_alpha=50.0)}
        results = score_candidates_thompson(
            candidates=[("A", 0.8)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            rng_seed=42,
        )
        assert results[0].exploration_bonus == 0.0

    def test_zero_semantic_no_exploration(self):
        """Zero semantic similarity → no exploration bonus."""
        cfg = ActivationConfig(
            ts_enabled=True,
            ts_weight=0.1,
            rediscovery_weight=0.0,
        )
        states = {"A": _make_state("A", ts_alpha=50.0)}
        results = score_candidates_thompson(
            candidates=[("A", 0.0)],
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=time.time(),
            cfg=cfg,
            rng_seed=42,
        )
        assert results[0].exploration_bonus == 0.0

    def test_backward_compat_state(self):
        """ActivationState without ts_alpha/ts_beta defaults to 1.0."""
        state = ActivationState(node_id="X")
        assert state.ts_alpha == 1.0
        assert state.ts_beta == 1.0


class TestFeedbackRecording:
    @pytest.mark.asyncio
    async def test_positive_increments_alpha(self):
        from engram.activation.feedback import record_positive_feedback

        store = MemoryActivationStore()
        cfg = ActivationConfig(ts_positive_increment=2.0)
        state = ActivationState(node_id="A", ts_alpha=1.0, ts_beta=1.0)
        await store.set_activation("A", state)

        await record_positive_feedback("A", store, cfg)
        updated = await store.get_activation("A")
        assert updated.ts_alpha == 3.0
        assert updated.ts_beta == 1.0

    @pytest.mark.asyncio
    async def test_negative_increments_beta(self):
        from engram.activation.feedback import record_negative_feedback

        store = MemoryActivationStore()
        cfg = ActivationConfig(ts_negative_increment=1.5)
        state = ActivationState(node_id="A", ts_alpha=1.0, ts_beta=1.0)
        await store.set_activation("A", state)

        await record_negative_feedback("A", store, cfg)
        updated = await store.get_activation("A")
        assert updated.ts_alpha == 1.0
        assert updated.ts_beta == 2.5

    @pytest.mark.asyncio
    async def test_feedback_creates_state_if_missing(self):
        from engram.activation.feedback import record_positive_feedback

        store = MemoryActivationStore()
        cfg = ActivationConfig()

        await record_positive_feedback("new_entity", store, cfg)
        state = await store.get_activation("new_entity")
        assert state is not None
        assert state.ts_alpha == 2.0  # 1.0 default + 1.0 increment

    @pytest.mark.asyncio
    async def test_feedback_scope_isolation(self):
        """Feedback on one entity doesn't affect another."""
        from engram.activation.feedback import (
            record_negative_feedback,
            record_positive_feedback,
        )

        store = MemoryActivationStore()
        cfg = ActivationConfig()
        await store.set_activation("A", ActivationState(node_id="A"))
        await store.set_activation("B", ActivationState(node_id="B"))

        await record_positive_feedback("A", store, cfg)
        await record_negative_feedback("B", store, cfg)

        a = await store.get_activation("A")
        b = await store.get_activation("B")
        assert a.ts_alpha == 2.0
        assert a.ts_beta == 1.0
        assert b.ts_alpha == 1.0
        assert b.ts_beta == 2.0
