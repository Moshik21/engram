"""M2.2/M2.3/M2.4 — the usage-ranking core surgery (unit level).

Covers:
- M2.3: the multiplicative usage tiebreaker final = composite * (1 + beta*u)
  applied post-composite pre-sort; G3 tie-probe (used wins equal-sem ties);
  the overtake theorem bound (usage never beats semantics beyond the beta band).
- M2.2: reader neutralization — with the flag ON a POPULATED activation
  store scores/seeds/goal-primes byte-identically to an EMPTY one when the
  usage store is empty (activation readers replaced).
- M2.4: the bounded top-used store helper reads only usage_events.
"""

from __future__ import annotations

import math
import time

import pytest

from engram.activation.engine import compute_u
from engram.activation.spreading import identify_seeds
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import score_candidates
from engram.storage.memory.activation import MemoryActivationStore

NOW = 1_700_000_000.0


def _used_state(node_id: str, n_used: int = 5, age_s: float = 600.0) -> ActivationState:
    state = ActivationState(node_id=node_id)
    for k in range(n_used):
        state.record_usage_event(NOW - age_s - k, 0.3)
    return state


def _accessed_state(node_id: str, n: int = 20) -> ActivationState:
    """Hygiene history only (surfaced-style) — no usage events."""
    return ActivationState(
        node_id=node_id,
        access_history=[NOW - 60.0 * k for k in range(1, n + 1)],
        access_count=n,
    )


def _score(cfg, candidates, states, **kwargs):
    return score_candidates(
        candidates=candidates,
        spreading_bonuses=kwargs.pop("spreading_bonuses", {}),
        hop_distances=kwargs.pop("hop_distances", {}),
        seed_node_ids=kwargs.pop("seed_node_ids", set()),
        activation_states=states,
        now=NOW,
        cfg=cfg,
        **kwargs,
    )


class TestUsageMultiplier:
    def test_g3_used_wins_equal_sem_tie(self):
        """G3 wiring proof: equal-sem pair, the used one ranks first, 100%."""
        cfg = ActivationConfig(
            usage_ranking_enabled=True,
            usage_beta_route=0.30,
            exploration_weight=0.0,
            rediscovery_weight=0.0,
        )
        states = {"used_ent": _used_state("used_ent")}
        for _ in range(20):
            scored = _score(cfg, [("plain_ent", 0.6), ("used_ent", 0.6)], states)
            assert scored[0].node_id == "used_ent"
            assert scored[0].score > scored[1].score

    def test_multiplier_is_exact_composite_times_beta_u(self):
        cfg = ActivationConfig(
            usage_ranking_enabled=True,
            usage_beta_route=0.25,
            exploration_weight=0.0,
            rediscovery_weight=0.0,
        )
        state = _used_state("e1")
        u = compute_u(state, NOW, cfg)
        assert u > 0.0
        scored = _score(cfg, [("e1", 0.6)], {"e1": state})
        composite = cfg.weight_semantic * 0.6
        assert scored[0].score == pytest.approx(composite * (1.0 + 0.25 * u))

    def test_empty_usage_store_multiplier_is_noop(self):
        """u=0 => multiplier exactly 1.0 — scores byte-identical to no state."""
        cfg = ActivationConfig(usage_ranking_enabled=True, usage_beta_route=0.30)
        with_state = _score(cfg, [("e1", 0.6)], {"e1": ActivationState(node_id="e1")})
        without = _score(cfg, [("e1", 0.6)], {})
        assert with_state[0].score == without[0].score

    def test_overtake_theorem_bound(self):
        """X overtakes Y only if sem(X) > sem(Y)/(1+beta): saturated usage
        cannot rescue a semantically buried item."""
        cfg = ActivationConfig(
            usage_ranking_enabled=True,
            usage_beta_route=0.30,
            exploration_weight=0.0,
            rediscovery_weight=0.0,
        )
        # Saturated u ~= 1.0 (50 confirmed, recent)
        sat = ActivationState(node_id="buried")
        for k in range(50):
            sat.record_usage_event(NOW - 60.0 - k, 1.0)
        assert compute_u(sat, NOW, cfg) > 0.99

        # Outside the band: 0.30 * 1.30 < 0.50 — gold must win.
        scored = _score(cfg, [("gold", 0.50), ("buried", 0.30)], {"buried": sat})
        assert scored[0].node_id == "gold"

        # Inside the band: 0.45 * 1.30 > 0.50 — used item may flip (near-peer).
        scored = _score(cfg, [("gold", 0.50), ("buried", 0.45)], {"buried": sat})
        assert scored[0].node_id == "buried"

    def test_flag_off_populated_usage_store_is_inert(self):
        """Kill-switch: flag OFF ignores usage_events entirely."""
        cfg = ActivationConfig(
            usage_ranking_enabled=False,
            exploration_weight=0.0,
            rediscovery_weight=0.0,
        )
        used = _used_state("e1")
        used.access_history = []  # isolate: no hygiene history either
        with_usage = _score(cfg, [("e1", 0.6)], {"e1": used})
        without = _score(cfg, [("e1", 0.6)], {})
        assert with_usage[0].score == without[0].score


class TestReaderNeutralization:
    """M2.2: flag ON + populated ACTIVATION history + empty usage store ==
    empty store, byte-identical, reader by reader."""

    def test_scorer_ignores_access_history_under_flag(self):
        cfg = ActivationConfig(
            usage_ranking_enabled=True,
            exploration_weight=0.0,  # exploration/rediscovery read history by
            rediscovery_weight=0.0,  # design (kept per RF_target §3) — pinned off
        )
        populated = {"e1": _accessed_state("e1")}
        a = _score(cfg, [("e1", 0.6), ("e2", 0.4)], populated)
        b = _score(cfg, [("e1", 0.6), ("e2", 0.4)], {})
        assert [(r.node_id, r.score, r.activation) for r in a] == [
            (r.node_id, r.score, r.activation) for r in b
        ]
        assert all(r.activation == 0.0 for r in a)

    def test_seed_energy_ignores_activation_under_flag(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        populated = {"e1": _accessed_state("e1")}
        cands = [("e1", 0.8), ("e2", 0.7)]
        seeds_pop = identify_seeds(cands, populated, NOW, cfg)
        seeds_empty = identify_seeds(cands, {}, NOW, cfg)
        assert seeds_pop == seeds_empty
        # Empty-store energy formula: sem * 0.15 floor.
        assert dict(seeds_pop)["e1"] == pytest.approx(0.8 * 0.15)

    def test_temporal_mode_seed_energy_is_constant_under_flag(self):
        cfg = ActivationConfig(usage_ranking_enabled=True)
        populated = {"e1": _accessed_state("e1")}
        seeds = identify_seeds([("e1", 0.0)], populated, NOW, cfg, temporal_mode=True)
        assert seeds == [("e1", 0.15)]

    def test_flag_off_seed_energy_still_reads_activation(self):
        """Regression guard: the neutralization is flag-gated, not global."""
        cfg = ActivationConfig(usage_ranking_enabled=False)
        populated = {"e1": _accessed_state("e1")}
        seeds_pop = identify_seeds([("e1", 0.8)], populated, NOW, cfg)
        seeds_empty = identify_seeds([("e1", 0.8)], {}, NOW, cfg)
        assert dict(seeds_pop)["e1"] > dict(seeds_empty)["e1"]

    @pytest.mark.asyncio
    async def test_goal_priming_activation_reader_neutralized(self):
        from dataclasses import dataclass, field
        from unittest.mock import AsyncMock

        from engram.retrieval.goals import identify_active_goals

        @dataclass
        class _Entity:
            id: str
            name: str
            entity_type: str
            attributes: dict = field(default_factory=dict)
            deleted_at: object = None

        goal = _Entity(id="g1", name="Ship it", entity_type="Goal")
        graph_store = AsyncMock()

        async def find_entities(entity_type=None, group_id=None, limit=10, **kwargs):
            return [goal] if entity_type == "Goal" else []

        graph_store.find_entities = AsyncMock(side_effect=find_entities)
        graph_store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(
            return_value=_accessed_state("g1")
        )

        cfg = ActivationConfig(goal_priming_enabled=True, usage_ranking_enabled=True)
        goals = await identify_active_goals(graph_store, activation_store, "default", cfg)
        # act_level := 0.0 — same outcome as an empty store (floor filters it).
        assert goals == []
        # The store is not even consulted on this path.
        activation_store.get_activation.assert_not_called()


class TestTopUsedStoreHelper:
    @pytest.mark.asyncio
    async def test_top_used_orders_by_u_and_excludes_surfaced(self):
        cfg = ActivationConfig()
        store = MemoryActivationStore(cfg)
        now = time.time()
        # heavy user signal
        for k in range(10):
            await store.record_access("ent_hot", now - 60 * k, group_id="g", tier="confirmed")
        # light signal
        await store.record_access("ent_warm", now - 3600, group_id="g", tier="used")
        # surfaced-only control: high ACT-R activation, zero usage weight
        for k in range(30):
            await store.record_access("ent_ctrl", now - 30 * k, group_id="g", tier="surfaced")

        top = await store.get_top_used(group_id="g", limit=10, now=now)
        ids = [eid for eid, _u in top]
        assert ids == ["ent_hot", "ent_warm"]
        assert "ent_ctrl" not in ids
        us = dict(top)
        assert us["ent_hot"] > us["ent_warm"] > 0.0

    @pytest.mark.asyncio
    async def test_top_used_is_bounded_and_group_scoped(self):
        cfg = ActivationConfig()
        store = MemoryActivationStore(cfg)
        now = time.time()
        for i in range(15):
            await store.record_access(f"e{i:02d}", now - i, group_id="g", tier="used")
        await store.record_access("other", now, group_id="h", tier="confirmed")

        top = await store.get_top_used(group_id="g", limit=10, now=now)
        assert len(top) == 10
        assert all(not eid == "other" for eid, _ in top)

    @pytest.mark.asyncio
    async def test_top_used_matches_compute_u_exactly(self):
        cfg = ActivationConfig()
        store = MemoryActivationStore(cfg)
        now = time.time()
        await store.record_access("e1", now - 86400.0, group_id="g", tier="used")
        state = await store.get_activation("e1")
        expected = compute_u(state, now, cfg)
        top = await store.get_top_used(group_id="g", limit=5, now=now)
        assert top == [("e1", expected)]
        # sanity: the worked-formula value
        f = min(1.0, math.log1p(0.3) / math.log1p(cfg.usage_n_cap))
        r_prime = cfg.usage_r_floor + (1 - cfg.usage_r_floor) * 2.0 ** (
            -86400.0 / (cfg.usage_half_life_days * 86400.0)
        )
        assert expected == pytest.approx(f * r_prime)


class TestFrequencyTopUsedPool:
    @pytest.mark.asyncio
    async def test_flag_on_frequency_pool_reads_top_used_not_activation(self):
        from engram.retrieval.candidate_pool import _empty_pool, _top_used_pool

        class Spy:
            def __init__(self):
                self.top_used_calls = 0
                self.top_activated_calls = 0

            async def get_top_used(self, group_id=None, limit=10, now=None):
                self.top_used_calls += 1
                return [("e1", 0.5)]

            async def get_top_activated(self, group_id=None, limit=20, now=None):
                self.top_activated_calls += 1
                return []

        spy = Spy()
        out = await _top_used_pool("g", spy, 10, time.time())
        assert out == [("e1", 0.5)]
        assert spy.top_used_calls == 1
        assert spy.top_activated_calls == 0
        assert await _empty_pool() == []

    @pytest.mark.asyncio
    async def test_top_used_pool_zero_limit_or_missing_helper(self):
        from engram.retrieval.candidate_pool import _top_used_pool

        class NoHelper:
            pass

        assert await _top_used_pool("g", NoHelper(), 10, time.time()) == []

        class Spy:
            async def get_top_used(self, group_id=None, limit=10, now=None):  # pragma: no cover
                raise AssertionError("must not be called at limit 0")

        assert await _top_used_pool("g", Spy(), 0, time.time()) == []
