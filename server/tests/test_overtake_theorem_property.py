"""Overtake-theorem property test (RF_target_design section 3, reviewer-B gate).

The multiplicative tiebreaker final = composite * (1 + beta * u) with
beta <= beta_max = 0.30 and u in [0, 1] must satisfy, by construction:

    X overtakes Y  =>  sem(X) > sem(Y) / (1 + beta)

for EVERY sem pair and EVERY usage history — usage flips near-ties inside
the beta band and can never rescue a semantically buried item. The
example-based bound test in test_usage_ranking_core.py checks two fixed
pairs; this file checks the theorem over randomized (seeded) sem pairs and
randomized usage histories spanning u in [0, 1].
"""

from __future__ import annotations

import random

from engram.activation.engine import compute_u
from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import score_candidates

NOW = 1_700_000_000.0
BETA_MAX = 0.30
TRIALS = 300


def _cfg(beta: float) -> ActivationConfig:
    # exploration/rediscovery pinned off: the theorem is stated on the
    # composite = w_sem * sem baseline (design keeps those bonuses as
    # separate additive signals outside the theorem's scope).
    return ActivationConfig(
        usage_ranking_enabled=True,
        usage_beta_route=beta,
        exploration_weight=0.0,
        rediscovery_weight=0.0,
    )


def _random_usage_state(rng: random.Random, node_id: str) -> ActivationState:
    """Random tier-weighted history: u spans [0, 1] across draws."""
    state = ActivationState(node_id=node_id)
    n_events = rng.choice([0, 1, 2, 5, 10, 25, 50, 80])
    last_age_s = rng.uniform(0.0, 120.0 * 86400.0)  # 0 .. 120 days
    for k in range(n_events):
        weight = rng.choice([0.3, 0.5, 1.0])
        state.record_usage_event(NOW - last_age_s - k * 60.0, weight)
    return state


def _rank(cfg, sem_x, sem_y, state_x, state_y):
    states = {}
    if state_x is not None:
        states["x"] = state_x
    if state_y is not None:
        states["y"] = state_y
    return score_candidates(
        candidates=[("x", sem_x), ("y", sem_y)],
        spreading_bonuses={},
        hop_distances={},
        seed_node_ids=set(),
        activation_states=states,
        now=NOW,
        cfg=cfg,
    )


class TestOvertakeTheoremProperty:
    def test_random_pairs_never_violate_the_band(self):
        """For random sem pairs and random u in [0,1]: X ranks above Y only
        if sem(X) > sem(Y) / (1 + beta)."""
        rng = random.Random(0xE2A1)
        u_seen_low = False
        u_seen_high = False
        for _ in range(TRIALS):
            beta = rng.choice([0.05, 0.10, 0.25, BETA_MAX])
            cfg = _cfg(beta)
            sem_y = rng.uniform(0.05, 1.0)
            sem_x = rng.uniform(0.01, sem_y)  # X is the semantically weaker item
            state_x = _random_usage_state(rng, "x")
            state_y = _random_usage_state(rng, "y") if rng.random() < 0.5 else None

            u_x = compute_u(state_x, NOW, cfg)
            assert 0.0 <= u_x <= 1.0
            u_seen_low |= u_x < 0.05
            u_seen_high |= u_x > 0.60

            scored = _rank(cfg, sem_x, sem_y, state_x, state_y)
            if scored[0].node_id == "x":
                # The theorem's necessary condition for an overtake.
                assert sem_x * (1.0 + beta) > sem_y, (
                    f"buried item overtook outside the band: sem_x={sem_x:.4f} "
                    f"sem_y={sem_y:.4f} beta={beta} u_x={u_x:.4f}"
                )
        # The draw actually exercised both ends of the u range.
        assert u_seen_low and u_seen_high

    def test_outside_band_never_flips_even_at_saturated_u(self):
        """Constructive worst case: u = 1-saturated X, sem_x strictly outside
        the band => Y wins for every beta."""
        rng = random.Random(0xE2A2)
        for _ in range(100):
            beta = rng.choice([0.05, 0.10, 0.25, BETA_MAX])
            cfg = _cfg(beta)
            sem_y = rng.uniform(0.10, 1.0)
            # strictly outside the band (with a margin for float noise)
            sem_x = sem_y / (1.0 + beta) * rng.uniform(0.5, 0.999)
            sat = ActivationState(node_id="x")
            for k in range(60):
                sat.record_usage_event(NOW - 30.0 - k, 1.0)
            assert compute_u(sat, NOW, cfg) > 0.99
            scored = _rank(cfg, sem_x, sem_y, sat, None)
            assert scored[0].node_id == "y"

    def test_inside_band_flip_is_reachable(self):
        """The band is not vacuous: a saturated-u near-peer does flip."""
        cfg = _cfg(BETA_MAX)
        sat = ActivationState(node_id="x")
        for k in range(60):
            sat.record_usage_event(NOW - 30.0 - k, 1.0)
        scored = _rank(cfg, 0.45, 0.50, sat, None)
        assert scored[0].node_id == "x"
