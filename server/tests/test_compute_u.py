"""u-discrimination gate (RF goal M2.1).

Worked-number invariants from RF_target_design.md section 1: the picked signal
u = f * r' must reproduce the design table's numbers, discriminate on count and
age, and stay in [0, 1] under extreme inputs. The metamorphic assertions make a
hard-coded constant u fail this suite: two states differing only in n_eff must
produce different u, as must two states differing only in usage_last_ts.
"""

from __future__ import annotations

import pytest

from engram.activation.engine import compute_u
from engram.config import ActivationConfig
from engram.models.activation import DEFAULT_USAGE_TIER_WEIGHTS, ActivationState

NOW = 1_000_000_000.0
MINUTE = 60.0
DAY = 86400.0

W_USED = DEFAULT_USAGE_TIER_WEIGHTS["used"]
W_CONFIRMED = DEFAULT_USAGE_TIER_WEIGHTS["confirmed"]


@pytest.fixture
def cfg() -> ActivationConfig:
    return ActivationConfig()


def _state(events: list[tuple[float, float]]) -> ActivationState:
    state = ActivationState(node_id="e1")
    for ts, weight in events:
        state.record_usage_event(ts, weight)
    return state


class TestWorkedNumbers:
    """The design-table rows, verified numerically."""

    def test_one_surfaced_is_exactly_zero(self, cfg):
        # surfaced is ranker output: tier weight 0, never enters usage_events,
        # so n_eff == 0 and the loop is broken -- u is 0.0 EXACTLY, not small.
        assert DEFAULT_USAGE_TIER_WEIGHTS["surfaced"] == 0.0
        state = ActivationState(node_id="e1")
        assert compute_u(state, NOW, cfg) == 0.0

    def test_one_used_ten_minutes(self, cfg):
        state = _state([(NOW - 10 * MINUTE, W_USED)])
        assert compute_u(state, NOW, cfg) == pytest.approx(0.067, abs=0.005)

    def test_five_used_last_week_last_one_day(self, cfg):
        events = [(NOW - d * DAY, W_USED) for d in (6, 5, 4, 2, 1)]
        state = _state(events)
        assert compute_u(state, NOW, cfg) == pytest.approx(0.225, abs=0.01)

    def test_fifty_used_last_thirty_days_ago(self, cfg):
        events = [(NOW - (30 + i) * DAY, W_USED) for i in range(50)]
        state = _state(events)
        assert compute_u(state, NOW, cfg) == pytest.approx(0.296, abs=0.01)

    def test_one_confirmed_ten_minutes(self, cfg):
        state = _state([(NOW - 10 * MINUTE, W_CONFIRMED)])
        assert compute_u(state, NOW, cfg) == pytest.approx(0.176, abs=0.005)

    def test_fifty_confirmed_recent_saturates(self, cfg):
        events = [(NOW - MINUTE - i, W_CONFIRMED) for i in range(50)]
        state = _state(events)
        assert compute_u(state, NOW, cfg) >= 0.9


class TestSeparability:
    def test_count_separability_at_fixed_recency(self, cfg):
        """u(1 used) < u(5 used) < u(50 used), all last-touched 10 min ago."""
        one = _state([(NOW - 10 * MINUTE, W_USED)])
        five = _state([(NOW - 10 * MINUTE - i, W_USED) for i in range(5)])
        fifty = _state([(NOW - 10 * MINUTE - i, W_USED) for i in range(50)])
        u1 = compute_u(one, NOW, cfg)
        u5 = compute_u(five, NOW, cfg)
        u50 = compute_u(fifty, NOW, cfg)
        assert 0.0 < u1 < u5 < u50

    def test_age_separability_at_fixed_count(self, cfg):
        """Same 5-event history, older last-touch -> strictly smaller u."""
        u_by_age = []
        for age in (10 * MINUTE, 1 * DAY, 7 * DAY, 30 * DAY, 180 * DAY):
            state = _state([(NOW - age - i * DAY, W_USED) for i in range(5)])
            u_by_age.append(compute_u(state, NOW, cfg))
        assert u_by_age == sorted(u_by_age, reverse=True)
        assert len(set(u_by_age)) == len(u_by_age)  # strictly decreasing


class TestMetamorphicConstantKiller:
    """A hard-coded constant u MUST fail these."""

    def test_states_differing_only_in_n_eff_differ_in_u(self, cfg):
        # Identical usage_last_ts; only the weight sum differs.
        a = _state([(NOW - 10 * MINUTE, W_USED)])
        b = _state([(NOW - 20 * MINUTE, W_USED), (NOW - 10 * MINUTE, W_USED)])
        assert a.usage_last_ts == b.usage_last_ts
        assert a.n_eff != b.n_eff
        assert compute_u(a, NOW, cfg) != compute_u(b, NOW, cfg)

    def test_states_differing_only_in_last_ts_differ_in_u(self, cfg):
        # Identical n_eff; only the last-event timestamp differs.
        a = _state([(NOW - 10 * MINUTE, W_USED)])
        b = _state([(NOW - 30 * DAY, W_USED)])
        assert a.n_eff == b.n_eff
        assert a.usage_last_ts != b.usage_last_ts
        assert compute_u(a, NOW, cfg) != compute_u(b, NOW, cfg)

    def test_zero_versus_nonzero(self, cfg):
        empty = ActivationState(node_id="e1")
        touched = _state([(NOW - 10 * MINUTE, W_USED)])
        assert compute_u(empty, NOW, cfg) == 0.0
        assert compute_u(touched, NOW, cfg) > 0.0


class TestBounds:
    def test_huge_n_eff_capped_at_one(self, cfg):
        state = _state([(NOW - MINUTE, 1e12)])
        u = compute_u(state, NOW, cfg)
        assert 0.0 <= u <= 1.0
        assert u == pytest.approx(1.0, abs=1e-3)

    def test_negative_delta_clamped(self, cfg):
        """Future-dated event (clock skew) must not push u above 1."""
        state = _state([(NOW + 1 * DAY, W_CONFIRMED)])
        u = compute_u(state, NOW, cfg)
        assert 0.0 <= u <= 1.0
        # Clamped to delta=0: same as an event exactly at NOW.
        at_now = _state([(NOW, W_CONFIRMED)])
        assert u == compute_u(at_now, NOW, cfg)

    def test_zero_delta(self, cfg):
        state = _state([(NOW, W_CONFIRMED)])
        assert 0.0 <= compute_u(state, NOW, cfg) <= 1.0

    def test_extreme_age(self, cfg):
        """delta = 1e12 s: r underflows to 0, u falls to f * r_floor, in [0, 1]."""
        state = _state([(1.0, W_CONFIRMED)])
        u = compute_u(state, 1e12, cfg)
        assert 0.0 <= u <= 1.0
        assert u > 0.0  # r_floor keeps old-but-used alive

    def test_u_always_within_unit_interval(self, cfg):
        extremes = [
            [],
            [(NOW - 1e12, 1e12)],
            [(NOW + 1e12, 1e12)],
            [(NOW, 1e-12)],
            [(NOW - MINUTE, W_USED)] * 50,
        ]
        for events in extremes:
            state = _state(list(events))
            u = compute_u(state, NOW, cfg)
            assert 0.0 <= u <= 1.0


class TestConfigKnobs:
    def test_defaults(self, cfg):
        assert cfg.usage_n_cap == 50
        assert cfg.usage_half_life_days == 14.0
        assert cfg.usage_r_floor == 0.25
        assert cfg.usage_ranking_enabled is False

    def test_knob_bounds_enforced(self):
        with pytest.raises(ValueError):
            ActivationConfig(usage_n_cap=0)
        with pytest.raises(ValueError):
            ActivationConfig(usage_half_life_days=0.0)
        with pytest.raises(ValueError):
            ActivationConfig(usage_r_floor=1.0)

    def test_r_floor_zero_kills_ancient_u(self):
        cfg = ActivationConfig(usage_r_floor=0.0)
        state = _state([(1.0, W_CONFIRMED)])
        assert compute_u(state, 1e12, cfg) == 0.0
