"""Tests for the ACT-R activation engine."""

from __future__ import annotations

import math
import time

from engram.activation.engine import (
    batch_compute_activations,
    compute_activation,
    compute_base_level,
    normalize_activation,
    record_access,
)
from engram.config import ActivationConfig
from engram.models.activation import ActivationState


class TestComputeBaseLevel:
    def test_empty_history(self):
        cfg = ActivationConfig()
        assert compute_base_level([], time.time(), cfg) == -10.0

    def test_single_recent_access(self):
        """Single access 10s ago -> high base level."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 10]
        base = compute_base_level(history, now, cfg)
        # 10^(-0.5) = ~0.316, ln(0.316) = ~-1.15
        assert -1.5 < base < -0.8

    def test_single_old_access(self):
        """Single access 1 hour ago -> lower base level."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 3600]
        base = compute_base_level(history, now, cfg)
        # 3600^(-0.5) = ~0.0167, ln(0.0167) = ~-4.09
        assert -4.5 < base < -3.5

    def test_single_week_old_access(self):
        """Single access 7 days ago -> very low base level."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 604800]  # 7 days
        base = compute_base_level(history, now, cfg)
        assert base < -5.0

    def test_min_age_clamping(self):
        """Access at current time is clamped to min_age_seconds."""
        cfg = ActivationConfig(min_age_seconds=1.0)
        now = time.time()
        history = [now]  # age = 0
        base = compute_base_level(history, now, cfg)
        # 1.0^(-0.5) = 1.0, ln(1.0) = 0.0
        assert abs(base - 0.0) < 0.1

    def test_multiple_accesses_boost(self):
        """Multiple accesses should produce higher B than a single access."""
        cfg = ActivationConfig()
        now = time.time()
        single = [now - 60]
        multi = [now - 60, now - 120, now - 180, now - 240, now - 300]
        base_single = compute_base_level(single, now, cfg)
        base_multi = compute_base_level(multi, now, cfg)
        assert base_multi > base_single


class TestNormalizeActivation:
    def test_sigmoid_bounds(self):
        """Output is always in (0, 1)."""
        cfg = ActivationConfig()
        assert 0.0 < normalize_activation(-100.0, cfg) < 0.01
        assert 0.99 < normalize_activation(100.0, cfg) < 1.0

    def test_sigmoid_midpoint(self):
        """B = B_mid -> activation ~0.5."""
        cfg = ActivationConfig()
        act = normalize_activation(cfg.B_mid, cfg)
        assert abs(act - 0.5) < 0.01

    def test_decay_exponent_effect(self):
        """Higher decay exponent means faster decay."""
        now = time.time()
        history = [now - 3600]
        cfg_low = ActivationConfig(decay_exponent=0.3)
        cfg_high = ActivationConfig(decay_exponent=0.8)
        base_low = compute_base_level(history, now, cfg_low)
        base_high = compute_base_level(history, now, cfg_high)
        # Higher exponent -> faster decay -> lower base level
        assert base_high < base_low


class TestComputeActivation:
    def test_empty_history_near_zero(self):
        """Empty history -> very low activation."""
        cfg = ActivationConfig()
        act = compute_activation([], time.time(), cfg)
        assert act < 0.05

    def test_recent_access_high(self):
        """Single access 10s ago -> activation ~0.84."""
        cfg = ActivationConfig()
        now = time.time()
        act = compute_activation([now - 10], now, cfg)
        assert 0.7 < act < 0.95

    def test_one_hour_ago_medium(self):
        """Single access 1 hour ago -> activation ~0.49."""
        cfg = ActivationConfig()
        now = time.time()
        act = compute_activation([now - 3600], now, cfg)
        assert 0.3 < act < 0.6

    def test_week_old_low(self):
        """Single access 7 days ago -> activation ~0.17."""
        cfg = ActivationConfig()
        now = time.time()
        act = compute_activation([now - 604800], now, cfg)
        assert act < 0.25

    def test_very_cold_node(self):
        """Single access 30 days ago -> low activation."""
        cfg = ActivationConfig()
        now = time.time()
        act = compute_activation([now - 2592000], now, cfg)
        assert act < 0.15

    def test_pipeline_matches_manual(self):
        """Full pipeline matches step-by-step calculation."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 10]

        # Manual calculation
        age = 10.0
        raw_sum = age ** (-0.5)
        raw_b = math.log(raw_sum)
        x = (raw_b - cfg.B_mid) / cfg.B_scale
        expected = 1.0 / (1.0 + math.exp(-x))

        actual = compute_activation(history, now, cfg)
        assert abs(actual - expected) < 0.001

    def test_power_law_long_tail(self):
        """Old but frequent node retains meaningful activation."""
        cfg = ActivationConfig()
        now = time.time()
        # 20 accesses spread over last 7 days
        history = [now - i * 30240 for i in range(20)]  # every ~8.4 hours
        act = compute_activation(history, now, cfg)
        # Frequent access compensates for age
        assert act > 0.5

    def test_recency_dominates_for_single_access(self):
        """Recent single access > old single access."""
        cfg = ActivationConfig()
        now = time.time()
        recent = compute_activation([now - 60], now, cfg)
        old = compute_activation([now - 86400], now, cfg)
        assert recent > old


class TestRecordAccess:
    def test_appends_and_updates(self):
        cfg = ActivationConfig()
        state = ActivationState(node_id="test")
        now = time.time()
        record_access(state, now, cfg)
        assert len(state.access_history) == 1
        assert state.access_count == 1
        assert state.last_accessed == now

    def test_caps_history(self):
        cfg = ActivationConfig(max_history_size=10)
        state = ActivationState(node_id="test")
        now = time.time()
        for i in range(15):
            record_access(state, now + i, cfg)
        assert len(state.access_history) == 10
        assert state.access_count == 15
        # Should keep most recent
        assert state.access_history[-1] == now + 14


class TestConsolidatedStrength:
    def test_base_level_with_consolidated_strength(self):
        """Consolidated strength produces higher activation than without."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 3600]  # 1 hour ago
        base_without = compute_base_level(history, now, cfg, consolidated_strength=0.0)
        base_with = compute_base_level(history, now, cfg, consolidated_strength=5.0)
        assert base_with > base_without

    def test_base_level_zero_cs_unchanged(self):
        """cs=0.0 produces identical result to the default (no cs argument)."""
        cfg = ActivationConfig()
        now = time.time()
        history = [now - 60, now - 300, now - 3600]
        base_default = compute_base_level(history, now, cfg)
        base_zero = compute_base_level(history, now, cfg, consolidated_strength=0.0)
        assert abs(base_default - base_zero) < 1e-10

    def test_activation_preserves_after_compaction(self):
        """Compact + cs gives activation within 0.1% of original."""
        cfg = ActivationConfig()
        now = time.time()
        # 200 timestamps spread over 30 days
        history = [now - i * 3600 * 3.6 for i in range(200)]
        act_before = compute_activation(history, now, cfg)

        # Simulate compaction: keep recent half, compute cs for dropped half
        keep = history[:100]
        dropped = history[100:]
        cs = sum(max(cfg.min_age_seconds, now - t) ** (-cfg.decay_exponent) for t in dropped)

        act_after = compute_activation(keep, now, cfg, consolidated_strength=cs)
        # Within 0.1% relative error
        assert abs(act_before - act_after) / max(act_before, 1e-10) < 0.001


class TestBatchCompute:
    def test_batch_compute(self):
        cfg = ActivationConfig()
        now = time.time()
        states = {
            "ent_1": ActivationState(node_id="ent_1", access_history=[now - 10]),
            "ent_2": ActivationState(node_id="ent_2", access_history=[now - 3600]),
        }
        results = batch_compute_activations(states, now, cfg)
        assert len(results) == 2
        assert results["ent_1"] > results["ent_2"]
