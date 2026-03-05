"""Tests for consolidation pressure accumulation and triggering."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.pressure import ConsolidationPressure, PressureAccumulator
from engram.events.bus import EventBus


class TestConsolidationPressure:
    def test_fresh_pressure_zero(self):
        """Fresh accumulator with no events should have near-zero pressure."""
        cfg = ActivationConfig(consolidation_pressure_time_factor=0.0)
        p = ConsolidationPressure()
        assert p.compute(cfg) == 0.0

    def test_episode_event_increments(self):
        cfg = ActivationConfig(
            consolidation_pressure_weight_episode=1.0,
            consolidation_pressure_weight_entity=0.0,
            consolidation_pressure_weight_near_miss=0.0,
            consolidation_pressure_time_factor=0.0,
        )
        p = ConsolidationPressure()
        p.episodes_since_last = 10
        assert p.compute(cfg) == 10.0

    def test_entity_creation_increments(self):
        cfg = ActivationConfig(
            consolidation_pressure_weight_episode=0.0,
            consolidation_pressure_weight_entity=0.5,
            consolidation_pressure_weight_near_miss=0.0,
            consolidation_pressure_time_factor=0.0,
        )
        p = ConsolidationPressure()
        p.entities_created = 20
        assert p.compute(cfg) == 10.0

    def test_time_factor_increases_pressure(self):
        cfg = ActivationConfig(
            consolidation_pressure_weight_episode=0.0,
            consolidation_pressure_weight_entity=0.0,
            consolidation_pressure_weight_near_miss=0.0,
            consolidation_pressure_time_factor=0.01,
        )
        p = ConsolidationPressure()
        p.last_cycle_time = time.time() - 1000  # 1000 seconds ago
        pressure = p.compute(cfg)
        assert pressure >= 9.0  # ~10.0 = 0.01 * 1000

    def test_compute_formula_matches_weighted_sum(self):
        cfg = ActivationConfig(
            consolidation_pressure_weight_episode=2.0,
            consolidation_pressure_weight_entity=3.0,
            consolidation_pressure_weight_near_miss=5.0,
            consolidation_pressure_time_factor=0.0,
        )
        p = ConsolidationPressure()
        p.episodes_since_last = 4
        p.entities_created = 2
        p.failed_dedup_near_misses = 1
        # 2*4 + 3*2 + 5*1 = 8 + 6 + 5 = 19
        assert p.compute(cfg) == 19.0

    def test_reset_clears_all_counters(self):
        p = ConsolidationPressure()
        p.episodes_since_last = 10
        p.entities_created = 5
        p.entities_modified = 3
        p.failed_dedup_near_misses = 2
        p.last_cycle_time = 0.0

        p.reset()

        assert p.episodes_since_last == 0
        assert p.entities_created == 0
        assert p.entities_modified == 0
        assert p.failed_dedup_near_misses == 0
        assert p.last_cycle_time > 0  # reset to now

    def test_snapshot_returns_independent_copy(self):
        p = ConsolidationPressure()
        p.episodes_since_last = 5

        snap = p.snapshot()
        assert snap.episodes_since_last == 5

        # Modify original — snapshot should be unaffected
        p.episodes_since_last = 99
        assert snap.episodes_since_last == 5

    def test_custom_config_weights_affect_computation(self):
        cfg_low = ActivationConfig(
            consolidation_pressure_weight_episode=0.1,
            consolidation_pressure_time_factor=0.0,
        )
        cfg_high = ActivationConfig(
            consolidation_pressure_weight_episode=10.0,
            consolidation_pressure_time_factor=0.0,
        )
        p = ConsolidationPressure()
        p.episodes_since_last = 5

        assert p.compute(cfg_high) > p.compute(cfg_low)


class TestPressureAccumulator:
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        bus = EventBus()
        acc = PressureAccumulator()

        acc.start("test", bus)
        assert "test" in acc._tasks

        await acc.stop()
        assert len(acc._tasks) == 0

    @pytest.mark.asyncio
    async def test_episode_event_increments_counter(self):
        bus = EventBus()
        acc = PressureAccumulator()
        acc.start("test", bus)

        # Publish event
        bus.publish("test", "episode.completed", {})
        await asyncio.sleep(0.05)  # Let consumer process

        snap = acc.get_snapshot("test")
        assert snap is not None
        assert snap.episodes_since_last == 1

        await acc.stop()

    @pytest.mark.asyncio
    async def test_entity_creation_event(self):
        bus = EventBus()
        acc = PressureAccumulator()
        acc.start("test", bus)

        bus.publish("test", "graph.nodes_added", {"count": 3})
        await asyncio.sleep(0.05)

        snap = acc.get_snapshot("test")
        assert snap.entities_created == 3

        await acc.stop()

    @pytest.mark.asyncio
    async def test_per_group_isolation(self):
        bus = EventBus()
        acc = PressureAccumulator()
        acc.start("group_a", bus)
        acc.start("group_b", bus)

        bus.publish("group_a", "episode.completed", {})
        bus.publish("group_a", "episode.completed", {})
        bus.publish("group_b", "episode.completed", {})
        await asyncio.sleep(0.05)

        snap_a = acc.get_snapshot("group_a")
        snap_b = acc.get_snapshot("group_b")
        assert snap_a.episodes_since_last == 2
        assert snap_b.episodes_since_last == 1

        await acc.stop()

    @pytest.mark.asyncio
    async def test_reset_clears_pressure(self):
        bus = EventBus()
        acc = PressureAccumulator()
        acc.start("test", bus)

        bus.publish("test", "episode.completed", {})
        await asyncio.sleep(0.05)

        cfg = ActivationConfig(consolidation_pressure_time_factor=0.0)
        assert acc.get_pressure("test", cfg) > 0

        acc.reset("test")
        assert acc.get_pressure("test", cfg) == 0.0

        await acc.stop()

    @pytest.mark.asyncio
    async def test_get_pressure_unknown_group_returns_zero(self):
        acc = PressureAccumulator()
        cfg = ActivationConfig()
        assert acc.get_pressure("nonexistent", cfg) == 0.0

    @pytest.mark.asyncio
    async def test_get_snapshot_unknown_group_returns_none(self):
        acc = PressureAccumulator()
        assert acc.get_snapshot("nonexistent") is None

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        bus = EventBus()
        acc = PressureAccumulator()
        acc.start("test", bus)
        task1 = acc._tasks["test"]
        acc.start("test", bus)  # Should not create a second task
        assert acc._tasks["test"] is task1
        await acc.stop()


class TestSchedulerPressureIntegration:
    @pytest.mark.asyncio
    async def test_scheduler_triggers_on_pressure(self):
        """Scheduler should trigger when pressure exceeds threshold."""
        from engram.consolidation.scheduler import ConsolidationScheduler

        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_pressure_enabled=True,
            consolidation_pressure_threshold=5.0,
            consolidation_pressure_cooldown_seconds=30.0,
            consolidation_interval_seconds=86400.0,  # Very long so interval doesn't trigger
        )

        engine = AsyncMock()
        engine.is_running = False
        engine.run_cycle = AsyncMock()

        pressure = PressureAccumulator()
        # Manually set pressure high
        pressure._pressures["default"] = ConsolidationPressure(
            episodes_since_last=100,
            last_cycle_time=time.time() - 60,  # 60s ago, past cooldown
        )

        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            pressure=pressure,
        )
        # Set last cycle time far enough in the past for cooldown check
        scheduler._last_cycle_time = time.time() - 60

        iteration = 0

        async def fast_sleep(seconds):
            nonlocal iteration
            iteration += 1
            if iteration > 1:
                raise asyncio.CancelledError

        with patch("engram.consolidation.scheduler.asyncio.sleep", side_effect=fast_sleep):
            try:
                await scheduler._loop()
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_called_once_with(
            group_id="default",
            trigger="pressure",
        )

    @pytest.mark.asyncio
    async def test_scheduler_respects_cooldown(self):
        """Scheduler should NOT trigger if cooldown hasn't elapsed."""
        from engram.consolidation.scheduler import ConsolidationScheduler

        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_pressure_enabled=True,
            consolidation_pressure_threshold=5.0,
            consolidation_pressure_cooldown_seconds=3600.0,  # 1 hour cooldown
            consolidation_interval_seconds=86400.0,
        )

        engine = AsyncMock()
        engine.is_running = False
        engine.run_cycle = AsyncMock()

        pressure = PressureAccumulator()
        pressure._pressures["default"] = ConsolidationPressure(
            episodes_since_last=100,
            last_cycle_time=time.time(),  # Just now — cooldown not elapsed
        )

        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            pressure=pressure,
        )
        scheduler._last_cycle_time = time.time()  # Just ran

        iteration = 0

        async def fast_sleep(seconds):
            nonlocal iteration
            iteration += 1
            if iteration > 1:
                raise asyncio.CancelledError

        with patch("engram.consolidation.scheduler.asyncio.sleep", side_effect=fast_sleep):
            try:
                await scheduler._loop()
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_interval_and_pressure_coexist(self):
        """Both interval and pressure can trigger cycles via interval."""
        from engram.consolidation.scheduler import ConsolidationScheduler

        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_tiered_enabled=False,
            consolidation_pressure_enabled=True,
            consolidation_pressure_threshold=10000.0,  # Very high — won't trigger
            consolidation_pressure_cooldown_seconds=30.0,
            consolidation_interval_seconds=60.0,
        )

        engine = AsyncMock()
        engine.is_running = False
        engine.run_cycle = AsyncMock()

        pressure = PressureAccumulator()
        pressure._pressures["default"] = ConsolidationPressure(
            episodes_since_last=0,
        )

        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            pressure=pressure,
        )
        # Set last cycle time far enough in the past for interval trigger (>60s)
        scheduler._last_cycle_time = time.time() - 120.0

        iteration = 0

        async def fast_sleep(seconds):
            nonlocal iteration
            iteration += 1
            if iteration > 1:
                raise asyncio.CancelledError

        with patch("engram.consolidation.scheduler.asyncio.sleep", side_effect=fast_sleep):
            try:
                await scheduler._loop()
            except asyncio.CancelledError:
                pass

        # Should trigger from interval, not pressure
        engine.run_cycle.assert_called_once_with(
            group_id="default",
            trigger="scheduled",
        )
