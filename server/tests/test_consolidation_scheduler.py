"""Tests for the consolidation scheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.scheduler import ConsolidationScheduler

_SLEEP_PATH = "engram.consolidation.scheduler.asyncio.sleep"
_TIME_PATH = "engram.consolidation.scheduler.time.time"


class _FakeEngine:
    """Minimal mock of ConsolidationEngine for scheduler tests."""

    def __init__(self) -> None:
        self.is_running = False
        self.run_cycle = AsyncMock()


class _FakePressure:
    """Minimal mock of PressureAccumulator for scheduler tests."""

    def __init__(self, pressure_value: float = 0.0) -> None:
        self._value = pressure_value
        self.reset_called = False

    def get_pressure(self, group_id: str, cfg) -> float:
        return self._value

    def reset(self, group_id: str) -> None:
        self.reset_called = True


class TestConsolidationScheduler:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """is_active=True after start."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)
        scheduler.start()
        assert scheduler.is_active is True
        # Cleanup
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self):
        """is_active=False after stop."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)
        scheduler.start()
        assert scheduler.is_active is True
        await scheduler.stop()
        assert scheduler.is_active is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self):
        """stop when not started is fine."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)
        await scheduler.stop()  # Should not raise
        assert scheduler.is_active is False

    @pytest.mark.asyncio
    async def test_start_idempotent(self):
        """start twice doesn't create two tasks."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)
        scheduler.start()
        task1 = scheduler._task
        scheduler.start()
        task2 = scheduler._task
        assert task1 is task2
        # Cleanup
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_skips_when_disabled(self):
        """consolidation_enabled=False skips the cycle."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=False,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)

        with patch(
            _SLEEP_PATH,
            new_callable=AsyncMock,
        ) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_running(self):
        """engine.is_running prevents concurrent cycle."""
        engine = _FakeEngine()
        engine.is_running = True
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)

        with patch(
            _SLEEP_PATH,
            new_callable=AsyncMock,
        ) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_cycle_on_interval(self):
        """Mock sleep + time, verify run_cycle called with trigger='scheduled'."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="test_group",
        )

        # time.time() calls: start() sets _last_cycle_time,
        # then _loop checks elapsed, then updates _last_cycle_time after cycle
        time_values = [
            0.0,  # start() → _last_cycle_time = 0
            100.0,  # _loop: now = 100 → elapsed = 100 ≥ 60 → trigger
            100.0,  # _loop: _last_cycle_time update after run_cycle
        ]
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=time_values),
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_called_once_with(
            group_id="test_group",
            trigger="scheduled",
        )

    @pytest.mark.asyncio
    async def test_error_doesnt_stop_loop(self):
        """Exception in run_cycle doesn't kill the loop."""
        engine = _FakeEngine()
        engine.run_cycle.side_effect = [RuntimeError("boom"), None]
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(engine, cfg)

        # Time values for: start, loop iter 1, loop iter 2, post-cycle update
        time_values = [0.0, 100.0, 200.0, 200.0]
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=time_values),
        ):
            mock_sleep.side_effect = [
                None,
                None,
                asyncio.CancelledError(),
            ]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        assert engine.run_cycle.call_count == 2


class TestPressureTriggering:
    @pytest.mark.asyncio
    async def test_pressure_triggers_above_threshold(self):
        """High pressure → trigger='pressure'."""
        engine = _FakeEngine()
        pressure = _FakePressure(pressure_value=200.0)
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=3600.0,
            consolidation_pressure_enabled=True,
            consolidation_pressure_threshold=100.0,
            consolidation_pressure_cooldown_seconds=30.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="test",
            pressure=pressure,
        )

        # start sets _last_cycle_time=0, loop checks: elapsed=100 < 3600
        # but elapsed=100 >= cooldown=30 and pressure=200 >= 100 → trigger
        time_values = [0.0, 100.0, 100.0]
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=time_values),
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_called_once_with(
            group_id="test",
            trigger="pressure",
        )
        assert pressure.reset_called

    @pytest.mark.asyncio
    async def test_pressure_scheduled_below_threshold(self):
        """Low pressure but interval elapsed → trigger='scheduled'."""
        engine = _FakeEngine()
        pressure = _FakePressure(pressure_value=10.0)  # Below threshold
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_pressure_enabled=True,
            consolidation_pressure_threshold=100.0,
            consolidation_pressure_cooldown_seconds=30.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="test",
            pressure=pressure,
        )

        # elapsed=100 >= interval=60 → "scheduled"
        # pressure=10 < 100 → not "pressure" (but "scheduled" already set)
        time_values = [0.0, 100.0, 100.0]
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=time_values),
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_called_once_with(
            group_id="test",
            trigger="scheduled",
        )

    @pytest.mark.asyncio
    async def test_pressure_disabled_ignores_count(self):
        """pressure_enabled=False → always trigger='scheduled'."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_pressure_enabled=False,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="test",
        )

        # No pressure accumulator, interval trigger only
        time_values = [0.0, 100.0, 100.0]
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=time_values),
        ):
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        engine.run_cycle.assert_called_once_with(
            group_id="test",
            trigger="scheduled",
        )
