"""Tests for the consolidation scheduler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.scheduler import PHASE_TIERS, ConsolidationScheduler

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

    def get_pressure(self, group_id: str, cfg, hygiene_debt=None) -> float:
        return self._value

    def reset(self, group_id: str) -> None:
        self.reset_called = True


class _FakeTemporalScanner:
    def __init__(self) -> None:
        self.scan = AsyncMock()


class TestConsolidationScheduler:
    @pytest.mark.asyncio
    async def test_start_creates_task(self):
        """is_active=True after start."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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
            consolidation_tiered_enabled=False,
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

        engine.run_cycle.assert_called_once()
        kwargs = engine.run_cycle.call_args.kwargs
        assert kwargs["group_id"] == "test_group"
        assert kwargs["trigger"] == "scheduled"
        # Loop steward overlay always passes phase_names/cfg (may be None/base)
        assert "phase_names" in kwargs
        assert "cfg" in kwargs

    @pytest.mark.asyncio
    async def test_error_doesnt_stop_loop(self):
        """Exception in run_cycle doesn't kill the loop."""
        engine = _FakeEngine()
        engine.run_cycle.side_effect = [RuntimeError("boom"), None]
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_tiered_enabled=False,
        )
        scheduler = ConsolidationScheduler(engine, cfg)

        # Use iterator with default to avoid StopIteration inside the
        # coroutine (PEP 479 converts it to RuntimeError).
        time_seq = iter([0.0, 100.0, 200.0, 200.0])
        with (
            patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep,
            patch(_TIME_PATH, side_effect=lambda: next(time_seq, 999.0)),
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

    @pytest.mark.asyncio
    async def test_temporal_scanner_uses_explicit_graph_store(self):
        """Temporal scans use scheduler dependencies instead of app state."""
        engine = _FakeEngine()
        scanner = _FakeTemporalScanner()
        graph_store = object()
        cfg = ActivationConfig(
            consolidation_enabled=False,
            consolidation_interval_seconds=60.0,
            consolidation_tiered_enabled=False,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="brain",
            temporal_scanner=scanner,
            graph_store=graph_store,
        )

        with patch(_SLEEP_PATH, new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = [None, asyncio.CancelledError()]
            scheduler.start()
            try:
                await scheduler._task
            except asyncio.CancelledError:
                pass

        scanner.scan.assert_awaited_once_with("brain", graph_store)
        engine.run_cycle.assert_not_called()


class TestPressureTriggering:
    @pytest.mark.asyncio
    async def test_pressure_triggers_above_threshold(self):
        """High pressure → trigger='pressure'."""
        engine = _FakeEngine()
        pressure = _FakePressure(pressure_value=200.0)
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=3600.0,
            consolidation_tiered_enabled=False,
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

        engine.run_cycle.assert_called_once()
        kwargs = engine.run_cycle.call_args.kwargs
        assert kwargs["group_id"] == "test"
        assert kwargs["trigger"] == "pressure"
        assert "cfg" in kwargs
        assert pressure.reset_called

    @pytest.mark.asyncio
    async def test_pressure_scheduled_below_threshold(self):
        """Low pressure but interval elapsed → trigger='scheduled'."""
        engine = _FakeEngine()
        pressure = _FakePressure(pressure_value=10.0)  # Below threshold
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_tiered_enabled=False,
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

        engine.run_cycle.assert_called_once()
        kwargs = engine.run_cycle.call_args.kwargs
        assert kwargs["group_id"] == "test"
        assert kwargs["trigger"] == "scheduled"
        assert "cfg" in kwargs

    @pytest.mark.asyncio
    async def test_pressure_disabled_ignores_count(self):
        """pressure_enabled=False → always trigger='scheduled'."""
        engine = _FakeEngine()
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_interval_seconds=60.0,
            consolidation_tiered_enabled=False,
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

        engine.run_cycle.assert_called_once()
        kwargs = engine.run_cycle.call_args.kwargs
        assert kwargs["group_id"] == "test"
        assert kwargs["trigger"] == "scheduled"
        assert "cfg" in kwargs


class _FakeGraphStore:
    def __init__(self, open_work_count: int) -> None:
        self.open_work_count = open_work_count
        self.get_open_work_metrics = AsyncMock(
            return_value={"open_work_count": open_work_count},
        )


class _FakeConsolidationStore:
    def __init__(self) -> None:
        self.tier_times: dict[str, float] = {}
        self.save_calls: list[dict[str, float]] = []

    async def get_scheduler_tier_last_runs(self, group_id: str) -> dict[str, float]:
        return dict(self.tier_times)

    async def save_scheduler_tier_last_runs(
        self,
        group_id: str,
        tier_times: dict[str, float],
    ) -> None:
        self.tier_times.update(tier_times)
        self.save_calls.append(dict(tier_times))


class TestBacklogAndPersistence:
    @pytest.mark.asyncio
    async def test_backlog_triggers_run_cycle_with_warm_phases(self):
        """High open-work backlog runs warm-tier phases via scheduler loop."""
        engine = _FakeEngine()
        graph_store = _FakeGraphStore(open_work_count=2_000)
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_tiered_enabled=True,
            consolidation_tier_hot_seconds=3600.0,
            consolidation_tier_warm_seconds=7200.0,
            consolidation_tier_cold_seconds=21600.0,
            consolidation_open_work_backlog_enabled=True,
            consolidation_open_work_backlog_threshold=500,
            consolidation_open_work_backlog_cooldown_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            graph_store=graph_store,
        )
        scheduler._last_tier_time = {"hot": 1_000.0, "warm": 1_000.0, "cold": 1_000.0}
        scheduler._tier_times_loaded = True

        time_values = [1_100.0, 1_100.0, 1_100.0]
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

        engine.run_cycle.assert_called_once()
        call_kwargs = engine.run_cycle.call_args.kwargs
        assert call_kwargs["group_id"] == "default"
        assert "backlog" in call_kwargs["trigger"]
        assert "evidence_adjudication" in call_kwargs["phase_names"]
        assert "warm" in call_kwargs["trigger"]

    @pytest.mark.asyncio
    async def test_backlog_triggers_warm_tier_before_interval(self):
        engine = _FakeEngine()
        graph_store = _FakeGraphStore(open_work_count=2_000)
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_tiered_enabled=True,
            consolidation_tier_hot_seconds=3600.0,
            consolidation_tier_warm_seconds=7200.0,
            consolidation_tier_cold_seconds=21600.0,
            consolidation_open_work_backlog_enabled=True,
            consolidation_open_work_backlog_threshold=500,
            consolidation_open_work_backlog_cooldown_seconds=60.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            graph_store=graph_store,
        )
        scheduler._last_tier_time = {"hot": 1_000.0, "warm": 1_000.0, "cold": 1_000.0}

        due = await scheduler._get_due_phases(now=1_100.0)

        assert due is not None
        assert "evidence_adjudication" in due
        assert due.issubset(set(PHASE_TIERS))

    @pytest.mark.asyncio
    async def test_helix_consolidation_store_persists_scheduler_tiers(
        self,
        tmp_path,
        monkeypatch,
    ):
        from engram.config import HelixDBConfig
        from engram.storage.helix.consolidation import HelixConsolidationStore

        store = HelixConsolidationStore(
            HelixDBConfig(transport="native", data_dir=str(tmp_path)),
        )

        await store.save_scheduler_tier_last_runs(
            "native_brain",
            {"warm": 1_700_000_000.0},
        )

        restarted = HelixConsolidationStore(
            HelixDBConfig(transport="native", data_dir=str(tmp_path)),
        )
        loaded = await restarted.get_scheduler_tier_last_runs("native_brain")

        assert loaded["warm"] == 1_700_000_000.0
        await store.close()
        await restarted.close()

    @pytest.mark.asyncio
    async def test_tier_timestamps_persist_and_survive_restart(self):
        engine = _FakeEngine()
        store = _FakeConsolidationStore()
        store.tier_times = {"warm": 42.0}
        cfg = ActivationConfig(
            consolidation_enabled=True,
            consolidation_tiered_enabled=True,
            consolidation_tier_warm_seconds=7200.0,
        )
        scheduler = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            consolidation_store=store,
        )

        await scheduler._load_tier_times()

        assert scheduler._last_tier_time["warm"] == 42.0
        due = await scheduler._get_due_phases(now=8_000.0)
        assert due is not None
        assert "evidence_adjudication" in due

        scheduler._last_tier_time["warm"] = 200.0
        await scheduler._persist_tier_times({"warm"})
        assert store.tier_times["warm"] == 200.0

        restarted = ConsolidationScheduler(
            engine,
            cfg,
            default_group_id="default",
            consolidation_store=store,
        )
        await restarted._load_tier_times()
        assert restarted._last_tier_time["warm"] == 200.0

        not_due = await restarted._get_due_phases(now=300.0)
        assert not_due is None
