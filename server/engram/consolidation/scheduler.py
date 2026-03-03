"""Periodic consolidation scheduler."""

from __future__ import annotations

import asyncio
import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.pressure import PressureAccumulator

logger = logging.getLogger(__name__)

_PRESSURE_POLL_INTERVAL = 10.0  # seconds


class ConsolidationScheduler:
    """Runs consolidation cycles on a configurable interval.

    Supports two triggering modes:
    - Interval: run every consolidation_interval_seconds (default)
    - Pressure: poll every 10s, trigger when pressure exceeds threshold
    Both can coexist: either condition can trigger a cycle.
    """

    def __init__(
        self,
        engine: ConsolidationEngine,
        cfg: ActivationConfig,
        default_group_id: str = "default",
        pressure: PressureAccumulator | None = None,
    ) -> None:
        self._engine = engine
        self._cfg = cfg
        self._group_id = default_group_id
        self._pressure = pressure
        self._task: asyncio.Task | None = None
        self._last_cycle_time: float = time.time()

    def start(self) -> None:
        """Start the scheduler loop. Idempotent — won't create a second task."""
        if self._task and not self._task.done():
            return
        self._last_cycle_time = time.time()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the scheduler loop and wait for cleanup. Safe to call when not started."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    @property
    def is_active(self) -> bool:
        """Whether the scheduler loop is currently running."""
        return self._task is not None and not self._task.done()

    async def _loop(self) -> None:
        """Main loop: sleep → check conditions → run cycle → repeat."""
        pressure_enabled = (
            self._cfg.consolidation_pressure_enabled
            and self._pressure is not None
        )

        while True:
            if pressure_enabled:
                # Short poll for pressure checks
                await asyncio.sleep(_PRESSURE_POLL_INTERVAL)
            else:
                await asyncio.sleep(self._cfg.consolidation_interval_seconds)

            if not self._cfg.consolidation_enabled:
                continue

            if self._engine.is_running:
                logger.debug("Skipping scheduled cycle: engine already running")
                continue

            now = time.time()
            elapsed = now - self._last_cycle_time
            trigger: str | None = None

            # Check interval trigger
            if elapsed >= self._cfg.consolidation_interval_seconds:
                trigger = "scheduled"

            # Check pressure trigger
            if (
                pressure_enabled
                and elapsed >= self._cfg.consolidation_pressure_cooldown_seconds
            ):
                pressure_value = self._pressure.get_pressure(
                    self._group_id, self._cfg,
                )
                if pressure_value >= self._cfg.consolidation_pressure_threshold:
                    trigger = "pressure"

            if trigger is None:
                continue

            try:
                await self._engine.run_cycle(
                    group_id=self._group_id,
                    trigger=trigger,
                )
                self._last_cycle_time = time.time()
                if self._pressure:
                    self._pressure.reset(self._group_id)
            except Exception:
                logger.exception("Scheduled consolidation cycle failed")
