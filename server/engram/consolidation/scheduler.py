"""Periodic consolidation scheduler with three-tier phase scheduling."""

from __future__ import annotations

import asyncio
import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.pressure import PressureAccumulator

logger = logging.getLogger(__name__)

_PRESSURE_POLL_INTERVAL = 10.0  # seconds
_TIER_POLL_INTERVAL = 30.0  # seconds — check tiers every 30s

# Phase → scheduling tier mapping
PHASE_TIERS: dict[str, str] = {
    "triage": "hot",
    "merge": "warm",
    "infer": "warm",
    "evidence_adjudication": "warm",
    "compact": "warm",
    "mature": "warm",
    "semanticize": "warm",
    "reindex": "warm",
    "microglia": "warm",
    "replay": "cold",
    "prune": "cold",
    "schema": "cold",
    "graph_embed": "cold",
    "dream": "cold",
}


class ConsolidationScheduler:
    """Runs consolidation cycles on a configurable interval.

    Supports three scheduling modes:
    - Flat: run all phases every consolidation_interval_seconds (legacy)
    - Tiered: run hot/warm/cold phases at different intervals
    - Pressure: poll every 10s, trigger full cycle when pressure exceeds threshold

    Tiered scheduling (when enabled) runs phases at different frequencies:
    - Hot (triage): every 15 min
    - Warm (merge, infer, compact, mature, semanticize, reindex): every 2 hours
    - Cold (replay, prune, schema, graph_embed, dream): every 6 hours
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
        # Per-tier last run timestamps
        now = time.time()
        self._last_tier_time: dict[str, float] = {
            "hot": now,
            "warm": now,
            "cold": now,
        }

    def start(self) -> None:
        """Start the scheduler loop. Idempotent — won't create a second task."""
        if self._task and not self._task.done():
            return
        now = time.time()
        self._last_cycle_time = now
        self._last_tier_time = {"hot": now, "warm": now, "cold": now}
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
        pressure_enabled = self._cfg.consolidation_pressure_enabled and self._pressure is not None
        tiered = self._cfg.consolidation_tiered_enabled

        while True:
            if tiered:
                await asyncio.sleep(_TIER_POLL_INTERVAL)
            elif pressure_enabled:
                await asyncio.sleep(_PRESSURE_POLL_INTERVAL)
            else:
                await asyncio.sleep(self._cfg.consolidation_interval_seconds)

            if not self._cfg.consolidation_enabled:
                continue

            if self._engine.is_running:
                logger.debug("Skipping scheduled cycle: engine already running")
                continue

            now = time.time()

            # --- Tiered scheduling ---
            if tiered:
                due_phases = self._get_due_phases(now)
                if due_phases:
                    tier_names = sorted(
                        {PHASE_TIERS.get(p, "warm") for p in due_phases},
                    )
                    trigger = f"tiered:{'+'.join(tier_names)}"
                    try:
                        await self._engine.run_cycle(
                            group_id=self._group_id,
                            trigger=trigger,
                            phase_names=due_phases,
                        )
                        # Update per-tier timestamps
                        for tier in tier_names:
                            self._last_tier_time[tier] = time.time()
                        self._last_cycle_time = time.time()
                    except Exception:
                        logger.exception("Tiered consolidation cycle failed")
                # Also check pressure (triggers full cycle)
                if pressure_enabled:
                    pressure = self._pressure
                    elapsed = now - self._last_cycle_time
                    if (
                        pressure is not None
                        and elapsed >= self._cfg.consolidation_pressure_cooldown_seconds
                    ):
                        pressure_value = pressure.get_pressure(
                            self._group_id, self._cfg,
                        )
                        if pressure_value >= self._cfg.consolidation_pressure_threshold:
                            try:
                                await self._engine.run_cycle(
                                    group_id=self._group_id,
                                    trigger="pressure",
                                )
                                now2 = time.time()
                                self._last_cycle_time = now2
                                for tier in self._last_tier_time:
                                    self._last_tier_time[tier] = now2
                                pressure.reset(self._group_id)
                            except Exception:
                                logger.exception("Pressure consolidation cycle failed")
                continue

            # --- Flat (legacy) scheduling ---
            elapsed = now - self._last_cycle_time
            cycle_trigger: str | None = None

            if elapsed >= self._cfg.consolidation_interval_seconds:
                cycle_trigger = "scheduled"

            pressure = self._pressure
            if (
                pressure_enabled
                and pressure is not None
                and elapsed >= self._cfg.consolidation_pressure_cooldown_seconds
            ):
                pressure_value = pressure.get_pressure(
                    self._group_id,
                    self._cfg,
                )
                if pressure_value >= self._cfg.consolidation_pressure_threshold:
                    cycle_trigger = "pressure"

            if cycle_trigger is None:
                continue

            try:
                await self._engine.run_cycle(
                    group_id=self._group_id,
                    trigger=cycle_trigger,
                )
                self._last_cycle_time = time.time()
                if pressure is not None:
                    pressure.reset(self._group_id)
            except Exception:
                logger.exception("Scheduled consolidation cycle failed")

    def _get_due_phases(self, now: float) -> set[str] | None:
        """Determine which phases are due to run based on tier intervals."""
        cfg = self._cfg
        tier_intervals = {
            "hot": cfg.consolidation_tier_hot_seconds,
            "warm": cfg.consolidation_tier_warm_seconds,
            "cold": cfg.consolidation_tier_cold_seconds,
        }

        due: set[str] = set()
        for phase_name, tier in PHASE_TIERS.items():
            interval = tier_intervals.get(tier, cfg.consolidation_interval_seconds)
            last_run = self._last_tier_time.get(tier, 0.0)
            if now - last_run >= interval:
                due.add(phase_name)

        return due if due else None
