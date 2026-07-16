"""Periodic consolidation scheduler with three-tier phase scheduling."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.phase_registry import CONSOLIDATION_PHASE_TIERS
from engram.consolidation.pressure import ConsolidationPressure, PressureAccumulator

logger = logging.getLogger(__name__)

_PRESSURE_POLL_INTERVAL = 10.0  # seconds
_TIER_POLL_INTERVAL = 30.0  # seconds — check tiers every 30s

PHASE_TIERS = CONSOLIDATION_PHASE_TIERS
_WARM_TIER_PHASES = frozenset(
    phase_name for phase_name, tier in PHASE_TIERS.items() if tier == "warm"
)


class ConsolidationScheduler:
    """Runs consolidation cycles on a configurable interval.

    Supports three scheduling modes:
    - Flat: run all phases every consolidation_interval_seconds (legacy)
    - Tiered: run hot/warm/cold phases at different intervals
    - Pressure: poll every 10s, trigger full cycle when pressure exceeds threshold

    Tiered scheduling (when enabled) runs phases at different frequencies:
    - Hot (triage): every 15 min
    - Warm (merge, calibrate, infer, adjudication, compact, mature, semanticize,
      reindex, microglia): every 2 hours
    - Cold (replay, prune, schema, graph_embed, dream): every 6 hours
    """

    def __init__(
        self,
        engine: ConsolidationEngine,
        cfg: ActivationConfig,
        default_group_id: str = "default",
        pressure: PressureAccumulator | None = None,
        temporal_scanner: object | None = None,
        graph_store: object | None = None,
        consolidation_store: object | None = None,
    ) -> None:
        self._engine = engine
        self._cfg = cfg
        self._group_id = default_group_id
        self._pressure = pressure
        self._temporal_scanner = temporal_scanner
        self._graph_store = graph_store
        self._consolidation_store = consolidation_store
        self._task: asyncio.Task | None = None
        self._last_cycle_time: float = time.time()
        self._tier_times_loaded = False
        self._last_backlog_warm_trigger: float = 0.0
        # Per-tier last run timestamps (0.0 = never run)
        self._last_tier_time: dict[str, float] = {
            "hot": 0.0,
            "warm": 0.0,
            "cold": 0.0,
        }

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

    def _loop_steward_overlay(
        self,
        phase_names: set[str] | None,
    ) -> tuple[set[str] | None, ActivationConfig]:
        """Load active LoopAdjustment and produce phase set + effective cfg.

        Does not mutate process-boot ``self._cfg``.
        """
        try:
            from engram.loop_adjustment import (
                effective_activation_config,
                effective_phase_names,
                load_active_adjustment,
            )

            adj = load_active_adjustment(self._group_id)
            cfg_eff = effective_activation_config(self._cfg, adj)
            biased = effective_phase_names(phase_names, adj)
            return biased, cfg_eff
        except Exception:
            logger.debug("Loop steward overlay skipped", exc_info=True)
            return phase_names, self._cfg

    async def _load_tier_times(self) -> None:
        """Hydrate tier timestamps from durable storage once per process."""
        if self._tier_times_loaded:
            return
        self._tier_times_loaded = True
        store = self._consolidation_store
        if store is None:
            return
        loader = getattr(store, "get_scheduler_tier_last_runs", None)
        if not callable(loader):
            return
        try:
            persisted = await loader(self._group_id)
        except Exception:
            logger.debug("Failed to load scheduler tier timestamps", exc_info=True)
            return
        if persisted:
            self._last_tier_time.update(persisted)

    async def _persist_tier_times(self, tiers: set[str]) -> None:
        store = self._consolidation_store
        if store is None or not tiers:
            return
        saver = getattr(store, "save_scheduler_tier_last_runs", None)
        if not callable(saver):
            return
        payload = {
            tier: self._last_tier_time[tier] for tier in tiers if tier in self._last_tier_time
        }
        if not payload:
            return
        try:
            await saver(self._group_id, payload)
        except Exception:
            logger.debug("Failed to persist scheduler tier timestamps", exc_info=True)

    async def _get_open_work_count(self) -> int:
        graph_store = self._graph_store
        if graph_store is None:
            return 0
        try:
            metrics_loader = getattr(graph_store, "get_open_work_metrics", None)
            if callable(metrics_loader):
                metrics = await metrics_loader(self._group_id)
            else:
                stats_loader = getattr(graph_store, "get_stats", None)
                if not callable(stats_loader):
                    logger.debug(
                        "Graph store lacks open-work metrics loaders; skipping backlog signal",
                    )
                    return 0
                stats = await stats_loader(self._group_id, exact=True)
                metrics = stats.get("adjudication_metrics") or {}
            return int(metrics.get("open_work_count", 0) or 0)
        except Exception:
            logger.debug("Failed to read open-work backlog metrics", exc_info=True)
            return 0

    async def _hygiene_debt_for_pressure(self) -> Any | None:
        """Best-effort debt snapshot so pressure sees deferred/cue sludge."""
        if not getattr(self._cfg, "consolidation_pressure_include_hygiene_debt", True):
            return None
        graph_store = self._graph_store
        if graph_store is None:
            return None
        try:
            from engram.consolidation.hygiene_debt import collect_hygiene_debt_from_store

            return await collect_hygiene_debt_from_store(graph_store, self._group_id)
        except Exception:
            logger.debug("Failed to collect hygiene debt for pressure", exc_info=True)
            return None

    async def _backlog_warm_phases(self, now: float) -> set[str] | None:
        cfg = self._cfg
        if not cfg.consolidation_open_work_backlog_enabled:
            return None
        cooldown = cfg.consolidation_open_work_backlog_cooldown_seconds
        if now - self._last_backlog_warm_trigger < cooldown:
            return None
        open_work_count = await self._get_open_work_count()
        signal = ConsolidationPressure.open_work_backlog_signal(
            open_work_count,
            cfg.consolidation_open_work_backlog_threshold,
        )
        if signal <= 0.0:
            return None
        return set(_WARM_TIER_PHASES)

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

            await self._load_tier_times()

            # Run temporal intention scan on every poll (piggyback on hot tier)
            if self._temporal_scanner is not None and self._graph_store is not None:
                try:
                    await self._temporal_scanner.scan(self._group_id, self._graph_store)
                except Exception:
                    logger.debug("Temporal intention scan failed", exc_info=True)

            if not self._cfg.consolidation_enabled:
                continue

            if self._engine.is_running:
                logger.debug("Skipping scheduled cycle: engine already running")
                continue

            now = time.time()

            # --- Tiered scheduling ---
            if tiered:
                due_phases = await self._get_due_phases(now)
                if due_phases:
                    tier_names = sorted(
                        {PHASE_TIERS.get(p, "warm") for p in due_phases},
                    )
                    backlog_triggered = bool(due_phases & _WARM_TIER_PHASES) and (
                        now - self._last_tier_time.get("warm", 0.0)
                        < self._cfg.consolidation_tier_warm_seconds
                    )
                    trigger = (
                        f"tiered:backlog+{'+'.join(tier_names)}"
                        if backlog_triggered
                        else f"tiered:{'+'.join(tier_names)}"
                    )
                    try:
                        phases, cfg_eff = self._loop_steward_overlay(due_phases)
                        await self._engine.run_cycle(
                            group_id=self._group_id,
                            trigger=trigger,
                            phase_names=phases,
                            cfg=cfg_eff,
                        )
                        # Update per-tier timestamps
                        for tier in tier_names:
                            self._last_tier_time[tier] = time.time()
                        if backlog_triggered:
                            self._last_backlog_warm_trigger = time.time()
                        await self._persist_tier_times(set(tier_names))
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
                        debt = await self._hygiene_debt_for_pressure()
                        pressure_value = pressure.get_pressure(
                            self._group_id,
                            self._cfg,
                            hygiene_debt=debt,
                        )
                        if pressure_value >= self._cfg.consolidation_pressure_threshold:
                            try:
                                phases, cfg_eff = self._loop_steward_overlay(None)
                                await self._engine.run_cycle(
                                    group_id=self._group_id,
                                    trigger="pressure",
                                    phase_names=phases,
                                    cfg=cfg_eff,
                                )
                                now2 = time.time()
                                self._last_cycle_time = now2
                                for tier in self._last_tier_time:
                                    self._last_tier_time[tier] = now2
                                await self._persist_tier_times(set(self._last_tier_time))
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
                debt = await self._hygiene_debt_for_pressure()
                pressure_value = pressure.get_pressure(
                    self._group_id,
                    self._cfg,
                    hygiene_debt=debt,
                )
                if pressure_value >= self._cfg.consolidation_pressure_threshold:
                    cycle_trigger = "pressure"

            if cycle_trigger is None:
                continue

            try:
                phases, cfg_eff = self._loop_steward_overlay(None)
                await self._engine.run_cycle(
                    group_id=self._group_id,
                    trigger=cycle_trigger,
                    phase_names=phases,
                    cfg=cfg_eff,
                )
                self._last_cycle_time = time.time()
                if pressure is not None:
                    pressure.reset(self._group_id)
            except Exception:
                logger.exception("Scheduled consolidation cycle failed")

    async def _get_due_phases(self, now: float) -> set[str] | None:
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

        backlog_phases = await self._backlog_warm_phases(now)
        if backlog_phases:
            due.update(backlog_phases)

        return due if due else None
