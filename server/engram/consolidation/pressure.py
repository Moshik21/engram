"""Pressure-based consolidation triggering via event bus."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from engram.config import ActivationConfig
from engram.events.bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationPressure:
    """Accumulated pressure counters for a single group."""

    episodes_since_last: int = 0
    entities_created: int = 0
    entities_modified: int = 0
    failed_dedup_near_misses: int = 0
    last_cycle_time: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Reset counters after a consolidation cycle."""
        self.episodes_since_last = 0
        self.entities_created = 0
        self.entities_modified = 0
        self.failed_dedup_near_misses = 0
        self.last_cycle_time = time.time()

    def compute(self, cfg: ActivationConfig) -> float:
        """Compute weighted pressure score.

        pressure = w_ep * episodes + w_ent * entities + w_nm * near_misses + w_t * elapsed_seconds
        """
        elapsed = time.time() - self.last_cycle_time
        return (
            cfg.consolidation_pressure_weight_episode * self.episodes_since_last
            + cfg.consolidation_pressure_weight_entity * self.entities_created
            + cfg.consolidation_pressure_weight_near_miss * self.failed_dedup_near_misses
            + cfg.consolidation_pressure_time_factor * elapsed
        )

    def snapshot(self) -> ConsolidationPressure:
        """Return an independent copy of the current state."""
        return ConsolidationPressure(
            episodes_since_last=self.episodes_since_last,
            entities_created=self.entities_created,
            entities_modified=self.entities_modified,
            failed_dedup_near_misses=self.failed_dedup_near_misses,
            last_cycle_time=self.last_cycle_time,
        )


class PressureAccumulator:
    """Subscribes to EventBus and accumulates consolidation pressure per group."""

    def __init__(self) -> None:
        self._pressures: dict[str, ConsolidationPressure] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, group_id: str, event_bus: EventBus) -> None:
        """Subscribe to events for a group and start accumulating."""
        if group_id in self._tasks:
            return

        queue = event_bus.subscribe(group_id)
        self._queues[group_id] = queue
        self._pressures[group_id] = ConsolidationPressure()
        self._tasks[group_id] = asyncio.create_task(
            self._consume(group_id, queue),
        )

    async def stop(self) -> None:
        """Cancel all consumer tasks."""
        for task in self._tasks.values():
            task.cancel()
        for task in self._tasks.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self._queues.clear()

    def get_pressure(self, group_id: str, cfg: ActivationConfig) -> float:
        """Compute current pressure for a group."""
        pressure = self._pressures.get(group_id)
        if not pressure:
            return 0.0
        return pressure.compute(cfg)

    def reset(self, group_id: str) -> None:
        """Reset pressure counters for a group (after a cycle)."""
        pressure = self._pressures.get(group_id)
        if pressure:
            pressure.reset()

    def get_snapshot(self, group_id: str) -> ConsolidationPressure | None:
        """Get an independent copy of pressure state for a group."""
        pressure = self._pressures.get(group_id)
        if not pressure:
            return None
        return pressure.snapshot()

    async def _consume(self, group_id: str, queue: asyncio.Queue) -> None:
        """Consume events from the bus and update pressure counters."""
        while True:
            try:
                event = await queue.get()
                event_type = event.get("type", "")
                payload = event.get("payload", {})

                if event_type == "episode.completed":
                    self._pressures[group_id].episodes_since_last += 1
                elif event_type == "graph.nodes_added":
                    count = payload.get("count", 1)
                    self._pressures[group_id].entities_created += count

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Pressure accumulator error for group %s",
                    group_id,
                    exc_info=True,
                )
