"""Access history compaction phase: apply logarithmic bucketing to old timestamps."""

from __future__ import annotations

import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import CycleContext, PhaseResult

logger = logging.getLogger(__name__)

_DEFAULT_ENTITY_LIMIT = 10000


class AccessHistoryCompactionPhase(ConsolidationPhase):
    """Compact access_history arrays using logarithmic bucketing."""

    @property
    def name(self) -> str:
        return "compact"

    async def execute(
        self,
        group_id: str,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        cycle_id: str,
        dry_run: bool = False,
        context: CycleContext | None = None,
    ) -> tuple[PhaseResult, list[Any]]:
        t0 = time.perf_counter()
        horizon_days = cfg.consolidation_compaction_horizon_days
        keep_min = cfg.consolidation_compaction_keep_min
        use_log = cfg.consolidation_compaction_logarithmic
        now = time.time()
        max_age_seconds = horizon_days * 86400

        # Pull activated entities for this group
        activated = await activation_store.get_top_activated(
            group_id=group_id,
            limit=_DEFAULT_ENTITY_LIMIT,
            now=now,
        )

        items_processed = 0
        items_affected = 0

        for entity_id, state in activated:
            if not state.access_history:
                continue

            # Skip entities that haven't been accessed since last compaction
            if state.last_compacted > 0 and state.last_accessed <= state.last_compacted:
                continue

            items_processed += 1

            original_len = len(state.access_history)
            if use_log:
                compacted = logarithmic_compact(
                    state.access_history,
                    now,
                    max_age_seconds,
                    keep_min,
                )
            else:
                # Simple mode: just drop old timestamps and enforce keep_min
                compacted = [t for t in state.access_history if (now - t) <= max_age_seconds]
                if len(compacted) < keep_min and state.access_history:
                    compacted = sorted(state.access_history, reverse=True)[:keep_min]

            if len(compacted) < original_len:
                items_affected += 1
                if not dry_run:
                    dropped = set(state.access_history) - set(compacted)
                    if dropped:
                        state.consolidated_strength += compute_dropped_strength(
                            dropped,
                            now,
                            cfg.decay_exponent,
                            cfg.min_age_seconds,
                        )
                    state.access_history = compacted
                    state.last_compacted = now
                    await activation_store.set_activation(entity_id, state)

        return PhaseResult(
            phase=self.name,
            items_processed=items_processed,
            items_affected=items_affected,
            duration_ms=_elapsed_ms(t0),
        ), []  # No audit records for compaction


def compute_dropped_strength(
    dropped_timestamps: set[float],
    now: float,
    decay_exponent: float,
    min_age_seconds: float,
) -> float:
    """Compute the ACT-R contribution of dropped timestamps for preservation.

    Returns the sum of age^(-d) for each dropped timestamp, which can be
    added to consolidated_strength to preserve activation accuracy.
    """
    return float(
        sum(
            max(min_age_seconds, now - t) ** (-decay_exponent)
            for t in dropped_timestamps
        )
    )


def logarithmic_compact(
    history: list[float],
    now: float,
    max_age_seconds: float,
    keep_min: int,
) -> list[float]:
    """Apply logarithmic compaction to an access history.

    Rules:
    - Last 24h: keep all timestamps
    - 1-7 days: keep one timestamp per hour (representative = max in bucket)
    - 7+ days: keep one timestamp per day (representative = max in bucket)
    - Drop anything older than max_age_seconds
    - Always keep at least keep_min timestamps

    Pure function for easy unit testing.
    """
    if not history:
        return []

    recent: list[float] = []  # < 24h
    hourly_buckets: dict[int, float] = {}  # 1-7d, keyed by hour offset
    daily_buckets: dict[int, float] = {}  # 7d+, keyed by day offset

    one_day = 86400
    seven_days = 7 * one_day

    for t in history:
        age = now - t
        if age > max_age_seconds:
            continue  # Too old
        if age <= one_day:
            recent.append(t)
        elif age <= seven_days:
            bucket = int(age / 3600)  # Hour bucket
            if bucket not in hourly_buckets or t > hourly_buckets[bucket]:
                hourly_buckets[bucket] = t
        else:
            bucket = int(age / one_day)  # Day bucket
            if bucket not in daily_buckets or t > daily_buckets[bucket]:
                daily_buckets[bucket] = t

    result = recent + list(hourly_buckets.values()) + list(daily_buckets.values())
    result.sort(reverse=True)

    # Enforce keep_min from original (sorted descending = most recent first)
    if len(result) < keep_min and history:
        all_sorted = sorted(history, reverse=True)
        result = all_sorted[:keep_min]

    return result


def _elapsed_ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
