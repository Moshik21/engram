"""Read-side helpers for consolidation audit data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from engram.models.consolidation import ConsolidationCycle


@dataclass(slots=True)
class ConsolidationCycleDetail:
    """Complete read model for a consolidation cycle detail surface."""

    cycle: ConsolidationCycle
    merges: list[Any] = field(default_factory=list)
    identifier_reviews: list[Any] = field(default_factory=list)
    inferred_edges: list[Any] = field(default_factory=list)
    prunes: list[Any] = field(default_factory=list)
    reindexes: list[Any] = field(default_factory=list)
    replays: list[Any] = field(default_factory=list)
    dreams: list[Any] = field(default_factory=list)
    decision_traces: list[Any] = field(default_factory=list)
    decision_outcomes: list[Any] = field(default_factory=list)
    distillation_examples: list[Any] = field(default_factory=list)
    calibration_snapshots: list[Any] = field(default_factory=list)


class ConsolidationAuditReader:
    """Centralized reader for optional consolidation audit-store access."""

    def __init__(self, store: Any | None) -> None:
        self._store = store

    @property
    def available(self) -> bool:
        return self._store is not None

    async def latest_cycle(self, group_id: str) -> ConsolidationCycle | None:
        cycles = await self.recent_cycles(group_id, limit=1)
        return cycles[0] if cycles else None

    async def recent_cycles(
        self,
        group_id: str,
        *,
        limit: int = 10,
    ) -> list[ConsolidationCycle]:
        if self._store is None:
            return []
        return await self._store.get_recent_cycles(group_id, limit=limit)

    async def cycle_detail(
        self,
        cycle_id: str,
        group_id: str,
    ) -> ConsolidationCycleDetail | None:
        if self._store is None:
            return None

        cycle = await self._store.get_cycle(cycle_id, group_id)
        if cycle is None:
            return None

        return ConsolidationCycleDetail(
            cycle=cycle,
            merges=await self._records("get_merge_records", cycle_id, group_id),
            identifier_reviews=await self._records(
                "get_identifier_review_records",
                cycle_id,
                group_id,
            ),
            inferred_edges=await self._records("get_inferred_edges", cycle_id, group_id),
            prunes=await self._records("get_prune_records", cycle_id, group_id),
            reindexes=await self._records("get_reindex_records", cycle_id, group_id),
            replays=await self._records("get_replay_records", cycle_id, group_id),
            dreams=await self._records("get_dream_records", cycle_id, group_id),
            decision_traces=await self._records("get_decision_traces", cycle_id, group_id),
            decision_outcomes=await self._records(
                "get_decision_outcome_labels",
                cycle_id,
                group_id,
            ),
            distillation_examples=await self._records(
                "get_distillation_examples",
                cycle_id,
                group_id,
            ),
            calibration_snapshots=await self._records(
                "get_calibration_snapshots",
                cycle_id,
                group_id,
            ),
        )

    async def evaluation_context(
        self,
        group_id: str,
        *,
        cycle_limit: int,
    ) -> tuple[list[ConsolidationCycle], list[Any]]:
        recent_cycles = await self.recent_cycles(group_id, limit=cycle_limit)
        calibration_snapshots: list[Any] = []
        if self._store is None:
            return recent_cycles, calibration_snapshots

        get_snapshots = getattr(self._store, "get_calibration_snapshots", None)
        if get_snapshots is not None:
            for cycle in recent_cycles:
                calibration_snapshots.extend(await get_snapshots(cycle.id, group_id))
        return recent_cycles, calibration_snapshots

    async def _records(self, method_name: str, cycle_id: str, group_id: str) -> list[Any]:
        if self._store is None:
            return []
        getter = getattr(self._store, method_name, None)
        if getter is None:
            return []
        records = await getter(cycle_id, group_id)
        return list(records or [])
