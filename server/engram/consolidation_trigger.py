"""Consolidation trigger helpers for public control surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConsolidationTriggerResult:
    """Result of a public consolidation trigger."""

    cycle: Any
    graph_stats: dict


def _build_consolidation_engine(*args, **kwargs):
    from engram.consolidation.engine import ConsolidationEngine

    return ConsolidationEngine(*args, **kwargs)


class ConsolidationTriggerService:
    """Own ad hoc consolidation cycle construction and execution."""

    def __init__(
        self,
        *,
        graph_store: Any,
        activation_store: Any,
        search_index: Any,
        cfg: Any,
        extractor: Any,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._extractor = extractor

    async def trigger_consolidation_cycle(
        self,
        *,
        group_id: str,
        trigger: str,
        dry_run: bool,
        consolidation_store: Any | None = None,
    ) -> ConsolidationTriggerResult:
        """Run a public ad hoc consolidation cycle."""
        engine = _build_consolidation_engine(
            self._graph,
            self._activation,
            self._search,
            cfg=self._cfg,
            consolidation_store=consolidation_store,
            extractor=self._extractor,
        )
        graph_stats = await self._graph.get_stats(group_id)
        cycle = await engine.run_cycle(
            group_id=group_id,
            trigger=trigger,
            dry_run=dry_run,
        )
        return ConsolidationTriggerResult(cycle=cycle, graph_stats=graph_stats)

    def shared_sqlite_db(self) -> Any | None:
        """Return the shared sqlite handle used by lite graph stores, if any."""
        return getattr(self._graph, "_db", None)
