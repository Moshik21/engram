"""Consolidation engine: orchestrates phases and persists audit trail."""

from __future__ import annotations

import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.phases.compact import AccessHistoryCompactionPhase
from engram.consolidation.phases.dream import DreamSpreadingPhase
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.consolidation.phases.merge import EntityMergePhase
from engram.consolidation.phases.prune import PrunePhase
from engram.consolidation.phases.reindex import ReindexPhase
from engram.consolidation.phases.replay import EpisodeReplayPhase
from engram.consolidation.phases.triage import TriagePhase
from engram.consolidation.store import SQLiteConsolidationStore
from engram.events.bus import EventBus
from engram.models.consolidation import (
    ConsolidationCycle,
    CycleContext,
    DreamAssociationRecord,
    DreamRecord,
    InferredEdge,
    MergeRecord,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    TriageRecord,
)

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Orchestrates memory consolidation cycles."""

    def __init__(
        self,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        consolidation_store: SQLiteConsolidationStore | None = None,
        event_bus: EventBus | None = None,
        extractor: object | None = None,
        llm_client: object | None = None,
        graph_manager: object | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._store = consolidation_store
        self._event_bus = event_bus
        self._running = False
        self._cancelled = False

        self._phases = [
            TriagePhase(graph_manager=graph_manager),
            EpisodeReplayPhase(extractor=extractor),
            EntityMergePhase(llm_client=llm_client),
            EdgeInferencePhase(llm_client=llm_client, escalation_client=llm_client),
            PrunePhase(),
            AccessHistoryCompactionPhase(),
            ReindexPhase(),
            DreamSpreadingPhase(),
        ]

    @property
    def is_running(self) -> bool:
        return self._running

    def cancel(self) -> None:
        """Request cancellation of the current cycle (checked between phases)."""
        self._cancelled = True

    async def run_cycle(
        self,
        group_id: str,
        trigger: str = "manual",
        dry_run: bool | None = None,
    ) -> ConsolidationCycle:
        """Execute a full consolidation cycle.

        Returns the completed cycle with phase results.
        """
        if self._running:
            raise RuntimeError("A consolidation cycle is already running")

        if dry_run is None:
            dry_run = self._cfg.consolidation_dry_run

        self._running = True
        self._cancelled = False

        cycle = ConsolidationCycle(
            group_id=group_id,
            trigger=trigger,
            dry_run=dry_run,
            status="running",
        )

        # Persist initial cycle
        if self._store:
            await self._store.save_cycle(cycle)

        context = CycleContext()

        self._publish(
            group_id,
            "consolidation.started",
            {
                "cycle_id": cycle.id,
                "dry_run": dry_run,
                "trigger": trigger,
            },
        )

        try:
            for phase in self._phases:
                if self._cancelled:
                    cycle.status = "cancelled"
                    break

                self._publish(
                    group_id,
                    f"consolidation.phase.{phase.name}.started",
                    {
                        "cycle_id": cycle.id,
                        "phase": phase.name,
                    },
                )

                try:
                    result, records = await phase.execute(
                        group_id=group_id,
                        graph_store=self._graph,
                        activation_store=self._activation,
                        search_index=self._search,
                        cfg=self._cfg,
                        cycle_id=cycle.id,
                        dry_run=dry_run,
                        context=context,
                    )
                    cycle.phase_results.append(result)

                    # Persist audit records
                    if self._store:
                        for record in records:
                            if isinstance(record, MergeRecord):
                                await self._store.save_merge_record(record)
                            elif isinstance(record, InferredEdge):
                                await self._store.save_inferred_edge(record)
                            elif isinstance(record, PruneRecord):
                                await self._store.save_prune_record(record)
                            elif isinstance(record, ReindexRecord):
                                await self._store.save_reindex_record(record)
                            elif isinstance(record, ReplayRecord):
                                await self._store.save_replay_record(record)
                            elif isinstance(record, DreamAssociationRecord):
                                await self._store.save_dream_association_record(record)
                            elif isinstance(record, DreamRecord):
                                await self._store.save_dream_record(record)
                            elif isinstance(record, TriageRecord):
                                await self._store.save_triage_record(record)

                    self._publish(
                        group_id,
                        f"consolidation.phase.{phase.name}.completed",
                        {
                            "cycle_id": cycle.id,
                            "phase": phase.name,
                            "items_processed": result.items_processed,
                            "items_affected": result.items_affected,
                        },
                    )

                except Exception as exc:
                    logger.error(
                        "Phase %s failed (non-fatal): %s",
                        phase.name,
                        exc,
                        exc_info=True,
                    )
                    from engram.models.consolidation import PhaseResult

                    cycle.phase_results.append(
                        PhaseResult(
                            phase=phase.name,
                            status="error",
                            error=str(exc),
                        )
                    )
                    self._publish(
                        group_id,
                        f"consolidation.phase.{phase.name}.failed",
                        {
                            "cycle_id": cycle.id,
                            "phase": phase.name,
                            "error": str(exc),
                        },
                    )

            if cycle.status != "cancelled":
                cycle.status = "completed"

        except Exception as exc:
            cycle.status = "failed"
            cycle.error = str(exc)
            logger.error("Consolidation cycle failed: %s", exc, exc_info=True)

        finally:
            self._running = False
            cycle.completed_at = time.time()
            cycle.total_duration_ms = round(
                (cycle.completed_at - cycle.started_at) * 1000,
                1,
            )

            if self._store:
                await self._store.update_cycle(cycle)

            self._publish(
                group_id,
                "consolidation.completed",
                {
                    "cycle_id": cycle.id,
                    "status": cycle.status,
                    "duration_ms": cycle.total_duration_ms,
                    "phases": len(cycle.phase_results),
                },
            )

        return cycle

    def _publish(self, group_id: str, event_type: str, payload: dict) -> None:
        """Publish an event if event bus is available."""
        if self._event_bus:
            self._event_bus.publish(group_id, event_type, payload)
