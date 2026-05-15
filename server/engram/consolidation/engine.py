"""Consolidation engine: orchestrates phases and persists audit trail."""

from __future__ import annotations

import logging
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.audit_reader import (
    ConsolidationAuditReader,
    ConsolidationCycleDetail,
)
from engram.consolidation.capabilities import ConsolidationCapabilityValidator
from engram.consolidation.completion import ConsolidationCycleCompletionService
from engram.consolidation.events import ConsolidationEventPublisher
from engram.consolidation.finalization import ConsolidationFinalizationService
from engram.consolidation.learning import ConsolidationLearningService
from engram.consolidation.lifecycle import build_cycle_plan
from engram.consolidation.phase_catalog import build_consolidation_phases
from engram.consolidation.phase_runner import ConsolidationPhaseRunner
from engram.events.bus import EventBus
from engram.models.consolidation import (
    ConsolidationCycle,
    CycleContext,
    PhaseResult,
)
from engram.storage.protocols import ConsolidationStore

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    """Orchestrates memory consolidation cycles."""

    def __init__(
        self,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        consolidation_store: ConsolidationStore | None = None,
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
        self._audit_reader = ConsolidationAuditReader(consolidation_store)
        self._capabilities = ConsolidationCapabilityValidator(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
        )
        self._events = ConsolidationEventPublisher(event_bus)
        self._learning = ConsolidationLearningService(
            cfg=self._cfg,
            consolidation_store=self._store,
        )
        self._finalization = ConsolidationFinalizationService(
            graph_manager=graph_manager,
        )
        self._completion = ConsolidationCycleCompletionService(
            consolidation_store=self._store,
            learning_service=self._learning,
            finalization_service=self._finalization,
            event_publisher=self._events,
        )
        self._phase_runner = ConsolidationPhaseRunner(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            consolidation_store=self._store,
        )
        self._running = False
        self._cancelled = False

        self._phases = build_consolidation_phases(
            graph_manager=graph_manager,
            extractor=extractor,
            llm_client=llm_client,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    async def get_recent_evaluation_context(
        self,
        group_id: str,
        *,
        cycle_limit: int,
    ) -> tuple[list[ConsolidationCycle], list[Any]]:
        """Return recent cycles and calibration snapshots for evaluation reports."""
        return await self._audit_reader.evaluation_context(
            group_id,
            cycle_limit=cycle_limit,
        )

    @property
    def audit_store_available(self) -> bool:
        """Whether this runtime has a consolidation audit store attached."""
        return self._audit_reader.available

    async def get_latest_cycle(self, group_id: str) -> ConsolidationCycle | None:
        """Return the latest persisted consolidation cycle for a group."""
        return await self._audit_reader.latest_cycle(group_id)

    async def get_recent_cycles(
        self,
        group_id: str,
        *,
        limit: int = 10,
    ) -> list[ConsolidationCycle]:
        """Return recent persisted consolidation cycles for a group."""
        return await self._audit_reader.recent_cycles(group_id, limit=limit)

    async def get_cycle_detail(
        self,
        cycle_id: str,
        group_id: str,
    ) -> ConsolidationCycleDetail | None:
        """Return a full persisted consolidation cycle detail view."""
        return await self._audit_reader.cycle_detail(cycle_id, group_id)

    def cancel(self) -> None:
        """Request cancellation of the current cycle (checked between phases)."""
        self._cancelled = True

    async def run_cycle(
        self,
        group_id: str,
        trigger: str = "manual",
        dry_run: bool | None = None,
        phase_names: set[str] | None = None,
    ) -> ConsolidationCycle:
        """Execute a consolidation cycle.

        Args:
            phase_names: If set, only run phases whose name is in this set.
                         If None, run all phases (full cycle).

        Returns the completed cycle with phase results.
        """
        if self._running:
            raise RuntimeError("A consolidation cycle is already running")

        if dry_run is None:
            dry_run = self._cfg.consolidation_dry_run

        cycle = ConsolidationCycle(
            group_id=group_id,
            trigger=trigger,
            dry_run=dry_run,
            status="running",
        )
        cycle_plan = build_cycle_plan(
            cycle=cycle,
            phases=self._phases,
            phase_names=phase_names,
        )
        cycle_plan.validate_requested_phase_names()

        self._running = True
        self._cancelled = False

        # Persist initial cycle
        if self._store:
            await self._store.save_cycle(cycle)

        context = CycleContext(trigger=trigger)

        self._events.cycle_started(group_id, cycle_plan)

        try:
            self._capabilities.validate(
                tuple(
                    phase
                    for phase in self._phases
                    if phase.name in cycle_plan.selected_phase_names
                ),
                cfg=self._cfg,
            )

            for phase in self._phases:
                if self._cancelled:
                    cycle.status = "cancelled"
                    break

                # Skip phases not in the requested set (tiered scheduling)
                phase_plan = cycle_plan.phase_plan(phase.name)
                if phase_plan is None or not phase_plan.selected:
                    continue

                self._events.phase_started(
                    group_id,
                    cycle_id=cycle.id,
                    phase_plan=phase_plan,
                )

                try:
                    phase_run = await self._phase_runner.run_phase(
                        phase,
                        group_id=group_id,
                        cycle_id=cycle.id,
                        dry_run=dry_run,
                        context=context,
                    )
                    cycle.phase_results.append(phase_run.result)

                    self._events.graph_delta(
                        group_id,
                        removed_node_ids=phase_run.removed_node_ids,
                        dry_run=dry_run,
                    )

                    self._events.phase_completed(
                        group_id,
                        cycle_id=cycle.id,
                        result=phase_run.result,
                        phase_plan=phase_plan,
                    )

                except Exception as exc:
                    logger.error(
                        "Phase %s failed (non-fatal): %s",
                        phase.name,
                        exc,
                        exc_info=True,
                    )
                    error_result = PhaseResult(
                        phase=phase.name,
                        status="error",
                        error=str(exc),
                    )
                    cycle.phase_results.append(error_result)
                    self._events.phase_failed(
                        group_id,
                        cycle_id=cycle.id,
                        result=error_result,
                        phase_plan=phase_plan,
                    )

            if cycle.status != "cancelled":
                cycle.status = "completed"

        except Exception as exc:
            cycle.status = "failed"
            cycle.error = str(exc)
            logger.error("Consolidation cycle failed: %s", exc, exc_info=True)

        finally:
            self._running = False
            await self._completion.complete_cycle(
                group_id=group_id,
                cycle=cycle,
                context=context,
            )

        return cycle
