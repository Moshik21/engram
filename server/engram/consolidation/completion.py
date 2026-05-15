"""Post-cycle completion orchestration for consolidation cycles."""

from __future__ import annotations

import logging
import time

from engram.consolidation.events import ConsolidationEventPublisher
from engram.consolidation.finalization import ConsolidationFinalizationService
from engram.consolidation.learning import ConsolidationLearningService
from engram.models.consolidation import ConsolidationCycle, CycleContext
from engram.storage.protocols import ConsolidationStore

logger = logging.getLogger(__name__)


class ConsolidationCycleCompletionService:
    """Finalize a cycle after the engine has finished running phases."""

    def __init__(
        self,
        *,
        consolidation_store: ConsolidationStore | None,
        learning_service: ConsolidationLearningService,
        finalization_service: ConsolidationFinalizationService,
        event_publisher: ConsolidationEventPublisher,
    ) -> None:
        self._store = consolidation_store
        self._learning = learning_service
        self._finalization = finalization_service
        self._events = event_publisher

    async def complete_cycle(
        self,
        *,
        group_id: str,
        cycle: ConsolidationCycle,
        context: CycleContext,
    ) -> None:
        """Persist final cycle state, run post-cycle work, and publish completion."""
        finalization_payload: dict[str, object] | None = None
        cycle.completed_at = time.time()
        cycle.total_duration_ms = round(
            (cycle.completed_at - cycle.started_at) * 1000,
            1,
        )

        if self._store:
            await self._store.update_cycle(cycle)
            try:
                learning_result = await self._learning.analyze_cycle(cycle, context)
                self._events.learning_updated(
                    cycle.group_id,
                    cycle_id=cycle.id,
                    distillation_examples=learning_result.distillation_examples,
                    calibration_snapshots=learning_result.calibration_snapshots,
                )
            except Exception:
                logger.exception(
                    "Post-cycle distillation/calibration failed for cycle %s",
                    cycle.id,
                )

        if cycle.status == "completed":
            try:
                finalization_result = await self._finalization.refresh_after_cycle(group_id)
                finalization_payload = finalization_result.event_payload()
            except Exception:
                logger.warning(
                    "Pinned context refresh failed after cycle %s",
                    cycle.id,
                    exc_info=True,
                )

        self._events.cycle_completed(
            group_id,
            cycle,
            finalization=finalization_payload,
        )
