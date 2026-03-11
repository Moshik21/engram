"""Consolidation engine: orchestrates phases and persists audit trail."""

from __future__ import annotations

import logging
import time

from engram.config import ActivationConfig
from engram.consolidation.calibration import (
    build_calibration_snapshots,
    build_distillation_examples,
)
from engram.consolidation.phases.compact import AccessHistoryCompactionPhase
from engram.consolidation.phases.dream import DreamSpreadingPhase
from engram.consolidation.phases.edge_adjudication import EdgeAdjudicationPhase
from engram.consolidation.phases.evidence_adjudication import EvidenceAdjudicationPhase
from engram.consolidation.phases.graph_embed import GraphEmbedPhase
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.consolidation.phases.maturation import MaturationPhase
from engram.consolidation.phases.merge import EntityMergePhase
from engram.consolidation.phases.microglia import MicrogliaPhase
from engram.consolidation.phases.prune import PrunePhase
from engram.consolidation.phases.reindex import ReindexPhase
from engram.consolidation.phases.replay import EpisodeReplayPhase
from engram.consolidation.phases.schema_formation import SchemaFormationPhase
from engram.consolidation.phases.semantic_transition import SemanticTransitionPhase
from engram.consolidation.phases.triage import TriagePhase
from engram.events.bus import EventBus
from engram.models.consolidation import (
    ConsolidationCycle,
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    DreamAssociationRecord,
    DreamRecord,
    EvidenceAdjudicationRecord,
    GraphEmbedRecord,
    IdentifierReviewRecord,
    InferredEdge,
    MaturationRecord,
    MergeRecord,
    MicrogliaRecord,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    SchemaRecord,
    SemanticTransitionRecord,
    TriageRecord,
)
from engram.storage.protocols import ConsolidationStore

logger = logging.getLogger(__name__)

_AUDIT_PERSISTORS: tuple[tuple[type, str], ...] = (
    (MergeRecord, "save_merge_record"),
    (IdentifierReviewRecord, "save_identifier_review_record"),
    (InferredEdge, "save_inferred_edge"),
    (PruneRecord, "save_prune_record"),
    (ReindexRecord, "save_reindex_record"),
    (ReplayRecord, "save_replay_record"),
    (DreamAssociationRecord, "save_dream_association_record"),
    (DreamRecord, "save_dream_record"),
    (TriageRecord, "save_triage_record"),
    (GraphEmbedRecord, "save_graph_embed_record"),
    (MaturationRecord, "save_maturation_record"),
    (SemanticTransitionRecord, "save_semantic_transition_record"),
    (SchemaRecord, "save_schema_record"),
    (MicrogliaRecord, "save_microglia_record"),
    (EvidenceAdjudicationRecord, "save_evidence_adjudication_record"),
    (DecisionTrace, "save_decision_trace"),
    (DecisionOutcomeLabel, "save_decision_outcome_label"),
)


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
        self._event_bus = event_bus
        self._running = False
        self._cancelled = False

        self._phases = [
            TriagePhase(graph_manager=graph_manager),
            EntityMergePhase(llm_client=llm_client),
            EdgeInferencePhase(llm_client=llm_client, escalation_client=llm_client),
            EvidenceAdjudicationPhase(graph_manager=graph_manager),
            EdgeAdjudicationPhase(graph_manager=graph_manager),
            EpisodeReplayPhase(extractor=extractor),
            PrunePhase(),
            AccessHistoryCompactionPhase(),
            MaturationPhase(),
            SemanticTransitionPhase(),
            SchemaFormationPhase(),
            ReindexPhase(),
            GraphEmbedPhase(),
            MicrogliaPhase(),
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

        context = CycleContext(trigger=trigger)

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
            selected_phases = [
                phase for phase in self._phases
                if phase_names is None or phase.name in phase_names
            ]
            self._validate_phase_capabilities(selected_phases)

            for phase in self._phases:
                if self._cancelled:
                    cycle.status = "cancelled"
                    break

                # Skip phases not in the requested set (tiered scheduling)
                if phase_names is not None and phase.name not in phase_names:
                    continue

                self._publish(
                    group_id,
                    f"consolidation.phase.{phase.name}.started",
                    {
                        "cycle_id": cycle.id,
                        "phase": phase.name,
                    },
                )

                try:
                    trace_start = len(context.decision_traces)
                    outcome_start = len(context.decision_outcome_labels)
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
                            await self._persist_record(record)
                        for trace in context.decision_traces[trace_start:]:
                            await self._persist_record(trace)
                        for label in context.decision_outcome_labels[outcome_start:]:
                            await self._persist_record(label)

                    # Notify dashboard of removed nodes (prune/merge)
                    removed_ids: list[str] = []
                    for record in records:
                        if isinstance(record, PruneRecord):
                            removed_ids.append(record.entity_id)
                        elif isinstance(record, MergeRecord):
                            removed_ids.append(record.remove_id)
                    if removed_ids and not dry_run:
                        self._publish(
                            group_id,
                            "graph.delta",
                            {"nodesRemoved": removed_ids},
                        )

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
                try:
                    await self._run_post_cycle_analysis(cycle, context)
                except Exception:
                    logger.exception(
                        "Post-cycle distillation/calibration failed for cycle %s",
                        cycle.id,
                    )

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

    def _validate_phase_capabilities(self, phases: list) -> None:
        for phase in phases:
            self._validate_capability_group(
                phase.name,
                "graph_store",
                self._graph,
                phase.required_graph_store_methods(self._cfg),
            )
            self._validate_capability_group(
                phase.name,
                "activation_store",
                self._activation,
                phase.required_activation_store_methods(self._cfg),
            )
            self._validate_capability_group(
                phase.name,
                "search_index",
                self._search,
                phase.required_search_index_methods(self._cfg),
            )

    @staticmethod
    def _validate_capability_group(
        phase_name: str,
        target_name: str,
        target: object,
        required_methods: set[str],
    ) -> None:
        if not required_methods:
            return
        missing = sorted(method for method in required_methods if not hasattr(target, method))
        if missing:
            raise RuntimeError(
                f"Phase '{phase_name}' requires {target_name} methods: {', '.join(missing)}"
            )

    async def _persist_record(self, record: object) -> None:
        if self._store is None:
            return
        for record_type, method_name in _AUDIT_PERSISTORS:
            if isinstance(record, record_type):
                persist = getattr(self._store, method_name, None)
                if persist is None:
                    logger.warning(
                        "Consolidation store %s does not implement %s for %s",
                        type(self._store).__name__,
                        method_name,
                        record_type.__name__,
                    )
                    return
                await persist(record)
                return

    async def _run_post_cycle_analysis(
        self,
        cycle: ConsolidationCycle,
        context: CycleContext,
    ) -> None:
        if self._store is None:
            return

        distillation_examples = []
        if self._cfg.consolidation_distillation_enabled and context.decision_traces:
            distillation_examples = build_distillation_examples(
                cycle.id,
                cycle.group_id,
                context.decision_traces,
                context.decision_outcome_labels,
            )
            for example in distillation_examples:
                await self._store.save_distillation_example(example)

        snapshots = []
        if self._cfg.consolidation_calibration_enabled:
            recent_cycles = await self._store.get_recent_cycles(
                cycle.group_id,
                limit=self._cfg.consolidation_calibration_window_cycles,
            )
            traces: list[DecisionTrace] = []
            labels: list[DecisionOutcomeLabel] = []
            for recent_cycle in recent_cycles:
                if recent_cycle.id == cycle.id:
                    traces.extend(context.decision_traces)
                    labels.extend(context.decision_outcome_labels)
                    continue
                traces.extend(
                    await self._store.get_decision_traces(recent_cycle.id, cycle.group_id)
                )
                labels.extend(
                    await self._store.get_decision_outcome_labels(recent_cycle.id, cycle.group_id)
                )

            if traces:
                snapshots = build_calibration_snapshots(
                    cycle.id,
                    cycle.group_id,
                    traces,
                    labels,
                    window_cycles=len(recent_cycles),
                    min_examples=self._cfg.consolidation_calibration_min_examples,
                    bins=self._cfg.consolidation_calibration_bins,
                )
                for snapshot in snapshots:
                    snapshot.summary.setdefault(
                        "requested_window_cycles",
                        self._cfg.consolidation_calibration_window_cycles,
                    )
                    snapshot.summary.setdefault(
                        "cycles_observed",
                        len(recent_cycles),
                    )
                    await self._store.save_calibration_snapshot(snapshot)

        if distillation_examples or snapshots:
            self._publish(
                cycle.group_id,
                "consolidation.learning.updated",
                {
                    "cycle_id": cycle.id,
                    "distillation_examples": len(distillation_examples),
                    "calibration_snapshots": len(snapshots),
                },
            )

    def _publish(self, group_id: str, event_type: str, payload: dict) -> None:
        """Publish an event if event bus is available."""
        if self._event_bus:
            self._event_bus.publish(group_id, event_type, payload)
