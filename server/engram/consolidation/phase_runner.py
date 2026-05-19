"""Per-phase execution boundary for consolidation cycles."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.phases.base import ConsolidationPhase
from engram.models.consolidation import (
    CalibrationRecord,
    CycleContext,
    DecisionOutcomeLabel,
    DecisionTrace,
    DreamAssociationRecord,
    DreamRecord,
    EvidenceAdjudicationRecord,
    GraphEmbedRecord,
    IdentifierReviewRecord,
    ImmunityRecord,
    InferredEdge,
    MaturationRecord,
    MergeRecord,
    MicrogliaRecord,
    PhaseResult,
    PruneRecord,
    ReindexRecord,
    ReplayRecord,
    SchemaRecord,
    SemanticTransitionRecord,
    TriageRecord,
)
from engram.storage.protocols import ConsolidationStore

logger = logging.getLogger(__name__)

_AUDIT_PERSISTORS: tuple[tuple[type[Any], str], ...] = (
    (MergeRecord, "save_merge_record"),
    (IdentifierReviewRecord, "save_identifier_review_record"),
    (InferredEdge, "save_inferred_edge"),
    (PruneRecord, "save_prune_record"),
    (ReindexRecord, "save_reindex_record"),
    (ReplayRecord, "save_replay_record"),
    (DreamAssociationRecord, "save_dream_association_record"),
    (DreamRecord, "save_dream_record"),
    (ImmunityRecord, "save_immunity_record"),
    (TriageRecord, "save_triage_record"),
    (GraphEmbedRecord, "save_graph_embed_record"),
    (MaturationRecord, "save_maturation_record"),
    (SemanticTransitionRecord, "save_semantic_transition_record"),
    (SchemaRecord, "save_schema_record"),
    (MicrogliaRecord, "save_microglia_record"),
    (EvidenceAdjudicationRecord, "save_evidence_adjudication_record"),
    (CalibrationRecord, "save_calibration_record"),
    (DecisionTrace, "save_decision_trace"),
    (DecisionOutcomeLabel, "save_decision_outcome_label"),
)


@dataclass(frozen=True)
class ConsolidationPhaseRunOutcome:
    """Result of executing and auditing one consolidation phase."""

    result: PhaseResult
    records: tuple[object, ...]
    removed_node_ids: tuple[str, ...]


class ConsolidationPhaseRunner:
    """Execute one phase and persist its audit records."""

    def __init__(
        self,
        graph_store,
        activation_store,
        search_index,
        cfg: ActivationConfig,
        consolidation_store: ConsolidationStore | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._store = consolidation_store

    async def run_phase(
        self,
        phase: ConsolidationPhase,
        *,
        group_id: str,
        cycle_id: str,
        dry_run: bool,
        context: CycleContext,
    ) -> ConsolidationPhaseRunOutcome:
        """Execute a phase and persist its direct and context audit records."""
        trace_start = len(context.decision_traces)
        outcome_start = len(context.decision_outcome_labels)
        result, records = await phase.execute(
            group_id=group_id,
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            cycle_id=cycle_id,
            dry_run=dry_run,
            context=context,
        )

        record_tuple = tuple(records)
        if self._store:
            for record in record_tuple:
                await self._persist_record(record)
            for trace in context.decision_traces[trace_start:]:
                await self._persist_record(trace)
            for label in context.decision_outcome_labels[outcome_start:]:
                await self._persist_record(label)

        return ConsolidationPhaseRunOutcome(
            result=result,
            records=record_tuple,
            removed_node_ids=_removed_node_ids(record_tuple),
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


def _removed_node_ids(records: tuple[object, ...]) -> tuple[str, ...]:
    removed_ids: list[str] = []
    for record in records:
        if isinstance(record, PruneRecord):
            removed_ids.append(record.entity_id)
        elif isinstance(record, MergeRecord):
            removed_ids.append(record.remove_id)
        elif isinstance(record, ImmunityRecord):
            if record.decision == "pruned":
                removed_ids.append(record.node_id)
    return tuple(removed_ids)
