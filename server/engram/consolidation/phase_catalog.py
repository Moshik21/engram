"""Runtime construction for the ordered consolidation phase catalog."""

from __future__ import annotations

from engram.consolidation.phase_registry import validate_consolidation_phase_order
from engram.consolidation.phases.base import ConsolidationPhase
from engram.consolidation.phases.calibrate import CalibratePhase
from engram.consolidation.phases.compact import AccessHistoryCompactionPhase
from engram.consolidation.phases.dream import DreamSpreadingPhase
from engram.consolidation.phases.edge_adjudication import EdgeAdjudicationPhase
from engram.consolidation.phases.evidence_adjudication import EvidenceAdjudicationPhase
from engram.consolidation.phases.graph_embed import GraphEmbedPhase
from engram.consolidation.phases.immunity import ImmunityPhase
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.consolidation.phases.maturation import MaturationPhase
from engram.consolidation.phases.merge import EntityMergePhase
from engram.consolidation.phases.microglia import MicrogliaPhase
from engram.consolidation.phases.prune import PrunePhase
from engram.consolidation.phases.reflect import ObserverReflectPhase
from engram.consolidation.phases.reindex import ReindexPhase
from engram.consolidation.phases.replay import EpisodeReplayPhase
from engram.consolidation.phases.schema_formation import SchemaFormationPhase
from engram.consolidation.phases.semantic_transition import SemanticTransitionPhase
from engram.consolidation.phases.triage import TriagePhase


def build_consolidation_phases(
    *,
    graph_manager: object | None = None,
    extractor: object | None = None,
    llm_client: object | None = None,
) -> list[ConsolidationPhase]:
    """Build the runtime phase list and validate it against the shared registry."""
    phases: list[ConsolidationPhase] = [
        TriagePhase(graph_manager=graph_manager),
        EntityMergePhase(llm_client=llm_client),
        CalibratePhase(),
        EdgeInferencePhase(llm_client=llm_client, escalation_client=llm_client),
        EvidenceAdjudicationPhase(graph_manager=graph_manager),
        EdgeAdjudicationPhase(graph_manager=graph_manager),
        EpisodeReplayPhase(extractor=extractor),
        PrunePhase(),
        AccessHistoryCompactionPhase(),
        MaturationPhase(),
        SemanticTransitionPhase(),
        ObserverReflectPhase(extractor=extractor, llm_client=llm_client),
        SchemaFormationPhase(),
        ReindexPhase(),
        GraphEmbedPhase(),
        MicrogliaPhase(),
        ImmunityPhase(),
        DreamSpreadingPhase(),
    ]
    validate_consolidation_phase_order(phase.name for phase in phases)
    return phases
