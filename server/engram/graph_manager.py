"""Graph Manager — orchestrates extraction, entity resolution, and storage."""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from engram.benchmark_loader import BenchmarkLoadService
from engram.config import ActivationConfig, NerveCenterConfig
from engram.consolidation_trigger import ConsolidationTriggerResult, ConsolidationTriggerService
from engram.events.bus import EventBus
from engram.extraction.ambiguity import AmbiguityAnalyzer, AmbiguityGroup
from engram.extraction.apply import (
    ApplyEngine,
    apply_relationship_fact,
    merge_entity_attributes,
    resolve_relationship_temporals,
)
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.client_proposals import proposals_to_evidence
from engram.extraction.commit_policy import AdaptiveCommitPolicy, CommitThresholds
from engram.extraction.evidence import (
    CommitDecision,
    EvidenceBundle,
    EvidenceCandidate,
)
from engram.extraction.evidence_bridge import EvidenceBridge
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.extraction.models import ApplyOutcome, ProjectionBundle
from engram.extraction.narrow.pipeline import NarrowExtractionPipeline
from engram.extraction.narrow_adapter import NarrowExtractorAdapter
from engram.extraction.ollama_extractor import OllamaExtractor
from engram.extraction.planner import ProjectionPlanner
from engram.extraction.policy import ProjectionPolicy
from engram.extraction.post_apply import ProjectionPostProcessor
from engram.extraction.projector import EpisodeProjector
from engram.ingestion.adjudication_service import (
    AdjudicationResolutionOutcome,
    EvidenceAdjudicationService,
    EvidenceMaterializationOutcome,
)
from engram.ingestion.adjudication_service import (
    EvidenceMaterializationFailure as EvidenceMaterializationFailure,
)
from engram.ingestion.capture_service import EpisodeCaptureService
from engram.ingestion.decision_materializer import DecisionMaterializer
from engram.ingestion.entity_indexer import StructureAwareEntityIndexer
from engram.ingestion.episode_ingestion import EpisodeIngestionService
from engram.ingestion.project_bootstrap import ProjectBootstrapService
from engram.ingestion.projection_execution import (
    EvidenceProjectionExecutor,
    LegacyProjectionExecutor,
    ProjectionLifecycleResult,
)
from engram.ingestion.projection_service import EpisodeProjectionService
from engram.ingestion.projection_service import ProjectionError as ProjectionError
from engram.ingestion.projection_state import sync_projection_state
from engram.ingestion.worker_runtime import EpisodeWorkerRuntimeStores
from engram.models.consolidation import RelationshipApplyResult
from engram.models.entity import Entity
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.epistemic import ArtifactHit, EpistemicBundle, EvidenceClaim
from engram.models.relationship import Relationship
from engram.public_surface_policy import (
    ChatRuntimePolicy,
    ChatToolRecallPolicy,
    ExplicitRecallPacketPolicy,
    PublicSurfacePolicyService,
)
from engram.retrieval.artifacts import ArtifactSearchService
from engram.retrieval.budgets import (
    RecallBudget,
    budget_profile_for_source,
    recall_budget_for_profile,
    surface_for_source,
)
from engram.retrieval.confidence import RecallConfidenceApplier
from engram.retrieval.context import (
    ConversationRuntimeService,
    RecallConversationFingerprintRecorder,
)
from engram.retrieval.context_builder import MemoryContextBuilder
from engram.retrieval.control import RecallNeedController, RecallNeedThresholds
from engram.retrieval.entity_mutation import EntityMutationService
from engram.retrieval.entity_probe import EntityProbeRecallService
from engram.retrieval.episode_traversal import RecallEpisodeTraversal
from engram.retrieval.epistemic import EpistemicRoutingController
from engram.retrieval.epistemic_evidence import EpistemicEvidenceService
from engram.retrieval.epistemic_route import EpistemicRouteService
from engram.retrieval.feedback import (
    RecallCueFeedbackRecorder,
    RecallEntityAccessRecorder,
    RecallInteractionRecorder,
    RecallMemoryInteractionApplier,
)
from engram.retrieval.forgetting import MemoryForgettingService
from engram.retrieval.graph_state import GraphStateService
from engram.retrieval.identity_core import IdentityCoreService
from engram.retrieval.lookup import EntityFactLookupService
from engram.retrieval.memory_operations import (
    MemoryOperationMetricsCollector,
    MemoryOperationSample,
    memory_operation_sample_from_mapping,
)
from engram.retrieval.near_miss import RecallNearMissBuilder, RecallNearMissMaterializer
from engram.retrieval.packet_cache import MemoryPacketCache, MemoryPacketCacheHit
from engram.retrieval.post_process import RecallPostProcessor
from engram.retrieval.preference_feedback import PreferenceFeedbackRecorder
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.priming import RecallPrimingUpdater
from engram.retrieval.prospective import ProspectiveMemoryService
from engram.retrieval.response_state import RecallResponseStateService
from engram.retrieval.result_builder import RecallResultBuilder
from engram.retrieval.runtime_state import RuntimeStateService
from engram.retrieval.service import RecallService
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

if TYPE_CHECKING:
    from engram.consolidation.audit_reader import ConsolidationAuditReader

logger = logging.getLogger(__name__)


def _extract_message_text(blocks: object) -> str:
    """Join text-bearing Anthropic content blocks without assuming block types."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _session_cache_hit_signal(
    session_cache: dict[str, tuple[float, dict]] | None,
    results: list[dict],
) -> bool | None:
    """Return cache-hit telemetry only when the caller can infer it safely."""
    if session_cache is None or not results:
        return None
    return None


class GraphManager:
    """Orchestrates: extract -> upsert entities -> create relationships -> index -> activate."""

    def __init__(
        self,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        extractor: EntityExtractor,
        cfg: ActivationConfig | None = None,
        event_bus: EventBus | None = None,
        canonicalizer: PredicateCanonicalizer | None = None,
        reranker: object | None = None,
        community_store: object | None = None,
        predicate_cache: object | None = None,
        nerve_center_cfg: NerveCenterConfig | None = None,
        runtime_mode: str | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._extractor = extractor
        self._cfg = cfg or ActivationConfig()
        self._event_bus = event_bus
        self._canonicalizer = canonicalizer or PredicateCanonicalizer()
        self._reranker = reranker
        self._community_store = community_store
        self._predicate_cache = predicate_cache
        self._nerve_center_cfg = nerve_center_cfg or NerveCenterConfig()
        self._runtime_mode = runtime_mode or "unknown"
        self._public_surface_policy_service = PublicSurfacePolicyService(self._cfg)
        self._benchmark_load_service = BenchmarkLoadService(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
        )
        self._projection_planner = ProjectionPlanner(self._cfg)
        self._projection_policy = ProjectionPolicy(self._cfg)
        self._projector = EpisodeProjector(self._extractor)
        self._recall_result_builder = RecallResultBuilder(self._cfg)
        self._recall_response_state_service = RecallResponseStateService()
        self._recall_episode_traversal = RecallEpisodeTraversal(
            graph_store=self._graph,
            cfg=self._cfg,
            result_builder=self._recall_result_builder,
        )
        self._recall_near_miss_builder = RecallNearMissBuilder(self._graph)
        self._recall_priming_updater = RecallPrimingUpdater(
            graph_store=self._graph,
            cfg=self._cfg,
        )
        self._recall_working_memory_updater = RecallWorkingMemoryUpdater()
        self._recall_confidence_applier = RecallConfidenceApplier(
            cfg=self._cfg,
            search_index=self._search,
        )
        self._recall_fingerprint_recorder = RecallConversationFingerprintRecorder(
            cfg=self._cfg,
            search_index=self._search,
        )

        # Briefing cache: (group_id, topic_hint) -> (timestamp, text)
        self._briefing_cache: dict[tuple[str, str | None], tuple[float, str]] = {}

        # Content dedup: track hashes of recently extracted content to skip duplicates
        self._content_hashes: set[str] = set()
        self._content_hashes_inflight: set[str] = set()

        # Conversation context (Wave 2)
        if self._cfg.conv_context_enabled:
            from engram.retrieval.context import ConversationContext

            self._conv_context: ConversationContext | None = ConversationContext(
                alpha=self._cfg.conv_fingerprint_alpha,
            )
        else:
            self._conv_context = None
        self._conversation_runtime_service = ConversationRuntimeService(
            cfg=self._cfg,
            conv_context=self._conv_context,
            search_index=self._search,
        )
        self._last_near_misses: list[dict] = []

        # Surprise cache (Wave 3)
        if self._cfg.surprise_detection_enabled:
            from engram.retrieval.surprise import SurpriseCache

            self._surprise_cache: object | None = SurpriseCache(
                ttl_seconds=self._cfg.surprise_cache_ttl_seconds,
            )
        else:
            self._surprise_cache = None

        # Priming buffer (Wave 3): {entity_id: (boost, expiry_time)}
        self._priming_buffer: dict[str, tuple[float, float]] = {}

        # Prospective memory (Wave 4): triggered intentions consumed on read
        self._triggered_intentions: list = []

        # Goal priming cache (Brain Architecture)
        if self._cfg.goal_priming_enabled:
            from engram.retrieval.goals import GoalPrimingCache

            self._goal_priming_cache: object | None = GoalPrimingCache(
                ttl_seconds=self._cfg.goal_priming_cache_ttl_seconds,
            )
        else:
            self._goal_priming_cache = None

        self._recall_need_controller = RecallNeedController(self._cfg)
        self._memory_operation_metrics = MemoryOperationMetricsCollector()
        self._packet_cache = MemoryPacketCache(
            max_entries=self._cfg.recall_packet_cache_max_entries,
            default_ttl_seconds=self._cfg.recall_packet_cache_ttl_seconds,
            persistence_path=(
                self._cfg.recall_packet_cache_path
                if self._cfg.recall_packet_cache_persistence_enabled
                else None
            ),
        )
        self._recall_interaction_recorder = RecallInteractionRecorder(
            cfg=self._cfg,
            event_bus=self._event_bus,
            recall_need_controller=self._recall_need_controller,
        )
        self._recall_cue_feedback_recorder = RecallCueFeedbackRecorder(
            cfg=self._cfg,
            graph_store=self._graph,
            projection_policy=self._projection_policy,
            recall_need_controller=self._recall_need_controller,
            event_bus=self._event_bus,
        )
        self._recall_near_miss_materializer = RecallNearMissMaterializer(
            near_miss_builder=self._recall_near_miss_builder,
            cue_feedback_recorder=self._recall_cue_feedback_recorder,
        )
        self._epistemic_controller = EpistemicRoutingController(self._cfg)
        self._epistemic_route_service = EpistemicRouteService(
            cfg=self._cfg,
            conv_context=self._conv_context,
            get_graph_probe=self._epistemic_route_graph_probe,
            get_recall_need_thresholds=self._epistemic_route_thresholds,
            record_route=self._epistemic_route_record,
        )
        self._epistemic_evidence_service = EpistemicEvidenceService(
            cfg=self._cfg,
            build_route=self._epistemic_evidence_build_route,
            bootstrap_project=self._epistemic_evidence_bootstrap_project,
            recall=self._epistemic_evidence_recall,
            search_artifacts=self._epistemic_evidence_search_artifacts,
            get_runtime_state=self._epistemic_evidence_runtime_state,
            record_execution=self._epistemic_evidence_record_execution,
        )

        # Reconsolidation tracker (Brain Architecture Phase 2B)
        if self._cfg.reconsolidation_enabled:
            from engram.retrieval.reconsolidation import LabileWindowTracker

            self._labile_tracker: LabileWindowTracker | None = LabileWindowTracker(
                ttl=self._cfg.reconsolidation_window_seconds,
                max_entries=self._cfg.reconsolidation_max_entries,
            )
        else:
            self._labile_tracker = None
        self._recall_access_recorder = RecallEntityAccessRecorder(
            cfg=self._cfg,
            activation_store=self._activation,
            event_bus=self._event_bus,
            labile_tracker=self._labile_tracker,
        )
        self._recall_memory_interaction_applier = RecallMemoryInteractionApplier(
            cfg=self._cfg,
            graph_store=self._graph,
            activation_store=self._activation,
            cue_feedback_recorder=self._recall_cue_feedback_recorder,
            entity_access_recorder=self._recall_access_recorder,
            interaction_recorder=self._recall_interaction_recorder,
            recall_need_controller=self._recall_need_controller,
        )
        self._forgetting_service = MemoryForgettingService(
            graph_store=self._graph,
            activation_store=self._activation,
        )
        self._entity_mutation_service = EntityMutationService(
            graph_store=self._graph,
            activation_store=self._activation,
        )
        self._identity_core_service = IdentityCoreService(graph_store=self._graph)
        self._lookup_service = EntityFactLookupService(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
        )
        self._entity_probe_recall_service = EntityProbeRecallService(
            get_graph_store=lambda: self._graph,
            get_search_index=lambda: self._search,
            resolve_entity_name=self._entity_probe_resolve_entity_name,
        )
        self._preference_feedback_recorder = PreferenceFeedbackRecorder(
            graph_store=self._graph,
            cfg=self._cfg,
            event_bus=self._event_bus,
        )
        self._prospective_memory_service = ProspectiveMemoryService(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            publish_event=self._publish,
        )
        self._recall_primary_result_materializer = RecallPrimaryResultMaterializer(
            graph_store=self._graph,
            result_builder=self._recall_result_builder,
            cue_feedback_recorder=self._recall_cue_feedback_recorder,
            entity_access_recorder=self._recall_access_recorder,
            interaction_recorder=self._recall_interaction_recorder,
            working_memory_updater=self._recall_working_memory_updater,
        )
        self._recall_post_processor = RecallPostProcessor(
            episode_traversal=self._recall_episode_traversal,
            working_memory_updater=self._recall_working_memory_updater,
            priming_updater=self._recall_priming_updater,
            near_miss_materializer=self._recall_near_miss_materializer,
            confidence_applier=self._recall_confidence_applier,
            fingerprint_recorder=self._recall_fingerprint_recorder,
        )
        self._recall_service = RecallService(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            primary_materializer=self._recall_primary_result_materializer,
            post_processor=self._recall_post_processor,
            reranker=self._reranker,
            community_store=self._community_store,
            predicate_cache=self._predicate_cache,
        )
        self._context_builder = MemoryContextBuilder(
            graph_store=self._graph,
            activation_store=self._activation,
            cfg=self._cfg,
            recall=self._context_recall,
            list_intentions=self._context_list_intentions,
            resolve_entity_name=self._context_resolve_entity_name,
            publish_access_event=self._context_publish_access_event,
            briefing_cache=self._briefing_cache,
            get_cached_packets=self.get_cached_memory_packets,
            cache_packets=self.cache_memory_packets,
        )
        self._graph_state_service = GraphStateService(
            graph_store=self._graph,
            activation_store=self._activation,
            cfg=self._cfg,
            get_recall_metrics=self._graph_state_recall_metrics,
            get_memory_operation_metrics=self._graph_state_memory_operation_metrics,
            get_epistemic_metrics=self._graph_state_epistemic_metrics,
            resolve_entity_name=self._graph_state_resolve_entity_name,
        )
        from engram.lifecycle_summary import LifecycleSummaryService

        self._lifecycle_summary_service = LifecycleSummaryService(
            manager=self,
            activation_config=self._cfg,
        )
        self._artifact_search_service = ArtifactSearchService(
            graph_store=self._graph,
            search_index=self._search,
            cfg=self._cfg,
            bootstrap_project=self._artifact_search_bootstrap_project,
        )
        self._runtime_state_service = RuntimeStateService(
            cfg=self._cfg,
            runtime_mode=self._runtime_mode,
            list_project_artifacts=self._runtime_state_list_project_artifacts,
            artifact_is_stale=self._runtime_state_artifact_is_stale,
            get_recall_metrics=self._runtime_state_recall_metrics,
            get_memory_operation_metrics=self._runtime_state_memory_operation_metrics,
            get_epistemic_metrics=self._runtime_state_epistemic_metrics,
            get_packet_cache_summary=self._runtime_state_packet_cache_summary,
        )
        self._consolidation_trigger_service = ConsolidationTriggerService(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            extractor=self._extractor,
            nerve_center_cfg=self._nerve_center_cfg,
        )

        # Working memory buffer
        if self._cfg.working_memory_enabled:
            self._working_memory: WorkingMemoryBuffer | None = WorkingMemoryBuffer(
                capacity=self._cfg.working_memory_capacity,
                ttl_seconds=self._cfg.working_memory_ttl_seconds,
            )
        else:
            self._working_memory = None

        self._apply_engine = ApplyEngine(
            graph_store=self._graph,
            activation_store=self._activation,
            cfg=self._cfg,
            canonicalizer=self._canonicalizer,
            publish_access_event=self._publish_access_event,
            conv_context=self._conv_context,
            labile_tracker=self._labile_tracker,
            event_publisher=self._publish,
        )
        self._post_processor = ProjectionPostProcessor(
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            update_episode_status=self._update_episode_status,
            index_entity_with_structure=self._index_entity_with_structure,
            list_intentions=self.list_intentions,
            update_intention_fire=self._update_intention_fire,
        )
        self._entity_indexer = StructureAwareEntityIndexer(
            graph_store=self._graph,
            search_index=self._search,
            cfg=self._cfg,
        )
        self._decision_materializer = DecisionMaterializer(
            graph_store=self._graph,
            cfg=self._cfg,
            index_entity=self._index_entity_with_structure,
        )
        self._project_bootstrap_service = ProjectBootstrapService(
            graph_store=self._graph,
            activation_store=self._activation,
            cfg=self._cfg,
            publish_event=self._publish,
            store_episode=self._project_bootstrap_store_episode,
            sync_projection_state=self._project_bootstrap_sync_projection_state,
            ensure_relationship=self._project_bootstrap_ensure_relationship,
            index_entity=self._index_entity_with_structure,
            materialize_artifact_decisions=(self._project_bootstrap_materialize_artifact_decisions),
        )
        self._capture_service = EpisodeCaptureService(
            graph_store=self._graph,
            search_index=self._search,
            cfg=self._cfg,
            publish_event=self._publish,
            materialize_decisions=self._materialize_conversation_decisions,
        )

        # Evidence extraction pipeline (v2) — lazy init
        self._evidence_pipeline: NarrowExtractionPipeline | None = None
        self._commit_policy: AdaptiveCommitPolicy | None = None
        self._evidence_bridge: EvidenceBridge | None = None
        self._ambiguity_analyzer: AmbiguityAnalyzer | None = None
        if self._cfg.evidence_extraction_enabled:
            self._evidence_pipeline = NarrowExtractionPipeline(self._cfg)
            self._commit_policy = AdaptiveCommitPolicy(
                thresholds=CommitThresholds(
                    entity=self._cfg.evidence_commit_entity_threshold,
                    relationship=self._cfg.evidence_commit_relationship_threshold,
                    attribute=self._cfg.evidence_commit_attribute_threshold,
                    temporal=self._cfg.evidence_commit_temporal_threshold,
                ),
                adaptive=self._cfg.evidence_adaptive_thresholds,
            )
            self._evidence_bridge = EvidenceBridge()
            if self._cfg.edge_adjudication_enabled:
                self._ambiguity_analyzer = AmbiguityAnalyzer(self._graph)

        self._evidence_adjudication_service = EvidenceAdjudicationService(
            graph_store=self._graph,
            search_index=self._search,
            cfg=self._cfg,
            evidence_bridge=self._evidence_bridge,
            apply_engine=self._apply_engine,
            apply_bootstrap_part_of_edges=self._apply_bootstrap_part_of_edges,
            index_entity_with_structure=self._index_entity_with_structure,
            invalidate_briefing_cache=self.invalidate_briefing_cache,
        )
        self._legacy_projection_executor = LegacyProjectionExecutor(
            projector=self._projector,
            apply_engine=self._apply_engine,
            update_episode_status=self._update_episode_status,
            apply_bootstrap_part_of_edges=self._apply_bootstrap_part_of_edges,
        )
        self._evidence_projection_executor = EvidenceProjectionExecutor(
            graph_store=self._graph,
            cfg=self._cfg,
            build_evidence_bundle=self._build_evidence_bundle,
            build_adjudication_requests=(
                self._evidence_adjudication_service.build_adjudication_requests
            ),
            serialize_candidate_records=(
                self._evidence_adjudication_service.serialize_candidate_records
            ),
            serialize_evidence_records=(
                self._evidence_adjudication_service.serialize_evidence_records
            ),
            materialize_evidence=self._evidence_adjudication_service.materialize_evidence,
            apply_committed_ids=self._evidence_adjudication_service.apply_committed_ids,
            update_episode_status=self._update_episode_status,
            ambiguity_analyzer=self._ambiguity_analyzer,
            commit_policy=self._commit_policy,
        )
        self._projection_service = EpisodeProjectionService(
            graph_store=self._graph,
            cfg=self._cfg,
            projection_planner=self._projection_planner,
            evidence_projection_executor=self._evidence_projection_executor,
            legacy_projection_executor=self._legacy_projection_executor,
            content_hashes=self._content_hashes,
            content_hashes_inflight=self._content_hashes_inflight,
            update_episode_status=self._update_episode_status,
            sync_projection_state=self._sync_projection_state,
            get_episode_cue=self._get_episode_cue,
            publish_event=self._publish,
            should_use_evidence_pipeline=self._should_use_evidence_pipeline,
            run_surprise_detection=self._run_surprise_detection,
            run_prospective_memory=self._run_prospective_memory,
            publish_projection_graph_changes=self._publish_projection_graph_changes,
            index_projected_bundle=self._index_projected_bundle,
            store_emotional_encoding_context=self._store_emotional_encoding_context,
            invalidate_briefing_cache=self.invalidate_briefing_cache,
        )
        self._episode_ingestion_service = EpisodeIngestionService(
            store_episode=self._episode_ingestion_store_episode,
            project_episode=self._episode_ingestion_project_episode,
        )

    def _should_use_evidence_pipeline(
        self,
        *,
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
    ) -> bool:
        """Use v2 when the active source of structure is evidence-shaped.

        Tests and showcase baselines often inject deterministic/mock extractors
        with curated outputs that still rely on the legacy projection path.
        """
        has_client_proposals = bool(proposed_entities or proposed_relationships)
        extractor_supports_v2 = type(self._extractor) is EntityExtractor or isinstance(
            self._extractor, (NarrowExtractorAdapter, OllamaExtractor)
        )
        if not extractor_supports_v2:
            canned_result = getattr(self._extractor, "_result", None)
            if (
                isinstance(canned_result, ExtractionResult)
                and not canned_result.entities
                and not canned_result.relationships
            ):
                extractor_supports_v2 = True
        return (
            self._cfg.evidence_extraction_enabled
            and self._evidence_pipeline is not None
            and self._commit_policy is not None
            and self._evidence_bridge is not None
            and (extractor_supports_v2 or has_client_proposals)
        )

    def _build_evidence_bundle(
        self,
        *,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
    ) -> EvidenceBundle:
        """Resolve the active evidence source for a projection."""
        if self._cfg.evidence_client_proposals_enabled:
            proposal_candidates = proposals_to_evidence(
                proposed_entities,
                proposed_relationships,
                episode_id,
                group_id,
                model_tier,
            )
            if proposal_candidates:
                return EvidenceBundle(
                    episode_id=episode_id,
                    group_id=group_id,
                    candidates=proposal_candidates,
                    extractor_stats={
                        "client_proposals": {
                            "count": len(proposal_candidates),
                            "duration_ms": 0.0,
                        },
                    },
                    total_ms=0.0,
                )
        if self._evidence_pipeline is None:
            return EvidenceBundle(episode_id=episode_id, group_id=group_id)
        return self._evidence_pipeline.extract(
            text=text,
            episode_id=episode_id,
            group_id=group_id,
            cue=cue,
        )

    @staticmethod
    def _serialize_evidence_records(
        evidence_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        *,
        status: str,
        commit_reason: str | None = None,
    ) -> list[dict]:
        """Compatibility wrapper for evidence storage-row serialization."""
        return EvidenceAdjudicationService.serialize_evidence_records(
            evidence_pairs,
            status=status,
            commit_reason=commit_reason,
        )

    @staticmethod
    def _serialize_candidate_records(
        candidates: list[EvidenceCandidate],
        *,
        status: str,
        commit_reason: str | None = None,
    ) -> list[dict]:
        """Compatibility wrapper for evidence candidate storage-row serialization."""
        return EvidenceAdjudicationService.serialize_candidate_records(
            candidates,
            status=status,
            commit_reason=commit_reason,
        )

    @staticmethod
    def _candidate_payload_names(candidate: EvidenceCandidate) -> set[str]:
        names: set[str] = set()
        for key in ("name", "entity", "subject", "object", "nearby_entity"):
            value = candidate.payload.get(key)
            if not value:
                continue
            normalized = str(value).strip().lower()
            if normalized and normalized not in {"user", "none"}:
                names.add(normalized)
        return names

    def _build_adjudication_requests(
        self,
        episode_id: str,
        group_id: str,
        ambiguous_groups: list[AmbiguityGroup],
    ) -> list[object]:
        """Compatibility wrapper for building persisted adjudication work items."""
        return self._evidence_adjudication_service.build_adjudication_requests(
            episode_id,
            group_id,
            ambiguous_groups,
        )

    @staticmethod
    def _adjudication_instructions(tags: list[str]) -> str:
        """Compatibility wrapper for MCP-facing adjudication instructions."""
        return EvidenceAdjudicationService._adjudication_instructions(tags)

    async def get_episode_adjudications(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Compatibility API for open adjudication requests."""
        return await self._evidence_adjudication_service.get_episode_adjudications(
            episode_id,
            group_id,
        )

    @staticmethod
    def _committed_id_map(
        committed_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        *,
        entity_map: dict[str, str],
        claims,
        relationship_results: list[RelationshipApplyResult],
    ) -> dict[str, str]:
        """Compatibility wrapper for evidence-id materialization mapping."""
        return EvidenceAdjudicationService.committed_id_map(
            committed_pairs,
            entity_map=entity_map,
            claims=claims,
            relationship_results=relationship_results,
        )

    @staticmethod
    def _apply_committed_ids(
        evidence_rows: list[dict],
        committed_ids: dict[str, str],
    ) -> tuple[list[dict], list[dict]]:
        """Compatibility wrapper for committed/unresolved evidence splitting."""
        return EvidenceAdjudicationService.apply_committed_ids(
            evidence_rows,
            committed_ids,
        )

    @staticmethod
    def _rehydrate_evidence_pairs(
        evidence_rows: list[dict],
    ) -> list[tuple[EvidenceCandidate, CommitDecision]]:
        """Compatibility wrapper for stored evidence rehydration."""
        return EvidenceAdjudicationService.rehydrate_evidence_pairs(evidence_rows)

    @staticmethod
    def _evidence_projection_bundle(
        episode: Episode,
        entities,
        claims,
        recall_content: str | None = None,
    ) -> ProjectionBundle:
        """Compatibility wrapper for evidence materialization bundles."""
        return EvidenceAdjudicationService.evidence_projection_bundle(
            episode,
            entities,
            claims,
            recall_content=recall_content,
        )

    async def _index_materialized_bundle(
        self,
        *,
        bundle: ProjectionBundle,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> None:
        """Compatibility wrapper for indexing materialized evidence."""
        await self._evidence_adjudication_service.index_materialized_bundle(
            bundle=bundle,
            entity_map=entity_map,
            group_id=group_id,
            episode_id=episode_id,
        )

    async def materialize_evidence(
        self,
        *,
        episode: Episode,
        evidence_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        group_id: str,
        recall_content: str | None = None,
        on_before_relationships=None,
    ) -> EvidenceMaterializationOutcome:
        """Compatibility API for evidence materialization."""
        return await self._evidence_adjudication_service.materialize_evidence(
            episode=episode,
            evidence_pairs=evidence_pairs,
            group_id=group_id,
            recall_content=recall_content,
            on_before_relationships=on_before_relationships,
        )

    async def materialize_stored_evidence(
        self,
        episode_id: str,
        evidence_rows: list[dict],
        *,
        group_id: str = "default",
    ) -> EvidenceMaterializationOutcome:
        """Compatibility API for consolidation evidence materialization."""
        outcome = await self._evidence_adjudication_service.materialize_stored_evidence(
            episode_id,
            evidence_rows,
            group_id=group_id,
        )
        if outcome.materialized:
            self.invalidate_memory_packet_cache(group_id)
        return outcome

    async def submit_adjudication_resolution(
        self,
        request_id: str,
        *,
        entities: list[dict] | None = None,
        relationships: list[dict] | None = None,
        reject_evidence_ids: list[str] | None = None,
        source: str = "client_adjudication",
        model_tier: str = "default",
        rationale: str | None = None,
        group_id: str = "default",
    ) -> AdjudicationResolutionOutcome:
        """Compatibility API for resolving ambiguous evidence work items."""
        outcome = await self._evidence_adjudication_service.submit_adjudication_resolution(
            request_id,
            entities=entities,
            relationships=relationships,
            reject_evidence_ids=reject_evidence_ids,
            source=source,
            model_tier=model_tier,
            rationale=rationale,
            group_id=group_id,
        )
        if outcome.status in {"materialized", "rejected", "expired"}:
            self.invalidate_memory_packet_cache(group_id)
        return outcome

    @staticmethod
    def _is_meta_summary(text: str) -> bool:
        """Check if a summary fragment contains system-internal patterns."""
        from engram.utils.text_guards import is_meta_summary

        return is_meta_summary(text)

    @staticmethod
    def _merge_entity_attributes(
        existing: Entity,
        new_summary: str | None,
        new_pii: bool = False,
        new_pii_categories: list[str] | None = None,
        new_attributes: dict | None = None,
    ) -> dict:
        """Merge new attributes into an existing entity. Returns update dict."""
        return merge_entity_attributes(
            existing,
            new_summary,
            new_pii,
            new_pii_categories,
            new_attributes,
        )

    @staticmethod
    def _resolve_relationship_temporals(
        rel_data: dict,
        dt_now: datetime,
    ) -> tuple[datetime, datetime | None, float]:
        """Resolve temporal fields for an extracted relationship."""
        return resolve_relationship_temporals(rel_data, dt_now)

    @staticmethod
    async def _apply_relationship_fact(
        graph_store: GraphStore,
        canonicalizer: PredicateCanonicalizer,
        cfg: ActivationConfig,
        rel_data: dict,
        entity_map: dict[str, str],
        group_id: str,
        source_episode: str,
    ) -> RelationshipApplyResult:
        return await apply_relationship_fact(
            graph_store=graph_store,
            canonicalizer=canonicalizer,
            cfg=cfg,
            rel_data=rel_data,
            entity_map=entity_map,
            group_id=group_id,
            source_episode=source_episode,
        )

    def _publish(self, group_id: str, event_type: str, payload: dict | None = None) -> None:
        """Publish event if bus is configured."""
        if self._event_bus:
            self._event_bus.publish(group_id, event_type, payload)

    def get_recall_need_thresholds(self, group_id: str = "default") -> RecallNeedThresholds:
        """Return active recall-need thresholds for a group."""
        return self._recall_need_controller.get_thresholds(group_id)

    def record_memory_need_analysis(self, group_id: str, need) -> None:
        """Track a memory-need analyzer decision in the runtime controller."""
        self._recall_need_controller.record_analysis(group_id, need)

    def get_recall_metrics(self, group_id: str = "default") -> dict:
        """Return rolling recall metrics for stats surfaces."""
        return self._recall_need_controller.snapshot(group_id)

    def get_recall_feedback_summary(
        self,
        group_id: str = "default",
        memory_ids: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return compact per-memory feedback for packet trust summaries."""
        return self._recall_need_controller.memory_feedback_summary(
            group_id,
            list(memory_ids or []),
        )

    def record_memory_operation(
        self,
        group_id: str,
        sample: MemoryOperationSample | Mapping[str, Any],
    ) -> None:
        """Track a measured memory operation for value/latency reporting."""
        payload = (
            sample
            if isinstance(sample, MemoryOperationSample)
            else memory_operation_sample_from_mapping(sample)
        )
        self._memory_operation_metrics.record(group_id, payload)

    def get_memory_operation_metrics(self, group_id: str = "default") -> dict:
        """Return rolling memory operation cost metrics for stats surfaces."""
        return self._memory_operation_metrics.snapshot(group_id)

    def get_cached_memory_packets(
        self,
        group_id: str,
        *,
        scope: str,
        topic_hint: str | None = None,
        project_path: str | None = None,
    ) -> MemoryPacketCacheHit | None:
        """Return fresh cached packet payloads for a public recall surface."""
        if not self._cfg.recall_packet_cache_enabled:
            return None
        return self._packet_cache.get(
            group_id=group_id,
            scope=scope,
            topic_hint=topic_hint,
            project_path=project_path,
        )

    def cache_memory_packets(
        self,
        group_id: str,
        *,
        scope: str,
        packets: Sequence[Mapping[str, Any]],
        topic_hint: str | None = None,
        project_path: str | None = None,
        build_duration_ms: float = 0.0,
    ) -> dict:
        """Store serialized packet payloads in the runtime packet cache."""
        if not self._cfg.recall_packet_cache_enabled or not packets:
            return {}
        entry = self._packet_cache.put(
            group_id=group_id,
            scope=scope,
            packets=packets,
            topic_hint=topic_hint,
            project_path=project_path,
            ttl_seconds=self._cfg.recall_packet_cache_ttl_seconds,
            build_duration_ms=build_duration_ms,
        )
        return entry.to_dict()

    def invalidate_memory_packet_cache(
        self,
        group_id: str | None = None,
        *,
        entity_ids: Sequence[str] | None = None,
        episode_ids: Sequence[str] | None = None,
        relationship_ids: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> int:
        """Invalidate cached packets after graph or episode mutations."""
        return self._packet_cache.invalidate(
            group_id=group_id,
            entity_ids=entity_ids,
            episode_ids=episode_ids,
            relationship_ids=relationship_ids,
            scopes=scopes,
        )

    def clear_memory_packet_cache(self, group_id: str | None = None) -> int:
        """Clear packet cache entries for an operator/debug surface."""
        return self._packet_cache.clear(group_id=group_id)

    def get_memory_packet_cache_summary(self, group_id: str | None = None) -> dict:
        """Return packet cache health for runtime diagnostics."""
        return self._packet_cache.summary(group_id=group_id)

    def _memory_operation_budget(
        self,
        source: str,
        *,
        mode: str | None = None,
        max_results: int | None = None,
        max_packets: int | None = None,
        max_output_tokens: int | None = None,
    ) -> RecallBudget:
        """Resolve the budget profile attached to a memory operation."""
        return recall_budget_for_profile(
            self._cfg,
            budget_profile_for_source(source),
            surface=surface_for_source(source),
            mode=mode,
            max_results=max_results,
            max_packets=max_packets,
            max_output_tokens=max_output_tokens,
        )

    @staticmethod
    def _budget_fields(
        budget: RecallBudget,
        duration_ms: float,
    ) -> dict[str, Any]:
        """Return common budget telemetry fields for operation samples."""
        budget_miss = budget.exceeded(duration_ms)
        return {
            "budget_ms": budget.budget_ms,
            "budget_tokens": budget.budget_tokens,
            "budget_miss": budget_miss,
            "degraded": bool(budget.timeout_degrades and budget_miss),
        }

    def get_epistemic_metrics(self, group_id: str = "default") -> dict:
        """Return rolling epistemic routing metrics for stats surfaces."""
        return self._epistemic_controller.snapshot(group_id)

    def get_activation_config(self) -> ActivationConfig:
        """Return the active activation/runtime config for public surfaces."""
        return self._public_surface_policy_service.activation_config()

    def get_memory_need_config(self) -> ActivationConfig:
        """Return the active memory-need analyzer config for public surfaces."""
        return self._public_surface_policy_service.activation_config()

    def recall_need_graph_probe_enabled(self) -> bool:
        """Return whether public memory-need analysis should attach a graph probe."""
        return self._public_surface_policy_service.recall_need_graph_probe_enabled()

    def edge_adjudication_client_enabled(self) -> bool:
        """Return whether public write responses should include adjudication requests."""
        return self._public_surface_policy_service.edge_adjudication_client_enabled()

    def get_explicit_recall_packet_policy(self) -> ExplicitRecallPacketPolicy:
        """Return packet policy for explicit REST recall."""
        return self._public_surface_policy_service.explicit_recall_packet_policy()

    def get_chat_tool_recall_policy(self) -> ChatToolRecallPolicy:
        """Return recall interaction and packet policy for chat tool calls."""
        return self._public_surface_policy_service.chat_tool_recall_policy()

    def recall_usage_feedback_enabled(self) -> bool:
        """Return whether post-response recall usage feedback is enabled."""
        return self._public_surface_policy_service.recall_usage_feedback_enabled()

    def recall_need_post_response_safety_net_enabled(self) -> bool:
        """Return whether generic memory-free chat answers should retry once."""
        return self._public_surface_policy_service.recall_need_post_response_safety_net_enabled()

    def get_chat_runtime_policy(self) -> ChatRuntimePolicy:
        """Return route-facing policy for the knowledge chat endpoint."""
        return self._public_surface_policy_service.chat_runtime_policy()

    async def load_benchmark_corpus(
        self,
        *,
        group_id: str,
        seed: int,
        structure_aware: bool = False,
    ) -> dict:
        """Load a benchmark corpus into the active runtime stores."""
        return await self._benchmark_load_service.load_benchmark(
            group_id=group_id,
            seed=seed,
            structure_aware=structure_aware,
        )

    def get_conversation_context(self):
        """Return the active live conversation context, when enabled."""
        return self._conversation_runtime_service.get_context()

    def get_conversation_embed_fn(self):
        """Return the embedding function for live conversation turns, when available."""
        return self._conversation_runtime_service.get_embed_fn()

    def get_conversation_turn_count(self) -> int:
        """Return the number of live turns recorded in the active conversation context."""
        return self._conversation_runtime_service.get_turn_count()

    def get_conversation_top_entity_names(self, limit: int | None = None) -> list[str]:
        """Return session entity names tracked in the active conversation context."""
        return self._conversation_runtime_service.get_top_entity_names(limit)

    def get_conversation_recent_turns(self, limit: int | None = None) -> list[str]:
        """Return recent live turn text from the active conversation context."""
        return self._conversation_runtime_service.get_recent_turns(limit)

    async def ingest_conversation_turn(
        self,
        text: str,
        *,
        source: str,
        update_fingerprint: bool | None = None,
    ) -> None:
        """Record a live conversation turn through the runtime conversation service."""
        await self._conversation_runtime_service.ingest_turn(
            text,
            source=source,
            update_fingerprint=update_fingerprint,
        )

    async def _publish_access_event(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        await self._recall_access_recorder.publish_access_event(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            group_id=group_id,
            accessed_via=accessed_via,
        )

    async def _record_entity_access(
        self,
        entity: Entity,
        *,
        group_id: str,
        query: str,
        source: str,
        timestamp: float | None = None,
    ) -> None:
        """Record a true access event and open a reconsolidation window."""
        await self._recall_access_recorder.record_entity_access(
            entity,
            group_id=group_id,
            query=query,
            source=source,
            timestamp=timestamp,
        )

    async def apply_memory_interaction(
        self,
        memory_ids: list[str],
        *,
        interaction_type: str,
        group_id: str = "default",
        query: str = "",
        source: str = "recall_feedback",
        result_lookup: dict[str, dict] | None = None,
    ) -> None:
        """Apply post-recall interaction semantics to a set of entities."""
        if not memory_ids:
            return
        await self._recall_memory_interaction_applier.apply(
            memory_ids,
            interaction_type=interaction_type,
            group_id=group_id,
            query=query,
            source=source,
            result_lookup=result_lookup,
        )
        self.invalidate_memory_packet_cache(
            group_id,
            entity_ids=[
                memory_id
                for memory_id in memory_ids
                if memory_id and not memory_id.startswith(("cue:", "episode:"))
            ],
            episode_ids=[
                memory_id.split(":", 1)[1]
                for memory_id in memory_ids
                if memory_id.startswith(("cue:", "episode:")) and ":" in memory_id
            ],
        )

    async def _update_episode_status(
        self, episode_id: str, status: EpisodeStatus, group_id: str = "default", **extra: object
    ) -> None:
        """Update episode status and updated_at timestamp."""
        updates: dict = {"status": status.value}
        updates.update(extra)
        await self._graph.update_episode(episode_id, updates, group_id=group_id)

    async def _sync_projection_state(
        self,
        episode_id: str,
        state: EpisodeProjectionState,
        group_id: str = "default",
        *,
        reason: str | None = None,
        last_projected_at: datetime | None = None,
        episode_updates: dict[str, object] | None = None,
        cue_reason: str | None = None,
        cue_updates: dict[str, object] | None = None,
        cue_layer_enabled: bool | None = None,
        sync_cue: bool = True,
    ) -> None:
        await sync_projection_state(
            self._graph,
            episode_id,
            group_id=group_id,
            state=state,
            reason=reason,
            last_projected_at=last_projected_at,
            episode_updates=episode_updates,
            cue_layer_enabled=(
                self._cfg.cue_layer_enabled if cue_layer_enabled is None else cue_layer_enabled
            ),
            cue_reason=cue_reason,
            cue_updates=cue_updates,
            sync_cue=sync_cue,
            log_prefix="GraphManager",
        )

    async def _get_episode_cue(
        self,
        episode_id: str,
        group_id: str,
    ) -> EpisodeCue | None:
        getter = getattr(self._graph, "get_episode_cue", None)
        if getter is None or not callable(getter):
            return None
        cue = await getter(episode_id, group_id)
        return cue if isinstance(cue, EpisodeCue) else None

    @staticmethod
    def _episode_projection_state_value(episode: object | None) -> str | None:
        """Normalize episode projection state to its string value."""
        return RecallResultBuilder.episode_projection_state_value(episode)

    @staticmethod
    def _cue_result_payload(cue: EpisodeCue, *, hit_increment: int = 0) -> dict[str, object]:
        return RecallResultBuilder.cue_result_payload(cue, hit_increment=hit_increment)

    async def _apply_bootstrap_part_of_edges(
        self,
        episode: Episode,
        entity_map: dict[str, str],
        group_id: str,
    ) -> None:
        await self._post_processor.apply_bootstrap_part_of_edges(
            episode,
            entity_map,
            group_id,
        )

    async def _run_surprise_detection(
        self,
        *,
        entity_map: dict[str, str],
        group_id: str,
        now: float,
    ) -> None:
        await self._post_processor.run_surprise_detection(
            entity_map=entity_map,
            group_id=group_id,
            now=now,
            surprise_cache=self._surprise_cache,
        )

    async def _run_prospective_memory(
        self,
        *,
        content: str,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> None:
        trigger_matches = await self._post_processor.run_prospective_memory(
            content=content,
            entity_map=entity_map,
            group_id=group_id,
            episode_id=episode_id,
        )
        if trigger_matches:
            self._triggered_intentions = trigger_matches

    async def _publish_projection_graph_changes(
        self,
        *,
        bundle: ProjectionBundle,
        apply_outcome: ApplyOutcome,
        group_id: str,
        episode_id: str,
    ) -> None:
        await self._post_processor.publish_graph_changes(
            bundle=bundle,
            apply_outcome=apply_outcome,
            group_id=group_id,
            episode_id=episode_id,
            publish_event=self._publish,
        )

    async def _index_projected_bundle(
        self,
        *,
        bundle: ProjectionBundle,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> None:
        await self._post_processor.index_projected_bundle(
            bundle=bundle,
            entity_map=entity_map,
            group_id=group_id,
            episode_id=episode_id,
        )

    async def _store_emotional_encoding_context(
        self,
        *,
        episode_id: str,
        content: str,
        entity_map: dict[str, str],
        group_id: str,
    ) -> None:
        await self._post_processor.store_emotional_encoding_context(
            episode_id=episode_id,
            content=content,
            entity_map=entity_map,
            group_id=group_id,
        )

    async def _record_cue_hit(
        self,
        episode: Episode,
        score: float,
        query: str,
        *,
        interaction_type: str | None = None,
        near_miss: bool = False,
        count_hit: bool = True,
    ) -> None:
        """Track cue feedback and promote hot cues into scheduled projection."""
        await self._recall_cue_feedback_recorder.record_cue_feedback(
            episode,
            score,
            query,
            interaction_type=interaction_type,
            near_miss=near_miss,
            count_hit=count_hit,
        )

    async def store_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
        conversation_date: datetime | None = None,
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Store a raw episode without extraction. Fast path for bulk capture.

        Returns the episode ID. The episode is created with QUEUED status.
        Call project_episode() later to run extraction.
        """
        episode_id = await self._capture_service.store_episode(
            content=content,
            group_id=group_id,
            source=source,
            session_id=session_id,
            conversation_date=conversation_date,
            attachments=attachments,
        )
        self.invalidate_memory_packet_cache(
            group_id,
            scopes=["session_prime"],
        )
        return episode_id

    async def project_episode(
        self,
        episode_id: str,
        group_id: str = "default",
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
    ) -> ProjectionLifecycleResult:
        """Run extraction, resolution, and embedding on a stored episode.

        Raises on failure after setting FAILED status.
        """
        result = await self._projection_service.project_episode(
            episode_id,
            group_id=group_id,
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier=model_tier,
        )
        self.invalidate_memory_packet_cache(
            group_id,
            episode_ids=[episode_id],
        )
        if result.outcome == "projected":
            self.invalidate_memory_packet_cache(group_id)
        return result

    async def _episode_ingestion_store_episode(self, *args, **kwargs) -> str:
        """Late-bound store adapter for one-shot ingestion."""
        return await self.store_episode(*args, **kwargs)

    async def _episode_ingestion_project_episode(self, *args, **kwargs) -> object:
        """Late-bound project adapter for one-shot ingestion."""
        return await self.project_episode(*args, **kwargs)

    async def ingest_episode(
        self,
        content: str,
        group_id: str = "default",
        source: str | None = None,
        session_id: str | None = None,
        conversation_date: datetime | None = None,
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
        attachments: list[Attachment] | None = None,
    ) -> str:
        """Ingest a text episode: store, extract, resolve, link.

        Returns the episode ID. Thin wrapper over store_episode + project_episode.
        """
        return await self._episode_ingestion_service.ingest_episode(
            content=content,
            group_id=group_id,
            source=source,
            session_id=session_id,
            conversation_date=conversation_date,
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier=model_tier,
            attachments=attachments,
        )

    # ─── Project Bootstrap ──────────────────────────────────────────

    # Files to bootstrap into the artifact substrate (pattern → max chars)
    _BOOTSTRAP_FILES: list[tuple[str, int]] = ProjectBootstrapService.BOOTSTRAP_FILES

    async def bootstrap_project(
        self,
        project_path: str,
        group_id: str = "default",
        include_patterns: list[str] | None = None,
        session_id: str | None = None,
    ) -> dict:
        """Bootstrap a project: create Project entity and observe key files.

        Idempotent — if the Project entity exists and was bootstrapped
        within the last 24 hours, returns early. Otherwise re-observes
        files to pick up changes (cheap store_episode, no LLM).
        """
        return await self._project_bootstrap_service.bootstrap_project(
            project_path,
            group_id=group_id,
            include_patterns=include_patterns,
            session_id=session_id,
        )

    async def _project_bootstrap_store_episode(self, *args, **kwargs) -> str:
        """Late-bound episode capture adapter for project bootstrap."""
        return await self.store_episode(*args, **kwargs)

    async def _project_bootstrap_sync_projection_state(self, *args, **kwargs) -> None:
        """Late-bound projection-state adapter for project bootstrap."""
        await self._sync_projection_state(*args, **kwargs)

    async def _project_bootstrap_ensure_relationship(self, *args, **kwargs) -> None:
        """Late-bound relationship adapter for project bootstrap."""
        await self._ensure_relationship(*args, **kwargs)

    async def _project_bootstrap_materialize_artifact_decisions(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Late-bound artifact decision adapter for project bootstrap."""
        await self._materialize_artifact_decisions(*args, **kwargs)

    async def _observe_project_files(
        self,
        project_dir: object,  # Path
        project_name: str,
        group_id: str,
        session_id: str | None,
    ) -> list[str]:
        """Read, index, and optionally store bootstrapped project artifacts."""
        return await self._project_bootstrap_service.observe_project_files(
            project_dir,  # type: ignore[arg-type]
            project_name,
            group_id,
            session_id,
        )

    def _iter_bootstrap_files(
        self,
        project_dir: Path,
    ) -> list[tuple[Path, str, int]]:
        """Expand bootstrap patterns into concrete files."""
        return self._project_bootstrap_service.iter_bootstrap_files(project_dir)

    async def _resolve_project_entity_id(
        self,
        project_name: str,
        group_id: str,
    ) -> str | None:
        return await self._project_bootstrap_service.resolve_project_entity_id(
            project_name,
            group_id=group_id,
        )

    async def _upsert_artifact_entity(
        self,
        *,
        project_name: str,
        project_path: str,
        rel_path: str,
        artifact_class: str,
        content: str,
        content_hash: str,
        claims: list[EvidenceClaim],
        group_id: str,
        now_iso: str,
    ) -> tuple[Entity, bool]:
        """Create or update a bootstrapped Artifact entity."""
        return await self._project_bootstrap_service.upsert_artifact_entity(
            project_name=project_name,
            project_path=project_path,
            rel_path=rel_path,
            artifact_class=artifact_class,
            content=content,
            content_hash=content_hash,
            claims=claims,
            group_id=group_id,
            now_iso=now_iso,
        )

    @staticmethod
    def _claim_to_attr(claim: EvidenceClaim) -> dict:
        return ProjectBootstrapService.claim_to_attr(claim)

    @staticmethod
    def _merge_attributes(existing: dict | None, updates: dict) -> dict:
        return ProjectBootstrapService.merge_attributes(existing, updates)

    @staticmethod
    def _artifact_snippet(content: str) -> str:
        return ProjectBootstrapService.artifact_snippet(content)

    @staticmethod
    def _artifact_summary(
        project_name: str,
        rel_path: str,
        content: str,
        claims: list[EvidenceClaim],
    ) -> str:
        return ProjectBootstrapService.artifact_summary(project_name, rel_path, content, claims)

    async def _list_project_artifacts(
        self,
        *,
        group_id: str,
        project_path: str | None = None,
        limit: int = 200,
    ) -> list[Entity]:
        """Compatibility wrapper for artifact list reads."""
        return await self._artifact_search_service.list_project_artifacts(
            group_id=group_id,
            project_path=project_path,
            limit=limit,
        )

    @staticmethod
    def _artifact_is_stale(entity: Entity, stale_seconds: int) -> bool:
        return ArtifactSearchService.artifact_is_stale(entity, stale_seconds)

    @staticmethod
    def _artifact_lexical_score(query: str, entity: Entity) -> float:
        return ArtifactSearchService.artifact_lexical_score(query, entity)

    async def search_artifacts(
        self,
        *,
        query: str,
        group_id: str = "default",
        project_path: str | None = None,
        limit: int = 5,
    ) -> list[ArtifactHit]:
        """Search bootstrapped project artifacts by semantic or name match."""
        return await self._artifact_search_service.search_artifacts(
            query=query,
            group_id=group_id,
            project_path=project_path,
            limit=limit,
        )

    async def _artifact_search_bootstrap_project(self, *args, **kwargs) -> dict:
        """Late-bound project-bootstrap adapter for artifact search."""
        return await self.bootstrap_project(*args, **kwargs)

    async def get_runtime_state(
        self,
        *,
        group_id: str = "default",
        project_path: str | None = None,
    ) -> dict:
        """Return effective runtime/config state plus artifact freshness."""
        return await self._runtime_state_service.get_runtime_state(
            group_id=group_id,
            project_path=project_path,
        )

    async def trigger_consolidation_cycle(
        self,
        *,
        group_id: str = "default",
        trigger: str = "manual",
        dry_run: bool = True,
        consolidation_store: object | None = None,
    ) -> ConsolidationTriggerResult:
        """Compatibility API for ad hoc public consolidation triggers."""
        return await self._consolidation_trigger_service.trigger_consolidation_cycle(
            group_id=group_id,
            trigger=trigger,
            dry_run=dry_run,
            consolidation_store=consolidation_store,
        )

    def get_consolidation_shared_db(self) -> object | None:
        """Return a shared lite SQLite handle for consolidation audit fallback."""
        return self._consolidation_trigger_service.shared_sqlite_db()

    async def _runtime_state_list_project_artifacts(self, *args, **kwargs) -> list[Entity]:
        """Late-bound artifact-list adapter for runtime state."""
        return await self._list_project_artifacts(*args, **kwargs)

    def _runtime_state_artifact_is_stale(self, *args, **kwargs) -> bool:
        """Late-bound artifact-freshness adapter for runtime state."""
        return self._artifact_is_stale(*args, **kwargs)

    def _runtime_state_recall_metrics(self, group_id: str) -> dict:
        """Late-bound recall metrics adapter for runtime state."""
        return self.get_recall_metrics(group_id)

    def _runtime_state_memory_operation_metrics(self, group_id: str) -> dict:
        """Late-bound memory operation metrics adapter for runtime state."""
        return self.get_memory_operation_metrics(group_id)

    def _runtime_state_packet_cache_summary(self, group_id: str) -> dict:
        """Late-bound packet-cache adapter for runtime state."""
        return self.get_memory_packet_cache_summary(group_id)

    def _runtime_state_epistemic_metrics(self, group_id: str) -> dict:
        """Late-bound epistemic metrics adapter for runtime state."""
        return self.get_epistemic_metrics(group_id)

    def _epistemic_route_graph_probe(self):
        """Late-bound graph-probe adapter for epistemic routing."""
        from engram.retrieval.graph_probe import GraphProbe

        graph_probe = getattr(self, "_recall_need_graph_probe", None)
        if not isinstance(graph_probe, GraphProbe):
            graph_probe = GraphProbe(self._graph, self._activation)
            self._recall_need_graph_probe = graph_probe
        return graph_probe

    def get_recall_need_graph_probe(self):
        """Compatibility API for recall-need graph probes."""
        return self._epistemic_route_graph_probe()

    def _epistemic_route_thresholds(self, group_id: str):
        """Late-bound recall-need threshold adapter for epistemic routing."""
        return self.get_recall_need_thresholds(group_id)

    def _epistemic_route_record(
        self,
        group_id: str,
        mode: str,
        operator: str,
        scopes: list[str],
    ) -> None:
        """Late-bound route metrics adapter for epistemic routing."""
        self._epistemic_controller.record_route(
            group_id,
            mode,
            operator=operator,
            scopes=scopes,
        )

    async def _build_epistemic_route(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need=None,
    ):
        """Compatibility wrapper for epistemic question route construction."""
        return await self._epistemic_route_service.build_route(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )

    async def route_question(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need=None,
    ) -> dict:
        """Compatibility API for routed question frame and evidence plan reads."""
        return await self._epistemic_route_service.route_question(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )

    async def _epistemic_evidence_build_route(self, *args, **kwargs):
        """Late-bound route adapter for epistemic evidence gathering."""
        return await self._build_epistemic_route(*args, **kwargs)

    async def _epistemic_evidence_bootstrap_project(self, *args, **kwargs) -> dict:
        """Late-bound project-bootstrap adapter for epistemic evidence gathering."""
        return await self.bootstrap_project(*args, **kwargs)

    async def _epistemic_evidence_recall(self, *args, **kwargs) -> list[dict]:
        """Late-bound recall adapter for epistemic evidence gathering."""
        return await self.recall(*args, **kwargs)

    async def _epistemic_evidence_search_artifacts(self, *args, **kwargs) -> list[ArtifactHit]:
        """Late-bound artifact-search adapter for epistemic evidence gathering."""
        return await self.search_artifacts(*args, **kwargs)

    async def _epistemic_evidence_runtime_state(self, *args, **kwargs) -> dict:
        """Late-bound runtime-state adapter for epistemic evidence gathering."""
        return await self.get_runtime_state(*args, **kwargs)

    def _epistemic_evidence_record_execution(self, *args, **kwargs) -> None:
        """Late-bound execution metrics adapter for epistemic evidence gathering."""
        self._epistemic_controller.record_execution(*args, **kwargs)

    async def gather_epistemic_evidence(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need=None,
    ) -> EpistemicBundle:
        """Compatibility API for route-guided evidence gathering."""
        return await self._epistemic_evidence_service.gather_epistemic_evidence(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )

    @staticmethod
    def _surface_capabilities(surface: str, project_path: str | None) -> dict[str, bool]:
        return EpistemicRouteService.surface_capabilities(surface, project_path)

    @staticmethod
    def _claim_from_attr(claim_data: dict) -> EvidenceClaim:
        return ArtifactSearchService.claim_from_attr(claim_data)

    async def _materialize_artifact_decisions(
        self,
        artifact_entity: Entity,
        claims: list[EvidenceClaim],
        *,
        group_id: str,
    ) -> None:
        """Compatibility wrapper for artifact decision graph materialization."""
        await self._decision_materializer.materialize_artifact_decisions(
            artifact_entity,
            claims,
            group_id=group_id,
        )

    async def _materialize_conversation_decisions(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> None:
        """Compatibility wrapper for conversation decision graph materialization."""
        await self._decision_materializer.materialize_conversation_decisions(
            content,
            episode_id=episode_id,
            group_id=group_id,
        )

    async def _upsert_conversation_artifact(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> Entity:
        """Compatibility wrapper for conversation artifact upsert."""
        return await self._decision_materializer.upsert_conversation_artifact(
            content,
            episode_id=episode_id,
            group_id=group_id,
        )

    async def _upsert_decision_entity(
        self,
        claim: EvidenceClaim,
        *,
        group_id: str,
    ) -> Entity:
        """Compatibility wrapper for decision entity upsert."""
        return await self._decision_materializer.upsert_decision_entity(
            claim,
            group_id=group_id,
        )

    async def _ensure_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        group_id: str,
        source_episode: str | None = None,
    ) -> None:
        """Compatibility wrapper for idempotent relationship creation."""
        await self._decision_materializer.ensure_relationship(
            source_id,
            target_id,
            predicate,
            group_id=group_id,
            source_episode=source_episode,
        )

    @staticmethod
    def _is_decision_claim(claim: EvidenceClaim) -> bool:
        return DecisionMaterializer.is_decision_claim(claim)

    @staticmethod
    def _infer_decision_subject(content: str) -> str | None:
        return DecisionMaterializer.infer_decision_subject(content)

    async def _index_entity_with_structure(
        self,
        entity: Entity,
        group_id: str,
    ) -> None:
        """Build structure-aware embedding text that includes relationship predicates."""
        await self._entity_indexer.index_entity(entity, group_id)

    def _truncate_episode_content(self, ep: Episode, cue=None) -> str:
        """Truncate episode content based on memory tier.

        When tier-aware truncation is disabled, uses flat recall_episode_content_limit.
        When enabled: episodic uses recall_episode_content_limit,
        transitional uses recall_transitional_content_limit,
        semantic uses recall_semantic_content_limit.
        For transitional/semantic with a cue, prefer cue.cue_text over raw content.
        """
        return self._recall_result_builder.truncate_episode_content(ep, cue)

    async def recall(
        self,
        query: str,
        group_id: str = "default",
        limit: int = 10,
        *,
        record_access: bool = True,
        interaction_type: str | None = None,
        interaction_source: str = "recall",
        memory_need=None,
    ) -> list[dict]:
        """Retrieve relevant entities and their context using activation-aware scoring."""
        started = time.perf_counter()
        budget = self._memory_operation_budget(
            interaction_source,
            mode=interaction_source,
            max_results=limit,
        )
        try:
            recall_result = await self._recall_service.recall(
                query=query,
                group_id=group_id,
                limit=limit,
                record_access=record_access,
                interaction_type=interaction_type,
                interaction_source=interaction_source,
                conv_context=self._conv_context,
                working_memory=self._working_memory,
                priming_buffer=self._priming_buffer,
                goal_cache=self._goal_priming_cache,
                memory_need=memory_need,
            )
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 4)
            self.record_memory_operation(
                group_id,
                MemoryOperationSample(
                    operation="recall",
                    source=interaction_source,
                    mode=interaction_source,
                    status="error",
                    duration_ms=duration_ms,
                    **self._budget_fields(budget, duration_ms),
                ),
            )
            raise
        self._last_near_misses = recall_result.near_misses
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        self.record_memory_operation(
            group_id,
            MemoryOperationSample(
                operation="recall",
                source=interaction_source,
                mode=interaction_source,
                status="ok",
                duration_ms=duration_ms,
                result_count=len(recall_result.results),
                **self._budget_fields(budget, duration_ms),
            ),
        )

        return recall_result.results

    def drain_triggered_intention_views(self) -> list[dict]:
        """Return triggered intention views and clear the transient queue."""
        views = self._recall_response_state_service.triggered_intention_views(
            self._triggered_intentions,
        )
        self._triggered_intentions = []
        return views

    def get_last_near_miss_views(self) -> list[dict]:
        """Return latest near-miss payloads from the previous recall."""
        return self._recall_response_state_service.near_miss_views(self._last_near_misses)

    async def get_recall_item_access_count(self, entity_id: str) -> int:
        """Return access count for an entity in public recall presentation."""
        return await self._recall_response_state_service.get_access_count(
            self._activation,
            entity_id,
        )

    def get_surprise_connection_views(
        self,
        group_id: str,
        *,
        now: float,
        limit: int = 3,
    ) -> list[dict]:
        """Return cached surprise-connection views for recall responses."""
        return self._recall_response_state_service.surprise_connection_views(
            self._surprise_cache,
            group_id=group_id,
            now=now,
            limit=limit,
        )

    async def _entity_probe_resolve_entity_name(
        self,
        entity_id: str,
        group_id: str,
    ) -> str:
        """Late-bound entity-name adapter for entity-probe recall."""
        return await self.resolve_entity_name(entity_id, group_id)

    # ─── Lightweight recall (fast entity probe) ─────────────────────

    async def recall_lite(
        self,
        text: str,
        group_id: str,
        session_cache: dict[str, tuple[float, dict]] | None = None,
        token_budget: int = 300,
        cache_ttl: float = 300.0,
    ) -> list[dict]:
        """Fast entity-probe recall. ~3-5ms. Safe to run on every turn.

        Unlike full recall(), this skips embeddings, spreading activation, and reranking.
        It extracts entity mentions from text, probes the graph via FTS5, and returns
        compact entity summaries with top relationships.

        Args:
            text: User message to probe for entity mentions
            group_id: Memory partition
            session_cache: Optional shared cache dict mapping entity_id -> (timestamp, result_dict).
                Caller owns the cache; this method reads and writes to it.
            token_budget: Approximate token limit for returned context (~4 chars/token)
            cache_ttl: Seconds before cached entries expire

        Returns:
            List of compact entity dicts with keys: name, type, summary, confidence,
            identity_core, top_facts
        """
        started = time.perf_counter()
        budget = recall_budget_for_profile(
            self._cfg,
            "auto_lite",
            surface="mcp",
            mode="lite",
            max_results=5,
            max_output_tokens=token_budget,
        )
        try:
            results = await self._entity_probe_recall_service.recall_lite(
                text=text,
                group_id=group_id,
                session_cache=session_cache,
                token_budget=token_budget,
                cache_ttl=cache_ttl,
            )
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 4)
            self.record_memory_operation(
                group_id,
                MemoryOperationSample(
                    operation="recall_lite",
                    source="auto_recall",
                    mode="lite",
                    status="error",
                    duration_ms=duration_ms,
                    **self._budget_fields(budget, duration_ms),
                ),
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        self.record_memory_operation(
            group_id,
            MemoryOperationSample(
                operation="recall_lite",
                source="auto_recall",
                mode="lite",
                status="ok",
                duration_ms=duration_ms,
                result_count=len(results),
                cache_hit=_session_cache_hit_signal(session_cache, results),
                **self._budget_fields(budget, duration_ms),
            ),
        )
        return results

    async def recall_medium(
        self,
        text: str,
        group_id: str,
        session_cache: dict[str, tuple[float, dict]] | None = None,
        token_budget: int = 300,
        cache_ttl: float = 300.0,
        fts_weight: float = 0.3,
        vec_weight: float = 0.7,
    ) -> list[dict]:
        """FTS5 + embedding rerank recall. ~8-15ms. Disambiguates entities.

        Like recall_lite but reranks FTS5 candidates by embedding similarity,
        eliminating name-collision noise (e.g. "Alice" the coworker vs unrelated).
        Falls back to FTS5-only ranking when embeddings are unavailable.

        Args:
            text: User message to probe for entity mentions
            group_id: Memory partition
            session_cache: Optional shared cache (same as recall_lite)
            token_budget: Approximate token limit (~4 chars/token)
            cache_ttl: Cache entry TTL in seconds
            fts_weight: Weight for FTS5 rank position (0-1)
            vec_weight: Weight for embedding similarity (0-1)

        Returns:
            List of compact entity dicts (same format as recall_lite + freshness)
        """
        started = time.perf_counter()
        budget = recall_budget_for_profile(
            self._cfg,
            "auto_lite",
            surface="mcp",
            mode="medium",
            max_results=5,
            max_output_tokens=token_budget,
        )
        try:
            results = await self._entity_probe_recall_service.recall_medium(
                text=text,
                group_id=group_id,
                session_cache=session_cache,
                token_budget=token_budget,
                cache_ttl=cache_ttl,
                fts_weight=fts_weight,
                vec_weight=vec_weight,
            )
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 4)
            self.record_memory_operation(
                group_id,
                MemoryOperationSample(
                    operation="recall_medium",
                    source="auto_recall",
                    mode="medium",
                    status="error",
                    duration_ms=duration_ms,
                    **self._budget_fields(budget, duration_ms),
                ),
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        self.record_memory_operation(
            group_id,
            MemoryOperationSample(
                operation="recall_medium",
                source="auto_recall",
                mode="medium",
                status="ok",
                duration_ms=duration_ms,
                result_count=len(results),
                cache_hit=_session_cache_hit_signal(session_cache, results),
                **self._budget_fields(budget, duration_ms),
            ),
        )
        return results

    # ─── Entity name resolution ─────────────────────────────────────

    async def resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Compatibility API for resolving an entity ID to its name."""
        return await self._lookup_service.resolve_entity_name(entity_id, group_id)

    # ─── Search entities ────────────────────────────────────────────

    async def search_entities(
        self,
        group_id: str = "default",
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Compatibility API for read-only entity lookup."""
        return await self._lookup_service.search_entities(
            group_id=group_id,
            name=name,
            entity_type=entity_type,
            limit=limit,
        )

    # ─── Search facts ───────────────────────────────────────────────

    async def search_facts(
        self,
        group_id: str = "default",
        query: str = "",
        subject: str | None = None,
        predicate: str | None = None,
        include_expired: bool = False,
        include_epistemic: bool = False,
        limit: int = 10,
    ) -> list[dict]:
        """Compatibility API for read-only fact lookup."""
        return await self._lookup_service.search_facts(
            group_id=group_id,
            query=query,
            subject=subject,
            predicate=predicate,
            include_expired=include_expired,
            include_epistemic=include_epistemic,
            limit=limit,
        )

    async def _relationship_is_epistemic(
        self,
        relationship: Relationship,
        *,
        group_id: str,
    ) -> bool:
        """Compatibility wrapper for epistemic graph filtering."""
        return await self._lookup_service.relationship_is_epistemic(
            relationship,
            group_id=group_id,
        )

    # ─── Forget entity ──────────────────────────────────────────────

    async def forget_entity(
        self,
        entity_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Compatibility API for entity forgetting."""
        result = await self._forgetting_service.forget_entity(
            entity_name,
            group_id=group_id,
            reason=reason,
        )
        self.invalidate_memory_packet_cache(group_id)
        return result

    # ─── Forget fact ────────────────────────────────────────────────

    async def forget_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Compatibility API for fact invalidation."""
        result = await self._forgetting_service.forget_fact(
            subject_name,
            predicate,
            object_name,
            group_id=group_id,
            reason=reason,
        )
        self.invalidate_memory_packet_cache(group_id)
        return result

    async def mark_identity_core(
        self,
        entity_name: str,
        *,
        identity_core: bool = True,
        group_id: str = "default",
    ) -> dict:
        """Compatibility API for identity-core protection mutations."""
        return await self._identity_core_service.mark_identity_core(
            entity_name,
            identity_core=identity_core,
            group_id=group_id,
        )

    # ─── Preference-directed memory ─────────────────────────────────

    async def record_explicit_feedback(
        self, group_id: str, entity_id: str, rating: int, comment: str | None = None
    ) -> dict:
        """Compatibility API for preference-directed feedback reinforcement."""
        recorder = getattr(self, "_preference_feedback_recorder", None)
        if recorder is None:
            recorder = PreferenceFeedbackRecorder(
                graph_store=self._graph,
                cfg=self._cfg,
                event_bus=self._event_bus,
            )
        return await recorder.record_explicit_feedback(
            group_id=group_id,
            entity_id=entity_id,
            rating=rating,
            comment=comment,
        )

    # ─── Prospective memory (Wave 4) ────────────────────────────────

    async def create_intention(
        self,
        trigger_text: str,
        action_text: str,
        trigger_type: str = "activation",
        entity_name: str | None = None,
        entity_names: list[str] | None = None,
        threshold: float | None = None,
        priority: str = "normal",
        group_id: str = "default",
        context: str | None = None,
        see_also: list[str] | None = None,
        refresh_trigger: str = "manual",
    ) -> str:
        """Compatibility API for prospective memory intention creation."""
        return await self._prospective_memory_service.create_intention(
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            entity_name=entity_name,
            entity_names=entity_names,
            threshold=threshold,
            priority=priority,
            group_id=group_id,
            context=context,
            see_also=see_also,
            refresh_trigger=refresh_trigger,
        )

    async def _create_intention_v1(
        self,
        trigger_text: str,
        action_text: str,
        trigger_type: str = "semantic",
        entity_name: str | None = None,
        threshold: float | None = None,
        group_id: str = "default",
    ) -> str:
        """Compatibility wrapper for the legacy flat-table intention fallback."""
        return await self._prospective_memory_service._create_intention_v1(
            trigger_text,
            action_text,
            trigger_type,
            entity_name,
            threshold,
            group_id,
        )

    async def list_intentions(
        self,
        group_id: str = "default",
        enabled_only: bool = True,
    ) -> list:
        """Compatibility API for listing prospective memory intentions."""
        return await self._prospective_memory_service.list_intentions(
            group_id=group_id,
            enabled_only=enabled_only,
        )

    async def list_intention_views(
        self,
        group_id: str = "default",
        enabled_only: bool = True,
        *,
        surface: str,
    ) -> list[dict]:
        """Compatibility API for API/MCP intention list presentation."""
        return await self._prospective_memory_service.list_intention_views(
            group_id=group_id,
            enabled_only=enabled_only,
            surface=surface,
        )

    def effective_intention_threshold(self, threshold: float | None = None) -> float:
        """Compatibility API for prospective-memory threshold defaults."""
        return self._prospective_memory_service.effective_activation_threshold(threshold)

    async def dismiss_intention(
        self,
        intention_id: str,
        group_id: str = "default",
        hard: bool = False,
    ) -> None:
        """Compatibility API for dismissing prospective memory intentions."""
        await self._prospective_memory_service.dismiss_intention(
            intention_id=intention_id,
            group_id=group_id,
            hard=hard,
        )

    async def delete_intention(
        self,
        intention_id: str,
        group_id: str = "default",
    ) -> None:
        """Soft-delete an intention (backward compat)."""
        await self._prospective_memory_service.delete_intention(
            intention_id=intention_id,
            group_id=group_id,
        )

    async def migrate_flat_intentions(self, group_id: str = "default") -> int:
        """Compatibility API for migrating flat-table intentions."""
        return await self._prospective_memory_service.migrate_flat_intentions(group_id=group_id)

    async def _update_intention_fire(
        self,
        intention_id: str,
        group_id: str,
        episode_id: str | None = None,
    ) -> None:
        """Compatibility API for recording fired prospective memories."""
        await self._prospective_memory_service.update_intention_fire(
            intention_id=intention_id,
            group_id=group_id,
            episode_id=episode_id,
        )

    async def update_intention_meta(
        self,
        intention_id: str,
        group_id: str,
        updates: dict,
    ) -> None:
        """Compatibility API for updating prospective memory metadata."""
        await self._prospective_memory_service.update_intention_meta(
            intention_id=intention_id,
            group_id=group_id,
            updates=updates,
        )

    # ─── Get context ────────────────────────────────────────────────

    async def _context_recall(self, **kwargs) -> list[dict]:
        """Late-bound recall adapter for context builders and test fakes."""
        kwargs.setdefault("interaction_source", "context_recall")
        return await self.recall(**kwargs)

    async def _context_list_intentions(self, *args, **kwargs) -> list:
        """Late-bound intention adapter for context builders and test fakes."""
        return await self.list_intentions(*args, **kwargs)

    async def _context_resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Late-bound entity-name adapter for context builders and test fakes."""
        return await self.resolve_entity_name(entity_id, group_id)

    async def _context_publish_access_event(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        """Late-bound access-event adapter for context builders and test fakes."""
        await self._publish_access_event(
            entity_id,
            name,
            entity_type,
            group_id,
            accessed_via,
        )

    async def _entity_to_context_data(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        now: float,
        detail_level: str = "full",
    ) -> dict:
        """Compatibility wrapper for tests and older internal call sites."""
        return await self._context_builder.entity_to_context_data(
            entity_id,
            name,
            entity_type,
            summary,
            group_id,
            now,
            detail_level=detail_level,
        )

    @staticmethod
    def _render_tier(header: str, entities: list[dict], facts: list[str]) -> str:
        """Compatibility wrapper for tests and older internal call sites."""
        return MemoryContextBuilder.render_tier(header, entities, facts)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return MemoryContextBuilder.estimate_tokens(text)

    def invalidate_briefing_cache(self, group_id: str) -> None:
        """Clear briefing cache entries for the given group."""
        self._context_builder.invalidate_briefing_cache(group_id)

    def _template_briefing(
        self,
        structured_context: str,
        group_id: str,
        topic_hint: str | None,
    ) -> str:
        """Compatibility wrapper for tests and older internal call sites."""
        return self._context_builder.template_briefing(
            structured_context,
            group_id,
            topic_hint,
        )

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
        operation_source: str = "context",
    ) -> dict:
        """Build a tiered context summary through the shared retrieval service."""
        started = time.perf_counter()
        budget = self._memory_operation_budget(
            operation_source,
            mode=operation_source,
            max_results=max_tokens,
            max_output_tokens=max_tokens,
        )
        try:
            result = await self._context_builder.get_context(
                group_id=group_id,
                max_tokens=max_tokens,
                topic_hint=topic_hint,
                project_path=project_path,
                format=format,
            )
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 4)
            self.record_memory_operation(
                group_id,
                MemoryOperationSample(
                    operation="context",
                    source=operation_source,
                    mode=operation_source,
                    status="error",
                    duration_ms=duration_ms,
                    **self._budget_fields(budget, duration_ms),
                ),
            )
            raise
        duration_ms = round((time.perf_counter() - started) * 1000, 4)
        self.record_memory_operation(
            group_id,
            MemoryOperationSample(
                operation="context",
                source=operation_source,
                mode=operation_source,
                status="ok",
                duration_ms=duration_ms,
                result_count=int(result.get("entity_count") or 0),
                packet_count=len(result.get("cached_packets") or []),
                cache_hit=bool((result.get("packet_cache") or {}).get("hit")),
                **self._budget_fields(budget, duration_ms),
            ),
        )
        return result

    # ─── Get graph state ────────────────────────────────────────────

    def _graph_state_recall_metrics(self, group_id: str) -> dict:
        """Late-bound recall metrics adapter for graph-state readers."""
        return self.get_recall_metrics(group_id)

    def _graph_state_memory_operation_metrics(self, group_id: str) -> dict:
        """Late-bound memory operation metrics adapter for graph-state readers."""
        return self.get_memory_operation_metrics(group_id)

    def _graph_state_epistemic_metrics(self, group_id: str) -> dict:
        """Late-bound epistemic metrics adapter for graph-state readers."""
        return self.get_epistemic_metrics(group_id)

    async def _graph_state_resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Late-bound entity-name adapter for graph-state readers."""
        return await self.resolve_entity_name(entity_id, group_id)

    async def get_graph_state(
        self,
        group_id: str = "default",
        top_n: int = 20,
        include_edges: bool = False,
        entity_types: list[str] | None = None,
    ) -> dict:
        """Compatibility API for graph-state reads."""
        return await self._graph_state_service.get_graph_state(
            group_id=group_id,
            top_n=top_n,
            include_edges=include_edges,
            entity_types=entity_types,
        )

    async def get_dashboard_stats(
        self,
        *,
        group_id: str = "default",
        days: int = 30,
    ) -> dict:
        """Compatibility API for REST dashboard stats reads."""
        return await self._graph_state_service.get_dashboard_stats(
            group_id=group_id,
            days=days,
        )

    async def list_episode_summaries(
        self,
        *,
        group_id: str = "default",
        cursor: str | None = None,
        limit: int = 50,
        source: str | None = None,
        status: str | None = None,
    ) -> dict:
        """Compatibility API for REST episode dashboard listing reads."""
        return await self._graph_state_service.list_episode_summaries(
            group_id=group_id,
            cursor=cursor,
            limit=limit,
            source=source,
            status=status,
        )

    async def get_lifecycle_summary(
        self,
        *,
        group_id: str = "default",
        consolidation_engine: object | None = None,
        consolidation_reader: ConsolidationAuditReader | None = None,
        consolidation_scheduler: object | None = None,
        pressure_accumulator: object | None = None,
        activation_config: ActivationConfig | None = None,
        episode_limit: int = 5,
        cycle_limit: int = 10,
    ) -> dict:
        """Compatibility API for REST/MCP lifecycle summary reads."""
        return await self._lifecycle_summary_service.get_lifecycle_summary(
            group_id=group_id,
            consolidation_engine=consolidation_engine,
            consolidation_reader=consolidation_reader,
            consolidation_scheduler=consolidation_scheduler,
            pressure_accumulator=pressure_accumulator,
            activation_config=activation_config,
            episode_limit=episode_limit,
            cycle_limit=cycle_limit,
        )

    async def get_activation_snapshot(
        self,
        *,
        group_id: str = "default",
        limit: int = 50,
    ) -> dict:
        """Compatibility API for REST activation snapshot reads."""
        return await self._graph_state_service.get_activation_snapshot(
            group_id=group_id,
            limit=limit,
        )

    async def get_activation_curve(
        self,
        *,
        group_id: str = "default",
        entity_id: str,
        hours: int = 24,
        points: int = 48,
    ) -> dict | None:
        """Compatibility API for REST activation curve reads."""
        return await self._graph_state_service.get_activation_curve(
            group_id=group_id,
            entity_id=entity_id,
            hours=hours,
            points=points,
        )

    async def get_graph_neighborhood(
        self,
        *,
        group_id: str = "default",
        center: str | None = None,
        depth: int = 2,
        max_nodes: int = 2000,
        min_activation: float = 0.0,
    ) -> dict | None:
        """Compatibility API for dashboard graph-neighborhood reads."""
        return await self._graph_state_service.get_graph_neighborhood(
            group_id=group_id,
            center=center,
            depth=depth,
            max_nodes=max_nodes,
            min_activation=min_activation,
        )

    async def get_temporal_graph(
        self,
        *,
        group_id: str = "default",
        center: str,
        at_time: datetime,
        at_label: str,
        depth: int = 2,
        max_nodes: int = 2000,
    ) -> dict | None:
        """Compatibility API for dashboard temporal graph reads."""
        return await self._graph_state_service.get_temporal_graph(
            group_id=group_id,
            center=center,
            at_time=at_time,
            at_label=at_label,
            depth=depth,
            max_nodes=max_nodes,
        )

    def get_lifecycle_graph_store(self) -> GraphStore:
        """Return the graph store used by shared lifecycle summary reads."""
        return self._graph_state_service.get_graph_store()

    async def get_entity_profile(self, entity_id: str, group_id: str = "default") -> dict:
        """Compatibility API for entity profile resource reads."""
        return await self._graph_state_service.get_entity_profile(entity_id, group_id)

    async def get_entity_detail(self, entity_id: str, group_id: str = "default") -> dict | None:
        """Compatibility API for REST entity detail reads."""
        return await self._graph_state_service.get_entity_detail(entity_id, group_id)

    async def get_entity_neighbors(self, entity_id: str, group_id: str = "default") -> list[dict]:
        """Compatibility API for one-hop entity-neighbor resource reads."""
        return await self._graph_state_service.get_entity_neighbors(entity_id, group_id)

    async def close_runtime_resources(self) -> None:
        """Close runtime stores owned by this manager."""
        from engram.storage.bootstrap import close_if_supported

        await close_if_supported(self._search)
        await close_if_supported(self._activation)
        await close_if_supported(self._graph)

    def get_episode_worker_runtime_stores(self) -> EpisodeWorkerRuntimeStores:
        """Compatibility API exposing the stores needed by the episode worker."""
        return EpisodeWorkerRuntimeStores(
            graph=self._graph,
            activation=self._activation,
            search=self._search,
        )

    async def update_entity_profile(
        self,
        entity_id: str,
        updates: dict,
        *,
        group_id: str = "default",
    ) -> dict | None:
        """Compatibility API for REST entity profile updates."""
        return await self._entity_mutation_service.update_entity_profile(
            entity_id,
            updates,
            group_id=group_id,
        )

    async def get_all_adjudications(
        self,
        group_id: str,
        limit: int = 20,
        status: str = "pending",
    ) -> list[dict]:
        """Return adjudication requests plus their candidate evidence across all episodes."""
        requests = await self._graph.get_pending_adjudication_requests(group_id, limit=limit)
        if not requests:
            return []

        response = []
        for request in requests:
            episode_id = request["episode_id"]
            evidence_rows = await self._graph.get_episode_evidence(episode_id, group_id)
            by_id = {row["evidence_id"]: row for row in evidence_rows}
            response.append(
                {
                    "request_id": request["request_id"],
                    "episode_id": episode_id,
                    "ambiguity_tags": request.get("ambiguity_tags", []),
                    "selected_text": request.get("selected_text", ""),
                    "candidate_evidence": [
                        {
                            "evidence_id": row["evidence_id"],
                            "fact_class": row["fact_class"],
                            "payload": row.get("payload", {}),
                        }
                        for evidence_id in request.get("evidence_ids", [])
                        if (row := by_id.get(evidence_id)) is not None
                    ],
                    "instructions": self._evidence_adjudication_service._adjudication_instructions(
                        request.get("ambiguity_tags", []),
                    ),
                },
            )
        return response

    async def delete_entity_by_id(
        self,
        entity_id: str,
        *,
        group_id: str = "default",
    ) -> dict | None:
        """Compatibility API for REST entity soft deletes."""
        return await self._entity_mutation_service.delete_entity_by_id(
            entity_id,
            group_id=group_id,
        )
