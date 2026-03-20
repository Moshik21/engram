"""Graph Manager — orchestrates extraction, entity resolution, and storage."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from engram.config import ActivationConfig
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
from engram.extraction.cues import build_episode_cue
from engram.extraction.discourse import classify_discourse
from engram.extraction.evidence import (
    CommitDecision,
    EvidenceBundle,
    EvidenceCandidate,
    evidence_candidate_from_dict,
)
from engram.extraction.evidence_bridge import EvidenceBridge
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.extraction.models import ApplyOutcome, ProjectionBundle, ProjectionPlan
from engram.extraction.narrow.pipeline import NarrowExtractionPipeline
from engram.extraction.narrow_adapter import NarrowExtractorAdapter
from engram.extraction.ollama_extractor import OllamaExtractor
from engram.extraction.planner import ProjectionPlanner, summarize_plan
from engram.extraction.policy import ProjectionPolicy
from engram.extraction.post_apply import ProjectionPostProcessor
from engram.extraction.projector import EpisodeProjector
from engram.models.activation import ActivationState
from engram.models.adjudication import AdjudicationRequest
from engram.models.consolidation import RelationshipApplyResult
from engram.models.entity import Entity
from engram.models.episode import Attachment, Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.epistemic import ArtifactHit, EpistemicBundle, EvidenceClaim
from engram.models.recall import MemoryInteractionEvent
from engram.models.relationship import Relationship
from engram.retrieval.control import RecallNeedController, RecallNeedThresholds
from engram.retrieval.epistemic import (
    EpistemicRoutingController,
    apply_answer_contract_to_evidence_plan,
    apply_claim_states,
    artifact_class_for_path,
    build_evidence_plan,
    build_memory_claims,
    build_runtime_claims,
    extract_artifact_claims,
    extract_decision_claims,
    infer_claim_state,
    reconcile_claims,
    resolve_answer_contract,
    should_materialize_conversation_decision,
    summarize_claim_states,
)
from engram.retrieval.epistemic import (
    route_question as build_question_frame,
)
from engram.retrieval.feedback import publish_memory_interaction
from engram.retrieval.pipeline import retrieve
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex
from engram.utils.dates import utc_now, utc_now_iso

logger = logging.getLogger(__name__)

_EPISTEMIC_FACT_PREDICATES = {
    "DECIDED_IN",
    "DOCUMENTED_IN",
    "IMPLEMENTED_BY",
    "ANNOUNCED_AS",
    "SUPERSEDED_BY",
}


def _freshness_label(updated_at: datetime | None) -> str:
    """Compute freshness label from entity updated_at timestamp."""
    if not updated_at:
        return "unknown"
    age = (datetime.utcnow() - updated_at).days
    if age <= 7:
        return "fresh"
    if age <= 30:
        return "recent"
    if age <= 90:
        return "aging"
    return "stale"


class ProjectionError(RuntimeError):
    """Typed projection failure that distinguishes retryable failures."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class EvidenceMaterializationFailure(RuntimeError):  # noqa: N818
    """Raised when stored evidence cannot be materialized but the cycle may continue."""


@dataclass
class EvidenceMaterializationOutcome:
    """Summary of applying bridged evidence into graph state."""

    bundle: ProjectionBundle
    apply_outcome: ApplyOutcome = field(default_factory=ApplyOutcome)
    committed_ids: dict[str, str] = field(default_factory=dict)

    @property
    def materialized(self) -> bool:
        return bool(self.committed_ids)


@dataclass
class AdjudicationResolutionOutcome:
    """Result of resolving an adjudication request."""

    request_id: str
    status: str
    committed_ids: dict[str, str] = field(default_factory=dict)
    superseded_evidence_ids: list[str] = field(default_factory=list)
    replacement_evidence_ids: list[str] = field(default_factory=list)


class _TopActivatedEntry(TypedDict):
    id: str
    name: str
    entity_type: str
    summary: str | None
    activation: float
    access_count: int


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


def _coerce_int(value: object, default: int = 0) -> int:
    """Best-effort integer coercion for loose payloads and mocked test inputs."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


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
        self._runtime_mode = runtime_mode or "unknown"
        self._projection_planner = ProjectionPlanner(self._cfg)
        self._projection_policy = ProjectionPolicy(self._cfg)
        self._projector = EpisodeProjector(self._extractor)

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
        self._epistemic_controller = EpistemicRoutingController(self._cfg)

        # Reconsolidation tracker (Brain Architecture Phase 2B)
        if self._cfg.reconsolidation_enabled:
            from engram.retrieval.reconsolidation import LabileWindowTracker

            self._labile_tracker: LabileWindowTracker | None = LabileWindowTracker(
                ttl=self._cfg.reconsolidation_window_seconds,
                max_entries=self._cfg.reconsolidation_max_entries,
            )
        else:
            self._labile_tracker = None

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
        """Convert evidence candidates to storage rows with explicit status."""
        rows: list[dict] = []
        for evidence, _decision in evidence_pairs:
            rows.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "episode_id": evidence.episode_id,
                    "fact_class": evidence.fact_class,
                    "confidence": evidence.confidence,
                    "source_type": evidence.source_type,
                    "extractor_name": evidence.extractor_name,
                    "payload": evidence.payload,
                    "source_span": evidence.source_span,
                    "corroborating_signals": evidence.corroborating_signals,
                    "ambiguity_tags": evidence.ambiguity_tags,
                    "ambiguity_score": evidence.ambiguity_score,
                    "adjudication_request_id": evidence.adjudication_request_id,
                    "created_at": (
                        evidence.created_at.isoformat()
                        if hasattr(evidence.created_at, "isoformat")
                        else str(evidence.created_at)
                    ),
                    "status": status,
                    **({"commit_reason": commit_reason} if commit_reason else {}),
                },
            )
        return rows

    @staticmethod
    def _serialize_candidate_records(
        candidates: list[EvidenceCandidate],
        *,
        status: str,
        commit_reason: str | None = None,
    ) -> list[dict]:
        """Convert raw evidence candidates to storage rows."""
        return [
            {
                "evidence_id": candidate.evidence_id,
                "episode_id": candidate.episode_id,
                "fact_class": candidate.fact_class,
                "confidence": candidate.confidence,
                "source_type": candidate.source_type,
                "extractor_name": candidate.extractor_name,
                "payload": candidate.payload,
                "source_span": candidate.source_span,
                "corroborating_signals": candidate.corroborating_signals,
                "ambiguity_tags": candidate.ambiguity_tags,
                "ambiguity_score": candidate.ambiguity_score,
                "adjudication_request_id": candidate.adjudication_request_id,
                "created_at": (
                    candidate.created_at.isoformat()
                    if hasattr(candidate.created_at, "isoformat")
                    else str(candidate.created_at)
                ),
                "status": status,
                **({"commit_reason": commit_reason} if commit_reason else {}),
            }
            for candidate in candidates
        ]

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
    ) -> list[AdjudicationRequest]:
        """Create persisted work items for ambiguous evidence groups."""
        max_requests = max(1, self._cfg.edge_adjudication_max_requests_per_episode)
        if len(ambiguous_groups) > max_requests:
            overflow = ambiguous_groups[max_requests - 1 :]
            merged = AmbiguityGroup()
            for group in overflow:
                merged.candidates.extend(group.candidates)
                merged.ambiguity_tags.update(group.ambiguity_tags)
                if len(group.selected_text) > len(merged.selected_text):
                    merged.selected_text = group.selected_text
            merged.request_reason = "needs_adjudication:" + ",".join(sorted(merged.ambiguity_tags))
            ambiguous_groups = ambiguous_groups[: max_requests - 1] + [merged]
        requests: list[AdjudicationRequest] = []
        for group in ambiguous_groups:
            request = AdjudicationRequest(
                episode_id=episode_id,
                group_id=group_id,
                ambiguity_tags=sorted(group.ambiguity_tags),
                evidence_ids=[candidate.evidence_id for candidate in group.candidates],
                selected_text=group.selected_text,
                request_reason=group.request_reason,
            )
            for candidate in group.candidates:
                candidate.adjudication_request_id = request.request_id
            requests.append(request)
        return requests

    @staticmethod
    def _adjudication_instructions(tags: list[str]) -> str:
        """Return MCP-facing instructions for an adjudication request."""
        tag_list = ", ".join(tags)
        return (
            "Resolve only if highly confident. Use explicit entities and relationships "
            f"for the ambiguous case ({tag_list}); otherwise leave it unresolved."
        )

    async def get_episode_adjudications(
        self,
        episode_id: str,
        group_id: str = "default",
    ) -> list[dict]:
        """Return adjudication requests plus their current candidate evidence."""
        requests = await self._graph.get_episode_adjudications(episode_id, group_id)
        if not requests:
            return []
        evidence_rows = await self._graph.get_episode_evidence(episode_id, group_id)
        by_id = {row["evidence_id"]: row for row in evidence_rows}
        response: list[dict] = []
        for request in requests:
            if request.get("status") not in {"pending", "deferred", "error"}:
                continue
            response.append(
                {
                    "request_id": request["request_id"],
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
                    "instructions": self._adjudication_instructions(
                        request.get("ambiguity_tags", []),
                    ),
                },
            )
        return response

    @staticmethod
    def _committed_id_map(
        committed_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        *,
        entity_map: dict[str, str],
        claims,
        relationship_results: list[RelationshipApplyResult],
    ) -> dict[str, str]:
        """Resolve evidence IDs to durable entity/relationship IDs after apply."""
        committed_ids: dict[str, str] = {}
        for evidence, _decision in committed_pairs:
            entity_name = None
            if evidence.fact_class == "entity":
                entity_name = evidence.payload.get("name")
            elif evidence.fact_class == "attribute":
                entity_name = evidence.payload.get("entity")
            if entity_name and entity_name in entity_map:
                committed_ids[evidence.evidence_id] = entity_map[entity_name]

        for claim, result in zip(claims, relationship_results):
            evidence_id = (claim.raw_payload or {}).get("evidence_id")
            relationship_id = result.metadata.get("relationship_id") or result.metadata.get(
                "existing_relationship_id"
            )
            if not relationship_id:
                continue
            if evidence_id:
                committed_ids[evidence_id] = relationship_id
            for temporal_evidence_id in (claim.raw_payload or {}).get("temporal_evidence_ids", []):
                committed_ids[temporal_evidence_id] = relationship_id
        return committed_ids

    @staticmethod
    def _apply_committed_ids(
        evidence_rows: list[dict],
        committed_ids: dict[str, str],
    ) -> tuple[list[dict], list[dict]]:
        """Split evidence rows into committed and unresolved sets."""
        committed_rows: list[dict] = []
        unresolved_rows: list[dict] = []
        for row in evidence_rows:
            committed_id = committed_ids.get(row["evidence_id"])
            if committed_id:
                row["committed_id"] = committed_id
                committed_rows.append(row)
            else:
                unresolved_rows.append(row)
        return committed_rows, unresolved_rows

    @staticmethod
    def _rehydrate_evidence_pairs(
        evidence_rows: list[dict],
    ) -> list[tuple[EvidenceCandidate, CommitDecision]]:
        """Convert stored evidence rows back into committed evidence pairs."""
        pairs: list[tuple[EvidenceCandidate, CommitDecision]] = []
        for row in evidence_rows:
            candidate = evidence_candidate_from_dict(row)
            pairs.append(
                (
                    candidate,
                    CommitDecision(
                        evidence_id=candidate.evidence_id,
                        action="commit",
                        reason=row.get("commit_reason", "approved_for_materialization"),
                        effective_confidence=candidate.confidence,
                    ),
                ),
            )
        return pairs

    @staticmethod
    def _evidence_projection_bundle(
        episode: Episode,
        entities,
        claims,
        recall_content: str | None = None,
    ) -> ProjectionBundle:
        """Build a minimal ProjectionBundle for evidence-driven apply/index flows."""
        selected_text = recall_content or episode.content
        return ProjectionBundle(
            episode_id=episode.id,
            plan=ProjectionPlan(
                episode_id=episode.id,
                strategy="evidence_materialized",
                spans=[],
                selected_text=selected_text,
                selected_chars=len(selected_text),
                total_chars=len(episode.content or ""),
            ),
            entities=entities,
            claims=claims,
        )

    async def _index_materialized_bundle(
        self,
        *,
        bundle: ProjectionBundle,
        entity_map: dict[str, str],
        group_id: str,
        episode_id: str,
    ) -> None:
        """Index entities and the source episode without touching episode status."""
        try:
            for candidate in bundle.entities:
                entity_id = entity_map.get(candidate.name)
                if not entity_id:
                    continue
                entity = await self._graph.get_entity(entity_id, group_id)
                if entity is None:
                    continue
                if self._cfg.structure_aware_embeddings:
                    await self._index_entity_with_structure(entity, group_id)
                else:
                    await self._search.index_entity(entity)

            episode = await self._graph.get_episode_by_id(episode_id, group_id)
            if episode:
                await self._search.index_episode(episode)
        except Exception as embed_err:
            logger.warning(
                "Embedding failed for episode %s (non-fatal): %s",
                episode_id,
                embed_err,
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
        """Bridge evidence into graph writes, index touched nodes, and return committed IDs."""
        entities, claims = self._evidence_bridge.bridge(evidence_pairs)  # type: ignore[union-attr]
        bundle = self._evidence_projection_bundle(
            episode,
            entities,
            claims,
            recall_content=recall_content,
        )
        if not bundle.entities and not bundle.claims:
            return EvidenceMaterializationOutcome(bundle=bundle)

        apply_outcome = await self._apply_engine.apply_entities(
            bundle.entities,
            episode,
            group_id,
            recall_content=recall_content,
        )
        entity_map = apply_outcome.entity_map
        await self._apply_bootstrap_part_of_edges(episode, entity_map, group_id)
        if on_before_relationships is not None:
            await on_before_relationships()
        apply_outcome.relationship_results = await self._apply_engine.apply_relationships(
            bundle.claims,
            entity_map=entity_map,
            meta_entity_names=apply_outcome.meta_entity_names,
            group_id=group_id,
            source_episode=episode.id,
        )
        committed_ids = self._committed_id_map(
            evidence_pairs,
            entity_map=entity_map,
            claims=bundle.claims,
            relationship_results=apply_outcome.relationship_results,
        )
        if not committed_ids:
            return EvidenceMaterializationOutcome(
                bundle=bundle,
                apply_outcome=apply_outcome,
                committed_ids={},
            )
        await self._index_materialized_bundle(
            bundle=bundle,
            entity_map=entity_map,
            group_id=group_id,
            episode_id=episode.id,
        )
        self.invalidate_briefing_cache(group_id)
        return EvidenceMaterializationOutcome(
            bundle=bundle,
            apply_outcome=apply_outcome,
            committed_ids=committed_ids,
        )

    async def materialize_stored_evidence(
        self,
        episode_id: str,
        evidence_rows: list[dict],
        *,
        group_id: str = "default",
    ) -> EvidenceMaterializationOutcome:
        """Materialize stored evidence rows into graph state for consolidation."""
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if episode is None:
            raise EvidenceMaterializationFailure(
                f"episode_not_found:{episode_id}",
            )
        return await self.materialize_evidence(
            episode=episode,
            evidence_pairs=self._rehydrate_evidence_pairs(evidence_rows),
            group_id=group_id,
            recall_content=episode.content,
        )

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
        """Resolve an ambiguous request through the shared materialization path."""
        request = await self._graph.get_adjudication_request(request_id, group_id)
        if request is None:
            raise ValueError(f"Adjudication request not found: {request_id}")
        if request.get("status") not in {"pending", "deferred", "error"}:
            raise ValueError(
                f"Adjudication request is not open: {request_id}:{request.get('status')}",
            )

        episode_id = request["episode_id"]
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if episode is None:
            raise EvidenceMaterializationFailure(f"episode_not_found:{episode_id}")

        resolution_payload = {
            "entities": entities or [],
            "relationships": relationships or [],
            "reject_evidence_ids": reject_evidence_ids or [],
            "model_tier": model_tier,
            **({"rationale": rationale} if rationale else {}),
        }
        attempt_count = int(request.get("attempt_count", 0) or 0) + 1

        episode_evidence = await self._graph.get_episode_evidence(episode_id, group_id)
        request_evidence = [
            row
            for row in episode_evidence
            if row["evidence_id"] in set(request.get("evidence_ids", []))
        ]
        active_request_evidence = [
            row
            for row in request_evidence
            if row.get("status") in {"pending", "deferred", "approved"}
        ]

        rejected_ids = set(reject_evidence_ids or [])
        for row in active_request_evidence:
            if row["evidence_id"] not in rejected_ids:
                continue
            await self._graph.update_evidence_status(
                row["evidence_id"],
                "rejected",
                updates={
                    "commit_reason": f"rejected_by_adjudication:{request_id}",
                },
                group_id=group_id,
            )

        replacement_candidates = proposals_to_evidence(
            entities,
            relationships,
            episode_id,
            group_id,
            model_tier,
            source_type=source,
            confidence_bonus=0.05,
            adjudication_request_id=request_id,
            rationale=rationale,
            source_span=request.get("selected_text") or episode.content,
        )

        if not replacement_candidates:
            remaining_unresolved = [
                row for row in active_request_evidence if row["evidence_id"] not in rejected_ids
            ]
            status = "rejected" if not remaining_unresolved else "deferred"
            await self._graph.update_adjudication_request(
                request_id,
                {
                    "status": status,
                    "resolution_source": source,
                    "resolution_payload": resolution_payload,
                    "attempt_count": attempt_count,
                },
                group_id,
            )
            return AdjudicationResolutionOutcome(
                request_id=request_id,
                status=status,
                superseded_evidence_ids=[],
                replacement_evidence_ids=[],
            )

        replacement_pairs = [
            (
                candidate,
                CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="commit",
                    reason=f"resolved_by_{source}",
                    effective_confidence=candidate.confidence,
                ),
            )
            for candidate in replacement_candidates
        ]
        materialization = await self.materialize_evidence(
            episode=episode,
            evidence_pairs=replacement_pairs,
            group_id=group_id,
            recall_content=request.get("selected_text") or episode.content,
        )
        replacement_rows = self._serialize_candidate_records(
            replacement_candidates,
            status="committed",
            commit_reason=f"materialized_from_adjudication:{request_id}",
        )
        replacement_rows, unresolved_replacements = self._apply_committed_ids(
            replacement_rows,
            materialization.committed_ids,
        )
        if replacement_rows:
            await self._graph.store_evidence(
                replacement_rows,
                group_id=group_id,
                default_status="committed",
            )
        if unresolved_replacements:
            for row in unresolved_replacements:
                row["status"] = "deferred"
                row["commit_reason"] = "adjudication_unmaterialized"
            await self._graph.store_evidence(
                unresolved_replacements,
                group_id=group_id,
                default_status="deferred",
            )

        if not materialization.materialized:
            await self._graph.update_adjudication_request(
                request_id,
                {
                    "status": "deferred",
                    "resolution_source": source,
                    "resolution_payload": resolution_payload,
                    "attempt_count": attempt_count,
                },
                group_id,
            )
            return AdjudicationResolutionOutcome(
                request_id=request_id,
                status="deferred",
                replacement_evidence_ids=[
                    row["evidence_id"] for row in replacement_rows + unresolved_replacements
                ],
            )

        superseded_ids: list[str] = []
        for row in active_request_evidence:
            if row["evidence_id"] in rejected_ids:
                continue
            await self._graph.update_evidence_status(
                row["evidence_id"],
                "superseded",
                updates={
                    "commit_reason": f"superseded_by_adjudication:{request_id}",
                },
                group_id=group_id,
            )
            superseded_ids.append(row["evidence_id"])

        await self._graph.update_adjudication_request(
            request_id,
            {
                "status": "materialized",
                "resolution_source": source,
                "resolution_payload": resolution_payload,
                "attempt_count": attempt_count,
            },
            group_id,
        )
        return AdjudicationResolutionOutcome(
            request_id=request_id,
            status="materialized",
            committed_ids=materialization.committed_ids,
            superseded_evidence_ids=superseded_ids,
            replacement_evidence_ids=[
                row["evidence_id"] for row in replacement_rows + unresolved_replacements
            ],
        )

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

    def get_epistemic_metrics(self, group_id: str = "default") -> dict:
        """Return rolling epistemic routing metrics for stats surfaces."""
        return self._epistemic_controller.snapshot(group_id)

    async def _publish_access_event(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        """Compute activation and publish an activation.access event."""
        if not self._event_bus:
            return
        from engram.activation.engine import compute_activation

        now = time.time()
        state = await self._activation.get_activation(entity_id)
        activation = 0.0
        if state:
            activation = compute_activation(state.access_history, now, self._cfg)
        self._publish(
            group_id,
            "activation.access",
            {
                "entityId": entity_id,
                "name": name,
                "entityType": entity_type,
                "activation": round(activation, 4),
                "accessedVia": accessed_via,
            },
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
        now = timestamp if timestamp is not None else time.time()
        await self._activation.record_access(entity.id, now, group_id=group_id)
        await self._publish_access_event(
            entity.id,
            entity.name,
            entity.entity_type,
            group_id,
            source,
        )

        if self._labile_tracker is not None:
            self._labile_tracker.mark_labile(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                query,
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

        valid_types = {
            "surfaced",
            "selected",
            "used",
            "confirmed",
            "dismissed",
            "corrected",
        }
        if interaction_type not in valid_types:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")

        should_record_access = interaction_type in {"used", "confirmed"}
        should_record_positive = interaction_type == "confirmed" and self._cfg.ts_enabled
        should_record_negative = interaction_type == "corrected" and self._cfg.ts_enabled
        should_publish = (
            self._cfg.recall_telemetry_enabled or self._cfg.recall_usage_feedback_enabled
        )

        seen_ids: set[str] = set()
        now = time.time()
        for memory_id in memory_ids:
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            metadata = result_lookup.get(memory_id, {}) if result_lookup else {}
            result_type = metadata.get("result_type")
            if result_type is None and isinstance(memory_id, str) and memory_id.startswith("cue:"):
                result_type = "cue_episode"
            if result_type == "cue_episode":
                episode_id = metadata.get("episode_id")
                if not episode_id and isinstance(memory_id, str) and memory_id.startswith("cue:"):
                    episode_id = memory_id.split(":", 1)[1]
                if not episode_id:
                    continue
                episode = await self._graph.get_episode_by_id(episode_id, group_id)
                if episode is None:
                    continue
                cue_score = metadata.get("score")
                await self._record_cue_hit(
                    episode,
                    float(cue_score) if cue_score is not None else 0.0,
                    query,
                    interaction_type=interaction_type,
                    count_hit=bool(metadata.get("count_hit", False)),
                )
                self._recall_need_controller.record_interaction(
                    group_id,
                    interaction_type,
                    result_type="cue_episode",
                )
                continue

            entity_name = metadata.get("entity_name")
            entity_type = metadata.get("entity_type")
            score = metadata.get("score")

            entity = await self._graph.get_entity(memory_id, group_id)
            if entity is not None:
                entity_name = entity.name
                entity_type = entity.entity_type

            recorded_access = False
            if should_record_access and entity is not None:
                await self._record_entity_access(
                    entity,
                    group_id=group_id,
                    query=query,
                    source=source,
                    timestamp=now,
                )
                recorded_access = True

            if should_record_positive:
                from engram.activation.feedback import record_positive_feedback

                await record_positive_feedback(memory_id, self._activation, self._cfg)

            if should_record_negative:
                from engram.activation.feedback import record_negative_feedback

                await record_negative_feedback(memory_id, self._activation, self._cfg)

            if should_publish:
                publish_memory_interaction(
                    self._event_bus,
                    MemoryInteractionEvent(
                        group_id=group_id,
                        entity_id=memory_id,
                        entity_name=entity_name,
                        entity_type=entity_type,
                        interaction_type=interaction_type,
                        source=source,
                        query=query,
                        score=score,
                        recorded_access=recorded_access,
                    ),
                )
            self._recall_need_controller.record_interaction(
                group_id,
                interaction_type,
                result_type="entity",
            )

    async def _update_episode_status(
        self, episode_id: str, status: EpisodeStatus, group_id: str = "default", **extra: object
    ) -> None:
        """Update episode status and updated_at timestamp."""
        updates: dict = {"status": status.value}
        updates.update(extra)
        await self._graph.update_episode(episode_id, updates, group_id=group_id)

    async def _update_projection_state(
        self,
        episode_id: str,
        state: EpisodeProjectionState,
        group_id: str = "default",
        *,
        reason: str | None = None,
        last_projected_at: datetime | None = None,
    ) -> None:
        updates: dict[str, object] = {"projection_state": state.value}
        if reason is not None:
            updates["last_projection_reason"] = reason
        if last_projected_at is not None:
            updates["last_projected_at"] = last_projected_at.isoformat()
        await self._graph.update_episode(episode_id, updates, group_id=group_id)

    async def _update_episode_cue(
        self,
        episode_id: str,
        group_id: str,
        updates: dict,
    ) -> None:
        if hasattr(self._graph, "update_episode_cue"):
            await self._graph.update_episode_cue(episode_id, updates, group_id=group_id)

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
        if episode is None:
            return None
        state = getattr(episode, "projection_state", None)
        value = getattr(state, "value", None)
        if isinstance(value, str):
            return value
        return state if isinstance(state, str) else None

    @staticmethod
    def _cue_result_payload(cue: EpisodeCue, *, hit_increment: int = 0) -> dict[str, object]:
        projection_state = (
            cue.projection_state.value
            if hasattr(cue.projection_state, "value")
            else cue.projection_state
        )
        return {
            "episode_id": cue.episode_id,
            "cue_text": cue.cue_text,
            "supporting_spans": cue.first_spans,
            "projection_state": projection_state,
            "route_reason": cue.route_reason,
            "hit_count": (cue.hit_count or 0) + hit_increment,
            "surfaced_count": cue.surfaced_count,
            "selected_count": cue.selected_count,
            "used_count": cue.used_count,
            "near_miss_count": cue.near_miss_count,
            "policy_score": cue.policy_score,
            "last_feedback_at": (
                cue.last_feedback_at.isoformat() if cue.last_feedback_at else None
            ),
            "last_projected_at": (
                cue.last_projected_at.isoformat() if cue.last_projected_at else None
            ),
        }

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
        cue = await self._get_episode_cue(episode.id, episode.group_id)
        if cue is None:
            return

        now_dt = utc_now()
        feedback_type = "near_miss" if near_miss else (interaction_type or "surfaced")
        feedback = self._projection_policy.apply_feedback(
            cue,
            interaction_type=feedback_type,
            score=score,
            count_hit=count_hit,
        )
        cue_updates: dict[str, object] = dict(feedback.updates)
        cue_updates["last_feedback_at"] = now_dt
        if not near_miss and "hit_count" in cue_updates:
            cue_updates["last_hit_at"] = now_dt

        current_projection_state = (
            episode.projection_state.value
            if hasattr(episode.projection_state, "value")
            else episode.projection_state
        )
        event_payload = {
            "episodeId": episode.id,
            "projectionState": current_projection_state,
            "interactionType": feedback_type,
            "score": round(score, 4),
            "query": query[:200],
        }
        if "hit_count" in cue_updates:
            event_payload["hitCount"] = cue_updates["hit_count"]
        if "policy_score" in cue_updates:
            event_payload["policyScore"] = cue_updates["policy_score"]
        self._publish(
            episode.group_id,
            "cue.hit" if not near_miss else "cue.near_miss",
            event_payload,
        )
        if not near_miss and interaction_type in {"surfaced", "selected"}:
            self._recall_need_controller.record_interaction(
                episode.group_id,
                interaction_type,
                result_type="cue_episode",
            )

        episode_projection_state = (
            episode.projection_state.value
            if hasattr(episode.projection_state, "value")
            else episode.projection_state
        )
        promotable_states = {
            EpisodeProjectionState.CUED.value,
            EpisodeProjectionState.CUE_ONLY.value,
            EpisodeProjectionState.QUEUED.value,
            EpisodeProjectionState.FAILED.value,
        }
        hit_count = _coerce_int(
            cue_updates.get("hit_count", cue.hit_count or 0),
            cue.hit_count or 0,
        )
        should_promote = (
            hit_count >= self._cfg.cue_recall_hit_threshold or feedback.should_promote
        ) and episode_projection_state in promotable_states
        if should_promote:
            promotion_reason = (
                "cue_recall_hits"
                if hit_count >= self._cfg.cue_recall_hit_threshold
                else (feedback.promotion_reason or "cue_policy")
            )
            await self._update_episode_status(
                episode.id,
                EpisodeStatus.QUEUED,
                group_id=episode.group_id,
                error=None,
            )
            cue_updates["projection_state"] = EpisodeProjectionState.SCHEDULED
            cue_updates["route_reason"] = promotion_reason
            await self._update_projection_state(
                episode.id,
                EpisodeProjectionState.SCHEDULED,
                group_id=episode.group_id,
                reason=promotion_reason,
            )
            self._publish(
                episode.group_id,
                "cue.promoted",
                {
                    "episodeId": episode.id,
                    "hitCount": hit_count,
                    "reason": promotion_reason,
                    "score": round(score, 4),
                    "policyScore": cue_updates.get("policy_score", cue.policy_score),
                },
            )
            self._publish(
                episode.group_id,
                "episode.projection_scheduled",
                {
                    "episodeId": episode.id,
                    "reason": promotion_reason,
                    "hitCount": hit_count,
                },
            )
        elif self._cfg.cue_policy_learning_enabled and "policy_score" in cue_updates:
            self._publish(
                episode.group_id,
                "cue.policy_updated",
                {
                    "episodeId": episode.id,
                    "interactionType": feedback_type,
                    "policyScore": cue_updates["policy_score"],
                    "projectionState": current_projection_state,
                },
            )

        await self._update_episode_cue(episode.id, episode.group_id, cue_updates)

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
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        episode = Episode(
            id=episode_id,
            content=content,
            source=source,
            status=EpisodeStatus.QUEUED,
            projection_state=EpisodeProjectionState.QUEUED,
            group_id=group_id,
            session_id=session_id,
            conversation_date=conversation_date,
            created_at=utc_now(),
            attachments=attachments or [],
        )
        await self._graph.create_episode(episode)
        self._publish(
            group_id,
            "episode.queued",
            {
                "episode": {
                    "episodeId": episode_id,
                    "content": content[:200] if content else "",
                    "source": source or "unknown",
                    "status": "queued",
                    "createdAt": (
                        episode.created_at.isoformat() + "Z" if episode.created_at else ""
                    ),
                    "updatedAt": (
                        episode.created_at.isoformat() + "Z" if episode.created_at else ""
                    ),
                    "entities": [],
                    "factsCount": 0,
                    "processingDurationMs": None,
                    "error": None,
                    "retryCount": 0,
                },
            },
        )

        if self._cfg.cue_layer_enabled:
            try:
                cue = build_episode_cue(episode, self._cfg)
                if cue is not None and hasattr(self._graph, "upsert_episode_cue"):
                    await self._graph.upsert_episode_cue(cue)
                    if self._cfg.cue_vector_index_enabled and hasattr(
                        self._search, "index_episode_cue"
                    ):
                        await self._search.index_episode_cue(cue)
                    await self._update_projection_state(
                        episode_id,
                        cue.projection_state,
                        group_id=group_id,
                        reason=cue.route_reason,
                    )
                    self._publish(
                        group_id,
                        "episode.cued",
                        {
                            "episodeId": episode_id,
                            "projectionState": cue.projection_state.value,
                            "cueScore": cue.cue_score,
                            "projectionPriority": cue.projection_priority,
                            "routeReason": cue.route_reason,
                        },
                    )
                    if cue.projection_state == EpisodeProjectionState.SCHEDULED:
                        self._publish(
                            group_id,
                            "episode.projection_scheduled",
                            {
                                "episodeId": episode_id,
                                "reason": cue.route_reason,
                                "projectionState": cue.projection_state.value,
                            },
                        )
                elif cue is None:
                    await self._update_projection_state(
                        episode_id,
                        EpisodeProjectionState.CUE_ONLY,
                        group_id=group_id,
                        reason="system_discourse",
                    )
            except Exception:
                logger.warning("Failed to generate/store episode cue", exc_info=True)
        if self._cfg.decision_graph_enabled and source != "auto:bootstrap" and content.strip():
            try:
                await self._materialize_conversation_decisions(
                    content,
                    episode_id=episode_id,
                    group_id=group_id,
                )
            except Exception:
                logger.warning("Failed to materialize conversation decisions", exc_info=True)
        return episode_id

    async def project_episode(
        self,
        episode_id: str,
        group_id: str = "default",
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
    ) -> None:
        """Run extraction, resolution, and embedding on a stored episode.

        Raises on failure after setting FAILED status.
        """
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        start_ms = time.monotonic()
        content = episode.content

        # Content dedup: skip if identical content was already extracted
        content_hash = hashlib.sha256((content or "").encode()).hexdigest()[:16]
        if content_hash in self._content_hashes or content_hash in self._content_hashes_inflight:
            logger.info(
                "project_episode: skipping duplicate content for episode %s",
                episode_id,
            )
            await self._update_episode_status(
                episode_id,
                EpisodeStatus.COMPLETED,
                group_id=group_id,
                skipped_triage=True,
            )
            await self._update_projection_state(
                episode_id,
                EpisodeProjectionState.CUE_ONLY,
                group_id=group_id,
                reason="duplicate_content",
            )
            return

        # Discourse gate: skip pure system meta-commentary
        if classify_discourse(content) == "system":
            logger.warning("project_episode: skipping system-discourse episode %s", episode_id)
            await self._update_episode_status(
                episode_id,
                EpisodeStatus.COMPLETED,
                group_id=group_id,
                skipped_meta=True,
            )
            self._content_hashes.add(content_hash)
            await self._update_projection_state(
                episode_id,
                EpisodeProjectionState.CUE_ONLY,
                group_id=group_id,
                reason="system_discourse",
            )
            await self._update_episode_cue(
                episode_id,
                group_id,
                {
                    "projection_state": EpisodeProjectionState.CUE_ONLY,
                    "route_reason": "system_discourse",
                },
            )
            return

        self._content_hashes_inflight.add(content_hash)

        try:
            await self._update_projection_state(
                episode_id,
                EpisodeProjectionState.PROJECTING,
                group_id=group_id,
                reason="projection_started",
            )
            await self._update_episode_cue(
                episode_id,
                group_id,
                {"projection_state": EpisodeProjectionState.PROJECTING},
            )
            cue = await self._get_episode_cue(episode_id, group_id)
            plan = self._projection_planner.plan(episode, cue)
            plan_summary = summarize_plan(plan)
            if plan.warnings:
                plan_summary["warnings"] = list(plan.warnings)
            self._publish(
                group_id,
                "episode.projection_started",
                {
                    "episodeId": episode_id,
                    **plan_summary,
                },
            )

            await self._update_episode_status(
                episode_id,
                EpisodeStatus.EXTRACTING,
                group_id=group_id,
            )

            used_evidence_materializer = False
            if self._should_use_evidence_pipeline(
                proposed_entities=proposed_entities,
                proposed_relationships=proposed_relationships,
            ):
                # NEW PATH: narrow extractors -> evidence -> commit -> bridge -> apply
                cue = await self._get_episode_cue(episode_id, group_id)
                evidence_bundle = self._build_evidence_bundle(
                    text=plan.selected_text,
                    episode_id=episode_id,
                    group_id=group_id,
                    cue=cue,
                    proposed_entities=proposed_entities,
                    proposed_relationships=proposed_relationships,
                    model_tier=model_tier,
                )
                if self._cfg.edge_adjudication_enabled and self._ambiguity_analyzer:
                    ambiguity_analysis = await self._ambiguity_analyzer.analyze(
                        text=plan.selected_text,
                        bundle=evidence_bundle,
                        group_id=group_id,
                    )
                    ambiguous_groups = ambiguity_analysis.ambiguous_groups
                    evidence_bundle = EvidenceBundle(
                        episode_id=evidence_bundle.episode_id,
                        group_id=evidence_bundle.group_id,
                        candidates=ambiguity_analysis.clean_candidates,
                        extractor_stats=evidence_bundle.extractor_stats,
                        total_ms=evidence_bundle.total_ms,
                    )
                    if ambiguous_groups:
                        requests = self._build_adjudication_requests(
                            episode_id,
                            group_id,
                            ambiguous_groups,
                        )
                        if requests:
                            await self._graph.store_adjudication_requests(
                                [request.to_dict() for request in requests],
                                group_id=group_id,
                            )
                            ambiguous_candidates = [
                                candidate
                                for group in ambiguous_groups
                                for candidate in group.candidates
                                if candidate.adjudication_request_id
                            ]
                            ambiguous_rows = self._serialize_candidate_records(
                                ambiguous_candidates,
                                status="pending",
                                commit_reason="needs_adjudication",
                            )
                            await self._graph.store_evidence(
                                ambiguous_rows,
                                group_id=group_id,
                            )
                raw_entity_count = await self._graph.get_entity_count(group_id)
                entity_count = raw_entity_count if isinstance(raw_entity_count, int) else 0
                decisions = self._commit_policy.evaluate(  # type: ignore[union-attr]
                    evidence_bundle,
                    entity_count,
                )
                committed = [
                    (ev, d)
                    for ev, d in zip(evidence_bundle.candidates, decisions)
                    if d.action == "commit"
                ]
                deferred = [
                    (ev, d)
                    for ev, d in zip(evidence_bundle.candidates, decisions)
                    if d.action == "defer"
                ]
                deferred_dicts = (
                    self._serialize_evidence_records(deferred, status="deferred")
                    if self._cfg.evidence_store_deferred and deferred
                    else []
                )
                committed_dicts = (
                    self._serialize_evidence_records(
                        committed,
                        status="committed",
                        commit_reason="committed_on_hot_path",
                    )
                    if committed
                    else []
                )

                await self._update_episode_status(
                    episode_id,
                    EpisodeStatus.RESOLVING,
                    group_id=group_id,
                )
                now = time.time()
                materialization = await self.materialize_evidence(
                    episode=episode,
                    evidence_pairs=committed,
                    group_id=group_id,
                    recall_content=plan.selected_text,
                    on_before_relationships=lambda: self._update_episode_status(
                        episode_id,
                        EpisodeStatus.WRITING,
                        group_id=group_id,
                    ),
                )
                used_evidence_materializer = True
                apply_outcome = materialization.apply_outcome
                entity_map = apply_outcome.entity_map
                committed_dicts, unmaterialized_rows = self._apply_committed_ids(
                    committed_dicts,
                    materialization.committed_ids,
                )
                if unmaterialized_rows and self._cfg.evidence_store_deferred:
                    for row in unmaterialized_rows:
                        row["status"] = "deferred"
                        row["commit_reason"] = None
                    deferred_dicts.extend(unmaterialized_rows)

                if deferred_dicts:
                    await self._graph.store_evidence(
                        deferred_dicts,
                        group_id=group_id,
                        default_status="deferred",
                    )
                if committed_dicts:
                    await self._graph.store_evidence(
                        committed_dicts,
                        group_id=group_id,
                        default_status="committed",
                    )
                bundle = materialization.bundle
            else:
                # EXISTING PATH: LLM extractor (unchanged)
                bundle = await self._projector.project(plan)
                if bundle.is_error:
                    raise ProjectionError(
                        f"extractor_{bundle.extractor_status}: "
                        f"{bundle.extractor_error or 'unknown_error'}",
                        retryable=bundle.retryable,
                    )

                await self._update_episode_status(
                    episode_id,
                    EpisodeStatus.RESOLVING,
                    group_id=group_id,
                )
                now = time.time()
                apply_outcome = await self._apply_engine.apply_entities(
                    bundle.entities,
                    episode,
                    group_id,
                    recall_content=plan.selected_text,
                )
                entity_map = apply_outcome.entity_map
                await self._apply_bootstrap_part_of_edges(
                    episode,
                    entity_map,
                    group_id,
                )

                await self._update_episode_status(
                    episode_id,
                    EpisodeStatus.WRITING,
                    group_id=group_id,
                )
                apply_outcome.relationship_results = await self._apply_engine.apply_relationships(
                    bundle.claims,
                    entity_map=entity_map,
                    meta_entity_names=apply_outcome.meta_entity_names,
                    group_id=group_id,
                    source_episode=episode_id,
                )

            await self._run_surprise_detection(
                entity_map=entity_map,
                group_id=group_id,
                now=now,
            )
            await self._run_prospective_memory(
                content=content,
                entity_map=entity_map,
                group_id=group_id,
                episode_id=episode_id,
            )
            await self._publish_projection_graph_changes(
                bundle=bundle,
                apply_outcome=apply_outcome,
                group_id=group_id,
                episode_id=episode_id,
            )
            if not used_evidence_materializer:
                await self._index_projected_bundle(
                    bundle=bundle,
                    entity_map=entity_map,
                    group_id=group_id,
                    episode_id=episode_id,
                )

            await self._update_episode_status(
                episode_id,
                EpisodeStatus.ACTIVATING,
                group_id=group_id,
            )
            await self._store_emotional_encoding_context(
                episode_id=episode_id,
                content=content,
                entity_map=entity_map,
                group_id=group_id,
            )

            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            await self._update_episode_status(
                episode_id,
                EpisodeStatus.COMPLETED,
                group_id=group_id,
                processing_duration_ms=elapsed_ms,
            )
            projected_at = utc_now()
            await self._update_projection_state(
                episode_id,
                EpisodeProjectionState.PROJECTED,
                group_id=group_id,
                reason="projected",
                last_projected_at=projected_at,
            )
            await self._update_episode_cue(
                episode_id,
                group_id,
                {
                    "projection_state": EpisodeProjectionState.PROJECTED,
                    "projection_attempts": (episode.retry_count or 0) + 1,
                    "last_projected_at": projected_at,
                },
            )
            self._content_hashes_inflight.discard(content_hash)
            self._content_hashes.add(content_hash)
            self._publish(
                group_id,
                "episode.completed",
                {
                    "episodeId": episode_id,
                    "status": "completed",
                    "entity_count": len(bundle.entities),
                    "relationship_count": len(bundle.claims),
                    "duration_ms": elapsed_ms,
                },
            )
            logger.info(
                "Ingested episode %s: %d entities, %d relationships",
                episode_id,
                len(bundle.entities),
                len(bundle.claims),
            )
            if not used_evidence_materializer:
                self.invalidate_briefing_cache(group_id)

        except Exception as e:
            self._content_hashes_inflight.discard(content_hash)
            logger.error("Failed to process episode %s: %s", episode_id, e)
            retry_count = (episode.retry_count or 0) + 1
            retryable = isinstance(e, ProjectionError) and e.retryable
            if retryable and retry_count <= self._cfg.projection_max_retries:
                fail_status = EpisodeStatus.RETRYING
                fail_projection_state = EpisodeProjectionState.FAILED
            elif retryable:
                fail_status = EpisodeStatus.DEAD_LETTER
                fail_projection_state = EpisodeProjectionState.DEAD_LETTER
            else:
                fail_status = EpisodeStatus.FAILED
                fail_projection_state = EpisodeProjectionState.FAILED
            await self._update_episode_status(
                episode_id,
                fail_status,
                group_id=group_id,
                error=str(e),
                retry_count=retry_count,
            )
            await self._update_projection_state(
                episode_id,
                fail_projection_state,
                group_id=group_id,
                reason=str(e),
            )
            await self._update_episode_cue(
                episode_id,
                group_id,
                {
                    "projection_state": fail_projection_state,
                    "projection_attempts": retry_count,
                },
            )
            self._publish(
                group_id,
                "episode.failed",
                {
                    "episodeId": episode_id,
                    "status": fail_status.value,
                    "error": str(e),
                    "retry_count": retry_count,
                },
            )
            raise

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
        episode_id = await self.store_episode(
            content, group_id, source, session_id,
            conversation_date=conversation_date,
            attachments=attachments,
        )
        try:
            await self.project_episode(
                episode_id,
                group_id,
                proposed_entities=proposed_entities,
                proposed_relationships=proposed_relationships,
                model_tier=model_tier,
            )
        except Exception:
            pass  # project_episode already sets FAILED status
        return episode_id

    # ─── Project Bootstrap ──────────────────────────────────────────

    # Files to bootstrap into the artifact substrate (pattern → max chars)
    _BOOTSTRAP_FILES: list[tuple[str, int]] = [
        ("README.md", 2000),
        ("package.json", 3000),
        ("pyproject.toml", 3000),
        ("Makefile", 3000),
        (".env.example", 2000),
        ("docker-compose.yml", 3000),
        ("CLAUDE.md", 2500),
        ("docs/design/**/*.md", 4000),
        ("docs/vision/**/*.md", 4000),
        ("skills/**/SKILL.md", 3500),
    ]

    async def bootstrap_project(
        self,
        project_path: str,
        group_id: str = "default",
        session_id: str | None = None,
    ) -> dict:
        """Bootstrap a project: create Project entity and observe key files.

        Idempotent — if the Project entity exists and was bootstrapped
        within the last 24 hours, returns early. Otherwise re-observes
        files to pick up changes (cheap store_episode, no LLM).
        """
        from pathlib import Path as _Path

        p = _Path(project_path).expanduser()
        project_name = p.name
        if not project_name or str(p) in (str(_Path.home()), "/"):
            return {"status": "skipped", "reason": "invalid_path"}

        now_iso = utc_now_iso()

        # Check for existing Project entity
        existing = await self._graph.find_entities(
            name=project_name,
            entity_type="Project",
            group_id=group_id,
            limit=1,
        )

        if existing:
            entity = existing[0]
            entity_id = entity.id
            attrs = entity.attributes or {}
            last_bs = attrs.get("last_bootstrapped")

            # Check staleness
            if last_bs:
                try:
                    last_dt = datetime.fromisoformat(last_bs)
                    age_seconds = (utc_now() - last_dt).total_seconds()
                    if age_seconds < self._cfg.artifact_bootstrap_stale_seconds:
                        return {
                            "status": "already_bootstrapped",
                            "project_entity_id": entity_id,
                        }
                except (ValueError, TypeError):
                    pass  # Malformed timestamp — refresh

            # Stale or never timestamped — refresh files
            files_observed = await self._observe_project_files(
                p,
                project_name,
                group_id,
                session_id,
            )

            # Update timestamp
            import json as _json

            merged_attrs = {**attrs, "last_bootstrapped": now_iso}
            await self._graph.update_entity(
                entity_id,
                {"attributes": _json.dumps(merged_attrs)},
                group_id=group_id,
            )
            await self._activation.record_access(
                entity_id,
                time.time(),
                group_id=group_id,
            )

            self._publish(
                group_id,
                "project.refreshed",
                {
                    "project_name": project_name,
                    "project_entity_id": entity_id,
                    "files_observed": files_observed,
                },
            )

            return {
                "status": "refreshed",
                "project_entity_id": entity_id,
                "files_observed": files_observed,
            }

        # Create Project entity
        entity_id = f"ent_{uuid.uuid4().hex[:12]}"
        entity = Entity(
            id=entity_id,
            name=project_name,
            entity_type="Project",
            summary=f"Software project at {project_path}",
            attributes={
                "project_path": str(p),
                "last_bootstrapped": now_iso,
            },
            group_id=group_id,
        )
        await self._graph.create_entity(entity)
        await self._index_entity_with_structure(entity, group_id)
        await self._activation.record_access(
            entity_id,
            time.time(),
            group_id=group_id,
        )

        files_observed = await self._observe_project_files(
            p,
            project_name,
            group_id,
            session_id,
        )

        self._publish(
            group_id,
            "project.bootstrapped",
            {
                "project_name": project_name,
                "project_entity_id": entity_id,
                "files_observed": files_observed,
            },
        )

        return {
            "status": "bootstrapped",
            "project_entity_id": entity_id,
            "files_observed": files_observed,
        }

    async def _observe_project_files(
        self,
        project_dir: object,  # Path
        project_name: str,
        group_id: str,
        session_id: str | None,
    ) -> list[str]:
        """Read, index, and optionally store bootstrapped project artifacts."""
        files_observed: list[str] = []
        project_path = str(project_dir)
        project_entity_id = await self._resolve_project_entity_id(project_name, group_id)
        now_iso = utc_now_iso()
        seen_rel_paths: set[str] = set()

        for filepath, rel_path, max_chars in self._iter_bootstrap_files(project_dir):  # type: ignore[arg-type]
            if rel_path in seen_rel_paths:
                continue
            seen_rel_paths.add(rel_path)
            try:
                raw_content = filepath.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                continue

            truncated = raw_content[:max_chars]
            artifact_class = artifact_class_for_path(rel_path)
            content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
            claims = extract_artifact_claims(
                truncated,
                rel_path=rel_path,
                artifact_class=artifact_class,
                project_name=project_name,
                timestamp=now_iso,
            )
            artifact_entity, changed = await self._upsert_artifact_entity(
                project_name=project_name,
                project_path=project_path,
                rel_path=rel_path,
                artifact_class=artifact_class,
                content=truncated,
                content_hash=content_hash,
                claims=claims,
                group_id=group_id,
                now_iso=now_iso,
            )
            if project_entity_id is not None:
                await self._ensure_relationship(
                    artifact_entity.id,
                    project_entity_id,
                    "PART_OF",
                    group_id=group_id,
                )
            if changed:
                tagged = f"[project-bootstrap|{project_name}|{rel_path}]\n{truncated}"
                episode_id = await self.store_episode(
                    content=tagged,
                    group_id=group_id,
                    source="auto:bootstrap",
                    session_id=session_id,
                )
                # Mark bootstrap episodes as CUE_ONLY to prevent entity extraction
                # from documentation — these are project artifacts, not user knowledge
                await self._graph.update_episode(
                    episode_id,
                    {
                        "projection_state": EpisodeProjectionState.CUE_ONLY.value,
                        "status": EpisodeStatus.COMPLETED.value,
                    },
                    group_id=group_id,
                )
                await self._graph.update_entity(
                    artifact_entity.id,
                    {
                        "attributes": json.dumps(
                            self._merge_attributes(
                                artifact_entity.attributes,
                                {"last_episode_id": episode_id},
                            )
                        )
                    },
                    group_id=group_id,
                )
            if self._cfg.decision_graph_enabled and claims:
                await self._materialize_artifact_decisions(
                    artifact_entity,
                    claims,
                    group_id=group_id,
                )
            files_observed.append(rel_path)
        return files_observed

    def _iter_bootstrap_files(
        self,
        project_dir: Path,
    ) -> list[tuple[Path, str, int]]:
        """Expand bootstrap patterns into concrete files."""
        matches: list[tuple[Path, str, int]] = []
        for pattern, max_chars in self._BOOTSTRAP_FILES:
            for filepath in sorted(project_dir.glob(pattern)):
                if not filepath.is_file():
                    continue
                try:
                    rel_path = filepath.relative_to(project_dir).as_posix()
                except ValueError:
                    continue
                matches.append((filepath, rel_path, max_chars))
        return matches

    async def _resolve_project_entity_id(
        self,
        project_name: str,
        group_id: str,
    ) -> str | None:
        existing = await self._graph.find_entities(
            name=project_name,
            entity_type="Project",
            group_id=group_id,
            limit=1,
        )
        if existing:
            return existing[0].id
        return None

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
        artifact_key = f"{group_id}:{project_path}:{rel_path}"
        artifact_id = f"art_{hashlib.sha256(artifact_key.encode()).hexdigest()[:12]}"
        attributes = {
            "project_path": project_path,
            "rel_path": rel_path,
            "artifact_class": artifact_class,
            "content_hash": content_hash,
            "last_observed_at": now_iso,
            "stale_after": self._cfg.artifact_bootstrap_stale_seconds,
            "snippet": self._artifact_snippet(content),
            "claims": [self._claim_to_attr(claim) for claim in claims],
        }
        entity = await self._graph.get_entity(artifact_id, group_id)
        if not isinstance(entity, Entity):
            entity = None
        if entity is None:
            entity = Entity(
                id=artifact_id,
                name=rel_path,
                entity_type="Artifact",
                summary=self._artifact_summary(project_name, rel_path, content, claims),
                attributes=attributes,
                group_id=group_id,
            )
            await self._graph.create_entity(entity)
            await self._index_entity_with_structure(entity, group_id)
            await self._activation.record_access(entity.id, time.time(), group_id=group_id)
            return entity, True

        current_hash = (entity.attributes or {}).get("content_hash")
        changed = current_hash != content_hash
        merged_attrs = self._merge_attributes(entity.attributes, attributes)
        updates: dict[str, object] = {"attributes": json.dumps(merged_attrs)}
        if changed:
            updates["summary"] = self._artifact_summary(project_name, rel_path, content, claims)
        await self._graph.update_entity(entity.id, updates, group_id=group_id)
        entity.attributes = merged_attrs
        if changed:
            entity.summary = str(updates["summary"])
            await self._index_entity_with_structure(entity, group_id)
        return entity, changed

    @staticmethod
    def _claim_to_attr(claim: EvidenceClaim) -> dict:
        return {
            "subject": claim.subject,
            "predicate": claim.predicate,
            "object": claim.object,
            "source_type": claim.source_type,
            "authority_type": claim.authority_type,
            "externalization_state": claim.externalization_state,
            "timestamp": claim.timestamp,
            "confidence": claim.confidence,
            "provenance": claim.provenance,
        }

    @staticmethod
    def _merge_attributes(existing: dict | None, updates: dict) -> dict:
        merged = dict(existing or {})
        merged.update(updates)
        return merged

    @staticmethod
    def _artifact_snippet(content: str) -> str:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:240]
        return content[:240]

    @staticmethod
    def _artifact_summary(
        project_name: str,
        rel_path: str,
        content: str,
        claims: list[EvidenceClaim],
    ) -> str:
        summary_parts = [f"{project_name} artifact {rel_path}"]
        if claims:
            summary_parts.append(
                "; ".join(f"{claim.predicate}={claim.object}" for claim in claims[:3])
            )
        else:
            summary_parts.append(GraphManager._artifact_snippet(content))
        return " — ".join(part for part in summary_parts if part)[:500]

    async def _list_project_artifacts(
        self,
        *,
        group_id: str,
        project_path: str | None = None,
        limit: int = 200,
    ) -> list[Entity]:
        artifacts = await self._graph.find_entities(
            entity_type="Artifact",
            group_id=group_id,
            limit=limit,
        )
        if project_path is None:
            return artifacts
        return [
            entity
            for entity in artifacts
            if (entity.attributes or {}).get("project_path") == project_path
        ]

    @staticmethod
    def _artifact_is_stale(entity: Entity, stale_seconds: int) -> bool:
        attrs = entity.attributes or {}
        last_observed = attrs.get("last_observed_at")
        if not last_observed:
            return True
        try:
            observed_dt = datetime.fromisoformat(last_observed)
        except (TypeError, ValueError):
            return True
        return (utc_now() - observed_dt).total_seconds() >= stale_seconds

    async def search_artifacts(
        self,
        *,
        query: str,
        group_id: str = "default",
        project_path: str | None = None,
        limit: int = 5,
    ) -> list[ArtifactHit]:
        """Search bootstrapped project artifacts by semantic or name match."""
        if project_path and self._cfg.artifact_bootstrap_enabled:
            await self.bootstrap_project(project_path, group_id=group_id)

        hits: list[ArtifactHit] = []
        seen_ids: set[str] = set()
        scored_ids = await self._search.search(
            query=query,
            entity_types=["Artifact"],
            group_id=group_id,
            limit=max(limit * 4, 10),
        )
        if not scored_ids:
            fallback = await self._graph.find_entities(
                name=query,
                entity_type="Artifact",
                group_id=group_id,
                limit=max(limit * 2, 10),
            )
            scored_ids = [(entity.id, 0.4) for entity in fallback]

        for entity_id, score in scored_ids:
            if entity_id in seen_ids:
                continue
            seen_ids.add(entity_id)
            entity = await self._graph.get_entity(entity_id, group_id)
            if entity is None:
                continue
            attrs = entity.attributes or {}
            if project_path and attrs.get("project_path") != project_path:
                continue
            claims = [
                self._claim_from_attr(claim_data)
                for claim_data in attrs.get("claims", [])[:4]
                if isinstance(claim_data, dict)
            ]
            hits.append(
                ArtifactHit(
                    artifact_id=entity.id,
                    path=str(attrs.get("rel_path") or entity.name),
                    artifact_class=str(attrs.get("artifact_class") or "artifact"),
                    snippet=str(attrs.get("snippet") or entity.summary or ""),
                    last_observed_at=attrs.get("last_observed_at"),
                    score=score,
                    stale=self._artifact_is_stale(
                        entity,
                        int(attrs.get("stale_after") or self._cfg.artifact_bootstrap_stale_seconds),
                    ),
                    supporting_claims=claims,
                )
            )
            if len(hits) >= limit:
                break
        return hits

    async def get_runtime_state(
        self,
        *,
        group_id: str = "default",
        project_path: str | None = None,
    ) -> dict:
        """Return effective runtime/config state plus artifact freshness."""
        artifacts = await self._list_project_artifacts(
            group_id=group_id,
            project_path=project_path,
        )
        stale_seconds = int(self._cfg.artifact_bootstrap_stale_seconds)
        stale_count = sum(
            1 for artifact in artifacts if self._artifact_is_stale(artifact, stale_seconds)
        )
        fresh_count = max(0, len(artifacts) - stale_count)
        last_observed = None
        for artifact in artifacts:
            observed = (artifact.attributes or {}).get("last_observed_at")
            if observed and (last_observed is None or observed > last_observed):
                last_observed = observed
        return {
            "projectName": Path(project_path).name if project_path else "Engram",
            "runtime": {
                "mode": self._runtime_mode,
            },
            "activation": {
                "consolidationProfile": self._cfg.consolidation_profile,
                "recallProfile": self._cfg.recall_profile,
                "integrationProfile": self._cfg.integration_profile,
            },
            "features": {
                "epistemicRoutingEnabled": self._cfg.epistemic_routing_enabled,
                "artifactBootstrapEnabled": self._cfg.artifact_bootstrap_enabled,
                "artifactRecallEnabled": self._cfg.artifact_recall_enabled,
                "runtimeExecutorEnabled": self._cfg.epistemic_runtime_executor_enabled,
                "decisionGraphEnabled": self._cfg.decision_graph_enabled,
                "epistemicReconcileEnabled": self._cfg.epistemic_reconcile_enabled,
                "answerContractEnabled": self._cfg.answer_contract_enabled,
                "claimStateModelingEnabled": self._cfg.claim_state_modeling_enabled,
                "recallNeedAnalyzerEnabled": self._cfg.recall_need_analyzer_enabled,
                "recallNeedGraphProbeEnabled": self._cfg.recall_need_graph_probe_enabled,
            },
            "artifactBootstrap": {
                "enabled": self._cfg.artifact_bootstrap_enabled,
                "projectPath": project_path,
                "artifactCount": len(artifacts),
                "freshArtifactCount": fresh_count,
                "staleArtifactCount": stale_count,
                "lastObservedAt": last_observed,
                "staleAfterSeconds": stale_seconds,
            },
            "stats": {
                "recallMetrics": self.get_recall_metrics(group_id),
                "epistemicMetrics": self.get_epistemic_metrics(group_id),
            },
            "generatedAt": utc_now_iso(),
        }

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
        """Create the question frame and evidence plan for a turn."""
        if memory_need is None:
            from engram.retrieval.graph_probe import GraphProbe
            from engram.retrieval.need import analyze_memory_need

            graph_probe = None
            if self._cfg.recall_need_graph_probe_enabled:
                graph_probe = getattr(self, "_recall_need_graph_probe", None)
                if not isinstance(graph_probe, GraphProbe):
                    graph_probe = GraphProbe(self._graph, self._activation)
                    self._recall_need_graph_probe = graph_probe
            memory_need = await analyze_memory_need(
                question,
                recent_turns=recent_turns or [],
                session_entity_names=session_entity_names or [],
                mode="chat" if surface == "rest" else "auto_recall",
                graph_probe=graph_probe,
                group_id=group_id,
                conv_context=self._conv_context,
                cfg=self._cfg,
                thresholds=self.get_recall_need_thresholds(group_id),
            )

        surface_capabilities = self._surface_capabilities(surface, project_path)
        frame = build_question_frame(
            question,
            memory_need=memory_need,
            recent_turns=recent_turns,
            project_path=project_path,
            surface_capabilities=surface_capabilities,
        )
        plan = build_evidence_plan(
            frame,
            surface_capabilities=surface_capabilities,
            cfg=self._cfg,
        )
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=[],
        )
        plan = apply_answer_contract_to_evidence_plan(
            question,
            frame=frame,
            plan=plan,
            answer_contract=answer_contract,
            memory_need=memory_need,
        )
        self._epistemic_controller.record_route(
            group_id,
            frame.mode,
            operator=answer_contract.operator,
            scopes=answer_contract.relevant_scopes,
        )
        return frame, plan, memory_need, answer_contract

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
        """Return a routed question frame and evidence plan."""
        frame, plan, routed_need, answer_contract = await self._build_epistemic_route(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )
        payload = {
            "questionFrame": frame.to_dict(),
            "evidencePlan": plan.to_dict(),
            "answerContract": answer_contract.to_dict(),
            "recommendedNextSources": plan.recommended_next_sources,
        }
        if routed_need is not None:
            payload["memoryNeed"] = routed_need.to_payload(
                source="epistemic_route",
                mode=surface,
                turn_preview=question[:160],
            )
        return payload

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
        """Route a question, gather planned evidence, and reconcile it."""
        frame, plan, routed_need, _initial_contract = await self._build_epistemic_route(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )

        query_text = getattr(routed_need, "query_hint", None) or question
        memory_query = plan.source_queries.get("memory") or query_text
        artifact_query = plan.source_queries.get("artifacts") or query_text

        if plan.use_artifacts and project_path and self._cfg.artifact_bootstrap_enabled:
            await self.bootstrap_project(project_path, group_id=group_id)

        memory_results: list[dict] = []
        artifact_hits: list[ArtifactHit] = []
        runtime_state: dict | None = None

        if plan.use_memory:
            memory_results = await self.recall(
                query=memory_query,
                group_id=group_id,
                limit=max(1, plan.memory_budget),
                record_access=False,
            )
        if plan.use_artifacts:
            artifact_hits = await self.search_artifacts(
                query=artifact_query,
                group_id=group_id,
                project_path=project_path,
                limit=max(1, plan.artifact_budget),
            )
        if plan.use_runtime or plan.use_implementation:
            runtime_state = await self.get_runtime_state(
                group_id=group_id,
                project_path=project_path,
            )

        memory_claims = build_memory_claims(memory_results)
        artifact_claims = [claim for hit in artifact_hits for claim in hit.supporting_claims]
        runtime_claims = build_runtime_claims(runtime_state or {})
        implementation_claims: list[EvidenceClaim] = []
        all_claims = apply_claim_states(
            memory_claims + artifact_claims + runtime_claims + implementation_claims
        )
        claim_state_summary = summarize_claim_states(all_claims)
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=all_claims,
        )

        reconciliation = reconcile_claims(
            frame,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            answer_contract=answer_contract,
        )
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=all_claims,
            reconciliation=reconciliation,
        )
        reconciliation = reconcile_claims(
            frame,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            answer_contract=answer_contract,
        )

        artifact_stale_miss = bool(
            runtime_state
            and runtime_state.get("artifactBootstrap", {}).get("staleArtifactCount", 0)
            and not artifact_hits
        )
        self._epistemic_controller.record_execution(
            group_id,
            reconciliation,
            plan,
            answer_contract=answer_contract,
            artifact_stale_miss=artifact_stale_miss,
        )

        return EpistemicBundle(
            question_frame=frame,
            evidence_plan=plan,
            reconciliation=reconciliation,
            answer_contract=answer_contract,
            memory_claims=memory_claims,
            artifact_claims=artifact_claims,
            runtime_claims=runtime_claims,
            implementation_claims=implementation_claims,
            artifact_hits=artifact_hits,
            memory_results=memory_results,
            runtime_state=runtime_state,
            claim_state_summary=claim_state_summary,
        )

    @staticmethod
    def _surface_capabilities(surface: str, project_path: str | None) -> dict[str, bool]:
        return {
            "workspace_available": surface == "mcp" and bool(project_path),
            "native_workspace_search": surface == "mcp" and bool(project_path),
            "artifact_bootstrap": bool(project_path),
        }

    @staticmethod
    def _claim_from_attr(claim_data: dict) -> EvidenceClaim:
        return EvidenceClaim(
            subject=str(claim_data.get("subject", "")),
            predicate=str(claim_data.get("predicate", "")),
            object=str(claim_data.get("object", "")),
            source_type=str(
                claim_data.get("source_type") or claim_data.get("sourceType") or "artifact"
            ),
            authority_type=str(
                claim_data.get("authority_type") or claim_data.get("authorityType") or "canonical"
            ),
            externalization_state=str(
                claim_data.get("externalization_state")
                or claim_data.get("externalizationState")
                or "documented"
            ),
            claim_state=str(
                claim_data.get("claim_state") or claim_data.get("claimState") or "mentioned"
            ),
            timestamp=claim_data.get("timestamp"),
            confidence=float(claim_data.get("confidence", 0.0) or 0.0),
            provenance=dict(claim_data.get("provenance") or {}),
        )

    async def _materialize_artifact_decisions(
        self,
        artifact_entity: Entity,
        claims: list[EvidenceClaim],
        *,
        group_id: str,
    ) -> None:
        attrs = artifact_entity.attributes or {}
        artifact_class = str(attrs.get("artifact_class") or "design_doc")
        link_predicate = "DOCUMENTED_IN"
        if artifact_class == "config":
            link_predicate = "IMPLEMENTED_BY"
        elif artifact_class in {"readme", "skill"}:
            link_predicate = "ANNOUNCED_AS"
        for claim in claims:
            if not self._is_decision_claim(claim):
                continue
            decision = await self._upsert_decision_entity(claim, group_id=group_id)
            await self._ensure_relationship(
                decision.id,
                artifact_entity.id,
                link_predicate,
                group_id=group_id,
            )

    async def _materialize_conversation_decisions(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> None:
        subject = self._infer_decision_subject(content)
        if subject is None:
            return
        claims: list[EvidenceClaim] = []
        for chunk in re.split(r"[\n.!?]+", content):
            if not should_materialize_conversation_decision(chunk):
                continue
            claims.extend(
                extract_decision_claims(
                    chunk,
                    subject=subject,
                    source_type="memory",
                    authority_type="historical",
                    externalization_state="discussed",
                    provenance={"episode_id": episode_id},
                )
            )
        filtered_claims: list[EvidenceClaim] = []
        for claim in claims:
            if not self._is_decision_claim(claim):
                continue
            claim.claim_state = infer_claim_state(claim)
            if claim.claim_state != "decided":
                continue
            filtered_claims.append(claim)
        claims = filtered_claims
        if not claims:
            return
        artifact = await self._upsert_conversation_artifact(
            content,
            episode_id=episode_id,
            group_id=group_id,
        )
        for claim in claims:
            decision = await self._upsert_decision_entity(claim, group_id=group_id)
            await self._ensure_relationship(
                decision.id,
                artifact.id,
                "DECIDED_IN",
                group_id=group_id,
                source_episode=episode_id,
            )

    async def _upsert_conversation_artifact(
        self,
        content: str,
        *,
        episode_id: str,
        group_id: str,
    ) -> Entity:
        artifact_id = f"art_conv_{episode_id.split('_')[-1]}"
        existing = await self._graph.get_entity(artifact_id, group_id)
        if existing is not None:
            return existing
        artifact = Entity(
            id=artifact_id,
            name=f"conversation:{episode_id}",
            entity_type="Artifact",
            summary=f"Conversation record for decision provenance: {content[:180]}",
            attributes={
                "artifact_class": "conversation_record",
                "source_episode": episode_id,
                "snippet": content[:240],
                "last_observed_at": utc_now_iso(),
                "stale_after": self._cfg.artifact_bootstrap_stale_seconds,
            },
            group_id=group_id,
        )
        await self._graph.create_entity(artifact)
        await self._index_entity_with_structure(artifact, group_id)
        return artifact

    async def _upsert_decision_entity(
        self,
        claim: EvidenceClaim,
        *,
        group_id: str,
    ) -> Entity:
        prefix = f"{claim.subject}:{claim.predicate}"
        existing = await self._graph.find_entities(
            name=prefix,
            entity_type="Decision",
            group_id=group_id,
            limit=20,
        )
        for candidate in existing:
            attrs = candidate.attributes or {}
            if (
                attrs.get("canonical_predicate") == claim.predicate
                and attrs.get("subject") == claim.subject
                and attrs.get("decision_object") == claim.object
            ):
                merged_attrs = self._merge_attributes(
                    attrs,
                    {
                        "last_seen_at": utc_now_iso(),
                        "authority_type": claim.authority_type,
                        "externalization_state": claim.externalization_state,
                        "source_type": claim.source_type,
                    },
                )
                await self._graph.update_entity(
                    candidate.id,
                    {"attributes": json.dumps(merged_attrs)},
                    group_id=group_id,
                )
                candidate.attributes = merged_attrs
                return candidate

        decision = Entity(
            id=f"dec_{uuid.uuid4().hex[:12]}",
            name=f"{claim.subject}:{claim.predicate}:{claim.object[:80]}",
            entity_type="Decision",
            summary=f"{claim.subject} -> {claim.predicate} -> {claim.object}"[:500],
            attributes={
                "subject": claim.subject,
                "canonical_predicate": claim.predicate,
                "decision_object": claim.object,
                "authority_type": claim.authority_type,
                "externalization_state": claim.externalization_state,
                "source_type": claim.source_type,
                "last_seen_at": utc_now_iso(),
            },
            group_id=group_id,
        )
        await self._graph.create_entity(decision)
        await self._index_entity_with_structure(decision, group_id)
        for candidate in existing:
            attrs = candidate.attributes or {}
            if (
                attrs.get("canonical_predicate") == claim.predicate
                and attrs.get("subject") == claim.subject
                and attrs.get("decision_object") != claim.object
            ):
                await self._ensure_relationship(
                    candidate.id,
                    decision.id,
                    "SUPERSEDED_BY",
                    group_id=group_id,
                )
        return decision

    async def _ensure_relationship(
        self,
        source_id: str,
        target_id: str,
        predicate: str,
        *,
        group_id: str,
        source_episode: str | None = None,
    ) -> None:
        existing = await self._graph.find_existing_relationship(
            source_id,
            target_id,
            predicate,
            group_id,
        )
        if existing is not None:
            return
        await self._graph.create_relationship(
            Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=source_id,
                target_id=target_id,
                predicate=predicate,
                group_id=group_id,
                source_episode=source_episode,
            )
        )

    @staticmethod
    def _is_decision_claim(claim: EvidenceClaim) -> bool:
        return claim.predicate in {
            "public_launch_path",
            "full_mode_default_behavior",
            "integration_profile",
            "recall_profile",
            "consolidation_profile",
            "decision_statement",
        } or claim.predicate.startswith("config:engram_activation__")

    @staticmethod
    def _infer_decision_subject(content: str) -> str | None:
        lowered = content.lower()
        if "engram" in lowered or "openclaw" in lowered or "full mode" in lowered:
            return "Engram"
        if "project" in lowered or "repo" in lowered:
            return "Project"
        return None

    async def _index_entity_with_structure(
        self,
        entity: Entity,
        group_id: str,
    ) -> None:
        """Build structure-aware embedding text that includes relationship predicates."""
        rels = await self._graph.get_relationships(
            entity.id,
            active_only=True,
            group_id=group_id,
        )

        # Sort by predicate weight (high-signal predicates first)
        predicate_weights = self._cfg.predicate_weights
        default_weight = self._cfg.predicate_weight_default
        rels_sorted = sorted(
            rels,
            key=lambda r: predicate_weights.get(r.predicate, default_weight),
            reverse=True,
        )

        natural_names = self._cfg.predicate_natural_names
        max_rels = self._cfg.structure_max_relationships
        rel_parts: list[str] = []
        for r in rels_sorted[:max_rels]:
            pred_natural = natural_names.get(r.predicate, r.predicate.lower().replace("_", " "))
            if r.source_id == entity.id:
                target = await self._graph.get_entity(r.target_id, group_id)
                target_name = target.name if target else r.target_id
                rel_parts.append(f"{entity.name} {pred_natural} {target_name}")
            else:
                source = await self._graph.get_entity(r.source_id, group_id)
                source_name = source.name if source else r.source_id
                rel_parts.append(f"{source_name} {pred_natural} {entity.name}")

        # Build enriched text: "{name}. {type}. {summary}. Relationships: ..."
        parts = [entity.name]
        if entity.entity_type:
            parts.append(entity.entity_type)
        if entity.summary:
            parts.append(entity.summary)

        text = ". ".join(parts) + "."
        if rel_parts:
            text += " Relationships: " + ", ".join(rel_parts)

        enriched = Entity(
            id=entity.id,
            name=text,
            entity_type=entity.entity_type,
            summary=None,
            group_id=entity.group_id,
        )
        await self._search.index_entity(enriched)

    def _truncate_episode_content(self, ep: Episode, cue=None) -> str:
        """Truncate episode content based on memory tier.

        When tier-aware truncation is disabled, uses flat recall_episode_content_limit.
        When enabled: episodic uses recall_episode_content_limit,
        transitional uses recall_transitional_content_limit,
        semantic uses recall_semantic_content_limit.
        For transitional/semantic with a cue, prefer cue.cue_text over raw content.
        """
        limit = self._cfg.recall_episode_content_limit

        if self._cfg.recall_tier_aware_truncation_enabled:
            tier = getattr(ep, "memory_tier", "episodic") or "episodic"
            if tier == "transitional":
                limit = self._cfg.recall_transitional_content_limit
                # Prefer cue text for transitional tier
                if cue is not None and hasattr(cue, "cue_text") and cue.cue_text:
                    content = cue.cue_text
                    return content[:limit] if limit > 0 else content
            elif tier == "semantic":
                limit = self._cfg.recall_semantic_content_limit
                # Prefer cue text for semantic tier
                if cue is not None and hasattr(cue, "cue_text") and cue.cue_text:
                    content = cue.cue_text
                    return content[:limit] if limit > 0 else content

        content = ep.content
        if limit > 0:
            return content[:limit]
        return content

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
        # Request extra results for near-miss detection (Wave 2)
        fetch_limit = limit
        if self._cfg.conv_near_miss_enabled and self._conv_context is not None:
            fetch_limit = limit + self._cfg.conv_near_miss_window

        # Ranking feedback should only learn from true usage, not passive surfacing.
        record_feedback = record_access
        if interaction_type in {"surfaced", "selected", "dismissed", "corrected"}:
            record_feedback = False
        elif interaction_type in {"used", "confirmed"}:
            record_feedback = True

        scored_results = await retrieve(
            query=query,
            group_id=group_id,
            graph_store=self._graph,
            activation_store=self._activation,
            search_index=self._search,
            cfg=self._cfg,
            limit=fetch_limit,
            working_memory=self._working_memory,
            reranker=self._reranker,
            community_store=self._community_store,
            predicate_cache=self._predicate_cache,
            conv_context=self._conv_context,
            priming_buffer=self._priming_buffer if self._cfg.retrieval_priming_enabled else None,
            goal_cache=self._goal_priming_cache,
            record_feedback=record_feedback,
            memory_need=memory_need,
        )

        # Split primary results and near-misses (Wave 2)
        primary_results = scored_results[:limit]
        near_miss_results = scored_results[limit:] if self._cfg.conv_near_miss_enabled else []

        now = time.time()
        results = []
        seen_episode_ids: set[str] = set()
        for sr in primary_results:
            if sr.result_type in {"episode", "cue_episode"}:
                if sr.node_id in seen_episode_ids:
                    continue
                # Fetch episode data — do NOT record access for episodes
                ep = await self._graph.get_episode_by_id(sr.node_id, group_id)
                if ep:
                    if (
                        self._episode_projection_state_value(ep)
                        == EpisodeProjectionState.MERGED.value
                    ):
                        continue
                    seen_episode_ids.add(ep.id)
                    linked_entities = await self._graph.get_episode_entities(sr.node_id)

                    # Populate working memory buffer for episodes
                    if self._working_memory is not None:
                        self._working_memory.add(
                            sr.node_id,
                            "episode",
                            sr.score,
                            query,
                            now,
                        )

                    if sr.result_type == "cue_episode":
                        cue = None
                        if hasattr(self._graph, "get_episode_cue"):
                            cue = await self._graph.get_episode_cue(sr.node_id, group_id)
                        if cue is None:
                            continue

                        await self._record_cue_hit(
                            ep,
                            sr.score,
                            query,
                            interaction_type=interaction_type,
                        )
                        cue = await self._get_episode_cue(ep.id, group_id) or cue
                        results.append(
                            {
                                "cue": self._cue_result_payload(cue, hit_increment=1),
                                "episode": {
                                    "id": ep.id,
                                    "source": ep.source,
                                    "created_at": (
                                        ep.created_at.isoformat() if ep.created_at else None
                                    ),
                                    "conversation_date": (
                                        ep.conversation_date.isoformat()
                                        if ep.conversation_date
                                        else None
                                    ),
                                },
                                "score": sr.score,
                                "score_breakdown": {
                                    "semantic": sr.semantic_similarity,
                                    "activation": sr.activation,
                                    "edge_proximity": sr.edge_proximity,
                                    "exploration_bonus": sr.exploration_bonus,
                                },
                                "result_type": "cue_episode",
                                "linked_entities": linked_entities,
                            }
                        )
                        continue

                    ep_result: dict = {
                        "episode": {
                            "id": ep.id,
                            "content": self._truncate_episode_content(ep),
                            "source": ep.source,
                            "created_at": ep.created_at.isoformat() if ep.created_at else None,
                            "conversation_date": (
                                ep.conversation_date.isoformat()
                                if ep.conversation_date
                                else None
                            ),
                        },
                        "score": sr.score,
                        "score_breakdown": {
                            "semantic": sr.semantic_similarity,
                            "activation": sr.activation,
                            "edge_proximity": sr.edge_proximity,
                            "exploration_bonus": sr.exploration_bonus,
                        },
                        "result_type": "episode",
                        "linked_entities": linked_entities,
                    }
                    if sr.chunk_context:
                        ep_result["chunk_context"] = sr.chunk_context
                    results.append(ep_result)
            else:
                entity = await self._graph.get_entity(sr.node_id, group_id)
                if entity:
                    rels = await self._graph.get_relationships(sr.node_id, group_id=group_id)

                    # Record access only for true recall usage, not passive surfacing.
                    if record_access:
                        await self._record_entity_access(
                            entity,
                            group_id=group_id,
                            query=query,
                            source=interaction_source,
                            timestamp=now,
                        )

                    # Populate working memory buffer for entities
                    if self._working_memory is not None:
                        self._working_memory.add(
                            sr.node_id,
                            "entity",
                            sr.score,
                            query,
                            now,
                        )

                    result_dict = {
                        "result_type": "entity",
                        "entity": {
                            "id": entity.id,
                            "name": entity.name,
                            "type": entity.entity_type,
                            "summary": entity.summary,
                        },
                        "score": sr.score,
                        "score_breakdown": {
                            "semantic": sr.semantic_similarity,
                            "activation": sr.activation,
                            "edge_proximity": sr.edge_proximity,
                            "exploration_bonus": sr.exploration_bonus,
                            "hop_distance": sr.hop_distance,
                            "planner_support": sr.planner_support,
                        },
                        "relationships": [
                            {
                                "id": r.id,
                                "predicate": r.predicate,
                                "source_id": r.source_id,
                                "target_id": r.target_id,
                                "weight": r.weight,
                                "polarity": r.polarity,
                            }
                            for r in rels[:5]
                        ],
                    }
                    if sr.planner_intents:
                        result_dict["supporting_intents"] = sr.planner_intents
                    if sr.recall_trace:
                        result_dict["recall_trace"] = sr.recall_trace

                    # Add warmth metadata for Intention entities
                    if entity.entity_type == "Intention" and self._cfg.prospective_graph_embedded:
                        try:
                            from engram.models.prospective import IntentionMeta

                            meta = IntentionMeta(**(entity.attributes or {}))
                            warmth_ratio = (
                                sr.activation / meta.activation_threshold
                                if meta.activation_threshold > 0
                                else 0.0
                            )
                            result_dict["intention_meta"] = {
                                "warmth_ratio": round(warmth_ratio, 4),
                                "fire_count": meta.fire_count,
                                "max_fires": meta.max_fires,
                                "action_text": meta.action_text,
                                "priority": meta.priority,
                            }
                        except Exception:
                            pass

                    if interaction_type and (
                        self._cfg.recall_telemetry_enabled
                        or self._cfg.recall_usage_feedback_enabled
                    ):
                        publish_memory_interaction(
                            self._event_bus,
                            MemoryInteractionEvent(
                                group_id=group_id,
                                entity_id=entity.id,
                                entity_name=entity.name,
                                entity_type=entity.entity_type,
                                interaction_type=interaction_type,
                                source=interaction_source,
                                query=query,
                                score=sr.score,
                                recorded_access=record_access,
                            ),
                        )
                    if interaction_type:
                        self._recall_need_controller.record_interaction(
                            group_id,
                            interaction_type,
                            result_type="entity",
                        )

                    results.append(result_dict)

        # --- Entity-linked episode traversal ---
        # For top-scoring entities in results, follow graph links to find
        # additional episodes where those entities were mentioned.
        if self._cfg.entity_episode_traversal_enabled:
            entity_scores: list[tuple[str, float]] = []
            for r in results:
                if r.get("result_type") == "entity":
                    ent_payload = r.get("entity")
                    if isinstance(ent_payload, dict) and ent_payload.get("id"):
                        entity_scores.append((ent_payload["id"], r.get("score", 0.0)))
            # Sort by score descending and take top-N entities
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            top_entities = entity_scores[: self._cfg.entity_episode_max_entities]

            for ent_id, ent_score in top_entities:
                try:
                    linked_ep_ids = await self._graph.get_episodes_for_entity(
                        ent_id,
                        group_id=group_id,
                        limit=self._cfg.entity_episode_max_per_entity,
                    )
                except Exception:
                    continue
                for ep_id in linked_ep_ids:
                    if ep_id in seen_episode_ids:
                        continue
                    ep = await self._graph.get_episode_by_id(ep_id, group_id)
                    if not ep:
                        continue
                    if (
                        self._episode_projection_state_value(ep)
                        == EpisodeProjectionState.MERGED.value
                    ):
                        continue
                    seen_episode_ids.add(ep.id)
                    linked_entities = await self._graph.get_episode_entities(ep.id)
                    traversal_score = ent_score * self._cfg.entity_episode_weight
                    results.append(
                        {
                            "episode": {
                                "id": ep.id,
                                "content": self._truncate_episode_content(ep),
                                "source": ep.source,
                                "created_at": (
                                    ep.created_at.isoformat() if ep.created_at else None
                                ),
                                "conversation_date": (
                                    ep.conversation_date.isoformat()
                                    if ep.conversation_date
                                    else None
                                ),
                            },
                            "score": traversal_score,
                            "score_breakdown": {
                                "semantic": 0.0,
                                "activation": 0.0,
                                "edge_proximity": 0.0,
                                "exploration_bonus": 0.0,
                                "entity_traversal": True,
                                "parent_entity_id": ent_id,
                            },
                            "result_type": "episode",
                            "linked_entities": linked_entities,
                        }
                    )

        # --- Temporal contiguity effect ---
        # For top-scoring recalled episodes, fetch adjacent episodes from same session
        if self._cfg.temporal_contiguity_enabled:
            episode_items: list[tuple[str, float]] = []
            for r in results:
                if r.get("result_type") == "episode":
                    ep_payload = r.get("episode")
                    if isinstance(ep_payload, dict) and ep_payload.get("id"):
                        episode_items.append((ep_payload["id"], r.get("score", 0.0)))
            # Sort by score descending and take top-N
            episode_items.sort(key=lambda x: x[1], reverse=True)
            top_episodes = episode_items[: self._cfg.temporal_contiguity_max_adjacent]

            for ep_id, ep_score in top_episodes:
                try:
                    adjacent = await self._graph.get_adjacent_episodes(
                        ep_id,
                        group_id=group_id,
                        limit=self._cfg.temporal_contiguity_max_adjacent,
                    )
                except Exception:
                    continue
                for adj_ep in adjacent:
                    if adj_ep.id in seen_episode_ids:
                        continue
                    if (
                        self._episode_projection_state_value(adj_ep)
                        == EpisodeProjectionState.MERGED.value
                    ):
                        continue
                    seen_episode_ids.add(adj_ep.id)
                    linked_entities = await self._graph.get_episode_entities(adj_ep.id)
                    contiguity_score = ep_score * self._cfg.temporal_contiguity_weight
                    results.append(
                        {
                            "episode": {
                                "id": adj_ep.id,
                                "content": self._truncate_episode_content(adj_ep),
                                "source": adj_ep.source,
                                "created_at": (
                                    adj_ep.created_at.isoformat()
                                    if adj_ep.created_at
                                    else None
                                ),
                                "conversation_date": (
                                    getattr(adj_ep, "conversation_date", None).isoformat()
                                    if getattr(adj_ep, "conversation_date", None)
                                    else None
                                ),
                            },
                            "score": contiguity_score,
                            "score_breakdown": {
                                "semantic": 0.0,
                                "activation": 0.0,
                                "edge_proximity": 0.0,
                                "exploration_bonus": 0.0,
                                "temporal_contiguity": True,
                                "parent_episode_id": ep_id,
                            },
                            "result_type": "episode",
                            "linked_entities": linked_entities,
                        }
                    )

        if self._query_prefers_current_state(query) and any(
            result.get("result_type") == "entity" for result in results
        ):
            results = [
                result
                for result in results
                if result.get("result_type") not in {"episode", "cue_episode"}
            ]

        # Track the query in working memory
        if self._working_memory is not None:
            self._working_memory.add_query(query, now)

        # Retrieval priming (Wave 3): boost 1-hop neighbors for follow-up queries
        if self._cfg.retrieval_priming_enabled and results:
            priming_now = time.time()
            expiry = priming_now + self._cfg.retrieval_priming_ttl_seconds
            for r in results[: self._cfg.retrieval_priming_top_n]:
                if r.get("result_type") != "entity":
                    continue
                entity_payload = r.get("entity")
                entity_id = entity_payload.get("id") if isinstance(entity_payload, dict) else None
                if not entity_id:
                    continue
                try:
                    neighbors = await self._graph.get_active_neighbors_with_weights(
                        entity_id,
                        group_id,
                    )
                    for neighbor_info in neighbors[: self._cfg.retrieval_priming_max_neighbors]:
                        nid = neighbor_info[0]
                        weight = neighbor_info[1]
                        self._priming_buffer[nid] = (
                            self._cfg.retrieval_priming_boost * weight,
                            expiry,
                        )
                except Exception:
                    pass

        # Build near-miss list (Wave 2)
        self._last_near_misses = []
        if near_miss_results:
            for nm in near_miss_results:
                if nm.result_type == "entity":
                    entity = await self._graph.get_entity(nm.node_id, group_id)
                    if entity:
                        self._last_near_misses.append(
                            {
                                "result_type": "entity",
                                "entity": {"name": entity.name, "type": entity.entity_type},
                                "score": round(nm.score, 4),
                            }
                        )
                    continue

                if nm.result_type != "cue_episode":
                    continue

                episode = await self._graph.get_episode_by_id(nm.node_id, group_id)
                cue = await self._get_episode_cue(nm.node_id, group_id)
                if episode is None or cue is None:
                    continue
                if (
                    self._episode_projection_state_value(episode)
                    == EpisodeProjectionState.MERGED.value
                ):
                    continue

                await self._record_cue_hit(
                    episode,
                    nm.score,
                    query,
                    interaction_type=interaction_type,
                    near_miss=True,
                )
                cue = await self._get_episode_cue(episode.id, group_id) or cue
                self._last_near_misses.append(
                    {
                        "result_type": "cue_episode",
                        "cue": self._cue_result_payload(cue),
                        "score": round(nm.score, 4),
                    }
                )

        # --- Relevance confidence scoring ---
        if self._cfg.relevance_confidence_enabled and results:
            try:
                from engram.retrieval.relevance import RelevanceScorer

                provider = getattr(self._search, "_provider", None)
                if provider and provider.dimension() > 0:
                    scorer = RelevanceScorer(provider)

                    # Collect text inputs for scoring
                    entity_summaries: dict[str, str] = {}
                    episode_contents: dict[str, str] = {}
                    chunk_texts: dict[str, str] = {}

                    for r in results:
                        if r.get("result_type") == "entity":
                            ent = r.get("entity", {})
                            if ent.get("id") and ent.get("summary"):
                                entity_summaries[ent["id"]] = ent["summary"]
                        elif r.get("result_type") in {"episode", "cue_episode"}:
                            ep = r.get("episode", {})
                            ep_id = ep.get("id", "")
                            if ep_id and ep.get("content"):
                                episode_contents[ep_id] = ep["content"]
                            chunk = r.get("chunk_context") or ""
                            if not chunk and r.get("result_type") == "cue_episode":
                                cue_data = r.get("cue", {})
                                chunk = cue_data.get("compressed_content", "")
                            if ep_id and chunk:
                                chunk_texts[ep_id] = chunk

                    # Reuse query_vec from search to avoid redundant embed call
                    query_vec = getattr(self._search, "_last_query_vec", None)

                    # Build temporary ScoredResult list for scoring
                    from engram.retrieval.scorer import ScoredResult

                    temp_scored: list[ScoredResult] = []
                    for r in results:
                        rt = r.get("result_type", "entity")
                        bd = r.get("score_breakdown", {})
                        node_id = ""
                        if rt == "entity":
                            node_id = r.get("entity", {}).get("id", "")
                        elif rt in {"episode", "cue_episode"}:
                            node_id = r.get("episode", {}).get("id", "")
                        temp_scored.append(
                            ScoredResult(
                                node_id=node_id,
                                score=r.get("score", 0.0),
                                semantic_similarity=bd.get("semantic", 0.0),
                                activation=bd.get("activation", 0.0),
                                spreading=0.0,
                                edge_proximity=bd.get("edge_proximity", 0.0),
                                result_type=rt,
                            )
                        )

                    await scorer.score_results(
                        query=query,
                        results=temp_scored,
                        entity_summaries=entity_summaries,
                        episode_contents=episode_contents,
                        chunk_texts=chunk_texts,
                        query_vec=query_vec,
                    )

                    # Write relevance back into result dicts
                    for r, ts in zip(results, temp_scored):
                        bd = r.get("score_breakdown")
                        if isinstance(bd, dict):
                            bd["relevance_confidence"] = round(ts.relevance_confidence, 4)
            except Exception:
                logger.debug("Relevance scoring failed, continuing without it", exc_info=True)

        # Update conversation fingerprint (Wave 2)
        if self._conv_context is not None and self._cfg.conv_fingerprint_enabled:
            from engram.retrieval.context import ConversationFingerprinter

            embed_fn = None
            provider = getattr(self._search, "_provider", None)
            if provider and hasattr(provider, "embed_query"):
                embed_fn = provider.embed_query
            await ConversationFingerprinter.ingest_turn(
                self._conv_context,
                query,
                embed_fn,
                source=f"recall_query:{interaction_source}",
                update_fingerprint=False,
            )

        return results

    # ─── Lightweight recall (fast entity probe) ─────────────────────

    # Pre-compiled patterns for entity mention extraction
    _RE_PROPER_NOUNS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
    _RE_QUOTED = re.compile(r'"([^"]{2,})"')
    _RE_AT_MENTION = re.compile(r"@(\w+)")
    _RE_HASHTAG = re.compile(r"#(\w+)")
    _RE_ALL_CAPS = re.compile(r"\b[A-Z]{2,}\b")

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
        if not text or not text.strip() or self._graph is None:
            return []

        # ── Step 1: Extract potential entity mentions from text ──────
        mentions: list[str] = []

        # Proper noun sequences (e.g. "John Smith", "React Router")
        mentions.extend(self._RE_PROPER_NOUNS.findall(text))

        # Quoted strings
        mentions.extend(self._RE_QUOTED.findall(text))

        # @-mentions and #hashtags
        mentions.extend(self._RE_AT_MENTION.findall(text))
        mentions.extend(self._RE_HASHTAG.findall(text))

        # Acronyms / all-caps tokens (e.g. "API", "AWS", "SQL")
        mentions.extend(self._RE_ALL_CAPS.findall(text))

        # Deduplicate while preserving order
        seen_lower: set[str] = set()
        unique_mentions: list[str] = []
        for m in mentions:
            key = m.strip().lower()
            if key and key not in seen_lower and len(key) >= 2:
                seen_lower.add(key)
                unique_mentions.append(m.strip())

        if not unique_mentions:
            return []

        now = time.time()
        if session_cache is None:
            session_cache = {}

        # ── Step 2-4: Probe graph for each mention ──────────────────
        tokens_per_entity = 40
        identity_core_results: list[dict] = []
        normal_results: list[dict] = []

        for mention in unique_mentions:
            # Find best matching entity via FTS5
            candidates = await self._graph.find_entity_candidates(
                mention, group_id, limit=3
            )
            if not candidates:
                continue

            entity = candidates[0]

            # Check session cache
            if entity.id in session_cache:
                ts, cached_result = session_cache[entity.id]
                if now - ts < cache_ttl:
                    if cached_result.get("identity_core"):
                        identity_core_results.append(cached_result)
                    else:
                        normal_results.append(cached_result)
                    continue

            # Fetch top relationships (limit=3 for compactness)
            rels = await self._graph.get_relationships(
                entity.id, group_id=group_id
            )
            top_rels = rels[:3]

            # Resolve target/source names for fact strings
            top_facts: list[str] = []
            for rel in top_rels:
                # Determine the "other" entity in the relationship
                other_id = (
                    rel.target_id if rel.source_id == entity.id else rel.source_id
                )
                other_name = await self.resolve_entity_name(other_id, group_id)
                if rel.source_id == entity.id:
                    top_facts.append(f"{rel.predicate} {other_name}")
                else:
                    top_facts.append(f"{other_name} {rel.predicate}")

            # Determine confidence tier from entity memory tier
            attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
            mat_tier = attrs.get("mat_tier", "episodic")
            if mat_tier == "semantic":
                confidence = "known"
            elif mat_tier == "transitional":
                confidence = "likely"
            else:
                confidence = "recent"

            freshness = _freshness_label(getattr(entity, "updated_at", None))
            result_dict = {
                "name": entity.name,
                "type": entity.entity_type,
                "summary": (entity.summary or "")[:120],
                "confidence": confidence,
                "identity_core": bool(getattr(entity, "identity_core", False)),
                "top_facts": top_facts,
                "freshness": freshness,
            }

            # Update session cache
            session_cache[entity.id] = (now, result_dict)

            if result_dict["identity_core"]:
                identity_core_results.append(result_dict)
            else:
                normal_results.append(result_dict)

        # ── Step 7: Pack into token budget ───────────────────────────
        # Identity-core entities are always included (free against budget)
        results: list[dict] = list(identity_core_results)
        remaining_budget = token_budget  # identity_core is free

        for entry in normal_results:
            if remaining_budget < tokens_per_entity:
                break
            results.append(entry)
            remaining_budget -= tokens_per_entity

        # If over budget, truncate summaries on non-identity entries
        if remaining_budget < 0:
            for entry in results:
                if not entry.get("identity_core"):
                    entry["summary"] = (entry.get("summary") or "")[:60]

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
        if not text or not text.strip() or self._graph is None:
            return []

        # ── Step 1: Extract entity mentions (shared with recall_lite) ──
        mentions: list[str] = []
        mentions.extend(self._RE_PROPER_NOUNS.findall(text))
        mentions.extend(self._RE_QUOTED.findall(text))
        mentions.extend(self._RE_AT_MENTION.findall(text))
        mentions.extend(self._RE_HASHTAG.findall(text))
        mentions.extend(self._RE_ALL_CAPS.findall(text))

        seen_lower: set[str] = set()
        unique_mentions: list[str] = []
        for m in mentions:
            key = m.strip().lower()
            if key and key not in seen_lower and len(key) >= 2:
                seen_lower.add(key)
                unique_mentions.append(m.strip())

        if not unique_mentions:
            return []

        now = time.time()
        if session_cache is None:
            session_cache = {}

        # ── Step 2: FTS5 candidate generation ──────────────────────────
        all_candidates: list[tuple[str, object, int]] = []  # (mention, entity, fts_rank)
        seen_entity_ids: set[str] = set()

        for mention in unique_mentions:
            candidates = await self._graph.find_entity_candidates(
                mention, group_id, limit=3
            )
            for rank, entity in enumerate(candidates):
                if entity.id not in seen_entity_ids:
                    seen_entity_ids.add(entity.id)
                    all_candidates.append((mention, entity, rank))

        if not all_candidates:
            return []

        # ── Step 3: Embedding rerank ───────────────────────────────────
        entity_ids = [eid for eid in seen_entity_ids]
        sim_scores: dict[str, float] = {}
        if self._search is not None:
            try:
                sim_scores = await self._search.compute_similarity(
                    text, entity_ids, group_id
                )
            except Exception:
                logger.debug("recall_medium embedding rerank failed", exc_info=True)

        # Score: combine FTS rank position with embedding similarity
        scored: list[tuple[float, str, object]] = []
        for _mention, entity, fts_rank in all_candidates:
            fts_score = 1.0 / (1 + fts_rank)  # rank 0 → 1.0, rank 1 → 0.5, etc.
            emb_score = sim_scores.get(entity.id, 0.0)
            final = fts_weight * fts_score + vec_weight * emb_score
            scored.append((final, entity.id, entity))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Step 4: Build result dicts (same format as recall_lite) ────
        tokens_per_entity = 40
        identity_core_results: list[dict] = []
        normal_results: list[dict] = []

        for _score, entity_id, entity in scored:
            # Check session cache
            if entity_id in session_cache:
                ts, cached_result = session_cache[entity_id]
                if now - ts < cache_ttl:
                    if cached_result.get("identity_core"):
                        identity_core_results.append(cached_result)
                    else:
                        normal_results.append(cached_result)
                    continue

            rels = await self._graph.get_relationships(
                entity_id, group_id=group_id
            )
            top_facts: list[str] = []
            for rel in rels[:3]:
                other_id = (
                    rel.target_id if rel.source_id == entity_id else rel.source_id
                )
                other_name = await self.resolve_entity_name(other_id, group_id)
                if rel.source_id == entity_id:
                    top_facts.append(f"{rel.predicate} {other_name}")
                else:
                    top_facts.append(f"{other_name} {rel.predicate}")

            attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
            mat_tier = attrs.get("mat_tier", "episodic")
            if mat_tier == "semantic":
                confidence = "known"
            elif mat_tier == "transitional":
                confidence = "likely"
            else:
                confidence = "recent"

            freshness = _freshness_label(getattr(entity, "updated_at", None))
            result_dict = {
                "name": entity.name,
                "type": entity.entity_type,
                "summary": (entity.summary or "")[:120],
                "confidence": confidence,
                "identity_core": bool(getattr(entity, "identity_core", False)),
                "top_facts": top_facts,
                "freshness": freshness,
            }

            session_cache[entity_id] = (now, result_dict)

            if result_dict["identity_core"]:
                identity_core_results.append(result_dict)
            else:
                normal_results.append(result_dict)

        # ── Step 5: Pack into token budget ─────────────────────────────
        results: list[dict] = list(identity_core_results)
        remaining_budget = token_budget

        for entry in normal_results:
            if remaining_budget < tokens_per_entity:
                break
            results.append(entry)
            remaining_budget -= tokens_per_entity

        if remaining_budget < 0:
            for entry in results:
                if not entry.get("identity_core"):
                    entry["summary"] = (entry.get("summary") or "")[:60]

        return results

    def _query_prefers_current_state(self, query: str) -> bool:
        """Detect narrow queries asking for the current/latest state."""
        tokens = {match.group(0) for match in re.finditer(r"[a-z]+", query.lower())}
        if not tokens:
            return False
        return bool(tokens & {"now", "current", "currently"})

    # ─── Entity name resolution ─────────────────────────────────────

    async def resolve_entity_name(self, entity_id: str, group_id: str) -> str:
        """Resolve an entity ID to its name. Returns ID if not found."""
        entity = await self._graph.get_entity(entity_id, group_id)
        return entity.name if entity else entity_id

    # ─── Search entities ────────────────────────────────────────────

    async def search_entities(
        self,
        group_id: str = "default",
        name: str | None = None,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search entities by name (FTS5 fuzzy) and/or type. Read-only, does not record access."""
        from engram.activation.engine import compute_activation

        now = time.time()
        entities: list[Entity] = []

        if name:
            # Use FTS5 for fuzzy matching, then optionally filter by type
            search_hits = await self._search.search(query=name, group_id=group_id, limit=limit * 2)
            for eid, _score in search_hits:
                ent = await self._graph.get_entity(eid, group_id)
                if ent and (not entity_type or ent.entity_type == entity_type):
                    entities.append(ent)
                if len(entities) >= limit:
                    break

            # Fallback: if search returned nothing, try direct name lookup
            if not entities:
                entities = await self._graph.find_entities(
                    name=name, entity_type=entity_type, group_id=group_id, limit=limit
                )
        else:
            # Type-only search
            entities = await self._graph.find_entities(
                entity_type=entity_type, group_id=group_id, limit=limit
            )

        result = []
        entity_ids = [e.id for e in entities]
        states = await self._activation.batch_get(entity_ids)

        for entity in entities:
            state = states.get(entity.id)
            activation_score = 0.0
            access_count = 0
            if state:
                activation_score = compute_activation(state.access_history, now, self._cfg)
                access_count = state.access_count

            result.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "summary": entity.summary,
                    "lexical_regime": entity.lexical_regime,
                    "canonical_identifier": entity.canonical_identifier,
                    "identifier_label": entity.identifier_label,
                    "activation_score": round(activation_score, 4),
                    "created_at": entity.created_at.isoformat() if entity.created_at else None,
                    "updated_at": entity.updated_at.isoformat() if entity.updated_at else None,
                    "access_count": access_count,
                }
            )

        return result

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
        """Search for relationships/facts. Resolves entity names."""
        # Normalize predicate filter
        if predicate:
            predicate = predicate.upper().replace(" ", "_")

        relationships: list[Relationship] = []

        if subject:
            # Resolve subject name to entity ID
            subject_entities = await self._graph.find_entities(
                name=subject, group_id=group_id, limit=1
            )
            if not subject_entities:
                # Try FTS5 as fallback
                hits = await self._search.search(query=subject, group_id=group_id, limit=5)
                for eid, _ in hits:
                    ent = await self._graph.get_entity(eid, group_id)
                    if ent and ent.name.lower() == subject.lower():
                        subject_entities = [ent]
                        break

            if subject_entities:
                subject_id = subject_entities[0].id
                rels = await self._graph.get_relationships(
                    subject_id,
                    direction="outgoing",
                    predicate=predicate,
                    active_only=not include_expired,
                    group_id=group_id,
                )
                relationships.extend(rels)
        else:
            # FTS5 search, get relationships for top hits
            search_hits = await self._search.search(query=query, group_id=group_id, limit=limit)
            seen_rel_ids: set[str] = set()
            for eid, _ in search_hits:
                rels = await self._graph.get_relationships(
                    eid,
                    direction="both",
                    predicate=predicate,
                    active_only=not include_expired,
                    group_id=group_id,
                )
                for r in rels:
                    if r.id not in seen_rel_ids:
                        seen_rel_ids.add(r.id)
                        relationships.append(r)
                if len(relationships) >= limit:
                    break

        # Resolve entity names and format results
        result = []
        for r in relationships:
            if not include_epistemic and await self._relationship_is_epistemic(
                r,
                group_id=group_id,
            ):
                continue
            source_name = await self.resolve_entity_name(r.source_id, group_id)
            target_name = await self.resolve_entity_name(r.target_id, group_id)
            result.append(
                {
                    "subject": source_name,
                    "predicate": r.predicate,
                    "object": target_name,
                    "polarity": r.polarity,
                    "valid_from": r.valid_from.isoformat() if r.valid_from else None,
                    "valid_to": r.valid_to.isoformat() if r.valid_to else None,
                    "confidence": r.confidence,
                    "source_episode": r.source_episode,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
            )
            if len(result) >= limit:
                break

        return result

    async def _relationship_is_epistemic(
        self,
        relationship: Relationship,
        *,
        group_id: str,
    ) -> bool:
        if relationship.predicate in _EPISTEMIC_FACT_PREDICATES:
            return True
        source_entity = await self._graph.get_entity(relationship.source_id, group_id)
        target_entity = await self._graph.get_entity(relationship.target_id, group_id)
        source_type = getattr(source_entity, "entity_type", None)
        target_type = getattr(target_entity, "entity_type", None)
        return source_type in {"Decision", "Artifact"} or target_type in {
            "Decision",
            "Artifact",
        }

    # ─── Forget entity ──────────────────────────────────────────────

    async def forget_entity(
        self,
        entity_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Soft-delete an entity and clear its activation."""
        entities = await self._graph.find_entities(name=entity_name, group_id=group_id, limit=1)
        if not entities:
            return {"status": "error", "message": f"Entity '{entity_name}' not found."}

        entity = entities[0]
        await self._graph.delete_entity(entity.id, soft=True, group_id=group_id)
        await self._activation.clear_activation(entity.id)

        logger.info("Forgot entity %s (%s), reason: %s", entity.name, entity.id, reason)
        return {
            "status": "forgotten",
            "target_type": "entity",
            "target": entity.name,
            "valid_to": utc_now_iso(),
            "message": f"Entity '{entity.name}' has been forgotten.",
        }

    # ─── Forget fact ────────────────────────────────────────────────

    async def forget_fact(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        group_id: str = "default",
        reason: str | None = None,
    ) -> dict:
        """Invalidate a specific relationship (fact)."""
        predicate = predicate.upper().replace(" ", "_")

        # Resolve entity names to IDs
        subject_entities = await self._graph.find_entities(
            name=subject_name, group_id=group_id, limit=1
        )
        object_entities = await self._graph.find_entities(
            name=object_name, group_id=group_id, limit=1
        )

        if not subject_entities:
            return {"status": "error", "message": f"Subject '{subject_name}' not found."}
        if not object_entities:
            return {"status": "error", "message": f"Object '{object_name}' not found."}

        subject_id = subject_entities[0].id
        object_id = object_entities[0].id

        rels = await self._graph.get_relationships(
            subject_id,
            direction="outgoing",
            predicate=predicate,
            active_only=True,
            group_id=group_id,
        )
        target_rel = None
        for r in rels:
            if r.target_id == object_id:
                target_rel = r
                break

        if not target_rel:
            return {
                "status": "error",
                "message": (f"No active fact found: {subject_name} —{predicate}→ {object_name}."),
            }

        await self._graph.invalidate_relationship(target_rel.id, utc_now(), group_id=group_id)

        logger.info(
            "Forgot fact %s —%s→ %s, reason: %s",
            subject_name,
            predicate,
            object_name,
            reason,
        )
        return {
            "status": "forgotten",
            "target_type": "fact",
            "subject": subject_name,
            "predicate": predicate,
            "object": object_name,
            "valid_to": utc_now_iso(),
            "message": f"Fact '{subject_name} {predicate} {object_name}' has been forgotten.",
        }

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
        """Create a new prospective memory intention.

        In v2 (graph-embedded), creates an Entity node with type 'Intention'
        and TRIGGERED_BY edges to related entities. Falls back to flat table
        when prospective_graph_embedded=False.

        For pinned contexts (trigger_type="refresh_context"), the trigger_text
        serves as the topic query. Set refresh_trigger="after_consolidation"
        to auto-refresh the cached result after each consolidation cycle.
        """
        if not self._cfg.prospective_graph_embedded:
            # v1 fallback: flat table — map v2 trigger_type to v1 equivalent
            v1_type = "semantic" if trigger_type == "activation" else trigger_type
            return await self._create_intention_v1(
                trigger_text,
                action_text,
                v1_type,
                entity_name,
                threshold,
                group_id,
            )

        if trigger_type not in ("activation", "entity_mention", "refresh_context"):
            raise ValueError(f"Invalid trigger_type: {trigger_type}")
        if trigger_type == "entity_mention" and not entity_names and not entity_name:
            raise ValueError("entity_names (or entity_name) required for entity_mention trigger")

        from datetime import timedelta

        from engram.models.prospective import IntentionMeta

        now = utc_now()
        intention_id = f"int_{uuid.uuid4().hex[:12]}"

        # Resolve entity_names to IDs and create TRIGGERED_BY edges
        linked_entity_ids: list[str] = []
        resolved_names = entity_names or ([entity_name] if entity_name else [])
        for name in resolved_names:
            candidates = await self._graph.find_entity_candidates(name, group_id)
            if candidates:
                linked_entity_ids.append(candidates[0].id)

        meta = IntentionMeta(
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            activation_threshold=threshold or self._cfg.prospective_activation_threshold,
            max_fires=self._cfg.prospective_max_fires,
            fire_count=0,
            enabled=True,
            expires_at=(now + timedelta(days=self._cfg.prospective_ttl_days)).isoformat(),
            trigger_entity_ids=linked_entity_ids,
            cooldown_seconds=self._cfg.prospective_cooldown_seconds,
            priority=priority,
            origin="explicit",
            context=context,
            see_also=see_also,
            refresh_trigger=refresh_trigger,
        )

        # Create as Entity node
        entity = Entity(
            id=intention_id,
            name=trigger_text,
            entity_type="Intention",
            summary=action_text,
            group_id=group_id,
            attributes=meta.model_dump(),
            created_at=now,
            updated_at=now,
        )
        await self._graph.create_entity(entity)

        # Index for FTS5 + vector search
        await self._search.index_entity(entity)

        # Create TRIGGERED_BY edges
        for eid in linked_entity_ids:
            rel = Relationship(
                id=f"rel_{uuid.uuid4().hex[:12]}",
                source_id=intention_id,
                target_id=eid,
                predicate="TRIGGERED_BY",
                weight=0.9,
                group_id=group_id,
                source_episode=None,
            )
            await self._graph.create_relationship(rel)

        # Record initial access
        await self._activation.record_access(intention_id, time.time(), group_id=group_id)

        # Publish event
        self._publish(
            group_id,
            "intention.created",
            {
                "intentionId": intention_id,
                "triggerText": trigger_text,
                "actionText": action_text,
                "linkedEntityIds": linked_entity_ids,
                "threshold": meta.activation_threshold,
            },
        )

        logger.info("Created graph-embedded intention %s: %s", intention_id, trigger_text)
        return intention_id

    async def _create_intention_v1(
        self,
        trigger_text: str,
        action_text: str,
        trigger_type: str = "semantic",
        entity_name: str | None = None,
        threshold: float | None = None,
        group_id: str = "default",
    ) -> str:
        """v1 fallback: create intention in flat table."""
        from engram.models.prospective import Intention

        if trigger_type not in ("semantic", "entity_mention"):
            raise ValueError(f"Invalid trigger_type: {trigger_type}")
        if trigger_type == "entity_mention" and not entity_name:
            raise ValueError("entity_name required for entity_mention trigger")

        from datetime import timedelta

        now = utc_now()
        intention = Intention(
            id=f"int_{uuid.uuid4().hex[:12]}",
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type=trigger_type,
            entity_name=entity_name,
            threshold=threshold or self._cfg.prospective_similarity_threshold,
            max_fires=self._cfg.prospective_max_fires,
            fire_count=0,
            enabled=True,
            group_id=group_id,
            created_at=now,
            updated_at=now,
            expires_at=now + timedelta(days=self._cfg.prospective_ttl_days),
        )
        return await self._graph.create_intention(intention)

    async def list_intentions(
        self,
        group_id: str = "default",
        enabled_only: bool = True,
    ) -> list:
        """List intentions. v2 uses Entity nodes, v1 uses flat table."""
        if not self._cfg.prospective_graph_embedded:
            return await self._graph.list_intentions(group_id, enabled_only=enabled_only)

        from engram.models.prospective import IntentionMeta

        entities = await self._graph.find_entities(
            entity_type="Intention",
            group_id=group_id,
            limit=100,
        )

        result = []
        now = utc_now()
        for entity in entities:
            attrs = entity.attributes or {}
            try:
                meta = IntentionMeta(**attrs)
            except Exception:
                continue

            if enabled_only:
                if not meta.enabled:
                    continue
                if meta.fire_count >= meta.max_fires:
                    continue
                if meta.expires_at:
                    try:
                        exp = datetime.fromisoformat(meta.expires_at)
                        if exp <= now:
                            continue
                    except (ValueError, TypeError):
                        pass

            result.append(entity)
        return result

    async def dismiss_intention(
        self,
        intention_id: str,
        group_id: str = "default",
        hard: bool = False,
    ) -> None:
        """Dismiss an intention. Soft-delete disables it; hard-delete removes the entity."""
        if not self._cfg.prospective_graph_embedded:
            await self._graph.delete_intention(intention_id, group_id, soft=not hard)
            return

        if hard:
            delete_entity = self._graph.delete_entity
            delete_signature = inspect.signature(delete_entity)
            if "group_id" in delete_signature.parameters:
                group_param = delete_signature.parameters["group_id"]
                if group_param.kind is inspect.Parameter.KEYWORD_ONLY:
                    await delete_entity(intention_id, soft=False, group_id=group_id)
                else:
                    await delete_entity(intention_id, group_id)
            else:
                await delete_entity(intention_id, group_id)
        else:
            entity = await self._graph.get_entity(intention_id, group_id)
            if entity:
                attrs = dict(entity.attributes or {})
                attrs["enabled"] = False
                await self._graph.update_entity(
                    intention_id,
                    {"attributes": attrs},
                    group_id=group_id,
                )

        self._publish(
            group_id,
            "intention.dismissed",
            {
                "intentionId": intention_id,
                "hard": hard,
            },
        )

    async def delete_intention(
        self,
        intention_id: str,
        group_id: str = "default",
    ) -> None:
        """Soft-delete an intention (backward compat)."""
        await self.dismiss_intention(intention_id, group_id, hard=False)

    async def migrate_flat_intentions(self, group_id: str = "default") -> int:
        """Migrate flat-table intentions to graph-embedded Entity nodes.

        Returns the number of migrated intentions.
        """
        flat_intentions = await self._graph.list_intentions(group_id, enabled_only=False)
        migrated = 0
        for intention in flat_intentions:
            try:
                await self.create_intention(
                    trigger_text=intention.trigger_text,
                    action_text=intention.action_text,
                    trigger_type=(
                        "entity_mention"
                        if intention.trigger_type == "entity_mention"
                        else "activation"
                    ),
                    entity_name=intention.entity_name,
                    group_id=group_id,
                )
                # Disable old flat-table entry
                await self._graph.delete_intention(intention.id, group_id, soft=True)
                migrated += 1
            except Exception:
                logger.warning("Failed to migrate intention %s", intention.id, exc_info=True)
        logger.info("Migrated %d flat-table intentions to graph-embedded", migrated)
        return migrated

    async def _update_intention_fire(
        self,
        intention_id: str,
        group_id: str,
        episode_id: str | None = None,
    ) -> None:
        """Increment fire count and update last_fired for a graph-embedded intention."""
        entity = await self._graph.get_entity(intention_id, group_id)
        if not entity:
            return
        attrs = dict(entity.attributes or {})
        attrs["fire_count"] = attrs.get("fire_count", 0) + 1
        attrs["last_fired"] = utc_now_iso()
        await self._graph.update_entity(
            intention_id,
            {"attributes": attrs},
            group_id=group_id,
        )
        self._publish(
            group_id,
            "intention.triggered",
            {
                "intentionId": intention_id,
                "triggerText": attrs.get("trigger_text", ""),
                "actionText": attrs.get("action_text", ""),
                "activation": 0.0,
                "episodeId": episode_id,
            },
        )

    async def update_intention_meta(
        self,
        intention_id: str,
        group_id: str,
        updates: dict,
    ) -> None:
        """Update specific fields in an intention's IntentionMeta attributes.

        Args:
            intention_id: The intention entity ID.
            group_id: The group ID.
            updates: Dict of IntentionMeta field names to new values.
        """
        entity = await self._graph.get_entity(intention_id, group_id)
        if not entity:
            return
        attrs = dict(entity.attributes or {})
        attrs.update(updates)
        await self._graph.update_entity(
            intention_id,
            {"attributes": attrs},
            group_id=group_id,
        )

    # ─── Get context ────────────────────────────────────────────────

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
        """Build context data dict for a single entity with activation and facts.

        detail_level controls rendering resolution:
        - "full": name + type + activation + summary + attributes + up to 5 facts
        - "summary": name + type + summary + up to 2 facts
        - "mention": name + type only
        """
        result: dict = {
            "name": name,
            "type": entity_type,
            "detail_level": detail_level,
            "id": entity_id,
        }

        if detail_level == "mention":
            result["activation"] = 0.0
            result["summary"] = None
            result["facts"] = []
            result["attributes"] = None
            return result

        from engram.activation.engine import compute_activation

        state = await self._activation.get_activation(entity_id)
        act = 0.0
        if state:
            act = compute_activation(state.access_history, now, self._cfg)
        result["activation"] = act
        result["summary"] = summary

        max_facts = 5 if detail_level == "full" else 2
        facts: list[str] = []
        rels = await self._graph.get_relationships(
            entity_id,
            active_only=True,
            group_id=group_id,
        )
        for r in rels[:max_facts]:
            src = await self.resolve_entity_name(r.source_id, group_id)
            tgt = await self.resolve_entity_name(r.target_id, group_id)
            facts.append(f"{src} {r.predicate} {tgt}")
        result["facts"] = facts

        # Only fetch attributes for full detail
        if detail_level == "full":
            entity = await self._graph.get_entity(entity_id, group_id)
            result["attributes"] = entity.attributes if entity else None
        else:
            result["attributes"] = None

        return result

    @staticmethod
    def _render_tier(header: str, entities: list[dict], facts: list[str]) -> str:
        """Render a single context tier as markdown with variable resolution.

        Each entity dict may have a 'detail_level' key:
        - "full": name + type + activation + summary + attributes + facts
        - "summary": name + type + summary + facts (no attributes)
        - "mention": name + type only
        """
        lines = [header, ""]
        for ed in entities:
            detail = ed.get("detail_level", "full")

            if detail == "mention":
                lines.append(f"- {ed['name']} ({ed['type']})")
                continue

            summary_part = f" — {ed['summary']}" if ed.get("summary") else ""
            # Append top attributes inline (full detail only)
            if detail == "full":
                attrs = ed.get("attributes")
                if attrs:
                    attr_parts = [f"{k}: {v}" for k, v in list(attrs.items())[:5]]
                    summary_part += f" [{', '.join(attr_parts)}]"
            lines.append(f"- {ed['name']} ({ed['type']}, act={ed['activation']:.2f}){summary_part}")
            # Render per-entity facts inline
            for fact in ed.get("facts", []):
                lines.append(f"  - {fact}")

        # Also render any tier-level facts not already covered by entities
        entity_facts = set()
        for ed in entities:
            entity_facts.update(ed.get("facts", []))
        extra_facts = [f for f in facts if f not in entity_facts]
        if extra_facts:
            for fact in extra_facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4

    def invalidate_briefing_cache(self, group_id: str) -> None:
        """Clear briefing cache entries for the given group."""
        keys_to_remove = [k for k in self._briefing_cache if k[0] == group_id]
        for k in keys_to_remove:
            del self._briefing_cache[k]

    def _template_briefing(
        self,
        structured_context: str,
        group_id: str,
        topic_hint: str | None,
    ) -> str:
        """Render a brief narrative from structured context using templates.

        No LLM call — deterministic, instant, always available.
        Parses tier sections from the structured markdown and formats them
        into 2-3 natural sentences.
        """
        cache_key = (group_id, topic_hint)
        now = time.time()
        if cache_key in self._briefing_cache:
            ts, text = self._briefing_cache[cache_key]
            if now - ts < self._cfg.briefing_cache_ttl_seconds:
                return text

        sentences: list[str] = []

        # Parse tier sections
        tier1_lines: list[str] = []
        tier2_lines: list[str] = []
        tier3_lines: list[str] = []
        current_tier: list[str] | None = None

        for line in structured_context.split("\n"):
            stripped = line.strip()
            if "Identity" in stripped and stripped.startswith("#"):
                current_tier = tier1_lines
            elif "Project" in stripped and stripped.startswith("#"):
                current_tier = tier2_lines
            elif ("Recent" in stripped or "Activity" in stripped) and stripped.startswith("#"):
                current_tier = tier3_lines
            elif "Intention" in stripped and stripped.startswith("#"):
                current_tier = None
            elif current_tier is not None and stripped.startswith("- "):
                current_tier.append(stripped[2:].strip())

        # Sentence 1: Identity
        if tier1_lines:
            # Pick the first few identity facts
            identity_facts = tier1_lines[:3]
            sentences.append("Known context: " + "; ".join(identity_facts) + ".")

        # Sentence 2: Current project/topic
        if tier2_lines:
            project_facts = tier2_lines[:3]
            prefix = f"Currently working on {topic_hint}: " if topic_hint else "Current focus: "
            sentences.append(prefix + "; ".join(project_facts) + ".")

        # Sentence 3: Recent activity
        if tier3_lines:
            recent_facts = tier3_lines[:3]
            sentences.append("Recent activity: " + "; ".join(recent_facts) + ".")

        if sentences:
            briefing = " ".join(sentences)
        else:
            briefing = structured_context

        self._briefing_cache[cache_key] = (now, briefing)
        return briefing

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        """Build a tiered markdown context summary of the most activated memories.

        Tiers:
        1. Identity Core — always-included identity entities + top relationships
        2. Project Context — topic-biased entities (from project_path or topic_hint)
        3. Recent Activity — top-activated entities filling remaining budget
        """
        from pathlib import Path

        from engram.activation.engine import compute_activation

        now = time.time()
        seen_ids: set[str] = set()

        # Derive topic_hint from project_path if not provided
        project_entity_id: str | None = None
        if project_path:
            p = Path(project_path).expanduser()
            if p.name and str(p) != str(Path.home()):
                if not topic_hint:
                    topic_hint = p.name
                # Auto-create Project entity if missing
                existing_projects = await self._graph.find_entities(
                    name=p.name,
                    entity_type="Project",
                    group_id=group_id,
                    limit=1,
                )
                if existing_projects:
                    project_entity_id = existing_projects[0].id
                else:
                    project_entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                    proj_entity = Entity(
                        id=project_entity_id,
                        name=p.name,
                        entity_type="Project",
                        summary=f"Software project at {project_path}",
                        attributes={"project_path": str(p)},
                        group_id=group_id,
                    )
                    await self._graph.create_entity(proj_entity)
                    await self._activation.record_access(
                        project_entity_id,
                        now,
                        group_id=group_id,
                    )

        # ── Layer 1: Identity Core ──
        layer1_entities: list[dict] = []
        layer1_facts: list[str] = []

        if self._cfg.identity_core_enabled and hasattr(self._graph, "get_identity_core_entities"):
            try:
                core_entities = await self._graph.get_identity_core_entities(group_id)
                for ce in core_entities:
                    ed = await self._entity_to_context_data(
                        ce.id,
                        ce.name,
                        ce.entity_type,
                        ce.summary or "",
                        group_id,
                        now,
                        detail_level="full",
                    )
                    layer1_entities.append(ed)
                    layer1_facts.extend(ed["facts"])
                    seen_ids.add(ce.id)
            except Exception:
                logger.debug("Identity core lookup failed (non-fatal)", exc_info=True)

        layer1_entities.sort(key=lambda x: x["activation"], reverse=True)
        layer1_text = self._render_tier("## Identity", layer1_entities, layer1_facts)

        # ── Layer 2: Project Context ──
        layer2_entities: list[dict] = []
        layer2_facts: list[str] = []

        if topic_hint:
            results = await self.recall(query=topic_hint, group_id=group_id, limit=15)
            for r in results:
                if r.get("result_type") in {"episode", "cue_episode"}:
                    continue
                ent = r.get("entity")
                if not ent:
                    continue
                if ent["id"] in seen_ids:
                    continue
                # Variable resolution based on hop distance
                hop = r.get("score_breakdown", {}).get("hop_distance")
                if hop is None or hop == 0:
                    detail = "full"
                elif hop == 1:
                    detail = "summary"
                else:
                    detail = "mention"
                ed = await self._entity_to_context_data(
                    ent["id"],
                    ent["name"],
                    ent["type"],
                    ent.get("summary") or "",
                    group_id,
                    now,
                    detail_level=detail,
                )
                layer2_entities.append(ed)
                layer2_facts.extend(ed["facts"])
                seen_ids.add(ent["id"])

        # Inject Project entity neighbors (PART_OF-connected entities)
        if project_entity_id:
            try:
                neighbors = await self._graph.get_neighbors(
                    project_entity_id,
                    hops=1,
                    group_id=group_id,
                )
                for neighbor_ent, _rel in neighbors:
                    if neighbor_ent.id in seen_ids:
                        continue
                    ed = await self._entity_to_context_data(
                        neighbor_ent.id,
                        neighbor_ent.name,
                        neighbor_ent.entity_type,
                        neighbor_ent.summary or "",
                        group_id,
                        now,
                        detail_level="summary",
                    )
                    layer2_entities.append(ed)
                    layer2_facts.extend(ed["facts"])
                    seen_ids.add(neighbor_ent.id)
            except Exception:
                logger.debug("Project neighbor injection failed (non-fatal)", exc_info=True)

        if layer2_entities:
            layer2_entities.sort(key=lambda x: x["activation"], reverse=True)

        if layer2_entities:
            layer2_text = self._render_tier(
                f"## Project Context ({topic_hint})",
                layer2_entities,
                layer2_facts,
            )
        else:
            layer2_text = ""

        # ── Layer 3: Recent Activity ──
        layer3_entities: list[dict] = []
        layer3_facts: list[str] = []

        top = await self._activation.get_top_activated(group_id=group_id, limit=20)
        for eid, state in top:
            if eid in seen_ids:
                continue
            entity = await self._graph.get_entity(eid, group_id)
            if not entity:
                continue
            act = compute_activation(state.access_history, now, self._cfg)
            ed = await self._entity_to_context_data(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                group_id,
                now,
                detail_level="summary",
            )
            ed["activation"] = act  # use fresh computation
            layer3_entities.append(ed)
            layer3_facts.extend(ed["facts"])
            seen_ids.add(eid)

        layer3_entities.sort(key=lambda x: x["activation"], reverse=True)
        layer3_text = self._render_tier("## Recent Activity", layer3_entities, layer3_facts)

        # ── Layer 4: Active Intentions ──
        layer4_text = ""
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            try:
                from engram.models.prospective import IntentionMeta

                intention_entities = await self.list_intentions(group_id)
                intention_lines: list[str] = []
                for ie in intention_entities:
                    attrs = ie.attributes or {}
                    try:
                        meta = IntentionMeta(**attrs)
                    except Exception:
                        continue

                    intention_state: ActivationState | None = await self._activation.get_activation(
                        ie.id,
                    )
                    act = 0.0
                    if intention_state:
                        act = compute_activation(
                            intention_state.access_history,
                            now,
                            self._cfg,
                        )
                    warmth_ratio = (
                        act / meta.activation_threshold if meta.activation_threshold > 0 else 0.0
                    )

                    # Filter out dormant intentions (below lowest warmth level)
                    levels = self._cfg.prospective_warmth_levels
                    if warmth_ratio < levels[0]:
                        continue

                    if warmth_ratio >= 1.0:
                        label = "HOT"
                    elif warmth_ratio >= levels[2]:
                        label = "warm"
                    elif warmth_ratio >= levels[1]:
                        label = "warming"
                    else:
                        label = "cool"

                    intention_lines.append(
                        f"- [{label}] {meta.trigger_text} → {meta.action_text} "
                        f"(fires: {meta.fire_count}/{meta.max_fires})"
                    )
                    seen_ids.add(ie.id)

                if intention_lines:
                    layer4_text = "## Active Intentions\n\n" + "\n".join(intention_lines)
            except Exception:
                logger.debug("Intention tier in get_context failed (non-fatal)", exc_info=True)

        # ── Layer 5: Pinned Contexts ──
        layer5_text = ""
        pinned_contexts: list[dict] = []
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            try:
                from engram.models.prospective import IntentionMeta as _IntMeta5

                pinned_entities = await self.list_intentions(group_id, enabled_only=True)
                pinned_lines: list[str] = []
                for pe in pinned_entities:
                    attrs = pe.attributes or {}
                    try:
                        pmeta = _IntMeta5(**attrs)
                    except Exception:
                        continue
                    if pmeta.trigger_type != "refresh_context":
                        continue
                    if not pmeta.pinned_result:
                        continue
                    pinned_contexts.append(
                        {
                            "topic": pmeta.trigger_text,
                            "result": pmeta.pinned_result,
                            "last_refreshed": pmeta.last_refreshed,
                        }
                    )
                    pinned_lines.append(
                        f"### {pmeta.trigger_text}\n{pmeta.pinned_result}"
                    )
                if pinned_lines:
                    layer5_text = "## Pinned Contexts\n\n" + "\n\n".join(pinned_lines)
            except Exception:
                logger.debug("Pinned context tier in get_context failed (non-fatal)", exc_info=True)

        # ── Assemble ──
        all_entities = layer1_entities + layer2_entities + layer3_entities
        all_facts = layer1_facts + layer2_facts + layer3_facts
        # Deduplicate facts
        seen_facts: set[str] = set()
        unique_facts: list[str] = []
        for f in all_facts:
            if f not in seen_facts:
                seen_facts.add(f)
                unique_facts.append(f)

        all_layers = [layer1_text, layer2_text, layer3_text, layer4_text, layer5_text]
        sections = [s for s in all_layers if s]
        context_text = (
            "\n\n".join(sections) if sections else "## Active Memory Context\n\nNo memories loaded."
        )

        # Token estimate and truncation
        token_estimate = self._estimate_tokens(context_text)
        if token_estimate > max_tokens:
            char_budget = max_tokens * 4
            context_text = context_text[:char_budget]
            token_estimate = max_tokens

        # Record access for included entities
        for ed in all_entities:
            await self._activation.record_access(ed["id"], now, group_id=group_id)
            await self._publish_access_event(
                ed["id"],
                ed["name"],
                ed["type"],
                group_id,
                "context",
            )

        # Briefing format
        if format == "briefing" and self._cfg.briefing_enabled and all_entities:
            briefing = self._template_briefing(context_text, group_id, topic_hint)
            result = {
                "context": briefing,
                "entity_count": len(all_entities),
                "fact_count": len(unique_facts),
                "token_estimate": self._estimate_tokens(briefing),
                "format": "briefing",
            }
            if pinned_contexts:
                result["pinned_contexts"] = pinned_contexts
            return result

        result = {
            "context": context_text,
            "entity_count": len(all_entities),
            "fact_count": len(unique_facts),
            "token_estimate": token_estimate,
            "format": "structured",
        }
        if pinned_contexts:
            result["pinned_contexts"] = pinned_contexts
        return result

    # ─── Get graph state ────────────────────────────────────────────

    async def get_graph_state(
        self,
        group_id: str = "default",
        top_n: int = 20,
        include_edges: bool = False,
        entity_types: list[str] | None = None,
    ) -> dict:
        """Return graph statistics and top-activated nodes."""
        from engram.activation.engine import compute_activation

        now = time.time()

        stats = await self._graph.get_stats(group_id)
        type_counts = await self._graph.get_entity_type_counts(group_id)
        stats["entity_type_distribution"] = type_counts

        # Get top activated
        top = await self._activation.get_top_activated(group_id=group_id, limit=top_n * 2)

        top_activated: list[_TopActivatedEntry] = []
        active_count = 0
        dormant_count = 0

        for eid, state in top:
            entity = await self._graph.get_entity(eid, group_id)
            if not entity:
                continue
            if entity_types and entity.entity_type not in entity_types:
                continue

            act = compute_activation(state.access_history, now, self._cfg)
            if act > 0.3:
                active_count += 1
            else:
                dormant_count += 1

            if len(top_activated) < top_n:
                top_activated.append(
                    {
                        "id": entity.id,
                        "name": entity.name,
                        "entity_type": entity.entity_type,
                        "summary": entity.summary,
                        "activation": round(act, 4),
                        "access_count": state.access_count,
                    }
                )

        stats["active_entities"] = active_count
        stats["dormant_entities"] = dormant_count
        stats["recall_metrics"] = self.get_recall_metrics(group_id)
        stats["epistemic_metrics"] = self.get_epistemic_metrics(group_id)

        result: dict = {
            "stats": stats,
            "top_activated": top_activated,
            "group_id": group_id,
        }

        if include_edges:
            edges = []
            seen_rel_ids: set[str] = set()
            for ta in top_activated:
                rels = await self._graph.get_relationships(
                    ta["id"], active_only=True, group_id=group_id
                )
                for r in rels:
                    if r.id not in seen_rel_ids:
                        seen_rel_ids.add(r.id)
                        source_name = await self.resolve_entity_name(r.source_id, group_id)
                        target_name = await self.resolve_entity_name(r.target_id, group_id)
                        edges.append(
                            {
                                "source": source_name,
                                "target": target_name,
                                "predicate": r.predicate,
                                "weight": r.weight,
                            }
                        )
            result["edges"] = edges

        return result
