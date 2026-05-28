"""Evidence materialization and adjudication resolution service."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.ambiguity import AmbiguityGroup
from engram.extraction.client_proposals import proposals_to_evidence
from engram.extraction.evidence import (
    CommitDecision,
    EvidenceCandidate,
    evidence_candidate_from_dict,
)
from engram.extraction.models import ApplyOutcome, ProjectionBundle, ProjectionPlan
from engram.models.adjudication import AdjudicationRequest
from engram.models.consolidation import RelationshipApplyResult
from engram.models.episode import Episode

logger = logging.getLogger(__name__)


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


class EvidenceAdjudicationService:
    """Own evidence projection materialization and adjudication resolution."""

    def __init__(
        self,
        *,
        graph_store: Any,
        search_index: Any,
        cfg: ActivationConfig,
        evidence_bridge: Any,
        apply_engine: Any,
        apply_bootstrap_part_of_edges: Any,
        index_entity_with_structure: Any,
        invalidate_briefing_cache: Any,
    ) -> None:
        self._graph = graph_store
        self._search = search_index
        self._cfg = cfg
        self._evidence_bridge = evidence_bridge
        self._apply_engine = apply_engine
        self._apply_bootstrap_part_of_edges = apply_bootstrap_part_of_edges
        self._index_entity_with_structure = index_entity_with_structure
        self._invalidate_briefing_cache = invalidate_briefing_cache

    def build_adjudication_requests(
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
            merged.request_reason = "needs_adjudication:" + ",".join(
                sorted(merged.ambiguity_tags),
            )
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

    async def create_clarification_intents(
        self,
        requests: list[AdjudicationRequest],
    ) -> None:
        """Create high-activation ClarificationIntent entities for pending adjudications."""
        if not self._cfg.active_adjudication_enabled or not requests:
            return

        from engram.models.entity import Entity
        from engram.utils.dates import utc_now

        for request in requests:
            intent_id = f"intent_{request.request_id[:12]}"
            # Link to the original text and episode
            summary = (
                f"Clarification needed for '{request.selected_text}'. "
                f"Reason: {request.request_reason}. "
                f"Source Episode: {request.episode_id}"
            )
            intent = Entity(
                id=intent_id,
                name=f"Clarification: {request.selected_text}",
                entity_type="ClarificationIntent",
                summary=summary,
                group_id=request.group_id,
                activation_current=0.95,  # High activation to ensure recall
                created_at=utc_now(),
                updated_at=utc_now(),
                attributes={
                    "adjudication_request_id": request.request_id,
                    "ambiguity_tags": request.ambiguity_tags,
                    "target_text": request.selected_text,
                },
            )
            try:
                await self._graph.create_entity(intent)
                logger.info(
                    "Active Adjudication: Created intent %s for request %s",
                    intent_id,
                    request.request_id,
                )
            except Exception:
                logger.warning(
                    "Failed to create clarification intent for request %s",
                    request.request_id,
                    exc_info=True,
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
    def _adjudication_instructions(tags: list[str]) -> str:
        """Return MCP-facing instructions for an adjudication request."""
        tag_list = ", ".join(tags)
        return (
            "Resolve only if highly confident. Use explicit entities and relationships "
            f"for the ambiguous case ({tag_list}); otherwise leave it unresolved."
        )

    @staticmethod
    def committed_id_map(
        committed_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        *,
        entity_map: dict[str, str],
        claims: Any,
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
                "existing_relationship_id",
            )
            if not relationship_id:
                continue
            if evidence_id:
                committed_ids[evidence_id] = relationship_id
            for temporal_evidence_id in (claim.raw_payload or {}).get(
                "temporal_evidence_ids",
                [],
            ):
                committed_ids[temporal_evidence_id] = relationship_id
        return committed_ids

    @staticmethod
    def apply_committed_ids(
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
    def serialize_evidence_records(
        evidence_pairs: list[tuple[EvidenceCandidate, CommitDecision]],
        *,
        status: str,
        commit_reason: str | None = None,
    ) -> list[dict]:
        """Convert evidence candidates to storage rows with explicit status."""
        return [
            EvidenceAdjudicationService._serialize_candidate(candidate, status, commit_reason)
            for candidate, _decision in evidence_pairs
        ]

    @staticmethod
    def serialize_candidate_records(
        candidates: list[EvidenceCandidate],
        *,
        status: str,
        commit_reason: str | None = None,
    ) -> list[dict]:
        """Convert raw evidence candidates to storage rows."""
        return [
            EvidenceAdjudicationService._serialize_candidate(candidate, status, commit_reason)
            for candidate in candidates
        ]

    @staticmethod
    def rehydrate_evidence_pairs(
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
    def evidence_projection_bundle(
        episode: Episode,
        entities: Any,
        claims: Any,
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
                group_id=episode.group_id,
            ),
            entities=entities,
            claims=claims,
        )

    async def index_materialized_bundle(
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
        entities, claims = self._evidence_bridge.bridge(evidence_pairs)
        bundle = self.evidence_projection_bundle(
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
            conversation_date=episode.conversation_date,
        )
        committed_ids = self.committed_id_map(
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
        await self.index_materialized_bundle(
            bundle=bundle,
            entity_map=entity_map,
            group_id=group_id,
            episode_id=episode.id,
        )
        self._invalidate_briefing_cache(group_id)
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
            evidence_pairs=self.rehydrate_evidence_pairs(evidence_rows),
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
        replacement_rows = self.serialize_candidate_records(
            replacement_candidates,
            status="committed",
            commit_reason=f"materialized_from_adjudication:{request_id}",
        )
        replacement_rows, unresolved_replacements = self.apply_committed_ids(
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
    def _serialize_candidate(
        candidate: EvidenceCandidate,
        status: str,
        commit_reason: str | None,
    ) -> dict:
        return {
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
