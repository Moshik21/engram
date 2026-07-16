"""Projection execution path helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceBundle
from engram.extraction.models import ApplyOutcome, ProjectionBundle, ProjectionPlan
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue


class ProjectionError(RuntimeError):
    """Typed projection failure that distinguishes retryable failures."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ProjectionLifecycleResult:
    """Stable lifecycle result for one Project-stage attempt."""

    episode_id: str
    group_id: str
    outcome: str
    episode_status: EpisodeStatus
    projection_state: EpisodeProjectionState
    reason: str | None = None
    lifecycle_stage: str = "project"
    entity_count: int = 0
    relationship_count: int = 0
    duration_ms: int | None = None
    retry_count: int | None = None
    error: str | None = None
    used_evidence_materializer: bool = False
    plan_strategy: str | None = None

    def to_event_payload(self) -> dict[str, object]:
        """Return event-safe projection lifecycle fields."""
        payload: dict[str, object] = {
            "episodeId": self.episode_id,
            "status": self.episode_status.value,
            "outcome": self.outcome,
            "lifecycleStage": self.lifecycle_stage,
            "projectionState": self.projection_state.value,
            "entity_count": self.entity_count,
            "relationship_count": self.relationship_count,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.duration_ms is not None:
            payload["duration_ms"] = self.duration_ms
        if self.retry_count is not None:
            payload["retry_count"] = self.retry_count
        if self.error is not None:
            payload["error"] = self.error
        if self.plan_strategy is not None:
            payload["planStrategy"] = self.plan_strategy
        payload["usedEvidenceMaterializer"] = self.used_evidence_materializer
        return payload


@dataclass
class ProjectionExecutionOutcome:
    """Result of one projection execution path before final lifecycle handling."""

    bundle: ProjectionBundle
    apply_outcome: ApplyOutcome
    entity_map: dict[str, str]
    now: float
    used_evidence_materializer: bool = False


class LegacyProjectionExecutor:
    """Run the original extractor -> apply -> relationship write projection path."""

    def __init__(
        self,
        *,
        projector: Any,
        apply_engine: Any,
        update_episode_status: Any,
        apply_bootstrap_part_of_edges: Any,
    ) -> None:
        self._projector = projector
        self._apply_engine = apply_engine
        self._update_episode_status = update_episode_status
        self._apply_bootstrap_part_of_edges = apply_bootstrap_part_of_edges

    async def execute(
        self,
        *,
        episode: Episode,
        plan: ProjectionPlan,
        group_id: str,
    ) -> ProjectionExecutionOutcome:
        """Project through the legacy extractor and apply path."""
        bundle = await self._projector.project(plan)
        if bundle.is_error:
            raise ProjectionError(
                f"extractor_{bundle.extractor_status}: {bundle.extractor_error or 'unknown_error'}",
                retryable=bundle.retryable,
            )

        await self._update_episode_status(
            episode.id,
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
            episode.id,
            EpisodeStatus.WRITING,
            group_id=group_id,
        )
        apply_outcome.relationship_results = await self._apply_engine.apply_relationships(
            bundle.claims,
            entity_map=entity_map,
            meta_entity_names=apply_outcome.meta_entity_names,
            group_id=group_id,
            source_episode=episode.id,
            conversation_date=episode.conversation_date,
        )
        return ProjectionExecutionOutcome(
            bundle=bundle,
            apply_outcome=apply_outcome,
            entity_map=entity_map,
            now=now,
        )


class EvidenceProjectionExecutor:
    """Run the evidence extraction -> commit/defer -> materialize projection path."""

    def __init__(
        self,
        *,
        graph_store: Any,
        cfg: ActivationConfig,
        build_evidence_bundle: Any,
        build_adjudication_requests: Any,
        serialize_candidate_records: Any,
        serialize_evidence_records: Any,
        materialize_evidence: Any,
        apply_committed_ids: Any,
        update_episode_status: Any,
        ambiguity_analyzer: Any = None,
        commit_policy: Any = None,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._build_evidence_bundle = build_evidence_bundle
        self._build_adjudication_requests = build_adjudication_requests
        self._serialize_candidate_records = serialize_candidate_records
        self._serialize_evidence_records = serialize_evidence_records
        self._materialize_evidence = materialize_evidence
        self._apply_committed_ids = apply_committed_ids
        self._update_episode_status = update_episode_status
        self._ambiguity_analyzer = ambiguity_analyzer
        self._commit_policy = commit_policy

    async def execute(
        self,
        *,
        episode: Episode,
        plan: ProjectionPlan,
        group_id: str,
        cue: EpisodeCue | None = None,
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
    ) -> ProjectionExecutionOutcome:
        """Project through the evidence pipeline and materialize committed facts."""
        evidence_bundle = self._build_evidence_bundle(
            text=plan.selected_text,
            episode_id=episode.id,
            group_id=group_id,
            cue=cue,
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier=model_tier,
        )
        # Trust hardening: re-derive client-proposal evidence with deterministic
        # span verification + confidence de-weaponization. graph_manager builds the
        # bundle without episode content/date, so an unverified single-source
        # annotation would otherwise commit on first sight at the caller's tier.
        evidence_bundle = self._verify_proposal_bundle(
            evidence_bundle,
            episode=episode,
            group_id=group_id,
            proposed_entities=proposed_entities,
            proposed_relationships=proposed_relationships,
            model_tier=model_tier,
        )
        if self._cfg.streaming_evidence_enabled:
            from engram.extraction.streaming_projector import StreamingEvidenceProjector

            streamer = StreamingEvidenceProjector(group_id)
            streamer.broadcast_extraction_start(episode.id)
            # Broadcast the initial discovery
            # We wrap the candidates in a mock ExtractionResult
            from engram.extraction.models import ExtractionResult

            mock_result = ExtractionResult(
                entities=[c.entity for c in evidence_bundle.candidates if c.entity],
                relationships=[
                    c.relationship for c in evidence_bundle.candidates if c.relationship
                ],
            )
            streamer.broadcast_result(mock_result)

        evidence_bundle = await self._store_adjudication_work(
            evidence_bundle=evidence_bundle,
            plan=plan,
            episode_id=episode.id,
            group_id=group_id,
        )

        raw_entity_count = await self._graph.get_entity_count(group_id)
        entity_count = raw_entity_count if isinstance(raw_entity_count, int) else 0
        if self._commit_policy is None:
            raise ProjectionError("evidence_commit_policy_missing")
        decisions = self._commit_policy.evaluate(
            evidence_bundle,
            entity_count,
        )
        committed = [
            (ev, d) for ev, d in zip(evidence_bundle.candidates, decisions) if d.action == "commit"
        ]
        deferred = [
            (ev, d) for ev, d in zip(evidence_bundle.candidates, decisions) if d.action == "defer"
        ]
        rejected = [
            (ev, d) for ev, d in zip(evidence_bundle.candidates, decisions) if d.action == "reject"
        ]
        # Harness scoreboard: count client-proposal outcomes (no external extract).
        if (evidence_bundle.extractor_stats or {}).get("extraction_path") == "client_proposals" or (
            proposed_entities or proposed_relationships
        ):
            from engram.extraction.harness_metrics import record_client_proposal_outcomes

            span_defers = sum(1 for _ev, d in deferred if d.reason == "span_unverified")
            pred_rejects = sum(1 for _ev, d in rejected if d.reason == "predicate_not_allowed")
            id_conflicts = sum(1 for _ev, d in deferred if d.reason == "identity_core_conflict")
            record_client_proposal_outcomes(
                commits=len(committed),
                defers=len(deferred),
                rejects=len(rejected),
                span_unverified_defers=span_defers,
                predicate_rejects=pred_rejects,
                identity_conflicts=id_conflicts,
            )
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
            episode.id,
            EpisodeStatus.RESOLVING,
            group_id=group_id,
        )
        now = time.time()
        materialization = await self._materialize_evidence(
            episode=episode,
            evidence_pairs=committed,
            group_id=group_id,
            recall_content=plan.selected_text,
            on_before_relationships=lambda: self._update_episode_status(
                episode.id,
                EpisodeStatus.WRITING,
                group_id=group_id,
            ),
        )
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
        return ProjectionExecutionOutcome(
            bundle=materialization.bundle,
            apply_outcome=apply_outcome,
            entity_map=entity_map,
            now=now,
            used_evidence_materializer=True,
        )

    def _verify_proposal_bundle(
        self,
        evidence_bundle: EvidenceBundle,
        *,
        episode: Episode,
        group_id: str,
        proposed_entities: list[dict] | None,
        proposed_relationships: list[dict] | None,
        model_tier: str,
    ) -> EvidenceBundle:
        """Re-derive client-proposal candidates with span verification + trust caps.

        Only fires for proposal-sourced bundles (the bundle whose candidates were
        built from ``proposed_entities``/``proposed_relationships``). Non-proposal
        extractor bundles pass through unchanged.
        """
        if not (proposed_entities or proposed_relationships):
            return evidence_bundle
        stats = evidence_bundle.extractor_stats or {}
        if "client_proposals" not in stats and stats.get("extraction_path") != "client_proposals":
            return evidence_bundle

        from engram.extraction.client_proposals import proposals_to_evidence

        verified = proposals_to_evidence(
            proposed_entities,
            proposed_relationships,
            episode.id,
            group_id,
            model_tier,
            episode_content=episode.content,
            reference_date=episode.conversation_date or episode.created_at,
            verify_spans=True,
        )
        # Identity-core conflict tag when proposed summary would overwrite protected facts.
        verified = self._tag_identity_core_conflicts(verified, group_id=group_id)
        out_stats = dict(evidence_bundle.extractor_stats or {})
        out_stats["extraction_path"] = "client_proposals"
        out_stats["span_unverified"] = sum(
            1 for c in verified if "span_unverified" in (c.corroborating_signals or [])
        )
        return EvidenceBundle(
            episode_id=evidence_bundle.episode_id,
            group_id=evidence_bundle.group_id,
            candidates=verified,
            extractor_stats=out_stats,
            total_ms=evidence_bundle.total_ms,
        )

    def _tag_identity_core_conflicts(
        self,
        candidates: list,
        *,
        group_id: str,
    ) -> list:
        """Tag entity proposals that would silently overwrite identity_core summaries.

        Synchronous best-effort: when graph lookup helpers are async-only, skip
        tagging (commit path still protects via apply-time checks).
        """
        from engram.extraction.promotion import identity_core_summary_conflict

        # Apply-time check is the durable gate; here we only tag when a sync
        # cache is available on the graph store (tests can inject one).
        lookup = getattr(self._graph, "get_entity_by_name_sync", None)
        if not callable(lookup):
            return candidates
        for candidate in candidates:
            if candidate.fact_class != "entity":
                continue
            if candidate.source_type != "client_proposal":
                continue
            payload = candidate.payload or {}
            name = str(payload.get("name") or "")
            summary = payload.get("summary")
            if not name or not summary:
                continue
            existing = lookup(name, group_id)
            if existing is None:
                continue
            if not bool(getattr(existing, "identity_core", False)):
                continue
            if identity_core_summary_conflict(
                getattr(existing, "summary", None),
                str(summary),
                entity_type=str(payload.get("entity_type") or ""),
            ):
                signals = list(candidate.corroborating_signals or [])
                if "identity_core_conflict" not in signals:
                    signals.append("identity_core_conflict")
                candidate.corroborating_signals = signals
                candidate.confidence = min(float(candidate.confidence), 0.55)
        return candidates

    async def _store_adjudication_work(
        self,
        *,
        evidence_bundle: EvidenceBundle,
        plan: ProjectionPlan,
        episode_id: str,
        group_id: str,
    ) -> EvidenceBundle:
        if not self._cfg.edge_adjudication_enabled or not self._ambiguity_analyzer:
            return evidence_bundle

        # Hard path: span-verified harness proposals commit hot — do not route
        # them into the offline adjudication swamp.
        if (evidence_bundle.extractor_stats or {}).get("extraction_path") == "client_proposals":
            grounded = [
                c
                for c in evidence_bundle.candidates
                if c.source_type == "client_proposal"
                and "span_verified" in (c.corroborating_signals or [])
                and "span_unverified" not in (c.corroborating_signals or [])
                and "predicate_not_allowed" not in (c.corroborating_signals or [])
            ]
            if grounded and len(grounded) == len(evidence_bundle.candidates):
                return evidence_bundle

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
        if not ambiguous_groups:
            return evidence_bundle

        requests = self._build_adjudication_requests(
            episode_id,
            group_id,
            ambiguous_groups,
        )
        if not requests:
            return evidence_bundle

        await self._graph.store_adjudication_requests(
            [request.to_dict() for request in requests],
            group_id=group_id,
        )
        if self._cfg.active_adjudication_enabled:
            await self._evidence_adjudication_service.create_clarification_intents(requests)
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
        return evidence_bundle
