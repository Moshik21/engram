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
                f"extractor_{bundle.extractor_status}: "
                f"{bundle.extractor_error or 'unknown_error'}",
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
