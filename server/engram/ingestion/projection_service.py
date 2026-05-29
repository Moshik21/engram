"""Projection execution service for stored episodes."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.discourse import classify_discourse
from engram.extraction.models import ApplyOutcome
from engram.extraction.planner import summarize_plan
from engram.ingestion.projection_execution import (
    EvidenceProjectionExecutor,
    LegacyProjectionExecutor,
    ProjectionError,
    ProjectionLifecycleResult,
)
from engram.models.episode import EpisodeProjectionState, EpisodeStatus
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


class EpisodeProjectionService:
    """Run the Project stage for an episode while GraphManager remains the facade."""

    def __init__(
        self,
        *,
        graph_store: Any,
        cfg: ActivationConfig,
        projection_planner: Any,
        evidence_projection_executor: EvidenceProjectionExecutor,
        legacy_projection_executor: LegacyProjectionExecutor,
        content_hashes: set[str],
        content_hashes_inflight: set[str],
        update_episode_status: Any,
        sync_projection_state: Any,
        get_episode_cue: Any,
        publish_event: Any,
        should_use_evidence_pipeline: Any,
        run_surprise_detection: Any,
        run_prospective_memory: Any,
        publish_projection_graph_changes: Any,
        index_projected_bundle: Any,
        store_emotional_encoding_context: Any,
        invalidate_briefing_cache: Any,
        record_storage_counts: Any = None,
    ) -> None:
        self._graph = graph_store
        self._cfg = cfg
        self._projection_planner = projection_planner
        self._evidence_projection_executor = evidence_projection_executor
        self._legacy_projection_executor = legacy_projection_executor
        self._content_hashes = content_hashes
        self._content_hashes_inflight = content_hashes_inflight
        self._update_episode_status = update_episode_status
        self._sync_projection_state = sync_projection_state
        self._get_episode_cue = get_episode_cue
        self._publish = publish_event
        self._should_use_evidence_pipeline = should_use_evidence_pipeline
        self._run_surprise_detection = run_surprise_detection
        self._run_prospective_memory = run_prospective_memory
        self._publish_projection_graph_changes = publish_projection_graph_changes
        self._index_projected_bundle = index_projected_bundle
        self._store_emotional_encoding_context = store_emotional_encoding_context
        self._invalidate_briefing_cache = invalidate_briefing_cache
        self._record_storage_counts = record_storage_counts

    async def project_episode(
        self,
        episode_id: str,
        group_id: str = "default",
        proposed_entities: list[dict] | None = None,
        proposed_relationships: list[dict] | None = None,
        model_tier: str = "default",
    ) -> ProjectionLifecycleResult:
        """Run extraction, resolution, graph writes, indexing, and lifecycle updates."""
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if not episode:
            raise ValueError(f"Episode not found: {episode_id}")

        start_ms = time.monotonic()
        content = episode.content

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
            await self._sync_projection_state(
                episode_id,
                EpisodeProjectionState.CUE_ONLY,
                group_id=group_id,
                reason="duplicate_content",
                cue_reason="duplicate_content",
            )
            return ProjectionLifecycleResult(
                episode_id=episode_id,
                group_id=group_id,
                outcome="skipped",
                episode_status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                reason="duplicate_content",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )

        if classify_discourse(content) == "system":
            logger.warning("project_episode: skipping system-discourse episode %s", episode_id)
            await self._update_episode_status(
                episode_id,
                EpisodeStatus.COMPLETED,
                group_id=group_id,
                skipped_meta=True,
            )
            self._content_hashes.add(content_hash)
            await self._sync_projection_state(
                episode_id,
                EpisodeProjectionState.CUE_ONLY,
                group_id=group_id,
                reason="system_discourse",
                cue_reason="system_discourse",
            )
            return ProjectionLifecycleResult(
                episode_id=episode_id,
                group_id=group_id,
                outcome="skipped",
                episode_status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.CUE_ONLY,
                reason="system_discourse",
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )

        self._content_hashes_inflight.add(content_hash)

        try:
            await self._sync_projection_state(
                episode_id,
                EpisodeProjectionState.PROJECTING,
                group_id=group_id,
                reason="projection_started",
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

            if self._should_use_evidence_pipeline(
                proposed_entities=proposed_entities,
                proposed_relationships=proposed_relationships,
            ):
                evidence_outcome = await self._evidence_projection_executor.execute(
                    episode=episode,
                    plan=plan,
                    group_id=group_id,
                    cue=cue,
                    proposed_entities=proposed_entities,
                    proposed_relationships=proposed_relationships,
                    model_tier=model_tier,
                )
                bundle = evidence_outcome.bundle
                apply_outcome = evidence_outcome.apply_outcome
                entity_map = evidence_outcome.entity_map
                now = evidence_outcome.now
                used_evidence_materializer = evidence_outcome.used_evidence_materializer
            else:
                legacy_outcome = await self._legacy_projection_executor.execute(
                    episode=episode,
                    plan=plan,
                    group_id=group_id,
                )
                bundle = legacy_outcome.bundle
                apply_outcome = legacy_outcome.apply_outcome
                entity_map = legacy_outcome.entity_map
                now = legacy_outcome.now
                used_evidence_materializer = legacy_outcome.used_evidence_materializer

            self._record_projection_storage_counts(group_id, apply_outcome)

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
            # Report PERSISTED graph counts, not pre-apply candidate counts.
            # apply.py drops relationships whose endpoints were never committed
            # as entities (action="missing_entities"); reporting len(bundle.claims)
            # made an empty graph look successfully projected (a "phantom graph"
            # that silently contributes nothing to retrieval).
            # Auto-created endpoints (D1) are real new entities written to the graph
            # but never pass through apply_entities, so count them here so the
            # drop->create swap is visible rather than hidden.
            auto_created_endpoints = sum(
                len(getattr(r, "metadata", {}).get("auto_created_endpoints", []))
                for r in apply_outcome.relationship_results
            )
            persisted_entities = len(apply_outcome.new_entity_names) + auto_created_endpoints
            persisted_relationships = sum(
                1 for r in apply_outcome.relationship_results if getattr(r, "created", False)
            )
            dropped_missing = sum(
                1
                for r in apply_outcome.relationship_results
                if getattr(r, "action", None) == "missing_entities"
            )
            if dropped_missing:
                logger.warning(
                    "Episode %s: dropped %d relationship(s) with uncommitted "
                    "endpoints (missing_entities); persisted %d of %d candidate claims. "
                    "The extractor proposed edges whose endpoint entities were not committed.",
                    episode_id,
                    dropped_missing,
                    persisted_relationships,
                    len(bundle.claims),
                )
            await self._sync_projection_state(
                episode_id,
                EpisodeProjectionState.PROJECTED,
                group_id=group_id,
                reason="projected",
                last_projected_at=projected_at,
                cue_updates={
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
                ProjectionLifecycleResult(
                    episode_id=episode_id,
                    group_id=group_id,
                    outcome="projected",
                    episode_status=EpisodeStatus.COMPLETED,
                    projection_state=EpisodeProjectionState.PROJECTED,
                    reason="projected",
                    entity_count=persisted_entities,
                    relationship_count=persisted_relationships,
                    duration_ms=elapsed_ms,
                    used_evidence_materializer=used_evidence_materializer,
                    plan_strategy=plan.strategy,
                ).to_event_payload(),
            )
            logger.info(
                "Ingested episode %s: %d entities, %d relationships persisted "
                "(%d/%d candidate claims, %d dropped missing-endpoints)",
                episode_id,
                persisted_entities,
                persisted_relationships,
                persisted_relationships,
                len(bundle.claims),
                dropped_missing,
            )
            if not used_evidence_materializer:
                self._invalidate_briefing_cache(group_id)
            return ProjectionLifecycleResult(
                episode_id=episode_id,
                group_id=group_id,
                outcome="projected",
                episode_status=EpisodeStatus.COMPLETED,
                projection_state=EpisodeProjectionState.PROJECTED,
                reason="projected",
                entity_count=persisted_entities,
                relationship_count=persisted_relationships,
                duration_ms=elapsed_ms,
                used_evidence_materializer=used_evidence_materializer,
                plan_strategy=plan.strategy,
            )

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
            await self._sync_projection_state(
                episode_id,
                fail_projection_state,
                group_id=group_id,
                reason=str(e),
                cue_updates={
                    "projection_state": fail_projection_state,
                    "projection_attempts": retry_count,
                },
            )
            failure_result = ProjectionLifecycleResult(
                episode_id=episode_id,
                group_id=group_id,
                outcome="failed",
                episode_status=fail_status,
                projection_state=fail_projection_state,
                reason=str(e),
                retry_count=retry_count,
                error=str(e),
                duration_ms=int((time.monotonic() - start_ms) * 1000),
            )
            self._publish(
                group_id,
                "episode.failed",
                failure_result.to_event_payload(),
            )
            raise

    def _record_projection_storage_counts(
        self,
        group_id: str,
        apply_outcome: ApplyOutcome,
    ) -> None:
        if self._record_storage_counts is None:
            return
        relationship_count = sum(
            1
            for result in apply_outcome.relationship_results
            if getattr(result, "created", False)
        )
        auto_created_endpoints = sum(
            len(getattr(result, "metadata", {}).get("auto_created_endpoints", []))
            for result in apply_outcome.relationship_results
        )
        try:
            self._record_storage_counts(
                group_id,
                entities=len(apply_outcome.new_entity_names) + auto_created_endpoints,
                relationships=relationship_count,
            )
        except Exception:
            logger.debug("failed to record projection storage count delta", exc_info=True)
