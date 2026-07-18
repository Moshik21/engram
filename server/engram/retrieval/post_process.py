"""Post-materialization Recall processing."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from engram.config import ActivationConfig
from engram.retrieval.confidence import RecallConfidenceApplier
from engram.retrieval.context import (
    ConversationContext,
    RecallConversationFingerprintRecorder,
)
from engram.retrieval.episode_traversal import RecallEpisodeTraversal
from engram.retrieval.near_miss import RecallNearMissMaterializer
from engram.retrieval.priming import RecallPrimingUpdater
from engram.retrieval.result_selection import (
    filter_current_state_results,
    prefer_durable_facts,
)
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import RecallWorkingMemoryUpdater, WorkingMemoryBuffer

T = TypeVar("T")


@dataclass
class RecallPostProcessResult:
    """Final Recall results plus materialized near misses."""

    results: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]


class RecallPostProcessor:
    """Run Recall side effects after primary result materialization."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        episode_traversal: RecallEpisodeTraversal,
        working_memory_updater: RecallWorkingMemoryUpdater,
        priming_updater: RecallPrimingUpdater,
        near_miss_materializer: RecallNearMissMaterializer,
        confidence_applier: RecallConfidenceApplier,
        fingerprint_recorder: RecallConversationFingerprintRecorder,
    ) -> None:
        self._cfg = cfg
        self._episode_traversal = episode_traversal
        self._working_memory_updater = working_memory_updater
        self._priming_updater = priming_updater
        self._near_miss_materializer = near_miss_materializer
        self._confidence_applier = confidence_applier
        self._fingerprint_recorder = fingerprint_recorder

    async def process(
        self,
        results: list[dict[str, Any]],
        *,
        group_id: str,
        query: str,
        seen_episode_ids: set[str],
        near_miss_results: Sequence[ScoredResult],
        now: float,
        working_memory: WorkingMemoryBuffer | None,
        priming_buffer: dict[str, tuple[float, float]],
        conv_context: ConversationContext | None,
        interaction_type: str | None,
        interaction_source: str,
        stage_timings_ms: dict[str, float] | None = None,
        entity_candidates: list[tuple[str, float]] | None = None,
    ) -> RecallPostProcessResult:
        """Apply episode expansion, recall side effects, and final scoring."""
        # Only pass the candidate pool when present so existing traversal
        # stubs (and the default 'results' source) see today's exact call.
        traversal_kwargs: dict[str, Any] = (
            {"candidate_entity_scores": entity_candidates} if entity_candidates else {}
        )
        await self._bounded_stage(
            self._episode_traversal.append_entity_linked_episodes(
                results,
                group_id=group_id,
                seen_episode_ids=seen_episode_ids,
                **traversal_kwargs,
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_entity_episode_traversal",
            timeout_key="recall_entity_episode_traversal_timeout",
            timeout_field="entity_episode_traversal_timeout_ms",
            fallback=None,
        )
        await self._bounded_stage(
            self._episode_traversal.append_temporal_episodes(
                results,
                group_id=group_id,
                seen_episode_ids=seen_episode_ids,
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_temporal_contiguity",
            timeout_key="recall_temporal_contiguity_timeout",
            timeout_field="temporal_contiguity_timeout_ms",
            fallback=None,
        )

        filtered_results = prefer_durable_facts(filter_current_state_results(query, results))

        self._working_memory_updater.add_query(
            working_memory,
            query=query,
            now=now,
        )
        await self._bounded_stage(
            self._priming_updater.update(
                filtered_results,
                group_id=group_id,
                priming_buffer=priming_buffer,
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_priming_update",
            timeout_key="recall_priming_update_timeout",
            timeout_field="recall_priming_update_timeout_ms",
            fallback=None,
        )
        if (
            near_miss_results
            and self._cfg.retrieval_skip_secondary_graph_after_probe_timeout
            and _graph_probe_timed_out(stage_timings_ms)
        ):
            near_misses = []
            _set_stage_metric(
                stage_timings_ms,
                "recall_near_miss_materialize_skipped_probe_timeout",
                0.0,
            )
        else:
            near_misses = await self._bounded_stage(
                self._near_miss_materializer.materialize(
                    near_miss_results,
                    group_id=group_id,
                    query=query,
                    interaction_type=interaction_type,
                ),
                stage_timings_ms=stage_timings_ms,
                stage_key="recall_near_miss_materialize",
                timeout_key="recall_near_miss_materialize_timeout",
                timeout_field="recall_near_miss_materialize_timeout_ms",
                fallback=[],
            )
        await self._bounded_stage(
            self._confidence_applier.apply(query=query, results=filtered_results),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_confidence",
            timeout_key="recall_confidence_timeout",
            timeout_field="recall_confidence_timeout_ms",
            fallback=None,
        )
        await self._bounded_stage(
            self._fingerprint_recorder.record_recall_query(
                conv_context,
                query,
                interaction_source=interaction_source,
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_fingerprint_record",
            timeout_key="recall_fingerprint_record_timeout",
            timeout_field="recall_fingerprint_record_timeout_ms",
            fallback=None,
        )

        return RecallPostProcessResult(
            results=filtered_results,
            near_misses=near_misses,
        )

    async def _bounded_stage(
        self,
        awaitable: Awaitable[T],
        *,
        stage_timings_ms: dict[str, float] | None,
        stage_key: str,
        timeout_key: str,
        timeout_field: str,
        fallback: T,
    ) -> T:
        started = time.perf_counter()
        try:
            timeout_seconds = self._timeout_seconds(timeout_field)
            result = (
                await asyncio.wait_for(awaitable, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await awaitable
            )
            _set_stage_timing(stage_timings_ms, stage_key, started)
            return result
        except asyncio.TimeoutError:
            _set_stage_timing(stage_timings_ms, timeout_key, started)
            return fallback
        except asyncio.CancelledError:
            _set_stage_timing(stage_timings_ms, f"{stage_key}_cancelled", started)
            raise

    def _timeout_seconds(self, field_name: str) -> float | None:
        timeout_ms = int(getattr(self._cfg, field_name, 0) or 0)
        if timeout_ms <= 0:
            return None
        return timeout_ms / 1000.0


def _set_stage_timing(
    stage_timings_ms: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if stage_timings_ms is None:
        return
    stage_timings_ms[key] = round((time.perf_counter() - started) * 1000, 4)


def _set_stage_metric(
    stage_timings_ms: dict[str, float] | None,
    key: str,
    value: int | float,
) -> None:
    if stage_timings_ms is None:
        return
    stage_timings_ms[key] = round(float(value), 4)


def _graph_probe_timed_out(stage_timings_ms: dict[str, float] | None) -> bool:
    return bool(
        stage_timings_ms
        and (
            "recall_stats_timeout" in stage_timings_ms or "graph_expand_timeout" in stage_timings_ms
        )
    )
