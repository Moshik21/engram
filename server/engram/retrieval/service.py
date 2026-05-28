"""Recall stage orchestration service."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve
from engram.retrieval.post_process import RecallPostProcessor
from engram.retrieval.primary_results import RecallPrimaryResultMaterializer
from engram.retrieval.request_policy import (
    recall_fetch_limit,
    should_record_ranking_feedback,
    split_primary_and_near_miss_results,
)
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import WorkingMemoryBuffer
from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

RecallRetrieveFn = Callable[..., Awaitable[list[ScoredResult]]]


@dataclass
class RecallServiceResult:
    """Final Recall payloads plus near misses retained by API/MCP surfaces."""

    results: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]
    stage_timings_ms: dict[str, float]


class RecallService:
    """Orchestrate the Recall stage behind the GraphManager compatibility facade."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        search_index: SearchIndex,
        cfg: ActivationConfig,
        primary_materializer: RecallPrimaryResultMaterializer,
        post_processor: RecallPostProcessor,
        reranker: object | None = None,
        community_store: object | None = None,
        predicate_cache: object | None = None,
        retrieve_fn: RecallRetrieveFn = retrieve,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._search = search_index
        self._cfg = cfg
        self._primary_materializer = primary_materializer
        self._post_processor = post_processor
        self._reranker = reranker
        self._community_store = community_store
        self._predicate_cache = predicate_cache
        self._retrieve = retrieve_fn
        self._time = time_fn
        self._last_stage_timings_ms: dict[str, float] = {}

    def last_stage_timings(self) -> dict[str, float]:
        """Return partial or complete timings from the latest recall attempt."""
        return dict(self._last_stage_timings_ms)

    async def recall(
        self,
        *,
        query: str,
        group_id: str,
        limit: int,
        record_access: bool,
        interaction_type: str | None,
        interaction_source: str,
        conv_context: object | None,
        working_memory: WorkingMemoryBuffer | None,
        priming_buffer: dict[str, tuple[float, float]],
        goal_cache: object | None,
        memory_need: object | None,
    ) -> RecallServiceResult:
        """Run retrieval, primary materialization, and post-processing."""
        fetch_limit = recall_fetch_limit(
            self._cfg,
            limit,
            conv_context=conv_context,
        )
        record_feedback = should_record_ranking_feedback(
            record_access=record_access,
            interaction_type=interaction_type,
        )

        stage_timings_ms: dict[str, float] = {}
        self._last_stage_timings_ms = stage_timings_ms
        retrieve_started = time.perf_counter()
        try:
            scored_results = await self._retrieve(
                query=query,
                group_id=group_id,
                graph_store=self._graph,
                activation_store=self._activation,
                search_index=self._search,
                cfg=self._cfg,
                limit=fetch_limit,
                working_memory=working_memory,
                reranker=self._reranker,
                community_store=self._community_store,
                predicate_cache=self._predicate_cache,
                conv_context=conv_context,
                priming_buffer=priming_buffer if self._cfg.retrieval_priming_enabled else None,
                goal_cache=goal_cache,
                record_feedback=record_feedback,
                memory_need=memory_need,
                stage_timings_ms=stage_timings_ms,
            )
        except asyncio.CancelledError:
            stage_timings_ms["recall_retrieve_cancelled"] = _elapsed_ms(
                retrieve_started
            )
            raise
        stage_timings_ms["recall_retrieve"] = _elapsed_ms(retrieve_started)

        primary_results, near_miss_results = split_primary_and_near_miss_results(
            scored_results,
            limit,
            near_miss_enabled=self._cfg.conv_near_miss_enabled,
        )

        now = self._time()
        materialize_started = time.perf_counter()
        graph_timeout_seconds = _materialize_graph_timeout_seconds(
            self._cfg,
            stage_timings_ms,
        )
        if graph_timeout_seconds is not None:
            stage_timings_ms["recall_materialize_graph_effective_timeout_ms"] = (
                round(graph_timeout_seconds * 1000.0, 4)
            )
        try:
            primary_materialization = await self._primary_materializer.materialize(
                primary_results,
                group_id=group_id,
                query=query,
                record_access=record_access,
                interaction_type=interaction_type,
                interaction_source=interaction_source,
                now=now,
                working_memory=working_memory,
                graph_timeout_seconds=graph_timeout_seconds,
                side_effect_timeout_seconds=_timeout_seconds(
                    self._cfg,
                    "recall_primary_materialize_side_effect_timeout_ms",
                ),
                stage_timings_ms=stage_timings_ms,
            )
        except asyncio.CancelledError:
            stage_timings_ms["recall_materialize_cancelled"] = _elapsed_ms(
                materialize_started
            )
            raise
        stage_timings_ms["recall_materialize"] = _elapsed_ms(materialize_started)

        post_started = time.perf_counter()
        try:
            post_processed = await self._post_processor.process(
                primary_materialization.results,
                group_id=group_id,
                seen_episode_ids=primary_materialization.seen_episode_ids,
                query=query,
                near_miss_results=near_miss_results,
                now=now,
                working_memory=working_memory,
                priming_buffer=priming_buffer,
                conv_context=conv_context,
                interaction_type=interaction_type,
                interaction_source=interaction_source,
                stage_timings_ms=stage_timings_ms,
            )
        except asyncio.CancelledError:
            stage_timings_ms["recall_post_process_cancelled"] = _elapsed_ms(
                post_started
            )
            raise
        stage_timings_ms["recall_post_process"] = _elapsed_ms(post_started)

        return RecallServiceResult(
            results=post_processed.results,
            near_misses=post_processed.near_misses,
            stage_timings_ms=stage_timings_ms,
        )


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _timeout_seconds(cfg: ActivationConfig, field_name: str) -> float | None:
    timeout_ms = int(getattr(cfg, field_name, 0) or 0)
    if timeout_ms <= 0:
        return None
    return timeout_ms / 1000.0


def _materialize_graph_timeout_seconds(
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float],
) -> float | None:
    timeout_ms = int(getattr(cfg, "recall_primary_materialize_graph_timeout_ms", 0) or 0)
    if timeout_ms <= 0:
        return None
    probe_timed_out = (
        "recall_stats_timeout" in stage_timings_ms
        or "graph_expand_timeout" in stage_timings_ms
    )
    adaptive_cap_ms = int(
        getattr(
            cfg,
            "recall_primary_materialize_graph_timeout_after_probe_timeout_ms",
            0,
        )
        or 0
    )
    if probe_timed_out and adaptive_cap_ms > 0:
        timeout_ms = min(timeout_ms, adaptive_cap_ms)
    return timeout_ms / 1000.0
