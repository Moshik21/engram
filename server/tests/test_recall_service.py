"""Tests for Recall stage orchestration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from engram.config import ActivationConfig
from engram.retrieval.post_process import RecallPostProcessResult
from engram.retrieval.primary_results import RecallPrimaryMaterialization
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.service import RecallService
from engram.retrieval.working_memory import WorkingMemoryBuffer


def _score(
    node_id: str,
    *,
    result_type: str = "entity",
    score: float = 0.8,
) -> ScoredResult:
    return ScoredResult(
        node_id=node_id,
        score=score,
        semantic_similarity=score,
        activation=0.0,
        spreading=0.0,
        edge_proximity=0.0,
        result_type=result_type,
    )


@pytest.mark.asyncio
async def test_recall_service_orchestrates_retrieve_materialize_and_post_process() -> None:
    events: list[tuple[str, object]] = []
    scored_results = [
        _score("ent_primary"),
        _score("ep_primary", result_type="cue_episode", score=0.7),
        _score("ent_near", score=0.4),
    ]
    graph = object()
    activation = object()
    search = object()
    reranker = object()
    community_store = object()
    predicate_cache = object()
    conv_context = object()
    goal_cache = object()
    working_memory = WorkingMemoryBuffer()
    priming_buffer: dict[str, tuple[float, float]] = {}
    cfg = ActivationConfig(
        conv_near_miss_enabled=True,
        conv_near_miss_window=2,
        retrieval_priming_enabled=True,
    )

    async def retrieve_fn(**kwargs: Any) -> list[ScoredResult]:
        events.append(("retrieve", kwargs))
        return scored_results

    class PrimaryMaterializer:
        async def materialize(
            self,
            results: list[ScoredResult],
            **kwargs: Any,
        ) -> RecallPrimaryMaterialization:
            events.append(("primary", ([result.node_id for result in results], kwargs)))
            return RecallPrimaryMaterialization(
                results=[{"result_type": "entity", "entity": {"id": "ent_primary"}}],
                seen_episode_ids={"ep_primary"},
            )

    class PostProcessor:
        async def process(
            self,
            results: list[dict[str, Any]],
            **kwargs: Any,
        ) -> RecallPostProcessResult:
            events.append(("post", (results, kwargs)))
            return RecallPostProcessResult(
                results=[*results, {"result_type": "cue_episode", "episode": {"id": "ep_2"}}],
                near_misses=[{"result_type": "entity", "entity": {"name": "near"}}],
            )

    service = RecallService(
        graph_store=graph,
        activation_store=activation,
        search_index=search,
        cfg=cfg,
        primary_materializer=PrimaryMaterializer(),
        post_processor=PostProcessor(),
        reranker=reranker,
        community_store=community_store,
        predicate_cache=predicate_cache,
        retrieve_fn=retrieve_fn,
        time_fn=lambda: 123.0,
    )

    result = await service.recall(
        query="native Helix",
        group_id="native_brain",
        limit=2,
        record_access=True,
        interaction_type="surfaced",
        interaction_source="auto_recall",
        conv_context=conv_context,
        working_memory=working_memory,
        priming_buffer=priming_buffer,
        goal_cache=goal_cache,
        memory_need={"urgency": "high"},
    )

    assert [event[0] for event in events] == ["retrieve", "primary", "post"]
    retrieve_kwargs = events[0][1]
    assert isinstance(retrieve_kwargs, dict)
    assert retrieve_kwargs["query"] == "native Helix"
    assert retrieve_kwargs["group_id"] == "native_brain"
    assert retrieve_kwargs["graph_store"] is graph
    assert retrieve_kwargs["activation_store"] is activation
    assert retrieve_kwargs["search_index"] is search
    assert retrieve_kwargs["cfg"] is cfg
    assert retrieve_kwargs["limit"] == 4
    assert retrieve_kwargs["working_memory"] is working_memory
    assert retrieve_kwargs["reranker"] is reranker
    assert retrieve_kwargs["community_store"] is community_store
    assert retrieve_kwargs["predicate_cache"] is predicate_cache
    assert retrieve_kwargs["conv_context"] is conv_context
    assert retrieve_kwargs["priming_buffer"] is priming_buffer
    assert retrieve_kwargs["goal_cache"] is goal_cache
    assert retrieve_kwargs["record_feedback"] is False
    assert retrieve_kwargs["memory_need"] == {"urgency": "high"}
    assert retrieve_kwargs["budget_profile"] == "auto_deep"
    assert isinstance(retrieve_kwargs["stage_timings_ms"], dict)

    primary_ids, primary_kwargs = events[1][1]
    assert primary_ids == ["ent_primary", "ep_primary"]
    assert primary_kwargs["group_id"] == "native_brain"
    assert primary_kwargs["query"] == "native Helix"
    assert primary_kwargs["record_access"] is True
    assert primary_kwargs["interaction_type"] == "surfaced"
    assert primary_kwargs["interaction_source"] == "auto_recall"
    assert primary_kwargs["now"] == 123.0
    assert primary_kwargs["working_memory"] is working_memory
    assert primary_kwargs["graph_timeout_seconds"] == pytest.approx(0.3)
    assert primary_kwargs["side_effect_timeout_seconds"] == pytest.approx(0.025)
    assert primary_kwargs["stage_timings_ms"] is result.stage_timings_ms

    post_results, post_kwargs = events[2][1]
    assert post_results == [{"result_type": "entity", "entity": {"id": "ent_primary"}}]
    assert post_kwargs["seen_episode_ids"] == {"ep_primary"}
    assert [result.node_id for result in post_kwargs["near_miss_results"]] == ["ent_near"]
    assert post_kwargs["now"] == 123.0
    assert post_kwargs["working_memory"] is working_memory
    assert post_kwargs["priming_buffer"] is priming_buffer
    assert post_kwargs["conv_context"] is conv_context
    assert post_kwargs["interaction_type"] == "surfaced"
    assert post_kwargs["interaction_source"] == "auto_recall"

    assert result.results == [
        {"result_type": "entity", "entity": {"id": "ent_primary"}},
        {"result_type": "cue_episode", "episode": {"id": "ep_2"}},
    ]
    assert result.near_misses == [{"result_type": "entity", "entity": {"name": "near"}}]
    assert result.stage_timings_ms["recall_retrieve"] >= 0
    assert result.stage_timings_ms["recall_materialize"] >= 0
    assert result.stage_timings_ms["recall_post_process"] >= 0
    assert result.stage_timings_ms["recall_materialize_graph_effective_timeout_ms"] == 300


@pytest.mark.asyncio
async def test_recall_service_caps_materialization_after_probe_timeout() -> None:
    events: list[tuple[str, object]] = []
    scored_results = [_score("ent_primary")]
    cfg = ActivationConfig(
        recall_primary_materialize_graph_timeout_ms=50,
        recall_primary_materialize_graph_timeout_after_probe_timeout_ms=15,
    )

    async def retrieve_fn(**kwargs: Any) -> list[ScoredResult]:
        kwargs["stage_timings_ms"]["recall_stats_timeout"] = 75.0
        return scored_results

    class PrimaryMaterializer:
        async def materialize(
            self,
            results: list[ScoredResult],
            **kwargs: Any,
        ) -> RecallPrimaryMaterialization:
            events.append(("primary", kwargs))
            return RecallPrimaryMaterialization(
                results=[{"result_type": "entity", "entity": {"id": "ent_primary"}}],
                seen_episode_ids=set(),
            )

    class PostProcessor:
        async def process(
            self,
            results: list[dict[str, Any]],
            **kwargs: Any,
        ) -> RecallPostProcessResult:
            return RecallPostProcessResult(results=results, near_misses=[])

    service = RecallService(
        graph_store=object(),
        activation_store=object(),
        search_index=object(),
        cfg=cfg,
        primary_materializer=PrimaryMaterializer(),
        post_processor=PostProcessor(),
        retrieve_fn=retrieve_fn,
    )

    result = await service.recall(
        query="native Helix",
        group_id="native_brain",
        limit=1,
        record_access=True,
        interaction_type="used",
        interaction_source="mcp_recall",
        conv_context=None,
        working_memory=None,
        priming_buffer={},
        goal_cache=None,
        memory_need=None,
    )

    primary_kwargs = events[0][1]
    assert isinstance(primary_kwargs, dict)
    assert primary_kwargs["graph_timeout_seconds"] == pytest.approx(0.015)
    assert result.stage_timings_ms["recall_materialize_graph_effective_timeout_ms"] == 15


@pytest.mark.asyncio
async def test_recall_service_records_partial_timing_when_cancelled() -> None:
    async def retrieve_fn(**kwargs: Any) -> list[ScoredResult]:
        kwargs["stage_timings_ms"]["recall_embed"] = 12.0
        raise asyncio.CancelledError

    class PrimaryMaterializer:
        async def materialize(
            self,
            results: list[ScoredResult],
            **kwargs: Any,
        ) -> RecallPrimaryMaterialization:
            raise AssertionError("materialize should not run after cancellation")

    class PostProcessor:
        async def process(
            self,
            results: list[dict[str, Any]],
            **kwargs: Any,
        ) -> RecallPostProcessResult:
            raise AssertionError("post-process should not run after cancellation")

    service = RecallService(
        graph_store=object(),
        activation_store=object(),
        search_index=object(),
        cfg=ActivationConfig(),
        primary_materializer=PrimaryMaterializer(),
        post_processor=PostProcessor(),
        retrieve_fn=retrieve_fn,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.recall(
            query="native Helix",
            group_id="native_brain",
            limit=2,
            record_access=True,
            interaction_type="used",
            interaction_source="mcp_recall",
            conv_context=None,
            working_memory=None,
            priming_buffer={},
            goal_cache=None,
            memory_need=None,
        )

    timings = service.last_stage_timings()
    assert timings["recall_embed"] == 12.0
    assert timings["recall_retrieve_cancelled"] >= 0
