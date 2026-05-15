"""Tests for post-materialization Recall processing."""

from __future__ import annotations

from typing import Any

import pytest

from engram.retrieval.post_process import RecallPostProcessor
from engram.retrieval.scorer import ScoredResult
from engram.retrieval.working_memory import WorkingMemoryBuffer


def _score(
    node_id: str,
    *,
    result_type: str,
    score: float = 0.42,
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
async def test_process_runs_recall_tail_in_order_and_filters_current_state_results() -> None:
    events: list[tuple[str, object]] = []
    seen_episode_ids = {"ep_primary"}
    near_miss_results = [_score("ent_near", result_type="entity")]

    class EpisodeTraversal:
        async def append_entity_linked_episodes(
            self,
            results: list[dict[str, Any]],
            *,
            group_id: str,
            seen_episode_ids: set[str],
        ) -> None:
            events.append(("entity_linked", (group_id, set(seen_episode_ids))))
            seen_episode_ids.add("ep_linked")
            results.append(
                {
                    "result_type": "episode",
                    "episode": {"id": "ep_linked"},
                    "score": 0.7,
                }
            )

        async def append_temporal_episodes(
            self,
            results: list[dict[str, Any]],
            *,
            group_id: str,
            seen_episode_ids: set[str],
        ) -> None:
            events.append(("temporal", (group_id, set(seen_episode_ids))))
            seen_episode_ids.add("ep_temporal")
            results.append(
                {
                    "result_type": "cue_episode",
                    "episode": {"id": "ep_temporal"},
                    "score": 0.6,
                }
            )

    class WorkingMemoryUpdater:
        def add_query(
            self,
            buffer: WorkingMemoryBuffer | None,
            *,
            query: str,
            now: float,
        ) -> None:
            events.append(("working_memory", (query, now)))
            assert buffer is not None
            buffer.add_query(query, now)

    class PrimingUpdater:
        async def update(
            self,
            results: list[dict[str, Any]],
            *,
            group_id: str,
            priming_buffer: dict[str, tuple[float, float]],
        ) -> None:
            events.append(("priming", [result["result_type"] for result in results]))
            assert group_id == "native_brain"
            priming_buffer["ent_current"] = (0.2, 130.0)

    class NearMissMaterializer:
        async def materialize(
            self,
            scored_results: list[ScoredResult],
            *,
            group_id: str,
            query: str,
            interaction_type: str | None,
        ) -> list[dict[str, Any]]:
            events.append(("near_miss", (scored_results, group_id, query, interaction_type)))
            assert scored_results is near_miss_results
            return [{"result_type": "entity", "entity": {"name": "near"}}]

    class ConfidenceApplier:
        async def apply(self, *, query: str, results: list[dict[str, Any]]) -> None:
            events.append(("confidence", [result["result_type"] for result in results]))
            results[0].setdefault("score_breakdown", {})["relevance_confidence"] = 0.91

    class FingerprintRecorder:
        async def record_recall_query(
            self,
            conv_context: object | None,
            query: str,
            *,
            interaction_source: str,
        ) -> None:
            events.append(("fingerprint", (conv_context, query, interaction_source)))

    processor = RecallPostProcessor(
        episode_traversal=EpisodeTraversal(),
        working_memory_updater=WorkingMemoryUpdater(),
        priming_updater=PrimingUpdater(),
        near_miss_materializer=NearMissMaterializer(),
        confidence_applier=ConfidenceApplier(),
        fingerprint_recorder=FingerprintRecorder(),
    )
    working_memory = WorkingMemoryBuffer()
    priming_buffer: dict[str, tuple[float, float]] = {}
    conv_context = object()
    results = [
        {
            "result_type": "entity",
            "entity": {"id": "ent_current", "name": "Native Helix"},
            "score": 0.9,
        },
        {
            "result_type": "episode",
            "episode": {"id": "ep_primary"},
            "score": 0.8,
        },
    ]

    processed = await processor.process(
        results,
        group_id="native_brain",
        query="What is Engram using now?",
        seen_episode_ids=seen_episode_ids,
        near_miss_results=near_miss_results,
        now=100.0,
        working_memory=working_memory,
        priming_buffer=priming_buffer,
        conv_context=conv_context,
        interaction_type="surfaced",
        interaction_source="auto_recall",
    )

    assert [event[0] for event in events] == [
        "entity_linked",
        "temporal",
        "working_memory",
        "priming",
        "near_miss",
        "confidence",
        "fingerprint",
    ]
    assert processed.results == [
        {
            "result_type": "entity",
            "entity": {"id": "ent_current", "name": "Native Helix"},
            "score": 0.9,
            "score_breakdown": {"relevance_confidence": 0.91},
        }
    ]
    assert processed.near_misses == [
        {"result_type": "entity", "entity": {"name": "near"}}
    ]
    assert seen_episode_ids == {"ep_primary", "ep_linked", "ep_temporal"}
    assert working_memory.get_recent_queries() == ["What is Engram using now?"]
    assert priming_buffer == {"ent_current": (0.2, 130.0)}
    assert events[1][1] == ("native_brain", {"ep_primary", "ep_linked"})
    assert events[-1][1] == (conv_context, "What is Engram using now?", "auto_recall")
