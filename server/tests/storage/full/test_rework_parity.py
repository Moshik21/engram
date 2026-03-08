"""Full-mode parity coverage for the extraction/recall/consolidation rework."""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig, EmbeddingConfig
from engram.events.bus import EventBus
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from engram.models.episode import EpisodeProjectionState, EpisodeStatus
from engram.storage.vector.redis_search import RedisSearchIndex
from engram.worker import EpisodeWorker
from tests.conftest import MockExtractor

pytestmark = pytest.mark.requires_docker


class _TextFallbackProvider:
    """Deterministic local provider that forces text fallback at query time."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            token_count = float(max(len(text.split()), 1))
            char_count = float(max(len(text), 1))
            vowel_count = float(sum(ch.lower() in "aeiou" for ch in text) or 1)
            vectors.append([token_count, char_count, vowel_count, 1.0])
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        del text
        return []


def _cue_lookup(result: dict) -> dict:
    cue = result["cue"]
    return {
        "lookup_id": f"cue:{cue['episode_id']}",
        "result_type": "cue_episode",
        "episode_id": cue["episode_id"],
        "cue_text": cue["cue_text"],
        "supporting_spans": cue.get("supporting_spans", []),
        "score": result["score"],
        "count_hit": False,
    }


async def _wait_for(
    predicate,
    *,
    timeout: float = 5.0,
    interval: float = 0.1,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        if await predicate():
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("Timed out waiting for full-mode integration state to settle")
        await asyncio.sleep(interval)


async def _episode_projected(store, episode_id: str) -> bool:
    episode = await store.get_episode_by_id(episode_id, "default")
    if episode is None:
        return False
    return (
        episode.status == EpisodeStatus.COMPLETED
        and episode.projection_state == EpisodeProjectionState.PROJECTED
    )


async def _make_manager(
    *,
    graph_store,
    activation_store,
    redis_search_runtime,
    cfg: ActivationConfig,
    extraction_result: ExtractionResult,
    event_bus: EventBus,
) -> tuple[GraphManager, RedisSearchIndex]:
    search_index = RedisSearchIndex(
        redis_search_runtime["client"],
        provider=_TextFallbackProvider(),
        config=EmbeddingConfig(provider="test", model="test", dimensions=4),
        index_name=redis_search_runtime["index_name"],
        key_prefix=redis_search_runtime["key_prefix"],
    )
    await search_index.initialize()
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(extraction_result),
        cfg=cfg,
        event_bus=event_bus,
    )
    return manager, search_index


@pytest.mark.asyncio
async def test_full_mode_cue_usage_projects_episode_and_preserves_recall_parity(
    falkordb_graph_store,
    redis_activation_store,
    redis_search_runtime,
):
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        cue_recall_enabled=True,
        cue_policy_learning_enabled=True,
        cue_policy_schedule_threshold=0.6,
        cue_policy_select_weight=0.05,
        cue_policy_use_weight=0.35,
        cue_recall_hit_threshold=20,
        episode_retrieval_enabled=True,
        recall_usage_feedback_enabled=True,
        recall_telemetry_enabled=True,
        triage_enabled=False,
        worker_enabled=True,
        working_memory_enabled=False,
    )
    extraction_result = ExtractionResult(
        entities=[
            {
                "name": "React",
                "entity_type": "Technology",
                "summary": "UI library used for the dashboard migration",
            }
        ],
        relationships=[],
    )
    bus = EventBus()
    manager, search_index = await _make_manager(
        graph_store=falkordb_graph_store,
        activation_store=redis_activation_store,
        redis_search_runtime=redis_search_runtime,
        cfg=cfg,
        extraction_result=extraction_result,
        event_bus=bus,
    )
    worker = EpisodeWorker(manager, cfg)

    try:
        episode_id = await manager.store_episode(
            "React dashboard migration remains in scope for this sprint.",
            group_id="default",
            source="auto:prompt",
        )
        await falkordb_graph_store.update_episode(
            episode_id,
            {
                "projection_state": EpisodeProjectionState.CUE_ONLY,
                "last_projection_reason": "integration_reset",
            },
            group_id="default",
        )
        await falkordb_graph_store.update_episode_cue(
            episode_id,
            {
                "projection_state": EpisodeProjectionState.CUE_ONLY,
                "route_reason": "integration_reset",
                "policy_score": 0.45,
                "projection_priority": 0.45,
            },
            group_id="default",
        )

        recall_results = await manager.recall(
            "React dashboard migration",
            group_id="default",
            record_access=False,
            interaction_type="selected",
            interaction_source="chat_tool_select",
        )
        cue_result = next(
            result for result in recall_results if result.get("result_type") == "cue_episode"
        )

        worker.start("default", bus)
        try:
            await manager.apply_memory_interaction(
                [f"cue:{episode_id}"],
                group_id="default",
                interaction_type="used",
                source="chat_response",
                query="What remains in scope?",
                result_lookup={f"cue:{episode_id}": _cue_lookup(cue_result)},
            )
            await _wait_for(
                lambda: _episode_projected(falkordb_graph_store, episode_id),
            )
        finally:
            await worker.stop()

        stored_episode = await falkordb_graph_store.get_episode_by_id(episode_id, "default")
        cue_after = await falkordb_graph_store.get_episode_cue(episode_id, "default")
        assert stored_episode is not None
        assert cue_after is not None
        assert stored_episode.status == EpisodeStatus.COMPLETED
        assert stored_episode.projection_state == EpisodeProjectionState.PROJECTED
        assert cue_after.projection_state == EpisodeProjectionState.PROJECTED
        assert cue_after.used_count == 1
        assert await falkordb_graph_store.get_episode_entities(episode_id)

        entity_results = await manager.recall(
            "React",
            group_id="default",
            limit=5,
            record_access=False,
        )
        assert any(
            result.get("result_type") == "entity"
            and result["entity"]["name"] == "React"
            for result in entity_results
        )

        episode_results = await manager.recall(
            "scope sprint",
            group_id="default",
            limit=5,
            record_access=False,
        )
        assert any(
            result.get("result_type") == "episode"
            and result["episode"]["id"] == episode_id
            for result in episode_results
        )
    finally:
        await search_index.close()
