"""Tests for recall near-miss helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.near_miss import RecallNearMissBuilder, RecallNearMissMaterializer
from engram.retrieval.scorer import ScoredResult
from engram.utils.dates import utc_now


def _score(
    node_id: str,
    *,
    result_type: str,
    score: float = 0.63333,
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


def _episode(
    episode_id: str,
    *,
    projection_state: EpisodeProjectionState = EpisodeProjectionState.CUE_ONLY,
) -> Episode:
    return Episode(
        id=episode_id,
        content=f"{episode_id} content",
        source="test",
        status=EpisodeStatus.COMPLETED,
        projection_state=projection_state,
        group_id="native_brain",
        created_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_entity_near_miss_formats_entity_result() -> None:
    graph = AsyncMock()
    graph.get_entity = AsyncMock(
        return_value=Entity(
            id="ent_1",
            name="Native Helix",
            entity_type="Project",
            group_id="native_brain",
        )
    )

    result = await RecallNearMissBuilder(graph).entity_near_miss(
        _score("ent_1", result_type="entity"),
        group_id="native_brain",
    )

    assert result == {
        "result_type": "entity",
        "entity": {"name": "Native Helix", "type": "Project"},
        "score": 0.6333,
    }
    graph.get_entity.assert_awaited_once_with("ent_1", "native_brain")


@pytest.mark.asyncio
async def test_cue_context_returns_only_unmerged_episode_cues() -> None:
    graph = AsyncMock()
    cue = EpisodeCue(
        episode_id="ep_1",
        group_id="native_brain",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="mentions: native Helix",
        first_spans=["native Helix path"],
    )
    graph.get_episode_by_id = AsyncMock(return_value=_episode("ep_1"))
    graph.get_episode_cue = AsyncMock(return_value=cue)

    context = await RecallNearMissBuilder(graph).cue_context(
        _score("ep_1", result_type="cue_episode"),
        group_id="native_brain",
    )

    assert context is not None
    assert context.episode.id == "ep_1"
    assert context.cue is cue
    graph.get_episode_by_id.assert_awaited_once_with("ep_1", "native_brain")
    graph.get_episode_cue.assert_awaited_once_with("ep_1", "native_brain")


@pytest.mark.asyncio
async def test_cue_context_skips_merged_episodes() -> None:
    graph = AsyncMock()
    graph.get_episode_by_id = AsyncMock(
        return_value=_episode("ep_merged", projection_state=EpisodeProjectionState.MERGED)
    )
    graph.get_episode_cue = AsyncMock(
        return_value=EpisodeCue(
            episode_id="ep_merged",
            group_id="native_brain",
            projection_state=EpisodeProjectionState.MERGED,
        )
    )

    context = await RecallNearMissBuilder(graph).cue_context(
        _score("ep_merged", result_type="cue_episode"),
        group_id="native_brain",
    )

    assert context is None


def test_cue_near_miss_payload_uses_shared_cue_payload_contract() -> None:
    cue = EpisodeCue(
        episode_id="ep_1",
        group_id="native_brain",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="mentions: native Helix",
        first_spans=["native Helix path"],
        near_miss_count=1,
    )

    result = RecallNearMissBuilder.cue_near_miss(
        cue,
        _score("ep_1", result_type="cue_episode", score=0.71234),
    )

    assert result["result_type"] == "cue_episode"
    assert result["score"] == 0.7123
    assert result["cue"]["episode_id"] == "ep_1"
    assert result["cue"]["projection_state"] == "cue_only"
    assert result["cue"]["near_miss_count"] == 1


@pytest.mark.asyncio
async def test_materializer_records_cue_feedback_and_returns_near_misses() -> None:
    graph = AsyncMock()
    episode = _episode("ep_1")
    cue = EpisodeCue(
        episode_id=episode.id,
        group_id="native_brain",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        cue_text="mentions: native Helix",
        near_miss_count=1,
    )
    entity = Entity(
        id="ent_1",
        name="Native Helix",
        entity_type="Project",
        group_id="native_brain",
    )
    graph.get_entity = AsyncMock(return_value=entity)
    graph.get_episode_by_id = AsyncMock(return_value=episode)
    graph.get_episode_cue = AsyncMock(return_value=cue)
    cue_feedback = AsyncMock()

    results = await RecallNearMissMaterializer(
        near_miss_builder=RecallNearMissBuilder(graph),
        cue_feedback_recorder=cue_feedback,
    ).materialize(
        [
            _score("ent_1", result_type="entity"),
            _score("ep_1", result_type="cue_episode", score=0.71234),
        ],
        group_id="native_brain",
        query="native Helix",
        interaction_type="surfaced",
    )

    assert [result["result_type"] for result in results] == ["entity", "cue_episode"]
    assert results[1]["cue"]["near_miss_count"] == 1
    cue_feedback.record_cue_feedback.assert_awaited_once_with(
        episode,
        0.71234,
        "native Helix",
        interaction_type="surfaced",
        near_miss=True,
    )
