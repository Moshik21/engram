"""Tests for recall episode expansion."""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from engram.config import ActivationConfig
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.retrieval.episode_traversal import RecallEpisodeTraversal
from engram.retrieval.result_builder import RecallResultBuilder
from engram.utils.dates import utc_now


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


def _service(graph: AsyncMock, cfg: ActivationConfig) -> RecallEpisodeTraversal:
    return RecallEpisodeTraversal(
        graph_store=graph,
        cfg=cfg,
        result_builder=RecallResultBuilder(cfg),
    )


@pytest.mark.asyncio
async def test_entity_linked_traversal_appends_group_scoped_episode_results() -> None:
    graph = AsyncMock()
    episodes = {
        "ep_seen": _episode("ep_seen"),
        "ep_new": _episode("ep_new"),
        "ep_merged": _episode("ep_merged", projection_state=EpisodeProjectionState.MERGED),
    }
    graph.get_episodes_for_entity = AsyncMock(return_value=["ep_seen", "ep_new", "ep_merged"])
    graph.get_episode_by_id = AsyncMock(
        side_effect=lambda episode_id, group_id: episodes[episode_id]
    )
    graph.get_episode_entities = AsyncMock(return_value=["ent_parent"])
    cfg = ActivationConfig(
        entity_episode_traversal_enabled=True,
        entity_episode_max_entities=1,
        entity_episode_max_per_entity=5,
        entity_episode_weight=0.5,
    )
    results = [
        {
            "result_type": "entity",
            "entity": {"id": "ent_parent", "name": "Parent"},
            "score": 0.8,
        },
        {
            "result_type": "entity",
            "entity": {"id": "ent_low", "name": "Low"},
            "score": 0.7,
        },
    ]
    seen_episode_ids = {"ep_seen"}

    await _service(graph, cfg).append_entity_linked_episodes(
        results,
        group_id="native_brain",
        seen_episode_ids=seen_episode_ids,
    )

    assert [result.get("episode", {}).get("id") for result in results if "episode" in result] == [
        "ep_new"
    ]
    appended = results[-1]
    assert appended["score"] == pytest.approx(0.4)
    assert appended["score_breakdown"]["entity_traversal"] is True
    assert appended["score_breakdown"]["parent_entity_id"] == "ent_parent"
    assert appended["linked_entities"] == ["ent_parent"]
    assert seen_episode_ids == {"ep_seen", "ep_new"}
    graph.get_episodes_for_entity.assert_awaited_once_with(
        "ent_parent",
        group_id="native_brain",
        limit=5,
    )
    graph.get_episode_entities.assert_awaited_once_with("ep_new", group_id="native_brain")


@pytest.mark.asyncio
async def test_temporal_traversal_appends_adjacent_episode_results() -> None:
    graph = AsyncMock()
    adjacent = [
        _episode("ep_duplicate"),
        _episode("ep_adjacent"),
        _episode("ep_merged", projection_state=EpisodeProjectionState.MERGED),
    ]
    graph.get_adjacent_episodes = AsyncMock(return_value=adjacent)
    graph.get_episode_entities = AsyncMock(return_value=["ent_adjacent"])
    cfg = ActivationConfig(
        temporal_contiguity_enabled=True,
        temporal_contiguity_max_adjacent=2,
        temporal_contiguity_weight=0.25,
    )
    results = [
        {
            "result_type": "episode",
            "episode": {"id": "ep_parent"},
            "score": 0.9,
        },
        {
            "result_type": "episode",
            "episode": {"id": "ep_lower"},
            "score": 0.2,
        },
    ]
    seen_episode_ids = {"ep_parent", "ep_duplicate"}

    await _service(graph, cfg).append_temporal_episodes(
        results,
        group_id="native_brain",
        seen_episode_ids=seen_episode_ids,
    )

    appended = results[-1]
    assert appended["episode"]["id"] == "ep_adjacent"
    assert appended["score"] == pytest.approx(0.225)
    assert appended["score_breakdown"]["temporal_contiguity"] is True
    assert appended["score_breakdown"]["parent_episode_id"] == "ep_parent"
    assert appended["linked_entities"] == ["ent_adjacent"]
    assert seen_episode_ids == {"ep_parent", "ep_duplicate", "ep_adjacent"}
    graph.get_adjacent_episodes.assert_has_awaits(
        [
            call("ep_parent", group_id="native_brain", limit=2),
            call("ep_lower", group_id="native_brain", limit=2),
        ]
    )
    graph.get_episode_entities.assert_awaited_once_with("ep_adjacent", group_id="native_brain")


@pytest.mark.asyncio
async def test_traversal_read_errors_do_not_abort_recall_expansion() -> None:
    graph = AsyncMock()
    graph.get_episodes_for_entity = AsyncMock(side_effect=RuntimeError("lookup failed"))
    graph.get_adjacent_episodes = AsyncMock(side_effect=RuntimeError("lookup failed"))
    cfg = ActivationConfig(
        entity_episode_traversal_enabled=True,
        temporal_contiguity_enabled=True,
    )
    results = [
        {
            "result_type": "entity",
            "entity": {"id": "ent_parent", "name": "Parent"},
            "score": 0.8,
        },
        {
            "result_type": "episode",
            "episode": {"id": "ep_parent"},
            "score": 0.9,
        },
    ]
    seen_episode_ids = {"ep_parent"}

    service = _service(graph, cfg)
    await service.append_entity_linked_episodes(
        results,
        group_id="native_brain",
        seen_episode_ids=seen_episode_ids,
    )
    await service.append_temporal_episodes(
        results,
        group_id="native_brain",
        seen_episode_ids=seen_episode_ids,
    )

    assert len(results) == 2
    assert seen_episode_ids == {"ep_parent"}
