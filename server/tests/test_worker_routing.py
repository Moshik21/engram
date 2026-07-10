from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.worker_routing import EpisodeWorkerProjectionRouter
from engram.models.episode import EpisodeProjectionState
from engram.retrieval.triage_policy import TriageDecision


def _decision(action: str) -> TriageDecision:
    return TriageDecision(
        action=action,
        score=0.42,
        base_score=0.42,
        threshold_band="test",
        decision_source="test",
    )


def _graph(**overrides):
    defaults = {
        "get_episode_by_id": AsyncMock(return_value=None),
        "update_episode": AsyncMock(),
        "update_episode_cue": AsyncMock(),
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.asyncio
async def test_worker_router_skips_already_projecting_or_done_states() -> None:
    graph = _graph(
        get_episode_by_id=AsyncMock(
            side_effect=[
                {"projection_state": EpisodeProjectionState.PROJECTING.value},
                SimpleNamespace(projection_state=EpisodeProjectionState.PROJECTED),
                SimpleNamespace(projection_state=EpisodeProjectionState.DEAD_LETTER.value),
                SimpleNamespace(projection_state=EpisodeProjectionState.SCHEDULED.value),
                SimpleNamespace(projection_state=EpisodeProjectionState.CUED.value),
            ],
        )
    )
    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig())

    assert await router.should_skip_projection("ep", "brain") is True
    assert await router.should_skip_projection("ep", "brain") is True
    assert await router.should_skip_projection("ep", "brain") is True
    assert await router.should_skip_projection("ep", "brain", skip_scheduled=True) is True
    assert await router.should_skip_projection("ep", "brain") is False


@pytest.mark.asyncio
async def test_worker_router_route_decision_extract_returns_projection_flag() -> None:
    graph = _graph()
    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig(cue_layer_enabled=True))

    should_project = await router.route_decision("ep_extract", _decision("extract"), "brain")

    assert should_project is True
    graph.update_episode.assert_not_awaited()
    graph.update_episode_cue.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_router_route_decision_skip_syncs_episode_and_cue() -> None:
    graph = _graph()
    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig(cue_layer_enabled=True))

    should_project = await router.route_decision("ep_skip", _decision("skip"), "brain")

    assert should_project is False
    graph.update_episode.assert_awaited_once_with(
        "ep_skip",
        {
            "status": "completed",
            "skipped_triage": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "worker_skip_threshold",
        },
        group_id="brain",
    )
    graph.update_episode_cue.assert_awaited_once_with(
        "ep_skip",
        {
            "projection_state": EpisodeProjectionState.CUE_ONLY,
            "route_reason": "worker_skip_threshold",
        },
        group_id="brain",
    )


@pytest.mark.asyncio
async def test_worker_router_route_decision_defer_syncs_episode_and_cue() -> None:
    graph = _graph()
    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig(cue_layer_enabled=True))

    should_project = await router.route_decision("ep_defer", _decision("defer"), "brain")

    assert should_project is False
    graph.update_episode.assert_awaited_once_with(
        "ep_defer",
        {
            "projection_state": EpisodeProjectionState.SCHEDULED.value,
            "last_projection_reason": "worker_deferred_to_triage",
        },
        group_id="brain",
    )
    graph.update_episode_cue.assert_awaited_once_with(
        "ep_defer",
        {
            "projection_state": EpisodeProjectionState.SCHEDULED,
            "route_reason": "worker_deferred_to_triage",
        },
        group_id="brain",
    )


@pytest.mark.asyncio
async def test_worker_router_system_discourse_syncs_episode_and_cue() -> None:
    graph = _graph()
    router = EpisodeWorkerProjectionRouter(graph, ActivationConfig(cue_layer_enabled=True))

    await router.skip_system_discourse("ep_system", "brain")

    graph.update_episode.assert_awaited_once_with(
        "ep_system",
        {
            "status": "completed",
            "skipped_meta": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "system_discourse",
        },
        group_id="brain",
    )
    graph.update_episode_cue.assert_awaited_once_with(
        "ep_system",
        {
            "projection_state": EpisodeProjectionState.CUE_ONLY,
            "route_reason": "system_discourse",
        },
        group_id="brain",
    )
