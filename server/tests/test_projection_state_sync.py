"""Tests for shared projection/cue state synchronization."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from engram.ingestion.projection_state import sync_projection_state
from engram.models.episode import EpisodeProjectionState


@pytest.mark.asyncio
async def test_sync_projection_state_updates_episode_and_cue():
    graph = AsyncMock()
    projected_at = datetime(2026, 5, 11, 12, 0, 0)

    await sync_projection_state(
        graph,
        "ep_123",
        group_id="default",
        state=EpisodeProjectionState.PROJECTED,
        reason="projected",
        last_projected_at=projected_at,
        episode_updates={"status": "completed"},
        cue_reason="projected",
        cue_updates={"projection_attempts": 2, "last_projected_at": projected_at},
    )

    graph.update_episode.assert_awaited_once_with(
        "ep_123",
        {
            "status": "completed",
            "projection_state": EpisodeProjectionState.PROJECTED.value,
            "last_projection_reason": "projected",
            "last_projected_at": projected_at.isoformat(),
        },
        group_id="default",
    )
    graph.update_episode_cue.assert_awaited_once_with(
        "ep_123",
        {
            "projection_state": EpisodeProjectionState.PROJECTED,
            "route_reason": "projected",
            "projection_attempts": 2,
            "last_projected_at": projected_at,
        },
        group_id="default",
    )


@pytest.mark.asyncio
async def test_sync_projection_state_skips_cue_when_layer_disabled():
    graph = AsyncMock()

    await sync_projection_state(
        graph,
        "ep_123",
        group_id="default",
        state=EpisodeProjectionState.CUE_ONLY,
        reason="triage_skip_meta",
        cue_layer_enabled=False,
    )

    graph.update_episode.assert_awaited_once()
    graph.update_episode_cue.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_projection_state_does_not_fail_when_cue_sync_fails():
    graph = AsyncMock()
    graph.update_episode_cue = AsyncMock(side_effect=RuntimeError("cue write failed"))

    await sync_projection_state(
        graph,
        "ep_123",
        group_id="default",
        state=EpisodeProjectionState.FAILED,
        reason="extractor_error",
    )

    graph.update_episode.assert_awaited_once()
    graph.update_episode_cue.assert_awaited_once()
