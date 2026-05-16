from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.ingestion.worker_events import (
    EPISODE_PROJECTION_SCHEDULED_EVENT,
    EPISODE_QUEUED_EVENT,
    EpisodeWorkerEvent,
    load_full_auto_content,
    parse_episode_worker_event,
)


def test_parse_episode_worker_event_ignores_unhandled_events() -> None:
    assert parse_episode_worker_event({"type": "entity.created", "payload": {}}) is None
    assert (
        parse_episode_worker_event(
            {"type": EPISODE_QUEUED_EVENT, "payload": {"episode": {}}},
        )
        is None
    )
    assert (
        parse_episode_worker_event(
            {"type": EPISODE_PROJECTION_SCHEDULED_EVENT, "payload": {}},
        )
        is None
    )


def test_parse_episode_worker_event_normalizes_queued_event() -> None:
    event = parse_episode_worker_event(
        {
            "type": EPISODE_QUEUED_EVENT,
            "payload": {
                "episode": {
                    "episodeId": "ep_queued",
                    "content": "Alice works at Anthropic",
                    "source": "auto:prompt",
                }
            },
        }
    )

    assert event == EpisodeWorkerEvent(
        kind=EPISODE_QUEUED_EVENT,
        episode_id="ep_queued",
        content="Alice works at Anthropic",
        source="auto:prompt",
    )
    assert event is not None
    assert event.is_auto_turn is True
    assert event.is_scheduled_projection is False


def test_parse_episode_worker_event_normalizes_scheduled_projection() -> None:
    event = parse_episode_worker_event(
        {
            "type": EPISODE_PROJECTION_SCHEDULED_EVENT,
            "payload": {"episodeId": "ep_scheduled"},
        }
    )

    assert event == EpisodeWorkerEvent(
        kind=EPISODE_PROJECTION_SCHEDULED_EVENT,
        episode_id="ep_scheduled",
    )
    assert event is not None
    assert event.is_scheduled_projection is True
    assert event.is_auto_turn is False


@pytest.mark.asyncio
async def test_load_full_auto_content_expands_compact_auto_payload() -> None:
    graph = SimpleNamespace(
        get_episode_by_id=AsyncMock(
            return_value=SimpleNamespace(content="full stored auto content"),
        )
    )
    event = EpisodeWorkerEvent(
        kind=EPISODE_QUEUED_EVENT,
        episode_id="ep_auto",
        content="short",
        source="auto:response",
    )

    expanded = await load_full_auto_content(graph, event, "brain")

    assert expanded == EpisodeWorkerEvent(
        kind=EPISODE_QUEUED_EVENT,
        episode_id="ep_auto",
        content="full stored auto content",
        source="auto:response",
    )
    graph.get_episode_by_id.assert_awaited_once_with("ep_auto", "brain")


@pytest.mark.asyncio
async def test_load_full_auto_content_leaves_non_auto_or_long_content_alone() -> None:
    graph = SimpleNamespace(get_episode_by_id=AsyncMock())
    non_auto = EpisodeWorkerEvent(
        kind=EPISODE_QUEUED_EVENT,
        episode_id="ep_manual",
        content="short",
        source="manual",
    )
    long_auto = EpisodeWorkerEvent(
        kind=EPISODE_QUEUED_EVENT,
        episode_id="ep_long",
        content="x" * 200,
        source="auto:prompt",
    )

    assert await load_full_auto_content(graph, non_auto, "brain") is non_auto
    assert await load_full_auto_content(graph, long_auto, "brain") is long_auto
    graph.get_episode_by_id.assert_not_awaited()
