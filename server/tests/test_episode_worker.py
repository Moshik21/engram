"""Tests for the background EpisodeWorker."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.worker import EpisodeWorker


def _make_manager():
    """Create a mock GraphManager."""
    m = MagicMock()
    m.project_episode = AsyncMock()
    m._graph = MagicMock()
    m._graph.update_episode = AsyncMock()
    return m


def _make_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "triage_enabled": True,
        "triage_min_score": 0.2,
        "worker_enabled": True,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _queued_event(episode_id: str = "ep_1", content: str = "Alice works at Anthropic") -> dict:
    return {
        "seq": 1,
        "type": "episode.queued",
        "timestamp": 1000.0,
        "group_id": "default",
        "payload": {
            "episode": {
                "episodeId": episode_id,
                "content": content,
                "source": "test",
                "status": "queued",
            },
        },
    }


@pytest.mark.asyncio
async def test_worker_starts_and_subscribes():
    """Worker subscribes to EventBus on start."""
    manager = _make_manager()
    cfg = _make_cfg()
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    assert worker._task is not None
    assert worker._queue is not None
    assert len(bus._subscribers.get("default", [])) == 1

    await worker.stop()
    assert worker._task is None


@pytest.mark.asyncio
async def test_worker_processes_queued_event():
    """Worker calls project_episode for queued events."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)  # Extract everything
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Publish an event
    bus.publish("default", "episode.queued", _queued_event()["payload"])

    # Let the worker process
    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_1", "default")

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_ignores_non_queued_events():
    """Worker skips events that aren't episode.queued."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    bus.publish("default", "consolidation.started", {"cycle_id": "cyc_1"})
    bus.publish("default", "entity.created", {"entity_id": "ent_1"})

    await asyncio.sleep(0.1)

    manager.project_episode.assert_not_called()

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_scores_and_skips_low():
    """With triage enabled, low-score episodes are marked completed."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=True, triage_min_score=0.5)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Very short content → low score
    event = _queued_event(episode_id="ep_low", content="hi")
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    # Should NOT extract
    manager.project_episode.assert_not_called()
    # Should mark as completed
    manager._graph.update_episode.assert_called_once_with(
        "ep_low",
        {"status": "completed", "skipped_triage": True},
        group_id="default",
    )

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_scores_and_extracts_high():
    """With triage enabled, high-score episodes are extracted."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=True, triage_min_score=0.2)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Rich content → high score
    content = (
        "Alice and Bob discussed the Python project at Google. "
        "Charlie from Anthropic joined in San Francisco."
    )
    event = _queued_event(episode_id="ep_high", content=content)
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_high", "default")
    manager._graph.update_episode.assert_not_called()

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_extracts_all_when_triage_disabled():
    """With triage_enabled=False, all episodes are extracted regardless of score."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Even short content should be extracted
    event = _queued_event(episode_id="ep_any", content="hi")
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_any", "default")

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_handles_extraction_error():
    """project_episode failure doesn't crash the worker."""
    manager = _make_manager()
    manager.project_episode = AsyncMock(side_effect=RuntimeError("extraction failed"))
    cfg = _make_cfg(triage_enabled=False)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    bus.publish("default", "episode.queued", _queued_event("ep_fail")["payload"])

    await asyncio.sleep(0.1)

    # Worker should still be running
    assert worker._task is not None
    assert not worker._task.done()

    # Send another event to verify worker is alive
    manager.project_episode = AsyncMock()  # Reset to succeed
    bus.publish("default", "episode.queued", _queued_event("ep_ok")["payload"])
    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_ok", "default")

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_skips_system_discourse():
    """Worker skips meta-commentary episodes regardless of triage setting."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)  # Even without triage, meta is skipped
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # System meta-commentary content (2+ system pattern matches)
    meta_content = (
        "Entity ent_abc123 has activation score 0.91 in the retrieval pipeline"
    )
    event = _queued_event(episode_id="ep_meta", content=meta_content)
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    # Should NOT extract
    manager.project_episode.assert_not_called()
    # Should mark as completed with skipped_meta
    manager._graph.update_episode.assert_called_once_with(
        "ep_meta",
        {"status": "completed", "skipped_meta": True},
        group_id="default",
    )

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_allows_world_discourse():
    """Worker processes world-discourse episodes normally."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    event = _queued_event(
        episode_id="ep_world",
        content="Alice is a data scientist at Acme Corp in Berlin",
    )
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_world", "default")
    manager._graph.update_episode.assert_not_called()

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_stop_cancels_cleanly():
    """Worker stop cancels the task without errors."""
    manager = _make_manager()
    cfg = _make_cfg()
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    task = worker._task
    assert task is not None

    await worker.stop()

    assert worker._task is None
    assert worker._queue is None
    assert task.done()


@pytest.mark.asyncio
async def test_worker_idempotent_start():
    """Calling start twice doesn't create duplicate tasks."""
    manager = _make_manager()
    cfg = _make_cfg()
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)
    first_task = worker._task

    worker.start("default", bus)
    second_task = worker._task

    assert first_task is second_task
    assert len(bus._subscribers.get("default", [])) == 1

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_missing_episode_id_ignored():
    """Events without episodeId in payload are silently ignored."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=False)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Malformed event - no episodeId
    bus.publish("default", "episode.queued", {"episode": {"content": "no id"}})

    await asyncio.sleep(0.1)

    manager.project_episode.assert_not_called()

    await worker.stop()
