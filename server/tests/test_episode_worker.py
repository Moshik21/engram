"""Tests for the background EpisodeWorker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.models.episode import EpisodeProjectionState
from engram.retrieval.triage_policy import TriageDecision
from engram.worker import EpisodeWorker


def _make_manager():
    """Create a mock GraphManager."""
    m = MagicMock()
    m.project_episode = AsyncMock()
    m._graph = MagicMock()
    m._graph.update_episode = AsyncMock()
    m._graph.get_episode_by_id = AsyncMock(return_value=None)
    m._graph.get_episode_cue = AsyncMock(return_value=None)
    m._graph.update_episode_cue = AsyncMock()
    m._graph.upsert_episode_cue = AsyncMock()
    m._search = MagicMock()
    m._search.index_episode_cue = AsyncMock()
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


def _scheduled_event(episode_id: str = "ep_1") -> dict:
    return {
        "seq": 2,
        "type": "episode.projection_scheduled",
        "timestamp": 1001.0,
        "group_id": "default",
        "payload": {
            "episodeId": episode_id,
            "reason": "cue_recall_hits",
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
async def test_worker_processes_projection_scheduled_event():
    """Worker calls project_episode for scheduled projection events."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=True)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    bus.publish("default", "episode.projection_scheduled", _scheduled_event()["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_1", "default")

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_deduplicates_scheduled_cue_projection():
    """Queued and scheduled events for the same cue only project once."""
    manager = _make_manager()
    manager._graph.get_episode_by_id = AsyncMock(
        return_value=SimpleNamespace(projection_state=EpisodeProjectionState.SCHEDULED),
    )
    cfg = _make_cfg(triage_enabled=False, cue_layer_enabled=True)
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    bus.publish("default", "episode.queued", _queued_event()["payload"])
    bus.publish("default", "episode.projection_scheduled", _scheduled_event()["payload"])

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
    """With triage enabled (heuristic mode), low-score episodes are marked completed."""
    manager = _make_manager()
    cfg = _make_cfg(triage_enabled=True, triage_min_score=0.5, triage_multi_signal_enabled=False)
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
        {
            "status": "completed",
            "skipped_triage": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "worker_skip_threshold",
        },
        group_id="default",
    )

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_skip_syncs_cue_projection_state():
    """Worker skip routing keeps cue metadata aligned with episode state."""
    manager = _make_manager()
    cfg = _make_cfg(cue_layer_enabled=True)
    worker = EpisodeWorker(manager, cfg)

    await worker._route_episode(
        "ep_low",
        TriageDecision(
            action="skip",
            score=0.1,
            base_score=0.1,
            threshold_band="low",
            decision_source="test",
        ),
        "default",
    )

    manager._graph.update_episode_cue.assert_awaited_once_with(
        "ep_low",
        {
            "projection_state": EpisodeProjectionState.CUE_ONLY,
            "route_reason": "worker_skip_threshold",
        },
        group_id="default",
    )


@pytest.mark.asyncio
async def test_worker_defer_syncs_cue_projection_state():
    """Worker defer routing keeps cue metadata aligned with episode state."""
    manager = _make_manager()
    cfg = _make_cfg(cue_layer_enabled=True)
    worker = EpisodeWorker(manager, cfg)

    await worker._route_episode(
        "ep_mid",
        TriageDecision(
            action="defer",
            score=0.4,
            base_score=0.4,
            threshold_band="mid",
            decision_source="test",
        ),
        "default",
    )

    manager._graph.update_episode_cue.assert_awaited_once_with(
        "ep_mid",
        {
            "projection_state": EpisodeProjectionState.SCHEDULED,
            "route_reason": "worker_deferred_to_triage",
        },
        group_id="default",
    )


@pytest.mark.asyncio
async def test_worker_scores_and_extracts_high():
    """With triage enabled (heuristic mode), high-score episodes are extracted."""
    manager = _make_manager()
    # Disable multi-signal to use legacy heuristic path with triage_min_score
    cfg = _make_cfg(
        triage_enabled=True,
        triage_min_score=0.2,
        triage_multi_signal_enabled=False,
    )
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
async def test_worker_confidence_routing_extract():
    """Multi-signal scorer: high-confidence episodes extracted immediately."""
    manager = _make_manager()
    cfg = _make_cfg(
        triage_enabled=True,
        triage_multi_signal_enabled=True,
        worker_extract_threshold=0.30,  # Low threshold for test
        worker_skip_threshold=0.05,
    )
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    # Rich content with proper names + relationship verbs
    content = (
        "Alice works at Anthropic in San Francisco. "
        "Bob moved to Berlin and married Charlie last January. "
        "David graduated from Stanford and joined Google."
    )
    event = _queued_event(episode_id="ep_rich", content=content)
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_rich", "default")

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_confidence_routing_skip():
    """Multi-signal scorer: low-confidence episodes skipped immediately."""
    manager = _make_manager()
    cfg = _make_cfg(
        triage_enabled=True,
        triage_multi_signal_enabled=True,
        worker_extract_threshold=0.90,
        worker_skip_threshold=0.80,  # Very high skip threshold
    )
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    event = _queued_event(episode_id="ep_skip", content="ok sure")
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_not_called()
    manager._graph.update_episode.assert_called_once_with(
        "ep_skip",
        {
            "status": "completed",
            "skipped_triage": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "worker_skip_threshold",
        },
        group_id="default",
    )

    await worker.stop()


@pytest.mark.asyncio
async def test_worker_confidence_routing_defer():
    """Multi-signal scorer: mid-confidence episodes deferred to triage."""
    manager = _make_manager()
    cfg = _make_cfg(
        triage_enabled=True,
        triage_multi_signal_enabled=True,
        worker_extract_threshold=0.99,  # Almost nothing extracts
        worker_skip_threshold=0.01,  # Almost nothing skips
    )
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    content = "Alice mentioned Python yesterday."
    event = _queued_event(episode_id="ep_defer", content=content)
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    # Deferred to triage with scheduled projection metadata
    manager.project_episode.assert_not_called()
    manager._graph.update_episode.assert_called_once_with(
        "ep_defer",
        {
            "projection_state": EpisodeProjectionState.SCHEDULED.value,
            "last_projection_reason": "worker_deferred_to_triage",
        },
        group_id="default",
    )

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
    meta_content = "Entity ent_abc123 has activation score 0.91 in the retrieval pipeline"
    event = _queued_event(episode_id="ep_meta", content=meta_content)
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    # Should NOT extract
    manager.project_episode.assert_not_called()
    # Should mark as completed with skipped_meta
    manager._graph.update_episode.assert_called_once_with(
        "ep_meta",
        {
            "status": "completed",
            "skipped_meta": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "system_discourse",
        },
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
async def test_worker_extracts_durable_preference_even_with_low_score():
    """Durable guards bypass the worker confidence thresholds."""
    manager = _make_manager()
    cfg = _make_cfg(
        triage_enabled=True,
        triage_multi_signal_enabled=True,
        worker_extract_threshold=0.95,
        worker_skip_threshold=0.90,
    )
    bus = EventBus()

    worker = EpisodeWorker(manager, cfg)
    worker.start("default", bus)

    event = _queued_event(episode_id="ep_pref", content="I prefer Vim.")
    bus.publish("default", "episode.queued", event["payload"])

    await asyncio.sleep(0.1)

    manager.project_episode.assert_called_once_with("ep_pref", "default")
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
