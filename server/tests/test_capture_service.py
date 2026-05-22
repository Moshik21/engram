"""Tests for the explicit episode capture/cue service."""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_service import EpisodeCaptureService
from engram.models.episode import EpisodeProjectionState, EpisodeStatus


class FakeGraphStore:
    def __init__(self) -> None:
        self.episodes = {}
        self.cues = {}
        self.updates = []

    async def create_episode(self, episode):
        self.episodes[episode.id] = episode

    async def update_episode(self, episode_id: str, updates: dict, group_id: str = "default"):
        self.updates.append((episode_id, updates, group_id))
        episode = self.episodes[episode_id]
        if "projection_state" in updates:
            episode.projection_state = EpisodeProjectionState(updates["projection_state"])

    async def upsert_episode_cue(self, cue):
        self.cues[cue.episode_id] = cue


class FakeSearchIndex:
    def __init__(self) -> None:
        self.indexed_cues = []

    async def index_episode_cue(self, cue):
        self.indexed_cues.append(cue)


class SlowSearchIndex:
    def __init__(self) -> None:
        self.started = False

    async def index_episode_cue(self, cue):
        self.started = True
        await asyncio.sleep(1)


class FailingSearchIndex:
    async def index_episode_cue(self, cue):
        raise RuntimeError("index unavailable")


@pytest.mark.asyncio
async def test_capture_service_stores_episode_cue_and_events():
    graph = FakeGraphStore()
    search = FakeSearchIndex()
    events = []

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
        ),
        publish_event=lambda group_id, event, payload: events.append(
            (group_id, event, payload),
        ),
        materialize_decisions=materialize_decisions,
    )

    episode_id = await service.store_episode(
        "Alex moved to Phoenix in 2024 and is working on Engram extraction redesign",
        group_id="default",
        source="test",
    )

    episode = graph.episodes[episode_id]
    await service.drain_cue_indexing()
    assert episode.status == EpisodeStatus.QUEUED
    assert episode.source == "test"
    assert episode_id in graph.cues
    assert search.indexed_cues[0].episode_id == episode_id
    timings = service.last_stage_timings()
    assert timings["capture_store"] >= 0
    assert timings["cue_store"] >= 0
    assert timings["cue_index_enqueue"] >= 0
    assert timings["cue_index"] >= 0
    assert any(update[1]["projection_state"] for update in graph.updates)
    assert [event for _, event, _ in events][:2] == ["episode.queued", "episode.cued"]


@pytest.mark.asyncio
async def test_capture_service_records_write_through_storage_deltas():
    graph = FakeGraphStore()
    search = FakeSearchIndex()
    deltas = []

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=False,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
        record_storage_counts=lambda group_id, **counts: deltas.append(
            (group_id, counts),
        ),
    )

    episode_id = await service.store_episode(
        "Alex wants storage diagnostics to update without live Helix count scans.",
        group_id="brain",
        source="test",
    )

    assert episode_id in graph.cues
    assert deltas == [
        ("brain", {"episodes": 1}),
        ("brain", {"cues": 1}),
    ]


@pytest.mark.asyncio
async def test_capture_service_replays_durable_cue_index_outbox(tmp_path):
    graph = FakeGraphStore()
    outbox_path = tmp_path / "cue-index-outbox.sqlite3"

    async def materialize_decisions(*_args, **_kwargs):
        return None

    capture_service = EpisodeCaptureService(
        graph_store=graph,
        search_index=FailingSearchIndex(),
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            cue_index_outbox_path=str(outbox_path),
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    episode_id = await capture_service.store_episode(
        "Alex wants cue vector indexing to survive a process restart.",
        group_id="brain",
        source="test",
    )
    await capture_service.drain_cue_indexing()

    assert episode_id in graph.cues
    assert capture_service.cue_index_outbox_pending_count() == 1

    search = FakeSearchIndex()
    replay_service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            cue_index_outbox_path=str(outbox_path),
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    replayed = await replay_service.drain_cue_index_outbox(limit=10)

    assert replayed == 1
    assert search.indexed_cues[0].episode_id == episode_id
    assert replay_service.cue_index_outbox_pending_count() == 0


@pytest.mark.asyncio
async def test_capture_service_timeboxes_cue_vector_indexing():
    graph = FakeGraphStore()
    search = SlowSearchIndex()
    events = []

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            capture_cue_vector_index_timeout_ms=10,
        ),
        publish_event=lambda group_id, event, payload: events.append(
            (group_id, event, payload),
        ),
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Alex decided Engram capture should stay fast under native Helix.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started
    await service.drain_cue_indexing()

    assert episode_id in graph.cues
    assert search.started is True
    assert elapsed < 0.1
    assert any(update[0] == episode_id for update in graph.updates)
    assert [event for _, event, _ in events][:2] == ["episode.queued", "episode.cued"]


@pytest.mark.asyncio
async def test_capture_service_marks_system_discourse_cue_only():
    graph = FakeGraphStore()
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(cue_layer_enabled=True),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    episode_id = await service.store_episode(
        "Entity ent_abc123 has activation score 0.91 in the retrieval pipeline",
    )

    assert episode_id not in graph.cues
    assert graph.updates[-1] == (
        episode_id,
        {
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "system_discourse",
        },
        "default",
    )


@pytest.mark.asyncio
async def test_capture_service_runs_decision_materializer_with_episode_context():
    graph = FakeGraphStore()
    search = FakeSearchIndex()
    calls = []

    async def materialize_decisions(content: str, *, episode_id: str, group_id: str):
        calls.append((content, episode_id, group_id))

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=False,
            decision_graph_enabled=True,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    episode_id = await service.store_episode(
        "We decided Engram should keep one brain per person.",
        group_id="default",
        source="test",
    )

    assert calls == [
        (
            "We decided Engram should keep one brain per person.",
            episode_id,
            "default",
        ),
    ]


@pytest.mark.asyncio
async def test_capture_service_defers_decision_materializer_for_auto_sources():
    graph = FakeGraphStore()
    search = FakeSearchIndex()
    calls = []

    async def materialize_decisions(content: str, *, episode_id: str, group_id: str):
        calls.append((content, episode_id, group_id))

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=False,
            decision_graph_enabled=True,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    await service.store_episode(
        "We decided Engram hook capture should stay cheap.",
        group_id="default",
        source="auto:prompt",
    )

    assert calls == []
