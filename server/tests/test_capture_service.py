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
        self.deleted_groups = []

    async def create_episode(self, episode):
        self.episodes[episode.id] = episode

    async def delete_group(self, group_id: str):
        self.deleted_groups.append(group_id)
        self.episodes = {
            episode_id: episode
            for episode_id, episode in self.episodes.items()
            if episode.group_id != group_id
        }
        self.cues = {
            episode_id: cue
            for episode_id, cue in self.cues.items()
            if cue.group_id != group_id
        }

    async def update_episode(self, episode_id: str, updates: dict, group_id: str = "default"):
        self.updates.append((episode_id, updates, group_id))
        episode = self.episodes[episode_id]
        if "projection_state" in updates:
            episode.projection_state = EpisodeProjectionState(updates["projection_state"])

    async def upsert_episode_cue(self, cue):
        self.cues[cue.episode_id] = cue


class SlowCreateGraphStore(FakeGraphStore):
    async def create_episode(self, episode):
        await asyncio.sleep(0.06)
        await super().create_episode(episode)


class FakeSearchIndex:
    def __init__(self) -> None:
        self.indexed_cues = []

    async def index_episode_cue(self, cue):
        self.indexed_cues.append(cue)


class SlowSearchIndex:
    def __init__(self) -> None:
        self.started = False
        self.completed = False

    async def index_episode_cue(self, cue):
        self.started = True
        await asyncio.sleep(1)
        self.completed = True


class CountingSlowSearchIndex:
    def __init__(self, delay_seconds: float = 0.05) -> None:
        self.delay_seconds = delay_seconds
        self.active_index_writes = 0
        self.max_active_index_writes = 0

    async def index_episode_cue(self, cue):
        self.active_index_writes += 1
        self.max_active_index_writes = max(
            self.max_active_index_writes,
            self.active_index_writes,
        )
        try:
            await asyncio.sleep(self.delay_seconds)
        finally:
            self.active_index_writes -= 1


class TimestampSearchIndex:
    def __init__(self) -> None:
        self.started_at: float | None = None

    async def index_episode_cue(self, cue):
        self.started_at = asyncio.get_running_loop().time()


class SlowCueGraphStore(FakeGraphStore):
    async def upsert_episode_cue(self, cue):
        await asyncio.sleep(0.05)
        await super().upsert_episode_cue(cue)


class CountingSlowCueGraphStore(FakeGraphStore):
    def __init__(self, delay_seconds: float = 0.05) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds
        self.active_cue_writes = 0
        self.max_active_cue_writes = 0

    async def upsert_episode_cue(self, cue):
        self.active_cue_writes += 1
        self.max_active_cue_writes = max(
            self.max_active_cue_writes,
            self.active_cue_writes,
        )
        try:
            await asyncio.sleep(self.delay_seconds)
            await super().upsert_episode_cue(cue)
        finally:
            self.active_cue_writes -= 1


class SlowProjectionSyncGraphStore(FakeGraphStore):
    async def update_episode(self, episode_id: str, updates: dict, group_id: str = "default"):
        await asyncio.sleep(0.2)
        await super().update_episode(episode_id, updates, group_id)


class FailingSearchIndex:
    async def index_episode_cue(self, cue):
        raise RuntimeError("index unavailable")


@pytest.mark.asyncio
async def test_capture_service_warm_capture_store_does_not_retain_episode():
    graph = FakeGraphStore()
    events = []
    storage_deltas = []

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=FakeSearchIndex(),
        cfg=ActivationConfig(cue_layer_enabled=True),
        publish_event=lambda group_id, event, payload: events.append(
            (group_id, event, payload),
        ),
        materialize_decisions=materialize_decisions,
        record_storage_counts=lambda group_id, **deltas: storage_deltas.append(
            (group_id, deltas),
        ),
    )

    timings = await service.warm_capture_store()

    assert timings["capture_store_warmup"] >= 0
    assert timings["cue_store_warmup"] >= 0
    assert timings["capture_store_warmup_cleanup"] >= 0
    assert graph.episodes == {}
    assert graph.cues == {}
    assert graph.deleted_groups == ["__engram_capture_warmup__"]
    assert events == []
    assert storage_deltas == []


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
async def test_capture_service_defers_slow_raw_episode_store_without_losing_episode():
    graph = SlowCreateGraphStore()
    search = FakeSearchIndex()
    events = []
    storage_deltas = []

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=False,
            capture_store_timeout_ms=1,
        ),
        publish_event=lambda group_id, event, payload: events.append(
            (group_id, event, payload),
        ),
        materialize_decisions=materialize_decisions,
        record_storage_counts=lambda group_id, **deltas: storage_deltas.append(
            (group_id, deltas),
        ),
    )

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Raw capture should acknowledge before a slow native write completes.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started
    timings = service.last_stage_timings()

    assert elapsed < 0.03
    assert timings["capture_store_timeout"] >= 0
    assert "capture_store" not in timings
    assert episode_id not in graph.episodes
    assert events == []
    assert storage_deltas == []

    await service.drain_cue_indexing()

    assert episode_id in graph.episodes
    assert service.last_stage_timings()["capture_store"] >= 50
    assert [event for _, event, _ in events] == ["episode.queued"]
    assert storage_deltas == [("default", {"episodes": 1})]


@pytest.mark.asyncio
async def test_capture_service_accepts_per_write_store_timeout_override():
    graph = SlowCreateGraphStore()
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=False,
            capture_store_timeout_ms=1000,
        ),
        publish_event=lambda *_args, **_kwargs: None,
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Agent write should use a shorter per-call raw capture wait.",
        group_id="default",
        source="mcp",
        capture_store_timeout_ms=1,
    )
    elapsed = asyncio.get_running_loop().time() - started
    timings = service.last_stage_timings()

    assert elapsed < 0.03
    assert timings["capture_store_timeout"] >= 0
    assert "capture_store" not in timings
    assert episode_id not in graph.episodes

    await service.drain_cue_indexing()

    assert episode_id in graph.episodes


@pytest.mark.asyncio
async def test_capture_service_defers_slow_cue_storage_without_losing_cue():
    graph = SlowCueGraphStore()
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=False,
            capture_cue_store_timeout_ms=1,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    episode_id = await service.store_episode(
        "Alex wants capture acknowledgements to stay fast when cue writes stall.",
        group_id="default",
        source="test",
    )

    timings = service.last_stage_timings()
    assert timings["capture_store"] >= 0
    assert timings["cue_store_timeout"] >= 0
    assert episode_id not in graph.cues

    await service.drain_cue_indexing()

    assert episode_id in graph.cues
    assert graph.episodes[episode_id].projection_state != EpisodeProjectionState.QUEUED


@pytest.mark.asyncio
async def test_capture_service_default_cue_wait_is_agent_bounded():
    graph = CountingSlowCueGraphStore(delay_seconds=0.75)
    search = FakeSearchIndex()
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        cue_vector_index_enabled=False,
    )

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=cfg,
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Default capture should not wait a full second for slow cue persistence.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started
    timings = service.last_stage_timings()

    assert cfg.capture_cue_store_timeout_ms == 250
    assert elapsed < 0.5
    assert timings["cue_store_timeout"] < 500
    assert episode_id not in graph.cues

    await service.drain_cue_indexing()

    assert episode_id in graph.cues
    assert graph.episodes[episode_id].projection_state != EpisodeProjectionState.QUEUED


@pytest.mark.asyncio
async def test_capture_service_serializes_deferred_cue_storage():
    graph = CountingSlowCueGraphStore()
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=False,
            capture_cue_store_timeout_ms=1,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    await asyncio.gather(
        service.store_episode(
            "First capture should not let deferred cue writes saturate storage.",
            group_id="default",
            source="test",
        ),
        service.store_episode(
            "Second capture should queue cue storage behind the first cue write.",
            group_id="default",
            source="test",
        ),
    )
    await service.drain_cue_indexing()

    assert len(graph.cues) == 2
    assert graph.max_active_cue_writes == 1


@pytest.mark.asyncio
async def test_capture_service_serializes_background_cue_indexing():
    graph = FakeGraphStore()
    search = CountingSlowSearchIndex(delay_seconds=0.05)

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            capture_cue_store_timeout_ms=1000,
            capture_cue_vector_index_timeout_ms=1000,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    await asyncio.gather(
        *[
            service.store_episode(
                f"Capture {index} should not launch concurrent cue vector writes.",
                group_id="default",
                source="test",
            )
            for index in range(4)
        ],
    )
    await service.drain_cue_indexing()

    assert search.max_active_index_writes == 1


@pytest.mark.asyncio
async def test_capture_service_defers_cue_indexing_until_capture_quiet_period():
    graph = FakeGraphStore()
    search = TimestampSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            capture_cue_vector_index_quiet_period_ms=50,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    await service.store_episode(
        "Cue vector indexing should wait until the live capture burst is quiet.",
        group_id="default",
        source="test",
    )
    assert search.started_at is None

    await service.drain_cue_indexing()

    assert search.started_at is not None
    assert (search.started_at - started) >= 0.045
    assert service.last_stage_timings()["cue_index_quiet_wait"] >= 45


@pytest.mark.asyncio
async def test_capture_service_quiet_period_starts_after_slow_capture_store():
    graph = SlowCreateGraphStore()
    search = TimestampSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            capture_cue_vector_index_quiet_period_ms=50,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    await service.store_episode(
        "Cue vector indexing should not treat slow capture storage as idle time.",
        group_id="default",
        source="test",
    )
    assert search.started_at is None

    await service.drain_cue_indexing()

    assert search.started_at is not None
    assert (search.started_at - started) >= 0.1
    assert service.last_stage_timings()["cue_index_quiet_wait"] >= 45


@pytest.mark.asyncio
async def test_capture_service_queues_cue_storage_when_backlog_exists():
    graph = CountingSlowCueGraphStore(delay_seconds=0.2)
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=False,
            capture_cue_store_timeout_ms=1000,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    first = asyncio.create_task(
        service.store_episode(
            "First capture holds cue persistence open.",
            group_id="default",
            source="test",
        ),
    )
    for _ in range(50):
        if graph.active_cue_writes:
            break
        await asyncio.sleep(0.001)
    assert graph.active_cue_writes == 1

    started = asyncio.get_running_loop().time()
    second_id = await service.store_episode(
        "Second capture should enqueue cue storage without waiting for timeout.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started
    timings = service.last_stage_timings()

    assert second_id not in graph.cues
    assert elapsed < 0.05
    assert timings["cue_store_queued"] >= 0
    assert "cue_store_timeout" not in timings

    await first
    await service.drain_cue_indexing()

    assert second_id in graph.cues
    assert graph.max_active_cue_writes == 1


@pytest.mark.asyncio
async def test_capture_service_acknowledges_after_cue_persist_before_projection_sync():
    graph = SlowProjectionSyncGraphStore()
    search = FakeSearchIndex()

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=False,
            capture_cue_store_timeout_ms=1000,
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    started = asyncio.get_running_loop().time()
    episode_id = await service.store_episode(
        "Capture should acknowledge once cue persistence is durable.",
        group_id="default",
        source="test",
    )
    elapsed = asyncio.get_running_loop().time() - started
    timings = service.last_stage_timings()

    assert episode_id in graph.cues
    assert elapsed < 0.1
    assert timings["cue_store"] >= 0
    assert "cue_store_timeout" not in timings

    await service.drain_cue_indexing()

    assert graph.episodes[episode_id].projection_state != EpisodeProjectionState.QUEUED


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

    startup_replayed = await replay_service.drain_cue_index_outbox(
        limit=10,
        include_failed=False,
    )
    replayed = await replay_service.drain_cue_index_outbox(limit=10)

    assert startup_replayed == 0
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
    assert search.completed is True
    assert elapsed < 0.1
    assert service.last_stage_timings()["cue_index_timeout"] >= 0
    assert any(update[0] == episode_id for update in graph.updates)
    assert [event for _, event, _ in events][:2] == ["episode.queued", "episode.cued"]


@pytest.mark.asyncio
async def test_capture_service_soft_timeout_clears_successful_cue_index_outbox(tmp_path):
    graph = FakeGraphStore()
    search = CountingSlowSearchIndex(delay_seconds=0.03)
    outbox_path = tmp_path / "cue-index-outbox.sqlite3"

    async def materialize_decisions(*_args, **_kwargs):
        return None

    service = EpisodeCaptureService(
        graph_store=graph,
        search_index=search,
        cfg=ActivationConfig(
            cue_layer_enabled=True,
            cue_vector_index_enabled=True,
            capture_cue_vector_index_timeout_ms=1,
            cue_index_outbox_path=str(outbox_path),
        ),
        publish_event=lambda *_args: None,
        materialize_decisions=materialize_decisions,
    )

    await service.store_episode(
        "Cue vector indexing can exceed the soft threshold and still finish.",
        group_id="brain",
        source="test",
    )
    await service.drain_cue_indexing()

    assert search.max_active_index_writes == 1
    assert service.cue_index_outbox_pending_count() == 0
    assert service.last_stage_timings()["cue_index_timeout"] >= 0


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
