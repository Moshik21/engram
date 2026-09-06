"""The shell's continuous index-outbox drain.

Measured 2026-09-04: a 5/s import queued 4,223 of 5,898 rows in
episode_index_outbox because the shell replayed cue_index_outbox_replay_limit
rows once at startup and never again. The drain loop must (a) drain a backlog
batch after batch and idle when the outbox is empty, (b) wake when capture
enqueues a row nobody is indexing in-process without double-indexing rows
that ARE in flight, (c) back off instead of spinning when every row fails,
(d) yield to recall before each row, and (e) survive an outbox read failure.
"""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig
from engram.ingestion.capture_service import EpisodeCaptureService
from engram.ingestion.cue_index_outbox import CueIndexOutbox
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.retrieval import recall_activity


class FakeGraphStore:
    def __init__(self) -> None:
        self.episodes: dict[str, Episode] = {}

    async def create_episode(self, episode: Episode) -> None:
        self.episodes[episode.id] = episode

    async def update_episode(self, episode_id: str, updates: dict, group_id: str = "default"):
        return None


class RecordingIndex:
    def __init__(self) -> None:
        self.delay = 0.0
        self.episodes: list[str] = []
        self.cues: list[str] = []

    async def index_episode(self, episode: Episode) -> None:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.episodes.append(episode.id)

    async def index_episode_cue(self, cue: EpisodeCue) -> None:
        self.cues.append(cue.episode_id)


class FailingIndex(RecordingIndex):
    async def index_episode(self, episode: Episode) -> None:
        self.episodes.append(episode.id)
        raise RuntimeError("embedding provider down")


async def _noop_materialize(*_args, **_kwargs) -> None:
    return None


def _service(outbox: CueIndexOutbox, index, **cfg_overrides) -> EpisodeCaptureService:
    return EpisodeCaptureService(
        graph_store=FakeGraphStore(),
        search_index=index,
        cfg=ActivationConfig(cue_index_outbox_path=str(outbox.path), **cfg_overrides),
        publish_event=lambda *_args: None,
        materialize_decisions=_noop_materialize,
        cue_index_outbox=outbox,
    )


def _episode(episode_id: str) -> Episode:
    return Episode(id=episode_id, content=f"Row {episode_id} left by a previous shell lifetime.")


def _seed_previous_lifetime(outbox: CueIndexOutbox, *, episodes: int, cues: int = 0) -> None:
    for i in range(episodes):
        outbox.enqueue_episode(_episode(f"ep_old_{i}"))
    for i in range(cues):
        outbox.enqueue(
            EpisodeCue(episode_id=f"ep_cue_{i}", group_id="default", cue_text=f"cue {i}"),
        )


async def _wait_until(predicate, *, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


async def _stop(task: asyncio.Task) -> None:
    assert not task.done(), "drain loop must run until cancelled"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_drain_loop_drains_backlog_in_batches_then_idles(tmp_path) -> None:
    outbox = CueIndexOutbox(tmp_path / "outbox.sqlite3")
    _seed_previous_lifetime(outbox, episodes=3, cues=1)
    index = RecordingIndex()
    service = _service(outbox, index)

    # batch_limit=2 < 3 rows: the loop must keep going while rows succeed
    # instead of waiting a whole idle interval between batches.
    task = asyncio.create_task(
        service.run_cue_index_outbox_drain_loop(batch_limit=2, idle_interval_seconds=5.0),
    )
    await _wait_until(lambda: outbox.pending_episode_count() == 0 and outbox.pending_count() == 0)

    assert sorted(index.episodes) == ["ep_old_0", "ep_old_1", "ep_old_2"]
    assert index.cues == ["ep_cue_0"]
    assert service.index_outbox_status() == {
        "cues_pending": 0,
        "episodes_pending": 0,
        "inflight": 0,
        "drained": 4,
        "failed": 0,
    }

    # Empty outbox: idle, no re-indexing.
    await asyncio.sleep(0.1)
    assert len(index.episodes) == 3
    assert len(index.cues) == 1
    await _stop(task)


@pytest.mark.asyncio
async def test_drain_loop_wakes_on_capture_and_skips_rows_in_flight(tmp_path) -> None:
    outbox = CueIndexOutbox(tmp_path / "outbox.sqlite3")
    index = RecordingIndex()
    service = _service(outbox, index)
    # Idle interval far longer than the test: only a wake can drain anything.
    task = asyncio.create_task(
        service.run_cue_index_outbox_drain_loop(batch_limit=10, idle_interval_seconds=60.0),
    )
    await asyncio.sleep(0.05)  # loop is parked on an empty outbox

    # Slow indexing so the live capture's own task is still running when the
    # drain fetches its batch (the duplicate-row race the guard exists for).
    index.delay = 0.05
    # An older row nobody is indexing in-process (e.g. left by a lost task).
    outbox.enqueue_episode(_episode("ep_unclaimed"))
    # A live capture: enqueues (wakes the loop) AND schedules its own task.
    live_id = await service.store_episode("A live capture during the drain.", source="test")

    await _wait_until(lambda: "ep_unclaimed" in index.episodes, timeout=2.0)
    await service.drain_cue_indexing()
    await _wait_until(lambda: outbox.pending_episode_count() == 0, timeout=2.0)

    assert index.episodes.count(live_id) == 1, "live capture must not be indexed twice"
    assert index.episodes.count("ep_unclaimed") == 1
    await _stop(task)


@pytest.mark.asyncio
async def test_drain_loop_backs_off_when_every_row_fails(tmp_path) -> None:
    outbox = CueIndexOutbox(tmp_path / "outbox.sqlite3")
    _seed_previous_lifetime(outbox, episodes=2)
    index = FailingIndex()
    service = _service(outbox, index)

    task = asyncio.create_task(
        service.run_cue_index_outbox_drain_loop(batch_limit=10, idle_interval_seconds=0.2),
    )
    await asyncio.sleep(0.5)

    # One pass at t=0, then at most one per idle interval (0.2, 0.4): a spin
    # loop would have made hundreds of attempts in half a second.
    assert 2 <= len(index.episodes) <= 6, index.episodes
    assert outbox.pending_episode_count() == 2  # still retryable
    status = service.index_outbox_status()
    assert status["drained"] == 0
    assert status["failed"] == len(index.episodes)
    await _stop(task)


@pytest.mark.asyncio
async def test_drain_loop_yields_to_recall_before_each_row(tmp_path, monkeypatch) -> None:
    outbox = CueIndexOutbox(tmp_path / "outbox.sqlite3")
    _seed_previous_lifetime(outbox, episodes=2, cues=1)
    index = RecordingIndex()
    service = _service(outbox, index)
    yields = 0

    async def counting_wait_idle(*_args, **_kwargs) -> float:
        nonlocal yields
        yields += 1
        return 0.0

    monkeypatch.setattr(recall_activity, "wait_idle", counting_wait_idle)

    task = asyncio.create_task(
        service.run_cue_index_outbox_drain_loop(batch_limit=10, idle_interval_seconds=5.0),
    )
    await _wait_until(lambda: len(index.episodes) == 2 and len(index.cues) == 1)

    assert yields == 3
    await _stop(task)


@pytest.mark.asyncio
async def test_drain_loop_survives_outbox_read_failure(tmp_path, monkeypatch) -> None:
    outbox = CueIndexOutbox(tmp_path / "outbox.sqlite3")
    _seed_previous_lifetime(outbox, episodes=1)
    index = RecordingIndex()
    service = _service(outbox, index)
    real_pending_episodes = outbox.pending_episodes
    failures = {"left": 1}

    def flaky_pending_episodes(**kwargs):
        if failures["left"]:
            failures["left"] -= 1
            raise RuntimeError("database is locked")
        return real_pending_episodes(**kwargs)

    monkeypatch.setattr(outbox, "pending_episodes", flaky_pending_episodes)

    task = asyncio.create_task(
        service.run_cue_index_outbox_drain_loop(batch_limit=10, idle_interval_seconds=0.05),
    )
    await _wait_until(lambda: index.episodes == ["ep_old_0"], timeout=2.0)

    assert failures["left"] == 0
    await _stop(task)
