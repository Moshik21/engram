"""Ticket 26 — the episode->helix-id cold cliff, and the fan-out that multiplies it.

`_resolve_episode_helix_id` has no targeted route (the Entity twin is
`find_entity_by_entity_id`, schema.hx:261), so a cache miss can only be served
by `find_episodes_by_group`, which returns every episode row in the group.
Measured on the live brain: **533.2 ms cold vs 0.263 ms warm**, 4/4 alternating
trials.

Two amplifiers turn that one-off warm-up cost into a permanent T1:

1. **N misses issued N scans.** `episode_graph_signal` gathers up to 60 of these
   with no concurrency limit against a `max_workers=4` native executor.
2. **The work was thrown away.** `asyncio.wait_for` cancels the awaiting
   coroutines at `episode_graph_signal_timeout_ms=40`, so the `for item in
   results:` loop that populates the cache never ran — while the executor jobs,
   which cannot be cancelled mid-flight, ran to completion anyway. 60 full scans
   burned, zero rows cached, **and the next recall did it all again**. Cold, the
   stage could never converge.

SUBSTRATE. These tests run a real `HelixGraphStore` against an in-memory row
table on a `ThreadPoolExecutor(max_workers=4)` — the native transport's shape.
The resolution code, the gather and the timeout are production code; only the
storage engine is stubbed. Absolute milliseconds here are therefore synthetic
and no timing assertion is made. **Every assertion below is a COUNT** — scans
issued, reads started, peak in-flight — which is exact on either substrate.
(A native fixture is not available: engine init fails `No space left on device`
at 65.5 GiB free on this machine. Reported separately.)
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from engram.config import ActivationConfig, HelixDBConfig
from engram.retrieval.episode_graph_signal import (
    _EPISODE_GRAPH_READ_CONCURRENCY,
    EntityGraphSignal,
    apply_episode_graph_signal,
)
from engram.retrieval.scorer import ScoredResult
from engram.storage.helix.graph import HelixGraphStore

GROUP = "cliff"
NATIVE_WORKERS = 4


# ── Throwaway store ─────────────────────────────────────────────────


class RowTableTransport:
    """Stands in for the native transport: same executor shape, exact counters.

    ``row_cost_s`` is charged per row inside the worker thread, so a scan costs
    O(rows) the way ``find_episodes_by_group`` does, and — crucially — it is
    charged in a thread the event loop cannot cancel, reproducing the
    "cancelled the wrapper, kept burning the worker" behaviour that makes this
    a T1 rather than a latency nit.
    """

    def __init__(self, episode_count: int, row_cost_s: float = 0.0) -> None:
        self.rows = [
            {
                "id": 1000 + i,
                "episode_id": f"ep-{i:06d}",
                "group_id": GROUP,
                "content": "x" * 64,
            }
            for i in range(episode_count)
        ]
        self._by_helix_id = {row["id"]: row for row in self.rows}
        self._row_cost_s = row_cost_s
        self._executor = ThreadPoolExecutor(max_workers=NATIVE_WORKERS)
        self.calls: dict[str, int] = {}
        self.started: dict[str, int] = {}
        self.completed: dict[str, int] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=False)

    def _blocking(self, endpoint: str, payload: dict) -> list[dict]:
        self.started[endpoint] = self.started.get(endpoint, 0) + 1
        try:
            if endpoint in ("find_episodes_by_group", "find_episodes_all"):
                out = []
                for row in self.rows:
                    if self._row_cost_s:
                        time.sleep(self._row_cost_s)
                    out.append(dict(row))
                return out
            if endpoint == "get_episode_entities":
                row = self._by_helix_id.get(payload.get("id"))
                if row is None:
                    return []
                return [{"entity_id": "ent_hot", "group_id": GROUP}]
            raise AssertionError(f"unexpected endpoint {endpoint!r}")
        finally:
            self.completed[endpoint] = self.completed.get(endpoint, 0) + 1

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        self.calls[endpoint] = self.calls.get(endpoint, 0) + 1
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._blocking, endpoint, payload)


def _store(episode_count: int, row_cost_s: float = 0.0):
    transport = RowTableTransport(episode_count, row_cost_s=row_cost_s)
    store = HelixGraphStore(HelixDBConfig(), client=transport)
    return store, transport


# ── The scan itself ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_misses_issue_exactly_one_group_scan() -> None:
    """60 cold misses at once must cost ONE scan, not 60.

    Pre-fix this asserted 60: every coroutine checked the cache before any of
    them had filled it.
    """
    store, transport = _store(500)
    try:
        ids = [f"ep-{i:06d}" for i in range(60)]
        linked = await asyncio.gather(
            *(store.get_episode_entities(eid, group_id=GROUP) for eid in ids)
        )
    finally:
        transport.close()

    assert transport.calls["find_episodes_by_group"] == 1, transport.calls
    # Anti-inertness: the single scan must actually have RESOLVED all 60,
    # otherwise "1 scan" would be trivially satisfied by resolving nothing.
    assert transport.calls["get_episode_entities"] == 60
    assert all(row == ["ent_hot"] for row in linked)


@pytest.mark.asyncio
async def test_unresolvable_episode_does_not_rescan_on_every_call() -> None:
    """The 533 ms is paid once per TTL, not once per call, for a missing id.

    An id the scan does not find was never remembered, so the "0.263 ms warm"
    number never applied to deleted episodes, cross-group ids, or the ticket-21
    index gap: each one paid a fresh full scan forever.
    """
    store, transport = _store(500)
    try:
        for _ in range(10):
            assert await store.get_episode_entities("ep-999999", group_id=GROUP) == []
    finally:
        transport.close()

    assert transport.calls["find_episodes_by_group"] == 1, transport.calls


@pytest.mark.asyncio
async def test_scan_survives_caller_cancellation_and_warms_the_cache() -> None:
    """A stage timeout must not throw away a scan the executor already ran.

    This is the convergence property. Pre-fix, `wait_for` cancelled the
    coroutine before its caching loop ran, so the executor completed the scan
    and the result was discarded — every recall re-paid it.
    """
    store, transport = _store(400, row_cost_s=0.0002)  # ~80 ms scan
    try:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                store.get_episode_entities("ep-000007", group_id=GROUP),
                timeout=0.01,
            )
        assert transport.started["find_episodes_by_group"] == 1
        # The shielded scan is still running; give it the loop to finish on.
        for _ in range(200):
            if (GROUP, "ep-000007") in store._episode_group_id_cache:
                break
            await asyncio.sleep(0.005)

        assert store._episode_group_id_cache.get((GROUP, "ep-000007")) == 1007
        assert await store.get_episode_entities("ep-000007", group_id=GROUP) == ["ent_hot"]
        assert transport.calls["find_episodes_by_group"] == 1, transport.calls
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_unscoped_resolution_is_coalesced_too() -> None:
    """`find_episodes_all` is the same defect on the group_id=None path."""
    store, transport = _store(300)
    try:
        ids = [f"ep-{i:06d}" for i in range(30)]
        out = await asyncio.gather(*(store.get_episode_entities(eid) for eid in ids))
    finally:
        transport.close()

    assert transport.calls["find_episodes_all"] == 1, transport.calls
    assert all(row == ["ent_hot"] for row in out)


@pytest.mark.asyncio
async def test_resolution_still_answers_correctly() -> None:
    """Correctness guard: coalescing must not change what resolves to what."""
    store, transport = _store(50)
    try:
        assert await store._resolve_episode_helix_id("ep-000042", GROUP) == 1042
        assert await store._resolve_episode_helix_id("ep-000000", GROUP) == 1000
        assert await store._resolve_episode_helix_id("nope", GROUP) is None
        assert await store._resolve_episode_helix_id_unscoped("ep-000013") == 1013
        assert await store._resolve_episode_helix_id_unscoped("nope") is None
    finally:
        transport.close()


@pytest.mark.asyncio
async def test_failed_scan_raises_and_does_not_latch_the_ttl() -> None:
    """A broken scan must not degrade into "episode not found".

    Swallowing it would turn a transport failure into a silent wrong answer —
    the exact T1 shape of ticket 19 — and latching the TTL on a failure would
    make it stick for 5 s.
    """
    store, transport = _store(50)
    calls = {"n": 0}
    inner = transport.query

    async def flaky(endpoint, payload):
        if endpoint == "find_episodes_by_group":
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transport down")
        return await inner(endpoint, payload)

    transport.query = flaky  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="transport down"):
            await store._resolve_episode_helix_id("ep-000003", GROUP)
        # Not latched: the very next call retries instead of serving a stale miss.
        assert await store._resolve_episode_helix_id("ep-000003", GROUP) == 1003
    finally:
        transport.close()

    assert calls["n"] == 2


# ── The fan-out ─────────────────────────────────────────────────────


def _episode_candidates(n: int) -> list[ScoredResult]:
    return [
        ScoredResult(
            node_id=f"ep-{i:06d}",
            score=1.0 - i * 0.001,
            semantic_similarity=0.5,
            activation=0.0,
            spreading=0.0,
            edge_proximity=0.0,
            result_type="episode",
        )
        for i in range(n)
    ]


def _cfg(**overrides) -> ActivationConfig:
    base = {
        "episode_graph_signal_enabled": True,
        "episode_graph_signal_weight": 0.25,
        "episode_graph_signal_max_candidates": 60,
        "episode_graph_signal_timeout_ms": 0,
    }
    base.update(overrides)
    return ActivationConfig(**base)


def _set_metric(d, key, value):
    if d is not None:
        d[key] = round(float(value), 4)


def _add_timing(d, key, started):
    if d is not None:
        d[key] = round((time.perf_counter() - started) * 1000, 4)


async def _run_stage(store, cfg, candidates, timings):
    await apply_episode_graph_signal(
        candidates,
        [],
        entity_signal={"ent_hot": EntityGraphSignal(0.9, 0.0, None)},
        graph_store=store,
        group_id=GROUP,
        cfg=cfg,
        stage_timings_ms=timings,
        set_metric=_set_metric,
        add_timing=_add_timing,
    )


@pytest.mark.asyncio
async def test_stage_read_fan_out_is_bounded_and_counted() -> None:
    """Peak in-flight reads must be <= 8 while the pool is 60.

    Counted with a real in-flight counter, not inferred from timing: a bound
    inferred from a fast fixture is exactly the plausible-but-wrong number
    INSTRUMENT_AUDIT.md forbids.
    """
    store, transport = _store(200, row_cost_s=0.00002)
    peak = 0
    current = 0
    inner = store.get_episode_entities

    async def counting(episode_id, group_id=None):
        nonlocal peak, current
        current += 1
        peak = max(peak, current)
        try:
            return await inner(episode_id, group_id=group_id)
        finally:
            current -= 1

    store.get_episode_entities = counting  # type: ignore[method-assign]
    timings: dict[str, float] = {}
    candidates = _episode_candidates(60)
    try:
        await _run_stage(store, _cfg(), candidates, timings)
    finally:
        transport.close()

    assert timings["recall_episode_graph_signal_pool"] == 60
    assert peak <= 8, f"fan-out reached {peak} concurrent reads"
    # Not vacuous: the bound must actually have been hit, otherwise a removed
    # semaphore would pass here too.
    assert peak == 8, f"expected the bound to be exercised, saw peak={peak}"
    assert timings["recall_episode_graph_signal_inflight_max"] == 8
    assert timings["recall_episode_graph_signal_covered"] == 60


@pytest.mark.asyncio
async def test_bounded_fan_out_does_not_block_an_independent_stage() -> None:
    """The semaphore's ONLY measured benefit: head-of-line blocking.

    Bounding the fan-out does not reduce worker burn — measured, four arms:
    submissions fell 60 -> 8 while jobs actually run stayed 4 and worker time
    stayed 3043 ms vs 3042 ms, because `run_in_executor` cancels its own
    still-pending future anyway. What it does fix is an independent stage
    queueing behind the surplus: isolated, warm path, one foreign read
    submitted 5 ms in, median of 5 — 40.5 ms unbounded vs 5.2 ms bounded.

    Asserted here as a COUNT, not a duration: how many episode reads were still
    ahead of a foreign job in the executor queue.
    """
    store, transport = _store(200)
    await store._resolve_episode_helix_id("ep-000000", GROUP)  # warm path

    queue_depth_seen = 0
    inner = transport.query

    async def watching(endpoint, payload):
        nonlocal queue_depth_seen
        if endpoint == "get_episode_entities":
            queue_depth_seen = max(
                queue_depth_seen,
                transport.calls.get("get_episode_entities", 0)
                - transport.completed.get("get_episode_entities", 0),
            )
        return await inner(endpoint, payload)

    transport.query = watching  # type: ignore[method-assign]
    try:
        await _run_stage(store, _cfg(), _episode_candidates(60), {})
    finally:
        transport.close()

    # A foreign job submitted mid-stage sits behind at most this many episode
    # reads. Unbounded that is 60; the semaphore is what makes it small.
    assert queue_depth_seen <= _EPISODE_GRAPH_READ_CONCURRENCY, queue_depth_seen
    assert queue_depth_seen > NATIVE_WORKERS, (
        f"queue depth {queue_depth_seen} never exceeded the executor width, so "
        "this probe cannot distinguish a bounded fan-out from an unbounded one"
    )


@pytest.mark.asyncio
async def test_timed_out_stage_retains_its_scan_and_converges() -> None:
    """The T1 itself: a timed-out stage used to discard every scan it paid for.

    `wait_for` cancels the awaiting coroutines, so the pre-fix caching loop
    never ran while the executor jobs completed anyway. Measured pre-fix at
    2000 episodes: 60 scans submitted, 4 run, 3043 ms of worker time burned,
    **0 rows cached** — so the next recall re-paid the whole thing, forever.
    """
    store, transport = _store(400, row_cost_s=0.0005)  # ~200 ms per scan
    timings: dict[str, float] = {}
    candidates = _episode_candidates(60)
    try:
        await _run_stage(store, _cfg(episode_graph_signal_timeout_ms=40), candidates, timings)
        assert "recall_episode_graph_signal_timeout" in timings
        assert timings["recall_episode_graph_signal_inflight_max"] <= 8

        scans_committed = transport.started.get("find_episodes_by_group", 0)
        assert scans_committed == 1, f"{scans_committed} scans reached the executor"

        # Convergence: the shielded scan finishes, so the SECOND attempt is warm
        # and the stage completes. Pre-fix it re-paid the cliff on every recall.
        for _ in range(400):
            if (GROUP, "ep-000000") in store._episode_group_id_cache:
                break
            await asyncio.sleep(0.005)
        timings2: dict[str, float] = {}
        candidates2 = _episode_candidates(60)
        await _run_stage(store, _cfg(episode_graph_signal_timeout_ms=40), candidates2, timings2)
    finally:
        transport.close()

    assert "recall_episode_graph_signal_timeout" not in timings2, timings2
    assert timings2["recall_episode_graph_signal_covered"] == 60
    assert transport.calls["find_episodes_by_group"] == 1
