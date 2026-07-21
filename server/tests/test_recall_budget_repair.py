"""Recall budget repair: raised ceilings + native BM25 circuit breaker.

Covers the deep-recall autopsy production fixes:

1. Config ceilings raised so the stats probe (measured 240-720ms) and
   materialize graph reads (measured ~74ms/get_entity) fit their budgets on
   large native brains (small brains finish early — no cost).
2. Native BM25 circuit breaker: cancelled/capped native BM25 calls cannot be
   cancelled server-side — they keep running on the native thread pool and
   starve the 2-30ms vector lanes. After 2 consecutive over-budget BM25 calls
   the breaker opens and BM25 lanes are SKIPPED; vector+graph lanes carry
   recall. Half-open retry after 300s.
3. Fast-lane inversion: degraded fast fallback lanes prefer vector-first
   instead of BM25-only (the slowest path on native).

The breaker is scoped to the native Helix transport — lite/SQLite FTS5 and
HTTP transports must never engage it.
"""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig, EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.storage.helix.search import (
    _BM25_BREAKER_OPEN_AFTER,
    _BM25_BREAKER_RETRY_AFTER_SECONDS,
    Bm25CircuitBreaker,
    HelixSearchIndex,
    get_bm25_breaker_stats,
)

GROUP = "g1"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FixedProvider(EmbeddingProvider):
    """Deterministic embedding provider (no network)."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self._dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0] * self._dim

    def dimension(self) -> int:
        return self._dim


class _FakeHelixClient:
    """Fake shared HelixClient: slow BM25 endpoints, instant vector endpoints."""

    def __init__(self, *, bm25_delay: float = 0.0) -> None:
        self.bm25_delay = bm25_delay
        self.calls: list[str] = []

    def bm25_calls(self) -> int:
        return sum(1 for endpoint in self.calls if endpoint.endswith("_bm25"))

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        self.calls.append(endpoint)
        if endpoint.endswith("_bm25"):
            if self.bm25_delay:
                await asyncio.sleep(self.bm25_delay)
            return [{"episode_id": "ep_bm25", "group_id": GROUP, "score": 1.0}]
        if "vectors_filtered" in endpoint:
            return [{"episode_id": "ep_vec", "group_id": GROUP, "distance": 0.3}]
        return []


def _make_native_index(
    tmp_path,
    *,
    bm25_delay: float = 0.0,
    bm25_breaker_enabled: bool = True,
) -> tuple[HelixSearchIndex, _FakeHelixClient]:
    client = _FakeHelixClient(bm25_delay=bm25_delay)
    index = HelixSearchIndex(
        helix_config=HelixDBConfig(
            transport="native",
            data_dir=str(tmp_path / "recall-budget-repair"),
        ),
        provider=_FixedProvider(),
        embed_config=EmbeddingConfig(),
        embed_provider="test",
        embed_model="fixed",
        client=client,
        owns_client=False,
        bm25_breaker_enabled=bm25_breaker_enabled,
    )
    return index, client


def _fast_breaker(clock=None, **overrides) -> Bm25CircuitBreaker:
    kwargs = {
        "budget_ms": 10.0,
        "cancel_strike_ms": 10.0,
        "open_after": 2,
        "retry_after_seconds": 300.0,
    }
    kwargs.update(overrides)
    if clock is not None:
        kwargs["clock"] = clock
    return Bm25CircuitBreaker("test-breaker", **kwargs)


# ---------------------------------------------------------------------------
# 1. Config defaults pinned
# ---------------------------------------------------------------------------


def test_config_defaults_pinned():
    cfg = ActivationConfig()
    # Stats probe measured 240-720ms on an 8.8k-episode native brain.
    assert cfg.retrieval_stats_timeout_ms == 1500
    # get_entity measured ~74ms/call — 50ms/15ms caps failed every read.
    assert cfg.recall_primary_materialize_graph_timeout_ms == 300
    assert cfg.recall_primary_materialize_graph_timeout_after_probe_timeout_ms == 300
    # Breaker ships enabled (env kill switch).
    assert cfg.retrieval_bm25_breaker_enabled is True
    # Breaker shape pinned: N=2 consecutive strikes, 300s half-open retry.
    assert _BM25_BREAKER_OPEN_AFTER == 2
    assert _BM25_BREAKER_RETRY_AFTER_SECONDS == 300.0


# ---------------------------------------------------------------------------
# 2. Breaker unit behavior
# ---------------------------------------------------------------------------


def test_breaker_opens_after_two_consecutive_over_budget_calls():
    breaker = _fast_breaker()
    breaker.record_call(50.0)
    assert not breaker.is_open
    assert breaker.degraded  # one strike: degrade fast lanes already
    breaker.record_call(50.0)
    assert breaker.is_open
    snap = breaker.snapshot()
    assert snap["opens"] == 1
    assert snap["overBudgetCalls"] == 2


def test_breaker_healthy_call_resets_consecutive_strikes():
    breaker = _fast_breaker()
    breaker.record_call(50.0)
    breaker.record_call(2.0)  # healthy — resets
    assert not breaker.degraded
    breaker.record_call(50.0)
    assert not breaker.is_open  # still only 1 consecutive


def test_breaker_open_skips_calls_and_counts_them():
    breaker = _fast_breaker()
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    assert not breaker.allow_call()
    assert not breaker.allow_call()
    assert breaker.snapshot()["skippedCalls"] == 2


def test_breaker_half_open_probe_and_close():
    now = {"t": 0.0}
    breaker = _fast_breaker(clock=lambda: now["t"])
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    now["t"] = 299.0
    assert not breaker.allow_call()
    now["t"] = 301.0
    assert breaker.allow_call()  # single half-open probe
    assert not breaker.allow_call()  # probe in flight — others skipped
    breaker.record_call(2.0)  # probe healthy — closes
    assert not breaker.is_open
    assert breaker.snapshot()["closes"] == 1
    assert breaker.allow_call()


def test_breaker_failed_probe_reopens():
    now = {"t": 0.0}
    breaker = _fast_breaker(clock=lambda: now["t"])
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    now["t"] = 301.0
    assert breaker.allow_call()
    breaker.record_call(50.0)  # probe over budget — re-open window restarts
    assert breaker.is_open
    now["t"] = 301.0 + 299.0
    assert not breaker.allow_call()
    now["t"] = 301.0 + 301.0
    assert breaker.allow_call()


def test_breaker_cancelled_call_wall_time_counts_as_strike():
    breaker = _fast_breaker()
    # Cancelled after exceeding the cancel-strike threshold: the native call is
    # a zombie on the native pool — must count even though it never finished.
    breaker.record_call(1500.0, cancelled=True)
    breaker.record_call(1500.0, cancelled=True)
    assert breaker.is_open


def test_breaker_young_cancellation_is_inconclusive():
    breaker = _fast_breaker()
    breaker.record_call(50.0)  # strike 1
    breaker.record_call(1.0, cancelled=True)  # cancelled young: no info
    assert breaker.degraded  # strike NOT reset
    assert not breaker.is_open  # and no second strike either


# ---------------------------------------------------------------------------
# 3. HelixSearchIndex integration (fake slow-BM25 native backend)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_slow_bm25_opens_breaker_and_vector_lane_carries(tmp_path):
    index, client = _make_native_index(tmp_path, bm25_delay=0.05)
    index._bm25_breaker = _fast_breaker()

    r1 = await index.search_episodes("golden decision strategy", group_id=GROUP)
    assert not index._bm25_breaker.is_open
    r2 = await index.search_episodes("golden decision strategy", group_id=GROUP)
    assert index._bm25_breaker.is_open
    assert client.bm25_calls() == 2
    # Hybrid results still flowed while the breaker was accumulating strikes.
    assert r1 and r2

    # Breaker open: BM25 endpoint is NOT launched again; vector carries.
    r3 = await index.search_episodes("golden decision strategy", group_id=GROUP)
    assert client.bm25_calls() == 2
    assert [eid for eid, _ in r3] == ["ep_vec"]
    assert index._bm25_breaker.snapshot()["skippedCalls"] >= 1


@pytest.mark.asyncio
async def test_caller_timeout_cancellation_still_strikes_breaker(tmp_path):
    index, client = _make_native_index(tmp_path, bm25_delay=5.0)
    index._bm25_breaker = _fast_breaker()

    for _ in range(2):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                index.search_episodes("golden decision strategy", group_id=GROUP),
                timeout=0.05,
            )
    # The abandoned calls' wall time was observed at cancellation.
    assert index._bm25_breaker.is_open
    assert client.bm25_calls() == 2


@pytest.mark.asyncio
async def test_fast_lane_inversion_prefers_vector_when_degraded(tmp_path):
    index, client = _make_native_index(tmp_path)
    index._bm25_breaker = _fast_breaker()

    # Not degraded: fast lanes stay BM25-only (byte-identical behavior).
    hits = await index.search_episodes_fast("golden decision strategy", group_id=GROUP)
    assert client.bm25_calls() == 1
    assert [eid for eid, _ in hits] == ["ep_bm25"]

    # One over-budget BM25 call: degrade — fast lanes go vector-first.
    index._bm25_breaker.record_call(1500.0, cancelled=True)
    hits = await index.search_episodes_fast("golden decision strategy", group_id=GROUP)
    assert client.bm25_calls() == 1  # no new BM25 launch
    assert [eid for eid, _ in hits] == ["ep_vec"]

    cue_hits = await index.search_episode_cues_fast("golden decision strategy", group_id=GROUP)
    assert client.bm25_calls() == 1
    assert [eid for eid, _ in cue_hits] == ["ep_vec"]


@pytest.mark.asyncio
async def test_records_fast_returns_empty_fast_when_open(tmp_path):
    index, client = _make_native_index(tmp_path, bm25_delay=5.0)
    breaker = _fast_breaker()
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    index._bm25_breaker = breaker

    records = await asyncio.wait_for(
        index.search_episode_records_fast("golden decision strategy", group_id=GROUP),
        timeout=0.5,
    )
    assert records == []
    assert client.bm25_calls() == 0
    assert index.bm25_fallback_degraded() is True


@pytest.mark.asyncio
async def test_breaker_diagnostics_exposed(tmp_path):
    index, _client = _make_native_index(tmp_path)
    breaker = index._bm25_breaker
    assert breaker is not None
    breaker.record_call(5000.0)
    breaker.record_call(5000.0)

    stats = get_bm25_breaker_stats()
    key = str(tmp_path / "recall-budget-repair")
    assert key in stats
    assert stats[key]["open"] is True
    assert stats[key]["overBudgetCalls"] == 2

    # Mirrors the queryFailures pattern on the storage diagnostics dict.
    from engram.storage.diagnostics import _bm25_breaker_stats

    assert _bm25_breaker_stats()[key]["open"] is True


# ---------------------------------------------------------------------------
# 4. Non-native transports and lite are untouched
# ---------------------------------------------------------------------------


def test_breaker_never_engages_on_http_transport(tmp_path):
    index = HelixSearchIndex(
        helix_config=HelixDBConfig(transport="http"),
        provider=_FixedProvider(),
        embed_config=EmbeddingConfig(),
        client=_FakeHelixClient(),
        owns_client=False,
    )
    assert index._bm25_breaker is None
    assert index.bm25_fallback_degraded() is False


def test_breaker_kill_switch_disables_on_native(tmp_path):
    index, _client = _make_native_index(tmp_path, bm25_breaker_enabled=False)
    assert index._bm25_breaker is None
    assert index.bm25_fallback_degraded() is False


def test_lite_sqlite_index_has_no_breaker_surface():
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex

    assert not hasattr(HybridSearchIndex, "bm25_fallback_degraded")
    assert not hasattr(HybridSearchIndex, "_guarded_bm25_query")


@pytest.mark.asyncio
async def test_graph_candidate_search_joins_open_breaker(tmp_path):
    """The graph store's entity candidate search (projection/capture lanes)
    runs on the same native pool as recall — with the breaker OPEN it must
    skip its BM25 phase instead of feeding the zombie pool; with no breaker
    constructed (peek returns None) it behaves as before."""
    from engram.storage.helix.graph import HelixGraphStore
    from engram.storage.helix.search import _get_bm25_breaker

    data_dir = str(tmp_path / "native-brain")
    store = HelixGraphStore(
        HelixDBConfig(transport="native", data_dir=data_dir),
    )
    called: list[str] = []

    async def fake_query(endpoint, payload, *a, **k):
        called.append(endpoint)
        return []

    store._query = fake_query

    # No breaker constructed yet: BM25 phase runs.
    await store.find_entity_candidates("Melanie", "g1", limit=3)
    assert "search_entities_bm25_filtered" in called

    # Open the shared breaker for this data_dir: BM25 phase must be skipped.
    breaker = _get_bm25_breaker(data_dir)
    breaker.record_call(9_000.0)
    breaker.record_call(9_000.0)
    assert breaker.is_open
    called.clear()
    await store.find_entity_candidates("Melanie", "g1", limit=3)
    assert "search_entities_bm25_filtered" not in called
    assert "search_entities_bm25" not in called
    # Non-BM25 candidate phases still ran.
    assert called
