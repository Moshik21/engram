"""Persisted BM25 breaker state: fresh shells pre-arm instead of serving
2-4 minutes of degraded recall while the breaker re-collects its 2 strikes
and zombie native BM25 calls drain.

On OPEN (and failed half-open probes) the breaker writes a sidecar JSON in
~/.engram (ENGRAM_HOME honored — beside the activation snapshot, never inside
the graph data dir). A breaker constructed with persist=True that finds a
persisted open younger than 24h STARTS open, so the first query already skips
BM25; the half-open probe path still closes it if the brain got faster.
Missing/corrupt/stale sidecar = normal closed start. Lite and HTTP transports
never construct a breaker, so they never read or write the sidecar.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.storage.helix.search import (
    _BM25_BREAKER_STATE_MAX_AGE_SECONDS,
    Bm25CircuitBreaker,
    HelixSearchIndex,
    _bm25_breaker_state_path,
)

KEY = "/fake/native/data-dir"


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


class _FixedProvider(EmbeddingProvider):
    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self._dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [1.0] * self._dim

    def dimension(self) -> int:
        return self._dim


def _breaker(*, persist: bool = True, clock=None, wall_clock=None) -> Bm25CircuitBreaker:
    kwargs = {
        "budget_ms": 10.0,
        "cancel_strike_ms": 10.0,
        "open_after": 2,
        "retry_after_seconds": 300.0,
        "persist": persist,
    }
    if clock is not None:
        kwargs["clock"] = clock
    if wall_clock is not None:
        kwargs["wall_clock"] = wall_clock
    return Bm25CircuitBreaker(KEY, **kwargs)


def _open(breaker: Bm25CircuitBreaker) -> None:
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    assert breaker.is_open


def test_sidecar_written_on_open(engram_home: Path):
    _open(_breaker())
    path = _bm25_breaker_state_path()
    assert path == engram_home / "bm25-breaker-state.json"
    entry = json.loads(path.read_text())[KEY]
    assert isinstance(entry["opened_at_wall"], float)


def test_fresh_shell_pre_arms_from_persisted_open(
    engram_home: Path, caplog: pytest.LogCaptureFixture
):
    _open(_breaker())
    with caplog.at_level(logging.WARNING, logger="engram.storage.helix.search"):
        fresh = _breaker()  # new process: fresh object, same key
    assert fresh.is_open
    assert fresh.snapshot()["preArmed"] is True
    assert not fresh.allow_call()  # the FIRST query already skips BM25
    assert any("pre-armed from persisted state" in r.message for r in caplog.records)


def test_missing_sidecar_starts_closed(engram_home: Path):
    assert not _breaker().is_open


def test_corrupt_sidecar_starts_closed_and_recovers(engram_home: Path):
    path = _bm25_breaker_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json!!", encoding="utf-8")
    breaker = _breaker()
    assert not breaker.is_open
    # The breaker still functions: an open rewrites a valid sidecar.
    _open(breaker)
    assert "opened_at_wall" in json.loads(path.read_text())[KEY]


def test_stale_persisted_open_is_ignored(engram_home: Path):
    path = _bm25_breaker_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({KEY: {"opened_at_wall": 1000.0}}))
    breaker = _breaker(
        wall_clock=lambda: 1000.0 + _BM25_BREAKER_STATE_MAX_AGE_SECONDS + 1.0,
    )
    assert not breaker.is_open


def test_half_open_probe_closes_pre_armed_breaker_and_clears_sidecar(engram_home: Path):
    _open(_breaker())
    now = {"t": 0.0}
    fresh = _breaker(clock=lambda: now["t"])
    assert fresh.is_open
    now["t"] = 299.0
    assert not fresh.allow_call()
    now["t"] = 301.0
    assert fresh.allow_call()  # half-open probe still allowed when pre-armed
    fresh.record_call(2.0)  # brain got faster — healthy probe closes it
    assert not fresh.is_open
    assert fresh.snapshot()["preArmed"] is False
    assert KEY not in json.loads(_bm25_breaker_state_path().read_text())
    # The next shell starts closed again.
    assert not _breaker().is_open


def test_failed_probe_on_pre_armed_breaker_re_persists(engram_home: Path):
    _open(_breaker(wall_clock=lambda: 1000.0))
    now = {"t": 0.0}
    fresh = _breaker(clock=lambda: now["t"], wall_clock=lambda: 2000.0)
    now["t"] = 301.0
    assert fresh.allow_call()
    fresh.record_call(50.0)  # probe still over budget — re-open + re-persist
    assert fresh.is_open
    entry = json.loads(_bm25_breaker_state_path().read_text())[KEY]
    assert entry["opened_at_wall"] == 2000.0


def test_default_unit_breaker_never_touches_sidecar(engram_home: Path):
    breaker = Bm25CircuitBreaker("unit-key", budget_ms=10.0, open_after=2)
    breaker.record_call(50.0)
    breaker.record_call(50.0)
    assert breaker.is_open
    assert not _bm25_breaker_state_path().exists()


def test_native_index_construct_pre_arms_for_its_data_dir(engram_home: Path, tmp_path: Path):
    data_dir = str(tmp_path / "native-brain-prearm")
    path = _bm25_breaker_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({data_dir: {"opened_at_wall": time.time() - 60.0}}))

    index = HelixSearchIndex(
        helix_config=HelixDBConfig(transport="native", data_dir=data_dir),
        provider=_FixedProvider(),
        embed_config=EmbeddingConfig(),
        client=object(),
        owns_client=False,
    )

    assert index._bm25_breaker is not None
    assert index._bm25_breaker.is_open
    assert index.bm25_fallback_degraded() is True


def test_http_transport_never_reads_persisted_state(engram_home: Path):
    path = _bm25_breaker_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"http-key": {"opened_at_wall": time.time()}})
    path.write_text(payload)

    index = HelixSearchIndex(
        helix_config=HelixDBConfig(transport="http"),
        provider=_FixedProvider(),
        embed_config=EmbeddingConfig(),
        client=object(),
        owns_client=False,
    )

    assert index._bm25_breaker is None
    assert path.read_text() == payload  # sidecar byte-untouched


def test_lite_sqlite_has_no_breaker_and_never_touches_sidecar(engram_home: Path):
    from engram.storage.sqlite.hybrid_search import HybridSearchIndex

    assert not hasattr(HybridSearchIndex, "_bm25_breaker")
    assert not _bm25_breaker_state_path().exists()
