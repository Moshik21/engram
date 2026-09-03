"""The BM25 half-open probe must close the breaker even when the caller times out.

Live 2026-09-03: the breaker was persisted OPEN for a day; each half-open probe
was cancelled by the lane timeout before it could report, so the keyword lane
never came back although native BM25 answered in 18-700 ms.
"""

from __future__ import annotations

import asyncio

import pytest

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.storage.helix.search import HelixSearchIndex

pytestmark = pytest.mark.asyncio


class _Provider:
    async def embed(self, texts):
        return [[1.0] * 4 for _ in texts]

    async def embed_query(self, text):
        return [1.0] * 4

    def dimension(self) -> int:
        return 4


class _SlowClient:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def query(self, endpoint, payload):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return [{"episode_id": "ep_1", "group_id": "default", "score": 1.0}]


def _index(tmp_path, delay: float) -> HelixSearchIndex:
    client = _SlowClient(delay)
    return HelixSearchIndex(
        helix_config=HelixDBConfig(transport="native", data_dir=str(tmp_path / "d")),
        provider=_Provider(),
        embed_config=EmbeddingConfig(),
        embed_provider="test",
        embed_model="fixed",
        client=client,
        owns_client=False,
        bm25_breaker_enabled=True,
    )


async def test_probe_survives_caller_timeout_and_closes_the_breaker(tmp_path) -> None:
    index = _index(tmp_path, delay=0.15)
    breaker = index._bm25_breaker
    assert breaker is not None
    breaker._opened_at = breaker._clock() - 10_000  # long past the retry window
    assert breaker.is_open
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            index._guarded_bm25_query("search_episodes_bm25", {"query": "x", "k": 5}),
            timeout=0.03,
        )
    assert breaker.is_open, "the probe has not reported yet"
    await asyncio.sleep(0.25)
    assert not breaker.is_open, "a fast probe must close the breaker despite the caller's timeout"


async def test_slow_probe_reopens_instead_of_closing(tmp_path) -> None:
    from engram.storage.helix import search as search_mod

    index = _index(tmp_path, delay=0.05)
    breaker = index._bm25_breaker
    breaker._budget_ms = 10.0  # make 50 ms count as over budget
    breaker._opened_at = breaker._clock() - 10_000
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            index._guarded_bm25_query("search_episodes_bm25", {"query": "x", "k": 5}),
            timeout=0.01,
        )
    await asyncio.sleep(0.1)
    assert breaker.is_open, "an over-budget probe must keep the lane off"
    assert search_mod._BM25_BREAKER_RETRY_AFTER_SECONDS <= 60.0
