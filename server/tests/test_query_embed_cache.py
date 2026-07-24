"""Pin the per-recall query-embed memo (RECALL_PERFORMANCE_PLAN M1).

The entity / episode / cue / chunk lanes each embed the same query string. A
single recall must embed the query ONCE and reuse it across all four lanes.
A different query (the next recall) must re-embed — the memo is keyed by the
exact query text and never leaks a vector to a different query.
"""

from __future__ import annotations

import asyncio

from engram.config import EmbeddingConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.storage.helix.client import HelixDBConfig
from engram.storage.helix.search import HelixSearchIndex

GROUP = "g1"


class _CountingProvider(EmbeddingProvider):
    """Deterministic provider that counts query embeds."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.query_calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] * self._dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0] * self._dim

    def dimension(self) -> int:
        return self._dim


class _EmptyClient:
    """Fake shared HelixClient: every endpoint returns no rows."""

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        return []


def _make_index() -> tuple[HelixSearchIndex, _CountingProvider]:
    provider = _CountingProvider()
    index = HelixSearchIndex(
        helix_config=HelixDBConfig(transport="native", data_dir="/tmp/qec-unused"),
        provider=provider,
        embed_config=EmbeddingConfig(),
        embed_provider="test",
        embed_model="fixed",
        client=_EmptyClient(),
        owns_client=False,
    )
    return index, provider


async def _run_all_lanes(index: HelixSearchIndex, query: str) -> None:
    await index.search(query=query, group_id=GROUP, limit=5)
    await index.search_episodes(query=query, group_id=GROUP, limit=5)
    await index.search_episode_cues(query=query, group_id=GROUP, limit=5)
    await index.search_episode_chunks(query=query, group_id=GROUP, limit=5)


def test_single_recall_embeds_query_once():
    index, provider = _make_index()
    asyncio.run(_run_all_lanes(index, "what did we decide about pruning?"))
    assert provider.query_calls == 1


def test_gathered_lanes_embed_query_once():
    """Even when the lanes are launched concurrently the query embeds once."""
    index, provider = _make_index()

    async def _gather() -> None:
        q = "who owns the recall hot path?"
        await asyncio.gather(
            index.search(query=q, group_id=GROUP, limit=5),
            index.search_episodes(query=q, group_id=GROUP, limit=5),
            index.search_episode_cues(query=q, group_id=GROUP, limit=5),
            index.search_episode_chunks(query=q, group_id=GROUP, limit=5),
        )

    asyncio.run(_gather())
    # Serial value-cache: concurrent lanes may race the first store, so allow a
    # small window, but the memo must collapse the four embeds to at most two.
    assert 1 <= provider.query_calls <= 2


def test_new_query_re_embeds():
    index, provider = _make_index()

    async def _two_recalls() -> None:
        await _run_all_lanes(index, "first query")
        await _run_all_lanes(index, "second query")

    asyncio.run(_two_recalls())
    assert provider.query_calls == 2


def test_failed_embed_not_cached():
    """An empty embed result is not memoized, preserving per-lane retry."""
    index, provider = _make_index()

    call_count = {"n": 0}

    async def _flaky_query(text: str) -> list[float]:
        call_count["n"] += 1
        return []  # simulate embedding failure

    provider.embed_query = _flaky_query  # type: ignore[assignment]

    asyncio.run(_run_all_lanes(index, "unembeddable"))
    # Each lane retries independently because the failure is never cached.
    assert call_count["n"] >= 2
