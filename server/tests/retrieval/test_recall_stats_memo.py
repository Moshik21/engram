"""The recall stats probe must not cost every recall its wall.

Live 2026-09-02: the entity-count probe (used only to scale pool sizes by
sqrt(n/1000)) ran serially before the primary search on every recall and hit
its 1500 ms cap on 8/10 recalls under an indexing drain. The primary search
was then cancelled at 1.6 s. The count is memoised on the store; a timeout
backs off so a contended store is not re-probed on every recall.
"""

from __future__ import annotations

import asyncio

import pytest

from engram.config import ActivationConfig
from engram.retrieval.pipeline import retrieve

pytestmark = pytest.mark.asyncio


class _Index:
    async def search(self, query, group_id, limit=50):
        return []

    async def search_episodes(self, query, group_id, limit=10):
        return []

    async def compute_similarity(self, query, entity_ids, group_id=None):
        return {}


class _Activation:
    async def batch_get(self, entity_ids):
        return {}

    async def get_top_activated(self, **kwargs):
        return []


class _Store:
    def __init__(self, *, hang: bool):
        self.hang = hang
        self.stats_calls = 0

    async def get_stats(self, group_id, exact=True):
        self.stats_calls += 1
        if self.hang:
            await asyncio.sleep(60)
        return {"entity_count": 4000}

    async def get_entity(self, *a, **k):
        return None

    async def find_entities(self, **k):
        return []

    async def find_entity_candidates(self, *a, **k):
        return []

    async def get_relationships(self, *a, **k):
        return []

    async def get_active_neighbors_with_weights(self, *a, **k):
        return []

    async def get_episode_by_id(self, *a, **k):
        return None


def _cfg() -> ActivationConfig:
    return ActivationConfig(
        retrieval_stats_timeout_ms=50,
        graph_query_expansion_enabled=False,
        reranker_enabled=False,
    )


async def _recall(store) -> dict[str, float]:
    timings: dict[str, float] = {}
    await retrieve(
        query="anything",
        group_id="default",
        graph_store=store,
        activation_store=_Activation(),
        search_index=_Index(),
        cfg=_cfg(),
        limit=5,
        stage_timings_ms=timings,
    )
    return timings


async def test_second_recall_uses_the_memo_and_does_not_probe_again() -> None:
    store = _Store(hang=False)
    first = await _recall(store)
    second = await _recall(store)
    assert "recall_stats" in first and store.stats_calls == 1
    assert "recall_stats_cached" in second and "recall_stats" not in second
    assert store.stats_calls == 1


async def test_timeout_backs_off_instead_of_paying_the_cap_every_recall() -> None:
    store = _Store(hang=True)
    first = await _recall(store)
    second = await _recall(store)
    assert "recall_stats_timeout" in first and store.stats_calls == 1
    assert "recall_stats_backoff" in second and "recall_stats_timeout" not in second
    assert store.stats_calls == 1, "a contended store was re-probed inside the backoff"
    assert second["recall_stats_backoff"] < 5.0
