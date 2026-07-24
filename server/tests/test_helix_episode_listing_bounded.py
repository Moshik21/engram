"""GET /api/episodes must not materialise the whole store to serve one page.

Paginating `GET /api/episodes?limit=200` across a ~9.3k-episode native brain
took the shell down twice (RemoteDisconnected mid-scan, LaunchAgent restart).
The engine has no ordered/bounded episode page route, so the raw group rows
still arrive in full — but the listing only keeps `content[:200]`, so
hydrating every row into a decrypted, pydantic-validated Episode model was
pure waste that scaled with store size, not page size.

These tests fail loudly if that hydration ever becomes unbounded again.
"""

from __future__ import annotations

import tracemalloc
from datetime import datetime, timedelta, timezone

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.graph import HelixGraphStore

STORE_ROWS = 5000
PAGE_LIMIT = 200
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Long enough that N-vs-page hydration is unmistakable in tracemalloc.
CONTENT = "alpha beta gamma delta epsilon zeta eta theta " * 100


def _rows(count: int = STORE_ROWS, *, group_id: str = "brain") -> list[dict]:
    return [
        {
            "id": 1000 + index,
            "episode_id": f"ep_{index:06d}",
            "group_id": group_id,
            "content": f"episode {index} {CONTENT}",
            "source": "chat",
            "status": "completed",
            "projection_state": "projected",
            "created_at": (BASE + timedelta(seconds=index)).isoformat(),
            "updated_at": (BASE + timedelta(seconds=index)).isoformat(),
            "attachments_json": "[]",
        }
        for index in range(count)
    ]


def _store_with_rows(monkeypatch, rows: list[dict]) -> HelixGraphStore:
    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        return rows

    monkeypatch.setattr(store, "_query", fake_query)
    return store


def _count_hydrations(monkeypatch) -> list[str]:
    """Record every raw row that gets turned into an Episode model."""
    hydrated: list[str] = []
    original = HelixGraphStore._dict_to_episode

    def counting(self, d: dict, group_id: str | None = None):
        hydrated.append(d.get("episode_id", ""))
        return original(self, d, group_id)

    monkeypatch.setattr(HelixGraphStore, "_dict_to_episode", counting)
    return hydrated


@pytest.mark.asyncio
async def test_episode_page_hydrates_only_the_page_not_the_store(monkeypatch) -> None:
    rows = _rows()
    store = _store_with_rows(monkeypatch, rows)
    hydrated = _count_hydrations(monkeypatch)

    episodes, next_cursor = await store.get_episodes_paginated(
        group_id="brain",
        limit=PAGE_LIMIT,
    )

    assert len(episodes) == PAGE_LIMIT
    assert next_cursor is not None
    # The load-bearing assertion: hydration is bounded by the page, not by
    # the store. Old code path hydrated all STORE_ROWS.
    assert len(hydrated) == PAGE_LIMIT, (
        f"listing hydrated {len(hydrated)} Episode models for a {PAGE_LIMIT}-row page "
        f"out of a {len(rows)}-row store — full-store materialisation is back"
    )


@pytest.mark.asyncio
async def test_episode_page_peak_memory_does_not_scale_with_store_size(monkeypatch) -> None:
    """Peak allocation of one page must not grow with the number of stored rows."""
    small_rows = _rows(500)
    large_rows = _rows(STORE_ROWS)

    async def _peak_for(rows: list[dict]) -> int:
        store = _store_with_rows(monkeypatch, rows)
        tracemalloc.start()
        episodes, _cursor = await store.get_episodes_paginated(
            group_id="brain",
            limit=PAGE_LIMIT,
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(episodes) == PAGE_LIMIT
        return peak

    small_peak = await _peak_for(small_rows)
    large_peak = await _peak_for(large_rows)

    # The raw rows are supplied by the caller here, so anything the page
    # itself allocates should be flat. A 10x bigger store must not cost 2x
    # the page-serving allocation.
    assert large_peak < small_peak * 2, (
        f"peak allocation scaled with store size: {small_peak / 1e6:.1f}MB for 500 rows "
        f"vs {large_peak / 1e6:.1f}MB for {STORE_ROWS} rows"
    )


@pytest.mark.asyncio
async def test_episode_page_ordering_cursor_and_filters_are_unchanged(monkeypatch) -> None:
    """Bounded hydration must not change what the endpoint returns."""
    rows = _rows(600)
    rows[7]["source"] = "import"
    rows[9]["status"] = "failed"
    store = _store_with_rows(monkeypatch, rows)

    first, cursor = await store.get_episodes_paginated(group_id="brain", limit=10)
    assert [e.id for e in first] == [f"ep_{i:06d}" for i in range(599, 589, -1)]
    assert cursor == first[-1].created_at.isoformat()

    second, _next_cursor = await store.get_episodes_paginated(
        group_id="brain",
        cursor=cursor,
        limit=10,
    )
    assert [e.id for e in second] == [f"ep_{i:06d}" for i in range(589, 579, -1)]

    # Python-side safety-net filters still apply to the raw rows.
    by_source, _ = await store.get_episodes_paginated(
        group_id="brain",
        source="import",
        limit=10,
    )
    assert [e.id for e in by_source] == ["ep_000007"]

    by_status, _ = await store.get_episodes_paginated(
        group_id="brain",
        status="failed",
        limit=10,
    )
    assert [e.id for e in by_status] == ["ep_000009"]


@pytest.mark.asyncio
async def test_episode_page_still_warms_the_helix_id_cache_for_scanned_rows(monkeypatch) -> None:
    """Cold-cache id resolution relied on the listing warming every scanned row."""
    rows = _rows(500)
    store = _store_with_rows(monkeypatch, rows)

    await store.get_episodes_paginated(group_id="brain", limit=10)

    assert store._episode_group_id_cache[("brain", "ep_000000")] == 1000
    assert store._episode_group_id_cache[("brain", "ep_000499")] == 1499
