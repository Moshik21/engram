"""GET /api/episodes must not materialise the whole store to serve one page.

Ticket 22. Paginating `GET /api/episodes?limit=200` across a ~9.4k-episode
native brain took the shell down twice, and `limit` is validated `ge=1, le=200`
so 200 is an ACCEPTED value any caller can use. A first fix bounded HYDRATION to
the page (406 ms/89.2 MB -> 207 ms/61.1 MB against a 167 ms/60.5 MB transport
floor) and disclosed honestly that the cause was untouched: the store still
asked the engine for EVERY row in the group, content included, on every request.

The fix under test is the bounded page route in `schema.hx`
(`find_episodes_by_group_page` and its filter variants): ORDER<Desc> + RANGE run
inside the engine, so at most `limit + 1` rows ever cross the transport.

The load-bearing assertion in this file is therefore about the TRANSPORT, not
about hydration: `test_one_page_does_not_pull_the_whole_group_across_the_transport`
counts the rows the store asked for. It is red against the old path — which is
still reachable, and exercised here, on engines whose binary predates the route.
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
# Long enough that store-sized vs page-sized transfer is unmistakable.
CONTENT = "alpha beta gamma delta epsilon zeta eta theta " * 100

# endpoint -> the filter params that route's HelixQL signature actually declares.
# Modelled per-route rather than "apply whatever the payload carries", because a
# permissive fake lets a store that sends `src` to the unfiltered route look
# correct — it did, until neuter C caught it.
PAGE_ROUTES = {
    "find_episodes_by_group_page": (),
    "find_episodes_by_source_page": ("src",),
    "find_episodes_by_status_page": ("st",),
    "find_episodes_by_source_status_page": ("src", "st"),
}
_PAGE_FILTER_FIELD = {"src": "source", "st": "status"}


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


class _FakeEngine:
    """Stands in for the PyO3 engine's route table."""

    def __init__(self, routes: set[str]) -> None:
        self._routes = routes

    def has_route(self, endpoint: str) -> bool:
        return endpoint in self._routes


class _FakeTransport:
    def __init__(self, engine: _FakeEngine) -> None:
        self._engine = engine


class _FakeClient:
    def __init__(self, engine: _FakeEngine) -> None:
        self._native_transport = _FakeTransport(engine)


class _RecordingStore:
    """A HelixGraphStore whose transport honours the real route semantics.

    The engine-side routes are modelled faithfully — filter, ORDER<Desc> on
    created_at, RANGE(0, limit) — because a fake that returns every row for
    every endpoint would let a store that ignores the bound look correct.
    """

    def __init__(self, rows: list[dict], *, has_page_routes: bool = True) -> None:
        self.rows = rows
        self.store = HelixGraphStore(HelixDBConfig())
        routes = {"find_episode_by_episode_id"} | (
            set(PAGE_ROUTES) if has_page_routes else set()
        )
        self.store._helix_client = _FakeClient(_FakeEngine(routes))
        self.calls: list[tuple[str, dict]] = []
        self.rows_returned = 0
        self.store._query = self._query  # type: ignore[method-assign]

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        self.calls.append((endpoint, dict(payload)))
        if endpoint in PAGE_ROUTES:
            declared = PAGE_ROUTES[endpoint]
            rows = [
                row
                for row in self.rows
                if row["group_id"] == payload["gid"]
                and row["created_at"] < payload["before"]
                and all(
                    row.get(_PAGE_FILTER_FIELD[param]) == payload[param] for param in declared
                )
            ]
            rows.sort(key=lambda row: row["created_at"], reverse=True)
            result = rows[: payload["limit"]]
        elif endpoint == "find_episodes_by_source":
            result = [r for r in self.rows if r.get("source") == payload["src"]]
        elif endpoint == "find_episodes_by_status":
            result = [r for r in self.rows if r.get("status") == payload["st"]]
        else:
            result = list(self.rows)
        self.rows_returned += len(result)
        return result


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
async def test_one_page_does_not_pull_the_whole_group_across_the_transport() -> None:
    """THE ticket-22 assertion: the store is never asked for the whole group."""
    harness = _RecordingStore(_rows())

    episodes, next_cursor = await harness.store.get_episodes_paginated(
        group_id="brain",
        limit=PAGE_LIMIT,
    )

    assert len(episodes) == PAGE_LIMIT
    assert next_cursor is not None
    assert [endpoint for endpoint, _ in harness.calls] == ["find_episodes_by_group_page"]
    # limit + 1: the extra row is the has-more probe, not a page row.
    assert harness.rows_returned <= PAGE_LIMIT + 1, (
        f"the listing pulled {harness.rows_returned} rows across the transport to "
        f"serve a {PAGE_LIMIT}-row page out of a {STORE_ROWS}-row store — "
        "full-group materialisation is back"
    )


@pytest.mark.asyncio
async def test_transport_volume_is_flat_in_store_size() -> None:
    """Doubling the store must not change what one page costs to serve."""
    small = _RecordingStore(_rows(500))
    large = _RecordingStore(_rows(STORE_ROWS))

    await small.store.get_episodes_paginated(group_id="brain", limit=PAGE_LIMIT)
    await large.store.get_episodes_paginated(group_id="brain", limit=PAGE_LIMIT)

    assert small.rows_returned == large.rows_returned == PAGE_LIMIT + 1


@pytest.mark.asyncio
async def test_every_filter_combination_is_bounded() -> None:
    """`source` and `status` are accepted params, so each combination is a DoS
    surface of its own; a Python-side re-filter cannot fix one after truncation."""
    combos = [
        ({}, "find_episodes_by_group_page"),
        ({"source": "chat"}, "find_episodes_by_source_page"),
        ({"status": "completed"}, "find_episodes_by_status_page"),
        (
            {"source": "chat", "status": "completed"},
            "find_episodes_by_source_status_page",
        ),
    ]
    for kwargs, expected_endpoint in combos:
        harness = _RecordingStore(_rows())
        episodes, _cursor = await harness.store.get_episodes_paginated(
            group_id="brain",
            limit=PAGE_LIMIT,
            **kwargs,
        )
        assert [endpoint for endpoint, _ in harness.calls] == [expected_endpoint]
        assert len(episodes) == PAGE_LIMIT
        assert harness.rows_returned <= PAGE_LIMIT + 1, (
            f"{expected_endpoint} pulled {harness.rows_returned} rows for a "
            f"{PAGE_LIMIT}-row page"
        )


@pytest.mark.asyncio
async def test_cursor_pages_are_bounded_too() -> None:
    """A cursor page is the same DoS surface: `?cursor=...&limit=200`."""
    harness = _RecordingStore(_rows())
    _first, cursor = await harness.store.get_episodes_paginated(
        group_id="brain",
        limit=PAGE_LIMIT,
    )
    harness.rows_returned = 0
    harness.calls.clear()

    second, _next_cursor = await harness.store.get_episodes_paginated(
        group_id="brain",
        cursor=cursor,
        limit=PAGE_LIMIT,
    )

    assert len(second) == PAGE_LIMIT
    assert harness.calls[0][1]["before"] == cursor
    assert harness.rows_returned <= PAGE_LIMIT + 1


@pytest.mark.asyncio
async def test_page_peak_memory_does_not_scale_with_store_size(monkeypatch) -> None:
    """Peak allocation of one page must not grow with the number of stored rows."""

    async def _peak_for(row_count: int) -> int:
        harness = _RecordingStore(_rows(row_count))
        await harness.store.get_episodes_paginated(group_id="brain", limit=PAGE_LIMIT)
        tracemalloc.start()
        episodes, _cursor = await harness.store.get_episodes_paginated(
            group_id="brain",
            limit=PAGE_LIMIT,
        )
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(episodes) == PAGE_LIMIT
        return peak

    small_peak = await _peak_for(500)
    large_peak = await _peak_for(STORE_ROWS)

    assert large_peak < small_peak * 1.25, (
        f"peak allocation scaled with store size: {small_peak / 1e6:.2f}MB for 500 rows "
        f"vs {large_peak / 1e6:.2f}MB for {STORE_ROWS} rows"
    )


@pytest.mark.asyncio
async def test_ordering_cursor_and_filters_are_unchanged(monkeypatch) -> None:
    """Bounding the page must not change what the endpoint returns."""
    rows = _rows(600)
    rows[7]["source"] = "import"
    rows[9]["status"] = "failed"
    harness = _RecordingStore(rows)
    store = harness.store

    first, cursor = await store.get_episodes_paginated(group_id="brain", limit=10)
    assert [e.id for e in first] == [f"ep_{i:06d}" for i in range(599, 589, -1)]
    assert cursor == first[-1].created_at.isoformat()

    second, _next_cursor = await store.get_episodes_paginated(
        group_id="brain",
        cursor=cursor,
        limit=10,
    )
    assert [e.id for e in second] == [f"ep_{i:06d}" for i in range(589, 579, -1)]

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
async def test_last_page_reports_no_cursor() -> None:
    """The has-more probe must not invent a cursor at the end of the listing."""
    harness = _RecordingStore(_rows(15))
    episodes, cursor = await harness.store.get_episodes_paginated(group_id="brain", limit=20)
    assert len(episodes) == 15
    assert cursor is None


@pytest.mark.asyncio
async def test_the_page_hydrates_only_the_page(monkeypatch) -> None:
    """Hydration stays bounded as well (the earlier, partial fix)."""
    harness = _RecordingStore(_rows())
    hydrated = _count_hydrations(monkeypatch)

    await harness.store.get_episodes_paginated(group_id="brain", limit=PAGE_LIMIT)

    assert len(hydrated) == PAGE_LIMIT


@pytest.mark.asyncio
async def test_the_page_caches_its_own_helix_ids() -> None:
    """The listing still warms the id cache — but only for rows it actually saw.

    The pre-ticket-22 listing warmed the id cache for EVERY row in the group,
    because it scanned every row anyway. That side effect is gone by
    construction: an engine-bounded page cannot see rows outside itself. The
    replacement for cold id resolution is `find_episode_by_episode_id`
    (ticket 34), not a full scan disguised as a listing.
    """
    harness = _RecordingStore(_rows(500))

    await harness.store.get_episodes_paginated(group_id="brain", limit=10)

    cache = harness.store._episode_group_id_cache
    assert cache[("brain", "ep_000499")] == 1499
    assert cache[("brain", "ep_000490")] == 1490
    assert ("brain", "ep_000000") not in cache
    assert len(cache) == 10


@pytest.mark.asyncio
async def test_an_engine_without_the_route_falls_back_and_says_so(monkeypatch) -> None:
    """Engines whose binary predates the route keep working — and are counted.

    This is a live state, not a hypothetical: an install that has not rebuilt
    native has no page route, so the fallback must produce the same answer.
    """
    from engram.storage.helix import graph as graph_module

    monkeypatch.setattr(
        graph_module, "_EPISODE_PAGE_STATS", {"bounded": 0, "unbounded_fallback": 0}
    )
    monkeypatch.setattr(graph_module, "_EPISODE_PAGE_FALLBACK_WARNED", False)

    bounded = _RecordingStore(_rows(600))
    legacy = _RecordingStore(_rows(600), has_page_routes=False)

    expected, expected_cursor = await bounded.store.get_episodes_paginated(
        group_id="brain",
        limit=10,
    )
    actual, actual_cursor = await legacy.store.get_episodes_paginated(
        group_id="brain",
        limit=10,
    )

    assert [e.id for e in actual] == [e.id for e in expected]
    assert actual_cursor == expected_cursor
    assert [endpoint for endpoint, _ in legacy.calls] == ["find_episodes_by_group"]
    # ... and it pays the full-group price, which is the whole point of the ticket.
    assert legacy.rows_returned == 600
    assert graph_module.get_episode_page_stats() == {"bounded": 1, "unbounded_fallback": 1}


@pytest.mark.asyncio
async def test_get_episode_by_id_uses_the_targeted_route(monkeypatch) -> None:
    """Ticket 34: fetching one episode must not scan the group (ticket 22's cost)."""
    harness = _RecordingStore(_rows(600))

    async def _query(endpoint: str, payload: dict) -> list[dict]:
        harness.calls.append((endpoint, dict(payload)))
        if endpoint == "find_episode_by_episode_id":
            rows = [
                r
                for r in harness.rows
                if r["episode_id"] == payload["eid"] and r["group_id"] == payload["gid"]
            ]
            harness.rows_returned += len(rows)
            return rows
        rows = list(harness.rows)
        harness.rows_returned += len(rows)
        return rows

    harness.store._query = _query  # type: ignore[method-assign]

    episode = await harness.store.get_episode_by_id("ep_000123", "brain")

    assert episode is not None and episode.id == "ep_000123"
    assert [endpoint for endpoint, _ in harness.calls] == ["find_episode_by_episode_id"]
    assert harness.rows_returned == 1
    assert harness.store._episode_group_id_cache[("brain", "ep_000123")] == 1123


@pytest.mark.asyncio
async def test_get_episode_by_id_absence_does_not_trigger_a_group_scan() -> None:
    """A miss through the targeted route is a real absence, not a reason to scan."""
    harness = _RecordingStore(_rows(600))

    async def _query(endpoint: str, payload: dict) -> list[dict]:
        harness.calls.append((endpoint, dict(payload)))
        if endpoint == "find_episode_by_episode_id":
            return []
        rows = list(harness.rows)
        harness.rows_returned += len(rows)
        return rows

    harness.store._query = _query  # type: ignore[method-assign]

    assert await harness.store.get_episode_by_id("ep_999999", "brain") is None
    assert [endpoint for endpoint, _ in harness.calls] == ["find_episode_by_episode_id"]
    assert harness.rows_returned == 0
