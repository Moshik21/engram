"""M0 keystone tests: by-id probes, index-consistency drain, BM25 self-heal.

Covers (agent-experience goal M0.1-M0.4):
- exact by-id probe fallback when the engine lacks the routes
- bounded consistency drain: duplicate repair, orphan recording, cursors
- machinery-skip hook on the episode vector backfill
- create_entity/create_episode BM25 doc-conflict self-heal (bootstrap-500
  regression class: planted stale doc, adopt-existing, re-key, repeat->debt)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.storage.index_completeness import (
    ByIdVectorProbeUnavailableError,
    _ids_with_vectors,
    backfill_missing_episode_vectors,
    run_index_consistency_drain,
)

GROUP = "g"


def _episode(eid: str, content: str = "real content", salience: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        id=eid,
        group_id=GROUP,
        content=content,
        deleted_at=None,
        created_at=None,
        salience_class=salience,
        encoding_context="",
        source="",
    )


class FakeConsistencyIndex:
    """Vector-row surface for the consistency drain.

    ``rows`` is a list of dicts with id/episode_id (+ chunk_index for chunks)
    per kind; the paged listing serves stable slices, the by-id probe answers
    globally, deletes remove rows (mirroring the tombstone contract).
    """

    def __init__(self, rows_by_kind: dict[str, list[dict]]):
        self.rows = {k: list(v) for k, v in rows_by_kind.items()}
        self.deleted: list[tuple[str, str]] = []
        self.probe_unavailable = False

    async def list_vector_rows_page(self, kind, group_id, start, end):
        if self.probe_unavailable:
            raise ByIdVectorProbeUnavailableError(kind)
        return [dict(r) for r in self.rows.get(kind, [])[start:end]]

    async def find_vector_rows_by_episode_ids(self, kind, episode_ids, group_id):
        wanted = set(episode_ids)
        return [dict(r) for r in self.rows.get(kind, []) if r["episode_id"] in wanted]

    async def delete_vector_row(self, kind, helix_id):
        for row in self.rows.get(kind, []):
            if row["id"] == helix_id:
                self.rows[kind].remove(row)
                self.deleted.append((kind, helix_id))
                return True
        return False


class FakeGraph:
    def __init__(self, episode_ids):
        self.episodes = [_episode(e) for e in episode_ids]

    async def get_episodes(self, group_id=None, limit=50_000):
        return list(self.episodes)

    async def get_episode(self, episode_id, group_id):
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        return None


def _row(hid: str, eid: str, **extra) -> dict:
    return {"id": hid, "episode_id": eid, "group_id": GROUP, "data": [0.1], **extra}


@pytest.mark.asyncio
async def test_drain_repairs_duplicates_even_split_across_pages():
    # ep2's duplicate rows sit far apart — a page-local census would miss the
    # pair; the global by-id re-probe must not.
    rows = [_row("v1", "ep1"), _row("v2", "ep2")] + [_row(f"f{i}", f"epf{i}") for i in range(3)]
    rows.append(_row("v9", "ep2"))  # duplicate, later "page"
    index = FakeConsistencyIndex({"cue": rows})
    graph = FakeGraph([r["episode_id"] for r in rows])

    report = await run_index_consistency_drain(
        graph, index, GROUP, kinds=("cue",), page_size=3, max_rows=100
    )

    cue = report["kinds"]["cue"]
    # Cue duplicates are DEBT-ONLY (never deleted): question-space cues make
    # per-episode cue multiplicity legitimate and CueVec carries no per-cue
    # discriminator (DUP_REPAIR_KINDS excludes "cue").
    assert cue["duplicate_keys"] == 1
    assert cue["duplicate_rows_deleted"] == 0
    assert index.deleted == []
    assert cue["sweep_complete"] is True
    assert cue["cursor_next"] == 0  # sweep done -> restart next window


@pytest.mark.asyncio
async def test_drain_repairs_episode_duplicates_split_across_pages():
    # Same page-split scenario on a REPAIRABLE kind: deletion still works.
    rows = [_row("v1", "ep1"), _row("v2", "ep2")] + [_row(f"f{i}", f"epf{i}") for i in range(3)]
    rows.append(_row("v9", "ep2"))
    index = FakeConsistencyIndex({"episode": rows})
    graph = FakeGraph([r["episode_id"] for r in rows])

    report = await run_index_consistency_drain(
        graph, index, GROUP, kinds=("episode",), page_size=3, max_rows=100
    )

    ep = report["kinds"]["episode"]
    assert ep["duplicate_keys"] == 1
    assert ep["duplicate_rows_deleted"] == 1
    assert ("episode", "v9") in index.deleted


@pytest.mark.asyncio
async def test_drain_dry_run_and_repair_budget():
    rows = [_row("a1", "ep1"), _row("a2", "ep1"), _row("b1", "ep2"), _row("b2", "ep2")]
    index = FakeConsistencyIndex({"episode": rows})
    graph = FakeGraph(["ep1", "ep2"])

    dry = await run_index_consistency_drain(graph, index, GROUP, kinds=("episode",), dry_run=True)
    assert dry["kinds"]["episode"]["duplicate_keys"] == 2
    assert index.deleted == []

    bounded = await run_index_consistency_drain(
        graph, index, GROUP, kinds=("episode",), max_repairs=1
    )
    assert bounded["kinds"]["episode"]["duplicate_rows_deleted"] == 1
    assert len(index.deleted) == 1
    assert bounded["repairs_remaining_budget"] == 0


@pytest.mark.asyncio
async def test_drain_orphans_recorded_not_deleted_by_default():
    rows = [_row("v1", "ep1"), _row("v2", "ghost")]
    index = FakeConsistencyIndex({"episode": rows})
    graph = FakeGraph(["ep1"])

    report = await run_index_consistency_drain(graph, index, GROUP, kinds=("episode",))
    ep = report["kinds"]["episode"]
    assert ep["orphan_rows_found"] == 1
    assert ep["orphan_rows_deleted"] == 0
    assert index.deleted == []

    repaired = await run_index_consistency_drain(
        graph, index, GROUP, kinds=("episode",), repair_orphans=True
    )
    assert repaired["kinds"]["episode"]["orphan_rows_deleted"] == 1
    assert ("episode", "v2") in index.deleted


@pytest.mark.asyncio
async def test_drain_chunk_duplicates_key_on_chunk_index():
    rows = [
        _row("c1", "ep1", chunk_index=0),
        _row("c2", "ep1", chunk_index=1),  # different chunk -> NOT a duplicate
        _row("c3", "ep1", chunk_index=1),  # same chunk -> duplicate
    ]
    index = FakeConsistencyIndex({"chunk": rows})
    graph = FakeGraph(["ep1"])

    report = await run_index_consistency_drain(graph, index, GROUP, kinds=("chunk",))
    chunk = report["kinds"]["chunk"]
    assert chunk["duplicate_keys"] == 1
    assert chunk["duplicate_rows_deleted"] == 1
    assert ("chunk", "c3") in index.deleted


@pytest.mark.asyncio
async def test_drain_probe_unavailable_reports_and_stops():
    index = FakeConsistencyIndex({"cue": [_row("v1", "ep1")]})
    index.probe_unavailable = True
    report = await run_index_consistency_drain(FakeGraph(["ep1"]), index, GROUP, kinds=("cue",))
    assert report["kinds"]["cue"]["stopped"] == "probe_unavailable"
    assert report["kinds"]["cue"]["rows_scanned"] == 0


@pytest.mark.asyncio
async def test_drain_without_surface_is_skipped():
    report = await run_index_consistency_drain(FakeGraph([]), object(), GROUP)
    assert "skipped" in report


@pytest.mark.asyncio
async def test_ids_with_vectors_falls_back_to_census_when_probe_unavailable():
    class Index:
        async def get_episode_embeddings(self, ids, group_id=None):
            raise ByIdVectorProbeUnavailableError("find_episode_vectors_by_ids")

        async def _embed_text(self, text):
            return [0.1]

        async def _vector_search_episodes(self, vec, k, group_id):
            return [("ep1", 0.9)], [], []

    present, exact = await _ids_with_vectors(
        Index(),
        ["ep1", "ep2"],
        GROUP,
        probe_attr="get_episode_embeddings",
        census_attr="_vector_search_episodes",
    )
    assert present == {"ep1"}
    assert exact is False  # census path -> inexact -> cursor stays in force


@pytest.mark.asyncio
async def test_episode_backfill_machinery_skip_hook():
    episodes = [_episode("ep1"), _episode("ep2", salience="machinery")]

    class Graph:
        async def get_episodes(self, group_id=None, limit=50_000):
            return episodes

    class Index:
        _embeddings_enabled = True

        def __init__(self):
            self.indexed = []

        async def get_episode_embeddings(self, ids, group_id=None):
            return {}

        async def _embed_texts(self, texts):
            return [[0.1]]

        async def index_episode(self, ep):
            self.indexed.append(ep.id)

    index = Index()
    result = await backfill_missing_episode_vectors(
        Graph(),
        index,
        GROUP,
        skip_episode=lambda ep: getattr(ep, "salience_class", "") == "machinery",
    )
    assert index.indexed == ["ep1"]
    assert result.missing_before == 1  # machinery episode never counted as debt


# ---------------------------------------------------------------------------
# M0.3 BM25 doc-conflict self-heal (bootstrap-500 regression class)
# ---------------------------------------------------------------------------


def _entity(eid: str = "ent-1"):
    from engram.models.entity import Entity

    return Entity(id=eid, name="Konner", group_id=GROUP, entity_type="Person", summary="s")


def _graph_store_with_planted_stale_doc(conflict_ids: set[str], existing: dict[str, str]):
    """HelixGraphStore whose transport 500s AddN for ids with a stale doc."""
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(HelixDBConfig())

    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint, payload):
        calls.append((endpoint, dict(payload)))
        if endpoint in ("create_entity", "create_episode"):
            key = payload.get("entity_id") or payload.get("episode_id")
            if key in conflict_ids:
                raise RuntimeError(
                    f"native query {endpoint!r} failed: "
                    f'{{"error":"BM25 document 42 already exists"}}'
                )
            return [{"id": f"hid-{key}"}]
        return []

    async def fake_resolve(external_id, group_id):
        return existing.get(external_id)

    store._query = fake_query  # type: ignore[method-assign]
    store._resolve_entity_helix_id = fake_resolve  # type: ignore[method-assign]
    store._resolve_episode_helix_id = fake_resolve  # type: ignore[method-assign]
    store._encrypt = lambda gid, text: text  # type: ignore[method-assign]
    return store, calls


@pytest.fixture(autouse=True)
def _reset_bm25_conflict_stats():
    from engram.storage.helix import graph as helix_graph

    saved = dict(helix_graph._BM25_CONFLICT_STATS)
    saved_ids = list(helix_graph._BM25_CONFLICT_IDS)
    for key in helix_graph._BM25_CONFLICT_STATS:
        helix_graph._BM25_CONFLICT_STATS[key] = 0
    helix_graph._BM25_CONFLICT_IDS.clear()
    yield
    helix_graph._BM25_CONFLICT_STATS.update(saved)
    helix_graph._BM25_CONFLICT_IDS[:] = saved_ids


@pytest.mark.asyncio
async def test_create_entity_conflict_adopts_existing_row():
    from engram.storage.helix.graph import get_bm25_conflict_stats

    store, calls = _graph_store_with_planted_stale_doc({"ent-1"}, {"ent-1": "hid-existing"})
    entity = _entity("ent-1")
    result = await store.create_entity(entity)
    assert result == "ent-1"
    assert entity.id == "ent-1"
    stats = get_bm25_conflict_stats()
    assert stats["counts"]["conflicts"] == 1
    assert stats["counts"]["adopted_existing"] == 1
    assert stats["counts"]["rekeyed"] == 0
    # exactly ONE create attempt — no retry loop
    assert sum(1 for ep, _ in calls if ep == "create_entity") == 1


@pytest.mark.asyncio
async def test_create_entity_orphan_doc_rekeys_once_loudly(caplog):
    import logging

    from engram.storage.helix.graph import get_bm25_conflict_stats

    store, calls = _graph_store_with_planted_stale_doc({"ent-1"}, {})
    entity = _entity("ent-1")
    with caplog.at_level(logging.ERROR):
        result = await store.create_entity(entity)
    assert result == "ent-1~bm25r"
    assert entity.id == "ent-1~bm25r"  # caller's object follows storage
    assert any("re-keying" in r.message for r in caplog.records)
    stats = get_bm25_conflict_stats()
    assert stats["counts"]["rekeyed"] == 1
    assert "entity:ent-1" in stats["ids"]
    creates = [p for ep, p in calls if ep == "create_entity"]
    assert [p["entity_id"] for p in creates] == ["ent-1", "ent-1~bm25r"]


@pytest.mark.asyncio
async def test_create_entity_repeat_conflict_on_rekeyed_id_raises_as_debt():
    from engram.storage.helix.graph import get_bm25_conflict_stats

    store, calls = _graph_store_with_planted_stale_doc({"ent-1", "ent-1~bm25r"}, {})
    with pytest.raises(RuntimeError, match="BM25 document"):
        await store.create_entity(_entity("ent-1"))
    stats = get_bm25_conflict_stats()
    assert stats["counts"]["failed"] == 1
    # bounded: original + one re-key attempt, never a loop
    assert sum(1 for ep, _ in calls if ep == "create_entity") == 2


@pytest.mark.asyncio
async def test_create_entity_non_bm25_errors_pass_through():
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore, get_bm25_conflict_stats

    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint, payload):
        raise RuntimeError("connection refused")

    store._query = fake_query  # type: ignore[method-assign]
    store._encrypt = lambda gid, text: text  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="connection refused"):
        await store.create_entity(_entity())
    assert get_bm25_conflict_stats()["counts"]["conflicts"] == 0


@pytest.mark.asyncio
async def test_create_episode_conflict_rekeys_bootstrap_regression():
    """Bootstrap-500 regression: a planted stale doc must not 500 the write."""
    from engram.models.episode import Episode

    store, calls = _graph_store_with_planted_stale_doc({"ep-1"}, {})
    episode = Episode(id="ep-1", group_id=GROUP, content="hello")
    result = await store.create_episode(episode)
    assert result == "ep-1~bm25r"
    assert episode.id == "ep-1~bm25r"
    creates = [p for ep, p in calls if ep == "create_episode"]
    assert [p["episode_id"] for p in creates] == ["ep-1", "ep-1~bm25r"]


# ---------------------------------------------------------------------------
# search.py by-id probe surface (fake transport)
# ---------------------------------------------------------------------------


def _search_index(query_impl, native_routes: set[str] | None = None):
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.storage.helix.search import HelixSearchIndex

    class Provider:
        def dimension(self):
            return 1

    class FakeEngine:
        def has_route(self, name):
            return name in (native_routes or set())

    class FakeClient:
        def __init__(self):
            self._native_transport = (
                SimpleNamespace(_engine=FakeEngine()) if native_routes is not None else None
            )

        async def query(self, endpoint, payload):
            return await query_impl(endpoint, payload)

    return HelixSearchIndex(
        HelixDBConfig(),
        Provider(),
        EmbeddingConfig(),
        client=FakeClient(),
        bm25_breaker_enabled=False,
    )


@pytest.mark.asyncio
async def test_get_episode_embeddings_exact_map():
    async def query_impl(endpoint, payload):
        assert endpoint == "find_episode_vectors_by_ids"
        return [
            {"id": "h1", "episode_id": "ep1", "data": [0.5, 0.5]},
            {"id": "h2", "episode_id": "ep1", "data": [0.5, 0.5]},  # dup -> first wins
        ]

    index = _search_index(query_impl)
    found = await index.get_episode_embeddings(["ep1", "ep2"], group_id=GROUP)
    assert found == {"ep1": [0.5, 0.5]}


@pytest.mark.asyncio
async def test_by_id_probe_raises_unavailable_when_native_route_missing():
    async def query_impl(endpoint, payload):  # pragma: no cover - never reached
        raise AssertionError("query must not run when the route is missing")

    index = _search_index(query_impl, native_routes=set())
    assert index.by_id_probe_available("cue") is False
    with pytest.raises(ByIdVectorProbeUnavailableError):
        await index.find_vector_rows_by_episode_ids("cue", ["ep1"], GROUP)


@pytest.mark.asyncio
async def test_by_id_probe_translates_missing_route_errors():
    async def query_impl(endpoint, payload):
        raise RuntimeError(f"Query '{endpoint}' failed: route not found")

    index = _search_index(query_impl)
    with pytest.raises(ByIdVectorProbeUnavailableError):
        await index.get_cue_embeddings(["ep1"], group_id=GROUP)


@pytest.mark.asyncio
async def test_delete_vector_row_tolerates_already_deleted():
    async def query_impl(endpoint, payload):
        raise RuntimeError("VectorAlreadyDeleted: 42")

    index = _search_index(query_impl)
    assert await index.delete_vector_row("cue", "42") is True


@pytest.mark.asyncio
async def test_delete_vector_row_counts_real_failures():
    async def query_impl(endpoint, payload):
        raise RuntimeError("engine exploded")

    index = _search_index(query_impl)
    assert await index.delete_vector_row("cue", "42") is False
