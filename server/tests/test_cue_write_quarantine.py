"""Episode ids in ~/.engram/cue-quarantine.txt are never written through update_episode_cue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.storage.helix import graph as graph_mod


@pytest.mark.asyncio
async def test_quarantined_episode_cue_is_not_written(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    (tmp_path / "cue-quarantine.txt").write_text(
        "# crashes LMDB on write\nep_bad 1f17c952\nep_other\n"
    )
    graph_mod._cue_quarantine_cache = None
    calls: list[str] = []

    async def query(endpoint, payload):
        calls.append(endpoint)
        return []

    from types import MethodType

    store = SimpleNamespace(_query=query, _find_cue_by_episode_route=lambda: "find_cue_by_episode")
    store._find_cue_rows = MethodType(graph_mod.HelixGraphStore._find_cue_rows, store)
    await graph_mod.HelixGraphStore.update_episode_cue(store, "ep_bad", {"cue_text": ""}, "default")
    assert calls == [], "quarantined cue reached the store"
    await graph_mod.HelixGraphStore.update_episode_cue(
        store, "ep_fine", {"cue_text": ""}, "default"
    )
    assert calls == ["find_cue_by_episode"], "non-quarantined cue must still be looked up"


def test_missing_quarantine_file_means_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    graph_mod._cue_quarantine_cache = None
    assert graph_mod._cue_write_quarantine() == frozenset()


def test_cue_lookup_prefers_the_indexed_route_when_present() -> None:
    """The scanning route leaks natively; the indexed one is used whenever the engine has it."""
    from types import MethodType, SimpleNamespace

    from engram.storage.helix.graph import HelixGraphStore

    def store_with_routes(routes: set[str]):
        engine = SimpleNamespace(has_route=lambda name: name in routes)
        store = SimpleNamespace(
            _helix_client=SimpleNamespace(_native_transport=SimpleNamespace(_engine=engine))
        )
        store._native_route_missing = MethodType(HelixGraphStore._native_route_missing, store)
        return store

    new = store_with_routes({"find_cue_by_episode", "find_cue_by_episode_indexed"})
    old = store_with_routes({"find_cue_by_episode"})
    assert HelixGraphStore._find_cue_by_episode_route(new) == "find_cue_by_episode_indexed"
    assert HelixGraphStore._find_cue_by_episode_route(old) == "find_cue_by_episode"


@pytest.mark.asyncio
async def test_indexed_cue_lookup_maps_no_value_found_to_empty() -> None:
    """collect_to_obj raises on no match; the adapter returns [] like the scanning route."""
    from types import MethodType

    from engram.storage.helix.graph import HelixGraphStore

    async def query(endpoint, payload):
        assert endpoint == "find_cue_by_episode_indexed"
        raise RuntimeError("native query 'find_cue_by_episode_indexed' failed: No value found")

    store = SimpleNamespace(
        _query=query, _find_cue_by_episode_route=lambda: "find_cue_by_episode_indexed"
    )
    store._find_cue_rows = MethodType(HelixGraphStore._find_cue_rows, store)
    assert await store._find_cue_rows("ep_missing", "default") == []

    async def query_other(endpoint, payload):
        raise RuntimeError("native query timed out")

    store._query = query_other
    with pytest.raises(RuntimeError):
        await store._find_cue_rows("ep_x", "default")


@pytest.mark.asyncio
async def test_exact_name_lookup_prefers_the_indexed_route_and_maps_absence(tmp_path) -> None:
    """The durable probe's exact-name lookup is one index read, not a label scan."""
    from types import MethodType

    from engram.storage.helix.graph import HelixGraphStore

    calls: list[str] = []

    async def query(endpoint, payload):
        calls.append(endpoint)
        if endpoint == "find_entity_exact_name_indexed":
            if payload["name_exact"] == "Ghost":
                raise RuntimeError("Query failed: Graph error: No value found")
            return [{"id": 7, "entity_id": "ent_k", "name": "Konner", "group_id": "default"}]
        raise AssertionError(f"scan route must not run: {endpoint}")

    engine = SimpleNamespace(has_route=lambda n: n == "find_entity_exact_name_indexed")
    store = SimpleNamespace(
        _query=query,
        _helix_client=SimpleNamespace(_native_transport=SimpleNamespace(_engine=engine)),
        _dict_to_entity=lambda d, g: SimpleNamespace(id=d["entity_id"], name=d["name"]),
    )
    for name in ("_native_route_missing", "_is_missing_route_error", "find_entities_exact_name"):
        setattr(store, name, MethodType(getattr(HelixGraphStore, name), store))
    hits = await store.find_entities_exact_name("Konner", "default")
    assert [h.id for h in hits] == ["ent_k"]
    assert await store.find_entities_exact_name("Ghost", "default") == []
    assert calls == ["find_entity_exact_name_indexed"] * 2
