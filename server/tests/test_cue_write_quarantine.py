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
