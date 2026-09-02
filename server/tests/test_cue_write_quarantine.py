"""Episode ids in ~/.engram/cue-quarantine.txt are never written through update_episode_cue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.storage.helix import graph as graph_mod


@pytest.mark.asyncio
async def test_quarantined_episode_cue_is_not_written(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    (tmp_path / "cue-quarantine.txt").write_text("# crashes LMDB on write\nep_bad 1f17c952\nep_other\n")
    graph_mod._cue_quarantine_cache = None
    calls: list[str] = []

    async def query(endpoint, payload):
        calls.append(endpoint)
        return []

    store = SimpleNamespace(_query=query)
    await graph_mod.HelixGraphStore.update_episode_cue(store, "ep_bad", {"cue_text": ""}, "default")
    assert calls == [], "quarantined cue reached the store"
    await graph_mod.HelixGraphStore.update_episode_cue(store, "ep_fine", {"cue_text": ""}, "default")
    assert calls == ["find_cue_by_episode"], "non-quarantined cue must still be looked up"


def test_missing_quarantine_file_means_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    graph_mod._cue_quarantine_cache = None
    assert graph_mod._cue_write_quarantine() == frozenset()
