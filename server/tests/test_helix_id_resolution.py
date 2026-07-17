"""Tests for Helix internal ID resolution via the targeted entity_id query."""

from __future__ import annotations

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.graph import HelixGraphStore
from engram.storage.helix.native_transport import NativeQueryError


def _group_rows() -> list[dict]:
    return [
        {"id": 101, "entity_id": "ent_a", "name": "Alpha", "group_id": "g1"},
        {"id": 202, "entity_id": "ent_b", "name": "Beta", "group_id": "g1"},
    ]


@pytest.mark.asyncio
async def test_resolve_entity_helix_id_uses_targeted_query(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == "find_entity_by_entity_id"
        assert payload == {"eid": "ent_b", "gid": "g1"}
        return [{"id": 202, "entity_id": "ent_b", "name": "Beta", "group_id": "g1"}]

    monkeypatch.setattr(store, "_query", fake_query)

    hid = await store._resolve_entity_helix_id("ent_b", "g1")

    assert hid == 202
    assert calls == [("find_entity_by_entity_id", {"eid": "ent_b", "gid": "g1"})]
    assert store._entity_group_id_cache[("g1", "ent_b")] == 202


@pytest.mark.asyncio
async def test_resolve_entity_helix_id_falls_back_when_route_missing(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[str] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append(endpoint)
        if endpoint == "find_entity_by_entity_id":
            raise NativeQueryError(endpoint, "route not found")
        assert endpoint == "find_entities_by_group"
        assert payload == {"gid": "g1"}
        return _group_rows()

    monkeypatch.setattr(store, "_query", fake_query)

    hid = await store._resolve_entity_helix_id("ent_b", "g1")

    assert hid == 202
    assert calls == ["find_entity_by_entity_id", "find_entities_by_group"]
    # Scan fallback keeps warming the cache for the whole group.
    assert store._entity_group_id_cache[("g1", "ent_a")] == 101
    assert store._entity_group_id_cache[("g1", "ent_b")] == 202


@pytest.mark.asyncio
async def test_resolve_entity_helix_id_falls_back_on_empty_targeted_result(
    monkeypatch,
) -> None:
    """Engines that swallow a missing route as [] still resolve via the scan."""
    store = HelixGraphStore(HelixDBConfig())
    calls: list[str] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append(endpoint)
        if endpoint == "find_entity_by_entity_id":
            return []
        assert endpoint == "find_entities_by_group"
        return _group_rows()

    monkeypatch.setattr(store, "_query", fake_query)

    hid = await store._resolve_entity_helix_id("ent_a", "g1")

    assert hid == 101
    assert calls == ["find_entity_by_entity_id", "find_entities_by_group"]


@pytest.mark.asyncio
async def test_resolve_entity_helix_id_returns_none_when_absent(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_entity_by_entity_id":
            return []
        assert endpoint == "find_entities_by_group"
        return _group_rows()

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_entity_helix_id("ent_missing", "g1") is None


@pytest.mark.asyncio
async def test_resolve_entity_helix_id_serves_from_cache(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    store._entity_group_id_cache[("g1", "ent_b")] = 202

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        raise AssertionError(f"unexpected query {endpoint!r}")

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_entity_helix_id("ent_b", "g1") == 202
