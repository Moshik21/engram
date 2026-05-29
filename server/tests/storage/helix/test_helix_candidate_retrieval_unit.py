"""Unit coverage for Helix find_entity_candidates recall parity."""

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.graph import HelixGraphStore


def _row(entity_id: str, name: str, group_id: str = "brain") -> dict:
    return {
        "entity_id": entity_id,
        "name": name,
        "entity_type": "Thing",
        "summary": "",
        "group_id": group_id,
        "is_deleted": False,
    }


@pytest.mark.asyncio
async def test_find_entity_candidates_uses_bm25_before_contains(monkeypatch):
    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint in {"find_entities_exact_name", "find_entities_by_canonical"}:
            return []
        if endpoint == "search_entities_bm25_filtered":
            return [_row("e_python", "Python")]
        raise AssertionError(f"unexpected endpoint after BM25 hit: {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    candidates = await store.find_entity_candidates("python", "brain", limit=1)

    assert [candidate.id for candidate in candidates] == ["e_python"]
    assert [endpoint for endpoint, _ in calls] == [
        "find_entities_exact_name",
        "search_entities_bm25_filtered",
    ]
    assert calls[1][1] == {"query": "python", "k": 30, "gid": "brain"}


@pytest.mark.asyncio
async def test_find_entity_candidates_falls_back_to_unfiltered_bm25(monkeypatch):
    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_exact_name":
            return []
        if endpoint == "search_entities_bm25_filtered":
            return []
        if endpoint == "search_entities_bm25":
            return [
                _row("other", "Python", group_id="other"),
                _row("target", "Python", group_id="brain"),
            ]
        raise AssertionError(f"unexpected endpoint after BM25 fallback: {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    candidates = await store.find_entity_candidates("python", "brain", limit=1)

    assert [candidate.id for candidate in candidates] == ["target"]
    assert [endpoint for endpoint, _ in calls] == [
        "find_entities_exact_name",
        "search_entities_bm25_filtered",
        "search_entities_bm25",
    ]


@pytest.mark.asyncio
async def test_find_entity_candidates_tokenizes_punctuation_for_helix_contains(
    monkeypatch,
):
    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint in {
            "find_entities_exact_name",
            "search_entities_bm25_filtered",
            "search_entities_bm25",
            "find_entities_by_name",
        }:
            if endpoint == "find_entities_by_name" and payload["name_query"] == "ACT":
                return [_row("act_r", "ACT-R")]
            return []
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    candidates = await store.find_entity_candidates("ACT_R", "brain", limit=3)

    assert [candidate.id for candidate in candidates] == ["act_r"]
    assert ("find_entities_by_name", {"name_query": "ACT", "gid": "brain"}) in calls


@pytest.mark.asyncio
async def test_find_entity_candidates_uses_prefix_fallback(monkeypatch):
    store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint in {
            "find_entities_exact_name",
            "search_entities_bm25_filtered",
            "search_entities_bm25",
            "find_entities_by_name",
        }:
            return []
        if endpoint == "find_entities_by_name_prefix":
            assert payload == {"prefix": "Ale", "gid": "brain"}
            return [_row("alex", "Alex"), _row("alex_chen", "Alex Chen")]
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    candidates = await store.find_entity_candidates("Alexa", "brain", limit=3)

    assert [candidate.id for candidate in candidates] == ["alex", "alex_chen"]
    assert calls[-1] == ("find_entities_by_name_prefix", {"prefix": "Ale", "gid": "brain"})
