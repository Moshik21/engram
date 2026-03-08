"""Unit tests for FalkorDB consolidation helpers without Docker."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import FalkorDBConfig
from engram.models.episode import Episode
from engram.models.relationship import Relationship
from engram.storage.falkordb.graph import FalkorDBGraphStore


def _make_store() -> FalkorDBGraphStore:
    return FalkorDBGraphStore(
        FalkorDBConfig(host="localhost", port=6379, graph_name="test_graph"),
    )


class _FakeResult:
    def __init__(self, result_set):
        self.result_set = result_set


@pytest.mark.asyncio
async def test_create_episode_persists_consolidation_fields():
    store = _make_store()
    store._query = AsyncMock(return_value=_FakeResult([]))

    episode = Episode(
        id="ep1",
        content="hello",
        group_id="test",
        memory_tier="semantic",
        consolidation_cycles=3,
        entity_coverage=0.75,
    )

    await store.create_episode(episode)

    params = store._query.call_args.args[1]
    assert params["memory_tier"] == "semantic"
    assert params["consolidation_cycles"] == 3
    assert params["entity_coverage"] == 0.75


def test_node_to_episode_round_trips_tier_fields():
    store = _make_store()
    node = SimpleNamespace(
        properties={
            "id": "ep1",
            "content": "hello",
            "group_id": "test",
            "status": "completed",
            "memory_tier": "transitional",
            "consolidation_cycles": 2,
            "entity_coverage": 0.6,
            "encoding_context": "ctx",
            "created_at": "2026-03-01T12:00:00",
            "updated_at": "2026-03-01T12:05:00",
        },
    )

    episode = store._node_to_episode(node, "test")

    assert episode.memory_tier == "transitional"
    assert episode.consolidation_cycles == 2
    assert episode.entity_coverage == 0.6
    assert episode.encoding_context == "ctx"


@pytest.mark.asyncio
async def test_create_relationship_persists_polarity():
    store = _make_store()
    store._query = AsyncMock(return_value=_FakeResult([]))

    rel = Relationship(
        id="rel1",
        source_id="a",
        target_id="b",
        predicate="KNOWS",
        polarity="negative",
        group_id="test",
    )

    await store.create_relationship(rel)

    params = store._query.call_args.args[1]
    assert params["polarity"] == "negative"


@pytest.mark.asyncio
async def test_maturation_queries_parse_results():
    store = _make_store()
    store._query = AsyncMock(
        side_effect=[
            _FakeResult([[5]]),
            _FakeResult([["2026-01-01T00:00:00", "2026-03-01T00:00:00"]]),
            _FakeResult([["WORKS_AT"], ["KNOWS"]]),
        ],
    )

    count = await store.get_entity_episode_count("ent1", "test")
    span = await store.get_entity_temporal_span("ent1", "test")
    predicates = await store.get_entity_relationship_types("ent1", "test")

    assert count == 5
    assert span == ("2026-01-01T00:00:00", "2026-03-01T00:00:00")
    assert predicates == ["WORKS_AT", "KNOWS"]


@pytest.mark.asyncio
async def test_structural_and_cooccurrence_queries_parse_results():
    store = _make_store()
    store._query = AsyncMock(
        side_effect=[
            _FakeResult([["a", "b", 4], ["c", "d", 3]]),
            _FakeResult([[7]]),
        ],
    )

    structural = await store.find_structural_merge_candidates("test", min_shared_neighbors=3)
    cooccurrence = await store.get_episode_cooccurrence_count("a", "b", "test")

    assert structural == [("a", "b", 4), ("c", "d", 3)]
    assert cooccurrence == 7
