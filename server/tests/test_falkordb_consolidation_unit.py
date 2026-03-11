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


@pytest.mark.asyncio
async def test_store_evidence_persists_explicit_status_fields():
    store = _make_store()
    store._query = AsyncMock(return_value=_FakeResult([]))

    await store.store_evidence(
        [
            {
                "evidence_id": "evi_1",
                "episode_id": "ep1",
                "fact_class": "entity",
                "confidence": 0.9,
                "source_type": "client_proposal",
                "payload": {"name": "Alice", "entity_type": "Person"},
                "status": "committed",
                "commit_reason": "committed_on_hot_path",
                "committed_id": "ent_1",
            },
        ],
        group_id="test",
        default_status="committed",
    )

    params = store._query.call_args.args[1]
    assert params["status"] == "committed"
    assert params["commit_reason"] == "committed_on_hot_path"
    assert params["committed_id"] == "ent_1"
    assert params["resolved_at"] is not None


@pytest.mark.asyncio
async def test_get_pending_evidence_parses_deferred_rows():
    store = _make_store()
    store._query = AsyncMock(
        return_value=_FakeResult(
            [
                [
                    SimpleNamespace(
                        properties={
                            "evidence_id": "evi_1",
                            "episode_id": "ep1",
                            "group_id": "test",
                            "fact_class": "entity",
                            "confidence": 0.65,
                            "source_type": "narrow_extractor",
                            "extractor_name": "identity",
                            "payload_json": '{"name":"Alice","entity_type":"Person"}',
                            "signals_json": '["proper_name"]',
                            "status": "deferred",
                            "deferred_cycles": 2,
                            "created_at": "2026-03-09T00:00:00",
                        },
                    ),
                ],
            ],
        ),
    )

    pending = await store.get_pending_evidence(group_id="test")

    assert len(pending) == 1
    assert pending[0]["status"] == "deferred"
    assert pending[0]["payload"]["name"] == "Alice"
    query = store._query.call_args.args[0]
    assert "'approved'" in query


@pytest.mark.asyncio
async def test_get_entity_count_uses_entity_query():
    store = _make_store()
    store._query = AsyncMock(return_value=_FakeResult([[5]]))

    count = await store.get_entity_count("test")

    assert count == 5


@pytest.mark.asyncio
async def test_store_evidence_persists_ambiguity_fields():
    store = _make_store()
    store._query = AsyncMock(return_value=_FakeResult([]))

    await store.store_evidence(
        [
            {
                "evidence_id": "evi_2",
                "episode_id": "ep1",
                "fact_class": "relationship",
                "confidence": 0.61,
                "source_type": "narrow_extractor",
                "payload": {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
                "ambiguity_tags": ["negation_scope"],
                "ambiguity_score": 0.66,
                "adjudication_request_id": "adj_123",
                "status": "pending",
                "commit_reason": "needs_adjudication",
            },
        ],
        group_id="test",
    )

    params = store._query.call_args.args[1]
    assert params["ambiguity_tags_json"] == '["negation_scope"]'
    assert params["ambiguity_score"] == 0.66
    assert params["adjudication_request_id"] == "adj_123"


@pytest.mark.asyncio
async def test_adjudication_request_round_trip_helpers():
    store = _make_store()
    store._query = AsyncMock(
        side_effect=[
            _FakeResult([]),
            _FakeResult(
                [
                    [
                        SimpleNamespace(
                            properties={
                                "request_id": "adj_123",
                                "episode_id": "ep1",
                                "group_id": "test",
                                "status": "pending",
                                "ambiguity_tags_json": '["coreference"]',
                                "evidence_ids_json": '["evi_1"]',
                                "selected_text": "She reminded me about the dentist.",
                                "request_reason": "needs_adjudication:coreference",
                                "attempt_count": 0,
                                "created_at": "2026-03-09T00:00:00",
                            },
                        ),
                    ],
                ],
            ),
            _FakeResult(
                [
                    [
                        SimpleNamespace(
                            properties={
                                "request_id": "adj_123",
                                "episode_id": "ep1",
                                "group_id": "test",
                                "status": "pending",
                                "ambiguity_tags_json": '["coreference"]',
                                "evidence_ids_json": '["evi_1"]',
                                "selected_text": "She reminded me about the dentist.",
                                "request_reason": "needs_adjudication:coreference",
                                "attempt_count": 0,
                                "created_at": "2026-03-09T00:00:00",
                            },
                        ),
                    ],
                ],
            ),
        ],
    )

    await store.store_adjudication_requests(
        [
            {
                "request_id": "adj_123",
                "episode_id": "ep1",
                "ambiguity_tags": ["coreference"],
                "evidence_ids": ["evi_1"],
                "selected_text": "She reminded me about the dentist.",
                "request_reason": "needs_adjudication:coreference",
                "created_at": "2026-03-09T00:00:00",
            },
        ],
        group_id="test",
    )
    pending = await store.get_pending_adjudication_requests(group_id="test")
    fetched = await store.get_adjudication_request("adj_123", group_id="test")

    assert pending[0]["request_id"] == "adj_123"
    assert pending[0]["ambiguity_tags"] == ["coreference"]
    assert fetched["selected_text"] == "She reminded me about the dentist."
