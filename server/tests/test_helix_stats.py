from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig, HelixDBConfig
from engram.models.episode_cue import EpisodeCue
from engram.storage.helix.graph import HelixGraphStore


@pytest.mark.parametrize(
    ("search_kwargs", "expected_endpoint", "expected_payload"),
    [
        ({}, "find_entities_all", {}),
        ({"name": "Python"}, "find_entities_by_name_all", {"name_query": "Python"}),
        ({"entity_type": "Technology"}, "find_entities_by_type_all", {"etype": "Technology"}),
        (
            {"name": "Python", "entity_type": "Technology"},
            "find_entities_by_name_and_type_all",
            {"name_query": "Python", "etype": "Technology"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_helix_find_entities_without_group_searches_all_groups(
    monkeypatch,
    search_kwargs: dict,
    expected_endpoint: str,
    expected_payload: dict,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == expected_endpoint
        assert payload == expected_payload
        return [
            {
                "id": 101,
                "entity_id": "ent_python_a",
                "name": "Python",
                "entity_type": "Technology",
                "group_id": "brain_a",
            },
            {
                "id": 202,
                "entity_id": "ent_python_b",
                "name": "Python",
                "entity_type": "Technology",
                "group_id": "brain_b",
            },
        ]

    monkeypatch.setattr(store, "_query", fake_query)

    entities = await store.find_entities(**search_kwargs, limit=10)

    assert {(entity.id, entity.group_id) for entity in entities} == {
        ("ent_python_a", "brain_a"),
        ("ent_python_b", "brain_b"),
    }
    assert calls == [(expected_endpoint, expected_payload)]
    assert store._entity_group_id_cache[("brain_a", "ent_python_a")] == 101
    assert store._entity_group_id_cache[("brain_b", "ent_python_b")] == 202


@pytest.mark.asyncio
async def test_helix_get_episodes_without_group_searches_all_groups(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == "find_episodes_all"
        assert payload == {}
        return [
            {
                "id": 303,
                "episode_id": "ep_a",
                "group_id": "brain_a",
                "content": "Brain A episode",
                "status": "pending",
                "created_at": "2026-05-14T08:00:00",
            },
            {
                "id": 404,
                "episode_id": "ep_b",
                "group_id": "brain_b",
                "content": "Brain B episode",
                "status": "pending",
                "created_at": "2026-05-14T09:00:00",
            },
        ]

    monkeypatch.setattr(store, "_query", fake_query)

    episodes = await store.get_episodes(limit=10)

    assert {(episode.id, episode.group_id) for episode in episodes} == {
        ("ep_a", "brain_a"),
        ("ep_b", "brain_b"),
    }
    assert calls == [("find_episodes_all", {})]
    assert store._episode_group_id_cache[("brain_a", "ep_a")] == 303
    assert store._episode_group_id_cache[("brain_b", "ep_b")] == 404


@pytest.mark.asyncio
async def test_helix_stats_without_group_uses_all_group_queries(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_all":
            return [
                {"id": 101, "entity_id": "ent_a", "group_id": "brain_a"},
                {"id": 202, "entity_id": "ent_b", "group_id": "brain_b"},
            ]
        if endpoint == "find_episodes_all":
            return [
                {
                    "id": 303,
                    "episode_id": "ep_a",
                    "group_id": "brain_a",
                    "projection_state": "projected",
                    "retry_count": 0,
                    "processing_duration_ms": 20,
                    "created_at": "2026-05-14T08:00:00",
                    "last_projected_at": "2026-05-14T08:00:01",
                },
                {
                    "id": 404,
                    "episode_id": "ep_b",
                    "group_id": "brain_b",
                    "projection_state": "queued",
                },
            ]
        if endpoint == "get_outgoing_edges":
            return []
        if endpoint == "find_cue_by_episode":
            is_ep_a = payload["ep_id"] == "ep_a"
            return [
                {
                    "episode_id": payload["ep_id"],
                    "group_id": payload["gid"],
                    "cue_text": f"{payload['gid']} cue",
                    "projection_state": "projected" if is_ep_a else "queued",
                    "hit_count": 2 if is_ep_a else 0,
                    "surfaced_count": 3 if is_ep_a else 1,
                    "selected_count": 1 if is_ep_a else 1,
                    "used_count": 1 if is_ep_a else 0,
                    "near_miss_count": 1 if is_ep_a else 0,
                    "policy_score": 0.7 if is_ep_a else 0.3,
                    "projection_attempts": 2 if is_ep_a else 1,
                }
            ]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    async def fake_episode_entities(
        episode_id: str,
        group_id: str | None = None,
    ) -> list[str]:
        assert episode_id == "ep_a"
        assert group_id is None
        return ["ent_a"]

    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_episode_entities", fake_episode_entities)

    stats = await store.get_stats()

    assert stats["entities"] == 2
    assert stats["episodes"] == 2
    assert stats["cue_metrics"]["cue_count"] == 2
    assert stats["cue_metrics"]["projected_cue_count"] == 1
    assert stats["cue_metrics"]["cue_hit_count"] == 2
    assert stats["cue_metrics"]["cue_hit_episode_count"] == 1
    assert stats["cue_metrics"]["cue_hit_episode_rate"] == 0.5
    assert stats["cue_metrics"]["cue_surfaced_count"] == 4
    assert stats["cue_metrics"]["cue_selected_count"] == 2
    assert stats["cue_metrics"]["cue_used_count"] == 1
    assert stats["cue_metrics"]["cue_near_miss_count"] == 1
    assert stats["cue_metrics"]["avg_policy_score"] == 0.5
    assert stats["cue_metrics"]["avg_projection_attempts"] == 1.5
    assert stats["projection_metrics"]["state_counts"]["projected"] == 1
    assert stats["projection_metrics"]["yield"]["linked_entity_count"] == 1
    assert ("find_entities_all", {}) in calls
    assert ("find_episodes_all", {}) in calls
    assert ("find_cue_by_episode", {"ep_id": "ep_a", "gid": "brain_a"}) in calls
    assert ("find_cue_by_episode", {"ep_id": "ep_b", "gid": "brain_b"}) in calls


@pytest.mark.asyncio
async def test_helix_stats_prefers_bulk_cue_and_projected_entity_queries(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_by_group":
            return [
                {"id": "h_ent_1", "entity_id": "ent_a", "group_id": "native_brain"},
                {"id": "h_ent_2", "entity_id": "ent_b", "group_id": "native_brain"},
            ]
        if endpoint == "find_episodes_by_group":
            return [
                {
                    "id": "h_ep_projected",
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "projection_state": "projected",
                    "retry_count": 0,
                    "processing_duration_ms": 20,
                    "created_at": "2026-05-13T12:00:00",
                    "last_projected_at": "2026-05-13T12:00:01",
                },
                {
                    "id": "h_ep_queued",
                    "episode_id": "ep_queued",
                    "group_id": "native_brain",
                    "projection_state": "queued",
                },
            ]
        if endpoint == "get_outgoing_edges":
            return []
        if endpoint == "find_cues_by_group":
            return [
                {
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "cue_text": "native projected cue",
                    "projection_state": "projected",
                    "hit_count": 5,
                    "surfaced_count": 6,
                    "selected_count": 4,
                    "used_count": 3,
                    "near_miss_count": 1,
                    "policy_score": 0.9,
                    "projection_attempts": 2,
                }
            ]
        if endpoint == "get_projected_episode_entities_by_group":
            assert payload == {"gid": "native_brain", "projection_state": "projected"}
            return [{"entity_id": "ent_a"}, {"entity_id": "ent_b"}]
        if endpoint in {"find_evidence_by_status", "find_adjudications_by_status"}:
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    async def unexpected_episode_entities(*_args, **_kwargs) -> list[str]:
        raise AssertionError("bulk stats path should not query per-episode entities")

    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_episode_entities", unexpected_episode_entities)

    stats = await store.get_stats("native_brain")

    endpoints = [endpoint for endpoint, _payload in calls]
    assert "find_cues_by_group" in endpoints
    assert "find_cue_by_episode" not in endpoints
    assert "get_projected_episode_entities_by_group" in endpoints
    assert stats["cue_metrics"]["cue_count"] == 1
    assert stats["cue_metrics"]["cue_hit_count"] == 5
    assert stats["projection_metrics"]["state_counts"]["projected"] == 1
    assert stats["projection_metrics"]["yield"]["linked_entity_count"] == 2
    assert (
        stats["projection_metrics"]["yield"]["avg_linked_entities_per_projected_episode"]
        == 2.0
    )


@pytest.mark.asyncio
async def test_native_helix_fast_stats_use_count_routes_before_full_scans(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig(transport="native"))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "count_entities_by_group":
            return [{"count": 9}]
        if endpoint == "count_episodes_by_group":
            return [{"count": 12}]
        if endpoint == "count_relationships_by_group":
            return [{"count": 7}]
        if endpoint == "count_cues_by_group":
            return [{"count": 10}]
        if endpoint in {
            "find_evidence_by_status",
            "find_adjudications_by_status",
            "find_pending_evidence",
            "find_pending_adjudications",
        }:
            raise AssertionError(
                f"recall fast stats must not scan adjudication queues: {endpoint}"
            )
        raise AssertionError(f"unexpected full Helix stats query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    stats = await store.get_stats("native_brain", exact=False)

    assert stats["entities"] == 9
    assert stats["episodes"] == 12
    assert stats["relationships"] == 7
    assert stats["cue_metrics"]["cue_count"] == 10
    assert stats["cue_metrics"]["episodes_without_cues"] == 2
    endpoints = [endpoint for endpoint, _payload in calls]
    assert "find_entities_by_group" not in endpoints
    assert "find_episodes_by_group" not in endpoints
    assert "find_cues_by_group" not in endpoints
    assert "adjudication_metrics" not in stats


@pytest.mark.asyncio
async def test_native_helix_fast_stats_drive_recall_without_timeout(monkeypatch) -> None:
    """Recall hot path uses count-only stats and avoids adjudication scans."""
    from engram.retrieval.pipeline import retrieve
    from engram.storage.helix.graph import HelixGraphStore

    store = HelixGraphStore(HelixDBConfig(transport="native"))
    adjudication_calls = 0

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        nonlocal adjudication_calls
        if endpoint.startswith("count_"):
            return [{"count": 9}]
        if endpoint in {
            "find_evidence_by_status",
            "find_adjudications_by_status",
            "find_pending_evidence",
            "find_pending_adjudications",
        }:
            adjudication_calls += 1
            raise AssertionError(
                f"recall fast stats must not scan adjudication queues: {endpoint}"
            )
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)
    store.get_active_neighbors_with_weights = AsyncMock(return_value=[])

    search = AsyncMock()
    search.search = AsyncMock(return_value=[("ent_a", 0.9)])
    search.search_episodes = AsyncMock(return_value=[])
    search.search_episode_cues = AsyncMock(return_value=[])
    search.compute_similarity = AsyncMock(return_value={})
    search._embeddings_enabled = False

    activation = AsyncMock()
    activation.batch_get = AsyncMock(return_value={})
    activation.get = AsyncMock(return_value=None)

    cfg = ActivationConfig(
        episode_retrieval_enabled=False,
        cue_recall_enabled=False,
        chunk_search_enabled=False,
        graph_query_expansion_enabled=False,
        retrieval_stats_timeout_ms=25,
        retrieval_primary_search_timeout_ms=0,
    )
    stage_timings: dict[str, float] = {}

    await retrieve(
        query="native recall",
        group_id="native_brain",
        graph_store=store,
        activation_store=activation,
        search_index=search,
        cfg=cfg,
        stage_timings_ms=stage_timings,
    )

    assert adjudication_calls == 0
    assert "recall_stats" in stage_timings
    assert stage_timings["recall_stats"] < 25
    assert "recall_stats_timeout" not in stage_timings


@pytest.mark.asyncio
async def test_native_helix_exact_stats_cache_adjudication_metrics(monkeypatch) -> None:
    store = HelixGraphStore(
        HelixDBConfig(transport="native", adjudication_metrics_cache_ttl_seconds=60.0),
    )
    adjudication_calls = 0

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        nonlocal adjudication_calls
        if endpoint == "find_entities_by_group":
            return [{"id": "h_ent_1", "entity_id": "ent_a", "group_id": "native_brain"}]
        if endpoint == "find_episodes_by_group":
            return [
                {
                    "id": "h_ep_projected",
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "projection_state": "projected",
                    "retry_count": 0,
                    "processing_duration_ms": 20,
                    "created_at": "2026-05-13T12:00:00",
                    "last_projected_at": "2026-05-13T12:00:01",
                }
            ]
        if endpoint == "get_outgoing_edges":
            return []
        if endpoint == "find_cues_by_group":
            return [
                {
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "cue_text": "native projected cue",
                    "projection_state": "projected",
                }
            ]
        if endpoint == "get_projected_episode_entities_by_group":
            return [{"entity_id": "ent_a"}]
        if endpoint in {"find_evidence_by_status", "find_adjudications_by_status"}:
            adjudication_calls += 1
            if endpoint == "find_evidence_by_status" and payload["st"] == "deferred":
                return [{"evidence_id": "ev_deferred", "status": "deferred"}]
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    first = await store.get_stats("native_brain")
    second = await store.get_stats("native_brain")

    assert first["adjudication_metrics"]["open_work_count"] == 1
    assert second["adjudication_metrics"]["open_work_count"] == 1
    assert adjudication_calls == 6


@pytest.mark.asyncio
async def test_native_helix_stats_default_to_exact_projection_metrics(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig(transport="native"))
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_by_group":
            return [{"id": "h_ent_1", "entity_id": "ent_a", "group_id": "native_brain"}]
        if endpoint == "find_episodes_by_group":
            return [
                {
                    "id": "h_ep_projected",
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "projection_state": "projected",
                    "retry_count": 0,
                    "processing_duration_ms": 20,
                    "created_at": "2026-05-13T12:00:00",
                    "last_projected_at": "2026-05-13T12:00:01",
                }
            ]
        if endpoint == "get_outgoing_edges":
            return []
        if endpoint == "find_cues_by_group":
            return [
                {
                    "episode_id": "ep_projected",
                    "group_id": "native_brain",
                    "cue_text": "native projected cue",
                    "projection_state": "projected",
                }
            ]
        if endpoint == "get_projected_episode_entities_by_group":
            return [{"entity_id": "ent_a"}]
        if endpoint in {"find_evidence_by_status", "find_adjudications_by_status"}:
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    stats = await store.get_stats("native_brain")

    endpoints = [endpoint for endpoint, _payload in calls]
    assert "count_episodes_by_group" not in endpoints
    assert "find_episodes_by_group" in endpoints
    assert stats["projection_metrics"]["state_counts"]["projected"] == 1
    assert stats["projection_metrics"]["yield"]["linked_entity_count"] == 1


@pytest.mark.asyncio
async def test_helix_episode_cue_round_trips_feedback_fields(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    stored_payload: dict = {}

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_cue_by_episode":
            if stored_payload:
                return [{"id": 101, **stored_payload}]
            return []
        if endpoint == "create_episode_cue":
            stored_payload.update(payload)
            return [{"id": 101, **payload}]
        if endpoint == "update_cue":
            stored_payload.update(payload)
            return [{"id": payload["id"], **stored_payload}]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)
    last_feedback_at = datetime(2026, 5, 14, 12, 30, tzinfo=timezone.utc)

    await store.upsert_episode_cue(
        EpisodeCue(
            episode_id="ep_feedback",
            group_id="native_brain",
            cue_text="feedback cue",
            entity_mentions=[{"name": "Engram"}],
            temporal_markers=["today"],
            quote_spans=["feedback cue"],
            contradiction_keys=["engram:state"],
            first_spans=["feedback"],
            hit_count=3,
            surfaced_count=4,
            selected_count=2,
            used_count=1,
            near_miss_count=1,
            policy_score=0.82,
            projection_attempts=2,
            last_feedback_at=last_feedback_at,
        )
    )

    assert stored_payload["hit_count"] == 3
    assert stored_payload["surfaced_count"] == 4
    assert stored_payload["selected_count"] == 2
    assert stored_payload["used_count"] == 1
    assert stored_payload["near_miss_count"] == 1
    assert stored_payload["policy_score"] == 0.82
    assert stored_payload["projection_attempts"] == 2

    cue = await store.get_episode_cue("ep_feedback", "native_brain")
    assert cue is not None
    assert cue.entity_mentions == [{"name": "Engram"}]
    assert cue.temporal_markers == ["today"]
    assert cue.quote_spans == ["feedback cue"]
    assert cue.contradiction_keys == ["engram:state"]
    assert cue.first_spans == ["feedback"]
    assert cue.hit_count == 3
    assert cue.surfaced_count == 4
    assert cue.selected_count == 2
    assert cue.used_count == 1
    assert cue.near_miss_count == 1
    assert cue.policy_score == 0.82
    assert cue.projection_attempts == 2
    assert cue.last_feedback_at == last_feedback_at

    await store.update_episode_cue(
        "ep_feedback",
        {"selected_count": 3, "used_count": 2},
        group_id="native_brain",
    )

    assert stored_payload["selected_count"] == 3
    assert stored_payload["used_count"] == 2
    assert stored_payload["surfaced_count"] == 4


@pytest.mark.asyncio
async def test_helix_update_episode_cue_falls_back_to_episode_key_without_id(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_cue_by_episode":
            return [
                {
                    "episode_id": "ep_feedback",
                    "group_id": "native_brain",
                    "cue_text": "feedback cue",
                    "supporting_spans_json": "[]",
                    "projection_state": "cued",
                    "hit_count": 1,
                    "surfaced_count": 1,
                    "selected_count": 0,
                    "used_count": 0,
                }
            ]
        if endpoint == "update_cue_by_episode":
            return [payload]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)
    last_feedback_at = datetime(2026, 5, 14, 12, 30, tzinfo=timezone.utc)

    await store.update_episode_cue(
        "ep_feedback",
        {
            "surfaced_count": 2,
            "selected_count": 1,
            "last_feedback_at": last_feedback_at,
        },
        group_id="native_brain",
    )

    assert calls[0] == (
        "find_cue_by_episode",
        {"ep_id": "ep_feedback", "gid": "native_brain"},
    )
    endpoint, payload = calls[1]
    assert endpoint == "update_cue_by_episode"
    assert payload["ep_id"] == "ep_feedback"
    assert payload["gid"] == "native_brain"
    assert payload["surfaced_count"] == 2
    assert payload["selected_count"] == 1
    assert payload["hit_count"] == 1
    assert payload["last_feedback_at"] == last_feedback_at.isoformat()
    assert "id" not in payload


@pytest.mark.asyncio
async def test_helix_upsert_episode_cue_falls_back_to_episode_key_without_id(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_cue_by_episode":
            return [
                {
                    "episode_id": "ep_feedback",
                    "group_id": "native_brain",
                    "cue_text": "old cue",
                    "supporting_spans_json": "[]",
                    "projection_state": "cued",
                }
            ]
        if endpoint == "update_cue_by_episode":
            return [payload]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    await store.upsert_episode_cue(
        EpisodeCue(
            episode_id="ep_feedback",
            group_id="native_brain",
            cue_text="updated cue",
            surfaced_count=3,
            selected_count=2,
            used_count=1,
        )
    )

    assert calls[0] == (
        "find_cue_by_episode",
        {"ep_id": "ep_feedback", "gid": "native_brain"},
    )
    endpoint, payload = calls[1]
    assert endpoint == "update_cue_by_episode"
    assert payload["ep_id"] == "ep_feedback"
    assert payload["gid"] == "native_brain"
    assert payload["cue_text"] == "updated cue"
    assert payload["surfaced_count"] == 3
    assert payload["selected_count"] == 2
    assert payload["used_count"] == 1
    assert "id" not in payload
    assert "created_at" not in payload


@pytest.mark.asyncio
async def test_helix_paginated_episodes_without_group_filters_all_groups(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == "find_episodes_all"
        assert payload == {}
        return [
            {
                "id": 303,
                "episode_id": "ep_keep",
                "group_id": "brain_a",
                "content": "Keep",
                "source": "chat",
                "status": "completed",
                "created_at": "2026-05-14T09:00:00",
            },
            {
                "id": 404,
                "episode_id": "ep_wrong_source",
                "group_id": "brain_b",
                "content": "Wrong source",
                "source": "import",
                "status": "completed",
                "created_at": "2026-05-14T08:00:00",
            },
            {
                "id": 505,
                "episode_id": "ep_wrong_status",
                "group_id": "brain_c",
                "content": "Wrong status",
                "source": "chat",
                "status": "failed",
                "created_at": "2026-05-14T07:00:00",
            },
        ]

    monkeypatch.setattr(store, "_query", fake_query)

    episodes, next_cursor = await store.get_episodes_paginated(
        source="chat",
        status="completed",
        limit=10,
    )

    assert [episode.id for episode in episodes] == ["ep_keep"]
    assert episodes[0].group_id == "brain_a"
    assert next_cursor is None
    assert calls == [("find_episodes_all", {})]


@pytest.mark.asyncio
async def test_helix_dashboard_analytics_without_group_use_all_group_queries(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "engram.storage.helix.graph.utc_now",
        lambda: datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc),
    )

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_all":
            return [
                {
                    "id": 101,
                    "entity_id": "ent_person",
                    "name": "Alice",
                    "entity_type": "Person",
                    "group_id": "brain_a",
                    "created_at": "2026-05-14T08:00:00",
                },
                {
                    "id": 202,
                    "entity_id": "ent_project",
                    "name": "Engram",
                    "entity_type": "Project",
                    "group_id": "brain_b",
                    "created_at": "2026-05-14T09:00:00",
                },
            ]
        if endpoint == "find_episodes_all":
            return [
                {
                    "id": 303,
                    "episode_id": "ep_a",
                    "group_id": "brain_a",
                    "created_at": "2026-05-14T10:00:00",
                }
            ]
        if endpoint == "get_outgoing_edges" and payload == {"id": 101}:
            return [{"group_id": "brain_a"}]
        if endpoint in {"get_outgoing_edges", "get_incoming_edges"}:
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    top_connected = await store.get_top_connected(limit=2)
    growth = await store.get_growth_timeline(days=3)
    type_counts = await store.get_entity_type_counts()

    assert top_connected[0]["id"] == "ent_person"
    assert top_connected[0]["edgeCount"] == 1
    assert growth == [{"date": "2026-05-14", "episodes": 1, "entities": 2}]
    assert type_counts == {"Person": 1, "Project": 1}
    assert calls.count(("find_entities_all", {})) == 3
    assert ("find_episodes_all", {}) in calls


@pytest.mark.asyncio
async def test_helix_unscoped_entity_resolver_requires_unique_entity_id(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        assert endpoint == "find_entities_all"
        assert payload == {}
        return [
            {"id": 101, "entity_id": "ent_shared", "group_id": "brain_a"},
            {"id": 202, "entity_id": "ent_shared", "group_id": "brain_b"},
        ]

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_entity_helix_id_unscoped("ent_shared") is None
    assert store._entity_id_cache["ent_shared"] is None
    assert store._entity_group_id_cache[("brain_a", "ent_shared")] == 101
    assert store._entity_group_id_cache[("brain_b", "ent_shared")] == 202


@pytest.mark.asyncio
async def test_helix_unscoped_neighbor_reads_resolve_unique_entity_across_groups(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    edge = {
        "rel_id": "rel_a",
        "source_id": "ent_a",
        "target_id": "ent_b",
        "predicate": "KNOWS",
        "weight": 0.75,
        "group_id": "brain_a",
    }
    neighbor = {
        "id": 202,
        "entity_id": "ent_b",
        "name": "Bob",
        "entity_type": "Person",
        "group_id": "brain_a",
    }

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_entities_all":
            return [
                {
                    "id": 101,
                    "entity_id": "ent_a",
                    "name": "Alice",
                    "entity_type": "Person",
                    "group_id": "brain_a",
                }
            ]
        if endpoint == "get_outgoing_edges" and payload == {"id": 101}:
            return [edge]
        if endpoint == "get_outgoing_neighbors" and payload == {"id": 101}:
            return [neighbor]
        if endpoint in {"get_incoming_edges", "get_incoming_neighbors"}:
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    weighted = await store.get_active_neighbors_with_weights("ent_a")
    pairs = await store.get_neighbors("ent_a", hops=1)

    assert weighted == [("ent_b", 0.75, "KNOWS", "Person")]
    assert len(pairs) == 1
    entity, rel = pairs[0]
    assert entity.id == "ent_b"
    assert rel.id == "rel_a"
    assert calls.count(("find_entities_all", {})) == 1
    assert ("find_entities_by_group", {"gid": "default"}) not in calls


@pytest.mark.asyncio
async def test_helix_unscoped_episode_resolver_requires_unique_episode_id(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        assert endpoint == "find_episodes_all"
        assert payload == {}
        return [
            {"id": 303, "episode_id": "ep_shared", "group_id": "brain_a"},
            {"id": 404, "episode_id": "ep_shared", "group_id": "brain_b"},
        ]

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_episode_helix_id_unscoped("ep_shared") is None
    assert store._episode_id_cache["ep_shared"] is None
    assert store._episode_group_id_cache[("brain_a", "ep_shared")] == 303
    assert store._episode_group_id_cache[("brain_b", "ep_shared")] == 404


@pytest.mark.asyncio
async def test_helix_unscoped_episode_link_paths_resolve_unique_ids(
    monkeypatch,
) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_episodes_all":
            return [{"id": 303, "episode_id": "ep_a", "group_id": "brain_a"}]
        if endpoint == "find_entities_all":
            return [{"id": 101, "entity_id": "ent_a", "group_id": "brain_a"}]
        if endpoint == "get_episode_entities":
            assert payload == {"id": 303}
            return [{"entity_id": "ent_a", "group_id": "brain_a"}]
        if endpoint == "link_episode_entity":
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store.get_episode_entities("ep_a") == ["ent_a"]
    await store.link_episode_entity("ep_a", "ent_a")

    assert ("find_episodes_all", {}) in calls
    assert ("find_entities_all", {}) in calls
    assert ("link_episode_entity", {"episode_id": 303, "entity_id": 101}) in calls
    assert ("find_episodes_by_group", {"gid": "default"}) not in calls
    assert ("find_entities_by_group", {"gid": "default"}) not in calls


@pytest.mark.asyncio
async def test_helix_stats_counts_episodes_cues_and_projection_yield(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_entities_by_group":
            assert payload == {"gid": "native_brain"}
            return [{"id": "h_ent_1"}, {"id": "h_ent_2"}]
        if endpoint == "find_episodes_by_group":
            assert payload == {"gid": "native_brain"}
            return [
                {
                    "episode_id": "ep_projected",
                    "projection_state": "projected",
                    "retry_count": 0,
                    "processing_duration_ms": 20,
                    "created_at": "2026-05-13T12:00:00",
                    "last_projected_at": "2026-05-13T12:00:01.500000",
                },
                {
                    "episode_id": "ep_failed",
                    "projection_state": "failed",
                    "retry_count": 1,
                    "processing_duration_ms": 5,
                },
            ]
        if endpoint == "get_outgoing_edges":
            return [{"rel_id": "rel_1"}] if payload["id"] == "h_ent_1" else []
        if endpoint == "find_cue_by_episode":
            if payload["ep_id"] == "ep_projected":
                return [
                    {
                        "episode_id": "ep_projected",
                        "group_id": "native_brain",
                        "cue_text": "native projected cue",
                        "projection_state": "projected",
                        "hit_count": 5,
                        "surfaced_count": 6,
                        "selected_count": 4,
                        "used_count": 3,
                        "near_miss_count": 1,
                        "policy_score": 0.9,
                        "projection_attempts": 2,
                    }
                ]
            return []
        if endpoint == "find_evidence_by_status":
            assert payload["gid"] == "native_brain"
            if payload["st"] == "pending":
                return [{"evidence_id": "ev_pending", "status": "pending"}]
            if payload["st"] == "deferred":
                return [{"evidence_id": "ev_deferred", "status": "deferred"}]
            if payload["st"] == "approved":
                return []
        if endpoint == "find_adjudications_by_status":
            assert payload["gid"] == "native_brain"
            if payload["st"] == "pending":
                return [{"request_id": "adj_pending", "status": "pending"}]
            if payload["st"] == "deferred":
                return []
            if payload["st"] == "error":
                return [{"request_id": "adj_error", "status": "error"}]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    async def fake_episode_entities(
        episode_id: str,
        group_id: str | None = None,
    ) -> list[str]:
        assert episode_id == "ep_projected"
        assert group_id == "native_brain"
        return ["ent_a", "ent_b"]

    monkeypatch.setattr(store, "_query", fake_query)
    monkeypatch.setattr(store, "get_episode_entities", fake_episode_entities)

    stats = await store.get_stats("native_brain")

    assert stats["entities"] == 2
    assert stats["relationships"] == 1
    assert stats["episodes"] == 2
    assert stats["cue_metrics"]["cue_count"] == 1
    assert stats["cue_metrics"]["episodes_without_cues"] == 1
    assert stats["cue_metrics"]["cue_coverage"] == 0.5
    assert stats["cue_metrics"]["cue_hit_count"] == 5
    assert stats["cue_metrics"]["cue_hit_episode_count"] == 1
    assert stats["cue_metrics"]["cue_surfaced_count"] == 6
    assert stats["cue_metrics"]["cue_selected_count"] == 4
    assert stats["cue_metrics"]["cue_used_count"] == 3
    assert stats["cue_metrics"]["cue_near_miss_count"] == 1
    assert stats["cue_metrics"]["avg_policy_score"] == 0.9
    assert stats["cue_metrics"]["avg_projection_attempts"] == 2.0
    assert stats["cue_metrics"]["cue_to_projection_conversion_rate"] == 1.0
    assert stats["projection_metrics"]["state_counts"]["projected"] == 1
    assert stats["projection_metrics"]["state_counts"]["failed"] == 1
    assert stats["projection_metrics"]["attempted_episode_count"] == 2
    assert stats["projection_metrics"]["total_attempts"] == 3
    assert stats["projection_metrics"]["avg_processing_duration_ms"] == 20
    assert stats["projection_metrics"]["avg_time_to_projection_ms"] == 1500
    assert stats["projection_metrics"]["yield"]["linked_entity_count"] == 2
    assert (
        stats["projection_metrics"]["yield"]["avg_linked_entities_per_projected_episode"]
        == 2.0
    )
    assert stats["adjudication_metrics"]["open_evidence_count"] == 2
    assert stats["adjudication_metrics"]["pending_evidence_count"] == 1
    assert stats["adjudication_metrics"]["deferred_evidence_count"] == 1
    assert stats["adjudication_metrics"]["open_request_count"] == 2
    assert stats["adjudication_metrics"]["pending_request_count"] == 1
    assert stats["adjudication_metrics"]["error_request_count"] == 1
    assert stats["adjudication_metrics"]["open_work_count"] == 4


@pytest.mark.asyncio
async def test_helix_pending_evidence_includes_open_non_pending_statuses(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == "find_evidence_by_status"
        assert payload["gid"] == "native_brain"
        status = payload["st"]
        if status == "pending":
            return [
                {
                    "id": "h_ev_pending",
                    "evidence_id": "ev_pending",
                    "group_id": "native_brain",
                    "status": "pending",
                    "confidence": 0.1,
                }
            ]
        if status == "deferred":
            return [
                {
                    "id": "h_ev_deferred",
                    "evidence_id": "ev_deferred",
                    "group_id": "native_brain",
                    "status": "deferred",
                    "confidence": 0.9,
                }
            ]
        if status == "approved":
            return [
                {
                    "id": "h_ev_approved",
                    "evidence_id": "ev_approved",
                    "group_id": "native_brain",
                    "status": "approved",
                    "confidence": 0.5,
                }
            ]
        raise AssertionError(f"unexpected status {status}")

    monkeypatch.setattr(store, "_query", fake_query)

    evidence = await store.get_pending_evidence("native_brain")

    assert [call[1]["st"] for call in calls] == ["pending", "deferred", "approved"]
    assert [item["evidence_id"] for item in evidence] == [
        "ev_deferred",
        "ev_approved",
        "ev_pending",
    ]
    assert {item["status"] for item in evidence} == {"pending", "deferred", "approved"}


@pytest.mark.asyncio
async def test_helix_update_evidence_finds_deferred_open_status(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "find_evidence_by_status":
            if payload["st"] == "deferred":
                return [
                    {
                        "id": "h_ev_deferred",
                        "evidence_id": "ev_deferred",
                        "group_id": "native_brain",
                        "status": "deferred",
                        "confidence": 0.4,
                    }
                ]
            return []
        if endpoint == "get_evidence":
            return [
                {
                    "deferred_cycles": 0,
                    "confidence": 0.4,
                }
            ]
        if endpoint == "update_evidence":
            assert payload["id"] == "h_ev_deferred"
            assert payload["status"] == "committed"
            assert payload["committed_id"] == "rel_1"
            assert payload["deferred_cycles"] == 0
            assert payload["confidence"] == 0.4
            return []
        raise AssertionError(f"unexpected endpoint {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    await store.update_evidence_status(
        "ev_deferred",
        "committed",
        {"committed_id": "rel_1"},
        group_id="native_brain",
    )

    assert calls[-1][0] == "update_evidence"
    assert calls[-1][1] == {
        "id": "h_ev_deferred",
        "status": "committed",
        "resolved_at": calls[-1][1]["resolved_at"],
        "commit_reason": "",
        "committed_id": "rel_1",
        "deferred_cycles": 0,
        "confidence": 0.4,
    }


@pytest.mark.asyncio
async def test_helix_update_evidence_normalizes_null_optional_strings(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    store._evidence_id_cache["ev_deferred"] = "h_ev_deferred"
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        if endpoint == "get_evidence":
            return [{"deferred_cycles": 2, "confidence": 0.61}]
        return []

    monkeypatch.setattr(store, "_query", fake_query)

    await store.update_evidence_status(
        "ev_deferred",
        "committed",
        {"commit_reason": None, "committed_id": None},
        group_id="native_brain",
    )

    assert calls[0][0] == "get_evidence"
    assert calls[1] == (
        "update_evidence",
        {
            "id": "h_ev_deferred",
            "status": "committed",
            "resolved_at": calls[1][1]["resolved_at"],
            "commit_reason": "",
            "committed_id": "",
            "deferred_cycles": 2,
            "confidence": 0.61,
        },
    )


@pytest.mark.asyncio
async def test_helix_pending_adjudications_include_deferred_and_error(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    calls: list[tuple[str, dict]] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        calls.append((endpoint, payload))
        assert endpoint == "find_adjudications_by_status"
        assert payload["gid"] == "native_brain"
        status = payload["st"]
        if status == "pending":
            return [
                {
                    "id": "h_adj_pending",
                    "request_id": "adj_pending",
                    "group_id": "native_brain",
                    "status": "pending",
                    "created_at": "2026-05-14T12:02:00Z",
                }
            ]
        if status == "deferred":
            return [
                {
                    "id": "h_adj_deferred",
                    "request_id": "adj_deferred",
                    "group_id": "native_brain",
                    "status": "deferred",
                    "created_at": "2026-05-14T12:01:00Z",
                }
            ]
        if status == "error":
            return [
                {
                    "id": "h_adj_error",
                    "request_id": "adj_error",
                    "group_id": "native_brain",
                    "status": "error",
                    "created_at": "2026-05-14T12:03:00Z",
                }
            ]
        raise AssertionError(f"unexpected status {status}")

    monkeypatch.setattr(store, "_query", fake_query)

    requests = await store.get_pending_adjudication_requests("native_brain")

    assert [call[1]["st"] for call in calls] == ["pending", "deferred", "error"]
    assert [item["request_id"] for item in requests] == [
        "adj_deferred",
        "adj_pending",
        "adj_error",
    ]
    assert {item["status"] for item in requests} == {"pending", "deferred", "error"}
    assert store._adjudication_id_cache["adj_deferred"] == "h_adj_deferred"


@pytest.mark.asyncio
async def test_helix_entity_id_cache_is_group_scoped(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    group_scans: list[str] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_entities_exact_name":
            return []
        if endpoint == "find_entities_by_group":
            gid = payload["gid"]
            group_scans.append(gid)
            helix_id = 101 if gid == "brain_a" else 202
            return [{"id": helix_id, "entity_id": "ent_shared", "group_id": gid}]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_entity_helix_id("ent_shared", "brain_a") == 101
    assert await store._resolve_entity_helix_id("ent_shared", "brain_b") == 202
    assert await store._resolve_entity_helix_id("ent_shared", "brain_a") == 101

    assert group_scans == ["brain_a", "brain_b"]
    assert store._entity_group_id_cache[("brain_a", "ent_shared")] == 101
    assert store._entity_group_id_cache[("brain_b", "ent_shared")] == 202
    assert store._entity_id_cache["ent_shared"] is None


@pytest.mark.asyncio
async def test_helix_episode_id_cache_is_group_scoped(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    group_scans: list[str] = []
    episode_entity_queries: list[int] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "find_episodes_by_group":
            gid = payload["gid"]
            group_scans.append(gid)
            helix_id = 303 if gid == "brain_a" else 404
            return [{"id": helix_id, "episode_id": "ep_shared", "group_id": gid}]
        if endpoint == "get_episode_entities":
            episode_entity_queries.append(payload["id"])
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store._resolve_episode_helix_id("ep_shared", "brain_a") == 303
    assert await store._resolve_episode_helix_id("ep_shared", "brain_b") == 404
    assert await store._resolve_episode_helix_id("ep_shared", "brain_a") == 303

    await store.get_episode_entities("ep_shared", group_id="brain_b")

    assert group_scans == ["brain_a", "brain_b"]
    assert episode_entity_queries == [404]
    assert store._episode_group_id_cache[("brain_a", "ep_shared")] == 303
    assert store._episode_group_id_cache[("brain_b", "ep_shared")] == 404
    assert store._episode_id_cache["ep_shared"] is None


@pytest.mark.asyncio
async def test_helix_link_episode_entity_uses_group_scoped_caches(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    store._cache_episode(303, "ep_shared", "brain_a")
    store._cache_entity(101, "ent_shared", "brain_a")
    store._cache_episode(404, "ep_shared", "brain_b")
    store._cache_entity(202, "ent_shared", "brain_b")
    link_payloads: list[dict] = []

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "link_episode_entity":
            link_payloads.append(payload)
            return []
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    await store.link_episode_entity("ep_shared", "ent_shared", group_id="brain_b")

    assert link_payloads == [{"episode_id": 404, "entity_id": 202}]


@pytest.mark.asyncio
async def test_helix_get_episode_entities_filters_entity_group(monkeypatch) -> None:
    store = HelixGraphStore(HelixDBConfig())
    store._cache_episode(404, "ep_shared", "brain_b")

    async def fake_query(endpoint: str, payload: dict) -> list[dict]:
        if endpoint == "get_episode_entities":
            assert payload == {"id": 404}
            return [
                {"entity_id": "ent_a", "group_id": "brain_a"},
                {"entity_id": "ent_b", "group_id": "brain_b"},
            ]
        raise AssertionError(f"unexpected Helix query {endpoint}")

    monkeypatch.setattr(store, "_query", fake_query)

    assert await store.get_episode_entities("ep_shared", group_id="brain_b") == ["ent_b"]


@pytest.mark.asyncio
async def test_helix_graph_store_closes_owned_shared_client() -> None:
    client = AsyncMock()
    store = HelixGraphStore(HelixDBConfig(), client=client, owns_client=True)

    await store.close()

    client.close.assert_awaited_once()
    assert store._helix_client is None


@pytest.mark.asyncio
async def test_helix_graph_store_does_not_close_borrowed_shared_client() -> None:
    client = AsyncMock()
    store = HelixGraphStore(HelixDBConfig(), client=client, owns_client=False)

    await store.close()

    client.close.assert_not_awaited()
    assert store._helix_client is client
