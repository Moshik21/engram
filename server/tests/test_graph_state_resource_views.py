from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from engram.config import ActivationConfig
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.retrieval.graph_state import GraphStateService


class FakeGraphStore:
    def __init__(self) -> None:
        self.entity = SimpleNamespace(
            id="ent_alex",
            name="Alex",
            entity_type="Person",
            summary="Operator",
            lexical_regime="canonical",
            canonical_identifier="alex",
            identifier_label=True,
            activation_current=0.42,
            access_count=3,
            last_accessed=datetime(2026, 5, 15, 1, 0, tzinfo=timezone.utc),
            created_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 15, tzinfo=timezone.utc),
        )
        self.neighbor = SimpleNamespace(
            id="ent_project",
            name="Engram",
            entity_type="Project",
            summary="Memory runtime",
            activation_current=0.2,
            access_count=1,
            last_accessed=None,
            created_at=datetime(2026, 5, 15, 0, 30, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 15, 0, 30, tzinfo=timezone.utc),
        )
        self.relationship = SimpleNamespace(
            id="rel_1",
            source_id="ent_alex",
            target_id="ent_project",
            predicate="WORKS_ON",
            weight=0.8,
            valid_from=None,
            valid_to=None,
            created_at=datetime(2026, 5, 15, 2, 0, tzinfo=timezone.utc),
        )
        self.entities = {
            self.entity.id: self.entity,
            self.neighbor.id: self.neighbor,
        }
        self.episode = Episode(
            id="ep_1",
            content="remembered dashboard context " * 20,
            source="api",
            status=EpisodeStatus.COMPLETED,
            group_id="brain",
            conversation_date=datetime(2026, 5, 14, 12, 0, tzinfo=timezone.utc),
            created_at=datetime(2026, 5, 15, 4, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 5, 15, 4, 5, tzinfo=timezone.utc),
            retry_count=1,
            processing_duration_ms=42,
            projection_state=EpisodeProjectionState.SCHEDULED,
            last_projection_reason="cue_policy_used",
            last_projected_at=datetime(2026, 5, 15, 4, 10, tzinfo=timezone.utc),
        )
        self.cue = EpisodeCue(
            episode_id="ep_1",
            group_id="brain",
            projection_state=EpisodeProjectionState.SCHEDULED,
            cue_text="dashboard cue " * 40,
            route_reason="cue_policy_used",
            hit_count=3,
            surfaced_count=2,
            selected_count=1,
            used_count=1,
            near_miss_count=1,
            policy_score=0.81,
            projection_attempts=2,
            last_feedback_at=datetime(2026, 5, 15, 4, 8, tzinfo=timezone.utc),
            last_projected_at=datetime(2026, 5, 15, 4, 10, tzinfo=timezone.utc),
        )

    async def get_entity(self, entity_id: str, group_id: str):
        if group_id == "brain":
            return self.entities.get(entity_id)
        return None

    async def get_stats(self, group_id: str):
        assert group_id == "brain"
        return {"entities": 2, "relationships": 1, "episodes": 1}

    async def get_entity_type_counts(self, group_id: str):
        assert group_id == "brain"
        return {"Person": 1, "Project": 1}

    async def get_top_connected(self, *, group_id: str, limit: int):
        assert group_id == "brain"
        assert limit == 10
        return [
            {
                "id": "ent_alex",
                "name": "Alex",
                "entityType": "Person",
                "edgeCount": 1,
            }
        ]

    async def get_growth_timeline(self, *, group_id: str, days: int):
        assert group_id == "brain"
        assert days == 7
        return [{"date": "2026-05-15", "entities": 2, "episodes": 1}]

    async def get_relationships(self, entity_id: str, *, active_only: bool, group_id: str):
        assert active_only is True
        assert entity_id == self.entity.id
        assert group_id == "brain"
        return [self.relationship]

    async def batch_get_entities(self, entity_ids: list[str], group_id: str):
        assert group_id == "brain"
        return {entity_id: self.entities[entity_id] for entity_id in entity_ids}

    async def get_neighbors(
        self,
        entity_id: str,
        *,
        hops: int,
        group_id: str,
        max_results: int = 5000,
    ):
        assert hops == 1
        assert entity_id == self.entity.id
        assert group_id == "brain"
        assert max_results > 0
        return [(self.neighbor, self.relationship)]

    async def find_entities(self, *, group_id: str, limit: int):
        assert group_id == "brain"
        return list(self.entities.values())[:limit]

    async def get_all_edges(self, *, group_id: str, entity_ids: set[str], limit: int):
        assert group_id == "brain"
        assert limit > 0
        if {self.relationship.source_id, self.relationship.target_id} <= entity_ids:
            return [self.relationship]
        return []

    async def get_relationships_at(self, entity_id: str, at_time: datetime, *, group_id: str):
        assert group_id == "brain"
        assert at_time == datetime(2026, 5, 15, 3, 0, tzinfo=timezone.utc)
        return [self.relationship] if entity_id == self.entity.id else []

    async def get_episodes_paginated(
        self,
        *,
        group_id: str,
        cursor: str | None,
        limit: int,
        source: str | None,
        status: str | None,
    ):
        assert group_id == "brain"
        assert cursor == "cursor_1"
        assert limit == 1
        assert source == "api"
        assert status == "completed"
        return [self.episode], "cursor_2"

    async def get_episode_cue(self, episode_id: str, group_id: str):
        assert episode_id == "ep_1"
        assert group_id == "brain"
        return self.cue


class FakeActivationStore:
    async def get_activation(self, entity_id: str):
        assert entity_id == "ent_alex"
        return SimpleNamespace(access_history=[], access_count=2, last_accessed="recent")

    async def get_top_activated(self, *, group_id: str, limit: int):
        assert group_id == "brain"
        assert limit in {4, 40}
        return [
            (
                "ent_alex",
                SimpleNamespace(access_history=[1.0], access_count=3, last_accessed=1.0),
            )
        ]

    async def batch_get(self, entity_ids: list[str]):
        return {
            "ent_alex": SimpleNamespace(access_history=[], access_count=2, last_accessed=0.0),
            "ent_project": SimpleNamespace(access_history=[], access_count=1, last_accessed=0.0),
        } | {
            entity_id: SimpleNamespace(access_history=[], access_count=0, last_accessed=0.0)
            for entity_id in entity_ids
            if entity_id not in {"ent_alex", "ent_project"}
        }


async def _resolve_entity_name(entity_id: str, group_id: str) -> str:
    assert group_id == "brain"
    return {
        "ent_alex": "Alex",
        "ent_project": "Engram",
    }[entity_id]


def _service() -> GraphStateService:
    return GraphStateService(
        graph_store=FakeGraphStore(),
        activation_store=FakeActivationStore(),
        cfg=ActivationConfig(),
        get_recall_metrics=lambda _group_id: {},
        get_memory_operation_metrics=lambda _group_id: {},
        get_epistemic_metrics=lambda _group_id: {},
        resolve_entity_name=_resolve_entity_name,
    )


@pytest.mark.asyncio
async def test_graph_state_service_builds_entity_profile_resource_view() -> None:
    profile = await _service().get_entity_profile("ent_alex", "brain")

    assert profile["id"] == "ent_alex"
    assert profile["name"] == "Alex"
    assert profile["activation"]["access_count"] == 2
    assert profile["facts"] == [
        {
            "predicate": "WORKS_ON",
            "object": "Engram",
            "valid_from": None,
            "valid_to": None,
        }
    ]
    assert profile["created_at"] == "2026-05-15T00:00:00+00:00"


@pytest.mark.asyncio
async def test_graph_state_service_reports_missing_entity_profile() -> None:
    profile = await _service().get_entity_profile("missing", "brain")

    assert profile == {"error": "Entity not found", "entity_id": "missing"}


@pytest.mark.asyncio
async def test_graph_state_service_builds_rest_entity_detail_view() -> None:
    detail = await _service().get_entity_detail("ent_alex", "brain")

    assert detail is not None
    assert detail["id"] == "ent_alex"
    assert detail["entityType"] == "Person"
    assert detail["lexicalRegime"] == "canonical"
    assert detail["canonicalIdentifier"] == "alex"
    assert detail["identifierLabel"] is True
    assert detail["activationCurrent"] == 0.42
    assert detail["accessCount"] == 3
    assert detail["lastAccessed"] == "2026-05-15T01:00:00+00:00"
    assert detail["facts"] == [
        {
            "id": "rel_1",
            "predicate": "WORKS_ON",
            "direction": "outgoing",
            "other": {
                "id": "ent_project",
                "name": "Engram",
                "entityType": "Project",
            },
            "weight": 0.8,
            "validFrom": None,
            "validTo": None,
            "createdAt": "2026-05-15T02:00:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_graph_state_service_rest_entity_detail_missing_returns_none() -> None:
    assert await _service().get_entity_detail("missing", "brain") is None


@pytest.mark.asyncio
async def test_graph_state_service_builds_dashboard_stats_view() -> None:
    stats = await _service().get_dashboard_stats(group_id="brain", days=7)

    assert stats["stats"]["entities"] == 2
    assert stats["stats"]["entity_type_distribution"] == {"Person": 1, "Project": 1}
    assert stats["topActivated"][0]["id"] == "ent_alex"
    assert stats["topActivated"][0]["entityType"] == "Person"
    assert stats["topActivated"][0]["accessCount"] == 3
    assert stats["topConnected"] == [
        {
            "id": "ent_alex",
            "name": "Alex",
            "entityType": "Person",
            "edgeCount": 1,
        }
    ]
    assert stats["growthTimeline"] == [{"date": "2026-05-15", "entities": 2, "episodes": 1}]
    assert stats["groupId"] == "brain"


@pytest.mark.asyncio
async def test_graph_state_service_builds_episode_summary_listing_view() -> None:
    listing = await _service().list_episode_summaries(
        group_id="brain",
        cursor="cursor_1",
        limit=1,
        source="api",
        status="completed",
    )

    assert listing["nextCursor"] == "cursor_2"
    assert listing["total"] == 1
    item = listing["items"][0]
    assert item["episodeId"] == "ep_1"
    assert item["content"] == ("remembered dashboard context " * 20)[:200]
    assert item["source"] == "api"
    assert item["status"] == "completed"
    assert item["projectionState"] == "scheduled"
    assert item["lastProjectionReason"] == "cue_policy_used"
    assert item["lastProjectedAt"] == "2026-05-15T04:10:00Z"
    assert item["conversationDate"] == "2026-05-14T12:00:00Z"
    assert item["createdAt"] == "2026-05-15T04:00:00Z"
    assert item["updatedAt"] == "2026-05-15T04:05:00Z"
    assert item["retryCount"] == 1
    assert item["processingDurationMs"] == 42
    assert item["entities"] == []
    assert item["factsCount"] == 0
    assert item["cue"]["cueText"] == ("dashboard cue " * 40)[:240]
    assert item["cue"]["projectionState"] == "scheduled"
    assert item["cue"]["routeReason"] == "cue_policy_used"
    assert item["cue"]["hitCount"] == 3
    assert item["cue"]["policyScore"] == pytest.approx(0.81)
    assert item["cue"]["lastFeedbackAt"] == "2026-05-15T04:08:00Z"


@pytest.mark.asyncio
async def test_graph_state_service_builds_activation_snapshot_view() -> None:
    snapshot = await _service().get_activation_snapshot(group_id="brain", limit=2)

    assert len(snapshot["topActivated"]) == 1
    item = snapshot["topActivated"][0]
    assert item["entityId"] == "ent_alex"
    assert item["name"] == "Alex"
    assert item["entityType"] == "Person"
    assert item["accessCount"] == 3
    assert item["lastAccessedAt"] == datetime.fromtimestamp(1.0).isoformat()
    assert item["decayRate"] == ActivationConfig().decay_exponent


@pytest.mark.asyncio
async def test_graph_state_service_builds_activation_curve_view() -> None:
    curve = await _service().get_activation_curve(
        group_id="brain",
        entity_id="ent_alex",
        hours=1,
        points=3,
    )

    assert curve is not None
    assert curve["entityId"] == "ent_alex"
    assert curve["entityName"] == "Alex"
    assert len(curve["curve"]) == 3
    assert curve["hours"] == 1
    assert curve["points"] == 3
    assert curve["formula"].startswith("B_i = ln(")


@pytest.mark.asyncio
async def test_graph_state_service_activation_curve_missing_returns_none() -> None:
    curve = await _service().get_activation_curve(
        group_id="brain",
        entity_id="missing",
    )

    assert curve is None


@pytest.mark.asyncio
async def test_graph_state_service_builds_dashboard_neighborhood_view() -> None:
    neighborhood = await _service().get_graph_neighborhood(
        center="ent_alex",
        group_id="brain",
        depth=1,
        max_nodes=5,
    )

    assert neighborhood is not None
    assert neighborhood["centerId"] == "ent_alex"
    assert neighborhood["representation"]["scope"] == "neighborhood"
    assert neighborhood["representation"]["displayedNodeCount"] == 2
    assert {node["id"] for node in neighborhood["nodes"]} == {
        "ent_alex",
        "ent_project",
    }
    assert neighborhood["edges"] == [
        {
            "id": "rel_1",
            "source": "ent_alex",
            "target": "ent_project",
            "predicate": "WORKS_ON",
            "weight": 0.8,
            "validFrom": None,
            "validTo": None,
            "createdAt": "2026-05-15T02:00:00+00:00",
        }
    ]


@pytest.mark.asyncio
async def test_graph_state_service_builds_dashboard_neighborhood_auto_center() -> None:
    neighborhood = await _service().get_graph_neighborhood(group_id="brain", max_nodes=5)

    assert neighborhood is not None
    assert neighborhood["centerId"] == "ent_alex"
    assert neighborhood["representation"]["representedEntityCount"] == 2
    assert neighborhood["representation"]["representedEdgeCount"] == 1
    assert neighborhood["totalInNeighborhood"] == 2


@pytest.mark.asyncio
async def test_graph_state_service_dashboard_neighborhood_missing_center() -> None:
    assert await _service().get_graph_neighborhood(center="missing", group_id="brain") is None


@pytest.mark.asyncio
async def test_graph_state_service_builds_dashboard_temporal_graph_view() -> None:
    temporal = await _service().get_temporal_graph(
        center="ent_alex",
        group_id="brain",
        at_time=datetime(2026, 5, 15, 3, 0, tzinfo=timezone.utc),
        at_label="2026-05-15T03:00:00+00:00",
        depth=1,
        max_nodes=5,
    )

    assert temporal is not None
    assert temporal["centerId"] == "ent_alex"
    assert temporal["at"] == "2026-05-15T03:00:00+00:00"
    assert temporal["representation"]["scope"] == "temporal"
    assert temporal["representation"]["representedEntityCount"] == 2
    assert temporal["nodes"][0]["activationCurrent"] == 0.0
    assert temporal["edges"][0]["id"] == "rel_1"


@pytest.mark.asyncio
async def test_graph_state_service_dashboard_temporal_graph_missing_center() -> None:
    temporal = await _service().get_temporal_graph(
        center="missing",
        group_id="brain",
        at_time=datetime(2026, 5, 15, 3, 0, tzinfo=timezone.utc),
        at_label="2026-05-15T03:00:00+00:00",
    )

    assert temporal is None


@pytest.mark.asyncio
async def test_graph_state_service_builds_entity_neighbors_resource_view() -> None:
    neighbors = await _service().get_entity_neighbors("ent_alex", "brain")

    assert neighbors == [
        {
            "entity": {
                "id": "ent_project",
                "name": "Engram",
                "entity_type": "Project",
                "summary": "Memory runtime",
            },
            "relationship": {
                "predicate": "WORKS_ON",
                "source_id": "ent_alex",
                "target_id": "ent_project",
                "weight": 0.8,
            },
        }
    ]
