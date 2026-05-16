from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.models.atlas import (
    AtlasBridge,
    AtlasRegion,
    AtlasSnapshot,
    AtlasSnapshotSummary,
)
from engram.retrieval.atlas_surface import (
    atlas_region_not_found_payload,
    atlas_region_or_snapshot_not_found_payload,
    atlas_snapshot_not_found_payload,
    build_api_atlas_history_surface,
    build_api_atlas_region_surface,
    build_api_atlas_surface,
    build_atlas_representation,
    build_atlas_snapshot_payload,
)


def _snapshot() -> AtlasSnapshot:
    return AtlasSnapshot(
        id="atlas_1",
        group_id="brain_1",
        generated_at="2026-05-15T12:00:00+00:00",
        represented_entity_count=12,
        represented_edge_count=8,
        displayed_node_count=2,
        displayed_edge_count=1,
        total_entities=20,
        total_relationships=18,
        total_regions=2,
        hottest_region_id="region_1",
        fastest_growing_region_id="region_2",
        truncated=True,
        regions=[
            AtlasRegion(
                id="region_1",
                label="Work memory",
                subtitle="AI architecture",
                kind="theme",
                member_count=7,
                represented_edge_count=5,
                activation_score=0.83,
                growth_7d=2,
                growth_30d=4,
                dominant_entity_types={"Project": 3},
                hub_entity_ids=["ent_1", "ent_2"],
                center_entity_id="ent_1",
                latest_entity_created_at="2026-05-15T11:00:00+00:00",
                x=0.1,
                y=-0.2,
                z=0.3,
            )
        ],
        bridges=[
            AtlasBridge(
                id="bridge_1",
                source="region_1",
                target="region_2",
                weight=0.7,
                relationship_count=3,
            )
        ],
    )


def _summary(snapshot_id: str) -> AtlasSnapshotSummary:
    return AtlasSnapshotSummary(
        id=snapshot_id,
        group_id="brain_1",
        generated_at="2026-05-15T12:00:00+00:00",
        represented_entity_count=12,
        represented_edge_count=8,
        displayed_node_count=2,
        displayed_edge_count=1,
        total_entities=20,
        total_relationships=18,
        total_regions=2,
        hottest_region_id="region_1",
        fastest_growing_region_id="region_2",
        truncated=True,
    )


def test_build_atlas_representation_includes_snapshot_when_present() -> None:
    payload = build_atlas_representation(
        scope="atlas",
        layout="precomputed",
        represented_entity_count=12,
        represented_edge_count=8,
        displayed_node_count=2,
        displayed_edge_count=1,
        truncated=True,
        snapshot_id="atlas_1",
    )

    assert payload == {
        "scope": "atlas",
        "layout": "precomputed",
        "representedEntityCount": 12,
        "representedEdgeCount": 8,
        "displayedNodeCount": 2,
        "displayedEdgeCount": 1,
        "truncated": True,
        "snapshotId": "atlas_1",
    }


def test_build_atlas_snapshot_payload_preserves_dashboard_contract() -> None:
    payload = build_atlas_snapshot_payload(_snapshot())

    assert payload["representation"] == {
        "scope": "atlas",
        "layout": "precomputed",
        "representedEntityCount": 12,
        "representedEdgeCount": 8,
        "displayedNodeCount": 2,
        "displayedEdgeCount": 1,
        "truncated": True,
        "snapshotId": "atlas_1",
    }
    assert payload["generatedAt"] == "2026-05-15T12:00:00+00:00"
    assert payload["regions"] == [
        {
            "id": "region_1",
            "label": "Work memory",
            "subtitle": "AI architecture",
            "kind": "theme",
            "memberCount": 7,
            "representedEdgeCount": 5,
            "activationScore": 0.83,
            "growth7d": 2,
            "growth30d": 4,
            "dominantEntityTypes": {"Project": 3},
            "hubEntityIds": ["ent_1", "ent_2"],
            "centerEntityId": "ent_1",
            "latestEntityCreatedAt": "2026-05-15T11:00:00+00:00",
            "x": 0.1,
            "y": -0.2,
            "z": 0.3,
        }
    ]
    assert payload["bridges"] == [
        {
            "id": "bridge_1",
            "source": "region_1",
            "target": "region_2",
            "weight": 0.7,
            "relationshipCount": 3,
        }
    ]
    assert payload["stats"] == {
        "totalEntities": 20,
        "totalRelationships": 18,
        "totalRegions": 2,
        "hottestRegionId": "region_1",
        "fastestGrowingRegionId": "region_2",
    }


@pytest.mark.asyncio
async def test_build_api_atlas_surface_loads_and_formats_snapshot() -> None:
    service = AsyncMock()
    service.get_snapshot.return_value = _snapshot()

    result = await build_api_atlas_surface(
        service,
        group_id="brain_1",
        refresh=True,
        snapshot_id="atlas_1",
    )

    assert result.status_code == 200
    assert result.payload["representation"]["snapshotId"] == "atlas_1"
    assert result.log_warning is None
    service.get_snapshot.assert_awaited_once_with(
        "brain_1",
        force=True,
        snapshot_id="atlas_1",
    )


@pytest.mark.asyncio
async def test_build_api_atlas_surface_returns_lookup_payload() -> None:
    service = AsyncMock()
    service.get_snapshot.side_effect = LookupError("missing")

    result = await build_api_atlas_surface(
        service,
        group_id="brain_1",
        refresh=False,
        snapshot_id="atlas_missing",
    )

    assert result.status_code == 404
    assert result.payload == atlas_snapshot_not_found_payload()
    assert result.log_warning == "Atlas snapshot lookup failed: missing"


@pytest.mark.asyncio
async def test_build_api_atlas_history_surface_formats_summaries() -> None:
    service = AsyncMock()
    service.list_snapshots.return_value = [_summary("atlas_2"), _summary("atlas_1")]

    payload = await build_api_atlas_history_surface(
        service,
        group_id="brain_1",
        limit=10,
    )

    assert [item["id"] for item in payload["items"]] == ["atlas_2", "atlas_1"]
    assert payload["items"][0]["generatedAt"] == "2026-05-15T12:00:00+00:00"
    assert payload["items"][0]["totalEntities"] == 20
    assert payload["items"][0]["truncated"] is True
    service.list_snapshots.assert_awaited_once_with("brain_1", limit=10)


@pytest.mark.asyncio
async def test_build_api_atlas_region_surface_returns_region_payload() -> None:
    service = AsyncMock()
    service.get_region_payload.return_value = {"region": {"id": "region_1"}}

    result = await build_api_atlas_region_surface(
        service,
        group_id="brain_1",
        region_id="region_1",
        refresh=True,
        snapshot_id="atlas_1",
    )

    assert result.status_code == 200
    assert result.payload == {"region": {"id": "region_1"}}
    assert result.log_warning is None
    service.get_region_payload.assert_awaited_once_with(
        "brain_1",
        "region_1",
        force=True,
        snapshot_id="atlas_1",
    )


@pytest.mark.asyncio
async def test_build_api_atlas_region_surface_returns_missing_region_payload() -> None:
    service = AsyncMock()
    service.get_region_payload.return_value = None

    result = await build_api_atlas_region_surface(
        service,
        group_id="brain_1",
        region_id="missing_region",
        refresh=False,
        snapshot_id=None,
    )

    assert result.status_code == 404
    assert result.payload == atlas_region_not_found_payload("missing_region")
    assert result.log_warning is None


@pytest.mark.asyncio
async def test_build_api_atlas_region_surface_returns_lookup_payload() -> None:
    service = AsyncMock()
    service.get_region_payload.side_effect = LookupError("snapshot missing")

    result = await build_api_atlas_region_surface(
        service,
        group_id="brain_1",
        region_id="region_1",
        refresh=False,
        snapshot_id="atlas_missing",
    )

    assert result.status_code == 404
    assert result.payload == atlas_region_or_snapshot_not_found_payload()
    assert result.log_warning == "Region or snapshot lookup failed: snapshot missing"
