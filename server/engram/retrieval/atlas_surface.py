"""REST atlas surface helpers for dashboard graph routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.responses import JSONResponse

from engram.models.atlas import AtlasSnapshot, AtlasSnapshotSummary


@dataclass(frozen=True)
class ApiAtlasSurface:
    """REST atlas payload plus HTTP status and optional warning context."""

    status_code: int
    payload: dict
    log_warning: str | None = None


def atlas_snapshot_not_found_payload() -> dict:
    """Return the REST 404 payload for missing atlas snapshots."""
    return {"detail": "Atlas snapshot not found"}


def atlas_region_or_snapshot_not_found_payload() -> dict:
    """Return the REST 404 payload for missing atlas region lookups."""
    return {"detail": "Region or snapshot not found"}


def atlas_region_not_found_payload(region_id: str) -> dict:
    """Return the REST 404 payload for a missing atlas region."""
    return {"detail": f"Region '{region_id}' not found"}


def build_api_atlas_json_response(surface: ApiAtlasSurface, logger: Any) -> JSONResponse:
    """Return an atlas JSON response and log lookup warnings outside route bodies."""
    if surface.log_warning:
        logger.warning(surface.log_warning)
    return JSONResponse(status_code=surface.status_code, content=surface.payload)


def build_atlas_representation(
    *,
    scope: str,
    layout: str,
    represented_entity_count: int,
    represented_edge_count: int,
    displayed_node_count: int,
    displayed_edge_count: int,
    truncated: bool,
    snapshot_id: str | None = None,
) -> dict:
    """Return the shared graph representation metadata payload."""
    payload = {
        "scope": scope,
        "layout": layout,
        "representedEntityCount": represented_entity_count,
        "representedEdgeCount": represented_edge_count,
        "displayedNodeCount": displayed_node_count,
        "displayedEdgeCount": displayed_edge_count,
        "truncated": truncated,
    }
    if snapshot_id:
        payload["snapshotId"] = snapshot_id
    return payload


def build_atlas_snapshot_payload(snapshot: AtlasSnapshot) -> dict:
    """Return the REST atlas snapshot payload."""
    return {
        "representation": build_atlas_representation(
            scope="atlas",
            layout="precomputed",
            represented_entity_count=snapshot.represented_entity_count,
            represented_edge_count=snapshot.represented_edge_count,
            displayed_node_count=snapshot.displayed_node_count,
            displayed_edge_count=snapshot.displayed_edge_count,
            truncated=snapshot.truncated,
            snapshot_id=snapshot.id,
        ),
        "generatedAt": snapshot.generated_at,
        "regions": [
            {
                "id": region.id,
                "label": region.label,
                "subtitle": region.subtitle,
                "kind": region.kind,
                "memberCount": region.member_count,
                "representedEdgeCount": region.represented_edge_count,
                "activationScore": region.activation_score,
                "growth7d": region.growth_7d,
                "growth30d": region.growth_30d,
                "dominantEntityTypes": region.dominant_entity_types,
                "hubEntityIds": region.hub_entity_ids,
                "centerEntityId": region.center_entity_id,
                "latestEntityCreatedAt": region.latest_entity_created_at,
                "x": region.x,
                "y": region.y,
                "z": region.z,
            }
            for region in snapshot.regions
        ],
        "bridges": [
            {
                "id": bridge.id,
                "source": bridge.source,
                "target": bridge.target,
                "weight": bridge.weight,
                "relationshipCount": bridge.relationship_count,
            }
            for bridge in snapshot.bridges
        ],
        "stats": {
            "totalEntities": snapshot.total_entities,
            "totalRelationships": snapshot.total_relationships,
            "totalRegions": snapshot.total_regions,
            "hottestRegionId": snapshot.hottest_region_id,
            "fastestGrowingRegionId": snapshot.fastest_growing_region_id,
        },
    }


def build_atlas_history_item(snapshot: AtlasSnapshotSummary) -> dict:
    """Return one REST atlas history row."""
    return {
        "id": snapshot.id,
        "generatedAt": snapshot.generated_at,
        "representedEntityCount": snapshot.represented_entity_count,
        "representedEdgeCount": snapshot.represented_edge_count,
        "displayedNodeCount": snapshot.displayed_node_count,
        "displayedEdgeCount": snapshot.displayed_edge_count,
        "totalEntities": snapshot.total_entities,
        "totalRelationships": snapshot.total_relationships,
        "totalRegions": snapshot.total_regions,
        "hottestRegionId": snapshot.hottest_region_id,
        "fastestGrowingRegionId": snapshot.fastest_growing_region_id,
        "truncated": snapshot.truncated,
    }


async def build_api_atlas_surface(
    atlas_service: Any,
    *,
    group_id: str,
    refresh: bool,
    snapshot_id: str | None,
) -> ApiAtlasSurface:
    """Load and present the REST atlas snapshot surface."""
    try:
        snapshot = await atlas_service.get_snapshot(
            group_id,
            force=refresh,
            snapshot_id=snapshot_id,
        )
    except LookupError as exc:
        return ApiAtlasSurface(
            status_code=404,
            payload=atlas_snapshot_not_found_payload(),
            log_warning=f"Atlas snapshot lookup failed: {exc}",
        )
    return ApiAtlasSurface(status_code=200, payload=build_atlas_snapshot_payload(snapshot))


async def build_api_atlas_history_surface(
    atlas_service: Any,
    *,
    group_id: str,
    limit: int,
) -> dict:
    """Load and present the REST atlas history surface."""
    snapshots = await atlas_service.list_snapshots(group_id, limit=limit)
    return {"items": [build_atlas_history_item(snapshot) for snapshot in snapshots]}


async def build_api_atlas_region_surface(
    atlas_service: Any,
    *,
    group_id: str,
    region_id: str,
    refresh: bool,
    snapshot_id: str | None,
) -> ApiAtlasSurface:
    """Load and present the REST atlas region drill-down surface."""
    try:
        payload = await atlas_service.get_region_payload(
            group_id,
            region_id,
            force=refresh,
            snapshot_id=snapshot_id,
        )
    except LookupError as exc:
        return ApiAtlasSurface(
            status_code=404,
            payload=atlas_region_or_snapshot_not_found_payload(),
            log_warning=f"Region or snapshot lookup failed: {exc}",
        )
    if payload is None:
        return ApiAtlasSurface(
            status_code=404,
            payload=atlas_region_not_found_payload(region_id),
        )
    return ApiAtlasSurface(status_code=200, payload=payload)
