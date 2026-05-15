"""Graph neighborhood, atlas, and temporal graph endpoints for the dashboard."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_atlas_service, get_manager
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


def _build_representation(
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


@router.get("/atlas")
async def get_atlas(
    request: Request,
    refresh: bool = Query(False, description="Force a fresh snapshot rebuild"),
    snapshot_id: str | None = Query(
        None,
        description="Load a specific persisted atlas snapshot",
    ),
) -> JSONResponse:
    """Return a stable, display-bounded atlas of the whole memory graph."""
    tenant = get_tenant(request)
    try:
        snapshot = await get_atlas_service().get_snapshot(
            tenant.group_id,
            force=refresh,
            snapshot_id=snapshot_id,
        )
    except LookupError as exc:
        logger.warning("Atlas snapshot lookup failed: %s", exc)
        return JSONResponse(status_code=404, content={"detail": "Atlas snapshot not found"})
    payload = {
        "representation": _build_representation(
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
    return JSONResponse(content=payload)


@router.get("/atlas/history")
async def get_atlas_history(
    request: Request,
    limit: int = Query(24, ge=1, le=120, description="Max snapshots to return"),
) -> JSONResponse:
    """Return atlas snapshot history for timeline scrubbing."""
    tenant = get_tenant(request)
    snapshots = await get_atlas_service().list_snapshots(tenant.group_id, limit=limit)
    return JSONResponse(
        content={
            "items": [
                {
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
                for snapshot in snapshots
            ],
        }
    )


@router.get("/regions/{region_id}")
async def get_region(
    request: Request,
    region_id: str,
    refresh: bool = Query(False, description="Force a fresh snapshot rebuild"),
    snapshot_id: str | None = Query(
        None,
        description="Load a specific persisted atlas snapshot",
    ),
) -> JSONResponse:
    """Return a bounded drill-down for one atlas region."""
    tenant = get_tenant(request)
    try:
        payload = await get_atlas_service().get_region_payload(
            tenant.group_id,
            region_id,
            force=refresh,
            snapshot_id=snapshot_id,
        )
    except LookupError as exc:
        logger.warning("Region or snapshot lookup failed: %s", exc)
        return JSONResponse(status_code=404, content={"detail": "Region or snapshot not found"})
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"detail": f"Region '{region_id}' not found"},
        )
    return JSONResponse(content=payload)


@router.get("/neighborhood")
async def get_neighborhood(
    request: Request,
    center: str | None = Query(None, description="Entity ID to center on"),
    depth: int = Query(2, ge=1, le=5, description="Number of hops"),
    max_nodes: int = Query(2000, ge=1, le=10000, description="Max nodes to return"),
    min_activation: float = Query(0.0, ge=0.0, le=1.0, description="Min activation filter"),
) -> JSONResponse:
    """Return a subgraph neighborhood centered on an entity.

    When no center is specified, loads ALL entities (up to max_nodes) with all
    edges between them, so the dashboard gets the full graph on initial load.
    """
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    payload = await manager.get_graph_neighborhood(
        group_id=group_id,
        center=center,
        depth=depth,
        max_nodes=max_nodes,
        min_activation=min_activation,
    )
    if payload is None:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})
    return JSONResponse(content=payload)


@router.get("/at")
async def get_graph_at(
    request: Request,
    center: str = Query(..., description="Entity ID to center on"),
    at: str = Query(..., description="ISO 8601 timestamp for point-in-time query"),
    depth: int = Query(2, ge=1, le=5, description="BFS hops"),
    max_nodes: int = Query(2000, ge=1, le=10000, description="Max nodes"),
) -> JSONResponse:
    """Return a temporal subgraph — edges active at a specific point in time."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    try:
        at_time = datetime.fromisoformat(at)
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"detail": f"Invalid ISO 8601 timestamp: '{at}'"},
        )

    payload = await manager.get_temporal_graph(
        group_id=group_id,
        center=center,
        at_time=at_time,
        at_label=at,
        depth=depth,
        max_nodes=max_nodes,
    )
    if payload is None:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})
    return JSONResponse(content=payload)
