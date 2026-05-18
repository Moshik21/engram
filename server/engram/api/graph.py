"""Graph neighborhood, atlas, and temporal graph endpoints for the dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.api.deps import get_atlas_service, get_manager
from engram.retrieval.atlas_surface import (
    build_api_atlas_history_surface,
    build_api_atlas_json_response,
    build_api_atlas_region_surface,
    build_api_atlas_surface,
)
from engram.retrieval.graph_state import (
    build_api_graph_neighborhood_surface,
    build_api_temporal_graph_surface,
)
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/graph", tags=["graph"])


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
    result = await build_api_atlas_surface(
        get_atlas_service(),
        group_id=tenant.group_id,
        refresh=refresh,
        snapshot_id=snapshot_id,
    )
    return build_api_atlas_json_response(result, logger)


@router.get("/atlas/history")
async def get_atlas_history(
    request: Request,
    limit: int = Query(24, ge=1, le=120, description="Max snapshots to return"),
) -> JSONResponse:
    """Return atlas snapshot history for timeline scrubbing."""
    tenant = get_tenant(request)
    payload = await build_api_atlas_history_surface(
        get_atlas_service(),
        group_id=tenant.group_id,
        limit=limit,
    )
    return JSONResponse(content=payload)


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
    result = await build_api_atlas_region_surface(
        get_atlas_service(),
        group_id=tenant.group_id,
        region_id=region_id,
        refresh=refresh,
        snapshot_id=snapshot_id,
    )
    return build_api_atlas_json_response(result, logger)


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
    result = await build_api_graph_neighborhood_surface(
        manager,
        group_id=group_id,
        center=center,
        depth=depth,
        max_nodes=max_nodes,
        min_activation=min_activation,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)


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

    result = await build_api_temporal_graph_surface(
        manager,
        group_id=group_id,
        center=center,
        at=at,
        depth=depth,
        max_nodes=max_nodes,
    )
    return JSONResponse(status_code=result.status_code, content=result.payload)
