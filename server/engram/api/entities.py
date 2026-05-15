"""Entity CRUD and search endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engram.api.deps import get_manager
from engram.retrieval.entity_surface import (
    build_api_entity_delete_surface,
    build_api_entity_detail_surface,
    build_api_entity_update_surface,
    entity_not_found_payload,
)
from engram.retrieval.lookup import build_api_entity_search_surface
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/entities", tags=["entities"])


class EntityPatchBody(BaseModel):
    """Request body for PATCH /api/entities/{entity_id}."""

    name: str | None = None
    summary: str | None = None


@router.get("/search")
async def search_entities(
    request: Request,
    q: str | None = Query(None, description="Search query"),
    type: str | None = Query(None, description="Filter by entity type"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
) -> JSONResponse:
    """Search entities by name and/or type."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    payload = await build_api_entity_search_surface(
        manager,
        group_id=group_id,
        name=q,
        entity_type=type,
        limit=limit,
    )
    return JSONResponse(content=payload)


@router.get("/{entity_id}")
async def get_entity(request: Request, entity_id: str) -> JSONResponse:
    """Get entity detail with facts."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    detail = await build_api_entity_detail_surface(
        manager,
        group_id=group_id,
        entity_id=entity_id,
    )
    if detail is None:
        return JSONResponse(status_code=404, content=entity_not_found_payload(entity_id))
    return JSONResponse(content=detail)


@router.get("/{entity_id}/neighbors")
async def get_entity_neighbors(
    request: Request,
    entity_id: str,
    depth: int = Query(2, ge=1, le=5),
    max_nodes: int = Query(2000, ge=1, le=10000),
    min_activation: float = Query(0.0, ge=0.0, le=1.0),
) -> JSONResponse:
    """Get neighborhood subgraph centered on this entity."""
    from engram.api.graph import get_neighborhood

    # Delegate to the graph neighborhood endpoint with this entity as center
    return await get_neighborhood(
        request=request,
        center=entity_id,
        depth=depth,
        max_nodes=max_nodes,
        min_activation=min_activation,
    )


@router.patch("/{entity_id}")
async def patch_entity(request: Request, entity_id: str, body: EntityPatchBody) -> JSONResponse:
    """Update entity name and/or summary."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    updated = await build_api_entity_update_surface(
        manager,
        group_id=group_id,
        entity_id=entity_id,
        name=body.name,
        summary=body.summary,
    )
    if updated is None:
        return JSONResponse(status_code=404, content=entity_not_found_payload(entity_id))
    return JSONResponse(content=updated)


@router.delete("/{entity_id}")
async def delete_entity(request: Request, entity_id: str) -> JSONResponse:
    """Soft-delete an entity."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    result = await build_api_entity_delete_surface(
        manager,
        group_id=group_id,
        entity_id=entity_id,
    )
    if result is None:
        return JSONResponse(status_code=404, content=entity_not_found_payload(entity_id))
    return JSONResponse(content=result)
