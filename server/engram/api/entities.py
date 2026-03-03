"""Entity CRUD and search endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engram.activation.engine import compute_activation
from engram.api.deps import get_manager
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

    results = await manager.search_entities(
        group_id=group_id,
        name=q,
        entity_type=type,
        limit=limit,
    )

    # Convert to camelCase
    items = []
    for r in results:
        items.append(
            {
                "id": r["id"],
                "name": r["name"],
                "entityType": r["entity_type"],
                "summary": r["summary"],
                "activationCurrent": r["activation_score"],
                "accessCount": r["access_count"],
                "createdAt": r["created_at"],
                "updatedAt": r["updated_at"],
            }
        )

    return JSONResponse(content={"items": items, "total": len(items)})


@router.get("/{entity_id}")
async def get_entity(request: Request, entity_id: str) -> JSONResponse:
    """Get entity detail with facts."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    now = time.time()

    entity = await manager._graph.get_entity(entity_id, group_id)
    if not entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{entity_id}' not found"})

    # Get activation
    state = await manager._activation.get_activation(entity_id)
    activation_current = 0.0
    access_count = 0
    last_accessed = None
    if state and state.access_history:
        activation_current = compute_activation(state.access_history, now, manager._cfg)
        access_count = state.access_count
        if state.last_accessed:
            from datetime import datetime, timezone

            last_accessed = datetime.fromtimestamp(
                state.last_accessed, tz=timezone.utc
            ).isoformat()
    else:
        # Fallback to entity-level fields
        activation_current = getattr(entity, "activation_current", 0.0) or 0.0
        access_count = getattr(entity, "access_count", 0) or 0
        la = getattr(entity, "last_accessed", None)
        if la:
            last_accessed = la.isoformat() if hasattr(la, "isoformat") else str(la)

    # Get relationships as facts
    rels = await manager._graph.get_relationships(entity_id, active_only=True, group_id=group_id)
    facts = []
    for r in rels:
        if r.source_id == entity_id:
            direction = "outgoing"
            other_id = r.target_id
        else:
            direction = "incoming"
            other_id = r.source_id

        other_entity = await manager._graph.get_entity(other_id, group_id)
        other_info = (
            {
                "id": other_entity.id,
                "name": other_entity.name,
                "entityType": other_entity.entity_type,
            }
            if other_entity
            else {"id": other_id, "name": other_id, "entityType": "Unknown"}
        )

        facts.append(
            {
                "id": r.id,
                "predicate": r.predicate,
                "direction": direction,
                "other": other_info,
                "weight": r.weight,
                "validFrom": r.valid_from.isoformat() if r.valid_from else None,
                "validTo": r.valid_to.isoformat() if r.valid_to else None,
                "createdAt": r.created_at.isoformat() if r.created_at else None,
            }
        )

    return JSONResponse(
        content={
            "id": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "summary": entity.summary,
            "activationCurrent": round(activation_current, 4),
            "accessCount": access_count,
            "lastAccessed": last_accessed,
            "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
            "facts": facts,
        }
    )


@router.get("/{entity_id}/neighbors")
async def get_entity_neighbors(
    request: Request,
    entity_id: str,
    depth: int = Query(2, ge=1, le=5),
    max_nodes: int = Query(200, ge=1, le=1000),
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
async def patch_entity(
    request: Request, entity_id: str, body: EntityPatchBody
) -> JSONResponse:
    """Update entity name and/or summary."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    entity = await manager._graph.get_entity(entity_id, group_id)
    if not entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{entity_id}' not found"})

    updates = {}
    if body.name is not None:
        updates["name"] = body.name
    if body.summary is not None:
        updates["summary"] = body.summary

    if updates:
        await manager._graph.update_entity(entity_id, updates, group_id=group_id)

    # Fetch updated entity
    updated = await manager._graph.get_entity(entity_id, group_id)
    return JSONResponse(
        content={
            "id": updated.id,
            "name": updated.name,
            "entityType": updated.entity_type,
            "summary": updated.summary,
            "createdAt": updated.created_at.isoformat() if updated.created_at else None,
            "updatedAt": updated.updated_at.isoformat() if updated.updated_at else None,
        }
    )


@router.delete("/{entity_id}")
async def delete_entity(request: Request, entity_id: str) -> JSONResponse:
    """Soft-delete an entity."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()

    entity = await manager._graph.get_entity(entity_id, group_id)
    if not entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{entity_id}' not found"})

    await manager._graph.delete_entity(entity_id, soft=True, group_id=group_id)
    await manager._activation.clear_activation(entity_id)

    return JSONResponse(
        content={"status": "deleted", "id": entity_id, "name": entity.name}
    )
