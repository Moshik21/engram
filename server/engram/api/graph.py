"""Graph neighborhood and temporal graph endpoints for dashboard visualization."""

from __future__ import annotations

import time
from datetime import datetime

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.activation.engine import compute_activation
from engram.api.deps import get_manager
from engram.security.middleware import get_tenant

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("/neighborhood")
async def get_neighborhood(
    request: Request,
    center: str | None = Query(None, description="Entity ID to center on"),
    depth: int = Query(2, ge=1, le=5, description="Number of hops"),
    max_nodes: int = Query(200, ge=1, le=1000, description="Max nodes to return"),
    min_activation: float = Query(0.0, ge=0.0, le=1.0, description="Min activation filter"),
) -> JSONResponse:
    """Return a subgraph neighborhood centered on an entity."""
    tenant = get_tenant(request)
    group_id = tenant.group_id
    manager = get_manager()
    now = time.time()

    # Resolve center entity
    if not center:
        # Pick top-activated entity
        top = await manager._activation.get_top_activated(group_id=group_id, limit=1)
        if top:
            center = top[0][0]
        else:
            # Pick first entity from graph
            first = await manager._graph.find_entities(group_id=group_id, limit=1)
            if first:
                center = first[0].id
            else:
                return JSONResponse(
                    content={
                        "centerId": None,
                        "nodes": [],
                        "edges": [],
                        "truncated": False,
                        "totalInNeighborhood": 0,
                    }
                )

    # Fetch center entity
    center_entity = await manager._graph.get_entity(center, group_id)
    if not center_entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})

    # Fetch neighbors
    neighbor_pairs = await manager._graph.get_neighbors(center, hops=depth, group_id=group_id)

    # Collect unique entities and edges
    entities_map: dict[str, object] = {center: center_entity}
    edges_map: dict[str, object] = {}

    for entity, rel in neighbor_pairs:
        entities_map[entity.id] = entity
        edges_map[rel.id] = rel

    # Batch get activation states
    entity_ids = list(entities_map.keys())
    states = await manager._activation.batch_get(entity_ids)

    # Build node list with activation
    nodes = []
    for eid, entity in entities_map.items():
        state = states.get(eid)
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
            # Fallback to entity-level fields (e.g. seeded data)
            activation_current = getattr(entity, "activation_current", 0.0) or 0.0
            access_count = getattr(entity, "access_count", 0) or 0
            la = getattr(entity, "last_accessed", None)
            if la:
                last_accessed = la.isoformat() if hasattr(la, "isoformat") else str(la)

        nodes.append(
            {
                "id": entity.id,
                "name": entity.name,
                "entityType": entity.entity_type,
                "summary": entity.summary,
                "activationCurrent": round(activation_current, 4),
                "accessCount": access_count,
                "lastAccessed": last_accessed,
                "createdAt": entity.created_at.isoformat() if entity.created_at else None,
                "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
            }
        )

    total_in_neighborhood = len(nodes)

    # Filter by min_activation
    if min_activation > 0.0:
        nodes = [n for n in nodes if n["activationCurrent"] >= min_activation]

    # Truncation
    truncated = False
    if len(nodes) > max_nodes:
        nodes.sort(key=lambda n: n["activationCurrent"], reverse=True)
        nodes = nodes[:max_nodes]
        truncated = True

    # Filter edges to only include those where both endpoints remain
    remaining_ids = {n["id"] for n in nodes}
    edges = []
    for rel in edges_map.values():
        if rel.source_id in remaining_ids and rel.target_id in remaining_ids:
            edges.append(
                {
                    "id": rel.id,
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "predicate": rel.predicate,
                    "weight": rel.weight,
                    "validFrom": rel.valid_from.isoformat() if rel.valid_from else None,
                    "validTo": rel.valid_to.isoformat() if rel.valid_to else None,
                    "createdAt": rel.created_at.isoformat() if rel.created_at else None,
                }
            )

    return JSONResponse(
        content={
            "centerId": center,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
            "totalInNeighborhood": total_in_neighborhood,
        }
    )


@router.get("/at")
async def get_graph_at(
    request: Request,
    center: str = Query(..., description="Entity ID to center on"),
    at: str = Query(..., description="ISO 8601 timestamp for point-in-time query"),
    depth: int = Query(2, ge=1, le=5, description="BFS hops"),
    max_nodes: int = Query(200, ge=1, le=1000, description="Max nodes"),
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

    center_entity = await manager._graph.get_entity(center, group_id)
    if not center_entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})

    # BFS with time-filtered edges
    visited: set[str] = {center}
    frontier: set[str] = {center}
    entities_map: dict[str, object] = {center: center_entity}
    edges_list: list[dict] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for eid in frontier:
            rels = await manager._graph.get_relationships_at(eid, at_time, group_id=group_id)
            for r in rels:
                other = r.target_id if r.source_id == eid else r.source_id
                edges_list.append(
                    {
                        "id": r.id,
                        "source": r.source_id,
                        "target": r.target_id,
                        "predicate": r.predicate,
                        "weight": r.weight,
                        "validFrom": r.valid_from.isoformat() if r.valid_from else None,
                        "validTo": r.valid_to.isoformat() if r.valid_to else None,
                        "createdAt": r.created_at.isoformat() if r.created_at else None,
                    }
                )
                if other not in visited:
                    visited.add(other)
                    next_frontier.add(other)
                    ent = await manager._graph.get_entity(other, group_id)
                    if ent:
                        entities_map[other] = ent
        frontier = next_frontier
        if not frontier:
            break

    # Build nodes
    nodes = []
    for eid, entity in list(entities_map.items())[:max_nodes]:
        nodes.append(
            {
                "id": entity.id,
                "name": entity.name,
                "entityType": entity.entity_type,
                "summary": entity.summary,
                "createdAt": entity.created_at.isoformat() if entity.created_at else None,
            }
        )

    # Deduplicate edges and filter to remaining node IDs
    remaining_ids = {n["id"] for n in nodes}
    seen_edge_ids: set[str] = set()
    edges = []
    for e in edges_list:
        in_scope = e["source"] in remaining_ids and e["target"] in remaining_ids
        if e["id"] not in seen_edge_ids and in_scope:
            seen_edge_ids.add(e["id"])
            edges.append(e)

    return JSONResponse(
        content={
            "centerId": center,
            "at": at,
            "nodes": nodes,
            "edges": edges,
            "truncated": len(entities_map) > max_nodes,
            "totalInNeighborhood": len(entities_map),
        }
    )
