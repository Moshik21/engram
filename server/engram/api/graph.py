"""Graph neighborhood, atlas, and temporal graph endpoints for the dashboard."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from engram.activation.engine import compute_activation
from engram.api.deps import get_atlas_service, get_manager
from engram.config import ActivationConfig
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


def _build_node(entity: Any, state: Any, now: float, cfg: ActivationConfig) -> dict[str, Any]:
    """Build a JSON-serialisable node dict from an entity + activation state."""
    activation_current = 0.0
    access_count = 0
    last_accessed = None
    if state and state.access_history:
        activation_current = compute_activation(state.access_history, now, cfg)
        access_count = state.access_count
        if state.last_accessed:
            from datetime import datetime as _dt
            from datetime import timezone

            last_accessed = _dt.fromtimestamp(state.last_accessed, tz=timezone.utc).isoformat()
    else:
        activation_current = getattr(entity, "activation_current", 0.0) or 0.0
        access_count = getattr(entity, "access_count", 0) or 0
        la = getattr(entity, "last_accessed", None)
        if la:
            last_accessed = la.isoformat() if hasattr(la, "isoformat") else str(la)

    return {
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


def _build_edge(rel: Any) -> dict[str, Any]:
    """Build a JSON-serialisable edge dict from a Relationship."""
    return {
        "id": rel.id,
        "source": rel.source_id,
        "target": rel.target_id,
        "predicate": rel.predicate,
        "weight": rel.weight,
        "validFrom": rel.valid_from.isoformat() if rel.valid_from else None,
        "validTo": rel.valid_to.isoformat() if rel.valid_to else None,
        "createdAt": rel.created_at.isoformat() if rel.created_at else None,
    }


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
    now = time.time()

    # ── Full-graph mode: no center specified → load everything ──
    if not center:
        all_entities = await manager._graph.find_entities(group_id=group_id, limit=max_nodes)
        if not all_entities:
            representation = _build_representation(
                scope="neighborhood",
                layout="force",
                represented_entity_count=0,
                represented_edge_count=0,
                displayed_node_count=0,
                displayed_edge_count=0,
                truncated=False,
            )
            return JSONResponse(
                content={
                    "centerId": None,
                    "nodes": [],
                    "edges": [],
                    "representation": representation,
                    "truncated": False,
                    "totalInNeighborhood": 0,
                }
            )

        entity_ids = {e.id for e in all_entities}
        states = await manager._activation.batch_get(list(entity_ids))

        nodes: list[dict[str, Any]] = []
        for entity in all_entities:
            node = _build_node(entity, states.get(entity.id), now, manager._cfg)
            if min_activation > 0.0 and node["activationCurrent"] < min_activation:
                continue
            nodes.append(node)

        # Pick the top-activated entity as the center
        best = max(nodes, key=lambda n: n["activationCurrent"]) if nodes else None
        resolved_center = best["id"] if best else None

        total = len(nodes)
        truncated = False
        if len(nodes) > max_nodes:
            nodes.sort(key=lambda n: n["activationCurrent"], reverse=True)
            nodes = nodes[:max_nodes]
            truncated = True

        # Bulk-fetch all edges between the included entities
        remaining_ids = {n["id"] for n in nodes}
        all_rels = await manager._graph.get_all_edges(
            group_id=group_id, entity_ids=remaining_ids, limit=max_nodes * 5
        )
        edges = [_build_edge(r) for r in all_rels]

        return JSONResponse(
            content={
                "centerId": resolved_center,
                "nodes": nodes,
                "edges": edges,
                "representation": _build_representation(
                    scope="neighborhood",
                    layout="force",
                    represented_entity_count=total,
                    represented_edge_count=len(all_rels),
                    displayed_node_count=len(nodes),
                    displayed_edge_count=len(edges),
                    truncated=truncated,
                ),
                "truncated": truncated,
                "totalInNeighborhood": total,
            }
        )

    # ── Centered BFS mode: specific center → neighborhood traversal ──
    center_entity = await manager._graph.get_entity(center, group_id)
    if not center_entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})

    neighbor_pairs = await manager._graph.get_neighbors(
        center,
        hops=depth,
        group_id=group_id,
        max_results=max_nodes * 3,
    )

    entities_map: dict[str, Any] = {center: center_entity}
    edges_map: dict[str, Any] = {}

    for entity, rel in neighbor_pairs:
        entities_map[entity.id] = entity
        edges_map[rel.id] = rel

    neighborhood_entity_ids = list(entities_map)
    states = await manager._activation.batch_get(neighborhood_entity_ids)

    neighborhood_nodes: list[dict[str, Any]] = []
    for eid, entity in entities_map.items():
        neighborhood_nodes.append(_build_node(entity, states.get(eid), now, manager._cfg))

    total_in_neighborhood = len(neighborhood_nodes)

    if min_activation > 0.0:
        neighborhood_nodes = [
            n for n in neighborhood_nodes if n["activationCurrent"] >= min_activation
        ]

    truncated = False
    if len(neighborhood_nodes) > max_nodes:
        neighborhood_nodes.sort(key=lambda n: n["activationCurrent"], reverse=True)
        neighborhood_nodes = neighborhood_nodes[:max_nodes]
        truncated = True

    remaining_ids = {n["id"] for n in neighborhood_nodes}
    edges = [
        _build_edge(r)
        for r in edges_map.values()
        if r.source_id in remaining_ids and r.target_id in remaining_ids
    ]

    return JSONResponse(
        content={
            "centerId": center,
            "nodes": neighborhood_nodes,
            "edges": edges,
            "representation": _build_representation(
                scope="neighborhood",
                layout="force",
                represented_entity_count=total_in_neighborhood,
                represented_edge_count=len(edges_map),
                displayed_node_count=len(neighborhood_nodes),
                displayed_edge_count=len(edges),
                truncated=truncated,
            ),
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

    center_entity = await manager._graph.get_entity(center, group_id)
    if not center_entity:
        return JSONResponse(status_code=404, content={"detail": f"Entity '{center}' not found"})

    # BFS with time-filtered edges (batch entity fetches per hop)
    visited: set[str] = {center}
    frontier: set[str] = {center}
    entities_map: dict[str, Any] = {center: center_entity}
    edges_list: list[dict] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for eid in frontier:
            rels = await manager._graph.get_relationships_at(eid, at_time, group_id=group_id)
            for r in rels:
                other = r.target_id if r.source_id == eid else r.source_id
                edges_list.append(_build_edge(r))
                if other not in visited:
                    visited.add(other)
                    next_frontier.add(other)
        # Batch-fetch all newly discovered entities for this hop
        if next_frontier:
            batch = await manager._graph.batch_get_entities(list(next_frontier), group_id)
            entities_map.update(batch)
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
                "activationCurrent": 0.0,
                "accessCount": 0,
                "lastAccessed": None,
                "createdAt": entity.created_at.isoformat() if entity.created_at else None,
                "updatedAt": entity.updated_at.isoformat() if entity.updated_at else None,
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
            "representation": _build_representation(
                scope="temporal",
                layout="force",
                represented_entity_count=len(entities_map),
                represented_edge_count=len(edges_list),
                displayed_node_count=len(nodes),
                displayed_edge_count=len(edges),
                truncated=len(entities_map) > max_nodes,
            ),
            "truncated": len(entities_map) > max_nodes,
            "totalInNeighborhood": len(entities_map),
        }
    )
