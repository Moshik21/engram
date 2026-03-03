"""Activation monitor API endpoints."""

from __future__ import annotations

import logging
import time
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from engram.activation.engine import compute_activation
from engram.security.middleware import get_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/activation", tags=["activation"])


def _get_deps(request: Request) -> tuple:
    """Extract dependencies from app state."""
    from engram.main import _app_state
    activation_store = _app_state.get("activation_store")
    graph_store = _app_state.get("graph_store")
    config = _app_state.get("config")
    if not activation_store or not graph_store or not config:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return activation_store, graph_store, config


@router.get("/snapshot")
async def get_activation_snapshot(request: Request, limit: int = 50):
    """Top activated entities with scores and access metadata."""
    activation_store, graph_store, config = _get_deps(request)
    tenant = get_tenant(request)
    group_id = tenant.group_id
    now = time.time()

    top = await activation_store.get_top_activated(group_id=group_id, limit=limit * 2)

    items = []
    for entity_id, state in top:
        entity = await graph_store.get_entity(entity_id, group_id)
        if not entity:
            continue

        activation = compute_activation(state.access_history, now, config.activation)
        items.append({
            "entityId": entity.id,
            "name": entity.name,
            "entityType": entity.entity_type,
            "currentActivation": round(activation, 4),
            "accessCount": state.access_count,
            "lastAccessedAt": (
                datetime.fromtimestamp(state.last_accessed).isoformat()
                if state.last_accessed
                else None
            ),
            "decayRate": config.activation.decay_exponent,
        })
        if len(items) >= limit:
            break

    # Sort by activation descending
    items.sort(key=lambda x: x["currentActivation"], reverse=True)
    return {"topActivated": items}


@router.get("/{entity_id}/curve")
async def get_activation_curve(
    request: Request,
    entity_id: str,
    hours: int = 24,
    points: int = 48,
):
    """Simulated ACT-R decay curve over past N hours."""
    activation_store, graph_store, config = _get_deps(request)
    tenant = get_tenant(request)
    group_id = tenant.group_id

    # Verify entity exists
    entity = await graph_store.get_entity(entity_id, group_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")

    state = await activation_store.get_activation(entity_id)
    access_history = state.access_history if state else []

    now = time.time()
    start_time = now - (hours * 3600)
    step = (now - start_time) / max(points - 1, 1)

    curve: list[dict] = []
    for i in range(points):
        t = start_time + (i * step)
        # Only use accesses that occurred before this point in time
        history_at_t = [ts for ts in access_history if ts <= t]
        if history_at_t:
            activation = compute_activation(history_at_t, t, config.activation)
        else:
            activation = 0.0
        curve.append({
            "timestamp": datetime.fromtimestamp(t).isoformat(),
            "activation": round(activation, 4),
        })

    # Identify access event timestamps within the window
    access_events = [
        datetime.fromtimestamp(ts).isoformat()
        for ts in access_history
        if start_time <= ts <= now
    ]

    d = config.activation.decay_exponent
    formula = f"B_i = ln(Σ t_j^{{-{d}}})"

    return {
        "entityId": entity_id,
        "entityName": entity.name,
        "curve": curve,
        "accessEvents": access_events,
        "formula": formula,
        "hours": hours,
        "points": points,
    }
