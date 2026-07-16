"""REST API for Loop Steward status (dashboard + operators)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/loop", tags=["loop"])


class LoopApplyBody(BaseModel):
    adjustment: dict[str, Any] = Field(default_factory=dict)
    skip_continuity_check: bool = True


@router.get("/status")
async def get_loop_status(request: Request) -> dict[str, Any]:
    """Active LoopAdjustment or none. Dual-reads Helix sidecar + file."""
    from engram.loop_adjustment import (
        load_active_adjustment_async,
        remaining_ttl_seconds,
        status_payload,
    )

    group_id = str(getattr(request.app.state, "default_group_id", None) or "default")
    # Prefer app-held consolidation store (Helix native sidecar path)
    store = getattr(request.app.state, "consolidation_store", None)
    if store is None:
        state = getattr(request.app, "state", None)
        store = getattr(state, "consolidation_store", None) if state else None
    # main.py uses _app_state dict pattern via dependency — fall back module
    try:
        from engram.main import _app_state

        store = store or _app_state.get("consolidation_store")
        group_id = str(_app_state.get("default_group_id") or group_id)
    except Exception:
        pass

    if store is not None:
        adj = await load_active_adjustment_async(group_id, graph_store=store, clear_if_expired=True)
        if adj is not None:
            return {
                "group_id": group_id,
                "active": True,
                "adjustment": adj.to_dict(),
                "remaining_ttl_seconds": round(remaining_ttl_seconds(adj), 1),
                "regime": adj.regime,
                "reason": adj.reason,
                "expires_at": adj.expires_at,
                "store": "graph",
            }
    return status_payload(group_id)


@router.post("/apply")
async def post_loop_apply(body: LoopApplyBody, request: Request) -> dict[str, Any]:
    from engram.config import ActivationConfig
    from engram.loop_adjustment import (
        LoopAdjustment,
        clamp_loop_adjustment,
        hard_caps_from_config,
        save_active_adjustment_async,
        stamp_applied,
        status_payload,
    )

    group_id = "default"
    store = None
    try:
        from engram.main import _app_state

        store = _app_state.get("consolidation_store")
        group_id = str(_app_state.get("default_group_id") or "default")
    except Exception:
        pass

    adj = LoopAdjustment.from_mapping(body.adjustment or {})
    adj.group_id = group_id
    result = clamp_loop_adjustment(adj, hard_caps=hard_caps_from_config(ActivationConfig()))
    if result.rejected:
        return {
            "status": "error",
            "error": result.reject_reason,
            "warnings": list(result.warnings),
        }
    stamped = stamp_applied(result.adjustment)
    await save_active_adjustment_async(stamped, graph_store=store)
    return {
        "status": "ok",
        "applied": True,
        "warnings": list(result.warnings),
        "adjustment": stamped.to_dict(),
        "status_payload": status_payload(group_id),
    }


@router.delete("/status")
@router.post("/clear")
async def clear_loop(request: Request) -> dict[str, Any]:
    from engram.loop_adjustment import clear_active_adjustment_async

    group_id = "default"
    store = None
    try:
        from engram.main import _app_state

        store = _app_state.get("consolidation_store")
        group_id = str(_app_state.get("default_group_id") or "default")
    except Exception:
        pass
    cleared = await clear_active_adjustment_async(
        group_id, cleared_by="api:loop_clear", graph_store=store
    )
    return {"status": "ok", "cleared": cleared, "group_id": group_id}
