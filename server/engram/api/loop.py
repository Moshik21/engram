"""REST API for Loop Steward status (dashboard + operators)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/loop", tags=["loop"])


class LoopApplyBody(BaseModel):
    # NOTE: apply is unconditional on this surface; the continuity gate is a
    # CLI-only rail (engram loop apply). A former skip_continuity_check field
    # here was accepted and silently ignored — removed rather than pretend.
    adjustment: dict[str, Any] = Field(default_factory=dict)


@router.get("/status")
async def get_loop_status(request: Request) -> dict[str, Any]:
    """Active LoopAdjustment or none.

    FILE-FIRST: every runtime consumer (scheduler overlay, mop budgets,
    worker routing) reads the file via load_active_adjustment, so status must
    report the same source of truth. The Helix sidecar is an audit mirror
    only — preferring it here showed adjustments the runtime wasn't honoring
    whenever the CLI (file-only writer) and API (dual writer) diverged.
    """
    from engram.loop_adjustment import status_payload

    group_id = "default"
    try:
        from engram.main import _app_state

        group_id = str(_app_state.get("default_group_id") or group_id)
    except Exception:
        pass
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
    # Live config (env/profile-resolved), not a default-constructed
    # ActivationConfig — operators who raised drain caps via env expect the
    # clamp ceiling to follow.
    try:
        live_cfg = _app_state.get("config").activation  # type: ignore[union-attr]
    except Exception:
        live_cfg = ActivationConfig()
    result = clamp_loop_adjustment(adj, hard_caps=hard_caps_from_config(live_cfg))
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
