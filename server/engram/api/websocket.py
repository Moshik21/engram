"""WebSocket endpoint for real-time dashboard updates."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket

from engram.api.deps import get_manager, get_notification_surface_service
from engram.api.websocket_auth import (
    close_dashboard_websocket_auth_failure,
    resolve_dashboard_websocket_tenant,
)
from engram.api.websocket_runtime import run_dashboard_websocket_session
from engram.events.bus import get_event_bus

router = APIRouter()


@router.websocket("/ws/dashboard")
async def dashboard_ws(websocket: WebSocket) -> None:
    """WebSocket for dashboard real-time events.

    Subscribes to the authenticated tenant group's event bus and forwards events.
    Commands:
      - {"type": "ping"} -> pong
      - {"type": "command", "command": "resync", "lastSeq": N} -> replay missed events
      - {"type": "command", "command": "subscribe.activation_monitor", "interval_ms": N}
      - {"type": "command", "command": "unsubscribe.activation_monitor"}
    """
    try:
        tenant = await resolve_dashboard_websocket_tenant(websocket)
    except ValueError:
        await close_dashboard_websocket_auth_failure(websocket)
        return

    await websocket.accept()
    group_id = tenant.group_id
    try:
        await run_dashboard_websocket_session(
            websocket,
            bus=get_event_bus(),
            group_id=group_id,
            manager=get_manager(),
            notification_surface=get_notification_surface_service(),
        )
    except asyncio.CancelledError:
        pass
