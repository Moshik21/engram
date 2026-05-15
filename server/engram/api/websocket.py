"""WebSocket endpoint for real-time dashboard updates."""

from __future__ import annotations

import asyncio
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engram.api.deps import get_config, get_manager, get_notification_surface_service
from engram.events.bus import get_event_bus

logger = logging.getLogger(__name__)

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
    # Authenticate BEFORE accepting the connection
    from engram.config import AuthConfig
    from engram.security.middleware import resolve_tenant_from_scope

    try:
        try:
            auth_config = get_config().auth
        except RuntimeError:
            auth_config = AuthConfig()
        # Try headers first, then fall back to query param token (for browser WS)
        try:
            tenant = await resolve_tenant_from_scope(websocket.headers, auth_config)
        except ValueError:
            token = websocket.query_params.get("token", "")
            if token:
                # Build fake headers with the token for reuse of existing auth logic
                from starlette.datastructures import Headers

                fake_headers = Headers({"authorization": f"Bearer {token}"})
                tenant = await resolve_tenant_from_scope(fake_headers, auth_config)
            else:
                raise
    except (ValueError, Exception):
        await websocket.close(code=4001, reason="Authentication required")
        return

    await websocket.accept()
    group_id = tenant.group_id
    manager = get_manager()
    notification_surface = get_notification_surface_service()
    bus = get_event_bus()
    queue = bus.subscribe(group_id)
    activation_task: asyncio.Task | None = None

    async def forward_events() -> None:
        """Read events from the bus queue and send to websocket.

        Flattens the ``payload`` dict into top-level keys so the frontend
        can read fields like ``episodeId``, ``status``, ``episode`` directly
        instead of through ``data.payload.episodeId``.
        """
        try:
            while True:
                event = await queue.get()
                flat = {k: v for k, v in event.items() if k != "payload"}
                if event.get("payload"):
                    flat.update(event["payload"])
                await websocket.send_json(flat)
        except (WebSocketDisconnect, Exception):
            pass

    async def activation_snapshot_loop(interval_ms: int) -> None:
        """Periodically compute and send activation snapshots."""
        interval_s = max(interval_ms / 1000.0, 0.5)  # minimum 500ms

        try:
            while True:
                await asyncio.sleep(interval_s)
                try:
                    snapshot = await manager.get_activation_snapshot(
                        group_id=group_id,
                        limit=20,
                    )
                    await websocket.send_json(
                        {
                            "type": "activation.snapshot",
                            "payload": {"topActivated": snapshot["topActivated"]},
                        }
                    )
                except Exception as e:
                    logger.debug("Activation snapshot error: %s", e)
        except asyncio.CancelledError:
            pass
        except (WebSocketDisconnect, Exception):
            pass

    async def receive_commands() -> None:
        """Read commands from the websocket client."""
        nonlocal activation_task
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "")

                if msg_type == "ping":
                    await websocket.send_json(
                        {
                            "type": "pong",
                            "timestamp": time.time(),
                        }
                    )

                elif msg_type == "command":
                    command = data.get("command", "")

                    if command == "resync":
                        last_seq = data.get("lastSeq", 0)
                        events, is_full = bus.get_events_since(group_id, last_seq)
                        await websocket.send_json(
                            {
                                "type": "resync",
                                "events": events,
                                "isFull": is_full,
                            }
                        )

                    elif command == "subscribe.activation_monitor":
                        interval_ms = data.get("interval_ms", 2000)
                        # Cancel existing subscription if any
                        if activation_task and not activation_task.done():
                            activation_task.cancel()
                            try:
                                await activation_task
                            except asyncio.CancelledError:
                                pass
                        activation_task = asyncio.create_task(activation_snapshot_loop(interval_ms))

                    elif command == "unsubscribe.activation_monitor":
                        if activation_task and not activation_task.done():
                            activation_task.cancel()
                            try:
                                await activation_task
                            except asyncio.CancelledError:
                                pass
                            activation_task = None

                    elif command == "dismiss_notification":
                        nid = data.get("id")
                        if nid and notification_surface:
                            notification_surface.dismiss_notifications(
                                group_id=group_id,
                                ids=[str(nid)],
                            )

        except (WebSocketDisconnect, Exception):
            pass

    try:
        await asyncio.gather(forward_events(), receive_commands())
    except Exception:
        pass
    finally:
        # Cleanup activation task
        if activation_task and not activation_task.done():
            activation_task.cancel()
            try:
                await activation_task
            except asyncio.CancelledError:
                pass
        bus.unsubscribe(group_id, queue)
