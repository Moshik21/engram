"""Dashboard WebSocket session runtime helpers."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from engram.api.websocket_surface import (
    build_dashboard_activation_snapshot_message,
    build_dashboard_pong_surface,
    build_dashboard_resync_surface,
    dismiss_dashboard_notification_command,
    flatten_dashboard_event,
)
from engram.events.bus import EventBus
from engram.graph_manager import GraphManager
from engram.notifications.surface import NotificationSurfaceService
from engram.retrieval.graph_state import build_api_activation_snapshot_surface

logger = logging.getLogger(__name__)


async def _forward_dashboard_events(
    websocket: WebSocket,
    queue: asyncio.Queue,
) -> None:
    """Forward subscribed brain events to the connected dashboard."""
    try:
        while True:
            event = await queue.get()
            await websocket.send_json(flatten_dashboard_event(event))
    except (WebSocketDisconnect, Exception):
        pass


async def _run_dashboard_activation_snapshots(
    websocket: WebSocket,
    manager: GraphManager,
    *,
    group_id: str,
    interval_ms: Any,
) -> None:
    """Periodically send activation snapshots for the connected brain."""
    try:
        interval_s = max(float(interval_ms) / 1000.0, 0.5)
    except (TypeError, ValueError):
        interval_s = 2.0

    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                snapshot = await build_api_activation_snapshot_surface(
                    manager,
                    group_id=group_id,
                    limit=20,
                )
                await websocket.send_json(
                    build_dashboard_activation_snapshot_message(snapshot)
                )
            except Exception as exc:
                logger.debug("Activation snapshot error: %s", exc)
    except asyncio.CancelledError:
        pass
    except (WebSocketDisconnect, Exception):
        pass


async def _cancel_dashboard_task(task: asyncio.Task | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def _receive_dashboard_commands(
    websocket: WebSocket,
    *,
    bus: EventBus,
    group_id: str,
    manager: GraphManager,
    notification_surface: NotificationSurfaceService | None,
) -> None:
    """Handle dashboard commands for the connected brain."""
    activation_task: asyncio.Task | None = None
    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json(build_dashboard_pong_surface())

            elif msg_type == "command":
                command = data.get("command", "")

                if command == "resync":
                    await websocket.send_json(
                        build_dashboard_resync_surface(
                            bus,
                            group_id=group_id,
                            last_seq=data.get("lastSeq", 0),
                        )
                    )

                elif command == "subscribe.activation_monitor":
                    await _cancel_dashboard_task(activation_task)
                    activation_task = asyncio.create_task(
                        _run_dashboard_activation_snapshots(
                            websocket,
                            manager,
                            group_id=group_id,
                            interval_ms=data.get("interval_ms", 2000),
                        )
                    )

                elif command == "unsubscribe.activation_monitor":
                    await _cancel_dashboard_task(activation_task)
                    activation_task = None

                elif command == "dismiss_notification":
                    dismiss_dashboard_notification_command(
                        notification_surface,
                        group_id=group_id,
                        notification_id=data.get("id"),
                    )

    except (WebSocketDisconnect, Exception):
        pass
    finally:
        await _cancel_dashboard_task(activation_task)


async def run_dashboard_websocket_session(
    websocket: WebSocket,
    *,
    bus: EventBus,
    group_id: str,
    manager: GraphManager,
    notification_surface: NotificationSurfaceService | None,
) -> None:
    """Run the dashboard WebSocket event and command loops."""
    queue = bus.subscribe(group_id)
    forward_task = asyncio.create_task(_forward_dashboard_events(websocket, queue))
    command_task = asyncio.create_task(
        _receive_dashboard_commands(
            websocket,
            bus=bus,
            group_id=group_id,
            manager=manager,
            notification_surface=notification_surface,
        )
    )
    tasks = {forward_task, command_task}

    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        bus.unsubscribe(group_id, queue)
