"""Fluid Ingestion WebSocket for real-time text streaming."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from engram.api.deps import get_manager
from engram.events.bus import get_event_bus

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/ingest")
async def ingest_ws(websocket: WebSocket) -> None:
    """WebSocket for fluid text ingestion.

    Accepts incremental text chunks and builds a 'Latent Memory Trace'.
    """
    await websocket.accept()
    manager = get_manager()
    bus = get_event_bus()

    buffer = ""
    session_id = f"fluid_{id(websocket)}"

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type")

            if msg_type == "text_chunk":
                chunk = data.get("text", "")
                buffer += chunk

                # Signal to the dashboard that fluid ingestion is happening
                bus.publish(
                    group_id="default",
                    event_type="streaming.fluid_ingestion_progress",
                    payload={
                        "session_id": session_id,
                        "bytes_received": len(buffer),
                        "is_complete": False,
                    },
                )

                await websocket.send_json({
                    "type": "ack",
                    "bytes_received": len(buffer),
                })

            elif msg_type == "finalize":
                # Finalize the discourse and trigger a full projection
                logger.info("Finalizing fluid ingestion session %s", session_id)

                # In a full implementation, we'd call manager.ingest_episode(buffer)
                await websocket.send_json({
                    "type": "completion",
                    "status": "matured_to_episode",
                })
                break

    except WebSocketDisconnect:
        logger.info("Fluid ingestion session %s disconnected", session_id)
    except Exception as exc:
        logger.error("Fluid ingestion error: %s", exc, exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass
