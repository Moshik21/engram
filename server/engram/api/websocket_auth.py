"""Dashboard WebSocket authentication helpers."""

from __future__ import annotations

from typing import Any

from fastapi import WebSocket
from starlette.datastructures import Headers

from engram.api.deps import get_config
from engram.config import AuthConfig
from engram.models.tenant import TenantContext
from engram.security.middleware import resolve_tenant_from_scope


async def resolve_dashboard_websocket_tenant(websocket: WebSocket) -> TenantContext:
    """Resolve the authenticated tenant for a dashboard WebSocket."""
    auth_config = _dashboard_websocket_auth_config()
    try:
        return await resolve_tenant_from_scope(websocket.headers, auth_config)
    except ValueError:
        token = _dashboard_websocket_query_token(websocket)
        if not token:
            raise
        return await resolve_tenant_from_scope(
            Headers({"authorization": f"Bearer {token}"}),
            auth_config,
        )


async def close_dashboard_websocket_auth_failure(websocket: WebSocket) -> None:
    """Close an unauthenticated dashboard WebSocket before accepting it."""
    await websocket.close(code=4001, reason="Authentication required")


def _dashboard_websocket_auth_config() -> AuthConfig:
    try:
        return get_config().auth
    except RuntimeError:
        return AuthConfig()


def _dashboard_websocket_query_token(websocket: Any) -> str:
    return str(websocket.query_params.get("token", ""))
