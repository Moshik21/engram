"""Tenant context middleware — ensures every request has a resolved TenantContext."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from engram.config import AuthConfig
from engram.models.tenant import TenantContext

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/metrics"}


class TenantContextMiddleware(BaseHTTPMiddleware):
    """Resolves auth credentials into a TenantContext.

    EVERY request except health checks must pass through this.
    When auth is disabled (default in dev), produces a default TenantContext.
    """

    def __init__(self, app, config: AuthConfig) -> None:
        super().__init__(app)
        self._config = config

    async def dispatch(self, request: Request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        tenant = await self._resolve_tenant(request)
        request.state.tenant = tenant

        response = await call_next(request)
        return response

    async def _resolve_tenant(self, request: Request) -> TenantContext:
        try:
            return await resolve_tenant_from_scope(request.headers, self._config)
        except ValueError as e:
            raise HTTPException(
                status_code=401,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )


async def resolve_tenant_from_scope(
    headers,
    config: AuthConfig,
) -> TenantContext:
    """Resolve tenant from headers — reusable for both HTTP and WebSocket."""
    if not config.enabled:
        return TenantContext(
            group_id=config.default_group_id,
            user_id=None,
            role="owner",
            auth_method="none",
        )

    auth_header = headers.get("authorization", "")
    if auth_header.startswith("Bearer ") and config.bearer_token:
        token = auth_header[7:]
        if token == config.bearer_token:
            return TenantContext(
                group_id=config.default_group_id,
                user_id=None,
                role="owner",
                auth_method="bearer",
            )

    raise ValueError("Valid authentication credentials required")


def get_tenant(request: Request) -> TenantContext:
    """Extract TenantContext from request. Fails loud if missing."""
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        raise RuntimeError(
            "TenantContext missing — this handler was not protected by "
            "TenantContextMiddleware. This is a bug."
        )
    return tenant
