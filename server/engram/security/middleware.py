"""Tenant context middleware — ensures every request has a resolved TenantContext."""

from __future__ import annotations

import logging

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from engram.config import AuthConfig
from engram.models.tenant import TenantContext

logger = logging.getLogger(__name__)

EXEMPT_PATHS = {"/health", "/metrics"}

# Module-level OIDC validator — initialized lazily when OIDC is enabled
_oidc_validator = None


def _get_oidc_validator(config: AuthConfig):
    """Lazily create and cache OIDCValidator singleton."""
    global _oidc_validator
    if _oidc_validator is None and config.oidc_enabled:
        from engram.security.oidc import OIDCValidator

        _oidc_validator = OIDCValidator(
            issuer=config.oidc_issuer,
            audience=config.oidc_audience,
            group_claim=config.oidc_group_claim,
        )
    return _oidc_validator


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

    # OIDC JWT validation (Clerk)
    if config.oidc_enabled and auth_header.startswith("Bearer "):
        token = auth_header[7:]
        validator = _get_oidc_validator(config)
        if validator:
            try:
                claims = await validator.validate_token(token)
                return TenantContext(
                    group_id=claims.get("group_id", config.default_group_id),
                    user_id=claims.get("sub"),
                    role="owner",
                    auth_method="oidc",
                )
            except ValueError:
                raise ValueError("Invalid or expired JWT token")

    # Bearer token fallback
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
    if not isinstance(tenant, TenantContext):
        raise RuntimeError(
            "TenantContext missing — this handler was not protected by "
            "TenantContextMiddleware. This is a bug."
        )
    return tenant
