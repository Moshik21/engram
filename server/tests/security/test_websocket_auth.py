"""Tests for WebSocket tenant isolation (WS1 fix)."""

from __future__ import annotations

import pytest

from engram.config import AuthConfig
from engram.security.middleware import resolve_tenant_from_scope


@pytest.mark.asyncio
class TestResolveTenantFromScope:
    async def test_raises_on_no_auth_when_enabled(self):
        """WS rejects unauthenticated connection when auth enabled."""
        config = AuthConfig(enabled=True, bearer_token="secret-token")
        with pytest.raises(ValueError, match="Valid authentication credentials required"):
            await resolve_tenant_from_scope({}, config)

    async def test_accepts_valid_bearer_token(self):
        """WS accepts connection with valid Bearer token."""
        config = AuthConfig(
            enabled=True,
            bearer_token="secret-token",
            default_group_id="my-group",
        )
        tenant = await resolve_tenant_from_scope({"authorization": "Bearer secret-token"}, config)
        assert tenant.group_id == "my-group"
        assert tenant.auth_method == "bearer"
        assert tenant.role == "owner"

    async def test_connects_without_auth_when_disabled(self):
        """WS connects without auth when auth disabled (dev mode)."""
        config = AuthConfig(enabled=False, default_group_id="dev-group")
        tenant = await resolve_tenant_from_scope({}, config)
        assert tenant.group_id == "dev-group"
        assert tenant.auth_method == "none"

    async def test_uses_config_default_group_id(self):
        """WS uses config's default_group_id (not query param)."""
        config = AuthConfig(
            enabled=True,
            bearer_token="tok",
            default_group_id="configured-group",
        )
        tenant = await resolve_tenant_from_scope({"authorization": "Bearer tok"}, config)
        assert tenant.group_id == "configured-group"

    async def test_raises_on_bad_token(self):
        """resolve_tenant_from_scope raises ValueError on bad token."""
        config = AuthConfig(enabled=True, bearer_token="correct-token")
        with pytest.raises(ValueError, match="Valid authentication credentials required"):
            await resolve_tenant_from_scope({"authorization": "Bearer wrong-token"}, config)
