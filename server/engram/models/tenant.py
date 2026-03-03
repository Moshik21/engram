"""Tenant context for multi-tenant isolation."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class TenantContext(BaseModel):
    """Immutable per-request tenant scope. Injected by auth middleware."""

    model_config = {"frozen": True}

    group_id: str
    user_id: str | None = None
    role: str = "owner"
    auth_method: str = "none"

    @field_validator("group_id")
    @classmethod
    def group_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("group_id cannot be empty")
        return v
