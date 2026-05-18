"""Route-facing health response helpers."""

from __future__ import annotations

from typing import Any

from engram.api.deps import get_config, get_graph_store, get_mode
from engram.api.health_surface import HealthResponse, build_api_health_surface


async def build_api_health_response(*, version: str) -> HealthResponse:
    """Build the public health response from current API dependencies."""
    return await build_api_health_surface(
        graph_store=_optional_graph_store(),
        default_group_id=_default_group_id(),
        version=version,
        mode=get_mode(),
    )


def _optional_graph_store() -> Any | None:
    try:
        return get_graph_store()
    except RuntimeError:
        return None


def _default_group_id() -> str:
    try:
        return str(get_config().default_group_id)
    except RuntimeError:
        return "default"
