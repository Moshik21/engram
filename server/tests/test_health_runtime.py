"""Tests for route-facing health response helpers."""

from __future__ import annotations

import pytest

from engram.api.health_runtime import build_api_health_response
from engram.api.health_surface import ServiceStatus
from engram.config import EngramConfig
from engram.main import _app_state


class _HealthGraphStore:
    def __init__(self) -> None:
        self.group_ids: list[str | None] = []

    async def get_stats(self, group_id: str | None = None) -> dict:
        self.group_ids.append(group_id)
        return {}


@pytest.mark.asyncio
async def test_build_api_health_response_uses_configured_default_group() -> None:
    graph_store = _HealthGraphStore()
    _app_state.clear()
    _app_state.update(
        {
            "graph_store": graph_store,
            "config": EngramConfig(default_group_id="native_brain"),
            "mode": "helix",
        }
    )
    try:
        response = await build_api_health_response(version="1.2.3")
    finally:
        _app_state.clear()

    assert response.status == ServiceStatus.HEALTHY
    assert response.version == "1.2.3"
    assert response.mode == "helix"
    assert graph_store.group_ids == ["native_brain"]


@pytest.mark.asyncio
async def test_build_api_health_response_falls_back_without_config_or_store() -> None:
    _app_state.clear()
    _app_state["mode"] = "lite"
    try:
        response = await build_api_health_response(version="1.2.3")
    finally:
        _app_state.clear()

    assert response.status == ServiceStatus.UNHEALTHY
    assert response.services == {"graph_store": ServiceStatus.UNHEALTHY}
