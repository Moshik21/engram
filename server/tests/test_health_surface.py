"""Tests for API health response assembly."""

from __future__ import annotations

import asyncio

import pytest

from engram.api.health_surface import (
    ServiceStatus,
    aggregate_service_statuses,
    build_api_health_surface,
    probe_graph_store_health,
)


class FakeGraphStore:
    def __init__(self, *, raises: bool = False) -> None:
        self.raises = raises
        self.group_ids: list[str] = []

    async def get_stats(self, *, group_id: str | None = None) -> dict:
        self.group_ids.append(group_id or "")
        if self.raises:
            raise RuntimeError("graph unavailable")
        return {}


class SlowGraphStore:
    async def get_stats(self, *, group_id: str | None = None) -> dict:
        await asyncio.sleep(0.2)
        return {}


def test_aggregate_service_statuses_reports_degraded_without_unhealthy() -> None:
    status = aggregate_service_statuses(
        {
            "graph_store": ServiceStatus.HEALTHY,
            "search_index": ServiceStatus.DEGRADED,
        }
    )

    assert status == ServiceStatus.DEGRADED


@pytest.mark.asyncio
async def test_probe_graph_store_health_uses_default_group() -> None:
    graph_store = FakeGraphStore()

    assert (
        await probe_graph_store_health(graph_store, group_id="native_brain")
        == ServiceStatus.HEALTHY
    )
    assert graph_store.group_ids == ["native_brain"]


@pytest.mark.asyncio
async def test_probe_graph_store_health_reports_unhealthy_when_missing_or_failing() -> None:
    assert (
        await probe_graph_store_health(None, group_id="brain")
        == ServiceStatus.UNHEALTHY
    )
    assert (
        await probe_graph_store_health(FakeGraphStore(raises=True), group_id="brain")
        == ServiceStatus.UNHEALTHY
    )


@pytest.mark.asyncio
async def test_probe_graph_store_health_reports_degraded_when_stats_timeout() -> None:
    assert (
        await probe_graph_store_health(
            SlowGraphStore(),
            group_id="brain",
            timeout_seconds=0.01,
        )
        == ServiceStatus.DEGRADED
    )


@pytest.mark.asyncio
async def test_build_api_health_surface_returns_public_payload() -> None:
    graph_store = FakeGraphStore()

    payload = await build_api_health_surface(
        graph_store=graph_store,
        default_group_id="native_brain",
        mode="helix",
        version="1.2.3",
    )

    assert payload.status == ServiceStatus.HEALTHY
    assert payload.mode == "helix"
    assert payload.version == "1.2.3"
    assert payload.services == {"graph_store": ServiceStatus.HEALTHY}
    assert graph_store.group_ids == ["native_brain"]
