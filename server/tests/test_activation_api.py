"""Tests for activation monitor API endpoints."""

from __future__ import annotations

import time

import httpx
import pytest
import pytest_asyncio

from engram.config import EngramConfig
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.models.entity import Entity


@pytest_asyncio.fixture
async def activation_client(tmp_path):
    """Create an httpx.AsyncClient with test data for activation endpoints."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "act_test.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    graph_store = _app_state["graph_store"]
    activation_store = _app_state["activation_store"]
    now = time.time()

    # Create test entities
    e1 = Entity(
        id="ent_alpha",
        name="Alpha",
        entity_type="Concept",
        summary="First entity",
        group_id="default",
    )
    e2 = Entity(
        id="ent_beta",
        name="Beta",
        entity_type="Technology",
        summary="Second entity",
        group_id="default",
    )
    await graph_store.create_entity(e1)
    await graph_store.create_entity(e2)

    # Record access for e1 (should have higher activation)
    await activation_store.record_access("ent_alpha", now - 10, group_id="default")
    await activation_store.record_access("ent_alpha", now, group_id="default")
    await activation_store.record_access("ent_beta", now - 3600, group_id="default")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    await _shutdown()


class TestActivationSnapshot:
    @pytest.mark.asyncio
    async def test_returns_sorted_entities(self, activation_client):
        """Snapshot returns entities sorted by activation descending."""
        resp = await activation_client.get("/api/activation/snapshot?limit=10")
        assert resp.status_code == 200
        data = resp.json()

        items = data["topActivated"]
        assert len(items) >= 2

        # Alpha was accessed more recently → higher activation
        alpha = next(i for i in items if i["entityId"] == "ent_alpha")
        beta = next(i for i in items if i["entityId"] == "ent_beta")
        assert alpha["currentActivation"] > beta["currentActivation"]

        # Verify fields present
        assert "name" in alpha
        assert "entityType" in alpha
        assert "accessCount" in alpha
        assert "decayRate" in alpha

    @pytest.mark.asyncio
    async def test_respects_limit(self, activation_client):
        """Snapshot respects the limit parameter."""
        resp = await activation_client.get("/api/activation/snapshot?limit=1")
        assert resp.status_code == 200
        items = resp.json()["topActivated"]
        assert len(items) == 1


class TestActivationCurve:
    @pytest.mark.asyncio
    async def test_returns_curve_points(self, activation_client):
        """Curve returns array of timestamp/activation points."""
        resp = await activation_client.get("/api/activation/ent_alpha/curve?hours=1&points=10")
        assert resp.status_code == 200
        data = resp.json()

        assert data["entityId"] == "ent_alpha"
        assert data["entityName"] == "Alpha"
        assert len(data["curve"]) == 10
        assert "formula" in data
        assert "accessEvents" in data

        # Each point has timestamp and activation
        for point in data["curve"]:
            assert "timestamp" in point
            assert "activation" in point
            assert 0.0 <= point["activation"] <= 1.0

    @pytest.mark.asyncio
    async def test_entity_not_found_returns_404(self, activation_client):
        """Requesting curve for nonexistent entity returns 404."""
        resp = await activation_client.get("/api/activation/ent_nonexistent/curve")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_curve_shows_decay(self, activation_client):
        """Activation decreases over time (decay curve)."""
        resp = await activation_client.get("/api/activation/ent_alpha/curve?hours=1&points=5")
        data = resp.json()
        curve = data["curve"]

        # Later points should have higher or equal activation (closer to now)
        # The last point (most recent) should be >= the first (oldest)
        assert curve[-1]["activation"] >= curve[0]["activation"]
