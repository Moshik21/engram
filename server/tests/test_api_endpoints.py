"""Tests for REST API endpoints (Phase 2)."""

from __future__ import annotations

import time
from datetime import datetime

import httpx
import pytest
import pytest_asyncio

from engram.config import EngramConfig
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship


@pytest_asyncio.fixture
async def api_client(tmp_path):
    """Create an httpx.AsyncClient wired to the FastAPI app with test data."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "api_test.db")},
    )
    app = create_app(config)

    # Manual startup
    await _startup(app, config)

    graph_store = _app_state["graph_store"]
    activation_store = _app_state["activation_store"]
    now = time.time()

    # Populate test data: 3 entities + 2 relationships
    e1 = Entity(
        id="ent_alice",
        name="Alice",
        entity_type="Person",
        summary="Engineer",
        group_id="default",
    )
    e2 = Entity(
        id="ent_bob",
        name="Bob",
        entity_type="Person",
        summary="Designer",
        group_id="default",
    )
    e3 = Entity(
        id="ent_engram",
        name="Engram",
        entity_type="Project",
        summary="Memory layer",
        group_id="default",
    )

    for e in [e1, e2, e3]:
        await graph_store.create_entity(e)

    r1 = Relationship(
        id="rel_builds",
        source_id="ent_alice",
        target_id="ent_engram",
        predicate="BUILDS",
        weight=1.0,
        group_id="default",
    )
    r2 = Relationship(
        id="rel_designs",
        source_id="ent_bob",
        target_id="ent_engram",
        predicate="DESIGNS",
        weight=0.8,
        group_id="default",
    )
    for r in [r1, r2]:
        await graph_store.create_relationship(r)

    # Record activation for Alice (high) and Bob (lower)
    await activation_store.record_access("ent_alice", now)
    await activation_store.record_access("ent_alice", now - 10)
    await activation_store.record_access("ent_bob", now - 3600)

    # Index entities for FTS5 search
    search_index = _app_state["search_index"]
    for e in [e1, e2, e3]:
        await search_index.index_entity(e)

    # Create test episodes
    for i in range(3):
        ep = Episode(
            id=f"ep_test_{i}",
            content=f"Test episode {i}",
            source="api" if i < 2 else "mcp",
            status=EpisodeStatus.COMPLETED,
            group_id="default",
            created_at=datetime(2025, 1, 10 + i, 12, 0, 0),
        )
        await graph_store.create_episode(ep)
    # Link episode 0 to Alice
    await graph_store.link_episode_entity("ep_test_0", "ent_alice")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await _shutdown()
    # Clear app state for next test
    _app_state.clear()


@pytest_asyncio.fixture
async def empty_client(tmp_path):
    """Client with an empty graph."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "empty_test.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await _shutdown()
    _app_state.clear()


# ─── TestGraphNeighborhood ────────────────────────────────────────


class TestGraphNeighborhood:
    @pytest.mark.asyncio
    async def test_returns_nodes_and_edges_auto_center(self, api_client):
        """Auto-picks center when none provided; returns nodes and edges."""
        resp = await api_client.get("/api/graph/neighborhood")
        assert resp.status_code == 200
        data = resp.json()
        assert data["centerId"] is not None
        assert len(data["nodes"]) > 0
        assert isinstance(data["edges"], list)
        assert "truncated" in data
        assert "totalInNeighborhood" in data

    @pytest.mark.asyncio
    async def test_with_center_id(self, api_client):
        """Centering on ent_alice returns Alice + neighbors."""
        resp = await api_client.get("/api/graph/neighborhood?center=ent_alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["centerId"] == "ent_alice"
        node_ids = {n["id"] for n in data["nodes"]}
        assert "ent_alice" in node_ids
        # Alice's neighbor Engram should be present
        assert "ent_engram" in node_ids

    @pytest.mark.asyncio
    async def test_max_nodes_pruning(self, api_client):
        """Setting max_nodes=2 truncates the result."""
        resp = await api_client.get("/api/graph/neighborhood?center=ent_engram&max_nodes=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) <= 2
        # With 3 nodes in neighborhood, truncated should be true
        if data["totalInNeighborhood"] > 2:
            assert data["truncated"] is True

    @pytest.mark.asyncio
    async def test_entity_not_found(self, api_client):
        """Requesting a nonexistent center returns 404."""
        resp = await api_client.get("/api/graph/neighborhood?center=ent_nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty(self, empty_client):
        """Empty graph returns empty neighborhood."""
        resp = await empty_client.get("/api/graph/neighborhood")
        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == []
        assert data["edges"] == []
        assert data["centerId"] is None


# ─── TestEntitySearch ─────────────────────────────────────────────


class TestEntitySearch:
    @pytest.mark.asyncio
    async def test_search_by_name(self, api_client):
        """Search for Alice by name."""
        resp = await api_client.get("/api/entities/search?q=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1
        names = {item["name"] for item in data["items"]}
        assert "Alice" in names

    @pytest.mark.asyncio
    async def test_search_by_type_filter(self, api_client):
        """Search with type filter returns only matching types."""
        resp = await api_client.get("/api/entities/search?type=Project")
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["entityType"] == "Project"

    @pytest.mark.asyncio
    async def test_search_empty_result(self, api_client):
        """Search for nonexistent entity returns empty or low-relevance results."""
        resp = await api_client.get("/api/entities/search?q=Zzzznotfound")
        assert resp.status_code == 200
        data = resp.json()
        # With multi-pool retrieval enabled, graph/activation pools may return
        # entities even for unmatched queries. Just verify a valid response.
        assert isinstance(data["total"], int)
        assert isinstance(data["items"], list)


# ─── TestEntityDetail ─────────────────────────────────────────────


class TestEntityDetail:
    @pytest.mark.asyncio
    async def test_get_entity_with_facts(self, api_client):
        """GET entity returns detail with facts."""
        resp = await api_client.get("/api/entities/ent_alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "ent_alice"
        assert data["name"] == "Alice"
        assert data["entityType"] == "Person"
        assert data["activationCurrent"] > 0
        assert isinstance(data["facts"], list)
        # Alice has BUILDS relationship
        predicates = {f["predicate"] for f in data["facts"]}
        assert "BUILDS" in predicates

    @pytest.mark.asyncio
    async def test_entity_not_found_404(self, api_client):
        """GET nonexistent entity returns 404."""
        resp = await api_client.get("/api/entities/ent_nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_entity_neighbors(self, api_client):
        """GET entity neighbors returns neighborhood."""
        resp = await api_client.get("/api/entities/ent_alice/neighbors")
        assert resp.status_code == 200
        data = resp.json()
        assert data["centerId"] == "ent_alice"
        node_ids = {n["id"] for n in data["nodes"]}
        assert "ent_alice" in node_ids


# ─── TestEntityMutations ─────────────────────────────────────────


class TestEntityMutations:
    @pytest.mark.asyncio
    async def test_patch_updates_entity(self, api_client):
        """PATCH updates name and summary."""
        resp = await api_client.patch(
            "/api/entities/ent_bob",
            json={"name": "Robert", "summary": "Lead Designer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Robert"
        assert data["summary"] == "Lead Designer"

        # Verify persistence
        resp2 = await api_client.get("/api/entities/ent_bob")
        assert resp2.status_code == 200
        assert resp2.json()["name"] == "Robert"

    @pytest.mark.asyncio
    async def test_delete_soft_deletes(self, api_client):
        """DELETE soft-deletes and entity becomes 404."""
        resp = await api_client.delete("/api/entities/ent_bob")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

        # Verify entity is gone
        resp2 = await api_client.get("/api/entities/ent_bob")
        assert resp2.status_code == 404


# ─── TestStats ────────────────────────────────────────────────────


class TestStats:
    @pytest.mark.asyncio
    async def test_returns_stats_populated(self, api_client):
        """Stats endpoint returns entity/relationship counts."""
        resp = await api_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["entities"] >= 3
        assert data["stats"]["relationships"] >= 2
        assert "topActivated" in data
        assert data["groupId"] == "default"

    @pytest.mark.asyncio
    async def test_returns_stats_empty_graph(self, empty_client):
        """Stats for empty graph returns zeros."""
        resp = await empty_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["entities"] == 0
        assert data["stats"]["relationships"] == 0
        assert data["topActivated"] == []

    @pytest.mark.asyncio
    async def test_stats_includes_top_connected(self, api_client):
        """Enhanced stats includes topConnected list."""
        resp = await api_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "topConnected" in data
        assert isinstance(data["topConnected"], list)
        if data["topConnected"]:
            item = data["topConnected"][0]
            assert "id" in item
            assert "edgeCount" in item

    @pytest.mark.asyncio
    async def test_stats_includes_growth_timeline(self, api_client):
        """Enhanced stats includes growthTimeline list."""
        resp = await api_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "growthTimeline" in data
        assert isinstance(data["growthTimeline"], list)
        if data["growthTimeline"]:
            day = data["growthTimeline"][0]
            assert "date" in day
            assert "episodes" in day
            assert "entities" in day


# ─── TestEpisodes ────────────────────────────────────────────────


class TestEpisodes:
    @pytest.mark.asyncio
    async def test_list_episodes(self, api_client):
        """GET /api/episodes returns episodes."""
        resp = await api_client.get("/api/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "nextCursor" in data
        assert len(data["items"]) >= 3

    @pytest.mark.asyncio
    async def test_episodes_pagination(self, api_client):
        """Cursor pagination returns pages."""
        resp = await api_client.get("/api/episodes?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 2
        # Should have a next cursor since there are 3 episodes
        assert data["nextCursor"] is not None

        # Fetch next page
        resp2 = await api_client.get(f"/api/episodes?limit=2&cursor={data['nextCursor']}")
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert len(data2["items"]) >= 1

    @pytest.mark.asyncio
    async def test_episodes_filter_by_source(self, api_client):
        """Filter episodes by source."""
        resp = await api_client.get("/api/episodes?source=api")
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["source"] == "api"

    @pytest.mark.asyncio
    async def test_episodes_filter_by_status(self, api_client):
        """Filter episodes by status."""
        resp = await api_client.get("/api/episodes?status=completed")
        assert resp.status_code == 200
        data = resp.json()
        for item in data["items"]:
            assert item["status"] == "completed"

    @pytest.mark.asyncio
    async def test_episodes_empty(self, empty_client):
        """Empty graph returns no episodes."""
        resp = await empty_client.get("/api/episodes")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["nextCursor"] is None


# ─── TestGraphAt ─────────────────────────────────────────────────


class TestGraphAt:
    @pytest.mark.asyncio
    async def test_temporal_query(self, api_client):
        """GET /api/graph/at returns temporal subgraph."""
        resp = await api_client.get("/api/graph/at?center=ent_alice&at=2099-01-01T00:00:00")
        assert resp.status_code == 200
        data = resp.json()
        assert data["centerId"] == "ent_alice"
        assert data["at"] == "2099-01-01T00:00:00"
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    @pytest.mark.asyncio
    async def test_temporal_query_not_found(self, api_client):
        """GET /api/graph/at returns 404 for unknown entity."""
        resp = await api_client.get("/api/graph/at?center=ent_nope&at=2099-01-01T00:00:00")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_temporal_query_bad_timestamp(self, api_client):
        """GET /api/graph/at returns 400 for invalid timestamp."""
        resp = await api_client.get("/api/graph/at?center=ent_alice&at=not-a-date")
        assert resp.status_code == 400
