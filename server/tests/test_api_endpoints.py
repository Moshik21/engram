"""Tests for REST API endpoints (Phase 2)."""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio

from engram.api.health import ServiceStatus, health_check
from engram.config import EngramConfig
from engram.evaluation.store import StoredRecallRuntimeMetricsSnapshot
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.models.consolidation import ConsolidationCycle, PhaseResult
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.relationship import Relationship


class _HealthGraphStore:
    def __init__(self) -> None:
        self.group_ids: list[str | None] = []

    async def get_stats(self, group_id: str | None = None) -> dict:
        self.group_ids.append(group_id)
        return {}


@pytest.mark.asyncio
async def test_health_uses_configured_default_group() -> None:
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
        response = await health_check()
    finally:
        _app_state.clear()

    assert response.status == ServiceStatus.HEALTHY
    assert graph_store.group_ids == ["native_brain"]


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


class TestConsolidationAPI:
    @pytest.mark.asyncio
    async def test_status_and_history_include_cycle_and_phase_errors(self, api_client):
        store = _app_state["consolidation_store"]
        cycle = ConsolidationCycle(
            group_id="default",
            trigger="manual",
            dry_run=True,
            status="failed",
            error="Capability validation failed",
            phase_results=[
                PhaseResult(
                    phase="triage",
                    status="error",
                    items_processed=2,
                    items_affected=0,
                    duration_ms=12.5,
                    error="missing graph method",
                )
            ],
        )
        cycle.completed_at = cycle.started_at + 1
        cycle.total_duration_ms = 1000.0
        await store.save_cycle(cycle)

        status_resp = await api_client.get("/api/consolidation/status")
        assert status_resp.status_code == 200
        latest = status_resp.json()["latest_cycle"]
        assert latest["id"] == cycle.id
        assert latest["status"] == "failed"
        assert latest["error"] == "Capability validation failed"
        assert latest["phases"][0]["duration_ms"] == 12.5
        assert latest["phases"][0]["error"] == "missing graph method"
        assert latest["summary"] == {
            "total_processed": 2,
            "total_affected": 0,
            "description": ("Dry run failed cycle: 2 items processed, 0 affected across 1 phases"),
        }

        history_resp = await api_client.get("/api/consolidation/history")
        assert history_resp.status_code == 200
        history_cycle = history_resp.json()["cycles"][0]
        assert history_cycle["id"] == cycle.id
        assert history_cycle["error"] == "Capability validation failed"
        assert history_cycle["phases"][0]["duration_ms"] == 12.5
        assert history_cycle["phases"][0]["error"] == "missing graph method"
        assert history_cycle["summary"]["total_processed"] == 2

        detail_resp = await api_client.get(f"/api/consolidation/cycle/{cycle.id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["error"] == "Capability validation failed"
        assert detail["phases"][0]["error"] == "missing graph method"
        assert detail["summary"]["description"].startswith("Dry run failed cycle:")


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
        assert data["representation"]["scope"] == "neighborhood"
        assert data["representation"]["layout"] == "force"

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
        assert data["representation"]["displayedNodeCount"] == len(data["nodes"])

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
        assert data["representation"]["scope"] == "neighborhood"


class TestGraphAtlas:
    @pytest.mark.asyncio
    async def test_returns_display_bounded_atlas(self, api_client):
        resp = await api_client.get("/api/graph/atlas")
        assert resp.status_code == 200
        data = resp.json()
        assert data["representation"]["scope"] == "atlas"
        assert data["representation"]["layout"] == "precomputed"
        assert data["stats"]["totalEntities"] == 3
        assert data["stats"]["totalRelationships"] == 2
        assert data["stats"]["totalRegions"] >= 1
        assert isinstance(data["regions"], list)
        assert isinstance(data["bridges"], list)
        if data["regions"]:
            region = data["regions"][0]
            assert "centerEntityId" in region
            assert "memberCount" in region
            assert "activationScore" in region
            assert "latestEntityCreatedAt" in region

    @pytest.mark.asyncio
    async def test_region_drill_down_returns_materialized_region(self, api_client):
        atlas_resp = await api_client.get("/api/graph/atlas")
        assert atlas_resp.status_code == 200
        atlas = atlas_resp.json()
        region_id = atlas["regions"][0]["id"]

        resp = await api_client.get(f"/api/graph/regions/{region_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["representation"]["scope"] == "region"
        assert data["representation"]["layout"] == "precomputed"
        assert data["region"]["id"] == region_id
        assert "activationScore" in data["region"]
        assert "growth7d" in data["region"]
        assert "latestEntityCreatedAt" in data["region"]
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        assert isinstance(data["topEntities"], list)
        assert isinstance(data["memberIds"], list)
        assert data["memberIds"]

    @pytest.mark.asyncio
    async def test_atlas_rebuild_preserves_region_ids_and_positions(self, api_client):
        atlas_service = _app_state["atlas_service"]

        first = await atlas_service.get_snapshot("default", force=True)
        second = await atlas_service.get_snapshot("default", force=True)

        first_regions = {region.id: (region.x, region.y, region.z) for region in first.regions}
        second_regions = {region.id: (region.x, region.y, region.z) for region in second.regions}
        assert first_regions
        assert first_regions == second_regions

    @pytest.mark.asyncio
    async def test_history_and_snapshot_lookup(self, api_client):
        first_resp = await api_client.get("/api/graph/atlas?refresh=true")
        assert first_resp.status_code == 200
        first = first_resp.json()

        second_resp = await api_client.get("/api/graph/atlas?refresh=true")
        assert second_resp.status_code == 200
        second = second_resp.json()

        first_snapshot_id = first["representation"]["snapshotId"]
        second_snapshot_id = second["representation"]["snapshotId"]
        assert first_snapshot_id != second_snapshot_id

        history_resp = await api_client.get("/api/graph/atlas/history?limit=10")
        assert history_resp.status_code == 200
        history = history_resp.json()["items"]
        history_ids = [item["id"] for item in history]
        assert second_snapshot_id in history_ids
        assert first_snapshot_id in history_ids
        assert history[0]["id"] == second_snapshot_id

        snapshot_resp = await api_client.get(f"/api/graph/atlas?snapshot_id={first_snapshot_id}")
        assert snapshot_resp.status_code == 200
        snapshot = snapshot_resp.json()
        assert snapshot["representation"]["snapshotId"] == first_snapshot_id

    @pytest.mark.asyncio
    async def test_region_snapshot_lookup_and_invalid_snapshot_404(self, api_client):
        atlas_resp = await api_client.get("/api/graph/atlas?refresh=true")
        assert atlas_resp.status_code == 200
        atlas = atlas_resp.json()
        region_id = atlas["regions"][0]["id"]
        snapshot_id = atlas["representation"]["snapshotId"]

        region_resp = await api_client.get(
            f"/api/graph/regions/{region_id}?snapshot_id={snapshot_id}"
        )
        assert region_resp.status_code == 200
        region = region_resp.json()
        assert region["representation"]["snapshotId"] == snapshot_id
        assert region["generatedAt"] == atlas["generatedAt"]

        missing_resp = await api_client.get(
            f"/api/graph/regions/{region_id}?snapshot_id=atlas_missing"
        )
        assert missing_resp.status_code == 404


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
        alice = next(item for item in data["items"] if item["name"] == "Alice")
        assert alice["lexicalRegime"] == "natural_language"
        assert alice["canonicalIdentifier"] is None

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
        assert data["lexicalRegime"] == "natural_language"
        assert data["canonicalIdentifier"] is None
        assert data["identifierLabel"] is False
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
        assert "recall_metrics" in data["stats"]
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
        assert data["stats"]["recall_metrics"]["trigger_count"] == 0
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

    @pytest.mark.asyncio
    async def test_stats_includes_cue_and_projection_metrics(self, api_client):
        graph_store = _app_state["graph_store"]
        await graph_store.update_episode(
            "ep_test_0",
            {
                "projection_state": EpisodeProjectionState.PROJECTED,
                "processing_duration_ms": 180,
                "last_projected_at": datetime(2025, 1, 10, 12, 0, 3),
            },
            group_id="default",
        )
        await graph_store.update_episode(
            "ep_test_1",
            {"projection_state": EpisodeProjectionState.FAILED},
            group_id="default",
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_test_0",
                group_id="default",
                projection_state=EpisodeProjectionState.PROJECTED,
                cue_text="Alice project context",
                hit_count=2,
                surfaced_count=1,
                selected_count=1,
                used_count=1,
                policy_score=0.8,
                projection_attempts=1,
                created_at=datetime(2025, 1, 10, 12, 0, 0),
                last_projected_at=datetime(2025, 1, 10, 12, 0, 3),
            )
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_test_1",
                group_id="default",
                projection_state=EpisodeProjectionState.FAILED,
                cue_text="failed projection",
                near_miss_count=1,
                policy_score=0.2,
                projection_attempts=2,
            )
        )

        resp = await api_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()

        cue_metrics = data["stats"]["cue_metrics"]
        projection_metrics = data["stats"]["projection_metrics"]
        assert cue_metrics["cue_count"] == 2
        assert cue_metrics["cue_coverage"] == pytest.approx(2 / 3, abs=1e-4)
        assert cue_metrics["cue_used_count"] == 1
        assert cue_metrics["cue_to_projection_conversion_rate"] == 0.5
        assert projection_metrics["state_counts"]["projected"] == 1
        assert projection_metrics["state_counts"]["failed"] == 1
        assert projection_metrics["total_attempts"] == 3
        assert projection_metrics["yield"]["linked_entity_count"] == 1

    @pytest.mark.asyncio
    async def test_stats_empty_graph_includes_zeroed_cue_metrics(self, empty_client):
        resp = await empty_client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["cue_metrics"]["cue_count"] == 0
        assert data["stats"]["cue_metrics"]["cue_coverage"] == 0.0
        assert data["stats"]["projection_metrics"]["state_counts"]["projected"] == 0
        assert data["stats"]["projection_metrics"]["failure_rate"] == 0.0


# ─── TestActivation ────────────────────────────────────────────────


class TestActivation:
    @pytest.mark.asyncio
    async def test_activation_snapshot_returns_dashboard_payload(self, api_client):
        resp = await api_client.get("/api/activation/snapshot?limit=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "topActivated" in data
        assert isinstance(data["topActivated"], list)
        if data["topActivated"]:
            item = data["topActivated"][0]
            assert "entityId" in item
            assert "currentActivation" in item
            assert "decayRate" in item

    @pytest.mark.asyncio
    async def test_activation_curve_returns_dashboard_payload(self, api_client):
        resp = await api_client.get("/api/activation/ent_alice/curve?hours=1&points=4")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entityId"] == "ent_alice"
        assert data["entityName"] == "Alice"
        assert len(data["curve"]) == 4
        assert "accessEvents" in data
        assert "formula" in data

    @pytest.mark.asyncio
    async def test_activation_curve_missing_entity_returns_404(self, api_client):
        resp = await api_client.get("/api/activation/ent_missing/curve")
        assert resp.status_code == 404


# ─── TestLifecycleSummary ────────────────────────────────────────


class TestLifecycleSummary:
    @pytest.mark.asyncio
    async def test_lifecycle_summary_maps_brain_loop_contract(self, api_client):
        graph_store = _app_state["graph_store"]
        await graph_store.update_episode(
            "ep_test_0",
            {
                "projection_state": EpisodeProjectionState.PROJECTED,
                "processing_duration_ms": 180,
                "last_projected_at": datetime(2025, 1, 10, 12, 0, 3),
            },
            group_id="default",
        )
        await graph_store.update_episode(
            "ep_test_1",
            {"projection_state": EpisodeProjectionState.FAILED},
            group_id="default",
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_test_0",
                group_id="default",
                projection_state=EpisodeProjectionState.PROJECTED,
                cue_text="Alice project context",
                hit_count=2,
                surfaced_count=1,
                selected_count=1,
                used_count=1,
                policy_score=0.8,
                projection_attempts=1,
                created_at=datetime(2025, 1, 10, 12, 0, 0),
                last_projected_at=datetime(2025, 1, 10, 12, 0, 3),
            )
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_test_1",
                group_id="default",
                projection_state=EpisodeProjectionState.FAILED,
                cue_text="failed projection",
                near_miss_count=1,
                policy_score=0.2,
                projection_attempts=2,
            )
        )

        resp = await api_client.get("/api/lifecycle/summary")
        assert resp.status_code == 200
        data = resp.json()

        assert data["groupId"] == "default"
        assert data["loop"] == ["capture", "cue", "project", "recall", "consolidate"]
        assert data["totals"]["episodes"] == 3
        assert data["totals"]["cues"] == 2
        assert data["totals"]["projected"] == 1
        assert data["capture"]["episodeCount"] == 3
        assert data["capture"]["latestEpisode"]["episodeId"] == "ep_test_2"
        assert data["cue"]["coverage"] == pytest.approx(2 / 3, abs=1e-4)
        assert data["cue"]["usedCount"] == 1
        assert data["project"]["status"] == "attention"
        assert data["project"]["stateCounts"]["queued"] == 1
        assert data["project"]["stateCounts"]["failed"] == 1
        assert data["recall"]["status"] in {"ready", "active"}
        assert isinstance(data["recall"]["topActivated"], list)
        assert data["recall"]["intentions"]["activeCount"] == 0
        assert data["consolidate"]["status"] == "ready"
        assert data["consolidate"]["latestCycle"] is None
        assert len(data["recentEpisodes"]) == 3

    @pytest.mark.asyncio
    async def test_lifecycle_summary_empty_graph_uses_zero_contract(self, empty_client):
        resp = await empty_client.get("/api/lifecycle/summary")
        assert resp.status_code == 200
        data = resp.json()

        assert data["totals"]["episodes"] == 0
        assert data["capture"]["latestEpisode"] is None
        assert data["cue"]["coverage"] == 0.0
        assert data["project"]["stateCounts"]["projected"] == 0
        assert data["recall"]["topActivated"] == []
        assert data["consolidate"]["cycleCount"] == 0


# ─── TestEvaluation ───────────────────────────────────────────────


class TestEvaluation:
    @pytest.mark.asyncio
    async def test_evaluation_api_records_labels_and_reports_quality(self, api_client):
        recall_resp = await api_client.post(
            "/api/evaluation/recall-samples",
            json={
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 3,
                "packetsUsed": 2,
                "falseRecalls": 1,
                "query": "What did I decide?",
            },
        )
        assert recall_resp.status_code == 201
        recall_data = recall_resp.json()
        assert recall_data["status"] == "stored"
        assert recall_data["groupId"] == "default"
        assert recall_data["sample"]["recallTriggered"] is True
        assert recall_data["sample"]["recallNeeded"] is True

        session_resp = await api_client.post(
            "/api/evaluation/session-samples",
            json={
                "baselineScore": 0.2,
                "memoryScore": 0.7,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": False,
                "scenario": "open-loop follow-up",
            },
        )
        assert session_resp.status_code == 201
        session_data = session_resp.json()
        assert session_data["sample"]["openLoopRecovered"] is True

        report_resp = await api_client.get("/api/evaluation/brain-loop/report")
        assert report_resp.status_code == 200
        report = report_resp.json()

        assert report["group_id"] == "default"
        assert report["loop"] == ["capture", "cue", "project", "recall", "consolidate"]
        assert report["recall"]["evaluation"]["status"] == "measured"
        assert report["recall"]["evaluation"]["memory_need_precision"] == 1.0
        assert report["recall"]["evaluation"]["memory_need_recall"] == 1.0
        assert report["recall"]["evaluation"]["missed_recall_rate"] == 0.0
        assert report["recall"]["evaluation"]["useful_packet_rate"] == pytest.approx(
            2 / 3,
            abs=1e-4,
        )
        assert report["recall"]["evaluation"]["false_recall_rate"] == pytest.approx(
            1 / 3,
            abs=1e-4,
        )
        assert report["recall"]["latency"]["analyzer_ms"] == {"avg_ms": 0.0, "p95_ms": 0.0}
        assert report["recall"]["latency"]["probe_ms"] == {"avg_ms": 0.0, "p95_ms": 0.0}
        assert report["recall"]["control"]["graph_override_count"] == 0
        assert report["recall"]["control"]["adaptive_thresholds_enabled"] is False
        assert report["recall"]["continuity"]["status"] == "measured"
        assert report["recall"]["continuity"]["session_continuity_lift"] == 0.5
        assert report["recall"]["continuity"]["open_loop_recovery_rate"] == 1.0
        assert report["recall"]["continuity"]["temporal_correctness"] == 0.0

    @pytest.mark.asyncio
    async def test_evaluation_report_empty_graph_uses_gap_contract(self, empty_client):
        resp = await empty_client.get("/api/evaluation/brain-loop/report")
        assert resp.status_code == 200
        report = resp.json()

        assert report["capture"]["status"] == "empty"
        assert report["recall"]["evaluation"]["status"] == "needs_samples"
        assert "recall quality needs labeled recall_samples input" in report["coverage_gaps"]

    @pytest.mark.asyncio
    async def test_evaluation_report_uses_saved_recall_runtime_snapshot(self, api_client):
        store = _app_state["evaluation_store"]
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                group_id="default",
                metrics={
                    "total_analyses": 2,
                    "trigger_count": 1,
                    "analyzer_latency_ms": {"avg": 7.0, "p95": 14.0},
                    "probe_latency_ms": {"avg": 4.0, "p95": 8.0},
                    "surfaced_count": 3,
                    "thresholds": {"resonance": 0.52},
                },
                source="test",
            )
        )

        resp = await api_client.get("/api/evaluation/brain-loop/report")
        assert resp.status_code == 200
        report = resp.json()

        assert report["recall"]["total_analyses"] == 2
        assert report["recall"]["trigger_count"] == 1
        assert report["recall"]["latency"]["analyzer_ms"]["p95_ms"] == 14.0
        assert report["recall"]["latency"]["probe_ms"]["avg_ms"] == 4.0
        assert report["recall"]["control"]["surfaced_count"] == 3
        assert report["recall"]["control"]["thresholds"]["resonance"] == 0.52
        assert "recall gate needs runtime analyses" not in report["coverage_gaps"]

    @pytest.mark.asyncio
    async def test_evaluation_report_persists_live_recall_runtime_snapshot(self, api_client):
        manager = _app_state["graph_manager"]
        store = _app_state["evaluation_store"]
        manager.record_memory_need_analysis(
            "default",
            SimpleNamespace(
                should_recall=True,
                trigger_family="keyword",
                analyzer_latency_ms=11.0,
                probe_triggered=True,
                probe_latency_ms=5.0,
                decision_path="graph_lift",
                graph_override_used=True,
            ),
        )

        resp = await api_client.get("/api/evaluation/brain-loop/report")
        assert resp.status_code == 200
        report = resp.json()
        saved = await store.get_latest_recall_metrics_snapshot("default")

        assert report["recall"]["total_analyses"] == 1
        assert report["recall"]["latency"]["analyzer_ms"]["p95_ms"] == 11.0
        assert report["recall"]["latency"]["probe_ms"]["p95_ms"] == 5.0
        assert saved["total_analyses"] == 1
        assert saved["trigger_count"] == 1
        assert saved["analyzer_latency_ms"]["p95"] == 11.0
        assert saved["graph_lift_rate"] == 1.0


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
    async def test_episodes_include_projection_and_cue_debug_fields(self, api_client):
        """Episode listings expose projection/cue rollout state for debugging."""
        graph_store = _app_state["graph_store"]
        projected_at = datetime(2026, 3, 5, 15, 0, 0)
        feedback_at = datetime(2026, 3, 5, 14, 30, 0)
        await graph_store.update_episode(
            "ep_test_1",
            {
                "projection_state": EpisodeProjectionState.SCHEDULED,
                "last_projection_reason": "cue_policy_used",
                "last_projected_at": projected_at.isoformat(),
            },
            group_id="default",
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_test_1",
                group_id="default",
                projection_state=EpisodeProjectionState.SCHEDULED,
                cue_text="React dashboard migration remains in scope",
                route_reason="cue_policy_used",
                hit_count=3,
                surfaced_count=1,
                selected_count=1,
                used_count=1,
                policy_score=0.81,
                projection_attempts=2,
                last_feedback_at=feedback_at,
                last_projected_at=projected_at,
            )
        )

        resp = await api_client.get("/api/episodes")
        assert resp.status_code == 200
        data = resp.json()
        item = next(ep for ep in data["items"] if ep["episodeId"] == "ep_test_1")

        assert item["projectionState"] == "scheduled"
        assert item["lastProjectionReason"] == "cue_policy_used"
        assert item["lastProjectedAt"] == "2026-03-05T15:00:00Z"
        assert item["cue"]["projectionState"] == "scheduled"
        assert item["cue"]["routeReason"] == "cue_policy_used"
        assert item["cue"]["hitCount"] == 3
        assert item["cue"]["policyScore"] == pytest.approx(0.81)
        assert item["cue"]["lastFeedbackAt"] == "2026-03-05T14:30:00Z"

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
