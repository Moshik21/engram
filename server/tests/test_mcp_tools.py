"""Tests for Week 4 MCP tools — exercises GraphManager methods directly."""

from __future__ import annotations

import json
import time
import uuid
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig, EngramConfig
from engram.evaluation.store import StoredRecallEvalSample, StoredSessionContinuitySample
from engram.graph_manager import GraphManager
from engram.mcp.server import SessionState
from engram.models.entity import Entity
from engram.models.epistemic import EvidenceClaim
from engram.models.relationship import Relationship
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.utils.dates import utc_now
from tests.conftest import MockExtractor

# ─── Fixtures ────────────────────────────────────────────────────────

GROUP = "test_group"


def _entity(eid, name, etype, summary):
    return Entity(
        id=eid,
        name=name,
        entity_type=etype,
        summary=summary,
        group_id=GROUP,
    )


def _rel(rid, src, tgt, pred, vf, **kw):
    return Relationship(
        id=rid,
        source_id=src,
        target_id=tgt,
        predicate=pred,
        weight=1.0,
        valid_from=vf,
        group_id=GROUP,
        **kw,
    )


@pytest_asyncio.fixture
async def rich_manager(tmp_path):
    """GraphManager with a richer dataset for Week 4 tool testing."""
    db_path = str(tmp_path / "mcp_tools_test.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    entities = [
        _entity("ent_alex", "Alex", "Person", "Software engineer"),
        _entity("ent_engram", "Engram", "Project", "Memory layer for AI agents"),
        _entity("ent_python", "Python", "Technology", "Programming language"),
        _entity("ent_fastapi", "FastAPI", "Technology", "Web framework"),
        _entity("ent_denver", "Denver", "Location", "City in Colorado"),
        _entity("ent_mesa", "Mesa", "Location", "City in Arizona"),
    ]
    for e in entities:
        await graph_store.create_entity(e)

    now_dt = entities[0].created_at
    rels = [
        _rel("rel_builds", "ent_alex", "ent_engram", "BUILDS", now_dt),
        _rel("rel_uses_py", "ent_engram", "ent_python", "USES", now_dt),
        _rel("rel_uses_fa", "ent_engram", "ent_fastapi", "USES", now_dt),
        _rel("rel_lives", "ent_alex", "ent_denver", "LIVES_IN", now_dt),
        _rel("rel_moved", "ent_alex", "ent_mesa", "MOVED_FROM", now_dt),
    ]
    for r in rels:
        await graph_store.create_relationship(r)

    now = time.time()
    await activation_store.record_access("ent_alex", now - 10, group_id=GROUP)
    await activation_store.record_access("ent_alex", now - 5, group_id=GROUP)
    await activation_store.record_access("ent_engram", now - 20, group_id=GROUP)
    await activation_store.record_access("ent_engram", now - 15, group_id=GROUP)
    await activation_store.record_access("ent_engram", now - 10, group_id=GROUP)
    await activation_store.record_access("ent_python", now - 100, group_id=GROUP)

    extractor = MockExtractor()
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        extractor,
    )
    yield manager
    await graph_store.close()


@pytest_asyncio.fixture
async def rich_manager_with_expired(tmp_path):
    """GraphManager with expired relationships for include_expired tests."""
    db_path = str(tmp_path / "mcp_expired_test.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    now_dt = utc_now()
    past_dt = now_dt - timedelta(days=30)

    entities = [
        _entity("ent_alex2", "Alex", "Person", "Software engineer"),
        _entity("ent_acme", "Acme Corp", "Organization", "Previous employer"),
        _entity("ent_vercel", "Vercel", "Organization", "Current employer"),
    ]
    for e in entities:
        await graph_store.create_entity(e)

    rels = [
        _rel(
            "rel_old_job",
            "ent_alex2",
            "ent_acme",
            "WORKS_AT",
            past_dt,
            valid_to=now_dt,
        ),
        _rel(
            "rel_new_job",
            "ent_alex2",
            "ent_vercel",
            "WORKS_AT",
            now_dt,
        ),
    ]
    for r in rels:
        await graph_store.create_relationship(r)

    extractor = MockExtractor()
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        extractor,
    )
    yield manager
    await graph_store.close()


@pytest_asyncio.fixture
async def decision_manager(tmp_path):
    """GraphManager with decision graph materialization enabled."""
    db_path = str(tmp_path / "decision_tools_test.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    activation_store = MemoryActivationStore(
        cfg=ActivationConfig(decision_graph_enabled=True),
    )
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)

    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=ActivationConfig(decision_graph_enabled=True),
    )
    yield manager
    await graph_store.close()


# ─── TestSearchEntities ──────────────────────────────────────────────


class TestSearchEntities:
    @pytest.mark.asyncio
    async def test_search_by_name_finds_entity(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Alex",
        )
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert "Alex" in names

    @pytest.mark.asyncio
    async def test_search_by_type_only(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            entity_type="Technology",
        )
        assert len(results) >= 2
        for r in results:
            assert r["entity_type"] == "Technology"

    @pytest.mark.asyncio
    async def test_search_by_name_and_type(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Python",
            entity_type="Technology",
        )
        assert len(results) >= 1
        assert results[0]["name"] == "Python"
        assert results[0]["entity_type"] == "Technology"

    @pytest.mark.asyncio
    async def test_search_returns_activation_score(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Alex",
        )
        assert len(results) >= 1
        assert results[0]["activation_score"] > 0

    @pytest.mark.asyncio
    async def test_search_empty_result(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="NonExistentXYZ",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_does_not_record_access(self, rich_manager):
        state_before = await rich_manager._activation.get_activation(
            "ent_alex",
        )
        count_before = state_before.access_count if state_before else 0

        await rich_manager.search_entities(group_id=GROUP, name="Alex")

        state_after = await rich_manager._activation.get_activation(
            "ent_alex",
        )
        count_after = state_after.access_count if state_after else 0
        assert count_after == count_before

    @pytest.mark.asyncio
    async def test_search_returns_expected_fields(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Engram",
        )
        assert len(results) >= 1
        r = results[0]
        for key in [
            "id",
            "name",
            "entity_type",
            "summary",
            "activation_score",
            "access_count",
            "created_at",
            "updated_at",
        ]:
            assert key in r


# ─── TestSearchFacts ─────────────────────────────────────────────────


class TestSearchFacts:
    @pytest.mark.asyncio
    async def test_search_by_subject_name(self, rich_manager):
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Alex",
            subject="Alex",
        )
        assert len(facts) >= 1
        subjects = [f["subject"] for f in facts]
        assert "Alex" in subjects

    @pytest.mark.asyncio
    async def test_search_with_predicate_filter(self, rich_manager):
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Alex",
            subject="Alex",
            predicate="LIVES_IN",
        )
        assert len(facts) >= 1
        for f in facts:
            assert f["predicate"] == "LIVES_IN"

    @pytest.mark.asyncio
    async def test_search_include_expired_facts(
        self,
        rich_manager_with_expired,
    ):
        active_facts = await rich_manager_with_expired.search_facts(
            group_id=GROUP,
            query="Alex",
            subject="Alex",
            predicate="WORKS_AT",
            include_expired=False,
        )
        active_objects = [f["object"] for f in active_facts]
        assert "Vercel" in active_objects
        assert "Acme Corp" not in active_objects

        all_facts = await rich_manager_with_expired.search_facts(
            group_id=GROUP,
            query="Alex",
            subject="Alex",
            predicate="WORKS_AT",
            include_expired=True,
        )
        all_objects = [f["object"] for f in all_facts]
        assert "Vercel" in all_objects
        assert "Acme Corp" in all_objects

    @pytest.mark.asyncio
    async def test_search_resolves_entity_names(self, rich_manager):
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Engram",
            subject="Engram",
            predicate="USES",
        )
        assert len(facts) >= 1
        for f in facts:
            assert not f["subject"].startswith("ent_")
            assert not f["object"].startswith("ent_")

    @pytest.mark.asyncio
    async def test_search_no_results(self, rich_manager):
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="NonExistentXYZ",
        )
        assert facts == []

    @pytest.mark.asyncio
    async def test_search_hides_epistemic_facts_by_default(self, decision_manager):
        await decision_manager.store_episode(
            "we decided the plan is to launch Engram through OpenClaw",
            group_id=GROUP,
            source="dashboard",
        )

        hidden = await decision_manager.search_facts(
            group_id=GROUP,
            query="OpenClaw",
        )
        assert hidden == []

        visible = await decision_manager.search_facts(
            group_id=GROUP,
            query="OpenClaw",
            include_epistemic=True,
        )
        assert any(fact["predicate"] == "DECIDED_IN" for fact in visible)

    @pytest.mark.asyncio
    async def test_question_form_does_not_materialize_decision_entity(self, decision_manager):
        await decision_manager.store_episode(
            "what did we decide about launching Engram publicly?",
            group_id=GROUP,
            source="dashboard",
        )

        decisions = await decision_manager._graph.find_entities(
            entity_type="Decision",
            group_id=GROUP,
            limit=20,
        )
        assert decisions == []

    @pytest.mark.asyncio
    async def test_explicit_commitment_materializes_decision_entity(self, decision_manager):
        await decision_manager.store_episode(
            "we decided the plan is to launch Engram through OpenClaw",
            group_id=GROUP,
            source="dashboard",
        )

        decisions = await decision_manager._graph.find_entities(
            entity_type="Decision",
            group_id=GROUP,
            limit=20,
        )
        assert len(decisions) == 1

    @pytest.mark.asyncio
    async def test_artifact_decision_materializer_links_decision_to_artifact(
        self,
        decision_manager,
    ):
        artifact = Entity(
            id="art_readme_decision",
            name="README.md",
            entity_type="Artifact",
            summary="README artifact",
            attributes={"artifact_class": "readme"},
            group_id=GROUP,
        )
        await decision_manager._graph.create_entity(artifact)

        await decision_manager._materialize_artifact_decisions(
            artifact,
            [
                EvidenceClaim(
                    subject="Engram",
                    predicate="public_launch_path",
                    object="OpenClaw",
                    source_type="artifact",
                    authority_type="canonical",
                    externalization_state="documented",
                    claim_state="decided",
                    confidence=0.9,
                )
            ],
            group_id=GROUP,
        )

        decisions = await decision_manager._graph.find_entities(
            entity_type="Decision",
            group_id=GROUP,
            limit=20,
        )
        assert len(decisions) == 1
        relationships = await decision_manager._graph.get_relationships(
            decisions[0].id,
            direction="outgoing",
            group_id=GROUP,
        )
        assert any(
            rel.target_id == artifact.id and rel.predicate == "ANNOUNCED_AS"
            for rel in relationships
        )


# ─── TestForgetEntity ────────────────────────────────────────────────


class TestForgetEntity:
    @pytest.mark.asyncio
    async def test_forget_entity_soft_deletes(self, rich_manager):
        result = await rich_manager.forget_entity("Mesa", GROUP)
        assert result["status"] == "forgotten"
        assert result["target_type"] == "entity"
        assert result["target"] == "Mesa"

        entity = await rich_manager._graph.get_entity("ent_mesa", GROUP)
        assert entity is None

    @pytest.mark.asyncio
    async def test_forget_entity_clears_activation(self, rich_manager):
        await rich_manager._activation.record_access(
            "ent_python",
            time.time(),
        )
        state = await rich_manager._activation.get_activation("ent_python")
        assert state is not None

        await rich_manager.forget_entity("Python", GROUP)

        state = await rich_manager._activation.get_activation("ent_python")
        assert state is None

    @pytest.mark.asyncio
    async def test_forget_entity_not_found(self, rich_manager):
        result = await rich_manager.forget_entity("NonExistentXYZ", GROUP)
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()

    @pytest.mark.asyncio
    async def test_forgotten_entity_not_in_recall(self, rich_manager):
        await rich_manager.forget_entity("Mesa", GROUP)
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Mesa",
        )
        names = [r["name"] for r in results]
        assert "Mesa" not in names


# ─── TestForgetFact ──────────────────────────────────────────────────


class TestForgetFact:
    @pytest.mark.asyncio
    async def test_forget_fact_invalidates_relationship(self, rich_manager):
        result = await rich_manager.forget_fact(
            subject_name="Alex",
            predicate="LIVES_IN",
            object_name="Denver",
            group_id=GROUP,
        )
        assert result["status"] == "forgotten"
        assert result["target_type"] == "fact"

        rels = await rich_manager._graph.get_relationships(
            "ent_alex",
            direction="outgoing",
            predicate="LIVES_IN",
            active_only=True,
        )
        target_ids = [r.target_id for r in rels]
        assert "ent_denver" not in target_ids

    @pytest.mark.asyncio
    async def test_forget_fact_not_found(self, rich_manager):
        result = await rich_manager.forget_fact(
            subject_name="Alex",
            predicate="WORKS_AT",
            object_name="Denver",
            group_id=GROUP,
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_forgotten_fact_not_in_search_facts(self, rich_manager):
        await rich_manager.forget_fact(
            subject_name="Alex",
            predicate="LIVES_IN",
            object_name="Denver",
            group_id=GROUP,
        )
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Alex",
            subject="Alex",
            predicate="LIVES_IN",
            include_expired=False,
        )
        objects = [f["object"] for f in facts]
        assert "Denver" not in objects

    @pytest.mark.asyncio
    async def test_forget_requires_valid_subject(self, rich_manager):
        result = await rich_manager.forget_fact(
            subject_name="NonExistent",
            predicate="LIVES_IN",
            object_name="Denver",
            group_id=GROUP,
        )
        assert result["status"] == "error"
        assert "not found" in result["message"].lower()


# ─── TestGetContext ──────────────────────────────────────────────────


class TestGetContext:
    @pytest.mark.asyncio
    async def test_get_context_without_topic(self, rich_manager):
        result = await rich_manager.get_context(group_id=GROUP)
        assert "context" in result
        assert "entity_count" in result
        assert "fact_count" in result
        assert "token_estimate" in result
        assert result["entity_count"] >= 0

    @pytest.mark.asyncio
    async def test_get_context_with_topic_hint(self, rich_manager):
        result = await rich_manager.get_context(
            group_id=GROUP,
            topic_hint="Engram",
        )
        assert "context" in result
        assert result["entity_count"] >= 0

    @pytest.mark.asyncio
    async def test_get_context_respects_max_tokens(self, rich_manager):
        result = await rich_manager.get_context(
            group_id=GROUP,
            max_tokens=50,
        )
        assert result["token_estimate"] <= 50

    @pytest.mark.asyncio
    async def test_get_context_empty_graph(self, tmp_path):
        """Empty graph returns valid context with zero counts."""
        db_path = str(tmp_path / "empty.db")
        gs = SQLiteGraphStore(db_path)
        await gs.initialize()
        acts = MemoryActivationStore(cfg=ActivationConfig())
        si = FTS5SearchIndex(db_path)
        await si.initialize(db=gs._db)
        ext = MockExtractor()
        mgr = GraphManager(gs, acts, si, ext)

        result = await mgr.get_context(group_id="empty")
        assert result["entity_count"] == 0
        assert result["fact_count"] == 0
        await gs.close()

    @pytest.mark.asyncio
    async def test_get_context_returns_markdown(self, rich_manager):
        result = await rich_manager.get_context(group_id=GROUP)
        ctx = result["context"]
        # Tiered context uses section headers like ## Identity, ## Recent Activity
        assert "##" in ctx

    @pytest.mark.asyncio
    async def test_get_context_returns_format_field(self, rich_manager):
        result = await rich_manager.get_context(group_id=GROUP)
        assert result.get("format") == "structured"


# ─── TestGetGraphState ───────────────────────────────────────────────


class TestGetGraphState:
    @pytest.mark.asyncio
    async def test_get_graph_state_basic_stats(self, rich_manager):
        result = await rich_manager.get_graph_state(group_id=GROUP)
        assert "stats" in result
        assert "top_activated" in result
        assert "group_id" in result
        assert result["stats"]["entities"] == 6
        assert result["stats"]["relationships"] == 5

    @pytest.mark.asyncio
    async def test_get_graph_state_top_activated(self, rich_manager):
        result = await rich_manager.get_graph_state(
            group_id=GROUP,
            top_n=3,
        )
        assert len(result["top_activated"]) <= 3
        if result["top_activated"]:
            names = [ta["name"] for ta in result["top_activated"]]
            assert "Engram" in names or "Alex" in names

    @pytest.mark.asyncio
    async def test_get_graph_state_with_edges(self, rich_manager):
        result = await rich_manager.get_graph_state(
            group_id=GROUP,
            top_n=5,
            include_edges=True,
        )
        assert "edges" in result
        assert len(result["edges"]) > 0
        edge = result["edges"][0]
        assert "source" in edge
        assert "target" in edge
        assert "predicate" in edge

    @pytest.mark.asyncio
    async def test_get_graph_state_filter_entity_types(self, rich_manager):
        result = await rich_manager.get_graph_state(
            group_id=GROUP,
            entity_types=["Technology"],
        )
        for ta in result["top_activated"]:
            assert ta["entity_type"] == "Technology"

    @pytest.mark.asyncio
    async def test_get_graph_state_type_distribution(self, rich_manager):
        result = await rich_manager.get_graph_state(group_id=GROUP)
        dist = result["stats"]["entity_type_distribution"]
        assert "Person" in dist
        assert "Technology" in dist
        assert dist["Technology"] == 2
        assert dist["Location"] == 2


class TestEpistemicArtifacts:
    @pytest.mark.asyncio
    async def test_bootstrap_creates_searchable_artifacts(self, rich_manager, tmp_path):
        (tmp_path / "README.md").write_text(
            "# Engram\nPublic launch path: OpenClaw distribution first.\n"
        )
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "design").mkdir()
        (tmp_path / "docs" / "design" / "launch.md").write_text(
            "# Launch\nWe decided to prioritize OpenClaw.\n"
        )

        result = await rich_manager.bootstrap_project(str(tmp_path), group_id=GROUP)
        assert result["status"] == "bootstrapped"

        hits = await rich_manager.search_artifacts(
            query="OpenClaw launch",
            group_id=GROUP,
            project_path=str(tmp_path),
            limit=5,
        )
        assert hits
        assert any("README.md" in hit.path or "launch.md" in hit.path for hit in hits)
        assert any(
            any(
                claim.object == "OpenClaw" or "OpenClaw" in claim.object
                for claim in hit.supporting_claims
            )
            for hit in hits
        )

    @pytest.mark.asyncio
    async def test_get_runtime_state_reports_artifact_freshness(self, rich_manager, tmp_path):
        (tmp_path / "README.md").write_text("# Engram\nCurrent default is rework.\n")
        await rich_manager.bootstrap_project(str(tmp_path), group_id=GROUP)

        state = await rich_manager.get_runtime_state(
            group_id=GROUP,
            project_path=str(tmp_path),
        )
        assert state["artifactBootstrap"]["artifactCount"] >= 1
        assert "epistemicMetrics" in state["stats"]

    @pytest.mark.asyncio
    async def test_bootstrap_refreshes_cached_runtime_state(self, rich_manager, tmp_path):
        from engram.ingestion.project_bootstrap import build_project_bootstrap_surface

        (tmp_path / "README.md").write_text("# Engram\nCache invalidation proof.\n")
        empty = await rich_manager.get_runtime_state(
            group_id=GROUP,
            project_path=str(tmp_path),
            live=True,
        )
        assert empty["artifactBootstrap"]["artifactCount"] == 0

        await build_project_bootstrap_surface(
            rich_manager,
            group_id=GROUP,
            project_path=str(tmp_path),
        )

        cached = await rich_manager.get_runtime_state(
            group_id=GROUP,
            project_path=str(tmp_path),
            live=False,
        )
        assert cached["artifactBootstrap"]["artifactCount"] >= 1
        assert cached["artifactBootstrap"]["lastObservedAt"] is not None
        assert cached["agentAdoption"]["status"] == "ready"
        assert cached["agentAdoption"]["requiredNextTools"] == ["get_context"]

    @pytest.mark.asyncio
    async def test_get_runtime_state_reports_artifacts_for_tmp_path_alias(
        self,
        rich_manager,
        tmp_path,
    ):
        (tmp_path / "README.md").write_text("# Engram\nBootstrap via tmp alias.\n")
        await rich_manager.bootstrap_project(str(tmp_path), group_id=GROUP)

        alias = str(tmp_path).replace("/private/tmp", "/tmp")
        if alias == str(tmp_path):
            pytest.skip("tmp_path is not under /private/tmp on this platform")

        state = await rich_manager.get_runtime_state(
            group_id=GROUP,
            project_path=alias,
            live=True,
        )
        assert state["artifactBootstrap"]["artifactCount"] >= 1
        assert state["artifactBootstrap"]["lastObservedAt"] is not None
        assert state["agentAdoption"]["status"] == "ready"

    @pytest.mark.asyncio
    async def test_gather_epistemic_evidence_prefers_artifacts_for_project_truth(
        self,
        rich_manager,
        tmp_path,
    ):
        rich_manager._cfg.artifact_bootstrap_enabled = True
        rich_manager._cfg.artifact_recall_enabled = True
        rich_manager._cfg.epistemic_runtime_executor_enabled = True
        rich_manager._cfg.epistemic_reconcile_enabled = True
        (tmp_path / "README.md").write_text(
            "# Engram\nThe public launch / distribution path is OpenClaw.\n"
        )
        await rich_manager.bootstrap_project(str(tmp_path), group_id=GROUP)

        bundle = await rich_manager.gather_epistemic_evidence(
            "what did we decide about launching Engram publicly?",
            group_id=GROUP,
            project_path=str(tmp_path),
            surface="rest",
        )

        assert bundle.question_frame.mode == "reconcile"
        assert bundle.reconciliation.status in {"artifact_only", "confirmed"}
        assert bundle.artifact_hits


# ─── TestSessionTracking ────────────────────────────────────────────


class TestSessionTracking:
    def test_session_id_is_uuid(self):
        session = SessionState()
        parsed = uuid.UUID(session.session_id)
        assert parsed.version == 4

    def test_session_defaults(self):
        session = SessionState(group_id="test")
        assert session.group_id == "test"
        assert session.episode_count == 0
        assert session.started_at is not None

    def test_session_episode_count_increments(self):
        session = SessionState()
        assert session.episode_count == 0
        session.episode_count += 1
        assert session.episode_count == 1


# ─── TestJSONResponses (via MCP tool wrappers) ──────────────────────


class TestJSONResponses:
    @pytest.mark.asyncio
    async def test_remember_returns_valid_json(self, graph_manager):
        """ingest_episode returns episode ID (MCP wraps with JSON)."""
        episode_id = await graph_manager.ingest_episode(
            content="Test memory content",
            group_id="default",
            source="test",
            session_id="test-session-123",
        )
        assert episode_id.startswith("ep_")

    @pytest.mark.asyncio
    async def test_recall_returns_list(self, graph_manager):
        """recall returns a list (MCP tool wraps with JSON)."""
        await graph_manager.ingest_episode(
            content="Python is great",
            group_id="default",
        )
        results = await graph_manager.recall(
            query="Python",
            group_id="default",
        )
        assert isinstance(results, list)
        if results:
            r = results[0]
            assert "entity" in r
            assert "score" in r
            assert "score_breakdown" in r

    @pytest.mark.asyncio
    async def test_mcp_remember_forwards_client_proposals(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            ingest_episode=AsyncMock(return_value="ep_test"),
            get_episode_adjudications=AsyncMock(return_value=[]),
            _triggered_intentions=[],
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_session", SessionState(group_id="default"))
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        monkeypatch.setattr(mcp_server, "_activation_cfg", None)
        monkeypatch.setattr(mcp_server, "_ingest_live_turn", AsyncMock())

        raw = await mcp_server.remember(
            content="Alice works at Google",
            proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
            proposed_relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            model_tier="opus",
        )

        manager.ingest_episode.assert_awaited_once()
        kwargs = manager.ingest_episode.await_args.kwargs
        assert kwargs["content"] == "Alice works at Google"
        assert kwargs["group_id"] == "default"
        assert kwargs["source"] == "mcp"
        assert kwargs["session_id"] == mcp_server._session.session_id
        assert kwargs["model_tier"] == "opus"
        assert kwargs["proposed_entities"][0]["name"] == "Alice"
        assert kwargs["proposed_relationships"][0]["predicate"] == "WORKS_AT"
        data = json.loads(raw)
        assert data["episode_id"] == "ep_test"
        assert data["operation"] == "remember"
        assert data["lifecycle"]["stage"] == "project"
        assert data["lifecycle"]["projection_status"] == "attempted"

    @pytest.mark.asyncio
    async def test_mcp_recall_packet_analysis_uses_active_group(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            recall=AsyncMock(
                return_value=[
                    {
                        "result_type": "entity",
                        "entity": {
                            "id": "ent_engram",
                            "name": "Engram",
                            "entity_type": "Project",
                            "summary": "Memory layer for AI agents",
                        },
                        "score": 0.9,
                    }
                ]
            ),
            get_recall_need_thresholds=Mock(return_value=None),
            get_last_near_miss_views=Mock(return_value=[]),
            get_surprise_connection_views=Mock(return_value=[]),
            get_recall_item_access_count=AsyncMock(return_value=0),
        )
        analyze = AsyncMock(return_value=SimpleNamespace())
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_session", SessionState(group_id=GROUP))
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)
        monkeypatch.setattr(
            mcp_server,
            "_activation_cfg",
            ActivationConfig(recall_packets_enabled=True),
        )
        from engram.retrieval import recall_surface

        monkeypatch.setattr(recall_surface, "analyze_memory_need", analyze)
        monkeypatch.setattr(recall_surface, "assemble_memory_packets", AsyncMock(return_value=[]))
        monkeypatch.setattr(mcp_server, "_recall_middleware", AsyncMock())

        raw = await mcp_server.recall("Engram packet routing", limit=3)

        payload = json.loads(raw)
        assert payload["operation"] == "recall"
        assert payload["query"] == "Engram packet routing"
        assert payload["lifecycle"]["stage"] == "recall"
        assert payload["lifecycle"]["recall_mode"] == "explicit"
        assert payload["total_candidates"] == 1
        analyze.assert_awaited_once()
        assert analyze.await_args.kwargs["group_id"] == GROUP

    @pytest.mark.asyncio
    async def test_mcp_recall_forwards_project_path(self, monkeypatch):
        from engram.mcp import server as mcp_server

        surface = AsyncMock(
            return_value={
                "operation": "recall",
                "query": "Engram packet routing",
                "results": [],
                "total_candidates": 0,
            }
        )
        monkeypatch.setattr(mcp_server, "_manager", SimpleNamespace())
        monkeypatch.setattr(mcp_server, "_session", SessionState(group_id=GROUP))
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)
        monkeypatch.setattr(mcp_server, "_activation_cfg", ActivationConfig())
        monkeypatch.setattr(
            mcp_server,
            "build_mcp_explicit_recall_tool_surface",
            surface,
        )

        raw = await mcp_server.recall(
            "Engram packet routing",
            limit=3,
            project_path="/tmp/engram",
        )

        payload = json.loads(raw)
        assert payload["operation"] == "recall"
        surface.assert_awaited_once()
        assert surface.await_args.kwargs["project_path"] == "/tmp/engram"

    @pytest.mark.asyncio
    async def test_mcp_recall_inherits_last_context_project_path(self, monkeypatch):
        from engram.mcp import server as mcp_server

        context_surface = AsyncMock(
            return_value={
                "context": "## Cached Memory Packets",
                "entity_count": 0,
                "fact_count": 0,
                "token_estimate": 4,
            }
        )
        recall_surface = AsyncMock(
            return_value={
                "operation": "recall",
                "query": "Engram packet routing",
                "results": [],
                "total_candidates": 0,
            }
        )
        session = SessionState(group_id=GROUP)
        monkeypatch.setattr(mcp_server, "_manager", SimpleNamespace())
        monkeypatch.setattr(mcp_server, "_session", session)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)
        monkeypatch.setattr(mcp_server, "_activation_cfg", ActivationConfig())
        monkeypatch.setattr(
            mcp_server,
            "build_mcp_context_tool_surface",
            context_surface,
        )
        monkeypatch.setattr(
            mcp_server,
            "build_mcp_explicit_recall_tool_surface",
            recall_surface,
        )

        await mcp_server.get_context(project_path="/tmp/engram")
        raw = await mcp_server.recall("Engram packet routing", limit=3)

        payload = json.loads(raw)
        assert payload["operation"] == "recall"
        assert session.last_project_path == "/tmp/engram"
        recall_surface.assert_awaited_once()
        assert recall_surface.await_args.kwargs["project_path"] == "/tmp/engram"

    @pytest.mark.asyncio
    async def test_mcp_claim_authority_returns_onboarding_contract(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            get_runtime_state=AsyncMock(
                return_value={
                    "runtime": {"mode": "helix"},
                    "artifactBootstrap": {
                        "enabled": True,
                        "projectPath": "/tmp/engram",
                        "artifactCount": 0,
                        "staleArtifactCount": 0,
                        "lastObservedAt": None,
                    },
                    "stats": {"recallMetrics": {}, "epistemicMetrics": {}},
                }
            )
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)

        raw = await mcp_server.claim_authority(
            project_path="/tmp/engram",
            user_message="I am actively building Engram for cross-context AI memory.",
            file_memory_present=True,
        )

        payload = json.loads(raw)
        assert payload["authority"]["source_of_truth"] == "portable_cross_context_memory"
        assert payload["onboarding"]["state"] == "fresh_runtime"
        assert payload["onboarding"]["recommended_actions"][0]["tool"] == "bootstrap_project"
        assert payload["agent_protocol"]["file_memory_present"] is True
        assert payload["agent_protocol"]["capture"]["tool"] == "remember"
        assert payload["agent_protocol"]["required_tools_before_answer"] == [
            "bootstrap_project",
            "get_context",
            "recall",
        ]
        assert payload["agent_protocol"]["verification"]["command"] == (
            "engram adoption --authority claim-authority.json --calls mcp-calls.jsonl"
        )
        assert payload["agent_protocol"]["verification"]["live_evidence_command"] == (
            "engram adoption --authority claim-authority.json "
            "--calls live-harness-transcript.json --require-live-evidence"
        )
        assert payload["agent_protocol"]["verification"]["live_evidence_schema"][
            "required_metadata_fields"
        ] == ["client", "capturedAt", "source"]
        assert payload["agent_protocol"]["verification"]["capture_required"] is True
        manager.get_runtime_state.assert_awaited_once_with(
            group_id=GROUP,
            project_path="/tmp/engram",
            live=False,
            timeout_seconds=None,
        )

    @pytest.mark.asyncio
    async def test_mcp_remember_surfaces_adjudication_requests(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            ingest_episode=AsyncMock(return_value="ep_test"),
            get_episode_adjudications=AsyncMock(
                return_value=[
                    {
                        "request_id": "adj_123",
                        "ambiguity_tags": ["negation_scope"],
                        "selected_text": "Alice works at Google, but maybe not anymore.",
                        "candidate_evidence": [
                            {
                                "evidence_id": "evi_1",
                                "fact_class": "relationship",
                                "payload": {"subject": "Alice"},
                            },
                        ],
                        "instructions": "Resolve only if highly confident.",
                    },
                ],
            ),
            _triggered_intentions=[],
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_session", SessionState(group_id="default"))
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        monkeypatch.setattr(
            mcp_server,
            "_activation_cfg",
            ActivationConfig(
                evidence_extraction_enabled=True,
                edge_adjudication_client_enabled=True,
            ),
        )
        monkeypatch.setattr(mcp_server, "_ingest_live_turn", AsyncMock())
        monkeypatch.setattr(mcp_server, "_auto_recall_lite", AsyncMock(return_value=None))
        monkeypatch.setattr(mcp_server, "_session_prime", AsyncMock(return_value=None))

        raw = await mcp_server.remember(content="ambiguous memory")

        data = json.loads(raw)
        assert data["adjudication_requests"][0]["request_id"] == "adj_123"
        assert data["operation"] == "remember"

    @pytest.mark.asyncio
    async def test_mcp_adjudicate_evidence_forwards_resolution(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            submit_adjudication_resolution=AsyncMock(
                return_value=SimpleNamespace(
                    status="materialized",
                    request_id="adj_123",
                    committed_ids={"evi_new": "rel_1"},
                    superseded_evidence_ids=["evi_old"],
                    replacement_evidence_ids=["evi_new"],
                ),
            ),
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_group_id", "default")

        raw = await mcp_server.adjudicate_evidence(
            request_id="adj_123",
            entities=[{"name": "Alice", "entity_type": "Person"}],
            relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            reject_evidence_ids=["evi_old"],
            model_tier="opus",
            rationale="high confidence resolution",
        )

        manager.submit_adjudication_resolution.assert_awaited_once_with(
            "adj_123",
            entities=[{"name": "Alice", "entity_type": "Person"}],
            relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            reject_evidence_ids=["evi_old"],
            source="client_adjudication",
            model_tier="opus",
            rationale="high confidence resolution",
            group_id="default",
        )
        assert '"status": "materialized"' in raw

    @pytest.mark.asyncio
    async def test_mcp_feedback_returns_error_payload_for_invalid_rating(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(record_explicit_feedback=AsyncMock())
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)

        raw = await mcp_server.feedback(entity_id="ent_1", rating=6)

        assert json.loads(raw) == {"error": "Rating must be between 1 and 5"}
        manager.record_explicit_feedback.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_records_recall_evaluation_sample(self, monkeypatch):
        from engram.mcp import server as mcp_server

        store = SimpleNamespace(save_recall_sample=AsyncMock())
        monkeypatch.setattr(mcp_server, "_evaluation_store", store)
        monkeypatch.setattr(mcp_server, "_group_id", "default")

        raw = await mcp_server.record_recall_evaluation(
            recall_triggered=True,
            recall_helped=False,
            recall_needed=True,
            packets_surfaced=4,
            packets_used=1,
            false_recalls=2,
            stale_packets=1,
            corrected_packets=1,
            query="What did I decide?",
            notes="one misleading packet",
        )

        store.save_recall_sample.assert_awaited_once()
        sample = store.save_recall_sample.await_args.args[0]
        assert isinstance(sample, StoredRecallEvalSample)
        assert sample.group_id == "default"
        assert sample.source == "mcp"
        assert sample.recall_needed is True
        assert sample.false_recalls == 2
        assert sample.stale_packets == 1
        assert sample.corrected_packets == 1

        data = json.loads(raw)
        assert data["status"] == "stored"
        assert data["operation"] == "record_recall_evaluation"
        assert data["group_id"] == "default"
        assert data["sample"]["recall_triggered"] is True
        assert data["sample"]["recall_needed"] is True
        assert data["sample"]["packets_used"] == 1
        assert data["sample"]["stale_packets"] == 1
        assert data["sample"]["corrected_packets"] == 1

    @pytest.mark.asyncio
    async def test_mcp_records_session_continuity_evaluation_sample(self, monkeypatch):
        from engram.mcp import server as mcp_server

        store = SimpleNamespace(save_session_sample=AsyncMock())
        monkeypatch.setattr(mcp_server, "_evaluation_store", store)
        monkeypatch.setattr(mcp_server, "_group_id", "default")

        raw = await mcp_server.record_session_continuity_evaluation(
            baseline_score=0.2,
            memory_score=0.8,
            open_loop_expected=True,
            open_loop_recovered=True,
            temporal_expected=True,
            temporal_correct=False,
            scenario="open-loop follow-up",
        )

        store.save_session_sample.assert_awaited_once()
        sample = store.save_session_sample.await_args.args[0]
        assert isinstance(sample, StoredSessionContinuitySample)
        assert sample.group_id == "default"
        assert sample.scenario == "open-loop follow-up"
        assert sample.open_loop_recovered is True

        data = json.loads(raw)
        assert data["status"] == "stored"
        assert data["operation"] == "record_session_continuity_evaluation"
        assert data["sample"]["memory_score"] == 0.8
        assert data["sample"]["temporal_correct"] is False

    @pytest.mark.asyncio
    async def test_mcp_get_evaluation_report_uses_saved_samples(self, monkeypatch):
        from engram.mcp import server as mcp_server

        recall_sample = StoredRecallEvalSample(
            group_id="default",
            recall_triggered=True,
            recall_helped=True,
            packets_surfaced=3,
            packets_used=2,
            false_recalls=1,
        )
        session_sample = StoredSessionContinuitySample(
            group_id="default",
            baseline_score=0.2,
            memory_score=0.7,
            open_loop_expected=True,
            open_loop_recovered=True,
            temporal_expected=True,
            temporal_correct=False,
        )
        manager = SimpleNamespace(
            get_graph_state=AsyncMock(
                return_value={
                    "stats": {
                        "episodes": 1,
                        "entities": 2,
                        "relationships": 1,
                        "cue_metrics": {
                            "cue_count": 1,
                            "cue_surfaced_count": 1,
                            "cue_used_count": 1,
                            "cue_to_projection_conversion_rate": 1.0,
                        },
                        "projection_metrics": {
                            "state_counts": {"projected": 1},
                            "yield": {
                                "linked_entity_count": 2,
                                "relationship_count": 1,
                                "avg_linked_entities_per_projected_episode": 2.0,
                            },
                        },
                        "recall_metrics": {
                            "total_analyses": 1,
                            "trigger_count": 1,
                            "used_count": 2,
                            "dismissed_count": 1,
                            "graph_override_count": 1,
                            "adaptive_thresholds_enabled": True,
                            "thresholds": {
                                "linguistic": 0.34,
                                "borderline": 0.2,
                                "resonance": 0.52,
                            },
                            "analyzer_latency_ms": {"avg": 9.5, "p95": 15.0},
                            "probe_latency_ms": {"avg": 4.0, "p95": 8.0},
                        },
                    }
                }
            )
        )
        store = SimpleNamespace(
            get_recall_samples=AsyncMock(return_value=[recall_sample.to_sample()]),
            get_session_samples=AsyncMock(return_value=[session_sample.to_sample()]),
            get_latest_recall_metrics_snapshot=AsyncMock(return_value={}),
            save_recall_metrics_snapshot=AsyncMock(),
            get_latest_memory_operation_metrics_snapshot=AsyncMock(return_value={}),
            save_memory_operation_metrics_snapshot=AsyncMock(),
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_evaluation_store", store)
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        cycle = SimpleNamespace(
            id="cyc_mcp_eval",
            status="completed",
            phase_results=[
                SimpleNamespace(
                    phase="triage",
                    status="success",
                    items_processed=2,
                    items_affected=1,
                )
            ],
        )
        calibration = SimpleNamespace(
            phase="triage",
            total_traces=3,
            labeled_examples=2,
            oracle_examples=0,
            abstain_count=0,
            accuracy=1.0,
            mean_confidence=0.8,
            expected_calibration_error=0.0,
        )
        monkeypatch.setattr(
            mcp_server,
            "_consolidation_store",
            SimpleNamespace(
                get_recent_cycles=AsyncMock(return_value=[cycle]),
                get_calibration_snapshots=AsyncMock(return_value=[calibration]),
            ),
        )

        raw = await mcp_server.get_evaluation_report(sample_limit=50)

        manager.get_graph_state.assert_awaited_once_with(
            group_id="default",
            top_n=10,
            include_edges=False,
        )
        store.get_recall_samples.assert_awaited_once_with("default", limit=50)
        store.get_session_samples.assert_awaited_once_with("default", limit=50)
        store.get_latest_recall_metrics_snapshot.assert_not_awaited()
        store.save_recall_metrics_snapshot.assert_awaited_once()
        snapshot = store.save_recall_metrics_snapshot.await_args.args[0]
        assert snapshot.group_id == "default"
        assert snapshot.metrics["total_analyses"] == 1
        data = json.loads(raw)
        assert data["group_id"] == "default"
        assert data["recall"]["evaluation"]["status"] == "measured"
        assert data["recall"]["evaluation"]["memory_need_precision"] == 1.0
        assert data["recall"]["latency"]["analyzer_ms"]["p95_ms"] == 15.0
        assert data["recall"]["latency"]["probe_ms"]["avg_ms"] == 4.0
        assert data["recall"]["control"]["graph_override_count"] == 1
        assert data["recall"]["control"]["thresholds"]["resonance"] == 0.52
        assert data["recall"]["continuity"]["session_continuity_lift"] == 0.5
        assert {signal["status"] for signal in data["evaluation_signals"].values()} == {"measured"}
        assert data["evaluation_signals"]["cue_usefulness"]["evidence_count"] == 1
        assert data["evaluation_signals"]["projection_yield"]["metric"] == 2.0
        assert data["evaluation_signals"]["false_recall"]["metric"] == 0.3333
        assert data["evaluation_signals"]["triage_calibration"]["metric"] == 0.0
        assert data["evaluation_signals"]["consolidation_effect"]["metric"] == 0.5

    @pytest.mark.asyncio
    async def test_mcp_get_evaluation_report_uses_saved_recall_runtime_snapshot(
        self,
        monkeypatch,
    ):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            get_graph_state=AsyncMock(
                return_value={
                    "stats": {
                        "episodes": 1,
                        "entities": 2,
                        "relationships": 1,
                        "recall_metrics": {"total_analyses": 0},
                    }
                }
            )
        )
        store = SimpleNamespace(
            get_recall_samples=AsyncMock(return_value=[]),
            get_session_samples=AsyncMock(return_value=[]),
            get_latest_recall_metrics_snapshot=AsyncMock(
                return_value={
                    "total_analyses": 3,
                    "trigger_count": 2,
                    "analyzer_latency_ms": {"avg": 9.0, "p95": 18.0},
                    "probe_latency_ms": {"avg": 4.0, "p95": 8.0},
                    "surfaced_count": 5,
                    "thresholds": {"resonance": 0.5},
                }
            ),
            save_recall_metrics_snapshot=AsyncMock(),
            get_latest_memory_operation_metrics_snapshot=AsyncMock(return_value={}),
            save_memory_operation_metrics_snapshot=AsyncMock(),
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_evaluation_store", store)
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        monkeypatch.setattr(
            mcp_server,
            "_consolidation_store",
            SimpleNamespace(get_recent_cycles=AsyncMock(return_value=[])),
        )

        raw = await mcp_server.get_evaluation_report(sample_limit=50)

        store.get_latest_recall_metrics_snapshot.assert_awaited_once_with("default")
        store.save_recall_metrics_snapshot.assert_not_awaited()
        data = json.loads(raw)
        assert data["recall"]["total_analyses"] == 3
        assert data["recall"]["trigger_count"] == 2
        assert data["recall"]["latency"]["analyzer_ms"]["p95_ms"] == 18.0
        assert data["recall"]["latency"]["probe_ms"]["avg_ms"] == 4.0
        assert data["recall"]["control"]["surfaced_count"] == 5
        assert data["recall"]["control"]["thresholds"]["resonance"] == 0.5
        assert "recall gate needs runtime analyses" not in data["coverage_gaps"]

    @pytest.mark.asyncio
    async def test_mcp_evaluation_report_reads_active_consolidation_store(self):
        from engram.evaluation.report_service import load_consolidation_evaluation_inputs

        cycle = SimpleNamespace(id="cyc_native", phase_results=[])
        store = SimpleNamespace(
            get_recent_cycles=AsyncMock(return_value=[cycle]),
            get_calibration_snapshots=AsyncMock(return_value=["snapshot"]),
        )

        cycles, snapshots = await load_consolidation_evaluation_inputs(
            store,
            group_id="native_brain",
            cycle_limit=7,
        )

        assert cycles == [cycle]
        assert snapshots == ["snapshot"]
        store.get_recent_cycles.assert_awaited_once_with("native_brain", limit=7)
        store.get_calibration_snapshots.assert_awaited_once_with("cyc_native", "native_brain")

    @pytest.mark.asyncio
    async def test_mcp_shutdown_closes_runtime_resources(self, monkeypatch):
        from engram.mcp import server as mcp_server

        worker = SimpleNamespace(stop=AsyncMock())
        publisher = SimpleNamespace(close=AsyncMock())
        evaluation_store = SimpleNamespace(close=AsyncMock())
        consolidation_store = SimpleNamespace(close=AsyncMock())
        manager = SimpleNamespace(close_runtime_resources=AsyncMock())
        bus = SimpleNamespace(remove_on_publish_hook=Mock())

        monkeypatch.setattr(mcp_server, "_episode_worker", worker)
        monkeypatch.setattr(mcp_server, "_redis_publisher", publisher)
        monkeypatch.setattr(mcp_server, "_evaluation_store", evaluation_store)
        monkeypatch.setattr(mcp_server, "_consolidation_store", consolidation_store)
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_session", SimpleNamespace())
        monkeypatch.setattr(mcp_server, "_recall_cooldown", SimpleNamespace())
        monkeypatch.setattr(mcp_server, "_activation_cfg", ActivationConfig())
        monkeypatch.setattr(mcp_server, "get_event_bus", lambda: bus)

        await mcp_server._shutdown()

        worker.stop.assert_awaited_once()
        bus.remove_on_publish_hook.assert_called_once_with(publisher)
        publisher.close.assert_awaited_once()
        evaluation_store.close.assert_awaited_once()
        consolidation_store.close.assert_awaited_once()
        manager.close_runtime_resources.assert_awaited_once()
        assert mcp_server._manager is None
        assert mcp_server._evaluation_store is None
        assert mcp_server._consolidation_store is None

    @pytest.mark.asyncio
    async def test_mcp_init_defers_evaluation_and_consolidation_stores(
        self,
        monkeypatch,
    ):
        from engram.mcp import server as mcp_server

        config = EngramConfig(_env_file=None)
        config.activation.worker_enabled = False
        config.activation.cue_index_outbox_enabled = False
        graph_store = SimpleNamespace(initialize=AsyncMock())
        activation_store = SimpleNamespace()
        search_index = SimpleNamespace()
        manager = SimpleNamespace(
            drain_cue_index_outbox=AsyncMock(return_value=0),
            close_runtime_resources=AsyncMock(),
        )
        evaluation_factory = AsyncMock(return_value=SimpleNamespace())
        consolidation_factory = AsyncMock(return_value=SimpleNamespace())

        monkeypatch.setattr(mcp_server, "_manager", None)
        monkeypatch.setattr(mcp_server, "_session", None)
        monkeypatch.setattr(mcp_server, "_evaluation_store", None)
        monkeypatch.setattr(mcp_server, "_consolidation_store", None)
        monkeypatch.setattr(mcp_server, "_runtime_config", None)
        monkeypatch.setattr(mcp_server, "_runtime_mode", None)
        monkeypatch.setattr(mcp_server, "_runtime_graph_store", None)
        monkeypatch.setattr(mcp_server, "_mcp_init_timings", {})
        monkeypatch.setattr(mcp_server, "EngramConfig", lambda: config)
        monkeypatch.setattr(
            mcp_server,
            "resolve_mode",
            AsyncMock(return_value=mcp_server.EngineMode.LITE),
        )
        monkeypatch.setattr(
            mcp_server,
            "create_stores",
            Mock(return_value=(graph_store, activation_store, search_index)),
        )
        monkeypatch.setattr(mcp_server, "initialize_search_index_for_graph", AsyncMock())
        monkeypatch.setattr(mcp_server, "create_extractor", Mock(return_value=object()))
        monkeypatch.setattr(mcp_server, "GraphManager", Mock(return_value=manager))
        monkeypatch.setattr(
            mcp_server,
            "create_evaluation_store_for_graph",
            evaluation_factory,
        )
        monkeypatch.setattr(
            mcp_server,
            "create_consolidation_store_for_graph",
            consolidation_factory,
        )

        await mcp_server._init()

        graph_store.initialize.assert_awaited_once()
        mcp_server.initialize_search_index_for_graph.assert_awaited_once()
        evaluation_factory.assert_not_awaited()
        consolidation_factory.assert_not_awaited()
        assert mcp_server._evaluation_store is None
        assert mcp_server._consolidation_store is None
        assert "mcp_evaluation_store_init" not in mcp_server._mcp_init_timings
        assert "mcp_consolidation_store_init" not in mcp_server._mcp_init_timings
        assert mcp_server._mcp_init_timings["mcp_init"] >= 0

    @pytest.mark.asyncio
    async def test_mcp_lazy_store_helpers_initialize_on_first_use(self, monkeypatch):
        from engram.mcp import server as mcp_server

        config = EngramConfig(_env_file=None)
        graph_store = SimpleNamespace()
        evaluation_store = SimpleNamespace()
        consolidation_store = SimpleNamespace()
        evaluation_factory = AsyncMock(return_value=evaluation_store)
        consolidation_factory = AsyncMock(return_value=consolidation_store)

        monkeypatch.setattr(mcp_server, "_evaluation_store", None)
        monkeypatch.setattr(mcp_server, "_consolidation_store", None)
        monkeypatch.setattr(mcp_server, "_runtime_config", config)
        monkeypatch.setattr(mcp_server, "_runtime_mode", mcp_server.EngineMode.LITE)
        monkeypatch.setattr(mcp_server, "_runtime_graph_store", graph_store)
        monkeypatch.setattr(mcp_server, "_mcp_init_timings", {})
        monkeypatch.setattr(
            mcp_server,
            "create_evaluation_store_for_graph",
            evaluation_factory,
        )
        monkeypatch.setattr(
            mcp_server,
            "create_consolidation_store_for_graph",
            consolidation_factory,
        )

        assert await mcp_server._get_evaluation_store() is evaluation_store
        assert await mcp_server._get_evaluation_store() is evaluation_store
        assert await mcp_server._get_consolidation_store() is consolidation_store
        assert await mcp_server._get_consolidation_store() is consolidation_store

        evaluation_factory.assert_awaited_once_with(
            config,
            graph_store=graph_store,
            mode=mcp_server.EngineMode.LITE,
        )
        consolidation_factory.assert_awaited_once_with(
            config,
            graph_store=graph_store,
            mode=mcp_server.EngineMode.LITE,
        )
        assert mcp_server._mcp_init_timings["mcp_evaluation_store_lazy_init"] >= 0
        assert mcp_server._mcp_init_timings["mcp_consolidation_store_lazy_init"] >= 0

    @pytest.mark.asyncio
    async def test_mcp_trigger_consolidation_includes_failure_errors(self, monkeypatch):
        from engram.consolidation_trigger import ConsolidationTriggerResult
        from engram.mcp import server as mcp_server

        cycle = SimpleNamespace(
            id="cyc_mcp_failed",
            status="failed",
            error="Phase 'triage' requires graph_store methods: missing_method",
            dry_run=True,
            trigger="mcp",
            started_at=1.0,
            completed_at=2.0,
            total_duration_ms=7.5,
            phase_results=[
                SimpleNamespace(
                    phase="triage",
                    status="error",
                    items_processed=0,
                    items_affected=0,
                    duration_ms=7.5,
                    error="missing_method",
                )
            ],
        )

        manager = SimpleNamespace(
            get_consolidation_shared_db=lambda: None,
            trigger_consolidation_cycle=AsyncMock(
                return_value=ConsolidationTriggerResult(
                    cycle=cycle,
                    graph_stats={"episodes": 3},
                )
            ),
        )

        monkeypatch.setattr(mcp_server, "_get_manager", lambda: manager)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)

        payload = json.loads(await mcp_server.trigger_consolidation(dry_run=True))

        manager.trigger_consolidation_cycle.assert_awaited_once_with(
            group_id=GROUP,
            trigger="mcp",
            dry_run=True,
            consolidation_store=None,
        )
        assert payload["status"] == "failed"
        assert payload["error"] == "Phase 'triage' requires graph_store methods: missing_method"
        assert payload["phases"][0]["error"] == "missing_method"
        assert payload["summary"]["total_processed"] == 0
        assert payload["summary"]["total_affected"] == 0
        assert payload["summary"]["description"].startswith("Dry run failed cycle:")

    @pytest.mark.asyncio
    async def test_mcp_trigger_consolidation_uses_active_audit_store_for_native_graph(
        self,
        monkeypatch,
    ):
        from engram.consolidation_trigger import ConsolidationTriggerResult
        from engram.mcp import server as mcp_server

        active_store = SimpleNamespace(name="active-native-store")
        cycle = SimpleNamespace(
            id="cyc_mcp_native_store",
            status="completed",
            error=None,
            dry_run=True,
            trigger="mcp",
            started_at=1.0,
            completed_at=2.0,
            total_duration_ms=7.5,
            phase_results=[],
        )

        manager = SimpleNamespace(
            get_consolidation_shared_db=lambda: None,
            trigger_consolidation_cycle=AsyncMock(
                return_value=ConsolidationTriggerResult(
                    cycle=cycle,
                    graph_stats={"episodes": 3},
                )
            ),
        )

        monkeypatch.setattr(mcp_server, "_get_manager", lambda: manager)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)
        monkeypatch.setattr(mcp_server, "_consolidation_store", active_store)

        payload = json.loads(await mcp_server.trigger_consolidation(dry_run=True))

        manager.trigger_consolidation_cycle.assert_awaited_once_with(
            group_id=GROUP,
            trigger="mcp",
            dry_run=True,
            consolidation_store=active_store,
        )
        assert payload["status"] == "completed"
        assert payload["cycle_id"] == "cyc_mcp_native_store"

    @pytest.mark.asyncio
    async def test_mcp_trigger_consolidation_reports_completed_phase_warnings(self, monkeypatch):
        from engram.consolidation_trigger import ConsolidationTriggerResult
        from engram.mcp import server as mcp_server

        cycle = SimpleNamespace(
            id="cyc_mcp_phase_warning",
            status="completed",
            error=None,
            dry_run=True,
            trigger="mcp",
            started_at=1.0,
            completed_at=2.0,
            total_duration_ms=7.5,
            phase_results=[
                SimpleNamespace(
                    phase="graph_embed",
                    status="error",
                    items_processed=1,
                    items_affected=0,
                    duration_ms=7.5,
                    error="optional vector index unavailable",
                )
            ],
        )

        manager = SimpleNamespace(
            get_consolidation_shared_db=lambda: None,
            trigger_consolidation_cycle=AsyncMock(
                return_value=ConsolidationTriggerResult(
                    cycle=cycle,
                    graph_stats={"episodes": 3},
                )
            ),
        )

        monkeypatch.setattr(mcp_server, "_get_manager", lambda: manager)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)

        payload = json.loads(await mcp_server.trigger_consolidation(dry_run=True))

        assert payload["status"] == "completed"
        assert payload["phase_issue"] == "graph_embed: optional vector index unavailable"
        assert payload["phases"][0]["error"] == "optional vector index unavailable"
        assert payload["summary"]["total_processed"] == 1
        assert payload["summary"]["total_affected"] == 0
        assert payload["summary"]["description"].startswith("Dry run cycle with warnings:")

    @pytest.mark.asyncio
    async def test_mcp_consolidation_status_includes_latest_cycle(self, monkeypatch):
        from engram.mcp import server as mcp_server

        cycle = SimpleNamespace(
            id="cyc_status_failed",
            status="failed",
            error="calibration failed",
            dry_run=True,
            trigger="mcp",
            started_at=1.0,
            completed_at=2.0,
            total_duration_ms=7.5,
            phase_results=[
                SimpleNamespace(
                    phase="calibrate",
                    status="error",
                    items_processed=2,
                    items_affected=0,
                    duration_ms=7.5,
                    error="no teacher labels",
                )
            ],
        )
        store = SimpleNamespace(get_recent_cycles=AsyncMock(return_value=[cycle]))

        monkeypatch.setattr(mcp_server, "_consolidation_store", store)
        monkeypatch.setattr(mcp_server, "_group_id", GROUP)

        payload = json.loads(await mcp_server.get_consolidation_status())

        store.get_recent_cycles.assert_awaited_once_with(GROUP, limit=1)
        assert payload["is_running"] is False
        assert payload["latest_cycle"]["id"] == "cyc_status_failed"
        assert payload["latest_cycle"]["error"] == "calibration failed"
        assert payload["latest_cycle"]["phases"][0]["error"] == "no teacher labels"
        assert payload["latest_cycle"]["summary"]["description"].startswith("Dry run failed cycle:")

    @pytest.mark.asyncio
    async def test_mcp_get_lifecycle_summary_uses_manager_facade(self, monkeypatch):
        from engram.mcp import server as mcp_server

        consolidation_store = SimpleNamespace(
            get_recent_cycles=AsyncMock(return_value=[]),
        )
        activation_config = ActivationConfig(decay_exponent=0.4)
        manager = SimpleNamespace(
            get_lifecycle_summary=AsyncMock(
                return_value={
                    "groupId": "default",
                    "loop": ["capture", "cue", "project", "recall", "consolidate"],
                    "totals": {"episodes": 2, "cues": 2},
                }
            )
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        monkeypatch.setattr(mcp_server, "_activation_cfg", activation_config)
        monkeypatch.setattr(mcp_server, "_consolidation_store", consolidation_store)

        raw = await mcp_server.get_lifecycle_summary(episode_limit=1, cycle_limit=3)

        manager.get_lifecycle_summary.assert_awaited_once()
        kwargs = manager.get_lifecycle_summary.await_args.kwargs
        assert kwargs["group_id"] == "default"
        assert kwargs["activation_config"] is activation_config
        assert kwargs["episode_limit"] == 1
        assert kwargs["cycle_limit"] == 3
        assert kwargs["consolidation_reader"].available is True
        assert kwargs["consolidation_engine"].is_running is False
        data = json.loads(raw)
        assert data["groupId"] == "default"
        assert data["loop"] == ["capture", "cue", "project", "recall", "consolidate"]
        assert data["totals"]["episodes"] == 2
        assert data["totals"]["cues"] == 2

    @pytest.mark.asyncio
    async def test_mcp_get_lifecycle_summary_clamps_limits(self, monkeypatch):
        from engram.mcp import server as mcp_server

        manager = SimpleNamespace(
            get_lifecycle_summary=AsyncMock(
                return_value={
                    "groupId": "default",
                    "loop": ["capture", "cue", "project", "recall", "consolidate"],
                    "totals": {},
                }
            )
        )
        monkeypatch.setattr(mcp_server, "_manager", manager)
        monkeypatch.setattr(mcp_server, "_group_id", "default")
        monkeypatch.setattr(mcp_server, "_activation_cfg", None)
        monkeypatch.setattr(mcp_server, "_consolidation_store", None)

        await mcp_server.get_lifecycle_summary(episode_limit=0, cycle_limit=0)

        manager.get_lifecycle_summary.assert_awaited_once_with(
            group_id="default",
            consolidation_engine=None,
            consolidation_reader=None,
            activation_config=None,
            episode_limit=1,
            cycle_limit=1,
        )


# ─── TestResolveEntityName ───────────────────────────────────────────


class TestResolveEntityName:
    @pytest.mark.asyncio
    async def test_resolve_existing_entity(self, rich_manager):
        name = await rich_manager.resolve_entity_name(
            "ent_alex",
            GROUP,
        )
        assert name == "Alex"

    @pytest.mark.asyncio
    async def test_resolve_missing_entity_returns_id(self, rich_manager):
        name = await rich_manager.resolve_entity_name(
            "ent_nonexistent",
            GROUP,
        )
        assert name == "ent_nonexistent"


# ─── TestClearActivation ────────────────────────────────────────────


class TestClearActivation:
    @pytest.mark.asyncio
    async def test_clear_activation_removes_state(self):
        store = MemoryActivationStore()
        await store.record_access("ent_1", time.time())
        assert await store.get_activation("ent_1") is not None

        await store.clear_activation("ent_1")
        assert await store.get_activation("ent_1") is None

    @pytest.mark.asyncio
    async def test_clear_activation_noop_for_missing(self):
        store = MemoryActivationStore()
        await store.clear_activation("nonexistent")


# ─── TestGetEntityTypeCounts ─────────────────────────────────────────


class TestGetEntityTypeCounts:
    @pytest.mark.asyncio
    async def test_returns_correct_counts(self, rich_manager):
        counts = await rich_manager._graph.get_entity_type_counts(GROUP)
        assert counts["Person"] == 1
        assert counts["Technology"] == 2
        assert counts["Location"] == 2
        assert counts["Project"] == 1

    @pytest.mark.asyncio
    async def test_empty_graph_returns_empty_dict(self, tmp_path):
        db_path = str(tmp_path / "empty_counts.db")
        gs = SQLiteGraphStore(db_path)
        await gs.initialize()
        counts = await gs.get_entity_type_counts("nonexistent")
        assert counts == {}
        await gs.close()
