"""Tests for Week 4 MCP tools — exercises GraphManager methods directly."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.mcp.server import SessionState
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
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
        _entity("ent_konner", "Konner", "Person", "Software engineer"),
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
        _rel("rel_builds", "ent_konner", "ent_engram", "BUILDS", now_dt),
        _rel("rel_uses_py", "ent_engram", "ent_python", "USES", now_dt),
        _rel("rel_uses_fa", "ent_engram", "ent_fastapi", "USES", now_dt),
        _rel("rel_lives", "ent_konner", "ent_denver", "LIVES_IN", now_dt),
        _rel("rel_moved", "ent_konner", "ent_mesa", "MOVED_FROM", now_dt),
    ]
    for r in rels:
        await graph_store.create_relationship(r)

    now = time.time()
    await activation_store.record_access("ent_konner", now - 10, group_id=GROUP)
    await activation_store.record_access("ent_konner", now - 5, group_id=GROUP)
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

    now_dt = datetime.utcnow()
    past_dt = now_dt - timedelta(days=30)

    entities = [
        _entity("ent_konner2", "Konner", "Person", "Software engineer"),
        _entity("ent_acme", "Acme Corp", "Organization", "Previous employer"),
        _entity("ent_vercel", "Vercel", "Organization", "Current employer"),
    ]
    for e in entities:
        await graph_store.create_entity(e)

    rels = [
        _rel(
            "rel_old_job",
            "ent_konner2",
            "ent_acme",
            "WORKS_AT",
            past_dt,
            valid_to=now_dt,
        ),
        _rel(
            "rel_new_job",
            "ent_konner2",
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


# ─── TestSearchEntities ──────────────────────────────────────────────


class TestSearchEntities:
    @pytest.mark.asyncio
    async def test_search_by_name_finds_entity(self, rich_manager):
        results = await rich_manager.search_entities(
            group_id=GROUP,
            name="Konner",
        )
        assert len(results) >= 1
        names = [r["name"] for r in results]
        assert "Konner" in names

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
            name="Konner",
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
            "ent_konner",
        )
        count_before = state_before.access_count if state_before else 0

        await rich_manager.search_entities(group_id=GROUP, name="Konner")

        state_after = await rich_manager._activation.get_activation(
            "ent_konner",
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
            query="Konner",
            subject="Konner",
        )
        assert len(facts) >= 1
        subjects = [f["subject"] for f in facts]
        assert "Konner" in subjects

    @pytest.mark.asyncio
    async def test_search_with_predicate_filter(self, rich_manager):
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Konner",
            subject="Konner",
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
            query="Konner",
            subject="Konner",
            predicate="WORKS_AT",
            include_expired=False,
        )
        active_objects = [f["object"] for f in active_facts]
        assert "Vercel" in active_objects
        assert "Acme Corp" not in active_objects

        all_facts = await rich_manager_with_expired.search_facts(
            group_id=GROUP,
            query="Konner",
            subject="Konner",
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
            subject_name="Konner",
            predicate="LIVES_IN",
            object_name="Denver",
            group_id=GROUP,
        )
        assert result["status"] == "forgotten"
        assert result["target_type"] == "fact"

        rels = await rich_manager._graph.get_relationships(
            "ent_konner",
            direction="outgoing",
            predicate="LIVES_IN",
            active_only=True,
        )
        target_ids = [r.target_id for r in rels]
        assert "ent_denver" not in target_ids

    @pytest.mark.asyncio
    async def test_forget_fact_not_found(self, rich_manager):
        result = await rich_manager.forget_fact(
            subject_name="Konner",
            predicate="WORKS_AT",
            object_name="Denver",
            group_id=GROUP,
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_forgotten_fact_not_in_search_facts(self, rich_manager):
        await rich_manager.forget_fact(
            subject_name="Konner",
            predicate="LIVES_IN",
            object_name="Denver",
            group_id=GROUP,
        )
        facts = await rich_manager.search_facts(
            group_id=GROUP,
            query="Konner",
            subject="Konner",
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
        assert "## Active Memory Context" in ctx


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
            assert "Engram" in names or "Konner" in names

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


# ─── TestResolveEntityName ───────────────────────────────────────────


class TestResolveEntityName:
    @pytest.mark.asyncio
    async def test_resolve_existing_entity(self, rich_manager):
        name = await rich_manager.resolve_entity_name(
            "ent_konner",
            GROUP,
        )
        assert name == "Konner"

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
