"""Tests for Knowledge Management API endpoints."""

from __future__ import annotations

import json
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from engram.api.knowledge import (
    AdjudicateBody,
    ChatMessage,
    RememberBody,
    _analyze_chat_memory_need,
    _apply_chat_recall_feedback,
    _build_chat_memory_guidance,
    _execute_tool,
    _hydrate_chat_context,
    _record_chat_assistant_turn,
)
from engram.api.knowledge import (
    adjudicate as adjudicate_handler,
)
from engram.api.knowledge import (
    remember as remember_handler,
)
from engram.config import ActivationConfig, EngramConfig
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.epistemic import (
    AnswerContract,
    EpistemicBundle,
    EvidencePlan,
    QuestionFrame,
    ReconciliationResult,
)
from engram.models.relationship import Relationship
from engram.models.tenant import TenantContext
from engram.retrieval.context import ConversationContext


@pytest_asyncio.fixture
async def knowledge_client(tmp_path):
    """Create an httpx.AsyncClient wired to the FastAPI app with test data."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "knowledge_test.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    graph_store = _app_state["graph_store"]
    activation_store = _app_state["activation_store"]
    search_index = _app_state["search_index"]
    now = time.time()

    # Populate test entities
    e1 = Entity(
        id="ent_alice",
        name="Alice",
        entity_type="Person",
        summary="Engineer at Acme",
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
        summary="Memory layer for AI",
        group_id="default",
    )
    for e in [e1, e2, e3]:
        await graph_store.create_entity(e)
        await search_index.index_entity(e)

    # Relationships
    r1 = Relationship(
        id="rel_works_on",
        source_id="ent_alice",
        target_id="ent_engram",
        predicate="WORKS_ON",
        weight=1.0,
        group_id="default",
    )
    await graph_store.create_relationship(r1)

    # Record activation
    await activation_store.record_access("ent_alice", now)
    await activation_store.record_access("ent_alice", now - 10)

    # Create a completed episode
    ep = Episode(
        id="ep_test_0",
        content="Alice is working on the Engram project.",
        source="dashboard",
        status=EpisodeStatus.COMPLETED,
        group_id="default",
        created_at=datetime(2025, 6, 1, 12, 0, 0),
    )
    await graph_store.create_episode(ep)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await _shutdown()
    _app_state.clear()


@pytest_asyncio.fixture
async def empty_knowledge_client(tmp_path):
    """Client with an empty graph."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "empty_knowledge.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await _shutdown()
    _app_state.clear()


# ─── Observe ─────────────────────────────────────────────────────


class TestObserve:
    @pytest.mark.asyncio
    async def test_observe_stores_episode(self, knowledge_client):
        """POST /observe creates a QUEUED episode."""
        resp = await knowledge_client.post(
            "/api/knowledge/observe",
            json={"content": "Test observation content"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "observed"
        assert data["episodeId"].startswith("ep_")

    @pytest.mark.asyncio
    async def test_observe_custom_source(self, knowledge_client):
        """POST /observe with custom source."""
        resp = await knowledge_client.post(
            "/api/knowledge/observe",
            json={"content": "From slack", "source": "slack"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "observed"


# ─── Remember ────────────────────────────────────────────────────


class TestRemember:
    @pytest.mark.asyncio
    async def test_remember_ingests_episode(self, knowledge_client):
        """POST /remember ingests with extraction."""
        resp = await knowledge_client.post(
            "/api/knowledge/remember",
            json={"content": "Bob likes Python programming"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "remembered"
        assert data["episodeId"].startswith("ep_")

    @pytest.mark.asyncio
    async def test_remember_forwards_client_proposals(self, monkeypatch):
        manager = MagicMock()
        manager.ingest_episode = AsyncMock(return_value="ep_123")
        monkeypatch.setattr("engram.api.knowledge.get_manager", lambda: manager)

        request = SimpleNamespace(
            state=SimpleNamespace(
                tenant=TenantContext(group_id="default", auth_method="test"),
            ),
        )
        body = RememberBody(
            content="Alice works at Google",
            proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
            proposed_relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            model_tier="opus",
        )

        response = await remember_handler(request, body)

        assert response.status_code == 200
        manager.ingest_episode.assert_awaited_once_with(
            content="Alice works at Google",
            group_id="default",
            source="dashboard",
            conversation_date=None,
            proposed_entities=[{"name": "Alice", "entity_type": "Person"}],
            proposed_relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            model_tier="opus",
        )

    @pytest.mark.asyncio
    async def test_remember_returns_adjudication_requests(self, monkeypatch):
        manager = MagicMock()
        manager.ingest_episode = AsyncMock(return_value="ep_123")
        manager.get_episode_adjudications = AsyncMock(
            return_value=[
                {
                    "request_id": "adj_123",
                    "ambiguity_tags": ["negation_scope"],
                    "selected_text": "Alice works at Google, but maybe not anymore.",
                    "candidate_evidence": [
                        {
                            "evidence_id": "evi_1",
                            "fact_class": "relationship",
                            "payload": {
                                "subject": "Alice",
                                "predicate": "WORKS_AT",
                                "object": "Google",
                            },
                        },
                    ],
                    "instructions": "Resolve only if highly confident.",
                },
            ],
        )
        manager._cfg = ActivationConfig(edge_adjudication_client_enabled=True)
        monkeypatch.setattr("engram.api.knowledge.get_manager", lambda: manager)

        request = SimpleNamespace(
            state=SimpleNamespace(
                tenant=TenantContext(group_id="default", auth_method="test"),
            ),
        )
        body = RememberBody(content="Alice works at Google, but maybe not anymore.")

        response = await remember_handler(request, body)

        assert response.status_code == 200
        data = json.loads(response.body)
        assert len(data["adjudicationRequests"]) == 1
        assert data["adjudicationRequests"][0]["ambiguityTags"] == ["negation_scope"]

    @pytest.mark.asyncio
    async def test_adjudicate_endpoint_materializes_resolution(self, monkeypatch):
        manager = MagicMock()
        manager.submit_adjudication_resolution = AsyncMock(
            return_value=SimpleNamespace(
                status="materialized",
                request_id="adj_123",
                committed_ids={"evi_new": "rel_1"},
                superseded_evidence_ids=["evi_old"],
                replacement_evidence_ids=["evi_new"],
            ),
        )
        monkeypatch.setattr("engram.api.knowledge.get_manager", lambda: manager)

        request = SimpleNamespace(
            state=SimpleNamespace(
                tenant=TenantContext(group_id="default", auth_method="test"),
            ),
        )
        body = AdjudicateBody(
            request_id="adj_123",
            entities=[{"name": "Alice", "entity_type": "Person"}],
            relationships=[
                {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
            ],
            rationale="The sentence is retracting the employment fact.",
        )

        response = await adjudicate_handler(request, body)

        assert response.status_code == 200
        data = json.loads(response.body)
        assert data["status"] == "materialized"
        assert data["requestId"] == "adj_123"
        assert data["committedIds"] == {"evi_new": "rel_1"}


class TestEpistemicEndpoints:
    @pytest.mark.asyncio
    async def test_route_endpoint_classifies_reconcile(self, knowledge_client):
        resp = await knowledge_client.post(
            "/api/knowledge/route",
            json={
                "question": "what did we decide about launching Engram publicly?",
                "project_path": "/tmp/engram",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["questionFrame"]["mode"] == "reconcile"
        assert data["answerContract"]["operator"] == "reconcile"
        assert data["evidencePlan"]["requiredNextSources"] == ["artifacts"]
        assert "facts" in data["evidencePlan"]["discouragedSources"]

    @pytest.mark.asyncio
    async def test_route_endpoint_classifies_inspect(self, knowledge_client):
        resp = await knowledge_client.post(
            "/api/knowledge/route",
            json={"question": "how do we install the OpenClaw skill?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["questionFrame"]["mode"] == "inspect"

    @pytest.mark.asyncio
    async def test_route_endpoint_exposes_compare_scopes(self, knowledge_client):
        resp = await knowledge_client.post(
            "/api/knowledge/route",
            json={"question": "is full mode rework by default?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answerContract"]["operator"] == "compare"
        assert "raw_default" in data["answerContract"]["relevantScopes"]
        assert "install_default" in data["answerContract"]["relevantScopes"]
        assert "runtime_current" in data["answerContract"]["relevantScopes"]
        assert data["evidencePlan"]["requiredNextSources"] == ["artifacts", "runtime"]

    @pytest.mark.asyncio
    async def test_facts_endpoint_hides_epistemic_edges_by_default(self, knowledge_client):
        graph_store = _app_state["graph_store"]
        search_index = _app_state["search_index"]

        decision = Entity(
            id="dec_launch",
            name="Engram:public_launch_path:OpenClaw",
            entity_type="Decision",
            summary="Decision to launch through OpenClaw",
            group_id="default",
        )
        artifact = Entity(
            id="art_readme",
            name="README.md",
            entity_type="Artifact",
            summary="README artifact",
            group_id="default",
        )
        await graph_store.create_entity(decision)
        await graph_store.create_entity(artifact)
        await search_index.index_entity(decision)
        await search_index.index_entity(artifact)
        await graph_store.create_relationship(
            Relationship(
                id="rel_decided_in",
                source_id=decision.id,
                target_id=artifact.id,
                predicate="DECIDED_IN",
                weight=1.0,
                group_id="default",
            )
        )

        hidden = await knowledge_client.get(
            "/api/knowledge/facts",
            params={"subject": decision.name},
        )
        assert hidden.status_code == 200
        assert hidden.json()["items"] == []

        visible = await knowledge_client.get(
            "/api/knowledge/facts",
            params={"subject": decision.name, "include_epistemic": "true"},
        )
        assert visible.status_code == 200
        assert any(item["predicate"] == "DECIDED_IN" for item in visible.json()["items"])

    @pytest.mark.asyncio
    async def test_artifact_search_endpoint_returns_bootstrapped_hits(
        self,
        knowledge_client,
        tmp_path,
    ):
        (tmp_path / "README.md").write_text("# Engram\nLaunch path: OpenClaw\n")
        bootstrap = await knowledge_client.post(
            "/api/knowledge/bootstrap",
            json={"project_path": str(tmp_path)},
        )
        assert bootstrap.status_code == 200

        resp = await knowledge_client.get(
            "/api/knowledge/artifacts/search",
            params={"q": "OpenClaw", "project_path": str(tmp_path)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"]
        assert data["items"][0]["artifactClass"] in {"readme", "design_doc", "skill", "config"}

    @pytest.mark.asyncio
    async def test_runtime_endpoint_returns_epistemic_metrics(self, knowledge_client, tmp_path):
        (tmp_path / "README.md").write_text("# Engram\nCurrent default is rework.\n")
        await knowledge_client.post(
            "/api/knowledge/bootstrap",
            json={"project_path": str(tmp_path)},
        )

        resp = await knowledge_client.get(
            "/api/knowledge/runtime",
            params={"project_path": str(tmp_path)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "epistemicMetrics" in data["stats"]
        assert data["artifactBootstrap"]["artifactCount"] >= 1


# ─── Recall ──────────────────────────────────────────────────────


class TestRecall:
    @pytest.mark.asyncio
    async def test_recall_returns_results(self, knowledge_client):
        """GET /recall returns recall results."""
        resp = await knowledge_client.get("/api/knowledge/recall?q=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert data["query"] == "Alice"

    @pytest.mark.asyncio
    async def test_recall_with_limit(self, knowledge_client):
        """GET /recall respects limit parameter."""
        resp = await knowledge_client.get("/api/knowledge/recall?q=Alice&limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) <= 2

    @pytest.mark.asyncio
    async def test_recall_empty_results(self, empty_knowledge_client):
        """GET /recall on empty graph returns empty items."""
        resp = await empty_knowledge_client.get("/api/knowledge/recall?q=nothing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_recall_result_structure(self, knowledge_client):
        """Recall results have expected camelCase fields."""
        resp = await knowledge_client.get("/api/knowledge/recall?q=Alice")
        data = resp.json()
        if data["items"]:
            item = data["items"][0]
            assert "resultType" in item
            assert "score" in item
            assert "scoreBreakdown" in item
            breakdown = item["scoreBreakdown"]
            assert "semantic" in breakdown
            assert "activation" in breakdown
            assert "edgeProximity" in breakdown
            assert "explorationBonus" in breakdown

    @pytest.mark.asyncio
    async def test_recall_formats_entity_results(self, knowledge_client):
        """Entity recall results stay explicitly typed on the API surface."""
        resp = await knowledge_client.get("/api/knowledge/recall?q=Alice")
        assert resp.status_code == 200
        data = resp.json()

        entity_items = [item for item in data["items"] if item.get("resultType") == "entity"]
        if entity_items:
            item = entity_items[0]
            assert item["entity"]["id"]
            assert item["entity"]["name"]

    @pytest.mark.asyncio
    async def test_recall_includes_packets(self, knowledge_client):
        resp = await knowledge_client.get("/api/knowledge/recall?q=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert "packets" in data
        assert isinstance(data["packets"], list)

    @pytest.mark.asyncio
    async def test_recall_formats_cue_episode_results(self, knowledge_client):
        manager = MagicMock()
        manager._cfg = type("Cfg", (), {"recall_packets_enabled": False})()
        manager.recall = AsyncMock(
            return_value=[
                {
                    "result_type": "cue_episode",
                    "cue": {
                        "episode_id": "ep_cue_1",
                        "cue_text": "Phoenix redesign and recall discussion",
                        "supporting_spans": ["We need a latent memory layer."],
                        "projection_state": "cue_only",
                        "route_reason": "entity_dense",
                        "hit_count": 2,
                        "used_count": 1,
                        "policy_score": 0.78,
                        "last_feedback_at": "2026-03-05T12:30:00",
                    },
                    "episode": {
                        "id": "ep_cue_1",
                        "source": "dashboard",
                        "created_at": "2026-03-05T12:00:00",
                    },
                    "score": 0.72,
                    "score_breakdown": {
                        "semantic": 0.72,
                        "activation": 0.0,
                        "edge_proximity": 0.0,
                        "exploration_bonus": 0.0,
                    },
                }
            ]
        )

        with patch("engram.api.knowledge.get_manager", return_value=manager):
            resp = await knowledge_client.get("/api/knowledge/recall?q=Phoenix")

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"][0]["resultType"] == "cue_episode"
        assert data["items"][0]["cue"]["cueText"] == "Phoenix redesign and recall discussion"
        assert data["items"][0]["cue"]["usedCount"] == 1
        assert data["items"][0]["cue"]["policyScore"] == pytest.approx(0.78)
        assert data["items"][0]["cue"]["lastFeedbackAt"] == "2026-03-05T12:30:00"


class TestChatRecallHelpers:
    @pytest.mark.asyncio
    async def test_execute_tool_recall_formats_cue_episode(self):
        manager = MagicMock()
        manager._cfg = type("Cfg", (), {"recall_packets_enabled": False})()
        manager.recall = AsyncMock(
            return_value=[
                {
                    "result_type": "cue_episode",
                    "cue": {
                        "episode_id": "ep_cue_1",
                        "cue_text": "Recall redesign note",
                        "supporting_spans": ["Need a latent memory substrate."],
                        "projection_state": "scheduled",
                    },
                    "episode": {"id": "ep_cue_1", "source": "dashboard"},
                    "score": 0.77,
                }
            ]
        )

        raw = await _execute_tool(manager, "default", "recall", {"query": "recall", "limit": 3})
        payload = json.loads(raw)
        assert payload["results"][0]["type"] == "cue_episode"
        assert payload["results"][0]["cueText"] == "Recall redesign note"


# ─── Facts ───────────────────────────────────────────────────────


class TestFacts:
    @pytest.mark.asyncio
    async def test_search_facts_by_subject(self, knowledge_client):
        """GET /facts with subject filter."""
        resp = await knowledge_client.get("/api/knowledge/facts?subject=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        if data["items"]:
            item = data["items"][0]
            assert "subject" in item
            assert "predicate" in item
            assert "object" in item


@pytest.mark.asyncio
class TestChatMemoryNeedHelpers:
    async def test_analyze_chat_memory_need_uses_history(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig()
        manager._conv_context = None
        manager._graph = AsyncMock()
        manager._activation = AsyncMock()

        need = await _analyze_chat_memory_need(
            "Did we decide on that yet?",
            manager,
            history=[
                type("Msg", (), {"role": "user", "content": "We were debating Redis yesterday."})(),
            ],
            session_entity_names=["Redis"],
        )
        assert need.need_type == "open_loop"
        assert need.should_recall is True
        assert need.query_hint == "Redis"

    async def test_build_chat_memory_guidance_for_none(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig()
        manager._conv_context = None
        manager._graph = AsyncMock()
        manager._activation = AsyncMock()
        guidance = _build_chat_memory_guidance(
            await _analyze_chat_memory_need("thanks", manager, history=None),
        )
        assert "does not look required" in guidance

    async def test_build_chat_memory_guidance_for_recall(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig()
        manager._conv_context = None
        manager._graph = AsyncMock()
        manager._activation = AsyncMock()
        guidance = _build_chat_memory_guidance(
            await _analyze_chat_memory_need(
                "How's the auth migration going?",
                manager,
                history=None,
                session_entity_names=["Auth Migration"],
            ),
        )
        assert "Memory is likely relevant" in guidance


@pytest.mark.asyncio
class TestChatContextHelpers:
    async def test_hydrate_chat_context_records_live_history_and_message(self):
        manager = MagicMock()
        manager._conv_context = ConversationContext()

        async def embed(text):
            return [1.0, 0.0]

        provider = MagicMock()
        provider.embed_query = embed
        manager._search = MagicMock()
        manager._search._provider = provider

        history = [
            ChatMessage(role="user", content="We were discussing Redis."),
            ChatMessage(role="assistant", content="I mentioned the cache tradeoffs."),
        ]

        await _hydrate_chat_context(manager, history, "Did we decide on Redis?")

        assert manager._conv_context._turn_count == 3
        assert manager._conv_context.get_recent_turns(5) == [
            "We were discussing Redis.",
            "I mentioned the cache tradeoffs.",
            "Did we decide on Redis?",
        ]
        entries = manager._conv_context.get_recent_turn_entries(5, live_only=False)
        assert [entry.source for entry in entries] == [
            "chat_user",
            "chat_assistant",
            "chat_user",
        ]

    async def test_record_chat_assistant_turn_updates_context(self):
        manager = MagicMock()
        manager._conv_context = ConversationContext()

        async def embed(text):
            return [1.0, 0.0]

        provider = MagicMock()
        provider.embed_query = embed
        manager._search = MagicMock()
        manager._search._provider = provider

        await _record_chat_assistant_turn(manager, "Here is the follow-up.")

        assert manager._conv_context._turn_count == 1
        assert manager._conv_context.get_recent_turns() == ["Here is the follow-up."]
        entries = manager._conv_context.get_recent_turn_entries(5, live_only=False)
        assert entries[0].source == "chat_assistant"

    @pytest.mark.asyncio
    async def test_search_facts_empty(self, empty_knowledge_client):
        """GET /facts on empty graph returns empty items."""
        resp = await empty_knowledge_client.get("/api/knowledge/facts?q=nothing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []


@pytest.mark.asyncio
class TestChatRecallFeedbackHelpers:
    async def test_execute_tool_uses_selected_semantics_when_usage_feedback_enabled(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig(
            recall_usage_feedback_enabled=True,
            recall_packets_enabled=False,
        )
        manager.recall = AsyncMock(
            return_value=[
                {
                    "entity": {
                        "id": "ent_react",
                        "name": "React",
                        "type": "Technology",
                        "summary": "UI library",
                    },
                    "score": 0.91,
                    "score_breakdown": {"activation": 0.1},
                    "relationships": [],
                }
            ]
        )

        payload = await _execute_tool(
            manager,
            "default",
            "recall",
            {"query": "React", "limit": 5},
        )

        call_kwargs = manager.recall.await_args.kwargs
        assert call_kwargs["record_access"] is False
        assert call_kwargs["interaction_type"] == "selected"
        assert call_kwargs["interaction_source"] == "chat_tool_select"

        parsed = json.loads(payload)
        assert parsed["results"][0]["name"] == "React"

    async def test_apply_chat_recall_feedback_marks_used_and_dismissed(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig(recall_usage_feedback_enabled=True)
        manager.apply_memory_interaction = AsyncMock()
        recall_results = [
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Memory system",
                },
                "score": 0.95,
            },
            {
                "entity": {
                    "id": "ent_redis",
                    "name": "Redis",
                    "type": "Technology",
                    "summary": "Cache",
                },
                "score": 0.71,
            },
        ]

        await _apply_chat_recall_feedback(
            manager,
            group_id="default",
            query="What should we use?",
            response_text="Engram should stay the focus for this iteration.",
            recall_results=recall_results,
        )

        assert manager.apply_memory_interaction.await_count == 2

        used_call = manager.apply_memory_interaction.await_args_list[0]
        assert used_call.args[0] == ["ent_engram"]
        assert used_call.kwargs["interaction_type"] == "used"
        assert used_call.kwargs["source"] == "chat_response"

        dismissed_call = manager.apply_memory_interaction.await_args_list[1]
        assert dismissed_call.args[0] == ["ent_redis"]
        assert dismissed_call.kwargs["interaction_type"] == "dismissed"

    async def test_apply_chat_recall_feedback_marks_used_and_dismissed_cues(self):
        manager = MagicMock()
        manager._cfg = ActivationConfig(recall_usage_feedback_enabled=True)
        manager.apply_memory_interaction = AsyncMock()
        recall_results = [
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_react",
                    "cue_text": "spans: React dashboard migration remains in scope",
                    "supporting_spans": ["React dashboard migration remains in scope"],
                },
                "episode": {"id": "ep_react", "source": "chat"},
                "score": 0.88,
            },
            {
                "result_type": "cue_episode",
                "cue": {
                    "episode_id": "ep_redis",
                    "cue_text": "spans: Redis cache rollout was postponed",
                    "supporting_spans": ["Redis cache rollout was postponed"],
                },
                "episode": {"id": "ep_redis", "source": "chat"},
                "score": 0.61,
            },
        ]

        await _apply_chat_recall_feedback(
            manager,
            group_id="default",
            query="What should stay in scope?",
            response_text="Keep the React dashboard migration in scope.",
            recall_results=recall_results,
        )

        assert manager.apply_memory_interaction.await_count == 2

        used_call = manager.apply_memory_interaction.await_args_list[0]
        assert used_call.args[0] == ["cue:ep_react"]
        assert used_call.kwargs["interaction_type"] == "used"
        assert used_call.kwargs["source"] == "chat_response"

        dismissed_call = manager.apply_memory_interaction.await_args_list[1]
        assert dismissed_call.args[0] == ["cue:ep_redis"]
        assert dismissed_call.kwargs["interaction_type"] == "dismissed"


# ─── Context ─────────────────────────────────────────────────────


class TestContext:
    @pytest.mark.asyncio
    async def test_get_context_returns_markdown(self, knowledge_client):
        """GET /context returns context summary."""
        resp = await knowledge_client.get("/api/knowledge/context")
        assert resp.status_code == 200
        data = resp.json()
        assert "context" in data
        assert "entityCount" in data
        assert "factCount" in data
        assert "tokenEstimate" in data
        assert isinstance(data["context"], str)

    @pytest.mark.asyncio
    async def test_get_context_with_topic(self, knowledge_client):
        """GET /context with topic_hint biases results."""
        resp = await knowledge_client.get("/api/knowledge/context?topic_hint=Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["context"], str)

    @pytest.mark.asyncio
    async def test_get_context_empty_graph(self, empty_knowledge_client):
        """GET /context on empty graph returns empty context."""
        resp = await empty_knowledge_client.get("/api/knowledge/context")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entityCount"] == 0


# ─── Forget ──────────────────────────────────────────────────────


class TestForget:
    @pytest.mark.asyncio
    async def test_forget_entity(self, knowledge_client):
        """POST /forget with entity_name soft-deletes entity."""
        resp = await knowledge_client.post(
            "/api/knowledge/forget",
            json={"entity_name": "Bob"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "forgotten"
        assert data["target_type"] == "entity"

    @pytest.mark.asyncio
    async def test_forget_entity_not_found(self, knowledge_client):
        """POST /forget with unknown entity returns 404."""
        resp = await knowledge_client.post(
            "/api/knowledge/forget",
            json={"entity_name": "NonExistent"},
        )
        assert resp.status_code == 404
        data = resp.json()
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_forget_fact(self, knowledge_client):
        """POST /forget with fact invalidates the relationship."""
        resp = await knowledge_client.post(
            "/api/knowledge/forget",
            json={
                "fact": {
                    "subject": "Alice",
                    "predicate": "WORKS_ON",
                    "object": "Engram",
                },
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "forgotten"
        assert data["target_type"] == "fact"

    @pytest.mark.asyncio
    async def test_forget_no_target_returns_400(self, knowledge_client):
        """POST /forget without entity_name or fact returns 400."""
        resp = await knowledge_client.post(
            "/api/knowledge/forget",
            json={},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_forget_with_reason(self, knowledge_client):
        """POST /forget accepts optional reason."""
        resp = await knowledge_client.post(
            "/api/knowledge/forget",
            json={"entity_name": "Bob", "reason": "outdated info"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "forgotten"


# ─── Chat ────────────────────────────────────────────────────────


def _make_create_response(text: str, stop_reason: str = "end_turn"):
    """Create a mock response from messages.create() with a text block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = stop_reason
    response.content = [text_block]
    return response


class TestChat:
    @staticmethod
    def _parse_sse_events(text: str) -> list:
        """Parse SSE events from response text, returning decoded JSON objects."""
        events = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return events

    @pytest.mark.asyncio
    async def test_chat_streams_sse(self, knowledge_client):
        """POST /chat returns AI SDK v6 UIMessageStream protocol."""
        mock_response = _make_create_response("Hello world")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={"message": "Tell me about Alice"},
            )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = self._parse_sse_events(resp.text)

        # Should start with start + start-step
        assert events[0]["type"] == "start"
        assert events[1]["type"] == "start-step"

        # Should have text-start, text-delta(s), text-end
        text_starts = [e for e in events if e["type"] == "text-start"]
        assert len(text_starts) == 1
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        assert len(text_deltas) >= 1
        # Reconstruct full text from deltas
        full_text = "".join(e["delta"] for e in text_deltas)
        assert full_text == "Hello world"

        # Should end with finish-step + finish
        assert events[-2]["type"] == "finish-step"
        assert events[-1]["type"] == "finish"
        assert events[-1]["finishReason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_history(self, knowledge_client):
        """POST /chat accepts conversation history."""
        mock_response = _make_create_response("OK")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={
                    "message": "Follow up question",
                    "history": [
                        {"role": "user", "content": "Who is Alice?"},
                        {"role": "assistant", "content": "Alice is an engineer."},
                    ],
                },
            )

        assert resp.status_code == 200

        # Verify messages passed to Claude include history
        call_kwargs = mock_client.messages.create.call_args[1]
        assert len(call_kwargs["messages"]) == 3  # 2 history + 1 new

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, knowledge_client):
        """POST /chat handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(side_effect=Exception("API key invalid"))

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={"message": "test"},
            )

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)

        # Should have error event + finish with error reason
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "API key invalid" in error_events[0]["errorText"]

        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finishReason"] == "error"

    @pytest.mark.asyncio
    async def test_chat_tool_use_loop(self, knowledge_client):
        """POST /chat executes agentic tool-use loop before final response."""
        # First call returns tool_use, second returns final text
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "search_facts"
        tool_block.input = {"subject": "Alice", "predicate": "WORKS_AT"}
        tool_block.id = "tu_1"

        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_response.content = [tool_block]

        final_response = _make_create_response("Alice works at Acme Corp.")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[tool_response, final_response],
        )

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={"message": "Where does Alice work?"},
            )

        assert resp.status_code == 200
        events = self._parse_sse_events(resp.text)

        # Should have called create twice (tool turn + final)
        assert mock_client.messages.create.call_count == 2

        # Second call should include tool results
        second_call = mock_client.messages.create.call_args_list[1][1]
        # Messages should have: user + assistant (tool_use) + user (tool_result)
        assert len(second_call["messages"]) == 3

        # Final text should be streamed
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        full_text = "".join(e["delta"] for e in text_deltas)
        assert full_text == "Alice works at Acme Corp."

    @pytest.mark.asyncio
    async def test_chat_retries_generic_response_once(self, knowledge_client):
        manager = _app_state["graph_manager"]
        manager._cfg = ActivationConfig(
            recall_need_analyzer_enabled=True,
            recall_need_structural_enabled=True,
            recall_need_post_response_safety_net_enabled=True,
        )

        generic_response = _make_create_response("That makes sense. Let me know if you want help.")
        grounded_response = _make_create_response("Alice is still the engineer working on Engram.")

        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            side_effect=[generic_response, grounded_response],
        )

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={"message": "How's the Engram project going?"},
            )

        assert resp.status_code == 200
        assert mock_client.messages.create.call_count == 2
        events = self._parse_sse_events(resp.text)
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        full_text = "".join(e["delta"] for e in text_deltas)
        assert full_text == "Alice is still the engineer working on Engram."

    @pytest.mark.asyncio
    async def test_chat_includes_answer_contract_guidance(self, knowledge_client):
        manager = _app_state["graph_manager"]
        manager._cfg.epistemic_routing_enabled = True
        manager._cfg.answer_contract_enabled = True
        manager._cfg.claim_state_modeling_enabled = True

        bundle = EpistemicBundle(
            question_frame=QuestionFrame(
                mode="inspect",
                domain="runtime",
                timeframe="current",
                expected_authorities=["current", "canonical"],
                expected_sources=["artifacts", "runtime"],
                requires_workspace=False,
                confidence=0.9,
            ),
            evidence_plan=EvidencePlan(
                use_artifacts=True,
                use_runtime=True,
                artifact_budget=5,
                runtime_budget=1,
            ),
            reconciliation=ReconciliationResult(
                status="confirmed",
                answer_hints=["Prefer scoped comparison over a flattened default answer."],
            ),
            answer_contract=AnswerContract(
                operator="compare",
                requested_truth_kind="mixed",
                relevant_scopes=["raw_default", "install_default", "runtime_current"],
                preferred_authorities=["canonical", "current"],
                preserve_temporal_distinction=True,
                include_provenance=True,
                allow_recommendation=False,
                confidence=0.9,
                guidance=[
                    "Answer by contrasting scopes instead of flattening to one default.",
                    "Distinguish shipped install defaults from raw code defaults.",
                    "Include the effective runtime state when it materially affects the answer.",
                ],
            ),
            claim_state_summary={"dominantState": "implemented"},
        )
        manager.gather_epistemic_evidence = AsyncMock(return_value=bundle)

        mock_response = _make_create_response("Raw default is off, install default is rework.")
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("engram.api.knowledge.anthropic.AsyncAnthropic", return_value=mock_client):
            resp = await knowledge_client.post(
                "/api/knowledge/chat",
                json={"message": "is full mode rework by default?"},
            )

        assert resp.status_code == 200
        system_prompt = mock_client.messages.create.call_args.kwargs["system"]
        prompt_text = "\n".join(part["text"] for part in system_prompt)
        assert "Answer-contract guidance for this turn:" in prompt_text
        assert "Distinguish shipped install defaults from raw code defaults." in prompt_text
