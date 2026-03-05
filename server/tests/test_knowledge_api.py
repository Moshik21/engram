"""Tests for Knowledge Management API endpoints."""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from engram.config import EngramConfig
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship


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
    async def test_search_facts_empty(self, empty_knowledge_client):
        """GET /facts on empty graph returns empty items."""
        resp = await empty_knowledge_client.get("/api/knowledge/facts?q=nothing")
        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []


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
