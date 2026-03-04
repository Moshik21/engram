"""Tests for Knowledge Management API endpoints."""

from __future__ import annotations

import json
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

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


def _make_stream_mock(text_chunks):
    """Create a mock async context manager that yields text chunks."""
    mock_stream_ctx = MagicMock()

    async def aenter(*_args):
        return mock_stream_ctx

    async def aexit(*_args):
        return False

    async def text_gen():
        for chunk in text_chunks:
            yield chunk

    mock_stream_ctx.__aenter__ = aenter
    mock_stream_ctx.__aexit__ = aexit
    mock_stream_ctx.text_stream = text_gen()
    return mock_stream_ctx


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
        mock_stream_ctx = _make_stream_mock(["Hello", " world"])

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx

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
        text_deltas = [e for e in events if e["type"] == "text-delta"]
        assert len(text_deltas) >= 2
        assert text_deltas[0]["delta"] == "Hello"
        assert text_deltas[1]["delta"] == " world"

        # Should end with finish-step + finish
        assert events[-2]["type"] == "finish-step"
        assert events[-1]["type"] == "finish"
        assert events[-1]["finishReason"] == "stop"

    @pytest.mark.asyncio
    async def test_chat_with_history(self, knowledge_client):
        """POST /chat accepts conversation history."""
        mock_stream_ctx = _make_stream_mock(["OK"])

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx

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
        call_kwargs = mock_client.messages.stream.call_args[1]
        assert len(call_kwargs["messages"]) == 3  # 2 history + 1 new

    @pytest.mark.asyncio
    async def test_chat_error_handling(self, knowledge_client):
        """POST /chat handles API errors gracefully."""
        mock_stream_ctx = MagicMock()

        async def aenter_error(*_args):
            raise Exception("API key invalid")

        async def aexit_noop(*_args):
            return False

        mock_stream_ctx.__aenter__ = aenter_error
        mock_stream_ctx.__aexit__ = aexit_noop

        mock_client = MagicMock()
        mock_client.messages.stream.return_value = mock_stream_ctx

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
