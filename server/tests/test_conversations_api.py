"""Tests for conversation API ownership checks."""

from __future__ import annotations

import httpx
import pytest
import pytest_asyncio

from engram.config import EngramConfig
from engram.main import _app_state, _shutdown, _startup, create_app


@pytest_asyncio.fixture
async def conversations_client(tmp_path):
    config = EngramConfig(
        mode="lite",
        default_group_id="group-a",
        auth={"default_group_id": "group-a"},
        sqlite={"path": str(tmp_path / "conversations_test.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, _app_state["conversation_store"]

    await _shutdown()
    _app_state.clear()


class TestConversationOwnership:
    @pytest.mark.asyncio
    async def test_get_messages_rejects_other_group(self, conversations_client):
        client, store = conversations_client
        owned_id = await store.create_conversation("group-a", title="Owned")
        foreign_id = await store.create_conversation("group-b", title="Foreign")
        await store.add_messages_bulk(
            owned_id,
            [{"role": "user", "content": "hello from group a"}],
            group_id="group-a",
        )
        await store.add_messages_bulk(
            foreign_id,
            [{"role": "user", "content": "hello from group b"}],
            group_id="group-b",
        )

        owned = await client.get(f"/api/conversations/{owned_id}/messages")
        assert owned.status_code == 200
        assert owned.json()["messages"][0]["content"] == "hello from group a"

        foreign = await client.get(f"/api/conversations/{foreign_id}/messages")
        assert foreign.status_code == 404

    @pytest.mark.asyncio
    async def test_append_messages_rejects_other_group(self, conversations_client):
        client, store = conversations_client
        foreign_id = await store.create_conversation("group-b", title="Foreign")

        resp = await client.post(
            f"/api/conversations/{foreign_id}/messages",
            json={"messages": [{"role": "assistant", "content": "should fail"}]},
        )

        assert resp.status_code == 404
        assert await store.get_messages(foreign_id, "group-b") == []

    @pytest.mark.asyncio
    async def test_chat_rejects_other_group_conversation_id(self, conversations_client):
        client, store = conversations_client
        foreign_id = await store.create_conversation("group-b", title="Foreign")

        resp = await client.post(
            "/api/knowledge/chat",
            json={
                "message": "What did we decide?",
                "conversation_id": foreign_id,
            },
        )

        assert resp.status_code == 404
        assert resp.json()["detail"] == "Conversation not found"
