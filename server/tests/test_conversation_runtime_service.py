from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.retrieval.context import (
    ConversationContext,
    ConversationRuntimeService,
    ingest_manager_conversation_turn,
    manager_conversation_context,
    manager_conversation_embed_fn,
    manager_conversation_recent_turns,
    manager_conversation_top_entity_names,
    manager_conversation_turn_count,
)


class _Provider:
    async def embed_query(self, text: str) -> list[float]:
        assert text
        return [1.0, 0.0]


class _SearchIndex:
    _provider = _Provider()


def test_conversation_runtime_service_surfaces_context_views() -> None:
    ctx = ConversationContext()
    ctx.add_turn("First turn", source="chat_user")
    ctx.add_turn("Second turn", source="chat_assistant")
    ctx.add_session_entity("ent_a", "Alpha", "concept", weight_increment=2.0)
    ctx.add_session_entity("ent_b", "Beta", "concept", weight_increment=1.0)

    service = ConversationRuntimeService(
        cfg=ActivationConfig(conv_multi_query_top_entities=1),
        conv_context=ctx,
        search_index=object(),
    )

    assert service.get_context() is ctx
    assert service.get_turn_count() == 2
    assert service.get_top_entity_names() == ["Alpha"]
    assert service.get_top_entity_names(2) == ["Alpha", "Beta"]
    assert service.get_recent_turns(1) == ["Second turn"]


@pytest.mark.asyncio
async def test_conversation_runtime_service_ingests_with_embedding_provider() -> None:
    ctx = ConversationContext()
    service = ConversationRuntimeService(
        cfg=ActivationConfig(),
        conv_context=ctx,
        search_index=_SearchIndex(),
    )

    await service.ingest_turn("Remember this context", source="chat_user")

    assert service.get_turn_count() == 1
    assert ctx.get_recent_turns() == ["Remember this context"]
    assert ctx.get_fingerprint() == [1.0, 0.0]


@pytest.mark.asyncio
async def test_conversation_runtime_service_noops_without_context() -> None:
    service = ConversationRuntimeService(
        cfg=ActivationConfig(),
        conv_context=None,
        search_index=object(),
    )

    await service.ingest_turn("Ignored", source="chat_user")

    assert service.get_context() is None
    assert service.get_turn_count() == 0
    assert service.get_top_entity_names() == []
    assert service.get_recent_turns() == []


def test_manager_conversation_facade_helpers_filter_expected_shapes() -> None:
    ctx = ConversationContext()

    def embed(text: str) -> list[float]:
        return [1.0]

    manager = SimpleNamespace(
        get_conversation_context=lambda: ctx,
        get_conversation_embed_fn=lambda: embed,
        get_conversation_turn_count=lambda: 3,
        get_conversation_top_entity_names=lambda limit=None: (
            ["Alpha", 123, "Beta"][:limit] if limit else ["Alpha", 123, "Beta"]
        ),
        get_conversation_recent_turns=lambda limit: ["one", None, "two"][:limit],
    )

    assert manager_conversation_context(manager) is ctx
    assert manager_conversation_embed_fn(manager) is embed
    assert manager_conversation_turn_count(manager) == 3
    assert manager_conversation_top_entity_names(manager) == ["Alpha", "Beta"]
    assert manager_conversation_top_entity_names(manager, 1) == ["Alpha"]
    assert manager_conversation_recent_turns(manager, 3) == ["one", "two"]


def test_manager_conversation_facade_helpers_return_defaults_for_missing_or_async() -> None:
    manager = SimpleNamespace(
        get_conversation_context=AsyncMock(return_value=ConversationContext()),
        get_conversation_embed_fn=AsyncMock(return_value=lambda text: [1.0]),
        get_conversation_turn_count=AsyncMock(return_value=3),
        get_conversation_top_entity_names=AsyncMock(return_value=["Alpha"]),
        get_conversation_recent_turns=AsyncMock(return_value=["one"]),
    )

    assert manager_conversation_context(SimpleNamespace()) is None
    assert manager_conversation_context(manager) is None
    assert manager_conversation_embed_fn(manager) is None
    assert manager_conversation_turn_count(manager) == 0
    assert manager_conversation_top_entity_names(manager) == []
    assert manager_conversation_recent_turns(manager, 3) == []


@pytest.mark.asyncio
async def test_ingest_manager_conversation_turn_supports_sync_and_async_facades() -> None:
    sync_manager = SimpleNamespace(ingest_conversation_turn=lambda text, *, source: None)
    await ingest_manager_conversation_turn(sync_manager, "hello", source="chat_user")

    async_manager = SimpleNamespace(ingest_conversation_turn=AsyncMock())
    await ingest_manager_conversation_turn(async_manager, "hello", source="chat_user")

    async_manager.ingest_conversation_turn.assert_awaited_once_with(
        "hello",
        source="chat_user",
    )


@pytest.mark.asyncio
async def test_ingest_manager_conversation_turn_noops_without_facade() -> None:
    await ingest_manager_conversation_turn(SimpleNamespace(), "hello", source="chat_user")
