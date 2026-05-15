from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.context import ConversationContext, ConversationRuntimeService


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
