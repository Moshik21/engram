from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.chat_tools import execute_chat_tool


@pytest.mark.asyncio
async def test_execute_chat_tool_search_entities_formats_llm_payload() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(
        return_value=[
            {
                "id": "ent_engram",
                "name": "Engram",
                "type": "Project",
                "summary": "AI memory runtime.",
            }
        ]
    )

    payload = await execute_chat_tool(
        manager,
        group_id="tenant_brain",
        tool_name="search_entities",
        tool_input={"name": "Engram", "limit": 50},
    )

    assert manager.search_entities.await_args.kwargs["group_id"] == "tenant_brain"
    assert manager.search_entities.await_args.kwargs["limit"] == 20
    assert payload == {
        "entities": [
            {
                "id": "ent_engram",
                "name": "Engram",
                "entityType": "Project",
                "summary": "AI memory runtime.",
            }
        ],
        "total": 1,
    }


@pytest.mark.asyncio
async def test_execute_chat_tool_search_facts_deduplicates_results() -> None:
    manager = MagicMock()
    manager.search_facts = AsyncMock(
        return_value=[
            {
                "subject": "Engram",
                "predicate": "USES",
                "object": "PyO3 native Helix",
                "confidence": 0.91,
            },
            {
                "subject": "Engram",
                "predicate": "USES",
                "object": "PyO3 native Helix",
                "confidence": 0.84,
            },
            {
                "subject": "Engram",
                "predicate": "HAS_LOOP",
                "object": "Capture -> Cue -> Project -> Recall -> Consolidate",
                "confidence": 0.88,
            },
        ]
    )

    payload = await execute_chat_tool(
        manager,
        group_id="tenant_brain",
        tool_name="search_facts",
        tool_input={"query": "Engram", "limit": 2},
    )

    assert manager.search_facts.await_args.kwargs["limit"] == 4
    assert payload == {
        "facts": [
            {
                "subject": "Engram",
                "predicate": "USES",
                "object": "PyO3 native Helix",
                "confidence": 0.91,
            },
            {
                "subject": "Engram",
                "predicate": "HAS_LOOP",
                "object": "Capture -> Cue -> Project -> Recall -> Consolidate",
                "confidence": 0.88,
            },
        ],
        "total": 2,
    }


@pytest.mark.asyncio
async def test_execute_chat_tool_unknown_tool_returns_error() -> None:
    payload = await execute_chat_tool(
        MagicMock(),
        group_id="tenant_brain",
        tool_name="missing_tool",
        tool_input={},
    )

    assert payload == {"error": "Unknown tool: missing_tool"}
