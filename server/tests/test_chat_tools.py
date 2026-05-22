from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.retrieval.chat_tools import (
    CHAT_TOOLS,
    execute_chat_tool,
    extract_message_text,
    retry_memory_grounded_response,
    run_chat_tool_use_loop,
)


def test_chat_tools_schema_exposes_recall_and_lookup_tools() -> None:
    assert [tool["name"] for tool in CHAT_TOOLS] == [
        "recall",
        "search_entities",
        "search_facts",
    ]
    assert CHAT_TOOLS[2]["input_schema"]["properties"]["include_epistemic"]["default"] is False


def test_extract_message_text_joins_text_blocks_defensively() -> None:
    assert (
        extract_message_text(
            [
                SimpleNamespace(text="Engram "),
                SimpleNamespace(text="brain"),
                SimpleNamespace(text=""),
                SimpleNamespace(other="ignored"),
            ]
        )
        == "Engram brain"
    )
    assert extract_message_text({"text": "not a list"}) == ""


@pytest.mark.asyncio
async def test_retry_memory_grounded_response_returns_retry_text() -> None:
    retry_response = SimpleNamespace(content=[SimpleNamespace(text="Specific remembered answer.")])
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(return_value=retry_response))
    )

    text = await retry_memory_grounded_response(
        client,
        system_prompt=[{"type": "text", "text": "Base prompt"}],
        loop_messages=[{"role": "user", "content": "Use memory"}],
        chat_need=SimpleNamespace(need_type="project_state"),
        prior_response="Got it.",
    )

    assert text == "Specific remembered answer."
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["messages"] == [{"role": "user", "content": "Use memory"}]
    assert kwargs["system"][-1]["text"].startswith("The previous draft stayed too generic")


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


@pytest.mark.asyncio
async def test_execute_chat_tool_recall_records_packet_budget_metrics() -> None:
    manager = MagicMock()
    manager.get_chat_tool_recall_policy.return_value = SimpleNamespace(
        record_access=True,
        interaction_type="used",
        interaction_source="chat_tool_use",
        packets_enabled=True,
        packet_limit=2,
    )
    manager.get_memory_need_config.return_value = SimpleNamespace(
        recall_budget_chat_ms=1000,
        auto_recall_token_budget=300,
        recall_budget_timeout_degrades=True,
        recall_need_graph_probe_enabled=False,
    )
    manager.recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Memory layer",
                },
                "score": 0.9,
            }
        ]
    )
    manager.record_memory_operation = MagicMock()
    packet = SimpleNamespace(
        packet_type="state_packet",
        title="State: Engram",
        summary="Memory layer",
        why_now="Relevant to chat.",
        confidence=0.9,
        evidence_lines=["Engram is a memory layer."],
        provenance=["entity:ent_engram"],
    )

    with (
        patch("engram.retrieval.chat_tools.analyze_memory_need", AsyncMock(return_value=None)),
        patch(
            "engram.retrieval.chat_tools.resolve_manager_recall_need_thresholds",
            AsyncMock(return_value=None),
        ),
        patch(
            "engram.retrieval.chat_tools.assemble_memory_packets",
            AsyncMock(return_value=[packet]),
        ),
    ):
        payload = await execute_chat_tool(
            manager,
            group_id="tenant_brain",
            tool_name="recall",
            tool_input={"query": "Engram", "limit": 5},
        )

    assert payload["budget"]["profile"] == "chat"
    assert payload["budget"]["degraded"] is False
    assert payload["packets"][0]["title"] == "State: Engram"
    sample = manager.record_memory_operation.call_args.args[1]
    assert sample.operation == "chat_recall_packets"
    assert sample.source == "chat_tool_use"
    assert sample.status == "ok"
    assert sample.packet_count == 1


@pytest.mark.asyncio
async def test_execute_chat_tool_recall_degrades_when_packet_budget_times_out() -> None:
    manager = MagicMock()
    manager.get_chat_tool_recall_policy.return_value = SimpleNamespace(
        record_access=True,
        interaction_type="used",
        interaction_source="chat_tool_use",
        packets_enabled=True,
        packet_limit=2,
    )
    manager.get_memory_need_config.return_value = SimpleNamespace(
        recall_budget_chat_ms=1,
        auto_recall_token_budget=300,
        recall_budget_timeout_degrades=True,
        recall_need_graph_probe_enabled=False,
    )
    manager.recall = AsyncMock(
        return_value=[
            {
                "entity": {
                    "id": "ent_engram",
                    "name": "Engram",
                    "type": "Project",
                    "summary": "Memory layer",
                },
                "score": 0.9,
            }
        ]
    )
    manager.record_memory_operation = MagicMock()

    async def slow_packet_assembly(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return []

    with (
        patch("engram.retrieval.chat_tools.analyze_memory_need", AsyncMock(return_value=None)),
        patch(
            "engram.retrieval.chat_tools.resolve_manager_recall_need_thresholds",
            AsyncMock(return_value=None),
        ),
        patch(
            "engram.retrieval.chat_tools.assemble_memory_packets",
            slow_packet_assembly,
        ),
    ):
        payload = await execute_chat_tool(
            manager,
            group_id="tenant_brain",
            tool_name="recall",
            tool_input={"query": "Engram", "limit": 5},
        )

    assert payload["results"][0]["name"] == "Engram"
    assert payload["packets"] == []
    assert payload["budget"]["degraded"] is True
    assert payload["budget"]["skipReason"] == "packet_timeout"
    sample = manager.record_memory_operation.call_args.args[1]
    assert sample.status == "degraded"
    assert sample.timeout is True
    assert sample.budget_miss is True


@pytest.mark.asyncio
async def test_run_chat_tool_use_loop_executes_tools_and_accumulates_events() -> None:
    manager = MagicMock()
    manager.search_facts = AsyncMock(
        return_value=[
            {
                "subject": "Engram",
                "predicate": "USES",
                "object": "PyO3 native Helix",
                "confidence": 0.91,
            }
        ]
    )
    tool_response = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                id="tool_1",
                name="search_facts",
                input={"query": "Engram", "limit": 1},
            )
        ],
    )
    final_response = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(text="Engram uses native Helix.")],
    )
    client = SimpleNamespace(
        messages=SimpleNamespace(create=AsyncMock(side_effect=[tool_response, final_response]))
    )

    result = await run_chat_tool_use_loop(
        client,
        manager=manager,
        group_id="tenant_brain",
        system_prompt=[],
        messages=[{"role": "user", "content": "What does Engram use?"}],
        tools=[{"name": "search_facts"}],
        max_tool_turns=3,
    )

    assert result.response is final_response
    assert result.facts == [
        {
            "subject": "Engram",
            "predicate": "USES",
            "object": "PyO3 native Helix",
            "confidence": 0.91,
        }
    ]
    assert len(result.loop_messages) == 3
    assert result.loop_messages[-1]["content"][0]["tool_use_id"] == "tool_1"
    assert client.messages.create.await_count == 2
