from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.consolidation.finalization import ConsolidationFinalizationService


@pytest.mark.asyncio
async def test_finalization_service_noops_without_graph_manager():
    service = ConsolidationFinalizationService(graph_manager=None)

    result = await service.refresh_after_cycle("test")

    assert result.refreshed_pinned_contexts == 0
    assert result.event_payload() == {"refreshedPinnedContexts": 0}


@pytest.mark.asyncio
async def test_finalization_service_refreshes_after_consolidation_pinned_contexts():
    pinned = SimpleNamespace(
        id="intent_pinned",
        attributes={
            "trigger_type": "refresh_context",
            "refresh_trigger": "after_consolidation",
            "trigger_text": "Engram architecture",
        },
    )
    manual = SimpleNamespace(
        id="intent_manual",
        attributes={
            "trigger_type": "refresh_context",
            "refresh_trigger": "manual",
            "trigger_text": "Manual only",
        },
    )
    missing_topic = SimpleNamespace(
        id="intent_missing_topic",
        attributes={
            "trigger_type": "refresh_context",
            "refresh_trigger": "after_consolidation",
            "trigger_text": "",
        },
    )
    graph_manager = SimpleNamespace(
        list_intentions=AsyncMock(return_value=[pinned, manual, missing_topic]),
        get_context=AsyncMock(return_value={"context": "project facts", "fact_count": 3}),
        update_intention_meta=AsyncMock(),
    )
    service = ConsolidationFinalizationService(graph_manager=graph_manager)

    result = await service.refresh_after_cycle("test")

    assert result.refreshed_pinned_contexts == 1
    graph_manager.list_intentions.assert_awaited_once_with(
        group_id="test",
        enabled_only=True,
    )
    graph_manager.get_context.assert_awaited_once_with(
        group_id="test",
        topic_hint="Engram architecture",
        format="structured",
    )
    graph_manager.update_intention_meta.assert_awaited_once()
    kwargs = graph_manager.update_intention_meta.await_args.kwargs
    assert kwargs["intention_id"] == "intent_pinned"
    assert kwargs["group_id"] == "test"
    assert json.loads(kwargs["updates"]["pinned_result"]) == {
        "context": "project facts",
        "fact_count": 3,
    }
    assert kwargs["updates"]["last_refreshed"]


@pytest.mark.asyncio
async def test_finalization_service_continues_after_single_refresh_failure():
    first = SimpleNamespace(
        id="intent_first",
        attributes={
            "trigger_type": "refresh_context",
            "refresh_trigger": "after_consolidation",
            "trigger_text": "First",
        },
    )
    second = SimpleNamespace(
        id="intent_second",
        attributes={
            "trigger_type": "refresh_context",
            "refresh_trigger": "after_consolidation",
            "trigger_text": "Second",
        },
    )
    graph_manager = SimpleNamespace(
        list_intentions=AsyncMock(return_value=[first, second]),
        get_context=AsyncMock(side_effect=[RuntimeError("context failed"), "second result"]),
        update_intention_meta=AsyncMock(),
    )
    service = ConsolidationFinalizationService(graph_manager=graph_manager)

    result = await service.refresh_after_cycle("test")

    assert result.refreshed_pinned_contexts == 1
    assert graph_manager.get_context.await_count == 2
    graph_manager.update_intention_meta.assert_awaited_once()
    assert graph_manager.update_intention_meta.await_args.kwargs["intention_id"] == "intent_second"
