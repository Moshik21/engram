from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.context_builder import (
    build_api_context_surface,
    build_mcp_context_surface,
)

CONTEXT_RESULT = {
    "context": "## Active Memory\nAlice works on Engram.",
    "entity_count": 2,
    "fact_count": 3,
    "token_estimate": 42,
    "format": "briefing",
}


@pytest.mark.asyncio
async def test_api_context_surface_maps_rest_keys() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_api_context_surface(
        manager,
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
    )

    assert payload == {
        "context": "## Active Memory\nAlice works on Engram.",
        "entityCount": 2,
        "factCount": 3,
        "tokenEstimate": 42,
        "format": "briefing",
    }
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=1200,
        topic_hint="Alice",
        project_path="/tmp/engram",
        format="briefing",
    )


@pytest.mark.asyncio
async def test_mcp_context_surface_preserves_raw_manager_shape() -> None:
    manager = MagicMock()
    manager.get_context = AsyncMock(return_value=CONTEXT_RESULT)

    payload = await build_mcp_context_surface(
        manager,
        group_id="native_brain",
        topic_hint="Alice",
    )

    assert payload == CONTEXT_RESULT
    manager.get_context.assert_awaited_once_with(
        group_id="native_brain",
        max_tokens=2000,
        topic_hint="Alice",
        project_path=None,
        format="structured",
    )
