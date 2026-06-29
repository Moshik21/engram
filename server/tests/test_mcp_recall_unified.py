"""MCP recall-unified retrieval and deprecated search compat shims."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import ActivationConfig
from engram.retrieval.lookup import (
    MCP_SEARCH_ENTITIES_DEPRECATION,
    build_mcp_entity_search_tool_surface,
)


@pytest.mark.asyncio
async def test_entity_search_compat_primes_recall_path() -> None:
    manager = MagicMock()
    manager.search_entities = AsyncMock(return_value=[])
    recall_middleware = AsyncMock()
    cfg = ActivationConfig()

    with patch(
        "engram.retrieval.recall_surface.build_mcp_recall_surface",
        new_callable=AsyncMock,
        return_value={"results": [], "total": 0},
    ) as recall_surface:
        payload = await build_mcp_entity_search_tool_surface(
            manager,
            group_id="default",
            name="Helix",
            limit=3,
            recall_middleware=recall_middleware,
            cfg=cfg,
        )

    recall_surface.assert_awaited_once()
    assert payload["preferRecall"] is True
    assert payload["deprecationNotice"] == MCP_SEARCH_ENTITIES_DEPRECATION
    assert payload["recallCompanion"]["query"] == "Helix"
    recall_middleware.assert_awaited_once()
