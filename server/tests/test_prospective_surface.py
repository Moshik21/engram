from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.prospective import (
    build_api_create_intention_surface,
    build_api_dismiss_intention_surface,
    build_intention_list_surface,
    build_mcp_create_intention_surface,
    build_mcp_dismiss_intention_surface,
)


@pytest.mark.asyncio
async def test_create_intention_surfaces_preserve_api_and_mcp_contracts() -> None:
    manager = MagicMock()
    manager.create_intention = AsyncMock(return_value="int_native")
    manager.effective_intention_threshold.return_value = 0.72

    api = await build_api_create_intention_surface(
        manager,
        group_id="native_brain",
        trigger_text="native Helix",
        action_text="Use PyO3 native first",
        trigger_type="activation",
        entity_names=["Helix"],
        threshold=0.72,
        priority="high",
        context="no Docker path",
        see_also=["doctor"],
        refresh_trigger="manual",
    )
    mcp = await build_mcp_create_intention_surface(
        manager,
        group_id="native_brain",
        trigger_text="native Helix",
        action_text="Use PyO3 native first",
        trigger_type="activation",
        entity_names=["Helix"],
        threshold=0.72,
        priority="high",
        context="no Docker path",
        see_also=["doctor"],
        refresh_trigger="manual",
    )

    assert manager.create_intention.await_count == 2
    assert api["intentionId"] == "int_native"
    assert api["triggerText"] == "native Helix"
    assert mcp["intention_id"] == "int_native"
    assert mcp["linked_entities"] == ["Helix"]
    assert mcp["threshold"] == 0.72


@pytest.mark.asyncio
async def test_list_and_dismiss_intention_surfaces() -> None:
    manager = MagicMock()
    manager.list_intention_views = AsyncMock(return_value=[{"id": "int_native"}])
    manager.dismiss_intention = AsyncMock()

    listed = await build_intention_list_surface(
        manager,
        group_id="native_brain",
        enabled_only=True,
        surface="api",
    )
    api_dismiss = await build_api_dismiss_intention_surface(
        manager,
        group_id="native_brain",
        intention_id="int_native",
        hard=False,
    )
    mcp_dismiss = await build_mcp_dismiss_intention_surface(
        manager,
        group_id="native_brain",
        intention_id="int_native",
        hard=True,
    )

    assert listed == {"intentions": [{"id": "int_native"}], "total": 1}
    assert api_dismiss == {
        "status": "dismissed",
        "intentionId": "int_native",
        "hard": False,
    }
    assert mcp_dismiss["intention_id"] == "int_native"
    assert mcp_dismiss["message"].endswith("deleted.")
