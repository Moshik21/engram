from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.prospective import (
    api_intention_not_found_payload,
    api_intention_validation_error_payload,
    build_api_create_intention_response_surface,
    build_api_create_intention_surface,
    build_api_dismiss_intention_response_surface,
    build_api_dismiss_intention_surface,
    build_intention_list_surface,
    build_mcp_create_intention_response_surface,
    build_mcp_create_intention_surface,
    build_mcp_dismiss_intention_response_surface,
    build_mcp_dismiss_intention_surface,
    mcp_intention_error_payload,
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


def test_api_intention_error_payloads() -> None:
    assert api_intention_validation_error_payload(ValueError("bad trigger")) == {
        "detail": "bad trigger"
    }
    assert api_intention_not_found_payload() == {"detail": "Intention not found"}
    assert mcp_intention_error_payload(ValueError("bad trigger")) == {
        "status": "error",
        "message": "bad trigger",
    }


@pytest.mark.asyncio
async def test_api_create_intention_response_surface_maps_validation_to_400() -> None:
    manager = MagicMock()
    manager.create_intention = AsyncMock(side_effect=ValueError("bad trigger"))

    result = await build_api_create_intention_response_surface(
        manager,
        group_id="native_brain",
        trigger_text="",
        action_text="Act",
        trigger_type="activation",
        entity_names=None,
        threshold=None,
        priority="normal",
        context=None,
        see_also=None,
        refresh_trigger="activation",
    )

    assert result.status_code == 400
    assert result.payload == {"detail": "bad trigger"}


@pytest.mark.asyncio
async def test_api_dismiss_intention_response_surface_maps_missing_to_404() -> None:
    manager = MagicMock()
    manager.dismiss_intention = AsyncMock(side_effect=LookupError("missing"))

    result = await build_api_dismiss_intention_response_surface(
        manager,
        group_id="native_brain",
        intention_id="int_missing",
        hard=False,
    )

    assert result.status_code == 404
    assert result.payload == api_intention_not_found_payload()


@pytest.mark.asyncio
async def test_api_intention_response_surfaces_preserve_success_status() -> None:
    manager = MagicMock()
    manager.create_intention = AsyncMock(return_value="int_native")
    manager.dismiss_intention = AsyncMock()

    created = await build_api_create_intention_response_surface(
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
    dismissed = await build_api_dismiss_intention_response_surface(
        manager,
        group_id="native_brain",
        intention_id="int_native",
        hard=True,
    )

    assert created.status_code == 200
    assert created.payload["intentionId"] == "int_native"
    assert dismissed.status_code == 200
    assert dismissed.payload == {
        "status": "dismissed",
        "intentionId": "int_native",
        "hard": True,
    }


@pytest.mark.asyncio
async def test_mcp_intention_response_surfaces_map_errors() -> None:
    manager = MagicMock()
    manager.create_intention = AsyncMock(side_effect=ValueError("bad trigger"))
    manager.dismiss_intention = AsyncMock(side_effect=LookupError("missing"))

    created = await build_mcp_create_intention_response_surface(
        manager,
        group_id="native_brain",
        trigger_text="",
        action_text="Act",
        trigger_type="activation",
        entity_names=None,
        threshold=None,
        priority="normal",
        context=None,
        see_also=None,
        refresh_trigger="activation",
    )
    dismissed = await build_mcp_dismiss_intention_response_surface(
        manager,
        group_id="native_brain",
        intention_id="int_missing",
        hard=False,
    )

    assert created == {"status": "error", "message": "bad trigger"}
    assert dismissed == {"status": "error", "message": "missing"}


@pytest.mark.asyncio
async def test_mcp_intention_response_surfaces_preserve_success_payloads() -> None:
    manager = MagicMock()
    manager.create_intention = AsyncMock(return_value="int_native")
    manager.effective_intention_threshold.return_value = 0.72
    manager.dismiss_intention = AsyncMock()

    created = await build_mcp_create_intention_response_surface(
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
    dismissed = await build_mcp_dismiss_intention_response_surface(
        manager,
        group_id="native_brain",
        intention_id="int_native",
        hard=True,
    )

    assert created["status"] == "created"
    assert created["intention_id"] == "int_native"
    assert dismissed["status"] == "dismissed"
    assert dismissed["intention_id"] == "int_native"
