from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.preference_feedback import (
    FeedbackRatingError,
    build_api_explicit_feedback_surface,
    build_explicit_feedback_surface,
    build_mcp_explicit_feedback_surface,
)


@pytest.mark.asyncio
async def test_explicit_feedback_surface_records_valid_feedback() -> None:
    manager = MagicMock()
    manager.record_explicit_feedback = AsyncMock(
        return_value={
            "status": "recorded",
            "entity_id": "ent_native",
            "edge_type": "PREFERS",
        }
    )

    result = await build_explicit_feedback_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_native",
        rating=5,
        comment="prefer native mode",
    )

    assert result["status"] == "recorded"
    manager.record_explicit_feedback.assert_awaited_once_with(
        group_id="native_brain",
        entity_id="ent_native",
        rating=5,
        comment="prefer native mode",
    )


@pytest.mark.asyncio
async def test_explicit_feedback_surface_rejects_invalid_rating() -> None:
    manager = MagicMock()
    manager.record_explicit_feedback = AsyncMock()

    with pytest.raises(FeedbackRatingError):
        await build_explicit_feedback_surface(
            manager,
            group_id="native_brain",
            entity_id="ent_native",
            rating=6,
            comment=None,
        )

    manager.record_explicit_feedback.assert_not_called()


@pytest.mark.asyncio
async def test_api_explicit_feedback_surface_maps_invalid_rating_to_400() -> None:
    manager = MagicMock()
    manager.record_explicit_feedback = AsyncMock()

    result = await build_api_explicit_feedback_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_native",
        rating=0,
        comment=None,
    )

    assert result.status_code == 400
    assert result.payload == {"error": "Rating must be between 1 and 5"}
    manager.record_explicit_feedback.assert_not_called()


@pytest.mark.asyncio
async def test_api_explicit_feedback_surface_maps_missing_entity_to_404() -> None:
    manager = MagicMock()
    manager.record_explicit_feedback = AsyncMock(side_effect=ValueError("Entity missing"))

    result = await build_api_explicit_feedback_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_missing",
        rating=5,
        comment=None,
    )

    assert result.status_code == 404
    assert result.payload == {"error": "Entity missing"}
    manager.record_explicit_feedback.assert_awaited_once_with(
        group_id="native_brain",
        entity_id="ent_missing",
        rating=5,
        comment=None,
    )


@pytest.mark.asyncio
async def test_mcp_explicit_feedback_surface_returns_error_payload_for_invalid_rating() -> None:
    manager = MagicMock()
    manager.record_explicit_feedback = AsyncMock()

    result = await build_mcp_explicit_feedback_surface(
        manager,
        group_id="native_brain",
        entity_id="ent_native",
        rating=6,
        comment=None,
    )

    assert result == {"error": "Rating must be between 1 and 5"}
    manager.record_explicit_feedback.assert_not_called()
