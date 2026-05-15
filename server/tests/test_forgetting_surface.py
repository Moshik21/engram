from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.forgetting import (
    build_api_forget_surface,
    build_mcp_forget_surface,
)


@pytest.mark.asyncio
async def test_api_forget_surface_preserves_entity_first_contract() -> None:
    manager = MagicMock()
    manager.forget_entity = AsyncMock(return_value={"status": "forgotten"})
    manager.forget_fact = AsyncMock()

    result = await build_api_forget_surface(
        manager,
        group_id="native_brain",
        entity_name="Engram",
        fact=SimpleNamespace(subject="Engram", predicate="USES", object="Helix"),
        reason="obsolete",
    )

    assert result == {"status": "forgotten"}
    manager.forget_entity.assert_awaited_once_with(
        entity_name="Engram",
        group_id="native_brain",
        reason="obsolete",
    )
    manager.forget_fact.assert_not_called()


@pytest.mark.asyncio
async def test_api_forget_surface_accepts_fact_objects_and_rejects_empty() -> None:
    manager = MagicMock()
    manager.forget_fact = AsyncMock(return_value={"status": "forgotten"})

    result = await build_api_forget_surface(
        manager,
        group_id="native_brain",
        entity_name=None,
        fact=SimpleNamespace(subject="Engram", predicate="USES", object="Helix"),
        reason=None,
    )

    assert result == {"status": "forgotten"}
    manager.forget_fact.assert_awaited_once_with(
        subject_name="Engram",
        predicate="USES",
        object_name="Helix",
        group_id="native_brain",
        reason=None,
    )
    with pytest.raises(ValueError):
        await build_api_forget_surface(
            manager,
            group_id="native_brain",
            entity_name=None,
            fact=None,
            reason=None,
        )


@pytest.mark.asyncio
async def test_mcp_forget_surface_requires_exactly_one_target() -> None:
    manager = MagicMock()
    manager.forget_fact = AsyncMock(return_value={"status": "forgotten"})

    missing = await build_mcp_forget_surface(
        manager,
        group_id="native_brain",
        entity_name=None,
        fact=None,
        reason=None,
    )
    both = await build_mcp_forget_surface(
        manager,
        group_id="native_brain",
        entity_name="Engram",
        fact={"subject": "Engram", "predicate": "USES", "object": "Helix"},
        reason=None,
    )
    fact = await build_mcp_forget_surface(
        manager,
        group_id="native_brain",
        entity_name=None,
        fact={"subject": "Engram", "predicate": "USES", "object": "Helix"},
        reason="obsolete",
    )

    assert missing["status"] == "error"
    assert both["status"] == "error"
    assert fact == {"status": "forgotten"}
    manager.forget_fact.assert_awaited_once_with(
        subject_name="Engram",
        predicate="USES",
        object_name="Helix",
        group_id="native_brain",
        reason="obsolete",
    )
