from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.ingestion.adjudication_surface import load_episode_adjudication_requests


@pytest.mark.asyncio
async def test_load_episode_adjudication_requests_supports_async_manager_facade() -> None:
    manager = SimpleNamespace(
        get_episode_adjudications=AsyncMock(return_value=[{"request_id": "adj_1"}])
    )

    result = await load_episode_adjudication_requests(
        manager,
        episode_id="ep_1",
        group_id="brain_a",
    )

    assert result == [{"request_id": "adj_1"}]
    manager.get_episode_adjudications.assert_awaited_once_with("ep_1", "brain_a")


@pytest.mark.asyncio
async def test_load_episode_adjudication_requests_supports_sync_compatibility_facade() -> None:
    manager = SimpleNamespace(
        get_episode_adjudications=Mock(return_value=[{"request_id": "adj_1"}])
    )

    result = await load_episode_adjudication_requests(
        manager,
        episode_id="ep_1",
        group_id="brain_a",
    )

    assert result == [{"request_id": "adj_1"}]
    manager.get_episode_adjudications.assert_called_once_with("ep_1", "brain_a")


@pytest.mark.asyncio
async def test_load_episode_adjudication_requests_returns_empty_without_facade() -> None:
    assert (
        await load_episode_adjudication_requests(
            SimpleNamespace(),
            episode_id="ep_1",
            group_id="brain_a",
        )
        == []
    )


@pytest.mark.asyncio
async def test_load_episode_adjudication_requests_returns_empty_for_unexpected_shape() -> None:
    manager = SimpleNamespace(get_episode_adjudications=AsyncMock(return_value={"bad": "shape"}))

    assert (
        await load_episode_adjudication_requests(
            manager,
            episode_id="ep_1",
            group_id="brain_a",
        )
        == []
    )
