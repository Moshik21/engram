from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from engram.ingestion.adjudication_surface import (
    build_api_adjudication_resolution_surface,
    build_mcp_adjudication_resolution_surface,
    load_episode_adjudication_requests,
)


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


@pytest.mark.asyncio
async def test_adjudication_resolution_surfaces_share_submission_and_shapes() -> None:
    outcome = SimpleNamespace(
        status="resolved",
        request_id="adj_1",
        committed_ids={"ev_1": "rel_1"},
        superseded_evidence_ids=["ev_old"],
        replacement_evidence_ids=["ev_new"],
    )
    manager = SimpleNamespace(submit_adjudication_resolution=AsyncMock(return_value=outcome))

    api = await build_api_adjudication_resolution_surface(
        manager,
        group_id="native_brain",
        request_id="adj_1",
        entities=[{"name": "Alice"}],
        relationships=[{"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"}],
        reject_evidence_ids=["ev_reject"],
        model_tier="opus",
        rationale="human approved",
    )
    mcp = await build_mcp_adjudication_resolution_surface(
        manager,
        group_id="native_brain",
        request_id="adj_1",
        entities=[{"name": "Alice"}],
        relationships=[{"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"}],
        reject_evidence_ids=["ev_reject"],
        model_tier="opus",
        rationale="human approved",
    )

    assert manager.submit_adjudication_resolution.await_count == 2
    assert manager.submit_adjudication_resolution.await_args.kwargs == {
        "entities": [{"name": "Alice"}],
        "relationships": [{"subject": "Alice", "predicate": "WORKS_AT", "object": "Engram"}],
        "reject_evidence_ids": ["ev_reject"],
        "source": "client_adjudication",
        "model_tier": "opus",
        "rationale": "human approved",
        "group_id": "native_brain",
    }
    assert api == {
        "status": "resolved",
        "requestId": "adj_1",
        "committedIds": {"ev_1": "rel_1"},
        "supersededEvidenceIds": ["ev_old"],
        "replacementEvidenceIds": ["ev_new"],
    }
    assert mcp == {
        "status": "resolved",
        "request_id": "adj_1",
        "committed_ids": {"ev_1": "rel_1"},
        "superseded_evidence_ids": ["ev_old"],
        "replacement_evidence_ids": ["ev_new"],
    }
