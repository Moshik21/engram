from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.ingestion.capture_surface import (
    build_api_observe_write_surface,
    build_api_remember_write_surface,
)

_EVENTS = [{"name": "Launch", "date": "2026-01-15", "source_span": "Launch"}]


@pytest.mark.asyncio
async def test_build_api_observe_write_surface_persists_event_annotations() -> None:
    manager = MagicMock()
    manager.store_episode = AsyncMock(return_value="ep_observe")
    manager._graph = MagicMock()
    manager._graph.store_evidence = AsyncMock()

    response = await build_api_observe_write_surface(
        manager,
        content="Launch happened.",
        group_id="native_brain",
        source="dashboard",
        events=_EVENTS,
    )

    # Event annotations ride the deferred-evidence pipeline on the cheap path.
    manager._graph.store_evidence.assert_awaited_once()
    assert response["status"] == "observed"


@pytest.mark.asyncio
async def test_build_api_remember_write_surface_merges_event_proposals() -> None:
    manager = MagicMock()
    manager.ingest_episode = AsyncMock(return_value="ep_remember")
    manager.edge_adjudication_client_enabled = MagicMock(return_value=False)
    manager.get_episode_adjudications = AsyncMock(return_value=[])

    response = await build_api_remember_write_surface(
        manager,
        content="Launch happened.",
        group_id="native_brain",
        source="dashboard",
        events=_EVENTS,
    )

    manager.ingest_episode.assert_awaited_once()
    call_kwargs = manager.ingest_episode.await_args.kwargs
    proposed_entities = call_kwargs["proposed_entities"] or []
    proposed_relationships = call_kwargs["proposed_relationships"] or []

    # The dated event is folded into the client-proposal payloads as an Event
    # node, a Date node, and an OCCURRED_ON edge.
    assert {"name": "Launch", "entity_type": "Event", "source_span": "Launch"} in (
        proposed_entities
    )
    assert any(
        rel["predicate"] == "OCCURRED_ON" and rel["object"] == "2026-01-15"
        for rel in proposed_relationships
    )
    assert response["status"] == "remembered"
