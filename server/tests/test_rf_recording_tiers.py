"""RF M1.5 — P9 records with an EXPLICIT tier="surfaced" kwarg.

Behavior coverage for M1.5/M1.6 (zero usage_events from surfacing, exactly one
(ts, 0.1) mentioned event per organic commit, bootstrap/re-commit exclusions)
lives in test_usage_event_tiers.py against the real store. This file pins the
one thing that suite cannot: the get_context delivery loop passes the tier
explicitly (grep-able intent), rather than leaning on the default — the P1
equivalent is pinned in test_recall_primary_results.py.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.retrieval.context_builder import MemoryContextBuilder

GROUP = "rf_tiers"


@pytest.mark.asyncio
async def test_get_context_delivery_passes_surfaced_tier() -> None:
    activation = MagicMock()
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(
        return_value=[
            (
                "ent_recent",
                ActivationState(node_id="ent_recent", access_history=[time.time() - 5]),
            )
        ]
    )
    graph = MagicMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(
        return_value=Entity(
            id="ent_recent",
            name="RecentEnt",
            entity_type="Concept",
            summary="recent",
            group_id=GROUP,
        )
    )
    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=ActivationConfig(identity_core_enabled=False, briefing_enabled=False),
        recall=AsyncMock(return_value=[]),
        list_intentions=AsyncMock(return_value=[]),
        resolve_entity_name=AsyncMock(return_value=""),
        publish_access_event=AsyncMock(),
    )

    result = await builder.get_context(group_id=GROUP)

    assert "RecentEnt" in result["context"]  # delivered, so recorded
    tiers = [call.kwargs.get("tier") for call in activation.record_access.await_args_list]
    assert tiers == ["surfaced"]
