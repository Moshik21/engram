"""Tests for retrieval priming updates."""

from __future__ import annotations

from unittest.mock import AsyncMock, call

import pytest

from engram.config import ActivationConfig
from engram.retrieval.priming import RecallPrimingUpdater


@pytest.mark.asyncio
async def test_priming_updates_buffer_for_top_entity_neighbors() -> None:
    graph = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(
        side_effect=[
            [("neighbor_a", 1.0, "RELATES_TO", "Thing"), ("neighbor_b", 0.5, "USES", "Thing")],
            [("neighbor_c", 0.25, "MENTIONS", "Thing")],
        ]
    )
    cfg = ActivationConfig(
        retrieval_priming_enabled=True,
        retrieval_priming_top_n=2,
        retrieval_priming_boost=0.2,
        retrieval_priming_ttl_seconds=30.0,
        retrieval_priming_max_neighbors=1,
    )
    buffer: dict[str, tuple[float, float]] = {}

    await RecallPrimingUpdater(
        graph_store=graph,
        cfg=cfg,
        time_fn=lambda: 100.0,
    ).update(
        [
            {"result_type": "entity", "entity": {"id": "ent_a"}, "score": 0.9},
            {"result_type": "entity", "entity": {"id": "ent_b"}, "score": 0.8},
            {"result_type": "entity", "entity": {"id": "ent_c"}, "score": 0.7},
        ],
        group_id="native_brain",
        priming_buffer=buffer,
    )

    assert buffer == {
        "neighbor_a": (0.2, 130.0),
        "neighbor_c": (0.05, 130.0),
    }
    graph.get_active_neighbors_with_weights.assert_has_awaits(
        [
            call("ent_a", "native_brain"),
            call("ent_b", "native_brain"),
        ]
    )


@pytest.mark.asyncio
async def test_priming_skips_non_entities_and_missing_ids() -> None:
    graph = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(return_value=[("neighbor", 1.0)])
    cfg = ActivationConfig(retrieval_priming_enabled=True)
    buffer: dict[str, tuple[float, float]] = {}

    await RecallPrimingUpdater(
        graph_store=graph,
        cfg=cfg,
        time_fn=lambda: 100.0,
    ).update(
        [
            {"result_type": "episode", "episode": {"id": "ep_1"}, "score": 0.9},
            {"result_type": "entity", "entity": {}, "score": 0.8},
            {"result_type": "entity", "entity": {"id": ""}, "score": 0.7},
        ],
        group_id="native_brain",
        priming_buffer=buffer,
    )

    assert buffer == {}
    graph.get_active_neighbors_with_weights.assert_not_awaited()


@pytest.mark.asyncio
async def test_priming_neighbor_errors_do_not_abort_updates() -> None:
    graph = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(
        side_effect=[RuntimeError("lookup failed"), [("neighbor_b", 0.5)]]
    )
    cfg = ActivationConfig(
        retrieval_priming_enabled=True,
        retrieval_priming_top_n=2,
        retrieval_priming_boost=0.2,
    )
    buffer: dict[str, tuple[float, float]] = {}

    await RecallPrimingUpdater(
        graph_store=graph,
        cfg=cfg,
        time_fn=lambda: 10.0,
    ).update(
        [
            {"result_type": "entity", "entity": {"id": "ent_a"}, "score": 0.9},
            {"result_type": "entity", "entity": {"id": "ent_b"}, "score": 0.8},
        ],
        group_id="native_brain",
        priming_buffer=buffer,
    )

    assert buffer["neighbor_b"] == pytest.approx((0.1, 40.0))


@pytest.mark.asyncio
async def test_priming_disabled_does_not_mutate_buffer() -> None:
    graph = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(return_value=[("neighbor", 1.0)])
    cfg = ActivationConfig(retrieval_priming_enabled=False)
    buffer: dict[str, tuple[float, float]] = {}

    await RecallPrimingUpdater(graph_store=graph, cfg=cfg).update(
        [{"result_type": "entity", "entity": {"id": "ent_a"}, "score": 0.9}],
        group_id="native_brain",
        priming_buffer=buffer,
    )

    assert buffer == {}
    graph.get_active_neighbors_with_weights.assert_not_awaited()
