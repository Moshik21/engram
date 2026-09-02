"""The pre-pipeline rescue cascade must not eat the deep pipeline's wall.

Measured live 2026-09-02 (10 REST recalls, 4000ms wall): durable_entity_first
1500ms + preflight 250ms + durable_entity_rescue 2000ms = 3750ms spent BEFORE
``manager.recall`` started; the pipeline was then cancelled at ~250ms and a
2500ms post-timeout rescue ran on top. Every recall degraded, 6.7s median.

The config's own arithmetic (``recall_deep_pipeline_wall_budget_enabled``)
sizes the pipeline's serial substages to the FULL wall, so any pre-pipeline
stage that is not near-zero violates the design. This pins the contract:
with every rescue probe hanging, the pipeline is still entered inside a
small fraction of the wall and gets most of it.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engram.config import ActivationConfig
from engram.retrieval.recall_surface import _run_explicit_recall_with_budget

WALL_MS = 2000


async def _hang(*_args, **_kwargs):
    await asyncio.sleep(60)


def _manager(recall):
    graph = SimpleNamespace(
        find_entities_exact_name=_hang,
        find_entity_candidates=_hang,
    )
    return SimpleNamespace(
        _graph=graph,
        recall=recall,
        fast_recall_fallback=_hang,
        search_entities=_hang,
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=True, max_packets=3),
        get_memory_need_config=lambda: ActivationConfig(
            recall_budget_explicit_ms=WALL_MS,
            recall_fast_preflight_enabled=True,
        ),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
    )


@pytest.mark.asyncio
async def test_pipeline_entered_inside_a_third_of_the_wall_when_every_probe_hangs() -> None:
    entered_at: list[float] = []
    started = time.perf_counter()

    async def recall(**_kwargs):
        entered_at.append((time.perf_counter() - started) * 1000)
        return [{"entity": {"id": "e1", "name": "x", "type": "Fact"}, "score": 1.0}]

    manager = _manager(recall)
    results, metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id="default",
        query="what did we decide about the packet cache",
        limit=5,
        cfg=manager.get_memory_need_config(),
        operation_source="api_recall",
    )
    assert entered_at, "deep pipeline was never entered"
    assert entered_at[0] < WALL_MS / 3, (
        f"pipeline entered at {entered_at[0]:.0f}ms of a {WALL_MS}ms wall"
    )
    assert results and metadata["status"] == "ok"
    timings = metadata["stage_timings_ms"]
    assert "durable_entity_rescue" not in timings, "duplicate pre-pipeline rescue must not run"


@pytest.mark.asyncio
async def test_post_timeout_rescue_is_bounded_by_the_remaining_wall() -> None:
    """After the pipeline times out, the rescue must not add seconds on top."""

    async def recall(**_kwargs):
        await asyncio.sleep(60)

    manager = _manager(recall)
    started = time.perf_counter()
    results, metadata = await _run_explicit_recall_with_budget(
        manager,
        group_id="default",
        query="helixdb bounded episode listing route",
        limit=5,
        cfg=manager.get_memory_need_config(),
        operation_source="api_recall",
    )
    elapsed = (time.perf_counter() - started) * 1000
    assert metadata["status"] == "degraded"
    assert elapsed < WALL_MS * 1.25, f"recall took {elapsed:.0f}ms against a {WALL_MS}ms wall"


@pytest.mark.asyncio
async def test_probes_that_timed_out_back_off_on_the_next_recall() -> None:
    """Second recall inside the backoff window skips both timed-out probes."""
    calls = {"exact": 0, "preflight": 0}

    async def hang_exact(*_a, **_k):
        calls["exact"] += 1
        await asyncio.sleep(60)

    async def hang_preflight(*_a, **_k):
        calls["preflight"] += 1
        await asyncio.sleep(60)

    async def recall(**_kwargs):
        return [{"entity": {"id": "e1", "name": "x", "type": "Fact"}, "score": 1.0}]

    manager = _manager(recall)
    manager._graph.find_entities_exact_name = hang_exact
    manager._graph.find_entity_candidates = hang_exact
    manager.fast_recall_fallback = hang_preflight
    manager.search_entities = hang_exact

    _, first = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="reranker on the default tier", limit=5,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    started = time.perf_counter()
    _, second = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="reranker on the default tier", limit=5,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    pre_ms = (time.perf_counter() - started) * 1000
    assert "durable_entity_first" in first["stage_timings_ms"]
    assert "recall_fast_preflight" in first["stage_timings_ms"]
    assert "durable_entity_first_backoff" in second["stage_timings_ms"]
    assert "recall_fast_preflight_backoff" in second["stage_timings_ms"]
    assert calls["exact"] == 1 and calls["preflight"] == 1, calls
    assert pre_ms < 200, f"second recall still paid {pre_ms:.0f}ms before the pipeline"
