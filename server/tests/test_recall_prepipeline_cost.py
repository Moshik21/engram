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
        # slower than the pipeline-first hedge, so the preflight is consulted
        await asyncio.sleep(0.6)
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
    _, second = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="reranker on the default tier", limit=5,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert "durable_entity_first" in first["stage_timings_ms"]
    assert "recall_fast_preflight" in first["stage_timings_ms"]
    assert "durable_entity_first_backoff" in second["stage_timings_ms"]
    assert "recall_fast_preflight_backoff" in second["stage_timings_ms"]
    assert calls["exact"] == 1 and calls["preflight"] == 1, "probes re-asked inside the backoff"
    probe_ms = sum(
        v for k, v in second["stage_timings_ms"].items()
        if k in ("durable_entity_first", "recall_fast_preflight")
    )
    assert probe_ms < 50, f"second recall still paid {probe_ms:.0f}ms of probes"


@pytest.mark.asyncio
async def test_short_preflight_page_does_not_replace_a_non_empty_pipeline() -> None:
    """2026-09-03: one bare cue row from preflight short-circuited a query whose
    episode the deep pipeline finds. A short page only answers when the pipeline
    is empty."""
    cue_row = {
        "result_type": "cue_episode",
        "cue": {"cue_text": "Thompson sampling removed (cue)"},
        "score": 0.5,
    }
    episode_row = {
        "result_type": "episode",
        "episode": {"id": "ep_ts", "content": "Thompson sampling removed: noise"},
        "score": 0.9,
    }

    async def recall(**_k):
        await asyncio.sleep(0.6)  # slower than the pipeline-first hedge
        return [episode_row]

    async def preflight(**_k):
        return [cue_row]

    async def none(*_a, **_k):
        return []

    from types import SimpleNamespace
    from unittest.mock import Mock

    manager = SimpleNamespace(
        _graph=SimpleNamespace(find_entities_exact_name=none, find_entity_candidates=none),
        recall=recall, fast_recall_fallback=preflight, search_entities=none,
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=True, max_packets=3),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=2000),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
    )
    results, md = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="why was Thompson sampling removed", limit=3,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert results == [episode_row], "the pipeline's episode must win over a one-row preflight page"
    assert md["stage_timings_ms"].get("recall_fast_preflight_short_page") == 1.0

    async def empty(**_k):
        return []

    manager.recall = empty
    results, md = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="why was Thompson sampling removed", limit=3,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert results == [cue_row] and md["fallback_status"] == "fast_preflight_hit"


@pytest.mark.asyncio
async def test_pipeline_first_hedge_never_consults_the_preflight_when_the_pipeline_is_fast() -> None:
    """2026-09-03 meter: ts-kill's answer was in the pipeline's top-3 while the
    live call returned the preflight's full page in 15 ms without it. The
    preflight is a latency hedge and runs only when the pipeline has not
    answered inside the preflight's own timeout."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    calls = {"preflight": 0}
    row = {"result_type": "episode", "episode": {"id": "ep_ts", "content": "Thompson noise"}, "score": 0.9}

    async def fast_recall(**_k):
        return [row]

    async def preflight(**_k):
        calls["preflight"] += 1
        return [{"result_type": "cue_episode", "cue": {"cue_text": "Thompson (cue)"}, "score": 0.5}] * 3

    async def none(*_a, **_k):
        return []

    manager = SimpleNamespace(
        _graph=SimpleNamespace(find_entities_exact_name=none, find_entity_candidates=none),
        recall=fast_recall, fast_recall_fallback=preflight, search_entities=none,
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=True, max_packets=3),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=2000),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
    )
    results, md = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="why was Thompson sampling removed", limit=3,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert results == [row]
    assert calls["preflight"] == 0, "a fast pipeline must not be pre-empted by the preflight"
    assert "recall_pipeline_first_hit" in md["stage_timings_ms"]

    async def slow_recall(**_k):
        await asyncio.sleep(0.6)
        return [row]

    manager.recall = slow_recall
    results, md = await _run_explicit_recall_with_budget(
        manager, group_id="default", query="why was Thompson sampling removed", limit=3,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert calls["preflight"] == 1 and md["fallback_status"] == "fast_preflight_hit"
    assert len(results) == 3, "a slow pipeline hands the answer to the preflight's full page"
