"""The outbox drain yields to an explicit recall in flight (recall_activity)."""

from __future__ import annotations

import asyncio
import time

import pytest

from engram.retrieval import recall_activity

pytestmark = pytest.mark.asyncio


async def test_wait_idle_returns_immediately_when_no_recall_is_running() -> None:
    waited = await recall_activity.wait_idle(max_wait=1.0)
    assert waited < 0.5


async def test_wait_idle_holds_while_a_recall_is_active_and_is_bounded() -> None:
    async def recall():
        async with recall_activity.active():
            await asyncio.sleep(0.4)

    task = asyncio.create_task(recall())
    await asyncio.sleep(0.05)
    assert recall_activity.inflight() == 1
    started = time.monotonic()
    waited = await recall_activity.wait_idle(max_wait=5.0)
    assert waited >= 0.3, f"drain did not wait for the recall ({waited:.2f}s)"
    assert recall_activity.inflight() == 0
    await task
    # bounded: a recall that never ends cannot stall the drain past max_wait
    async def stuck():
        async with recall_activity.active():
            await asyncio.sleep(10)

    stuck_task = asyncio.create_task(stuck())
    await asyncio.sleep(0.05)
    waited = await recall_activity.wait_idle(max_wait=0.3)
    assert 0.25 <= waited < 1.0
    stuck_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await stuck_task
    assert time.monotonic() - started < 3.0


async def test_explicit_recall_marks_itself_in_flight() -> None:
    """The recall surface wraps its budgeted run in recall_activity.active()."""
    from types import SimpleNamespace
    from unittest.mock import Mock

    from engram.config import ActivationConfig
    from engram.retrieval.recall_surface import _run_explicit_recall_with_budget

    seen: list[int] = []

    async def recall(**_k):
        seen.append(recall_activity.inflight())
        return []

    async def none(*_a, **_k):
        return []

    manager = SimpleNamespace(
        _graph=SimpleNamespace(find_entities_exact_name=none, find_entity_candidates=none),
        recall=recall,
        fast_recall_fallback=none,
        search_entities=none,
        record_memory_operation=Mock(),
        get_explicit_recall_packet_policy=lambda: SimpleNamespace(enabled=True, max_packets=3),
        get_memory_need_config=lambda: ActivationConfig(recall_budget_explicit_ms=2000),
        get_cached_memory_packets=Mock(return_value=None),
        get_recent_cached_memory_packets=Mock(return_value=[]),
    )
    await _run_explicit_recall_with_budget(
        manager, group_id="default", query="q", limit=5,
        cfg=manager.get_memory_need_config(), operation_source="api_recall",
    )
    assert seen == [1]
    assert recall_activity.inflight() == 0
