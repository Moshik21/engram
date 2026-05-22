from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.evaluation.label_service import (
    build_recall_evaluation_write_surface,
    build_session_continuity_evaluation_write_surface,
    persist_recall_eval_sample,
    persist_session_continuity_sample,
)


@pytest.mark.asyncio
async def test_persist_recall_eval_sample_clamps_counts_and_saves() -> None:
    store = AsyncMock()

    sample = await persist_recall_eval_sample(
        store,
        group_id="brain_a",
        recall_triggered=True,
        recall_helped=False,
        recall_needed=True,
        packets_surfaced=-2,
        packets_used=-1,
        false_recalls=-3,
        stale_packets=-4,
        corrected_packets=-5,
        source="mcp",
        query="native recall",
        notes="operator label",
    )

    assert sample.group_id == "brain_a"
    assert sample.recall_triggered is True
    assert sample.recall_helped is False
    assert sample.recall_needed is True
    assert sample.packets_surfaced == 0
    assert sample.packets_used == 0
    assert sample.false_recalls == 0
    assert sample.stale_packets == 0
    assert sample.corrected_packets == 0
    assert sample.source == "mcp"
    assert sample.query == "native recall"
    assert sample.notes == "operator label"
    store.save_recall_sample.assert_awaited_once_with(sample)


@pytest.mark.asyncio
async def test_persist_session_continuity_sample_saves() -> None:
    store = AsyncMock()

    sample = await persist_session_continuity_sample(
        store,
        group_id="brain_a",
        baseline_score=0.4,
        memory_score=0.9,
        open_loop_expected=True,
        open_loop_recovered=True,
        temporal_expected=True,
        temporal_correct=False,
        source="rest",
        scenario="handoff",
        notes="manual label",
    )

    assert sample.group_id == "brain_a"
    assert sample.baseline_score == 0.4
    assert sample.memory_score == 0.9
    assert sample.open_loop_expected is True
    assert sample.open_loop_recovered is True
    assert sample.temporal_expected is True
    assert sample.temporal_correct is False
    assert sample.source == "rest"
    assert sample.scenario == "handoff"
    assert sample.notes == "manual label"
    store.save_session_sample.assert_awaited_once_with(sample)


@pytest.mark.asyncio
async def test_recall_evaluation_write_surface_persists_and_presents() -> None:
    store = AsyncMock()

    payload = await build_recall_evaluation_write_surface(
        store,
        group_id="brain_a",
        surface="mcp",
        recall_triggered=True,
        recall_helped=True,
        packets_surfaced=2,
        packets_used=1,
        stale_packets=1,
        corrected_packets=1,
        source="mcp",
    )

    store.save_recall_sample.assert_awaited_once()
    assert payload["status"] == "stored"
    assert payload["operation"] == "record_recall_evaluation"
    assert payload["group_id"] == "brain_a"
    assert payload["sample"]["packets_used"] == 1
    assert payload["sample"]["stale_packets"] == 1
    assert payload["sample"]["corrected_packets"] == 1


@pytest.mark.asyncio
async def test_session_continuity_write_surface_persists_and_presents() -> None:
    store = AsyncMock()

    payload = await build_session_continuity_evaluation_write_surface(
        store,
        group_id="brain_a",
        surface="rest",
        baseline_score=0.2,
        memory_score=0.8,
        open_loop_expected=True,
        open_loop_recovered=True,
    )

    store.save_session_sample.assert_awaited_once()
    assert payload["status"] == "stored"
    assert payload["groupId"] == "brain_a"
    assert payload["sample"]["memoryScore"] == 0.8
