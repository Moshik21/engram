from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.retrieval.control import (
    RecallNeedThresholds,
    record_manager_memory_need_analysis,
    resolve_manager_recall_need_thresholds,
)


@pytest.mark.asyncio
async def test_resolve_manager_recall_need_thresholds_accepts_sync_and_async() -> None:
    sync_manager = MagicMock()
    sync_manager.get_recall_need_thresholds.return_value = RecallNeedThresholds(
        linguistic_score=0.4,
    )
    async_manager = MagicMock()
    async_manager.get_recall_need_thresholds = AsyncMock(
        return_value=RecallNeedThresholds(borderline_score=0.2),
    )

    assert (
        await resolve_manager_recall_need_thresholds(sync_manager, "brain")
    ).linguistic_score == 0.4
    assert (
        await resolve_manager_recall_need_thresholds(async_manager, "brain")
    ).borderline_score == 0.2


@pytest.mark.asyncio
async def test_record_manager_memory_need_analysis_accepts_sync_and_async() -> None:
    sync_manager = MagicMock()
    async_manager = MagicMock()
    async_manager.record_memory_need_analysis = AsyncMock()

    await record_manager_memory_need_analysis(sync_manager, "brain", object())
    await record_manager_memory_need_analysis(async_manager, "brain", object())

    sync_manager.record_memory_need_analysis.assert_called_once()
    async_manager.record_memory_need_analysis.assert_awaited_once()
