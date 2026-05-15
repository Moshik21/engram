"""Tests for offline capture replay orchestration."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.ingestion.offline_replay import OfflineReplayService


@pytest.mark.asyncio
async def test_offline_replay_stores_valid_entries_in_active_group():
    entries = [
        {
            "content": "offline replay content long enough",
            "source": "offline:test",
            "session_id": "session_1",
        },
        {"content": "too short"},
    ]
    store_episode = AsyncMock(return_value="ep_queued")
    service = OfflineReplayService(
        drain_queue=lambda: entries,
        dedup_check=lambda _content: False,
        store_episode=store_episode,
    )

    result = await service.replay_queue(group_id="tenant_brain")

    assert result.as_payload() == {"replayed": 1, "skipped": 1, "total": 2}
    store_episode.assert_awaited_once_with(
        content="offline replay content long enough",
        group_id="tenant_brain",
        source="offline:test",
        session_id="session_1",
    )


@pytest.mark.asyncio
async def test_offline_replay_counts_deduped_and_failed_entries_as_skipped():
    entries = [
        {"content": "deduped replay content long enough"},
        {"content": "failed replay content long enough"},
    ]
    store_episode = AsyncMock(side_effect=RuntimeError("store failed"))
    service = OfflineReplayService(
        drain_queue=lambda: entries,
        dedup_check=lambda content: content.startswith("deduped"),
        store_episode=store_episode,
    )

    result = await service.replay_queue(group_id="tenant_brain")

    assert result.as_payload() == {"replayed": 0, "skipped": 2, "total": 2}
    store_episode.assert_awaited_once_with(
        content="failed replay content long enough",
        group_id="tenant_brain",
        source="offline:replay",
        session_id=None,
    )
