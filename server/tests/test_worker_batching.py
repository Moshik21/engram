from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.worker_batching import EpisodeWorkerBatchMerger, PendingEpisode
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue


def _cfg(**overrides) -> ActivationConfig:
    defaults = {
        "cue_layer_enabled": True,
        "cue_vector_index_enabled": True,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


@pytest.mark.asyncio
async def test_worker_batch_merger_rebuilds_primary_and_retires_secondary_cues() -> None:
    cfg = _cfg()
    group_id = "brain"
    primary = PendingEpisode("ep_primary", "Alice asked about spreading activation", "auto:prompt")
    secondary = PendingEpisode(
        "ep_secondary",
        "Spreading activation is useful for memory recall",
        "auto:response",
    )
    merged_content = f"{primary.content}\n\n{secondary.content}"
    stored_primary = Episode(
        id=primary.episode_id,
        content=merged_content,
        source=primary.source,
        status=EpisodeStatus.QUEUED,
        group_id=group_id,
        projection_state=EpisodeProjectionState.CUED,
    )
    previous_primary_cue = EpisodeCue(
        episode_id=primary.episode_id,
        group_id=group_id,
        cue_text="old cue",
        hit_count=3,
        policy_score=0.9,
    )
    secondary_cue = EpisodeCue(
        episode_id=secondary.episode_id,
        group_id=group_id,
        cue_text="secondary cue",
    )
    graph = SimpleNamespace(
        update_episode=AsyncMock(),
        get_episode_by_id=AsyncMock(return_value=stored_primary),
        get_episode_cue=AsyncMock(side_effect=[previous_primary_cue, secondary_cue]),
        upsert_episode_cue=AsyncMock(),
        update_episode_cue=AsyncMock(),
    )
    search = SimpleNamespace(index_episode_cue=AsyncMock())
    merger = EpisodeWorkerBatchMerger(graph, search, cfg)

    result = await merger.merge([primary, secondary], group_id)

    assert result.primary_episode_id == primary.episode_id
    assert result.merged_content == merged_content
    assert result.retired_episode_ids == (secondary.episode_id,)
    graph.update_episode.assert_any_await(
        primary.episode_id,
        {"content": merged_content},
        group_id=group_id,
    )
    graph.upsert_episode_cue.assert_awaited_once()
    rebuilt_cue = graph.upsert_episode_cue.await_args.args[0]
    assert rebuilt_cue.episode_id == primary.episode_id
    assert rebuilt_cue.hit_count == previous_primary_cue.hit_count
    assert rebuilt_cue.policy_score >= previous_primary_cue.policy_score
    graph.update_episode_cue.assert_awaited_once_with(
        secondary.episode_id,
        {
            "projection_state": EpisodeProjectionState.MERGED,
            "route_reason": f"merged_into:{primary.episode_id}",
            "cue_text": "",
            "entity_mentions": [],
            "temporal_markers": [],
            "quote_spans": [],
            "contradiction_keys": [],
            "first_spans": [],
        },
        group_id=group_id,
    )
    assert search.index_episode_cue.await_count == 2


@pytest.mark.asyncio
async def test_worker_batch_merger_skips_cue_writes_when_cue_layer_disabled() -> None:
    cfg = _cfg(cue_layer_enabled=False)
    graph = SimpleNamespace(
        update_episode=AsyncMock(),
        get_episode_by_id=AsyncMock(),
        get_episode_cue=AsyncMock(),
        upsert_episode_cue=AsyncMock(),
        update_episode_cue=AsyncMock(),
    )
    search = SimpleNamespace(index_episode_cue=AsyncMock())
    merger = EpisodeWorkerBatchMerger(graph, search, cfg)

    await merger.merge(
        [
            PendingEpisode("ep_primary", "first", "auto:prompt"),
            PendingEpisode("ep_secondary", "second", "auto:response"),
        ],
        "brain",
    )

    graph.upsert_episode_cue.assert_not_awaited()
    graph.update_episode_cue.assert_not_awaited()
    search.index_episode_cue.assert_not_awaited()
