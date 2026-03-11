"""Tests for the triage consolidation phase."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.triage import TriagePhase
from engram.models.consolidation import CycleContext
from engram.models.episode import EpisodeProjectionState


@dataclass
class FakeEpisode:
    """Minimal episode stub for triage tests."""

    id: str
    content: str = ""
    status: str = "queued"
    source: str = "test"


def _make_graph_store(episodes: list[FakeEpisode] | None = None):
    """Create a mock graph store returning given episodes as QUEUED."""
    store = AsyncMock()
    store.get_episodes_paginated = AsyncMock(return_value=(episodes or [], None))
    store.update_episode = AsyncMock()
    return store


def _make_search_index():
    """Create a search index mock without accidental episode matches."""
    index = AsyncMock()
    index.search_episodes = AsyncMock(return_value=[])
    return index


def _make_cfg(**overrides) -> ActivationConfig:
    """Create an ActivationConfig with triage defaults + overrides."""
    defaults = {
        "triage_enabled": True,
        "triage_extract_ratio": 0.35,
        "triage_min_score": 0.2,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


@pytest.mark.asyncio
async def test_triage_scores_and_promotes():
    """Top-scoring episodes are promoted, rest are skipped."""
    episodes = [
        FakeEpisode(id="ep_high", content="Alex works at Anthropic in San Francisco since 2024"),
        FakeEpisode(id="ep_medium", content="Something about Python and testing"),
        FakeEpisode(id="ep_low", content="ok"),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=0.35)  # ~1 of 3

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=_make_search_index(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    assert result.phase == "triage"
    assert result.items_processed == 3
    # At least 1 promoted (max(1, int(3 * 0.35)) = 1)
    assert result.items_affected >= 1
    assert len(records) == 3

    extract_records = [r for r in records if r.decision == "extract"]
    skip_records = [r for r in records if r.decision == "skip"]
    assert len(extract_records) >= 1
    assert len(skip_records) >= 1

    # The highest-scored episode should be extracted
    assert extract_records[0].episode_id == "ep_high"
    assert extract_records[0].score > skip_records[0].score

    # project_episode called for promoted episodes
    assert manager.project_episode.call_count == len(extract_records)

    # update_episode called for skipped episodes
    assert graph.update_episode.call_count == len(skip_records)


@pytest.mark.asyncio
async def test_triage_respects_extract_ratio():
    """Correct number promoted based on ratio."""
    episodes = [
        FakeEpisode(
            id=f"ep_{i}",
            content=(
                f"Alice_{i} works at Acme Corp in Berlin since 2024 and prefers Python "
                f"for project {i}."
            ),
        )
        for i in range(10)
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=0.40)  # 4 of 10

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=_make_search_index(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    extract_count = sum(1 for r in records if r.decision == "extract")
    assert extract_count == 4  # max(1, int(10 * 0.40))
    assert result.items_processed == 10


@pytest.mark.asyncio
async def test_triage_allows_zero_extractions_below_threshold():
    """No episode is forced through when nothing clears the utility threshold."""
    episodes = [
        FakeEpisode(id="ep_1", content="ok"),
        FakeEpisode(id="ep_2", content="sure"),
        FakeEpisode(id="ep_3", content="hi"),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=0.35, triage_min_score=0.4)

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    assert result.items_affected == 0
    assert all(r.decision == "skip" for r in records)
    manager.project_episode.assert_not_called()


@pytest.mark.asyncio
async def test_triage_dry_run_scores_only():
    """Dry run scores but doesn't modify episodes."""
    episodes = [
        FakeEpisode(id="ep_1", content="Important fact about Alice"),
        FakeEpisode(id="ep_2", content="hi"),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg()

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=_make_search_index(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=True,
    )

    assert result.items_processed == 2
    assert result.items_affected == 0  # No actual promotion in dry run
    assert len(records) == 2
    # No side effects
    manager.project_episode.assert_not_called()
    graph.update_episode.assert_not_called()


@pytest.mark.asyncio
async def test_triage_empty_queue_noop():
    """Empty QUEUED list returns 0 processed."""
    graph = _make_graph_store([])
    phase = TriagePhase()
    cfg = _make_cfg()

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
    )

    assert result.items_processed == 0
    assert result.items_affected == 0
    assert records == []


@pytest.mark.asyncio
async def test_triage_disabled_skips():
    """triage_enabled=False produces skipped phase result."""
    phase = TriagePhase()
    cfg = _make_cfg(triage_enabled=False)

    result, records = await phase.execute(
        group_id="default",
        graph_store=AsyncMock(),
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
    )

    assert result.status == "skipped"
    assert records == []


@pytest.mark.asyncio
async def test_triage_handles_extraction_failure():
    """project_episode failure doesn't crash the phase."""
    episodes = [
        FakeEpisode(id="ep_fail", content="Important data about Bob from New York"),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock(side_effect=RuntimeError("extraction boom"))

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=1.0)  # promote all

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    assert result.items_processed == 1
    # items_affected = 0 because extraction failed
    assert result.items_affected == 0
    assert records[0].decision == "extract"


@pytest.mark.asyncio
async def test_triage_updates_cycle_context():
    """Promoted episode IDs are tracked in CycleContext."""
    episodes = [
        FakeEpisode(id="ep_ctx", content="Alice works at Acme Corp in Berlin"),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=1.0)
    context = CycleContext()

    await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
        context=context,
    )

    assert "ep_ctx" in context.triage_promoted_ids


@pytest.mark.asyncio
async def test_triage_scoring_heuristics():
    """Scoring produces expected relative ordering."""
    phase = TriagePhase()
    cfg = _make_cfg()

    @dataclass
    class Ep:
        content: str

    # Long with many proper nouns > short with none
    long_rich = Ep(
        content="Alice and Bob discussed the Python project at Google. "
        "They decided to use React and TypeScript for the frontend. "
        "Charlie from Anthropic joined the meeting in San Francisco."
    )
    short_empty = Ep(content="ok")
    medium = Ep(content="Some discussion about testing")

    score_long = phase._score_episode(long_rich, cfg)
    score_short = phase._score_episode(short_empty, cfg)
    score_medium = phase._score_episode(medium, cfg)

    assert score_long > score_medium > score_short
    assert score_short < 0.25  # Very low content


@pytest.mark.asyncio
async def test_triage_skips_system_discourse():
    """System-discourse episodes are filtered out before scoring."""
    episodes = [
        FakeEpisode(
            id="ep_meta",
            content="Entity ent_abc123 has activation score 0.91 in the retrieval pipeline",
        ),
        FakeEpisode(
            id="ep_real",
            content="Alice works at Anthropic in San Francisco since 2024",
        ),
    ]
    graph = _make_graph_store(episodes)
    manager = MagicMock()
    manager.project_episode = AsyncMock()

    phase = TriagePhase(graph_manager=manager)
    cfg = _make_cfg(triage_extract_ratio=1.0)  # promote all remaining

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    # The meta episode should be skipped, only real one processed
    meta_records = [r for r in records if r.decision == "skip_meta"]
    extract_records = [r for r in records if r.decision == "extract"]
    assert len(meta_records) == 1
    assert meta_records[0].episode_id == "ep_meta"
    assert len(extract_records) == 1
    assert extract_records[0].episode_id == "ep_real"

    # Meta episode marked completed
    graph.update_episode.assert_called_once_with(
        "ep_meta",
        {
            "status": "completed",
            "skipped_meta": True,
            "projection_state": EpisodeProjectionState.CUE_ONLY.value,
            "last_projection_reason": "triage_skip_meta",
        },
        group_id="default",
    )
    # Real episode extracted
    manager.project_episode.assert_called_once_with("ep_real", "default")


@pytest.mark.asyncio
async def test_triage_dry_run_skips_meta_without_update():
    """In dry run, meta episodes are recorded but not updated in DB."""
    episodes = [
        FakeEpisode(
            id="ep_meta",
            content="The extraction pipeline scored ent_abc with activation score 0.9",
        ),
    ]
    graph = _make_graph_store(episodes)

    phase = TriagePhase()
    cfg = _make_cfg()

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=True,
    )

    assert len(records) == 1
    assert records[0].decision == "skip_meta"
    # No side effects in dry run
    graph.update_episode.assert_not_called()


@pytest.mark.asyncio
async def test_triage_no_manager_skips_extraction():
    """Without graph_manager, promoted episodes are not extracted."""
    episodes = [
        FakeEpisode(id="ep_noman", content="Data about Alice in London working at OpenAI"),
    ]
    graph = _make_graph_store(episodes)

    # No graph_manager provided
    phase = TriagePhase(graph_manager=None)
    cfg = _make_cfg(triage_extract_ratio=1.0)

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=AsyncMock(),
        cfg=cfg,
        cycle_id="cyc_test",
        dry_run=False,
    )

    assert result.items_processed == 1
    assert result.items_affected == 0  # No manager means no extraction
    assert records[0].decision == "extract"
