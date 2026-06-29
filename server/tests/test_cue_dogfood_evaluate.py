"""Dogfood cue usefulness: surfaced recall increments stats and evaluation signals."""

from __future__ import annotations

import pytest

from engram.evaluation.brain_loop_report import build_brain_loop_report
from engram.graph_manager import GraphManager
from engram.lifecycle_summary import build_lifecycle_summary
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.utils.dates import utc_now
from tests.conftest import MockExtractor
from tests.test_consolidation_profiles import _quiet_sqlite_recall_config


@pytest.mark.asyncio
async def test_surfaced_recall_makes_cue_usefulness_measurable(tmp_path) -> None:
    cfg = _quiet_sqlite_recall_config()
    graph_store = SQLiteGraphStore(str(tmp_path / "cue_dogfood.db"))
    await graph_store.initialize()
    search_index = FTS5SearchIndex(graph_store._db_path)
    await search_index.initialize(db=graph_store._db)
    activation_store = MemoryActivationStore(cfg=cfg)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        MockExtractor(),
        cfg=cfg,
    )

    episode = Episode(
        id="ep_dogfood_cue",
        content="The migration to native Helix finished on Tuesday.",
        source="test",
        status=EpisodeStatus.COMPLETED,
        projection_state=EpisodeProjectionState.CUE_ONLY,
        group_id="default",
        created_at=utc_now(),
    )
    await graph_store.create_episode(episode)
    await graph_store.upsert_episode_cue(
        EpisodeCue(
            episode_id=episode.id,
            group_id="default",
            projection_state=EpisodeProjectionState.CUE_ONLY,
            cue_text="native Helix migration",
            hit_count=0,
            surfaced_count=0,
        ),
    )

    for _ in range(2):
        await manager.recall(
            "native Helix migration",
            group_id="default",
            limit=5,
            record_access=False,
            interaction_type="surfaced",
            interaction_source="recall",
        )

    stats = await graph_store.get_stats(group_id="default")
    cue_metrics = stats["cue_metrics"]
    assert cue_metrics["cue_hit_count"] > 0
    assert cue_metrics["cue_surfaced_count"] > 0

    report = build_brain_loop_report(stats, group_id="default")
    cue_usefulness = report["evaluation_signals"]["cue_usefulness"]
    assert cue_usefulness["status"] != "needs_feedback"
    assert cue_usefulness["status"] == "measured"

    lifecycle = await build_lifecycle_summary(
        group_id="default",
        manager=manager,
        graph_store=graph_store,
    )
    assert lifecycle["cue"]["hitCount"] > 0
    assert lifecycle["cue"]["surfacedCount"] > 0
    await graph_store.close()
