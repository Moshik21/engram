from __future__ import annotations

import argparse
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import EngramConfig
from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.lifecycle_cli import (
    build_lifecycle_summary_for_config,
    build_lifecycle_summary_from_args,
    configure_lifecycle_parser,
    format_lifecycle_summary_markdown,
)
from engram.lifecycle_summary import build_lifecycle_summary, build_mcp_lifecycle_summary_surface
from engram.models.consolidation import ConsolidationCycle, PhaseResult
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.storage.resolver import EngineMode
from engram.storage.sqlite.graph import SQLiteGraphStore


def _parse_lifecycle_args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    configure_lifecycle_parser(parser)
    return parser.parse_args(list(args))


@pytest.mark.asyncio
async def test_mcp_lifecycle_summary_surface_forwards_store_reader_and_clamped_limits() -> None:
    consolidation_store = SimpleNamespace(get_recent_cycles=AsyncMock(return_value=[]))
    activation_config = SimpleNamespace(decay_exponent=0.4)
    manager = SimpleNamespace(
        get_lifecycle_summary=AsyncMock(
            return_value={
                "groupId": "native_brain",
                "loop": ["capture", "cue", "project", "recall", "consolidate"],
            }
        )
    )

    result = await build_mcp_lifecycle_summary_surface(
        manager,
        group_id="native_brain",
        consolidation_store=consolidation_store,
        activation_config=activation_config,
        episode_limit=0,
        cycle_limit=0,
    )

    assert result["groupId"] == "native_brain"
    manager.get_lifecycle_summary.assert_awaited_once()
    kwargs = manager.get_lifecycle_summary.await_args.kwargs
    assert kwargs["group_id"] == "native_brain"
    assert kwargs["activation_config"] is activation_config
    assert kwargs["episode_limit"] == 1
    assert kwargs["cycle_limit"] == 1
    assert kwargs["consolidation_reader"].available is True
    assert kwargs["consolidation_engine"].is_running is False


@pytest.mark.asyncio
async def test_lifecycle_cli_reads_shared_lite_summary(tmp_path) -> None:
    db_path = tmp_path / "lifecycle.db"
    graph_store = SQLiteGraphStore(str(db_path))
    await graph_store.initialize()
    try:
        created_at = datetime(2026, 5, 12, 9, 0, 0)
        await graph_store.create_episode(
            Episode(
                id="ep_cli_1",
                content="Alice chose the Engram launch plan.",
                source="cli-test",
                status=EpisodeStatus.COMPLETED,
                group_id="cli_brain",
                created_at=created_at,
                updated_at=created_at,
                projection_state=EpisodeProjectionState.PROJECTED,
                last_projected_at=created_at,
                processing_duration_ms=25,
            )
        )
        await graph_store.upsert_episode_cue(
            EpisodeCue(
                episode_id="ep_cli_1",
                group_id="cli_brain",
                projection_state=EpisodeProjectionState.PROJECTED,
                cue_text="Alice Engram launch plan",
                hit_count=2,
                surfaced_count=1,
                selected_count=1,
                used_count=1,
                policy_score=0.75,
                projection_attempts=1,
            )
        )
    finally:
        await graph_store.close()

    args = _parse_lifecycle_args(
        "--sqlite-path",
        str(db_path),
        "--group-id",
        "cli_brain",
        "--episodes",
        "1",
        "--format",
        "json",
    )

    summary = await build_lifecycle_summary_from_args(args)

    assert summary["groupId"] == "cli_brain"
    assert summary["loop"] == ["capture", "cue", "project", "recall", "consolidate"]
    assert summary["totals"]["episodes"] == 1
    assert summary["totals"]["cues"] == 1
    assert summary["totals"]["projected"] == 1
    assert summary["cue"]["coverage"] == 1.0
    assert summary["project"]["status"] == "ready"
    assert summary["recentEpisodes"][0]["episodeId"] == "ep_cli_1"
    assert summary["recentEpisodes"][0]["cue"]["cueText"] == "Alice Engram launch plan"


def test_lifecycle_cli_accepts_helix_mode() -> None:
    args = _parse_lifecycle_args("--mode", "helix", "--format", "json")

    assert args.mode == "helix"
    assert args.sqlite_path is None


def test_lifecycle_cli_accepts_native_helix_data_dir(tmp_path) -> None:
    native_dir = tmp_path / "native-data"
    args = _parse_lifecycle_args(
        "--mode",
        "helix",
        "--helix-data-dir",
        str(native_dir),
        "--format",
        "json",
    )

    assert args.mode == "helix"
    assert args.helix_data_dir == native_dir


@pytest.mark.asyncio
async def test_lifecycle_summary_resolves_configured_mode(tmp_path, monkeypatch) -> None:
    requested_modes: list[str] = []

    async def fake_resolve_mode(mode: str) -> EngineMode:
        requested_modes.append(mode)
        return EngineMode.LITE

    monkeypatch.setattr("engram.lifecycle_cli.resolve_mode", fake_resolve_mode)
    config = EngramConfig(mode="helix")
    config.sqlite.path = str(tmp_path / "mode-aware.db")

    summary = await build_lifecycle_summary_for_config(
        config,
        episode_limit=0,
        cycle_limit=1,
        top_n=0,
    )

    assert requested_modes == ["helix"]
    assert summary["groupId"] == "default"
    assert summary["loop"] == ["capture", "cue", "project", "recall", "consolidate"]


@pytest.mark.asyncio
async def test_lifecycle_summary_applies_native_helix_data_dir(tmp_path, monkeypatch) -> None:
    native_dir = tmp_path / "native-data"
    requested_modes: list[str] = []

    class FakeGraphStore:
        async def initialize(self) -> None:
            pass

        async def close(self) -> None:
            pass

    class FakeSearchIndex:
        async def initialize(self) -> None:
            pass

        async def close(self) -> None:
            pass

    class FakeConsolidationStore:
        async def close(self) -> None:
            pass

    async def fake_resolve_mode(mode: str) -> EngineMode:
        requested_modes.append(mode)
        return EngineMode.HELIX

    def fake_create_lifecycle_stores(mode: EngineMode, config: EngramConfig):
        assert mode == EngineMode.HELIX
        assert config.helix.transport == "native"
        assert config.helix.data_dir == str(native_dir)
        return FakeGraphStore(), object(), FakeSearchIndex()

    async def fake_create_consolidation_store(mode, config, graph_store):
        assert mode == EngineMode.HELIX
        assert config.helix.data_dir == str(native_dir)
        return FakeConsolidationStore()

    async def fake_build_lifecycle_summary(**kwargs):
        assert kwargs["group_id"] == "native_brain"
        return {
            "groupId": kwargs["group_id"],
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
        }

    monkeypatch.setattr("engram.lifecycle_cli.resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(
        "engram.lifecycle_cli._create_lifecycle_stores",
        fake_create_lifecycle_stores,
    )
    monkeypatch.setattr(
        "engram.lifecycle_cli._create_consolidation_store",
        fake_create_consolidation_store,
    )
    monkeypatch.setattr("engram.lifecycle_cli.GraphManager", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "engram.lifecycle_cli.build_lifecycle_summary",
        fake_build_lifecycle_summary,
    )

    summary = await build_lifecycle_summary_for_config(
        EngramConfig(mode="helix"),
        helix_data_dir=native_dir,
        group_id="native_brain",
        episode_limit=0,
        cycle_limit=1,
        top_n=0,
    )

    assert requested_modes == ["helix"]
    assert summary["groupId"] == "native_brain"


def test_lifecycle_cli_markdown_renders_stage_snapshot() -> None:
    rendered = format_lifecycle_summary_markdown(
        {
            "groupId": "cli_brain",
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
            "totals": {
                "episodes": 1,
                "cues": 1,
                "projected": 1,
                "cycles": 0,
                "entities": 0,
            },
            "capture": {
                "status": "ready",
                "episodeCount": 1,
                "activeCount": 0,
                "latestEpisode": {"episodeId": "ep_cli_1"},
            },
            "cue": {
                "status": "ready",
                "coverage": 1.0,
                "usedCount": 1,
                "episodesWithoutCues": 0,
            },
            "project": {
                "status": "ready",
                "projectedCount": 1,
                "activeCount": 0,
                "failedCount": 0,
            },
            "recall": {
                "status": "ready",
                "activeEntityCount": 0,
                "topScore": 0,
                "triggerCount": 0,
                "intentions": {
                    "activeCount": 2,
                    "refreshContextCount": 1,
                    "afterConsolidationCount": 1,
                    "pinnedResultCount": 1,
                    "needsRefreshCount": 0,
                    "latestRefreshedAt": "2026-05-13T12:00:00Z",
                },
            },
            "consolidate": {
                "status": "ready",
                "isRunning": False,
                "cycleCount": 0,
                "latestCycle": None,
            },
        }
    )

    assert "# Engram Lifecycle" in rendered
    assert "- Group: `cli_brain`" in rendered
    assert "- Cue: `ready` | coverage 100.0%" in rendered
    assert (
        "- Recall: `ready` | active entities 0 | top score 0 | "
        "triggers 0 | intentions 2 | pinned 1"
    ) in rendered
    assert "- Consolidate: `ready`" in rendered


def test_lifecycle_cli_markdown_includes_latest_cycle_error() -> None:
    rendered = format_lifecycle_summary_markdown(
        {
            "groupId": "cli_brain",
            "consolidate": {
                "status": "attention",
                "isRunning": False,
                "cycleCount": 1,
                "latestCycle": {
                    "id": "cyc_failed",
                    "error": "calibration failed",
                },
            },
        }
    )

    assert "- Consolidate: `attention`" in rendered
    assert "latest `cyc_failed` | error `calibration failed`" in rendered


def test_lifecycle_cli_markdown_includes_latest_phase_error() -> None:
    rendered = format_lifecycle_summary_markdown(
        {
            "groupId": "cli_brain",
            "consolidate": {
                "status": "attention",
                "isRunning": False,
                "cycleCount": 1,
                "latestCycle": {
                    "id": "cyc_phase_error",
                    "status": "completed",
                    "error": None,
                    "phases": [
                        {
                            "phase": "graph_embed",
                            "status": "error",
                            "items_processed": 1,
                            "items_affected": 0,
                            "duration_ms": 5.0,
                            "error": "optional vector index unavailable",
                        }
                    ],
                },
            },
        }
    )

    assert "- Consolidate: `attention`" in rendered
    assert (
        "latest `cyc_phase_error` | error "
        "`graph_embed: optional vector index unavailable`"
    ) in rendered


@pytest.mark.asyncio
async def test_lifecycle_summary_includes_recall_intention_summary() -> None:
    manager = SimpleNamespace(
        get_graph_state=AsyncMock(return_value={"stats": {}, "top_activated": []}),
        list_intentions=AsyncMock(
            return_value=[
                SimpleNamespace(
                    attributes={
                        "trigger_type": "activation",
                        "trigger_text": "regular intention",
                    }
                ),
                SimpleNamespace(
                    attributes={
                        "trigger_type": "refresh_context",
                        "refresh_trigger": "after_consolidation",
                        "trigger_text": "native path",
                        "pinned_result": "cached native context",
                        "last_refreshed": "2026-05-13T12:00:00Z",
                    }
                ),
                SimpleNamespace(
                    attributes={
                        "trigger_type": "refresh_context",
                        "refresh_trigger": "after_consolidation",
                        "trigger_text": "unrefreshed",
                    }
                ),
            ]
        ),
    )

    summary = await build_lifecycle_summary(
        group_id="native_brain",
        manager=manager,
        episode_limit=0,
        cycle_limit=1,
        top_n=0,
    )

    manager.list_intentions.assert_awaited_once_with(
        group_id="native_brain",
        enabled_only=True,
    )
    assert summary["recall"]["status"] == "active"
    assert summary["recall"]["intentions"] == {
        "activeCount": 3,
        "refreshContextCount": 2,
        "afterConsolidationCount": 2,
        "pinnedResultCount": 1,
        "needsRefreshCount": 1,
        "latestRefreshedAt": "2026-05-13T12:00:00Z",
    }


@pytest.mark.asyncio
async def test_lifecycle_summary_uses_shared_consolidation_cycle_contract() -> None:
    cycle = ConsolidationCycle(
        id="cyc_shared_contract",
        group_id="native_brain",
        status="failed",
        error="calibration failed",
        dry_run=True,
        phase_results=[
            PhaseResult(
                phase="calibrate",
                status="error",
                items_processed=2,
                items_affected=0,
                duration_ms=12.0,
                error="no teacher labels",
            )
        ],
    )
    cycle.total_duration_ms = 20.0

    class FakeConsolidationStore:
        async def get_recent_cycles(self, group_id: str, *, limit: int) -> list[ConsolidationCycle]:
            assert group_id == "native_brain"
            assert limit == 1
            return [cycle]

    manager = SimpleNamespace(
        get_graph_state=AsyncMock(return_value={"stats": {}, "top_activated": []}),
    )

    summary = await build_lifecycle_summary(
        group_id="native_brain",
        manager=manager,
        consolidation_engine=SimpleNamespace(is_running=False),
        consolidation_reader=ConsolidationAuditReader(FakeConsolidationStore()),
        episode_limit=0,
        cycle_limit=1,
        top_n=0,
    )

    latest_cycle = summary["consolidate"]["latestCycle"]
    assert summary["consolidate"]["status"] == "attention"
    assert latest_cycle["id"] == "cyc_shared_contract"
    assert latest_cycle["error"] == "calibration failed"
    assert latest_cycle["total_duration_ms"] == 20.0
    assert latest_cycle["phases"] == [
        {
            "phase": "calibrate",
            "status": "error",
            "items_processed": 2,
            "items_affected": 0,
            "duration_ms": 12.0,
            "error": "no teacher labels",
        }
    ]


@pytest.mark.asyncio
async def test_lifecycle_summary_marks_completed_cycle_with_phase_error_attention() -> None:
    cycle = ConsolidationCycle(
        id="cyc_phase_error",
        group_id="native_brain",
        status="completed",
        dry_run=False,
        phase_results=[
            PhaseResult(
                phase="graph_embed",
                status="error",
                items_processed=1,
                items_affected=0,
                duration_ms=5.0,
                error="optional vector index unavailable",
            )
        ],
    )
    cycle.total_duration_ms = 5.0

    class FakeConsolidationStore:
        async def get_recent_cycles(self, group_id: str, *, limit: int) -> list[ConsolidationCycle]:
            assert group_id == "native_brain"
            assert limit == 1
            return [cycle]

    manager = SimpleNamespace(
        get_graph_state=AsyncMock(return_value={"stats": {}, "top_activated": []}),
    )

    summary = await build_lifecycle_summary(
        group_id="native_brain",
        manager=manager,
        consolidation_engine=SimpleNamespace(is_running=False),
        consolidation_reader=ConsolidationAuditReader(FakeConsolidationStore()),
        episode_limit=0,
        cycle_limit=1,
        top_n=0,
    )

    latest_cycle = summary["consolidate"]["latestCycle"]
    assert summary["consolidate"]["status"] == "attention"
    assert latest_cycle["status"] == "completed"
    assert latest_cycle["error"] is None
    assert latest_cycle["phases"][0]["error"] == "optional vector index unavailable"
