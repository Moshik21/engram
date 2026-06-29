"""Dogfood cue usefulness via observe capture, MCP recall, and evaluate CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from engram.config import ActivationConfig
from engram.evaluation.brain_loop_report import build_brain_loop_report
from engram.evaluation.store import SQLiteEvaluationStore
from engram.graph_manager import GraphManager
from engram.lifecycle_summary import build_lifecycle_summary
from engram.mcp.server import SessionState
from engram.retrieval.recall_surface import build_mcp_explicit_recall_tool_surface
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex
from tests.conftest import MockExtractor
from tests.test_consolidation_profiles import _quiet_sqlite_recall_config


async def _build_dogfood_manager(
    tmp_path: Path,
) -> tuple[GraphManager, SQLiteGraphStore, ActivationConfig]:
    cfg = _quiet_sqlite_recall_config()
    assert cfg.cue_layer_enabled is True
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
    return manager, graph_store, cfg


@pytest.mark.asyncio
async def test_observe_mcp_recall_makes_cue_usefulness_measurable(
    tmp_path,
    monkeypatch,
) -> None:
    """Observe -> cue, MCP recall surface + middleware, record_recall_evaluation."""
    from engram.mcp import server as mcp_server

    manager, graph_store, cfg = await _build_dogfood_manager(tmp_path)
    eval_store = SQLiteEvaluationStore(graph_store._db_path)
    await eval_store.initialize(db=graph_store._db)

    monkeypatch.setattr(mcp_server, "_manager", manager)
    monkeypatch.setattr(mcp_server, "_session", SessionState(group_id="default"))
    monkeypatch.setattr(mcp_server, "_group_id", "default")
    monkeypatch.setattr(mcp_server, "_activation_cfg", cfg)
    monkeypatch.setattr(mcp_server, "_evaluation_store", eval_store)

    episode_id = await manager.store_episode(
        "The migration to native Helix finished on Tuesday.",
        group_id="default",
        source="mcp_observe",
    )
    cue = await graph_store.get_episode_cue(episode_id, "default")
    assert cue is not None
    assert "helix" in cue.cue_text.lower() or "migration" in cue.cue_text.lower()

    middleware_tools: list[str | None] = []

    async def capture_middleware(
        query: str,
        result: dict,
        *,
        tool_name: str | None = None,
    ) -> None:
        middleware_tools.append(tool_name)

    session = SessionState(group_id="default")
    for _ in range(2):
        await build_mcp_explicit_recall_tool_surface(
            manager,
            group_id="default",
            query="native Helix migration",
            limit=5,
            cfg=cfg,
            session=session,
            recall_middleware=capture_middleware,
        )

    assert middleware_tools == ["recall", "recall"]

    raw_recall = await mcp_server.recall("native Helix migration", limit=5)
    recall_payload = json.loads(raw_recall)
    assert recall_payload.get("operation") == "recall"
    assert recall_payload.get("query") == "native Helix migration"

    raw_label = await mcp_server.record_recall_evaluation(
        recall_triggered=True,
        recall_helped=True,
        recall_needed=True,
        packets_surfaced=1,
        packets_used=1,
        source="cue_dogfood_test",
        query="native Helix migration",
    )
    label_payload = json.loads(raw_label)
    assert label_payload["status"] == "stored"
    assert label_payload["operation"] == "record_recall_evaluation"

    stats = await graph_store.get_stats(group_id="default")
    cue_metrics = stats["cue_metrics"]
    assert cue_metrics["cue_hit_count"] > 0
    assert cue_metrics["cue_surfaced_count"] > 0

    report = build_brain_loop_report(stats, group_id="default")
    cue_usefulness = report["evaluation_signals"]["cue_usefulness"]
    assert cue_usefulness["status"] == "measured"
    assert cue_usefulness["status"] != "needs_feedback"

    lifecycle = await build_lifecycle_summary(
        group_id="default",
        manager=manager,
        graph_store=graph_store,
    )
    assert lifecycle["cue"]["hitCount"] > 0
    assert lifecycle["cue"]["surfacedCount"] > 0
    await graph_store.close()


@pytest.mark.asyncio
async def test_evaluate_cli_reports_measurable_cue_usefulness(tmp_path) -> None:
    """Real engram evaluate --sqlite-path on dogfood brain reports cue usefulness."""
    manager, graph_store, cfg = await _build_dogfood_manager(tmp_path)
    episode_id = await manager.store_episode(
        "The migration to native Helix finished on Tuesday.",
        group_id="default",
        source="mcp_observe",
    )
    assert await graph_store.get_episode_cue(episode_id, "default") is not None

    session = SessionState(group_id="default")

    async def noop_middleware(*_args, **_kwargs) -> None:
        return None

    for _ in range(2):
        await build_mcp_explicit_recall_tool_surface(
            manager,
            group_id="default",
            query="native Helix migration",
            limit=5,
            cfg=cfg,
            session=session,
            recall_middleware=noop_middleware,
        )

    db_path = graph_store._db_path
    await graph_store.close()

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram",
            "evaluate",
            "--mode",
            "lite",
            "--sqlite-path",
            db_path,
            "--format",
            "json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["cue"]["hit_count"] > 0
    assert report["cue"]["surfaced_count"] > 0
    assert report["evaluation_signals"]["cue_usefulness"]["status"] == "measured"

    lifecycle_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram",
            "lifecycle",
            "--mode",
            "lite",
            "--sqlite-path",
            db_path,
            "--format",
            "json",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert lifecycle_proc.returncode == 0, lifecycle_proc.stderr
    lifecycle = json.loads(lifecycle_proc.stdout)
    assert lifecycle["cue"]["hitCount"] > 0
    assert lifecycle["cue"]["surfacedCount"] > 0
