"""Command helpers for local brain-loop lifecycle summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from engram.config import EngramConfig
from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.extraction.extractor import EntityExtractor
from engram.graph_manager import GraphManager
from engram.lifecycle_summary import build_lifecycle_summary
from engram.storage.bootstrap import initialize_search_index_for_graph, initialize_store_for_graph
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.resolver import EngineMode, resolve_mode
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


def configure_lifecycle_parser(parser: argparse.ArgumentParser) -> None:
    """Attach lifecycle summary options to a parser."""
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Engine mode to inspect. Defaults to auto unless --sqlite-path is supplied.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help="SQLite DB path for a lite-mode lifecycle snapshot. Defaults to config.",
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        help="Native Helix data directory for --mode helix lifecycle snapshots.",
    )
    parser.add_argument(
        "--group-id",
        help="Group/brain ID. Defaults to config.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Recent episodes to include.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Recent consolidation cycles to include.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Top activated entities to include.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )


async def build_lifecycle_summary_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build the shared lifecycle summary from parsed CLI arguments."""
    requested_mode = args.mode or ("lite" if args.sqlite_path is not None else "auto")
    config = EngramConfig(mode=requested_mode)
    return await build_lifecycle_summary_for_config(
        config,
        sqlite_path=args.sqlite_path,
        helix_data_dir=args.helix_data_dir,
        group_id=args.group_id,
        episode_limit=max(0, args.episodes),
        cycle_limit=max(1, args.cycles),
        top_n=max(0, args.top_n),
    )


async def build_lifecycle_summary_for_config(
    config: EngramConfig,
    *,
    sqlite_path: Path | None = None,
    helix_data_dir: Path | None = None,
    group_id: str | None = None,
    episode_limit: int = 5,
    cycle_limit: int = 10,
    top_n: int = 10,
) -> dict[str, Any]:
    """Build the shared lifecycle summary for the configured local brain."""
    if sqlite_path is not None:
        config.sqlite.path = str(sqlite_path.expanduser())
    if helix_data_dir is not None:
        config.helix.transport = "native"
        config.helix.data_dir = str(helix_data_dir.expanduser())

    resolved_group_id = group_id or config.default_group_id
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = _create_lifecycle_stores(mode, config)
    consolidation_store: Any | None = None

    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )
    try:
        consolidation_store = await _create_consolidation_store(mode, config, graph_store)
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            EntityExtractor(),
            cfg=config.activation,
            runtime_mode=mode.value,
        )
        consolidation_engine = SimpleNamespace(is_running=False)
        return await build_lifecycle_summary(
            group_id=resolved_group_id,
            manager=manager,
            graph_store=graph_store,
            consolidation_engine=consolidation_engine,
            consolidation_reader=ConsolidationAuditReader(consolidation_store),
            activation_config=config.activation,
            top_n=top_n,
            episode_limit=episode_limit,
            cycle_limit=cycle_limit,
        )
    finally:
        await _maybe_close(consolidation_store)
        await _maybe_close(search_index)
        await _maybe_close(graph_store)


def _create_lifecycle_stores(
    mode: EngineMode,
    config: EngramConfig,
) -> tuple[Any, Any, Any]:
    """Create stores for a local lifecycle snapshot."""
    if mode == EngineMode.LITE:
        db_path = str(config.get_sqlite_path())
        return (
            SQLiteGraphStore(db_path),
            MemoryActivationStore(cfg=config.activation),
            FTS5SearchIndex(db_path),
        )

    from engram.storage.factory import create_stores

    return create_stores(mode, config)


async def _create_consolidation_store(
    mode: EngineMode,
    config: EngramConfig,
    graph_store: Any,
) -> Any:
    """Create the consolidation audit store that matches runtime startup."""
    if mode == EngineMode.HELIX:
        from engram.storage.helix.consolidation import HelixConsolidationStore

        store = HelixConsolidationStore(
            config.helix,
            client=getattr(graph_store, "_helix_client", None),
        )
        await store.initialize()
        return store

    if config.postgres.dsn:
        from engram.storage.postgres.consolidation import PostgresConsolidationStore

        store = PostgresConsolidationStore(
            config.postgres.dsn,
            min_pool_size=config.postgres.min_pool_size,
            max_pool_size=config.postgres.max_pool_size,
        )
        await store.initialize()
        return store

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(str(config.get_sqlite_path()))
    await initialize_store_for_graph(store, graph_store=graph_store, mode=mode)
    return store


async def _maybe_close(resource: Any) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


async def run_lifecycle_command(args: argparse.Namespace) -> None:
    """Print a lifecycle summary for parsed CLI arguments."""
    summary = await build_lifecycle_summary_from_args(args)
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(format_lifecycle_summary_markdown(summary))


def format_lifecycle_summary_markdown(summary: dict[str, Any]) -> str:
    """Render a compact Markdown lifecycle snapshot."""
    totals = summary.get("totals") or {}
    capture = summary.get("capture") or {}
    cue = summary.get("cue") or {}
    project = summary.get("project") or {}
    recall = summary.get("recall") or {}
    intentions = recall.get("intentions") or {}
    consolidate = summary.get("consolidate") or {}
    latest_episode = capture.get("latestEpisode") or {}
    latest_cycle = consolidate.get("latestCycle") or {}
    latest_cycle_issue = cycle_issue_text(latest_cycle)
    latest_cycle_error_text = f" | error `{latest_cycle_issue}`" if latest_cycle_issue else ""

    loop = " -> ".join(str(stage).title() for stage in summary.get("loop") or [])
    lines = [
        "# Engram Lifecycle",
        "",
        f"- Group: `{summary.get('groupId', 'default')}`",
        f"- Loop: {loop or 'Capture -> Cue -> Project -> Recall -> Consolidate'}",
        (
            "- Totals: "
            f"{totals.get('episodes', 0)} episodes, "
            f"{totals.get('cues', 0)} cues, "
            f"{totals.get('projected', 0)} projected, "
            f"{totals.get('cycles', 0)} cycles, "
            f"{totals.get('entities', 0)} entities"
        ),
        "",
        "## Stages",
        "",
        (
            f"- Capture: `{capture.get('status', 'unknown')}` | "
            f"episodes {capture.get('episodeCount', 0)} | "
            f"active {capture.get('activeCount', 0)} | "
            f"latest `{latest_episode.get('episodeId', 'none')}`"
        ),
        (
            f"- Cue: `{cue.get('status', 'unknown')}` | "
            f"coverage {_percent(cue.get('coverage', 0.0))} | "
            f"used {cue.get('usedCount', 0)} | "
            f"without cues {cue.get('episodesWithoutCues', 0)}"
        ),
        (
            f"- Project: `{project.get('status', 'unknown')}` | "
            f"projected {project.get('projectedCount', 0)} | "
            f"active {project.get('activeCount', 0)} | "
            f"failed {project.get('failedCount', 0)}"
        ),
        (
            f"- Recall: `{recall.get('status', 'unknown')}` | "
            f"active entities {recall.get('activeEntityCount', 0)} | "
            f"top score {recall.get('topScore', 0)} | "
            f"triggers {recall.get('triggerCount', 0)} | "
            f"intentions {intentions.get('activeCount', 0)} | "
            f"pinned {intentions.get('pinnedResultCount', 0)}"
        ),
        (
            f"- Consolidate: `{consolidate.get('status', 'unknown')}` | "
            f"running {bool(consolidate.get('isRunning', False))} | "
            f"cycles {consolidate.get('cycleCount', 0)} | "
            f"latest `{latest_cycle.get('id', 'none')}`{latest_cycle_error_text}"
        ),
    ]
    return "\n".join(lines).strip() + "\n"


def cycle_issue_text(cycle: dict[str, Any]) -> str | None:
    error = cycle.get("error")
    if isinstance(error, str) and error.strip():
        return error

    phase_issue = cycle.get("phase_issue")
    if isinstance(phase_issue, str) and phase_issue.strip():
        return phase_issue

    phases = cycle.get("phases") or []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        phase_error = phase.get("error")
        if phase.get("status") != "error" and not (
            isinstance(phase_error, str) and phase_error.strip()
        ):
            continue
        phase_name = phase.get("phase")
        phase_label = phase_name if isinstance(phase_name, str) and phase_name else "phase"
        if isinstance(phase_error, str) and phase_error.strip():
            return f"{phase_label}: {phase_error}"
        return f"{phase_label}: phase error"

    return None


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"
