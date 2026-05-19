"""CLI runner for one-shot consolidation cycles."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.presenter import (
    cycle_phase_issue_text,
    serialize_cycle_summary,
)
from engram.extraction.factory import create_extractor
from engram.models.consolidation import ConsolidationCycle
from engram.storage.bootstrap import (
    close_if_supported,
    create_consolidation_store_for_graph,
    initialize_search_index_for_graph,
)
from engram.storage.factory import create_stores
from engram.storage.resolver import EngineMode, resolve_mode

logger = logging.getLogger(__name__)


def _print_cycle_result(
    cycle: ConsolidationCycle,
    *,
    profile: str,
    graph_stats: dict[str, Any],
) -> None:
    result = serialize_cycle_summary(cycle)
    result["cycle_id"] = result.pop("id")
    result["profile"] = profile
    result["graph_stats"] = graph_stats
    print(json.dumps(result, indent=2))

    summary = result["summary"]
    mode_str = "DRY RUN" if cycle.dry_run else "LIVE"
    phase_issue = cycle_phase_issue_text(cycle)
    summary_status = (
        "completed with warnings"
        if cycle.status == "completed" and phase_issue
        else "complete"
        if cycle.status == "completed"
        else cycle.status
    )
    print(
        f"\n[{mode_str}] Consolidation {summary_status}: "
        f"{summary['total_processed']} items processed, {summary['total_affected']} affected, "
        f"{cycle.total_duration_ms}ms"
    )

    if cycle.status == "completed" and phase_issue:
        print(f"Consolidation warning: {phase_issue}", file=sys.stderr)
        return

    if cycle.status != "completed":
        detail = f": {cycle.error}" if cycle.error else ""
        print(f"Consolidation {cycle.status}{detail}", file=sys.stderr)
        raise SystemExit(1)


async def run(args: argparse.Namespace) -> None:
    """Run a single consolidation cycle."""
    config = EngramConfig()
    mode = await resolve_mode(config.mode)
    graph_store = None
    activation_store = None
    search_index = None
    store = None

    try:
        graph_store, activation_store, search_index = create_stores(mode, config)

        await graph_store.initialize()
        await initialize_search_index_for_graph(
            search_index,
            graph_store=graph_store,
            mode=mode,
        )

        # Build activation config - use EngramConfig's nested activation
        # so env vars like ENGRAM_ACTIVATION__MICROGLIA_SCAN_EDGES_PER_CYCLE work.
        # CLI --profile overrides the env-derived profile.
        cfg = config.activation
        if args.profile != cfg.consolidation_profile:
            object.__setattr__(cfg, "consolidation_profile", args.profile)
            cfg.model_post_init(None)

        # CLI overrides
        if args.dry_run is not None:
            object.__setattr__(cfg, "consolidation_dry_run", args.dry_run)
        # Ensure enabled regardless of profile (user explicitly ran CLI)
        object.__setattr__(cfg, "consolidation_enabled", True)

        # Scan limit overrides (for one-time full scans)
        if args.scan_edges is not None:
            object.__setattr__(cfg, "microglia_scan_edges_per_cycle", args.scan_edges)
        if args.scan_entities is not None:
            object.__setattr__(
                cfg,
                "microglia_scan_entities_per_cycle",
                args.scan_entities,
            )

        group_id = args.group_id

        # Get graph stats before cycle
        stats = await graph_store.get_stats(group_id)
        print(f"Graph stats: {json.dumps(stats, indent=2)}")

        consolidation_sqlite_path = None
        if mode == EngineMode.FULL:
            consolidation_sqlite_path = Path.home() / ".engram" / "consolidation.db"
            consolidation_sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        store = await create_consolidation_store_for_graph(
            config,
            graph_store=graph_store,
            mode=mode,
            sqlite_path=consolidation_sqlite_path,
        )

        extractor = create_extractor(config)
        engine = ConsolidationEngine(
            graph_store,
            activation_store,
            search_index,
            cfg=cfg,
            consolidation_store=store,
            extractor=extractor,
        )

        phase_names = set(args.phases) if args.phases else None
        try:
            cycle = await engine.run_cycle(
                group_id=group_id,
                trigger="cli",
                dry_run=cfg.consolidation_dry_run,
                phase_names=phase_names,
            )
        except ValueError as exc:
            print(f"Consolidation failed: {exc}", file=sys.stderr)
            raise SystemExit(2) from None

        _print_cycle_result(
            cycle,
            profile=args.profile,
            graph_stats=stats,
        )
    finally:
        await close_if_supported(store)
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


def main() -> None:
    """Entry point for consolidation CLI."""
    parser = argparse.ArgumentParser(
        description="Run a one-shot memory consolidation cycle",
    )
    parser.add_argument(
        "--profile",
        choices=["off", "observe", "conservative", "standard"],
        default="observe",
        help="Consolidation profile preset (default: observe)",
    )
    parser.add_argument(
        "--group-id",
        default="default",
        help="Group ID to consolidate (default: default)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        dest="dry_run",
        help="Force dry-run mode",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Force live mode (actually apply changes)",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="Run only specific phases (e.g. --phases microglia dream)",
    )
    parser.add_argument(
        "--scan-edges",
        type=int,
        default=None,
        help="Override microglia_scan_edges_per_cycle (for one-time full scans)",
    )
    parser.add_argument(
        "--scan-entities",
        type=int,
        default=None,
        help="Override microglia_scan_entities_per_cycle (for one-time full scans)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(run(args))
