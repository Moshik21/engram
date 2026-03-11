"""CLI runner for one-shot consolidation cycles."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, cast

from engram.config import EngramConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.store import SQLiteConsolidationStore
from engram.extraction.factory import create_extractor
from engram.storage.factory import create_stores
from engram.storage.resolver import resolve_mode

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> None:
    """Run a single consolidation cycle."""
    config = EngramConfig()
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    search_initializer = getattr(search_index, "initialize", None)
    if search_initializer is not None:
        from engram.storage.resolver import EngineMode

        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await cast(Any, search_initializer)(db=graph_store._db)
        else:
            await cast(Any, search_initializer)()

    # Build activation config — use EngramConfig's nested activation
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
        object.__setattr__(cfg, "microglia_scan_entities_per_cycle", args.scan_entities)

    group_id = args.group_id

    # Get graph stats before cycle
    stats = await graph_store.get_stats(group_id)
    print(f"Graph stats: {json.dumps(stats, indent=2)}")

    # Create consolidation store
    store = None
    if hasattr(graph_store, "_db") and hasattr(graph_store._db, "execute"):
        # Lite mode — share the SQLite connection
        store = SQLiteConsolidationStore(":memory:")
        await store.initialize(db=graph_store._db)
    else:
        # Full mode — use a standalone SQLite file for consolidation audit
        import os

        data_dir = os.path.expanduser("~/.engram")
        os.makedirs(data_dir, exist_ok=True)
        consolidation_db = os.path.join(data_dir, "consolidation.db")
        store = SQLiteConsolidationStore(consolidation_db)
        await store.initialize()

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
    cycle = await engine.run_cycle(
        group_id=group_id,
        trigger="cli",
        dry_run=cfg.consolidation_dry_run,
        phase_names=phase_names,
    )

    # Print results
    result = {
        "cycle_id": cycle.id,
        "status": cycle.status,
        "dry_run": cycle.dry_run,
        "profile": args.profile,
        "graph_stats": stats,
        "phases": [
            {
                "phase": pr.phase,
                "status": pr.status,
                "items_processed": pr.items_processed,
                "items_affected": pr.items_affected,
            }
            for pr in cycle.phase_results
        ],
        "total_duration_ms": cycle.total_duration_ms,
    }
    print(json.dumps(result, indent=2))

    # Human summary
    total_processed = sum(pr.items_processed for pr in cycle.phase_results)
    total_affected = sum(pr.items_affected for pr in cycle.phase_results)
    mode_str = "DRY RUN" if cycle.dry_run else "LIVE"
    print(f"\n[{mode_str}] Consolidation complete: "
          f"{total_processed} items processed, {total_affected} affected, "
          f"{cycle.total_duration_ms}ms")

    await graph_store.close()


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
