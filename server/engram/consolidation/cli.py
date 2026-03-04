"""CLI runner for one-shot consolidation cycles."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from engram.config import ActivationConfig, EngramConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.store import SQLiteConsolidationStore
from engram.extraction.extractor import EntityExtractor
from engram.storage.factory import create_stores
from engram.storage.resolver import resolve_mode

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> None:
    """Run a single consolidation cycle."""
    config = EngramConfig()
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_stores(mode, config)

    await graph_store.initialize()
    if hasattr(search_index, "initialize"):
        from engram.storage.resolver import EngineMode

        if mode == EngineMode.LITE and hasattr(graph_store, "_db"):
            await search_index.initialize(db=graph_store._db)
        else:
            await search_index.initialize()

    # Build activation config with profile
    cfg = ActivationConfig(consolidation_profile=args.profile)

    # CLI overrides
    if args.dry_run is not None:
        object.__setattr__(cfg, "consolidation_dry_run", args.dry_run)
    # Ensure enabled regardless of profile (user explicitly ran CLI)
    object.__setattr__(cfg, "consolidation_enabled", True)

    group_id = args.group_id

    # Get graph stats before cycle
    stats = await graph_store.get_stats(group_id)
    print(f"Graph stats: {json.dumps(stats, indent=2)}")

    # Create consolidation store
    store = None
    if hasattr(graph_store, "_db"):
        store = SQLiteConsolidationStore(":memory:")
        await store.initialize(db=graph_store._db)

    extractor = EntityExtractor()
    engine = ConsolidationEngine(
        graph_store,
        activation_store,
        search_index,
        cfg=cfg,
        consolidation_store=store,
        extractor=extractor,
    )

    cycle = await engine.run_cycle(
        group_id=group_id,
        trigger="cli",
        dry_run=cfg.consolidation_dry_run,
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

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(run(args))
