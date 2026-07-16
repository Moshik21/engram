"""CLI for hybrid entity-vector index completeness and backfill."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def configure_index_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--group-id",
        default=None,
        help="Graph group id (default: config default_group_id)",
    )
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Engine mode override",
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help="Native Helix data directory",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=200,
        help="Max entities to index in one backfill run (default 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for batch_index_entities (default 32)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Measure coverage only; do not write vectors",
    )
    parser.add_argument(
        "--remeasure",
        action="store_true",
        help="After backfill, re-measure coverage",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--force-local",
        action="store_true",
        help="Open the graph even if a shell appears to be running (unsafe)",
    )


async def run_index_command(args: argparse.Namespace) -> int:
    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    try:
        with require_exclusive_local_access(force=bool(getattr(args, "force_local", False))):
            return await _run_index_command_locked(args)
    except ExclusiveAccessError as exc:
        print(f"index: {exc}", file=sys.stderr)
        return 2


async def _run_index_command_locked(args: argparse.Namespace) -> int:
    from engram.config import EngramConfig
    from engram.storage.bootstrap import (
        close_if_supported,
        create_local_runtime_stores,
        initialize_search_index_for_graph,
    )
    from engram.storage.index_completeness import (
        backfill_missing_entity_vectors,
        list_indexable_entities,
        measure_entity_vector_coverage,
    )
    from engram.storage.resolver import resolve_mode

    requested_mode = args.mode or "auto"
    config = EngramConfig(mode=requested_mode)
    if getattr(args, "helix_data_dir", None):
        config.helix.data_dir = str(args.helix_data_dir.expanduser())
        config.helix.transport = "native"
        if args.mode is None:
            config.mode = "helix"

    group_id = args.group_id or config.default_group_id
    graph_store = None
    search_index = None
    try:
        mode = await resolve_mode(config.mode)
        graph_store, _activation, search_index = create_local_runtime_stores(mode, config)
        await graph_store.initialize()
        await initialize_search_index_for_graph(
            search_index,
            graph_store=graph_store,
            mode=mode,
        )

        before = await measure_entity_vector_coverage(
            graph_store,
            search_index,
            group_id,
        )
        payload: dict[str, Any] = {
            "before": before.to_dict(),
            "backfill": None,
        }
        if not args.dry_run and before.missing_count > 0:
            entities, _, _ = await list_indexable_entities(graph_store, group_id)
            by_id = {str(e.id): e for e in entities}
            backfill = await backfill_missing_entity_vectors(
                graph_store,
                search_index,
                group_id,
                max_entities=int(args.max_entities),
                batch_size=int(args.batch_size),
                dry_run=False,
                remeasure=bool(args.remeasure),
                missing_ids=before.missing_ids,
                entities_by_id=by_id,
            )
            payload["backfill"] = backfill.to_dict()
            if args.remeasure:
                after = await measure_entity_vector_coverage(graph_store, search_index, group_id)
                payload["after"] = after.to_dict()
        elif args.dry_run:
            payload["backfill"] = {
                "dry_run": True,
                "would_index": min(before.missing_count, int(args.max_entities)),
            }

        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            b = payload["before"]
            print(
                f"Entity vector coverage: {b['vector_count']}/{b['indexable_count']} "
                f"({b['coverage']:.1%}) missing={b['missing_count']} group={group_id}"
            )
            if b["skipped_empty_name"] or b["skipped_deleted"]:
                print(
                    f"  skipped empty_name={b['skipped_empty_name']} deleted={b['skipped_deleted']}"
                )
            bf = payload.get("backfill")
            if bf and not bf.get("dry_run"):
                print(
                    f"Backfill: attempted={bf.get('attempted')} "
                    f"indexed={bf.get('indexed')} failed={bf.get('failed')}"
                )
            elif bf and bf.get("dry_run"):
                print(f"Dry-run: would index up to {bf.get('would_index')} entities")
            after = payload.get("after")
            if after:
                print(
                    f"After: {after['vector_count']}/{after['indexable_count']} "
                    f"({after['coverage']:.1%}) missing={after['missing_count']}"
                )
        return 0
    except Exception as exc:
        logger.exception("index command failed")
        print(f"index command failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if search_index is not None:
            await close_if_supported(search_index)
        if graph_store is not None:
            await close_if_supported(graph_store)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure and backfill hybrid entity-vector index completeness",
    )
    configure_index_parser(parser)
    args = parser.parse_args(argv)
    return asyncio.run(run_index_command(args))
