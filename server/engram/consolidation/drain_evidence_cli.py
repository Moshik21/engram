"""CLI for deferred evidence audit and junk drain."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from engram.config import EngramConfig
from engram.consolidation.evidence_drain import (
    audit_deferred_evidence,
    load_deferred_evidence,
    reject_junk_evidence,
)
from engram.storage.bootstrap import close_if_supported, initialize_search_index_for_graph
from engram.storage.factory import create_stores
from engram.storage.resolver import resolve_mode

logger = logging.getLogger(__name__)


def _apply_helix_data_dir(config: EngramConfig, helix_data_dir: str | None) -> None:
    if not helix_data_dir:
        return
    helix_cfg = config.helix.model_copy(update={"data_dir": helix_data_dir, "transport": "native"})
    object.__setattr__(config, "helix", helix_cfg)


def _print_audit(summary: Any) -> None:
    payload = {
        "total": summary.total,
        "keep": summary.keep,
        "reject_junk": summary.reject_junk,
        "by_reason": summary.by_reason,
        "samples": summary.samples,
    }
    print(json.dumps(payload, indent=2))


async def run(args: argparse.Namespace) -> None:
    config = EngramConfig()
    _apply_helix_data_dir(config, args.helix_data_dir)

    mode = await resolve_mode(config.mode)
    graph_store = None
    activation_store = None
    search_index = None

    try:
        graph_store, activation_store, search_index = create_stores(mode, config)
        await graph_store.initialize()
        await initialize_search_index_for_graph(
            search_index,
            graph_store=graph_store,
            mode=mode,
        )

        rows = await load_deferred_evidence(graph_store, args.group_id)
        if args.mode == "audit":
            summary = audit_deferred_evidence(rows)
            _print_audit(summary)
            return

        if args.mode == "reject-junk":
            if not args.yes and not args.dry_run:
                summary = audit_deferred_evidence(rows)
                print(
                    f"About to reject {summary.reject_junk} / {summary.total} deferred rows "
                    f"(keeping {summary.keep}). Re-run with --yes to apply.",
                    file=sys.stderr,
                )
                _print_audit(summary)
                raise SystemExit(2)

            result = await reject_junk_evidence(
                graph_store,
                group_id=args.group_id,
                rows=rows,
                dry_run=args.dry_run,
                batch_size=args.batch_size,
            )
            print(json.dumps(result, indent=2))
            if not args.dry_run:
                metrics_loader = getattr(graph_store, "get_open_work_metrics", None)
                if callable(metrics_loader):
                    metrics = await metrics_loader(args.group_id)
                    print(
                        json.dumps(
                            {"post_drain_open_work": metrics.get("open_work_count")},
                            indent=2,
                        ),
                    )
            return

        raise SystemExit(f"Unknown mode: {args.mode}")
    finally:
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit or drain deferred evidence backlog (reject obvious junk)",
    )
    parser.add_argument(
        "--mode",
        choices=["audit", "reject-junk"],
        default="audit",
        help="audit: preview only; reject-junk: reject classified junk rows",
    )
    parser.add_argument("--group-id", default="default", help="Graph group id")
    parser.add_argument(
        "--helix-data-dir",
        default=None,
        help="Native Helix data dir (sets transport=native for this run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For reject-junk: classify and count without writing",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="For reject-junk: apply rejections without interactive confirm",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Progress log interval while rejecting",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
