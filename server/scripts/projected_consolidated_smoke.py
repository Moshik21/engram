"""Run the deterministic lite-mode Project + Consolidate smoke.

Examples:
    uv run python scripts/projected_consolidated_smoke.py
    uv run python scripts/projected_consolidated_smoke.py --format json
    uv run python scripts/projected_consolidated_smoke.py \
      --sqlite-path /tmp/engram-smoke.db --replace
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from engram.evaluation.smoke import (
    format_smoke_report,
    run_projected_consolidated_smoke_for_args,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test that Capture -> Cue -> Project -> Recall -> Consolidate "
            "can produce a measured brain-loop report in lite mode."
        )
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help=(
            "SQLite DB path to use. Defaults to a temporary disposable DB. "
            "Existing paths require --replace."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Delete the supplied --sqlite-path before running.",
    )
    parser.add_argument(
        "--group-id",
        default="default",
        help="Group/brain ID for the smoke report.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format for the final report.",
    )
    return parser


async def _run(args: argparse.Namespace) -> None:
    report = await run_projected_consolidated_smoke_for_args(
        sqlite_path=args.sqlite_path,
        replace=args.replace,
        group_id=args.group_id,
    )
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(format_smoke_report(report), end="")


if __name__ == "__main__":
    asyncio.run(_run(_parser().parse_args()))
