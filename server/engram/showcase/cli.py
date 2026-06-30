"""CLI for the public Engram showcase."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from engram.showcase.export import export_showcase_payload
from engram.showcase.resources import prepare_showcase_db
from engram.showcase.runner import (
    format_showcase_run,
    run_showcase_beats,
    showcase_open_instructions,
)
from engram.showcase.seed import default_seed_output, seed_demo_db


def configure_showcase_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="showcase_command", required=True)

    seed_parser = subparsers.add_parser(
        "seed",
        help="Build the lite demo.db used by showcase run/export",
    )
    seed_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output sqlite path (default: engram/data/demo.db)",
    )
    seed_parser.add_argument(
        "--group-id",
        default="showcase",
        help="Graph group id for seeded episodes",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Replay the 3-beat Liam continuity script against bundled demo.db",
    )
    run_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override demo sqlite path",
    )
    run_parser.add_argument(
        "--open",
        action="store_true",
        help="After the script, print local serve/dashboard instructions",
    )
    run_parser.add_argument(
        "--no-open",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    run_parser.add_argument(
        "--port",
        type=int,
        default=8100,
        help="API port referenced by --open instructions",
    )

    export_parser = subparsers.add_parser(
        "export",
        help="Export structured showcase beats for the website theater",
    )
    export_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Override demo sqlite path",
    )
    export_parser.add_argument(
        "--format",
        choices=["json", "markdown", "both"],
        default="both",
    )
    export_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON output path",
    )
    export_parser.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help="Markdown output path",
    )


def run_showcase_command(args: argparse.Namespace) -> int:
    return asyncio.run(_run_showcase_command(args))


async def _run_showcase_command(args: argparse.Namespace) -> int:
    if args.showcase_command == "seed":
        out = args.out or default_seed_output()
        path = await seed_demo_db(out, group_id=args.group_id)
        print(f"Seeded showcase database: {path}")
        return 0

    if args.showcase_command == "run":
        prepared = prepare_showcase_db(db_path=args.db)
        results, runtime_db = await run_showcase_beats(prepared_db_path=prepared)
        print(format_showcase_run(results), end="")
        if args.open and not args.no_open:
            print(showcase_open_instructions(runtime_db, api_port=args.port), end="")
        if any(not item.passed for item in results):
            return 1
        return 0

    if args.showcase_command == "export":
        json_path = args.out
        markdown_path = args.markdown_out
        if args.format in {"json", "both"} and json_path is None:
            json_path = Path("showcase-export.json")
        if args.format in {"markdown", "both"} and markdown_path is None:
            markdown_path = Path("showcase-export.md")
        if args.format == "json":
            markdown_path = None
        if args.format == "markdown":
            json_path = None
        await export_showcase_payload(
            db_path=args.db,
            out_path=json_path,
            markdown_path=markdown_path,
        )
        if json_path is not None:
            print(f"Wrote {json_path}")
        if markdown_path is not None:
            print(f"Wrote {markdown_path}")
        return 0

    print(f"Unknown showcase command: {args.showcase_command}", file=sys.stderr)
    return 2
