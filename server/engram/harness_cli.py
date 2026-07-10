"""CLI entrypoints for harness adoption helpers."""

from __future__ import annotations

import argparse
import json
import sys

from engram.harness_adoption import write_priming_instruction


def configure_harness_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="harness_command", required=True)

    install_parser = subparsers.add_parser(
        "install-autocapture",
        help="Install Claude Code AutoCapture hooks (transcript capture)",
    )
    install_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )

    write_parser = subparsers.add_parser(
        "write-priming",
        help="Write managed Engram memory protocol instructions for an MCP-only client",
    )
    write_parser.add_argument(
        "--client",
        required=True,
        help="Harness client (cursor, windsurf, grok-build)",
    )
    write_parser.add_argument(
        "--project",
        dest="project_path",
        required=True,
        help="Project directory that should receive priming instructions",
    )
    write_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    score_parser = subparsers.add_parser(
        "scoreboard",
        help="Print harness-as-extractor metrics (client_proposal share, promote rate)",
    )
    score_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: json)",
    )
    score_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset process-local counters after printing",
    )


def run_harness_command(args: argparse.Namespace) -> int:
    if args.harness_command == "install-autocapture":
        from engram.setup import install_hooks

        result = install_hooks()
        payload = {
            "operation": "harness.install_autocapture",
            "status": "ok",
            "scripts": result.get("scripts", []),
            "settings_updated": bool(result.get("settings_updated")),
        }
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            print(f"Installed AutoCapture hooks ({len(payload['scripts'])} scripts)")
        return 0

    if args.harness_command == "scoreboard":
        from engram.extraction.harness_metrics import (
            harness_scoreboard_payload,
            reset_harness_metrics,
        )

        payload = harness_scoreboard_payload()
        if args.format == "json":
            print(json.dumps(payload, indent=2))
        else:
            metrics = payload.get("metrics") or {}
            print("Harness extractor scoreboard")
            print(f"  promote_rate: {metrics.get('promote_rate')}")
            print(f"  client_proposal_share: {metrics.get('client_proposal_share')}")
            print(f"  remember_with_proposals: {metrics.get('remember_with_proposals')}")
            print(
                f"  external_extractor_skipped: {metrics.get('external_extractor_skipped')}"
            )
            print(f"  north_star: {payload.get('north_star')}")
        if getattr(args, "reset", False):
            reset_harness_metrics()
        return 0

    if args.harness_command != "write-priming":
        print(f"Unknown harness command: {args.harness_command}", file=sys.stderr)
        return 2
    try:
        payload = write_priming_instruction(
            client=args.client,
            project_path=args.project_path,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"Wrote Engram priming instructions: {payload['path']}")
    return 0
