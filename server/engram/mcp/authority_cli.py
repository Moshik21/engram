"""CLI helpers for generating Engram memory-authority payloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.extraction.extractor import EntityExtractor
from engram.graph_manager import GraphManager
from engram.retrieval.memory_authority import build_mcp_memory_authority_surface
from engram.storage.bootstrap import (
    close_if_supported,
    create_local_runtime_stores,
    initialize_search_index_for_graph,
)
from engram.storage.resolver import resolve_mode


def configure_authority_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `engram authority` options to a parser."""
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Engine mode to inspect. Defaults to auto unless --sqlite-path is supplied.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help="SQLite DB path for a lite-mode authority payload. Defaults to config.",
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        help="Native Helix data directory for --mode helix authority payloads.",
    )
    parser.add_argument(
        "--group-id",
        help="Group/brain ID. Defaults to config.",
    )
    parser.add_argument(
        "--project-path",
        help="Current project path for bootstrap/onboarding guidance.",
    )
    parser.add_argument(
        "--user-message",
        help="Current user message used to compute recall and capture routing.",
    )
    parser.add_argument(
        "--file-memory-present",
        action="store_true",
        help="Mark project-local/file memory as visible to the client.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Write the JSON authority payload to this path.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format.",
    )


async def build_authority_payload_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build a claim_authority-compatible payload from local runtime state."""
    requested_mode = args.mode or ("lite" if args.sqlite_path is not None else "auto")
    config = EngramConfig(mode=requested_mode)
    if args.sqlite_path is not None:
        config.sqlite.path = str(args.sqlite_path.expanduser())
    if args.helix_data_dir is not None:
        config.helix.transport = "native"
        config.helix.data_dir = str(args.helix_data_dir.expanduser())

    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_local_runtime_stores(
        mode,
        config,
    )
    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )
    try:
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            EntityExtractor(),
            cfg=config.activation,
            runtime_mode=mode.value,
        )
        return await build_mcp_memory_authority_surface(
            manager,
            group_id=args.group_id or config.default_group_id,
            project_path=args.project_path,
            user_message=args.user_message,
            file_memory_present=bool(args.file_memory_present),
        )
    finally:
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


async def run_authority_command(args: argparse.Namespace) -> None:
    """Print or write an authority payload for parsed CLI arguments."""
    payload = await build_authority_payload_from_args(args)
    if args.out is not None:
        args.out.expanduser().write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.format == "markdown":
        print(format_authority_markdown(payload, output_path=args.out))
        return
    print(json.dumps(payload, indent=2, sort_keys=True))


def format_authority_markdown(
    payload: dict[str, Any],
    *,
    output_path: Path | None = None,
) -> str:
    """Render a compact operator view of a generated authority payload."""
    authority = payload.get("authority") or {}
    onboarding = payload.get("onboarding") or {}
    protocol = payload.get("agent_protocol") or {}
    capture = protocol.get("capture") or {}
    required = protocol.get("required_tools_before_answer") or []
    verification = protocol.get("verification") or {}
    lines = [
        "# Engram Memory Authority",
        "",
        f"- Source of truth: `{authority.get('source_of_truth', 'unknown')}`",
        f"- Onboarding state: `{onboarding.get('state', 'unknown')}`",
        f"- Should bootstrap: `{onboarding.get('should_bootstrap', False)}`",
        f"- File memory substitute: `{protocol.get('file_memory_is_substitute', False)}`",
        f"- Required before-answer tools: `{required}`",
        (
            "- Capture: "
            f"`{capture.get('destination')}` via `{capture.get('tool')}`"
        ),
    ]
    if output_path is not None:
        lines.append(f"- JSON written: `{output_path.expanduser()}`")
    lines.extend(
        [
            "",
            "## Verify",
            "",
            f"- Transcript command: `{verification.get('command')}`",
            f"- Live evidence command: `{verification.get('live_evidence_command')}`",
            "",
            "Use this JSON as `--authority claim-authority.json` for "
            "`engram adoption --template` or transcript validation.",
        ]
    )
    return "\n".join(lines)
