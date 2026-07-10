"""CLI for captain preference export/import and identity-core protection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from engram.identity.captain_export import (
    DEFAULT_CAPTAIN_PATH,
    read_captain_import_payload,
    write_captain_file,
)


def configure_captain_parser(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="captain_command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="Export identity_core entities to ~/.engram/captain.md",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAPTAIN_PATH,
        help="Captain markdown path (default: ~/.engram/captain.md)",
    )
    export_parser.add_argument(
        "--group-id",
        default=None,
        help="Graph group id (default: ENGRAM_GROUP_ID or 'default')",
    )

    import_parser = subparsers.add_parser(
        "import",
        help="Parse captain markdown into remember-ready items (no auto-write)",
    )
    import_parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CAPTAIN_PATH,
        help="Captain markdown path (default: ~/.engram/captain.md)",
    )

    sync_parser = subparsers.add_parser(
        "sync",
        help="Export identity_core entities to the captain file",
    )
    sync_parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CAPTAIN_PATH,
    )
    sync_parser.add_argument("--group-id", default=None)

    protect_parser = subparsers.add_parser(
        "protect",
        help="Mark named entities as identity_core (protect from prune/merge)",
    )
    protect_parser.add_argument(
        "names",
        nargs="+",
        help="Entity names to mark as identity_core",
    )
    protect_parser.add_argument("--group-id", default=None)
    protect_parser.add_argument(
        "--unprotect",
        action="store_true",
        help="Remove identity_core protection instead of adding it",
    )


def run_captain_command(args: argparse.Namespace) -> int:
    return asyncio.run(_run_captain_command(args))


async def _run_captain_command(args: argparse.Namespace) -> int:
    if args.captain_command == "import":
        try:
            payload = read_captain_import_payload(path=args.input)
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2))
        return 0

    from engram.config import EngramConfig
    from engram.graph_manager import GraphManager
    from engram.storage.bootstrap import (
        close_if_supported,
        create_local_runtime_stores,
        initialize_search_index_for_graph,
        resolve_mode,
    )

    group_id = args.group_id or os.environ.get("ENGRAM_GROUP_ID", "default")
    config = EngramConfig(mode="auto")
    mode = await resolve_mode(config.mode)
    graph_store, activation_store, search_index = create_local_runtime_stores(mode, config)
    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )
    try:
        if args.captain_command == "protect":
            manager = GraphManager(
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=search_index,
                extractor=None,
                cfg=config.activation,
            )
            results = []
            for name in args.names:
                result = await manager.mark_identity_core(
                    name,
                    identity_core=not args.unprotect,
                    group_id=group_id,
                )
                results.append(result)
            print(json.dumps({"status": "ok", "results": results}, indent=2))
            return 0 if all(r.get("status") == "updated" for r in results) else 1

        if not hasattr(graph_store, "get_identity_core_entities"):
            print("Active graph backend does not support identity_core export", file=sys.stderr)
            return 2
        entities = await graph_store.get_identity_core_entities(group_id)
        output_path = getattr(args, "output", DEFAULT_CAPTAIN_PATH)
        payload = write_captain_file(entities, path=output_path)
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)
