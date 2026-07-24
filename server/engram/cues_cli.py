"""CLI for the cue backfill sweep: ``engram cues backfill``.

Cue creation is capture-time only, so every episode stored before the cue layer
shipped has no cue row and nothing in the system ever writes one. This is the
operator drain for that historical debt. Dry run by default; ``--apply``
commits, mirroring ``engram backup compact``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CURSOR_STATE_KEY = "cue_backfill_cursors"


def configure_cues_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="cues_command", required=True)

    backfill_p = sub.add_parser(
        "backfill",
        help="Write cue rows for episodes captured before the cue layer (dry run by default)",
    )
    backfill_p.add_argument(
        "--group-id",
        default=None,
        help="Graph group id (default: config default_group_id)",
    )
    backfill_p.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Engine mode override",
    )
    backfill_p.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help="Native Helix data directory",
    )
    backfill_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max cue probes (and therefore writes) this run (default 500)",
    )
    backfill_p.add_argument(
        "--apply",
        action="store_true",
        help="Write the cues. Without this the run reports and changes nothing.",
    )
    backfill_p.add_argument(
        "--restart",
        action="store_true",
        help="Ignore the persisted resume cursor and sweep from the oldest episode",
    )
    backfill_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    backfill_p.add_argument(
        "--force-local",
        action="store_true",
        help="Open the graph even if a shell appears to be running (unsafe)",
    )


def _read_cursor(group_id: str) -> tuple[float, str] | None:
    from engram.hygiene_ops import _read_hygiene_state

    raw = _read_hygiene_state().get(_CURSOR_STATE_KEY)
    if not isinstance(raw, dict):
        return None
    value = raw.get(group_id)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return (float(value[0]), str(value[1]))
    except (TypeError, ValueError):
        return None


def _write_cursor(group_id: str, cursor: tuple[float, str]) -> None:
    from engram.hygiene_ops import _read_hygiene_state, _write_hygiene_state

    state = _read_hygiene_state()
    cursors = state.get(_CURSOR_STATE_KEY)
    cursors = dict(cursors) if isinstance(cursors, dict) else {}
    cursors[group_id] = list(cursor)
    state[_CURSOR_STATE_KEY] = cursors
    _write_hygiene_state(state)


async def run_cues_command(args: argparse.Namespace) -> int:
    from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

    if args.cues_command != "backfill":
        print(f"cues: unknown command {args.cues_command}", file=sys.stderr)
        return 2
    try:
        with require_exclusive_local_access(force=bool(getattr(args, "force_local", False))):
            return await _run_backfill_locked(args)
    except ExclusiveAccessError as exc:
        print(f"cues backfill: {exc}", file=sys.stderr)
        return 2


async def _run_backfill_locked(args: argparse.Namespace) -> int:
    from engram.config import EngramConfig
    from engram.ingestion.cue_backfill import (
        DEFAULT_BACKFILL_LIMIT,
        backfill_missing_episode_cues,
    )
    from engram.storage.bootstrap import open_local_stores

    config = EngramConfig(mode=args.mode or "auto")
    if getattr(args, "helix_data_dir", None):
        config.helix.data_dir = str(args.helix_data_dir.expanduser())
        config.helix.transport = "native"
        if args.mode is None:
            config.mode = "helix"

    activation = config.activation
    group_id = args.group_id or config.default_group_id
    limit = int(args.limit) if args.limit is not None else DEFAULT_BACKFILL_LIMIT
    cursor = None if args.restart else _read_cursor(group_id)

    # Cue derivation reads these. CLI-resolved config is KNOWN to disagree with
    # the launchd shell's (profile drift), and these three change what a cue
    # looks like — print them so the operator can diff against the live values
    # at GET /api/knowledge/runtime before committing a bulk write.
    derivation_config = {
        "consolidation_profile": getattr(activation, "consolidation_profile", None),
        "cue_layer_enabled": bool(getattr(activation, "cue_layer_enabled", False)),
        "emotional_salience_enabled": bool(
            getattr(activation, "emotional_salience_enabled", False)
        ),
        "cue_policy_learning_enabled": bool(
            getattr(activation, "cue_policy_learning_enabled", False)
        ),
    }

    try:
        async with open_local_stores(config, local_runtime=True) as stores:
            result = await backfill_missing_episode_cues(
                stores.graph_store,
                activation,
                group_id,
                limit=limit,
                apply=bool(args.apply),
                cursor=cursor,
            )
    except Exception as exc:
        logger.exception("cues backfill failed")
        print(f"cues backfill failed: {exc}", file=sys.stderr)
        return 1

    if args.apply and result.cursor_next is not None:
        _write_cursor(group_id, result.cursor_next)

    payload: dict[str, Any] = {
        "derivationConfig": derivation_config,
        "cursorIn": list(cursor) if cursor else None,
        **result.to_dict(),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        _print_text(payload, result, derivation_config)
    return 0


def _print_text(payload: dict[str, Any], result: Any, derivation_config: dict[str, Any]) -> None:
    mode = "DRY RUN (nothing written)" if result.dry_run else "APPLY"
    print(f"Cue backfill [{mode}] group={result.group_id}")
    print(
        "  derivation config: "
        + " ".join(f"{k}={v}" for k, v in derivation_config.items())
        + "\n  (verify against GET /api/knowledge/runtime — CLI config can differ from the shell)"
    )
    cursor_in = payload.get("cursorIn")
    print(f"  resume cursor in: {cursor_in if cursor_in else 'none (from oldest episode)'}")
    print(
        f"  scanned={result.scanned} probed={result.probed} "
        f"already_cued={result.already_cued} failed={result.failed}"
    )
    print(
        f"  skipped (capture would skip these too): "
        f"empty_content={result.skipped_empty_content} "
        f"by_policy={result.skipped_by_policy}"
    )
    if result.dry_run:
        print(f"  WOULD WRITE {result.would_write} cues (re-run with --apply to commit)")
    else:
        print(f"  wrote {result.written} cues (state_inherited={result.state_inherited})")
        print(f"  resume cursor out: {payload.get('cursor_next')}")
    print(
        "  remaining after this window: "
        + ("none — sweep complete" if result.complete else "more (re-run to continue)")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Episode cue operator commands")
    configure_cues_parser(parser)
    args = parser.parse_args(argv)
    return asyncio.run(run_cues_command(args))
