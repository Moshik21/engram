"""Cold-brain CLI: exclusive consolidation outside the hot shell.

Usage:
  engram brain run [--tier auto|hot|warm|cold|full] [--profile quiet|…]
  engram brain status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from engram.brain_runtime import (
    BrainStatus,
    exclusive_brain_lock,
    read_brain_status,
    utc_now_iso,
)
from engram.consolidation.phase_registry import CONSOLIDATION_PHASE_TIERS

logger = logging.getLogger(__name__)

# Map tier → phase names for one-shot cycles (mirrors scheduler tiers).
_TIER_PHASES: dict[str, set[str] | None] = {
    "full": None,  # all phases
    "auto": None,  # same as full for one-shot; LaunchAgent may pick
    "hot": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "hot"},
    "warm": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "warm"},
    "cold": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "cold"},
    "mop": {
        "evidence_adjudication",
        "edge_adjudication",
        "prune",
        "triage",
    },
}


def configure_brain_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="brain_command", required=True)

    run_p = sub.add_parser("run", help="Run one cold-brain consolidation cycle")
    run_p.add_argument(
        "--tier",
        choices=sorted(_TIER_PHASES.keys()),
        default="auto",
        help="Phase tier (default: auto = full cycle)",
    )
    run_p.add_argument(
        "--profile",
        choices=["observe", "quiet", "conservative", "standard"],
        default=None,
        help="Consolidation profile (default: env / quiet)",
    )
    run_p.add_argument(
        "--group-id",
        default="default",
        help="Group ID (default: default)",
    )
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        dest="dry_run",
        help="Force dry-run mode",
    )
    run_p.add_argument(
        "--no-dry-run",
        action="store_false",
        dest="dry_run",
        help="Force live mode",
    )
    run_p.add_argument(
        "--pause-shell",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stop the always-on shell while brain holds the graph "
            "(default: true; safer for Helix native exclusive open)"
        ),
    )
    run_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
    )
    run_p.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="Mop budget (tier=mop only; default 1000)",
    )

    status_p = sub.add_parser("status", help="Show last cold-brain run status")
    status_p.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
    )


def _shell_healthy(port: int = 8100) -> bool:
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _api_port() -> int:
    raw = os.environ.get("ENGRAM_API_PORT", "8100")
    try:
        return int(raw)
    except ValueError:
        return 8100


def _pause_shell() -> bool:
    """Stop LaunchAgent/shell if possible. Returns True if we stopped something."""
    if not _shell_healthy(_api_port()):
        return False
    # Prefer engramctl when on PATH
    for cmd in (
        ["engramctl", "stop"],
        [str(Path.home() / "Engram" / "installer" / "engramctl"), "stop"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Paused shell via %s", cmd[0])
                # wait until health fails
                for _ in range(30):
                    if not _shell_healthy(_api_port()):
                        return True
                    time.sleep(0.5)
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    logger.warning("Could not pause shell via engramctl; proceeding carefully")
    return False


def _resume_shell() -> None:
    for cmd in (
        ["engramctl", "start"],
        [str(Path.home() / "Engram" / "installer" / "engramctl"), "start"],
    ):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode == 0:
                logger.info("Resumed shell via %s", cmd[0])
                return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    logger.warning("Could not resume shell via engramctl")


async def _run_mop(args: argparse.Namespace) -> dict[str, Any]:
    """Debt drain via shared hygiene ops (no consolidation phase thrash)."""
    from engram.config import EngramConfig
    from engram.hygiene_ops import execute_hygiene_mop
    from engram.loop_adjustment import effective_activation_config, load_active_adjustment
    from engram.storage.bootstrap import (
        close_if_supported,
        create_local_runtime_stores,
        initialize_search_index_for_graph,
    )
    from engram.storage.resolver import resolve_mode

    config = EngramConfig()
    object.__setattr__(config, "runtime_role", "brain")
    # Avoid embed thrash during mop if model cache is incomplete.
    if not os.environ.get("ENGRAM_EMBEDDING__PROVIDER"):
        os.environ["ENGRAM_EMBEDDING__PROVIDER"] = "noop"

    profile = args.profile or (
        config.activation.consolidation_profile
        if config.activation.consolidation_profile not in {"off", ""}
        else "quiet"
    )
    group_id = args.group_id
    loop_adj = load_active_adjustment(group_id)
    activation_cfg = effective_activation_config(config.activation, loop_adj)
    dry_run = bool(args.dry_run) if args.dry_run is not None else False

    graph_store = activation_store = search_index = None
    try:
        mode = await resolve_mode(config.mode)
        graph_store, activation_store, search_index = create_local_runtime_stores(mode, config)
        await graph_store.initialize()
        await initialize_search_index_for_graph(search_index, graph_store=graph_store, mode=mode)
        report = await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            activation_cfg=activation_cfg,
            group_id=group_id,
            budget=max(1, int(getattr(args, "budget", 1000) or 1000)),
            dry_run=dry_run,
            loop_adj=loop_adj,
        )
        debt_before = report.get("debt") or {}
        debt_after = report.get("debt_after") or {}
        mop = report.get("mop") or {}
        rejected = 0
        for key in ("evidence_drain", "already_exists", "stale"):
            rejected += int((mop.get(key) or {}).get("rejected") or 0)
        return {
            "status": "completed",
            "error": None,
            "cycle_id": None,
            "profile": profile,
            "tier": "mop",
            "path": "hygiene_ops",
            "summary": {
                "total_processed": rejected,
                "total_affected": rejected,
                "deferred_before": debt_before.get("deferred_evidence"),
                "deferred_after": debt_after.get("deferred_evidence"),
                "open_work_after": debt_after.get("open_work"),
                "mop": mop,
            },
            "report": report,
        }
    finally:
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


async def _run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    # Mop is the debt drain path — do not load consolidation/cross-encoder.
    if getattr(args, "tier", None) == "mop":
        return await _run_mop(args)

    from engram.config import EngramConfig
    from engram.consolidation.engine import ConsolidationEngine
    from engram.consolidation.presenter import serialize_cycle_summary
    from engram.extraction.factory import create_extractor
    from engram.graph_manager import GraphManager
    from engram.storage.bootstrap import (
        close_if_supported,
        create_consolidation_store_for_graph,
        initialize_search_index_for_graph,
    )
    from engram.storage.factory import create_stores
    from engram.storage.resolver import EngineMode, resolve_mode

    config = EngramConfig()
    # Brain process role (does not start HTTP serve)
    object.__setattr__(config, "runtime_role", "brain")

    cfg = config.activation
    profile = args.profile or (
        cfg.consolidation_profile if cfg.consolidation_profile not in {"off", ""} else "quiet"
    )
    if profile != cfg.consolidation_profile:
        object.__setattr__(cfg, "consolidation_profile", profile)
        cfg.model_post_init(None)
    if args.dry_run is not None:
        object.__setattr__(cfg, "consolidation_dry_run", args.dry_run)
    object.__setattr__(cfg, "consolidation_enabled", True)
    # Cold brain owns projection drain for quiet installs
    if profile == "quiet":
        object.__setattr__(cfg, "worker_enabled", True)

    mode = await resolve_mode(config.mode)
    graph_store = None
    activation_store = None
    search_index = None
    store = None
    try:
        graph_store, activation_store, search_index = create_stores(mode, config)
        await graph_store.initialize()
        await initialize_search_index_for_graph(
            search_index,
            graph_store=graph_store,
            mode=mode,
        )
        consolidation_sqlite_path = None
        if mode == EngineMode.FULL:
            consolidation_sqlite_path = Path.home() / ".engram" / "consolidation.db"
            consolidation_sqlite_path.parent.mkdir(parents=True, exist_ok=True)

        store = await create_consolidation_store_for_graph(
            config,
            graph_store=graph_store,
            mode=mode,
            sqlite_path=consolidation_sqlite_path,
        )
        extractor = create_extractor(config)
        graph_manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=extractor,
            cfg=cfg,
        )
        engine = ConsolidationEngine(
            graph_store,
            activation_store,
            search_index,
            cfg=cfg,
            consolidation_store=store,
            extractor=extractor,
            graph_manager=graph_manager,
        )
        phase_names = _TIER_PHASES.get(args.tier)
        cycle = await engine.run_cycle(
            group_id=args.group_id,
            trigger="brain-cli",
            dry_run=cfg.consolidation_dry_run,
            phase_names=phase_names,
        )
        summary = serialize_cycle_summary(cycle)
        return {
            "cycle_id": summary.get("id") or getattr(cycle, "id", None),
            "status": cycle.status,
            "error": cycle.error,
            "summary": summary.get("summary") or {},
            "profile": profile,
            "tier": args.tier,
        }
    finally:
        await close_if_supported(store)
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


def run_brain_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "brain_command", None)
    if cmd == "status":
        status = read_brain_status()
        if getattr(args, "format", "text") == "json":
            print(json.dumps(status or {"ok": None, "message": "no runs yet"}, indent=2))
        elif not status:
            print("Brain: no runs yet (status file missing)")
        else:
            print(
                f"Brain: ok={status.get('ok')} tier={status.get('tier')} "
                f"profile={status.get('profile')} finished={status.get('finished_at')} "
                f"duration_s={status.get('duration_s')} paused_shell={status.get('paused_shell')}"
            )
            if status.get("error"):
                print(f"  error: {status['error']}", file=sys.stderr)
        return 0

    if cmd != "run":
        print(f"Unknown brain command: {cmd}", file=sys.stderr)
        return 2

    started = time.time()
    started_at = utc_now_iso()
    paused = False
    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        with exclusive_brain_lock():
            if args.pause_shell:
                paused = _pause_shell()
            try:
                result = asyncio.run(_run_cycle(args))
                if result.get("status") not in {None, "completed"}:
                    error = result.get("error") or result.get("status")
            finally:
                if paused:
                    _resume_shell()
    except RuntimeError as exc:
        error = str(exc)
        if getattr(args, "format", "text") == "json":
            print(json.dumps({"ok": False, "error": error}, indent=2))
        else:
            print(f"Brain: {error}", file=sys.stderr)
        return 1
    except Exception as exc:
        error = str(exc)
        logger.exception("Brain cycle failed")
    finally:
        finished_at = utc_now_iso()
        duration = time.time() - started
        ok = error is None and (result or {}).get("status", "completed") == "completed"
        BrainStatus(
            ok=ok,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=round(duration, 3),
            tier=getattr(args, "tier", "auto"),
            profile=(result or {}).get("profile") or getattr(args, "profile", None) or "quiet",
            paused_shell=paused,
            pid=os.getpid(),
            error=error,
            cycle_id=(result or {}).get("cycle_id"),
            summary=(result or {}).get("summary") or {},
        ).write()

    payload = {
        "ok": error is None,
        "error": error,
        "result": result,
        "paused_shell": paused,
        "duration_s": round(time.time() - started, 3),
    }
    if getattr(args, "format", "text") == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:
        if error:
            print(f"Brain run failed: {error}", file=sys.stderr)
        else:
            summary = (result or {}).get("summary") or {}
            extra = ""
            if summary.get("deferred_before") is not None:
                extra = (
                    f" deferred={summary.get('deferred_before')}→{summary.get('deferred_after')}"
                )
            print(
                f"Brain run ok tier={args.tier} profile={(result or {}).get('profile')} "
                f"processed={summary.get('total_processed')} "
                f"affected={summary.get('total_affected')} "
                f"paused_shell={paused}{extra}"
            )
    return 0 if error is None else 1
