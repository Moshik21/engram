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
    clear_pause_marker,
    exclusive_brain_lock,
    on_battery_power,
    read_brain_status,
    read_pause_marker,
    serve_process_alive,
    utc_now_iso,
    write_pause_marker,
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
    run_p.add_argument(
        "--force",
        action="store_true",
        help="Skip the battery-power gate and the no-work preflight",
    )
    run_p.add_argument(
        "--deadline-seconds",
        type=float,
        default=None,
        help=(
            "Hard runtime bound for the cycle (default: "
            "ENGRAM_BRAIN_DEADLINE_SECONDS or 1800; 0 disables)"
        ),
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


class BrainPauseError(RuntimeError):
    """The shell could not be confirmed stopped; the graph must not be opened."""


_ENGRAMCTL_CANDIDATES = (
    ["engramctl"],
    [str(Path.home() / "Engram" / "installer" / "engramctl")],
)


def _pause_shell() -> bool:
    """Confirm exclusive graph access by stopping a healthy shell.

    Returns True when this run stopped the shell, False when the shell is
    already fully down (no serve process). Raises BrainPauseError whenever the
    shell might still hold the graph — the caller must abort, never
    "proceed carefully".
    """
    port = _api_port()
    if not _shell_healthy(port):
        if serve_process_alive():
            raise BrainPauseError(
                "an 'engram serve' process exists but /health is not responding "
                "(starting or stopping); refusing to open the graph concurrently"
            )
        return False
    # Marker first: if we die after stopping the shell, the next run (or
    # engramctl) knows the shell was stranded by a brain window.
    write_pause_marker()
    for base in _ENGRAMCTL_CANDIDATES:
        cmd = [*base, "stop"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            # State unknown; keep the marker so resume is attempted.
            raise BrainPauseError(f"'{cmd[0]} stop' timed out; shell state unknown") from None
        if result.returncode == 0:
            logger.info("Paused shell via %s", cmd[0])
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if not _shell_healthy(port) and not serve_process_alive():
                    return True
                time.sleep(0.5)
            raise BrainPauseError("shell still up 60s after engramctl stop; refusing to proceed")
    # engramctl unavailable while the shell is healthy: nothing was stopped.
    clear_pause_marker()
    raise BrainPauseError(
        "could not stop the running shell (engramctl not found); "
        "aborting instead of double-opening the graph"
    )


def _resume_shell() -> bool:
    """Restart the shell; clear the pause marker only once health confirms."""
    for base in _ENGRAMCTL_CANDIDATES:
        cmd = [*base, "start"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if _shell_healthy(_api_port()):
                    logger.info("Resumed shell via %s", cmd[0])
                    clear_pause_marker()
                    return True
                time.sleep(1.0)
    logger.warning("Could not confirm shell resume; pause marker retained for the next run")
    return False


def _preflight_skip_no_work(args: argparse.Namespace) -> bool:
    """Ask the still-running shell whether a mop window has any actionable work.

    Most windows historically drained nothing while costing minutes of shell
    downtime; skipping them costs nothing — the debt is re-checked in 2h.
    """
    if getattr(args, "tier", None) != "mop" or not args.pause_shell:
        return False
    if getattr(args, "dry_run", None):
        return False
    if not _shell_healthy(_api_port()):
        return False
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{_api_port()}/api/hygiene/debt", timeout=8
        ) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return False
    should = (payload.get("pressure") or {}).get("should_trigger_mop")
    return should is False


def _install_sigterm_handler() -> None:
    """Convert SIGTERM into SystemExit so finally blocks (resume, status) run."""
    import signal

    def _terminate(signum: int, _frame: Any) -> None:
        raise SystemExit(128 + signum)

    try:
        signal.signal(signal.SIGTERM, _terminate)
    except (ValueError, OSError):  # non-main thread / unsupported platform
        pass


def _deadline_seconds(args: argparse.Namespace) -> float:
    override = getattr(args, "deadline_seconds", None)
    if override is not None:
        return float(override)
    raw = os.environ.get("ENGRAM_BRAIN_DEADLINE_SECONDS", "1800")
    try:
        return float(raw)
    except ValueError:
        return 1800.0


async def _run_cycle_with_deadline(args: argparse.Namespace) -> dict[str, Any]:
    deadline = _deadline_seconds(args)
    if deadline <= 0:
        return await _run_cycle(args)
    try:
        return await asyncio.wait_for(_run_cycle(args), timeout=deadline)
    except TimeoutError:
        raise RuntimeError(
            f"brain run exceeded {deadline:.0f}s deadline and was cancelled"
        ) from None


async def _run_mop(args: argparse.Namespace) -> dict[str, Any]:
    """Debt drain + metabolize passes via shared hygiene ops.

    Builds a GraphManager/extractor so the bounded evidence/edge adjudication
    and zero-LLM replay passes run inside the window — the queues these
    service have no other consumer under mop-only scheduling.
    """
    from engram.config import EngramConfig
    from engram.extraction.factory import create_extractor
    from engram.graph_manager import GraphManager
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
        extractor = create_extractor(config)
        graph_manager = GraphManager(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            extractor=extractor,
            cfg=activation_cfg,
        )
        report = await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            activation_cfg=activation_cfg,
            group_id=group_id,
            budget=max(1, int(getattr(args, "budget", 1000) or 1000)),
            dry_run=dry_run,
            loop_adj=loop_adj,
            graph_manager=graph_manager,
            extractor=extractor,
            skip_when_no_work=not bool(getattr(args, "force", False)),
        )
        debt_before = report.get("debt") or {}
        debt_after = report.get("debt_after") or {}
        mop = report.get("mop") or {}
        rejected = 0
        for key in ("evidence_drain", "already_exists", "stale"):
            rejected += int((mop.get(key) or {}).get("rejected") or 0)
        for key in ("evidence_adjudication", "edge_adjudication", "replay"):
            rejected += int((mop.get(key) or {}).get("items_affected") or 0)
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
        # Loop Steward overlay: without this the brain (the only process that
        # runs phases under the hot/cold split) ignored phase_boost/defer and
        # budget adjustments entirely — they were honored only by the
        # monolith-role scheduler.
        try:
            from engram.loop_adjustment import (
                effective_activation_config,
                effective_phase_names,
                load_active_adjustment,
            )

            loop_adj = load_active_adjustment(args.group_id)
            cfg_eff = effective_activation_config(cfg, loop_adj)
            phase_names = effective_phase_names(phase_names, loop_adj)
        except Exception:
            logger.debug("Loop steward overlay skipped", exc_info=True)
            cfg_eff = cfg
        cycle = await engine.run_cycle(
            group_id=args.group_id,
            trigger="brain-cli",
            dry_run=cfg.consolidation_dry_run,
            phase_names=phase_names,
            cfg=cfg_eff,
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

    force = bool(getattr(args, "force", False))
    # Stranded-shell recovery: a leftover pause marker with the shell down
    # means a previous window died after stopping the shell — bring it back
    # before anything else (including skip paths). Never while another brain
    # holds the lock: its window legitimately has the shell paused.
    from engram.brain_runtime import brain_lock_is_held

    if (
        read_pause_marker() is not None
        and not _shell_healthy(_api_port())
        and not brain_lock_is_held()
    ):
        logger.warning("Pause marker found with shell down; resuming stranded shell")
        _resume_shell()
    if not force and args.pause_shell and on_battery_power():
        print(
            "Brain run skipped: on battery power "
            "(sleep would strand the paused shell; --force to override)"
        )
        return 0
    if not force and _preflight_skip_no_work(args):
        print("Brain run skipped: shell reports no actionable hygiene work")
        return 0
    if not args.pause_shell and not force and _shell_healthy(_api_port()):
        print(
            "Brain: shell is running; --no-pause-shell would double-open the graph. "
            "Stop the shell or pass --force.",
            file=sys.stderr,
        )
        return 2

    _install_sigterm_handler()
    started = time.time()
    started_mono = time.monotonic()
    started_at = utc_now_iso()
    paused = False
    resume_needed = False
    lock_skipped = False
    signalled = False
    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        # NOTE: the shell is resumed in the OUTER finally, after the lock
        # context exits — the restarting shell waits on brain.lock, so
        # resuming while still holding it would stall every window.
        with exclusive_brain_lock():
            # A leftover marker means a previous window stranded the shell.
            stranded = read_pause_marker() is not None
            if args.pause_shell:
                paused = _pause_shell()
            resume_needed = paused or stranded
            result = asyncio.run(_run_cycle_with_deadline(args))
            if result.get("status") not in {None, "completed"}:
                error = result.get("error") or result.get("status")
    except BrainPauseError as exc:
        error = f"pause-shell failed: {exc}"
        logger.error("%s", error)
        # If the stop was issued (marker written) and the shell ended up down,
        # bring it back now instead of stranding it until the next window.
        if read_pause_marker() is not None and not _shell_healthy(_api_port()):
            resume_needed = True
    except RuntimeError as exc:
        if "Another brain process holds" in str(exc):
            # Lock contention: do NOT clobber the running winner's status file.
            lock_skipped = True
            error = str(exc)
            if getattr(args, "format", "text") == "json":
                print(json.dumps({"ok": False, "error": error, "skipped": True}, indent=2))
            else:
                print(f"Brain: {error}", file=sys.stderr)
            return 1
        error = str(exc)
        logger.exception("Brain cycle failed")
    except SystemExit as exc:
        signalled = True
        error = f"terminated by signal (exit {exc.code})"
        logger.warning("%s", error)
    except Exception as exc:
        error = str(exc)
        logger.exception("Brain cycle failed")
    finally:
        if not lock_skipped:
            if resume_needed:
                _resume_shell()
            finished_at = utc_now_iso()
            duration = time.time() - started
            duration_mono = time.monotonic() - started_mono
            slept = (duration - duration_mono) > 60.0
            if slept:
                logger.warning(
                    "System slept during brain run: wall=%.0fs monotonic=%.0fs",
                    duration,
                    duration_mono,
                )
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
                duration_monotonic_s=round(duration_mono, 3),
                system_slept=slept,
            ).write()
    if signalled:
        return 1

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
