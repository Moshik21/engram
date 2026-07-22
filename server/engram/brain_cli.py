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
# "mop" is absent on purpose: _run_cycle dispatches tier=mop to _run_mop
# before phase selection, so a phase set here would never be consulted.
_TIER_PHASES: dict[str, set[str] | None] = {
    "full": None,  # all phases
    "auto": None,  # same as full for one-shot; LaunchAgent may pick
    "hot": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "hot"},
    "warm": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "warm"},
    "cold": {name for name, tier in CONSOLIDATION_PHASE_TIERS.items() if tier == "cold"},
}


def configure_brain_parser(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="brain_command", required=True)

    run_p = sub.add_parser("run", help="Run one cold-brain consolidation cycle")
    run_p.add_argument(
        "--tier",
        choices=sorted({*_TIER_PHASES, "mop"}),
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
    run_p.add_argument(
        "--child",
        action="store_true",
        # Internal: cycle process spawned by the watchdog parent. The parent
        # owns pause/resume and the status file; the child only runs the
        # cycle under the flock and prints JSON.
        help=argparse.SUPPRESS,
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
    from engram.storage.bootstrap import open_local_stores

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

    # load_activation_snapshot: the shell saved the snapshot at shutdown and
    # was paused before these stores opened, so prune protections see real
    # usage. The brain must NOT save the snapshot back — the shell owns
    # writes; a stale brain save could clobber a newer shell save.
    async with open_local_stores(
        config,
        local_runtime=True,
        load_activation_snapshot=True,
    ) as stores:
        extractor = create_extractor(config)
        graph_manager = GraphManager(
            graph_store=stores.graph_store,
            activation_store=stores.activation_store,
            search_index=stores.search_index,
            extractor=extractor,
            cfg=activation_cfg,
        )
        report = await execute_hygiene_mop(
            graph_store=stores.graph_store,
            activation_store=stores.activation_store,
            search_index=stores.search_index,
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


async def _run_cycle(args: argparse.Namespace) -> dict[str, Any]:
    # Mop is the debt drain path — do not load consolidation/cross-encoder.
    if getattr(args, "tier", None) == "mop":
        return await _run_mop(args)

    from engram.config import EngramConfig
    from engram.consolidation.engine import ConsolidationEngine
    from engram.consolidation.presenter import serialize_cycle_summary
    from engram.extraction.factory import create_extractor
    from engram.graph_manager import GraphManager
    from engram.storage.bootstrap import open_local_stores
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
    consolidation_sqlite_path = None
    if mode == EngineMode.FULL:
        consolidation_sqlite_path = Path.home() / ".engram" / "consolidation.db"
        consolidation_sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    # load_activation_snapshot: warm ACT-R usage so prune protections apply.
    # Load-only — the shell owns snapshot writes (see _run_mop).
    async with open_local_stores(
        config,
        mode=mode,
        with_consolidation=True,
        consolidation_sqlite_path=consolidation_sqlite_path,
        load_activation_snapshot=True,
    ) as stores:
        graph_store = stores.graph_store
        activation_store = stores.activation_store
        search_index = stores.search_index
        store = stores.consolidation_store
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


# ─── Mop watchdog: parent/child process split ────────────────────
#
# A sync-blocked native call can freeze the runner's event loop: the asyncio
# deadline never fires and the SIGTERM handler (installed on the loop) is
# dead — only SIGKILL works (measured 2026-07-21: a mop froze mid-call and
# the shell stayed paused ~45min until manual kill -9). The parent therefore
# runs NO native code and never opens the graph; the cycle runs in a child
# process the parent can always kill, and the shell resume lives in a parent
# finally the child cannot block.

_WATCHDOG_TERM_GRACE_SECONDS = 15.0


def _watchdog_grace_seconds() -> float:
    """Wall grace the parent adds on top of the child's cycle deadline."""
    raw = os.environ.get("ENGRAM_BRAIN_WATCHDOG_GRACE_SECONDS", "120")
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 120.0


def _child_command(args: argparse.Namespace, deadline: float) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "engram",
        "brain",
        "run",
        "--tier",
        str(getattr(args, "tier", "auto")),
        "--group-id",
        str(getattr(args, "group_id", "default")),
        "--budget",
        str(getattr(args, "budget", 1000) or 1000),
        "--deadline-seconds",
        str(deadline),
        "--no-pause-shell",
        "--child",
        "--format",
        "json",
    ]
    if getattr(args, "profile", None):
        cmd += ["--profile", str(args.profile)]
    dry_run = getattr(args, "dry_run", None)
    if dry_run is True:
        cmd.append("--dry-run")
    elif dry_run is False:
        cmd.append("--no-dry-run")
    if getattr(args, "force", False):
        cmd.append("--force")
    return cmd


def _parse_child_payload(out: str) -> dict[str, Any] | None:
    start = out.find("{")
    if start < 0:
        return None
    try:
        payload = json.loads(out[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _spawn_child_and_watch(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str | None, bool]:
    """Run the cycle in a child process under a hard wall deadline.

    Returns (result, error, lock_skipped). The child is guaranteed dead when
    this returns (even on exception unwind): expiry escalates SIGTERM → 15s →
    SIGKILL, which a frozen event loop cannot ignore. A killed child releases
    the brain flock via the OS, so the restarting shell never waits on a
    corpse. Child stderr is inherited (live logs); stdout carries its JSON.
    """
    deadline = _deadline_seconds(args)
    wall = deadline + _watchdog_grace_seconds() if deadline > 0 else None
    proc = subprocess.Popen(
        _child_command(args, deadline),
        stdout=subprocess.PIPE,
        text=True,
    )
    started_mono = time.monotonic()
    killed = False
    out = ""
    try:
        try:
            out, _ = proc.communicate(timeout=wall)
        except subprocess.TimeoutExpired:
            killed = True
            proc.terminate()
            try:
                out, _ = proc.communicate(timeout=_WATCHDOG_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            # Unwinding for any other reason (e.g. parent SIGTERM): never
            # leave a child holding the graph.
            proc.terminate()
            try:
                proc.wait(timeout=_WATCHDOG_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    elapsed = time.monotonic() - started_mono
    if killed:
        error = f"mop watchdog killed a frozen cycle after {elapsed:.0f}s; window skipped"
        logger.error("%s (deadline=%.0fs grace=%.0fs)", error, deadline, _watchdog_grace_seconds())
        return None, error, False
    payload = _parse_child_payload(out or "")
    if payload is None:
        tail = (out or "").strip()[-400:]
        return (
            None,
            f"brain child exited rc={proc.returncode} without parseable result: {tail!r}",
            False,
        )
    if payload.get("skipped") is True:
        return None, str(payload.get("error") or "brain child skipped (lock contention)"), True
    result = payload.get("result")
    error = payload.get("error")
    if error is None and proc.returncode != 0:
        error = f"brain child exited rc={proc.returncode}"
    return (result if isinstance(result, dict) else None), error, False


def _write_brain_status(
    args: argparse.Namespace,
    *,
    result: dict[str, Any] | None,
    error: str | None,
    paused: bool,
    started: float,
    started_mono: float,
    started_at: str,
) -> None:
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


def _emit_run_payload(
    args: argparse.Namespace,
    *,
    result: dict[str, Any] | None,
    error: str | None,
    paused: bool,
    started: float,
) -> int:
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
                f"Brain run ok tier={getattr(args, 'tier', 'auto')} "
                f"profile={(result or {}).get('profile')} "
                f"processed={summary.get('total_processed')} "
                f"affected={summary.get('total_affected')} "
                f"paused_shell={paused}{extra}"
            )
    return 0 if error is None else 1


def _emit_lock_skip(args: argparse.Namespace, error: str) -> int:
    """Lock contention: do NOT clobber the running winner's status file."""
    if getattr(args, "format", "text") == "json":
        print(json.dumps({"ok": False, "error": error, "skipped": True}, indent=2))
    else:
        print(f"Brain: {error}", file=sys.stderr)
    return 1


def _run_watchdog_parent(args: argparse.Namespace) -> int:
    """Pause the shell, run the cycle in a killable child, ALWAYS resume.

    Import audit contract: this path (and everything it calls) must never
    construct stores or run native code — store/extractor imports live only
    inside _run_mop/_run_cycle, which execute in the child process.
    """
    from engram.brain_runtime import brain_lock_is_held, brain_lock_path

    _install_sigterm_handler()
    if brain_lock_is_held():
        # Probe before pausing so a concurrent window keeps its shell state;
        # the child's flock acquisition remains the authoritative gate.
        return _emit_lock_skip(
            args, f"Another brain process holds {brain_lock_path()}; skip or wait"
        )
    started = time.time()
    started_mono = time.monotonic()
    started_at = utc_now_iso()
    paused = False
    resume_needed = False
    skipped = False
    signalled = False
    result: dict[str, Any] | None = None
    error: str | None = None
    try:
        # A leftover marker means a previous window stranded the shell.
        stranded = read_pause_marker() is not None
        paused = _pause_shell()
        resume_needed = paused or stranded
        result, error, skipped = _spawn_child_and_watch(args)
    except BrainPauseError as exc:
        error = f"pause-shell failed: {exc}"
        logger.error("%s", error)
        # If the stop was issued (marker written) and the shell ended up down,
        # bring it back now instead of stranding it until the next window.
        if read_pause_marker() is not None and not _shell_healthy(_api_port()):
            resume_needed = True
    except SystemExit as exc:
        signalled = True
        error = f"terminated by signal (exit {exc.code})"
        logger.warning("%s", error)
    except Exception as exc:
        error = str(exc)
        logger.exception("Brain watchdog failed")
    finally:
        # The child is dead by the time _spawn_child_and_watch returns or
        # unwinds, so brain.lock is free for the restarting shell. Nothing in
        # this block waits on the child.
        if resume_needed:
            _resume_shell()
        if not skipped:
            _write_brain_status(
                args,
                result=result,
                error=error,
                paused=paused,
                started=started,
                started_mono=started_mono,
                started_at=started_at,
            )
    if signalled:
        return 1
    if skipped:
        return _emit_lock_skip(args, error or "brain child skipped (lock contention)")
    return _emit_run_payload(args, result=result, error=error, paused=paused, started=started)


def _run_cycle_inprocess(args: argparse.Namespace, *, child: bool) -> int:
    """Flock + cycle in THIS process (watchdog child, or --no-pause-shell).

    In child mode the parent owns pause/resume and the status file: this
    process must not resume the shell, clear the parent's pause marker, or
    write brain-status.json — its JSON stdout is the contract with the parent.
    """
    _install_sigterm_handler()
    started = time.time()
    started_mono = time.monotonic()
    started_at = utc_now_iso()
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
            if not child:
                # A leftover marker means a previous window stranded the
                # shell; in child mode the marker is the PARENT's.
                resume_needed = read_pause_marker() is not None
            result = asyncio.run(_run_cycle_with_deadline(args))
            if result.get("status") not in {None, "completed"}:
                error = result.get("error") or result.get("status")
    except RuntimeError as exc:
        if "Another brain process holds" in str(exc):
            lock_skipped = True
            return _emit_lock_skip(args, str(exc))
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
        if not lock_skipped and not child:
            if resume_needed:
                _resume_shell()
            _write_brain_status(
                args,
                result=result,
                error=error,
                paused=False,
                started=started,
                started_mono=started_mono,
                started_at=started_at,
            )
    if signalled:
        return 1
    return _emit_run_payload(args, result=result, error=error, paused=False, started=started)


def run_brain_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "brain_command", None)
    if cmd == "status":
        status = read_brain_status()
        try:
            from engram.ops_metrics import brain_status_anomalies, compute_shell_availability

            anomalies = brain_status_anomalies(status)
            availability = compute_shell_availability().to_dict()
        except Exception:
            anomalies = []
            availability = None
        if getattr(args, "format", "text") == "json":
            payload = dict(status or {"ok": None, "message": "no runs yet"})
            payload["anomalies"] = anomalies
            payload["shell_availability_24h"] = availability
            print(json.dumps(payload, indent=2, default=str))
        elif not status:
            print("Brain: no runs yet (status file missing)")
        else:
            print(
                f"Brain: ok={status.get('ok')} tier={status.get('tier')} "
                f"profile={status.get('profile')} finished={status.get('finished_at')} "
                f"duration_s={status.get('duration_s')} paused_shell={status.get('paused_shell')}"
            )
            if availability and availability.get("availability_pct") is not None:
                print(
                    f"  shell availability 24h: {availability['availability_pct']}% "
                    f"(outages={availability['outage_count']}, "
                    f"max={availability['max_outage_seconds']}s)"
                )
            for anomaly in anomalies:
                print(f"  anomaly: {anomaly}", file=sys.stderr)
            if status.get("error"):
                print(f"  error: {status['error']}", file=sys.stderr)
        return 0

    if cmd != "run":
        print(f"Unknown brain command: {cmd}", file=sys.stderr)
        return 2

    force = bool(getattr(args, "force", False))
    child = bool(getattr(args, "child", False))
    from engram.brain_runtime import brain_lock_is_held

    if not child:
        # Stranded-shell recovery: a leftover pause marker with the shell down
        # means a previous window died after stopping the shell — bring it
        # back before anything else (including skip paths). Never while
        # another brain holds the lock: its window legitimately has the shell
        # paused. The watchdog child skips this: the marker is its parent's.
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

    if args.pause_shell and not child:
        # Watchdog parent: pause, spawn the cycle as a killable child, and
        # resume in a finally a frozen child cannot block.
        return _run_watchdog_parent(args)
    return _run_cycle_inprocess(args, child=child)
