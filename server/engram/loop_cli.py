"""Operator CLI for Loop Steward: status / propose / apply / clear.

Harness subconscious surface — not public MCP.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from engram.loop_adjustment import (
    LoopAdjustment,
    clamp_loop_adjustment,
    clear_active_adjustment,
    hard_caps_from_config,
    propose_from_report,
    run_steward_once,
    save_active_adjustment,
    stamp_applied,
    status_payload,
)

logger = logging.getLogger(__name__)


def configure_loop_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        choices=[
            "status",
            "propose",
            "propose-from-report",
            "steward-once",
            "apply",
            "clear",
        ],
        help=(
            "status|propose|propose-from-report|steward-once|apply|clear — "
            "steward-once: sense→propose→apply if not healthy→optional mop"
        ),
    )
    parser.add_argument("--group-id", default="default")
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="JSON file for apply/propose (or use --json)",
    )
    parser.add_argument(
        "--json",
        dest="json_inline",
        default=None,
        help="Inline JSON string for apply/propose",
    )
    parser.add_argument(
        "--debt-json",
        type=Path,
        default=None,
        help="Hygiene report JSON file for propose-from-report",
    )
    parser.add_argument(
        "--regime",
        default=None,
        help="Optional regime override for propose (debt_heavy, intake_heavy, ...)",
    )
    parser.add_argument(
        "--reason",
        default=None,
        help="Optional reason for propose when building a minimal adjustment",
    )
    parser.add_argument(
        "--ttl-hours",
        type=int,
        default=12,
        help="TTL hours for propose-from-report (default 12)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="For propose-from-report: treat runtime as unreachable",
    )
    parser.add_argument(
        "--continuity-ok",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="For propose-from-report: continuity gate result",
    )
    parser.add_argument(
        "--latency-degraded",
        action="store_true",
        help="For propose-from-report: mark latency_degraded regime",
    )
    parser.add_argument(
        "--adjustment-path",
        type=Path,
        default=None,
        help="Override active adjustment store path (tests/dogfood)",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=None,
        help="Override audit jsonl path",
    )
    parser.add_argument(
        "--skip-continuity-check",
        action="store_true",
        help="Do not require continuity gate even if expected.continuity_must_pass",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
    )
    parser.add_argument(
        "--created-by",
        default=None,
        help="Actor for clear/apply/propose override of created_by",
    )
    parser.add_argument(
        "--mop",
        action="store_true",
        help="For steward-once: run bounded hygiene mop after apply",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=200,
        help="For steward-once --mop: mop budget floor (default 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="For steward-once: propose only; do not write active adjustment",
    )
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help="Storage mode for steward-once mop / live report",
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        default=None,
        help="Helix data dir for steward-once mop / offline report",
    )


def run_loop_command(args: argparse.Namespace) -> int:
    group_id = args.group_id or "default"
    path = args.adjustment_path
    audit = args.audit_path

    try:
        from engram.config import ActivationConfig

        caps = hard_caps_from_config(ActivationConfig())
    except Exception:
        caps = None

    if args.action == "status":
        payload = status_payload(group_id, path=path)
        _emit(payload, args.format)
        return 0

    if args.action == "clear":
        cleared = clear_active_adjustment(
            group_id,
            path=path,
            audit_path=audit,
            cleared_by=args.created_by or "harness",
        )
        payload = {
            "group_id": group_id,
            "cleared": cleared,
            "active": False,
        }
        _emit(payload, args.format)
        return 0

    if args.action in {"propose", "propose-from-report"}:
        return _run_propose(args, group_id=group_id, caps=caps)

    if args.action == "steward-once":
        return _run_steward_once(args, group_id=group_id, caps=caps, path=path, audit=audit)

    # apply
    raw = _load_apply_payload(args)
    if raw is None:
        print("apply requires --file or --json with a LoopAdjustment object", file=sys.stderr)
        return 2
    adj = LoopAdjustment.from_mapping(raw)
    if args.created_by:
        adj.created_by = args.created_by
    if adj.group_id != group_id:
        adj.group_id = group_id

    result = clamp_loop_adjustment(adj, hard_caps=caps)
    if result.rejected:
        print(f"apply rejected: {result.reject_reason}", file=sys.stderr)
        return 1

    clamped = result.adjustment
    if not args.skip_continuity_check and bool(
        (clamped.expected or {}).get("continuity_must_pass")
    ):
        gate = _continuity_gate()
        if gate.get("checked") and not gate.get("pass"):
            print(
                f"apply blocked: continuity_must_pass but check failed ({gate.get('detail')})",
                file=sys.stderr,
            )
            return 1

    stamped = stamp_applied(clamped)
    save_active_adjustment(stamped, path=path, audit_path=audit)
    payload = {
        "group_id": group_id,
        "applied": True,
        "warnings": list(result.warnings),
        "adjustment": stamped.to_dict(),
        "status": status_payload(group_id, path=path),
    }
    _emit(payload, args.format)
    return 0


def _run_steward_once(
    args: argparse.Namespace,
    *,
    group_id: str,
    caps: dict | None,
    path: Path | None,
    audit: Path | None,
) -> int:
    """Sense debt → propose → apply if needed → optional mop."""
    debt = _load_debt_payload(args)
    collected = False
    if debt is None:
        debt = _collect_live_or_offline_debt(args)
        collected = True

    # Only force offline regime when explicitly requested or sense failed entirely.
    if getattr(args, "offline", False):
        server_reachable: bool | None = False
    elif collected and debt is None:
        server_reachable = False
    else:
        server_reachable = None  # debt snapshot drives regime; health is orthogonal
    mop_fn = None
    if args.mop and not args.dry_run:

        def mop_fn(*, budget: int, dry_run: bool = False) -> dict[str, Any]:
            return _run_mop_sync(args, group_id=group_id, budget=budget, dry_run=dry_run)

    payload = run_steward_once(
        debt,
        group_id=group_id,
        created_by=args.created_by or "harness:steward-once",
        dry_run=bool(args.dry_run),
        do_mop=bool(args.mop),
        mop_budget=max(1, int(getattr(args, "budget", None) or 200)),
        server_reachable=server_reachable,
        continuity_ok=args.continuity_ok,
        latency_degraded=bool(args.latency_degraded) or None,
        hard_caps=caps,
        path=path,
        audit_path=audit,
        mop_fn=mop_fn,
    )
    # Prefer compact JSON for harnesses
    if args.format == "text":
        print(
            f"steward-once regime={payload.get('regime')} "
            f"applied={payload.get('applied')} "
            f"healthy_noop={payload.get('healthy_noop')} "
            f"dry_run={payload.get('dry_run')}"
        )
        if payload.get("reason"):
            print(f"  reason: {payload.get('reason')}")
        if payload.get("mop"):
            print(f"  mop: {payload.get('mop')}")
    else:
        print(json.dumps(payload, indent=2, default=str))
    return 0 if payload.get("status") == "ok" else 1


def _server_reachable(url: str = "http://127.0.0.1:8100/health") -> bool:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _collect_live_debt(port: int = 8100) -> dict[str, Any] | None:
    """Fetch the debt scoreboard from the running shell (no local graph open)."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/hygiene/debt", timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict) or "debt" not in payload:
        return None
    return {"debt": payload.get("debt"), "pressure": payload.get("pressure")}


def _collect_live_or_offline_debt(args: argparse.Namespace) -> dict[str, Any] | None:
    """Best-effort debt snapshot for steward-once when no --debt-json provided.

    Live path first: when the shell answers /health, sense debt over HTTP so
    this command NEVER opens the native graph next to a running shell. The
    local open only happens with the shell confirmed down, under the brain
    flock.
    """
    import asyncio

    if _server_reachable():
        return _collect_live_debt()

    async def _offline_report() -> dict[str, Any] | None:

        # Capture report by calling hygiene internals via report action
        # Prefer programmatic collect when possible
        from engram.config import EngramConfig
        from engram.consolidation.hygiene_debt import (
            collect_hygiene_debt_from_store,
            debt_pressure_contribution,
            debt_should_trigger_mop,
        )
        from engram.consolidation.pressure import ConsolidationPressure
        from engram.storage.bootstrap import (
            close_if_supported,
            create_local_runtime_stores,
            initialize_search_index_for_graph,
        )
        from engram.storage.resolver import resolve_mode

        config = EngramConfig(mode=args.mode or "auto")
        if args.helix_data_dir is not None:
            config.helix.data_dir = str(Path(args.helix_data_dir).expanduser())
            config.helix.transport = "native"
            if args.mode is None:
                config.mode = "helix"
        group_id = args.group_id or config.default_group_id
        graph_store = None
        activation_store = None
        search_index = None
        try:
            mode = await resolve_mode(config.mode)
            graph_store, activation_store, search_index = create_local_runtime_stores(mode, config)
            await graph_store.initialize()
            await initialize_search_index_for_graph(
                search_index, graph_store=graph_store, mode=mode
            )
            debt = await collect_hygiene_debt_from_store(graph_store, group_id)
            debt_pressure = debt_pressure_contribution(debt)
            event_pressure = ConsolidationPressure().compute(config.activation)
            should_mop = debt_should_trigger_mop(
                debt,
                pressure_threshold=float(config.activation.consolidation_pressure_threshold),
            )
            return {
                "debt": debt.to_dict(),
                "pressure": {
                    "total": event_pressure + debt_pressure,
                    "hygiene_debt": debt_pressure,
                    "event_bus": event_pressure,
                    "threshold": config.activation.consolidation_pressure_threshold,
                    "should_trigger_mop": should_mop,
                },
            }
        except Exception:
            return None
        finally:
            if search_index is not None:
                await close_if_supported(search_index)
            if activation_store is not None:
                await close_if_supported(activation_store)
            if graph_store is not None:
                await close_if_supported(graph_store)

    try:
        from engram.brain_runtime import ExclusiveAccessError, require_exclusive_local_access

        with require_exclusive_local_access():
            return asyncio.run(_offline_report())
    except ExclusiveAccessError as exc:
        logger.warning("steward-once local debt sense refused: %s", exc)
        return None
    except Exception:
        return None


def _run_mop_sync(
    args: argparse.Namespace,
    *,
    group_id: str,
    budget: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Run hygiene mop via existing CLI entry (sync wrapper)."""
    import asyncio
    from argparse import Namespace

    from engram.hygiene_cli import run_hygiene_command

    mop_args = Namespace(
        action="mop",
        group_id=group_id,
        mode=getattr(args, "mode", None),
        helix_data_dir=getattr(args, "helix_data_dir", None),
        dry_run=dry_run,
        budget=budget,
        format="json",
    )
    # Capture stdout from mop JSON emit
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = asyncio.run(run_hygiene_command(mop_args))
    text = buf.getvalue()
    try:
        # Find last JSON object in output
        start = text.find("{")
        if start >= 0:
            return json.loads(text[start:])
    except json.JSONDecodeError:
        pass
    return {"returncode": rc, "raw": text[-2000:]}


def _run_propose(
    args: argparse.Namespace,
    *,
    group_id: str,
    caps: dict | None,
) -> int:
    """Produce clamped adjustment JSON without writing active set-point."""
    if args.action == "propose-from-report" or args.debt_json is not None:
        debt = _load_debt_payload(args)
        result = propose_from_report(
            debt,
            group_id=group_id,
            created_by=args.created_by or "harness:propose-from-report",
            server_reachable=False if args.offline else None,
            continuity_ok=args.continuity_ok,
            latency_degraded=bool(args.latency_degraded) or None,
            ttl_hours=int(args.ttl_hours or 12),
            hard_caps=caps,
        )
    else:
        raw = _load_apply_payload(args)
        if raw is None:
            # Minimal propose from flags
            if not args.reason and not args.regime:
                print(
                    "propose requires --file/--json or --debt-json / propose-from-report",
                    file=sys.stderr,
                )
                return 2
            raw = {
                "group_id": group_id,
                "regime": args.regime or "debt_heavy",
                "reason": args.reason or "manual propose",
                "ttl_hours": int(args.ttl_hours or 12),
                "created_by": args.created_by or "harness:propose",
                "max_risk": "low",
                "budgets": {
                    "evidence_drain": 2000,
                    "already_exists": 500,
                    "stale_reject": 500,
                    "cue_hygiene": 500,
                    "adjudication_limit": 400,
                },
                "phase_boost": ["evidence_adjudication", "prune"],
                "phase_defer": ["graph_embed", "dream"],
                "intake": {"pattern_junk_reject": True, "auto_extract_min_score": 0.85},
                "expected": {"continuity_must_pass": True},
            }
        adj = LoopAdjustment.from_mapping(raw)
        if args.created_by:
            adj.created_by = args.created_by
        adj.group_id = group_id
        if args.regime:
            adj.regime = args.regime
        if args.reason:
            adj.reason = args.reason
        result = clamp_loop_adjustment(adj, hard_caps=caps)

    if result.rejected:
        print(f"propose rejected: {result.reject_reason}", file=sys.stderr)
        return 1

    # Verify no write: status must not auto-apply
    payload = {
        "group_id": group_id,
        "proposed": True,
        "wrote_active": False,
        "warnings": list(result.warnings),
        "adjustment": result.adjustment.to_dict(),
        "regime": result.adjustment.regime,
    }
    # Prefer pure JSON for piping into apply
    if args.format == "json":
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(
            f"Proposed regime={result.adjustment.regime} "
            f"(not written; pipe --json or save --file then apply)"
        )
        print(json.dumps(result.adjustment.to_dict(), indent=2, default=str))
    return 0


def _load_debt_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    debt_actions = {"propose-from-report", "steward-once"}
    if args.debt_json is not None:
        text = Path(args.debt_json).expanduser().read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            # Accept full hygiene report envelope
            if isinstance(data.get("debt"), dict):
                debt = dict(data["debt"])
                if isinstance(data.get("pressure"), dict):
                    debt["pressure"] = data["pressure"]
                    debt["should_trigger_mop"] = data["pressure"].get("should_trigger_mop")
                return debt
            return data
        return None
    if args.json_inline and args.action in debt_actions:
        data = json.loads(args.json_inline)
        return data if isinstance(data, dict) else None
    if not sys.stdin.isatty() and args.action in debt_actions:
        text = sys.stdin.read().strip()
        if text:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("debt"), dict):
                debt = dict(data["debt"])
                if isinstance(data.get("pressure"), dict):
                    debt["pressure"] = data["pressure"]
                    debt["should_trigger_mop"] = data["pressure"].get("should_trigger_mop")
                return debt
            return data if isinstance(data, dict) else None
    return None


def _load_apply_payload(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.file is not None:
        text = Path(args.file).expanduser().read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict) and "adjustment" in data and isinstance(data["adjustment"], dict):
            return data["adjustment"]
        return data if isinstance(data, dict) else None
    if args.json_inline:
        data = json.loads(args.json_inline)
        if isinstance(data, dict) and "adjustment" in data and isinstance(data["adjustment"], dict):
            return data["adjustment"]
        return data if isinstance(data, dict) else None
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            data = json.loads(text)
            if (
                isinstance(data, dict)
                and "adjustment" in data
                and isinstance(data["adjustment"], dict)
            ):
                return data["adjustment"]
            return data if isinstance(data, dict) else None
    return None


def _continuity_gate() -> dict[str, Any]:
    """Best-effort live continuity check; skip if server unreachable."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://127.0.0.1:8100/health", timeout=2) as resp:
            if resp.status != 200:
                return {"checked": False, "pass": True, "detail": "health_non_200"}
    except Exception:
        return {"checked": False, "pass": True, "detail": "server_unreachable"}

    try:
        import asyncio

        from engram.evaluation.continuity import run_continuity_against_live

        result = asyncio.run(
            run_continuity_against_live(
                server_url="http://127.0.0.1:8100",
                max_recall_ms=4000,
            )
        )
        if isinstance(result, dict):
            ok = bool(result.get("passed"))
        else:
            ok = bool(getattr(result, "passed", False))
        return {"checked": True, "pass": ok, "detail": "against_live"}
    except Exception as exc:
        return {"checked": False, "pass": True, "detail": f"continuity_error:{exc}"}


def _emit(payload: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, indent=2, default=str))
        return
    if not payload.get("active") and payload.get("applied") is None and "cleared" not in payload:
        print(f"Loop adjustment: none (group={payload.get('group_id')})")
        return
    if payload.get("cleared") is not None:
        print(f"Loop adjustment cleared={payload.get('cleared')} group={payload.get('group_id')}")
        return
    if payload.get("applied"):
        adj = payload.get("adjustment") or {}
        print(
            f"Loop adjustment applied regime={adj.get('regime')} "
            f"ttl_hours={adj.get('ttl_hours')} expires_at={adj.get('expires_at')}"
        )
        warnings = payload.get("warnings") or []
        if warnings:
            print(f"  warnings: {', '.join(warnings)}")
        return
    adj = payload.get("adjustment") or {}
    print(
        f"Loop adjustment: active regime={adj.get('regime')} "
        f"remaining_s={payload.get('remaining_ttl_seconds')} "
        f"expires_at={payload.get('expires_at')}"
    )
    print(f"  reason: {adj.get('reason')}")
    print(f"  created_by: {adj.get('created_by')}")
    budgets = adj.get("budgets") or {}
    if budgets:
        print(f"  budgets: {budgets}")
    if adj.get("phase_defer") or adj.get("phase_boost"):
        print(f"  boost={adj.get('phase_boost')} defer={adj.get('phase_defer')}")
