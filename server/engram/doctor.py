"""Operator diagnostics for local Engram installs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib import error, request

from engram.config import EngramConfig
from engram.evaluation.brain_loop_report import (
    EVALUATION_SIGNAL_ORDER,
    unmeasured_evaluation_signals,
)
from engram.evaluation.smoke import run_projected_consolidated_smoke_for_args
from engram.lifecycle_cli import build_lifecycle_summary_for_config, cycle_issue_text
from engram.storage.resolver import EngineMode, resolve_mode

CHECK_ORDER = (
    "config",
    "sqlite",
    "mode",
    "lifecycle_snapshot",
    "server",
    "mcp",
    "brain_loop_smoke",
    "hooks",
    "promotion_window",
    "mcp_surface",
    "continuity_smoke",
)


def configure_doctor_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `engram doctor` options to a parser."""
    parser.description = (
        "Run local diagnostics, lifecycle snapshot, and brain-loop smoke with "
        "evaluation-signal readiness."
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        help="Override configured engine mode for diagnostics.",
    )
    parser.add_argument(
        "--group-id",
        help="Group/brain ID for the brain-loop smoke. Defaults to config.",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help=(
            "Optional SQLite path for the brain-loop smoke. Defaults to a disposable "
            "temporary DB; existing paths require --replace."
        ),
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        help=(
            "Native Helix data directory for --mode helix lifecycle snapshots. "
            "The brain-loop smoke still uses disposable storage."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing --sqlite-path smoke DB.",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help=(
            "Skip the Capture -> Cue -> Project -> Recall -> Consolidate smoke "
            "and evaluation-signal readiness summary."
        ),
    )
    parser.add_argument(
        "--no-lifecycle",
        action="store_true",
        help="Skip the local Capture -> Cue -> Project -> Recall -> Consolidate snapshot.",
    )
    parser.add_argument(
        "--lifecycle-timeout",
        type=float,
        default=10.0,
        help="Maximum seconds to spend on the local lifecycle snapshot.",
    )
    parser.add_argument(
        "--smoke-timeout",
        type=float,
        default=45.0,
        help="Maximum seconds to spend on the brain-loop smoke.",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip the local REST health check.",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8100",
        help="REST server base URL for the health check.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1.0,
        help="HTTP health-check timeout in seconds.",
    )
    parser.add_argument(
        "--require-golden-loop",
        action="store_true",
        help=(
            "Also verify hooks/promotion-window/MCP surface and run continuity "
            "golden-path smoke (promote Decisions → cold get_context/recall)."
        ),
    )
    parser.add_argument(
        "--continuity-timeout",
        type=float,
        default=60.0,
        help="Maximum seconds for continuity golden-path smoke.",
    )


async def build_doctor_report(args: argparse.Namespace) -> dict[str, Any]:
    """Run diagnostics and return a JSON-serializable report."""
    checks: list[dict[str, Any]] = []
    config = _load_config(args, checks)
    resolved_mode: str | None = None

    if config is not None:
        _check_sqlite_config(config, checks)
        resolved_mode = await _check_mode_resolution(config, checks)
    else:
        _add_check(checks, "sqlite", "skipped", "config did not load")
        _add_check(checks, "mode", "skipped", "config did not load")

    lifecycle_summary = None
    if args.no_lifecycle:
        _add_check(checks, "lifecycle_snapshot", "skipped", "lifecycle snapshot skipped")
    elif config is None:
        _add_check(checks, "lifecycle_snapshot", "skipped", "config did not load")
    else:
        lifecycle_summary = await _bounded_doctor_stage(
            "lifecycle_snapshot",
            timeout_seconds=args.lifecycle_timeout,
            checks=checks,
            detail_prefix="lifecycle snapshot",
            awaitable=_check_lifecycle_snapshot(
                config,
                args,
                checks,
                resolved_mode=resolved_mode,
            ),
        )

    if args.skip_server:
        _add_check(checks, "server", "skipped", "server health check skipped")
        _add_check(checks, "mcp", "skipped", "MCP readiness check skipped")
    else:
        _check_server(args, checks)
        _check_mcp(args, checks)

    smoke_report = None
    if args.no_smoke:
        _add_check(checks, "brain_loop_smoke", "skipped", "brain-loop smoke skipped")
    elif config is None:
        _add_check(checks, "brain_loop_smoke", "skipped", "config did not load")
    else:
        smoke_report = await _bounded_doctor_stage(
            "brain_loop_smoke",
            timeout_seconds=args.smoke_timeout,
            checks=checks,
            detail_prefix="brain-loop smoke",
            awaitable=_check_brain_loop_smoke(
                config,
                args,
                checks,
                resolved_mode=resolved_mode,
            ),
        )

    continuity_report = None
    require_golden = bool(getattr(args, "require_golden_loop", False))
    if require_golden:
        _check_hooks_install(checks)
        _check_promotion_window(checks)
        _check_mcp_surface(checks)
        continuity_report = await _bounded_doctor_stage(
            "continuity_smoke",
            timeout_seconds=float(getattr(args, "continuity_timeout", 60.0) or 60.0),
            checks=checks,
            detail_prefix="continuity golden-path smoke",
            awaitable=_check_continuity_smoke(checks),
        )
    else:
        _add_check(
            checks,
            "hooks",
            "skipped",
            "pass --require-golden-loop to verify capture/promote hooks",
        )
        _add_check(
            checks,
            "promotion_window",
            "skipped",
            "pass --require-golden-loop to verify compaction window path",
        )
        _add_check(
            checks,
            "mcp_surface",
            "skipped",
            "pass --require-golden-loop to verify public MCP tool freeze",
        )
        _add_check(
            checks,
            "continuity_smoke",
            "skipped",
            "pass --require-golden-loop to run promote→cold-recall smoke",
        )

    checks = _sort_checks(checks)
    return {
        "status": _overall_status(checks),
        "checks": checks,
        "lifecycle_summary": lifecycle_summary,
        "smoke_report": smoke_report,
        "continuity_report": continuity_report,
    }


async def run_doctor_command(args: argparse.Namespace) -> None:
    """Print diagnostics and exit non-zero only on hard failures."""
    report = await build_doctor_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_doctor_report(report))
    if report["status"] == "fail":
        raise SystemExit(1)


def format_doctor_report(report: dict[str, Any]) -> str:
    """Render a compact Markdown diagnostic report."""
    lines = [
        "# Engram Doctor",
        "",
        f"Overall: `{report.get('status', 'unknown')}`",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks") or []:
        summary = f"- {check.get('name')}: `{check.get('status')}`"
        detail = check.get("detail")
        if detail:
            summary += f" - {detail}"
        lines.append(summary)

    smoke = report.get("smoke_report") or {}
    lifecycle = report.get("lifecycle_summary") or {}
    if lifecycle:
        totals = lifecycle.get("totals") or {}
        cue = lifecycle.get("cue") or {}
        project = lifecycle.get("project") or {}
        consolidate = lifecycle.get("consolidate") or {}
        latest_cycle = consolidate.get("latestCycle") or {}
        consolidate_issue = cycle_issue_text(latest_cycle)
        consolidate_issue_text = f" | error `{consolidate_issue}`" if consolidate_issue else ""
        lines.extend(
            [
                "",
                "## Lifecycle Snapshot",
                "",
                (
                    f"- Group: `{lifecycle.get('groupId')}` | Episodes: "
                    f"{totals.get('episodes', 0)} | Cues: {totals.get('cues', 0)} | "
                    f"Projected: {totals.get('projected', 0)} | Cycles: "
                    f"{totals.get('cycles', 0)}"
                ),
                (
                    f"- Cue coverage: {_percent(cue.get('coverage', 0.0))} | "
                    f"Project: `{project.get('status', 'unknown')}` | "
                    f"Consolidate: `{consolidate.get('status', 'unknown')}`"
                    f"{consolidate_issue_text}"
                ),
            ]
        )

    if smoke:
        totals = smoke.get("totals") or {}
        project = smoke.get("project") or {}
        consolidate = smoke.get("consolidate") or {}
        coverage_gaps = list(smoke.get("coverage_gaps") or [])
        evaluation_summary = _evaluation_signal_summary(smoke)
        lines.extend(
            [
                "",
                "## Brain Loop Smoke",
                "",
                (
                    f"- Group: `{smoke.get('group_id')}` | Episodes: "
                    f"{totals.get('episodes', 0)} | Projected: "
                    f"{project.get('projected_count', 0)} | Cycles: "
                    f"{consolidate.get('cycle_count', 0)}"
                ),
                f"- Coverage gaps: {len(coverage_gaps)}",
            ]
        )
        lines.extend(f"  - {gap}" for gap in coverage_gaps)
        if evaluation_summary is not None:
            lines.append(
                "- Evaluation signals: "
                f"{evaluation_summary['measured']}/{evaluation_summary['required']} "
                "measured"
            )
            lines.extend(f"  - {failure}" for failure in evaluation_summary["unmeasured"])

    continuity = report.get("continuity_report") or {}
    if continuity:
        lines.extend(
            [
                "",
                "## Continuity Golden Path",
                "",
                f"- Passed: `{continuity.get('passed')}`",
                f"- Context hit: `{continuity.get('context_hit')}`",
                f"- Recall hit: `{continuity.get('recall_hit')}`",
            ]
        )

    return "\n".join(lines).strip() + "\n"


def _load_config(
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
) -> EngramConfig | None:
    try:
        config = EngramConfig(mode=args.mode) if args.mode else EngramConfig()
    except Exception as exc:
        _add_check(checks, "config", "fail", f"config failed to load: {exc}")
        return None

    _add_check(
        checks,
        "config",
        "pass",
        "config loaded",
        {
            "configured_mode": config.mode,
            "default_group_id": config.default_group_id,
            "consolidation_profile": config.activation.consolidation_profile,
            "recall_profile": config.activation.recall_profile,
            "integration_profile": config.activation.integration_profile,
        },
    )
    return config


def _check_sqlite_config(config: EngramConfig, checks: list[dict[str, Any]]) -> None:
    sqlite_path = config.get_sqlite_path()
    parent = sqlite_path.parent
    metadata = {"sqlite_path": str(sqlite_path)}
    if sqlite_path.exists():
        _add_check(checks, "sqlite", "pass", "configured SQLite DB exists", metadata)
        return
    if parent.exists() and parent.is_dir():
        _add_check(
            checks,
            "sqlite",
            "pass",
            "configured SQLite parent directory exists",
            metadata,
        )
        return
    _add_check(
        checks,
        "sqlite",
        "warn",
        "configured SQLite parent directory does not exist yet",
        metadata,
    )


async def _check_mode_resolution(config: EngramConfig, checks: list[dict[str, Any]]) -> str | None:
    try:
        resolved = await resolve_mode(config.mode)
    except Exception as exc:
        _add_check(checks, "mode", "fail", f"mode resolution failed: {exc}")
        return None
    _add_check(
        checks,
        "mode",
        "pass",
        f"resolved mode: {resolved.value}",
        {"configured_mode": config.mode, "resolved_mode": resolved.value},
    )
    return resolved.value


def _check_server(args: argparse.Namespace, checks: list[dict[str, Any]]) -> None:
    base_url = str(args.server_url).rstrip("/")
    url = f"{base_url}/health"
    try:
        with request.urlopen(url, timeout=max(0.1, float(args.timeout))) as resp:
            if resp.status == 200:
                _add_check(
                    checks,
                    "server",
                    "pass",
                    "REST health check passed",
                    {"url": url, "status_code": resp.status},
                )
                return
            _add_check(
                checks,
                "server",
                "warn",
                f"REST health returned HTTP {resp.status}",
                {"url": url, "status_code": resp.status},
            )
    except (OSError, error.URLError, TimeoutError) as exc:
        _add_check(
            checks,
            "server",
            "warn",
            f"REST server not reachable: {exc}",
            {"url": url},
        )


def _check_mcp(args: argparse.Namespace, checks: list[dict[str, Any]]) -> None:
    base_url = str(args.server_url).rstrip("/")
    url = f"{base_url}/mcp"
    try:
        with request.urlopen(url, timeout=max(0.1, float(args.timeout))) as resp:
            if resp.status == 200:
                _add_check(
                    checks,
                    "mcp",
                    "pass",
                    "MCP streamable HTTP endpoint is reachable",
                    {"url": url, "status_code": resp.status},
                )
                return
            _add_check(
                checks,
                "mcp",
                "warn",
                f"MCP endpoint returned HTTP {resp.status}",
                {"url": url, "status_code": resp.status},
            )
    except (OSError, error.URLError, TimeoutError) as exc:
        _add_check(
            checks,
            "mcp",
            "warn",
            f"MCP endpoint not reachable: {exc}",
            {"url": url},
        )


async def _bounded_doctor_stage(
    name: str,
    *,
    timeout_seconds: float | None,
    checks: list[dict[str, Any]],
    detail_prefix: str,
    awaitable: Any,
) -> dict[str, Any] | None:
    timeout = _positive_timeout(timeout_seconds)
    if timeout is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        _add_check(
            checks,
            name,
            "warn",
            f"{detail_prefix} timed out after {timeout:g}s",
            {"timeout_seconds": timeout},
        )
        return None


async def _check_lifecycle_snapshot(
    config: EngramConfig,
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
    *,
    resolved_mode: str | None,
) -> dict[str, Any] | None:
    group_id = args.group_id or config.default_group_id
    try:
        summary = await build_lifecycle_summary_for_config(
            config,
            helix_data_dir=args.helix_data_dir,
            group_id=group_id,
            episode_limit=5,
            cycle_limit=10,
            top_n=10,
        )
    except Exception as exc:
        _add_check(
            checks,
            "lifecycle_snapshot",
            "fail",
            f"lifecycle snapshot failed: {exc}",
        )
        return None

    totals = summary.get("totals") or {}
    capture = summary.get("capture") or {}
    cue = summary.get("cue") or {}
    project = summary.get("project") or {}
    recall = summary.get("recall") or {}
    consolidate = summary.get("consolidate") or {}
    consolidate_issue = cycle_issue_text(consolidate.get("latestCycle") or {})
    stage_statuses = {
        "capture": capture.get("status"),
        "cue": cue.get("status"),
        "project": project.get("status"),
        "recall": recall.get("status"),
        "consolidate": consolidate.get("status"),
    }
    attention_stages = [stage for stage, status in stage_statuses.items() if status == "attention"]
    status = "warn" if attention_stages else "pass"
    if attention_stages:
        attention_labels = [
            f"{stage} ({consolidate_issue})"
            if stage == "consolidate" and consolidate_issue
            else stage
            for stage in attention_stages
        ]
        detail = "lifecycle snapshot loaded with attention: " + ", ".join(attention_labels)
    else:
        detail = "Capture -> Cue -> Project -> Recall -> Consolidate snapshot loaded"
    _add_check(
        checks,
        "lifecycle_snapshot",
        status,
        detail,
        {
            "group_id": summary.get("groupId"),
            "resolved_mode": resolved_mode,
            "configured_sqlite_path": str(config.get_sqlite_path()),
            "episodes": totals.get("episodes", 0),
            "cues": totals.get("cues", 0),
            "projected": totals.get("projected", 0),
            "cycles": totals.get("cycles", 0),
            "cue_coverage": cue.get("coverage", 0.0),
            "project_status": project.get("status"),
            "consolidate_status": consolidate.get("status"),
            "consolidate_issue": consolidate_issue,
            "stage_statuses": stage_statuses,
        },
    )
    return summary


async def _check_brain_loop_smoke(
    config: EngramConfig,
    args: argparse.Namespace,
    checks: list[dict[str, Any]],
    *,
    resolved_mode: str | None,
) -> dict[str, Any] | None:
    group_id = args.group_id or config.default_group_id
    smoke_mode = _smoke_mode_for_resolved_mode(resolved_mode)
    try:
        report = await run_projected_consolidated_smoke_for_args(
            sqlite_path=args.sqlite_path,
            replace=args.replace,
            group_id=group_id,
            mode=smoke_mode,
        )
    except SystemExit as exc:
        _add_check(checks, "brain_loop_smoke", "fail", f"brain-loop smoke failed: {exc.code}")
        return None
    except Exception as exc:
        _add_check(checks, "brain_loop_smoke", "fail", f"brain-loop smoke failed: {exc}")
        return None

    totals = report.get("totals") or {}
    project = report.get("project") or {}
    consolidate = report.get("consolidate") or {}
    evaluation_summary = _evaluation_signal_summary(report)
    metadata = {
        "group_id": report.get("group_id"),
        "mode": smoke_mode.value,
        "episodes": totals.get("episodes", 0),
        "projected": project.get("projected_count", 0),
        "cycles": consolidate.get("cycle_count", 0),
        "coverage_gaps": report.get("coverage_gaps") or [],
    }
    check_status = "pass"
    check_detail = (
        f"disposable {smoke_mode.value} Capture -> Cue -> Project -> "
        "Recall -> Consolidate smoke passed"
    )
    if evaluation_summary is not None:
        metadata["evaluation_signals"] = evaluation_summary
        if not evaluation_summary["ready"]:
            check_status = "fail"
            check_detail = (
                f"disposable {smoke_mode.value} Capture -> Cue -> Project -> "
                "Recall -> Consolidate smoke has unmeasured evaluation signals"
            )
    _add_check(
        checks,
        "brain_loop_smoke",
        check_status,
        check_detail,
        metadata,
    )
    return report


def _positive_timeout(value: Any) -> float | None:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return max(0.01, timeout)


def _smoke_mode_for_resolved_mode(resolved_mode: str | None) -> EngineMode:
    """Doctor smoke supports native Helix as the primary full-backend path."""
    if resolved_mode == EngineMode.HELIX.value:
        return EngineMode.HELIX
    return EngineMode.LITE


def _check_hooks_install(checks: list[dict[str, Any]]) -> None:
    """Verify Claude AutoCapture / PreCompact / promote nudge hooks exist."""
    hooks_dir = Path.home() / ".engram" / "hooks"
    required = (
        "capture-prompt.sh",
        "capture-response.sh",
        "session-start.sh",
        "session-end.sh",
        "pre-compact.sh",
        "session-promote-nudge.sh",
    )
    present = [name for name in required if (hooks_dir / name).is_file()]
    missing = [name for name in required if name not in present]
    metadata = {
        "hooks_dir": str(hooks_dir),
        "present": present,
        "missing": missing,
    }
    if not missing:
        _add_check(
            checks,
            "hooks",
            "pass",
            f"golden-loop hooks installed ({len(present)} scripts)",
            metadata,
        )
        return
    if present:
        _add_check(
            checks,
            "hooks",
            "warn",
            f"partial hooks install; missing: {', '.join(missing)} "
            "(run `engram hooks` or hooks/install-precompact.sh)",
            metadata,
        )
        return
    _add_check(
        checks,
        "hooks",
        "warn",
        "no Engram hooks under ~/.engram/hooks "
        "(run `engram hooks` for PreCompact + session promote)",
        metadata,
    )


def _check_promotion_window(checks: list[dict[str, Any]]) -> None:
    """Promotion window file is optional until first PreCompact; path must be writable."""
    from engram.extraction.promotion import default_promotion_window_path

    path = Path(default_promotion_window_path())
    metadata = {"path": str(path)}
    if path.is_file():
        _add_check(
            checks,
            "promotion_window",
            "pass",
            f"promotion window present at {path}",
            metadata,
        )
        return
    parent = path.parent
    if parent.exists() and parent.is_dir() and os.access(parent, os.W_OK):
        _add_check(
            checks,
            "promotion_window",
            "pass",
            "promotion window path writable (file created on first PreCompact)",
            metadata,
        )
        return
    _add_check(
        checks,
        "promotion_window",
        "warn",
        f"cannot write promotion window parent {parent}",
        metadata,
    )


def _check_mcp_surface(checks: list[dict[str, Any]]) -> None:
    """Public installs should freeze to the golden-loop tool set."""
    from engram.mcp.surface import PUBLIC_TOOLS, resolve_mcp_surface

    surface = resolve_mcp_surface()
    metadata = {
        "surface": surface,
        "public_tool_count": len(PUBLIC_TOOLS),
        "env": os.environ.get("ENGRAM_MCP_SURFACE", ""),
    }
    if surface == "public":
        _add_check(
            checks,
            "mcp_surface",
            "pass",
            f"MCP surface=public ({len(PUBLIC_TOOLS)} golden-loop tools)",
            metadata,
        )
        return
    _add_check(
        checks,
        "mcp_surface",
        "warn",
        f"MCP surface={surface} (agents see more than the golden loop; "
        "set ENGRAM_MCP_SURFACE=public for install clients)",
        metadata,
    )


async def _check_continuity_smoke(checks: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Run promote→cold get_context/recall product smoke."""
    from engram.evaluation.continuity import run_continuity_golden_path_smoke

    try:
        result = await run_continuity_golden_path_smoke()
    except Exception as exc:
        _add_check(
            checks,
            "continuity_smoke",
            "fail",
            f"continuity smoke crashed: {exc}",
        )
        return None

    status = "pass" if result.get("passed") else "fail"
    detail = (
        "continuity golden path PASS (promoted Decisions surface cold)"
        if result.get("passed")
        else "continuity golden path FAIL — cold get_context/recall missed Decisions"
    )
    _add_check(
        checks,
        "continuity_smoke",
        status,
        detail,
        {
            "passed": bool(result.get("passed")),
            "context_hit": bool(result.get("context_hit")),
            "recall_hit": bool(result.get("recall_hit")),
        },
    )
    return result if isinstance(result, dict) else None


def _add_check(
    checks: list[dict[str, Any]],
    name: str,
    status: str,
    detail: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    checks.append(
        {
            "name": name,
            "status": status,
            "detail": detail,
            "metadata": metadata or {},
        }
    )


def _sort_checks(checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(CHECK_ORDER)}
    return sorted(checks, key=lambda check: order.get(str(check.get("name")), len(order)))


def _overall_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(check.get("status")) for check in checks}
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "pass"


def _evaluation_signal_summary(
    report: Mapping[str, Any],
) -> dict[str, Any] | None:
    signals = report.get("evaluation_signals")
    if not isinstance(signals, Mapping):
        return None
    unmeasured = unmeasured_evaluation_signals(report)
    statuses: dict[str, str] = {}
    for signal in EVALUATION_SIGNAL_ORDER:
        payload = signals.get(signal)
        if isinstance(payload, Mapping):
            statuses[signal] = str(payload.get("status") or "missing")
        else:
            statuses[signal] = "missing"
    return {
        "required": len(EVALUATION_SIGNAL_ORDER),
        "measured": len(EVALUATION_SIGNAL_ORDER) - len(unmeasured),
        "ready": not unmeasured,
        "unmeasured": unmeasured,
        "statuses": statuses,
    }


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"
