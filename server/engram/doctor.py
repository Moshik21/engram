"""Operator diagnostics for local Engram installs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from urllib import error, request

from engram.config import EngramConfig
from engram.evaluation.smoke import run_projected_consolidated_smoke_for_args
from engram.lifecycle_cli import build_lifecycle_summary_for_config, cycle_issue_text
from engram.storage.resolver import EngineMode, resolve_mode

CHECK_ORDER = ("config", "sqlite", "mode", "lifecycle_snapshot", "server", "brain_loop_smoke")


def configure_doctor_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `engram doctor` options to a parser."""
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
        help="Skip the Capture -> Cue -> Project -> Recall -> Consolidate smoke.",
    )
    parser.add_argument(
        "--no-lifecycle",
        action="store_true",
        help="Skip the local Capture -> Cue -> Project -> Recall -> Consolidate snapshot.",
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
        lifecycle_summary = await _check_lifecycle_snapshot(
            config,
            args,
            checks,
            resolved_mode=resolved_mode,
        )

    if args.skip_server:
        _add_check(checks, "server", "skipped", "server health check skipped")
    else:
        _check_server(args, checks)

    smoke_report = None
    if args.no_smoke:
        _add_check(checks, "brain_loop_smoke", "skipped", "brain-loop smoke skipped")
    elif config is None:
        _add_check(checks, "brain_loop_smoke", "skipped", "config did not load")
    else:
        smoke_report = await _check_brain_loop_smoke(
            config,
            args,
            checks,
            resolved_mode=resolved_mode,
        )

    checks = _sort_checks(checks)
    return {
        "status": _overall_status(checks),
        "checks": checks,
        "lifecycle_summary": lifecycle_summary,
        "smoke_report": smoke_report,
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
        consolidate_issue_text = (
            f" | error `{consolidate_issue}`" if consolidate_issue else ""
        )
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
    attention_stages = [
        stage for stage, status in stage_statuses.items() if status == "attention"
    ]
    status = "warn" if attention_stages else "pass"
    if attention_stages:
        attention_labels = [
            f"{stage} ({consolidate_issue})"
            if stage == "consolidate" and consolidate_issue
            else stage
            for stage in attention_stages
        ]
        detail = "lifecycle snapshot loaded with attention: " + ", ".join(
            attention_labels
        )
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
    _add_check(
        checks,
        "brain_loop_smoke",
        "pass",
        (
            f"disposable {smoke_mode.value} Capture -> Cue -> Project -> "
            "Recall -> Consolidate smoke passed"
        ),
        {
            "group_id": report.get("group_id"),
            "mode": smoke_mode.value,
            "episodes": totals.get("episodes", 0),
            "projected": project.get("projected_count", 0),
            "cycles": consolidate.get("cycle_count", 0),
            "coverage_gaps": report.get("coverage_gaps") or [],
        },
    )
    return report


def _smoke_mode_for_resolved_mode(resolved_mode: str | None) -> EngineMode:
    """Doctor smoke supports native Helix as the primary full-backend path."""
    if resolved_mode == EngineMode.HELIX.value:
        return EngineMode.HELIX
    return EngineMode.LITE


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


def _percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"
