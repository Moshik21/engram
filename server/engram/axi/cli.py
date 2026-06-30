"""CLI dispatch for Engram's AXI surface."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from engram.axi.client import DEFAULT_SERVER_URL, DEFAULT_TIMEOUT_SECONDS, AxiRestClient
from engram.axi.hooks import (
    DEFAULT_HOOK_TIMEOUT_SECONDS,
    FOLLOWUP_TRACE_ORIGIN,
    HOOK_TRACE_ORIGIN,
    build_hook_print_payload,
    build_hook_status_payload,
    install_hook,
    uninstall_hook,
)
from engram.retrieval.artifacts import _normalize_project_path as _resolve_project_path
from engram.axi.surfaces import (
    VALUE_TIMEOUT_SECONDS,
    AxiResult,
    build_bootstrap_payload,
    build_context_payload,
    build_doctor_payload,
    build_home_payload,
    build_packet_cache_clear_payload,
    build_packet_cache_summary_payload,
    build_recall_payload,
    build_storage_payload,
    build_value_payload,
    build_write_payload,
)
from engram.axi.toon import render_toon


def configure_axi_parser(parser: argparse.ArgumentParser) -> None:
    """Configure `engram axi` arguments."""
    parser.description = "Compact agent-facing AXI interface for Engram"
    parser.set_defaults(_axi_timeout_default=DEFAULT_TIMEOUT_SECONDS)
    _add_common_args(parser)
    parser.add_argument(
        "--project",
        dest="project_path",
        default=None,
        help="Current project path for home/context runtime hints.",
    )
    parser.add_argument(
        "--topic",
        dest="topic_hint",
        default=None,
        help="Topic hint for home/context packets.",
    )

    common = argparse.ArgumentParser(add_help=False)
    _add_common_args(common, suppress_defaults=True)
    subparsers = parser.add_subparsers(dest="axi_command")

    context_parser = subparsers.add_parser(
        "context",
        parents=[common],
        help="Print compact Engram context.",
    )
    context_parser.add_argument("--topic", dest="topic_hint", default=argparse.SUPPRESS)
    context_parser.add_argument("--project", dest="project_path", default=argparse.SUPPRESS)

    recall_parser = subparsers.add_parser(
        "recall",
        parents=[common],
        help="Search Engram memory.",
    )
    recall_parser.add_argument("query")
    recall_parser.add_argument("--limit", type=int, default=5)
    recall_parser.add_argument("--project", dest="project_path", default=argparse.SUPPRESS)

    subparsers.add_parser(
        "storage",
        parents=[common],
        help="Print storage paths, counts, and size.",
    )

    value_parser = subparsers.add_parser(
        "value",
        parents=[common],
        help="Print compact memory value and latency status.",
    )
    value_parser.set_defaults(_axi_timeout_default=VALUE_TIMEOUT_SECONDS)

    hook_run_parser = subparsers.add_parser(
        "hook-run",
        parents=[common],
        help=argparse.SUPPRESS,
    )
    hook_run_parser.add_argument("--project", dest="project_path", default=argparse.SUPPRESS)

    packet_cache_parser = subparsers.add_parser(
        "packet-cache",
        parents=[common],
        help="Inspect or clear cached memory packets.",
    )
    packet_cache_parser.set_defaults(packet_cache_command="summary")
    packet_cache_subparsers = packet_cache_parser.add_subparsers(
        dest="packet_cache_command",
        required=False,
    )
    packet_cache_subparsers.default = "summary"
    packet_cache_subparsers.add_parser(
        "summary",
        parents=[common],
        help="Print packet-cache diagnostics without clearing entries.",
    )
    packet_cache_subparsers.add_parser(
        "clear",
        parents=[common],
        help="Clear tenant-local packet-cache entries.",
    )

    doctor_parser = subparsers.add_parser(
        "doctor",
        parents=[common],
        help="Run a compact AXI readiness probe.",
    )
    doctor_parser.add_argument("--project", dest="project_path", default=argparse.SUPPRESS)
    doctor_parser.add_argument(
        "--hooks",
        nargs="*",
        choices=["codex", "claude-code", "claude"],
        default=[],
        help="Also verify managed AXI startup hooks for the selected clients.",
    )
    doctor_parser.add_argument(
        "--require-hook-run",
        action="store_true",
        help="Fail unless each requested hook has metadata-only run evidence.",
    )
    doctor_parser.add_argument(
        "--require-followup",
        action="store_true",
        help="Fail unless each requested hook has context/recall follow-up evidence.",
    )
    doctor_parser.add_argument(
        "--home",
        default=None,
        help=argparse.SUPPRESS,
    )

    for command in ("observe", "remember"):
        write_parser = subparsers.add_parser(
            command,
            parents=[common],
            help=f"{command.title()} explicit stdin content.",
        )
        write_parser.add_argument(
            "--stdin",
            action="store_true",
            help="Read content from stdin. Required for write commands.",
        )
        write_parser.add_argument("--source", default="axi")
        write_parser.add_argument("--conversation-date", default=None)

    bootstrap_parser = subparsers.add_parser(
        "bootstrap",
        parents=[common],
        help="Bootstrap project artifacts through the running REST API.",
    )
    bootstrap_parser.add_argument("project_path")
    bootstrap_parser.add_argument(
        "--include",
        dest="include_patterns",
        action="append",
        default=None,
        help="Additional include glob. Repeatable.",
    )

    hooks_parser = subparsers.add_parser(
        "hooks",
        parents=[common],
        help="Print, install, inspect, or uninstall read-only AXI session-start hooks.",
    )
    hooks_subparsers = hooks_parser.add_subparsers(dest="hooks_command", required=True)

    for hook_command in ("print", "install", "status", "uninstall"):
        hook_parser = hooks_subparsers.add_parser(
            hook_command,
            parents=[common],
            help=f"{hook_command.title()} a managed AXI hook config.",
        )
        hook_parser.add_argument("client", choices=["codex", "claude-code", "claude"])
        hook_parser.add_argument(
            "--capture",
            action="store_true",
            help="Explicitly mark capture opt-in policy. Read-only is the default.",
        )
        hook_parser.add_argument(
            "--home",
            default=None,
            help=argparse.SUPPRESS,
        )
        hook_parser.add_argument(
            "--engram-command",
            default="engram",
            help="Executable path to place in the managed hook command.",
        )
        if hook_command in {"install", "uninstall"}:
            hook_parser.add_argument(
                "--dry-run",
                action="store_true",
                help="Print what would change without writing config.",
            )


def run_axi_command(args: argparse.Namespace) -> int:
    """Run an AXI command and print its output."""
    started = time.perf_counter()
    timeout_seconds = _resolved_timeout(args)
    client = AxiRestClient(
        server_url=args.server_url,
        timeout_seconds=timeout_seconds,
        auth_token=args.auth_token,
    )
    result = _dispatch(args, client)
    duration_ms = int((time.perf_counter() - started) * 1000)
    _print_result(result, json_output=args.json)
    _write_trace(args, result, duration_ms=duration_ms)
    return result.exit_code


def _add_common_args(
    parser: argparse.ArgumentParser,
    *,
    suppress_defaults: bool = False,
) -> None:
    default = argparse.SUPPRESS if suppress_defaults else None
    parser.add_argument(
        "--server-url",
        default=argparse.SUPPRESS if suppress_defaults else DEFAULT_SERVER_URL,
        help="Engram REST server URL.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else 800,
        help="Approximate output token budget.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=default,
        help="Print stable JSON instead of compact text.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=default,
        help="Disable output truncation for supported commands.",
    )
    parser.add_argument(
        "--auth-token",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="Bearer token for auth-enabled local runtimes.",
    )
    parser.add_argument(
        "--trace-file",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-client",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--trace-origin",
        default=argparse.SUPPRESS if suppress_defaults else None,
        help=argparse.SUPPRESS,
    )


def _dispatch(args: argparse.Namespace, client: AxiRestClient) -> AxiResult:
    command = getattr(args, "axi_command", None)
    if command is None:
        project_path = _normalize_project_path(getattr(args, "project_path", None))
        return build_home_payload(
            client,
            project_path=project_path,
            topic_hint=getattr(args, "topic_hint", None),
            budget=args.budget,
            trace_file=getattr(args, "trace_file", None),
            trace_client=getattr(args, "trace_client", None),
            followup_trace_origin=FOLLOWUP_TRACE_ORIGIN,
        )
    if command == "context":
        return build_context_payload(
            client,
            topic_hint=getattr(args, "topic_hint", None),
            project_path=_normalize_project_path(getattr(args, "project_path", None)),
            budget=args.budget,
            full=args.full,
        )
    if command == "recall":
        return build_recall_payload(
            client,
            query=args.query,
            limit=max(1, args.limit),
            budget=args.budget,
            project_path=_normalize_project_path(getattr(args, "project_path", None)),
            full=args.full,
        )
    if command == "storage":
        return build_storage_payload(client)
    if command == "value":
        return build_value_payload(client)
    if command == "hook-run":
        project_path = _normalize_project_path(
            _hook_input_project_path(
                _read_hook_input_stdin(),
                explicit_project_path=getattr(args, "project_path", None),
            )
        )
        setattr(args, "project_path", project_path)
        return build_home_payload(
            client,
            project_path=project_path,
            topic_hint=getattr(args, "topic_hint", None),
            budget=args.budget,
            trace_file=getattr(args, "trace_file", None),
            trace_client=getattr(args, "trace_client", None),
            followup_trace_origin=FOLLOWUP_TRACE_ORIGIN,
        )
    if command == "packet-cache":
        packet_cache_command = getattr(args, "packet_cache_command", None) or "summary"
        if packet_cache_command == "summary":
            return build_packet_cache_summary_payload(client)
        if packet_cache_command == "clear":
            return build_packet_cache_clear_payload(client)
        return AxiResult(
            payload={
                "operation": "packet-cache",
                "status": "error",
                "error": "Unknown packet-cache command",
            },
            exit_code=2,
        )
    if command == "doctor":
        result = build_doctor_payload(
            client,
            project_path=_normalize_project_path(getattr(args, "project_path", None)),
        )
        return _with_hook_doctor_checks(
            result,
            clients=getattr(args, "hooks", []),
            home=Path(args.home).expanduser() if getattr(args, "home", None) else None,
            require_hook_run=getattr(args, "require_hook_run", False),
            require_followup=getattr(args, "require_followup", False),
        )
    if command in {"observe", "remember"}:
        if not getattr(args, "stdin", False):
            return AxiResult(
                payload={
                    "operation": command,
                    "status": "error",
                    "error": f"`engram axi {command}` requires --stdin",
                    "next": [
                        {
                            "cmd": f"printf '%s' 'memory text' | engram axi {command} --stdin",
                            "reason": "Capture only explicit user-approved content",
                        }
                    ],
                },
                exit_code=2,
            )
        return build_write_payload(
            client,
            operation=command,
            content=sys.stdin.read(),
            source=args.source,
            conversation_date=args.conversation_date,
        )
    if command == "bootstrap":
        return build_bootstrap_payload(
            client,
            project_path=str(Path(args.project_path).expanduser()),
            include_patterns=args.include_patterns,
        )
    if command == "hooks":
        try:
            home = Path(args.home).expanduser() if getattr(args, "home", None) else None
            hook_timeout = _hook_timeout(_resolved_timeout(args))
            if args.hooks_command == "print":
                return build_hook_print_payload(
                    args.client,
                    home=home,
                    engram_command=args.engram_command,
                    server_url=args.server_url,
                    budget=args.budget,
                    timeout_seconds=hook_timeout,
                    capture=args.capture,
                )
            if args.hooks_command == "install":
                return install_hook(
                    args.client,
                    home=home,
                    engram_command=args.engram_command,
                    server_url=args.server_url,
                    budget=args.budget,
                    timeout_seconds=hook_timeout,
                    capture=args.capture,
                    dry_run=args.dry_run,
                )
            if args.hooks_command == "status":
                return build_hook_status_payload(
                    args.client,
                    home=home,
                )
            if args.hooks_command == "uninstall":
                return uninstall_hook(
                    args.client,
                    home=home,
                    dry_run=args.dry_run,
                )
        except ValueError as exc:
            return AxiResult(
                payload={
                    "operation": "hooks",
                    "status": "error",
                    "error": str(exc),
                },
                exit_code=2,
            )
    return AxiResult(
        payload={"operation": command or "home", "status": "error", "error": "Unknown command"},
        exit_code=2,
    )


def _with_hook_doctor_checks(
    result: AxiResult,
    *,
    clients: list[str],
    home: Path | None,
    require_hook_run: bool = False,
    require_followup: bool = False,
) -> AxiResult:
    if not clients:
        return result
    payload = dict(result.payload)
    checks = list(payload.get("checks") or [])
    hook_statuses: list[dict[str, object]] = []
    for client in clients:
        hook_result = build_hook_status_payload(client, home=home)
        hook_payload = hook_result.payload
        last_run = hook_payload.get("last_run")
        last_observed_run = hook_payload.get("last_observed_run")
        last_followup = hook_payload.get("last_followup")
        ready = hook_payload.get("ready") is True
        status = "pass" if ready else "fail"
        check: dict[str, object] = {"name": f"hook:{hook_payload.get('client')}", "status": status}
        issues = hook_payload.get("issues") or []
        if issues:
            check["detail"] = ",".join(str(item) for item in issues)
        elif not ready:
            check["detail"] = hook_payload.get("status") or "not_ready"
        elif require_hook_run and not last_run:
            check["status"] = "fail"
            check["detail"] = (
                "missing_session_start_origin" if last_observed_run else "missing_last_run"
            )
        elif (
            require_hook_run
            and isinstance(last_run, dict)
            and last_run.get("origin") != HOOK_TRACE_ORIGIN
        ):
            check["status"] = "fail"
            check["detail"] = "missing_session_start_origin"
        elif require_hook_run and hook_payload.get("last_run_stale_after_config_change"):
            check["status"] = "fail"
            check["detail"] = "stale_session_start_run"
        elif require_hook_run and hook_payload.get("last_run_project_root"):
            check["status"] = "fail"
            check["detail"] = "session_start_project_root"
        elif require_followup and not last_followup:
            check["status"] = "fail"
            check["detail"] = "missing_followup"
        checks.append(check)
        hook_statuses.append(
            {
                "client": hook_payload.get("client"),
                "status": hook_payload.get("status"),
                "ready": ready,
                "read_only": hook_payload.get("read_only"),
                "capture": hook_payload.get("capture"),
                "last_run": last_run,
                "last_run_stale_after_config_change": hook_payload.get(
                    "last_run_stale_after_config_change"
                ),
                "last_run_project_root": hook_payload.get("last_run_project_root"),
                "last_observed_run": last_observed_run,
                "last_followup": last_followup,
                "followup_summary": hook_payload.get("followup_summary"),
                "issues": issues,
            }
        )
    payload["checks"] = checks
    payload["hooks"] = hook_statuses
    if any(check.get("status") != "pass" for check in checks):
        payload["status"] = "fail"
        return AxiResult(payload=payload, exit_code=1)
    payload["status"] = "pass"
    return AxiResult(payload=payload, exit_code=0)


def _write_trace(args: argparse.Namespace, result: AxiResult, *, duration_ms: int) -> None:
    raw_trace_file = getattr(args, "trace_file", None)
    if not raw_trace_file:
        return
    command = getattr(args, "axi_command", None) or "home"
    if command == "hook-run":
        command = "home"
    elif command == "hooks":
        command = f"hooks.{getattr(args, 'hooks_command', 'unknown')}"
    elif command == "packet-cache":
        command = f"packet-cache.{getattr(args, 'packet_cache_command', 'unknown')}"
    payload = result.payload
    record = {
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "hookId": "engram-axi-context",
        "client": getattr(args, "trace_client", None) or "manual",
        "origin": getattr(args, "trace_origin", None) or "manual",
        "operation": command,
        "status": payload.get("status"),
        "exitCode": result.exit_code,
        "durationMs": duration_ms,
        "server": getattr(args, "server_url", None),
        "project": _normalize_project_path(getattr(args, "project_path", None)),
        "budget": getattr(args, "budget", None),
        "timeoutSeconds": _resolved_timeout(args),
    }
    record.update(_trace_payload_metadata(payload))
    path = Path(raw_trace_file).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        # Tracing must never break the agent-facing context packet.
        return


def _trace_payload_metadata(payload: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {}
    packet_cache = payload.get("packet_cache")
    lifecycle = payload.get("lifecycle")
    budget = payload.get("budget")
    packet_count = _first_int(
        _get_dict_value(packet_cache, "packet_count", "packetCount"),
        _get_dict_value(lifecycle, "packetCount", "packet_count"),
        payload.get("packet_count"),
        _list_len(payload.get("packets")),
    )
    result_count = _first_int(
        _get_dict_value(lifecycle, "resultCount", "result_count"),
        payload.get("total_result_count"),
        payload.get("result_count"),
        _list_len(payload.get("results")),
    )
    fallback_status = _first_str(
        _get_dict_value(lifecycle, "fallbackStatus", "fallback_status"),
        _get_dict_value(budget, "fallbackStatus", "fallback_status"),
    )
    skip_reason = _first_str(
        _get_dict_value(lifecycle, "skipReason", "skip_reason"),
        _get_dict_value(budget, "skipReason", "skip_reason"),
    )
    cache_hit = _trace_cache_hit(packet_cache, lifecycle, budget, fallback_status, skip_reason)
    budget_miss = _first_bool(
        _get_dict_value(budget, "budgetMiss", "budget_miss"),
        _get_dict_value(lifecycle, "budgetMiss", "budget_miss"),
    )
    degraded = _first_bool(
        _get_dict_value(budget, "degraded"),
        _get_dict_value(lifecycle, "degraded"),
        payload.get("status") == "degraded",
    )
    if cache_hit is not None:
        metadata["cacheHit"] = cache_hit
    if packet_count is not None:
        metadata["packetCount"] = packet_count
    if result_count is not None:
        metadata["resultCount"] = result_count
    if fallback_status:
        metadata["fallbackStatus"] = fallback_status
    if skip_reason:
        metadata["skipReason"] = skip_reason
    if budget_miss is not None:
        metadata["budgetMiss"] = budget_miss
    if degraded is not None:
        metadata["degraded"] = degraded
    entity_count = _first_int(payload.get("entity_count"), payload.get("entityCount"))
    fact_count = _first_int(payload.get("fact_count"), payload.get("factCount"))
    if entity_count is not None:
        metadata["entityCount"] = entity_count
    if fact_count is not None:
        metadata["factCount"] = fact_count
    return metadata


def _trace_cache_hit(
    packet_cache: object,
    lifecycle: object,
    budget: object,
    fallback_status: str | None,
    skip_reason: str | None,
) -> bool | None:
    explicit = _first_bool(
        _get_dict_value(packet_cache, "hit", "cacheHit", "cache_hit"),
        _get_dict_value(lifecycle, "cacheHit", "cache_hit"),
        _get_dict_value(budget, "cacheHit", "cache_hit"),
    )
    if explicit is not None:
        return explicit
    if fallback_status == "cache_satisfied" or skip_reason == "cache_satisfied":
        return True
    return None


def _get_dict_value(value: object, *keys: str) -> object | None:
    if not isinstance(value, dict):
        return None
    for key in keys:
        if key in value:
            return value.get(key)
    return None


def _list_len(value: object) -> int | None:
    return len(value) if isinstance(value, list) else None


def _first_int(*values: object) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(float(value))
            except ValueError:
                continue
    return None


def _first_bool(*values: object) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "hit"}:
                return True
            if normalized in {"0", "false", "no", "miss"}:
                return False
    return None


def _first_str(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _print_result(result: AxiResult, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result.payload, indent=2, sort_keys=True))
    else:
        print(render_toon(result.payload), end="")


def _read_hook_input_stdin() -> str:
    try:
        if sys.stdin.isatty():
            return ""
        return sys.stdin.read()
    except OSError:
        return ""


def _hook_input_project_path(raw_input: str, *, explicit_project_path: str | None) -> str | None:
    if explicit_project_path:
        return explicit_project_path
    text = raw_input.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    for key in ("cwd", "workspace", "workspaceRoot", "projectPath", "project_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_project_path(project_path: str | None) -> str | None:
    if project_path:
        resolved = _resolve_project_path(project_path)
        if not resolved:
            return None
        path = Path(resolved)
        if _is_filesystem_root(path):
            return None
        return resolved
    return _infer_current_project_path()


def _infer_current_project_path() -> str | None:
    try:
        path = Path.cwd().resolve()
    except OSError:
        return None
    if not _looks_like_project_directory(path):
        return None
    return str(path)


def _is_filesystem_root(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return resolved.parent == resolved


def _looks_like_project_directory(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / marker).exists()
        for marker in (
            ".git",
            "README.md",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "docs",
        )
    )


def _hook_timeout(timeout: float) -> float:
    if timeout == DEFAULT_TIMEOUT_SECONDS:
        return DEFAULT_HOOK_TIMEOUT_SECONDS
    return timeout


def _resolved_timeout(args: argparse.Namespace) -> float:
    timeout = getattr(args, "timeout", None)
    if timeout is not None:
        return float(timeout)
    return float(getattr(args, "_axi_timeout_default", DEFAULT_TIMEOUT_SECONDS))
