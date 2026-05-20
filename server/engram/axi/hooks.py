"""Managed hook generation and installation for Engram AXI."""

from __future__ import annotations

import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram.axi.client import DEFAULT_SERVER_URL
from engram.axi.surfaces import AxiResult

MANAGED_HOOK_ID = "engram-axi-context"
MANAGED_BY = "engram"
DEFAULT_HOOK_TIMEOUT_SECONDS = 3.0
DEFAULT_HOOK_BUDGET = 800
MIN_HOOK_PROCESS_TIMEOUT_MS = 5000
DEFAULT_HOOK_TRACE_FILENAME = "axi-hook-runs.jsonl"
HOOK_TRACE_ORIGIN = "session-start-hook"
FOLLOWUP_TRACE_ORIGIN = "agent-followup"


@dataclass(frozen=True)
class HookTarget:
    """A concrete hook target and the config that should be merged into it."""

    client: str
    path: Path
    payload: dict[str, Any]
    command: str
    trace_file: Path


def build_hook_command(
    *,
    engram_command: str = "engram",
    server_url: str = DEFAULT_SERVER_URL,
    budget: int = DEFAULT_HOOK_BUDGET,
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    trace_file: Path | str | None = None,
    trace_client: str | None = None,
    trace_origin: str | None = None,
) -> str:
    """Return the read-only AXI command used by session-start hooks."""
    command = engram_command.strip() or "engram"
    parts = [
        command,
        "axi",
        "--server-url",
        server_url,
        "--project",
        "$PWD",
        "--budget",
        str(max(1, budget)),
        "--timeout",
        f"{timeout_seconds:g}",
    ]
    if trace_file is not None:
        parts.extend(["--trace-file", str(trace_file)])
    if trace_client:
        parts.extend(["--trace-client", trace_client])
    if trace_origin:
        parts.extend(["--trace-origin", trace_origin])
    rendered: list[str] = []
    for part in parts:
        if part == "$PWD":
            rendered.append('"$PWD"')
        else:
            rendered.append(shlex.quote(part))
    return " ".join(rendered)


def build_hook_target(
    client: str,
    *,
    home: Path | None = None,
    engram_command: str = "engram",
    server_url: str = DEFAULT_SERVER_URL,
    budget: int = DEFAULT_HOOK_BUDGET,
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    capture: bool = False,
) -> HookTarget:
    """Build the managed hook payload for a supported client."""
    normalized = normalize_hook_client(client)
    home = home or Path.home()
    trace_file = home / ".engram" / DEFAULT_HOOK_TRACE_FILENAME
    command = build_hook_command(
        engram_command=engram_command,
        server_url=server_url,
        budget=budget,
        timeout_seconds=timeout_seconds,
        trace_file=trace_file,
        trace_client=normalized,
        trace_origin=HOOK_TRACE_ORIGIN,
    )
    if normalized == "codex":
        path = home / ".codex" / "hooks.json"
        payload = _codex_hook_payload(
            command,
            capture=capture,
            timeout_seconds=timeout_seconds,
        )
    elif normalized == "claude-code":
        path = home / ".claude" / "settings.json"
        payload = _claude_hook_payload(
            command,
            capture=capture,
            timeout_seconds=timeout_seconds,
        )
    else:
        raise ValueError(f"Unsupported AXI hook client: {client}")
    return HookTarget(
        client=normalized,
        path=path,
        payload=payload,
        command=command,
        trace_file=trace_file,
    )


def build_hook_print_payload(
    client: str,
    *,
    home: Path | None = None,
    engram_command: str = "engram",
    server_url: str = DEFAULT_SERVER_URL,
    budget: int = DEFAULT_HOOK_BUDGET,
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    capture: bool = False,
) -> AxiResult:
    """Return an inspectable hook payload without touching user config."""
    target = build_hook_target(
        client,
        home=home,
        engram_command=engram_command,
        server_url=server_url,
        budget=budget,
        timeout_seconds=timeout_seconds,
        capture=capture,
    )
    return AxiResult(
        payload={
            "operation": "hooks.print",
            "status": "ok",
            "client": target.client,
            "path": str(target.path),
            "managed_hook_id": MANAGED_HOOK_ID,
            "capture": capture,
            "command": target.command,
            "trace_file": str(target.trace_file),
            "config": target.payload,
        }
    )


def build_hook_status_payload(
    client: str,
    *,
    home: Path | None = None,
) -> AxiResult:
    """Inspect whether a managed AXI startup hook is installed."""
    target = build_hook_target(client, home=home)
    existing = _load_json_object(target.path, backup_on_error=False)
    if target.client == "codex":
        hook = _find_codex_hook(existing)
    elif target.client == "claude-code":
        hook = _find_claude_hook(existing)
    else:
        raise ValueError(f"Unsupported AXI hook client: {client}")

    if hook is None:
        return AxiResult(
            payload={
                "operation": "hooks.status",
                "status": "missing",
                "client": target.client,
                "path": str(target.path),
                "managed_hook_id": MANAGED_HOOK_ID,
                "installed": False,
                "ready": False,
                "next": [
                    {
                        "cmd": f"engram axi hooks install {target.client}",
                        "reason": "Install read-only AXI session-start context",
                    }
                ],
            },
        )

    issues = _hook_status_issues(hook)
    trace_file = _hook_trace_file(hook)
    last_observed_run = (
        _last_hook_trace(trace_file, client=target.client, operations={"home"})
        if trace_file
        else None
    )
    last_run = (
        _last_hook_trace(
            trace_file,
            client=target.client,
            operations={"home"},
            origins={HOOK_TRACE_ORIGIN},
            successful=True,
        )
        if trace_file
        else None
    )
    last_followup = (
        _last_hook_trace(
            trace_file,
            client=target.client,
            operations={"context", "recall"},
            origins={FOLLOWUP_TRACE_ORIGIN},
            successful=True,
        )
        if trace_file
        else None
    )
    return AxiResult(
        payload={
            "operation": "hooks.status",
            "status": "installed" if not issues else "attention",
            "client": target.client,
            "path": str(target.path),
            "managed_hook_id": MANAGED_HOOK_ID,
            "installed": True,
            "ready": not issues,
            "read_only": hook.get("read_only") is True,
            "capture": hook.get("capture") is True,
            "command": str(hook.get("command") or ""),
            "timeout_ms": hook.get("timeout_ms") or hook.get("timeout"),
            "trace_file": str(trace_file) if trace_file else None,
            "last_run": last_run,
            "last_observed_run": last_observed_run,
            "last_followup": last_followup,
            "issues": issues,
            "next": (
                _hook_next_actions(target.client)
                if not issues
                else _hook_repair_actions(target.client)
            ),
        },
    )


def install_hook(
    client: str,
    *,
    home: Path | None = None,
    engram_command: str = "engram",
    server_url: str = DEFAULT_SERVER_URL,
    budget: int = DEFAULT_HOOK_BUDGET,
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS,
    capture: bool = False,
    dry_run: bool = False,
) -> AxiResult:
    """Merge a managed AXI session-start hook into a client config file."""
    target = build_hook_target(
        client,
        home=home,
        engram_command=engram_command,
        server_url=server_url,
        budget=budget,
        timeout_seconds=timeout_seconds,
        capture=capture,
    )
    existing = _load_json_object(target.path, backup_on_error=not dry_run)
    if target.client == "codex":
        merged = _merge_codex_hooks(existing, target.payload)
    elif target.client == "claude-code":
        merged = _merge_claude_settings(existing, target.payload)
    else:
        raise ValueError(f"Unsupported AXI hook client: {client}")

    changed = merged != existing
    if not dry_run and changed:
        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_text(json.dumps(merged, indent=2) + "\n")

    return AxiResult(
        payload={
            "operation": "hooks.install",
            "status": "ok",
            "client": target.client,
            "path": str(target.path),
            "managed_hook_id": MANAGED_HOOK_ID,
            "capture": capture,
            "dry_run": dry_run,
            "changed": changed,
            "command": target.command,
            "trace_file": str(target.trace_file),
            "next": _hook_next_actions(target.client),
        }
    )


def uninstall_hook(
    client: str,
    *,
    home: Path | None = None,
    dry_run: bool = False,
) -> AxiResult:
    """Remove Engram's managed AXI hook from a client config file."""
    target = build_hook_target(client, home=home)
    existing = _load_json_object(target.path, backup_on_error=not dry_run)
    if target.client == "codex":
        updated = _remove_codex_hook(existing)
    elif target.client == "claude-code":
        updated = _remove_claude_hook(existing)
    else:
        raise ValueError(f"Unsupported AXI hook client: {client}")

    changed = updated != existing
    if not dry_run and changed:
        target.path.parent.mkdir(parents=True, exist_ok=True)
        target.path.write_text(json.dumps(updated, indent=2) + "\n")

    return AxiResult(
        payload={
            "operation": "hooks.uninstall",
            "status": "ok",
            "client": target.client,
            "path": str(target.path),
            "managed_hook_id": MANAGED_HOOK_ID,
            "dry_run": dry_run,
            "changed": changed,
        }
    )


def normalize_hook_client(client: str) -> str:
    """Normalize accepted hook client aliases."""
    normalized = client.strip().lower()
    if normalized in {"claude", "claude-code", "claude_code"}:
        return "claude-code"
    if normalized == "codex":
        return normalized
    raise ValueError(f"Unsupported AXI hook client: {client}")


def _codex_hook_payload(
    command: str,
    *,
    capture: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    process_timeout = _hook_process_timeout_seconds(timeout_seconds)
    return {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "startup|resume|clear",
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                            "timeout": process_timeout,
                            "statusMessage": "Loading Engram AXI context",
                            "id": MANAGED_HOOK_ID,
                            "managed_by": MANAGED_BY,
                            "read_only": True,
                            "capture": capture,
                            "capture_policy": _capture_policy(capture),
                            "trace_file": _trace_file_from_command(command),
                        }
                    ],
                }
            ]
        }
    }


def _claude_hook_payload(
    command: str,
    *,
    capture: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    process_timeout_ms = _hook_process_timeout_ms(timeout_seconds)
    return {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": command,
                            "timeout": process_timeout_ms,
                            "async": False,
                            "id": MANAGED_HOOK_ID,
                            "managed_by": MANAGED_BY,
                            "read_only": True,
                            "capture": capture,
                            "capture_policy": _capture_policy(capture),
                            "trace_file": _trace_file_from_command(command),
                        }
                    ],
                }
            ]
        }
    }


def _capture_policy(capture: bool) -> str:
    if capture:
        return (
            "explicit-opt-in; do not install transcript capture unless the user "
            "requested --capture"
        )
    return "disabled; read-only session-start context only"


def _hook_process_timeout_ms(timeout_seconds: float) -> int:
    # The home packet may perform health, runtime, and storage calls. Keep each
    # REST call tightly bounded while giving the overall hook process enough
    # room to return degraded partial output instead of being killed.
    return int(max(MIN_HOOK_PROCESS_TIMEOUT_MS / 1000, timeout_seconds * 3 + 1) * 1000)


def _hook_process_timeout_seconds(timeout_seconds: float) -> int:
    return max(
        int(MIN_HOOK_PROCESS_TIMEOUT_MS / 1000),
        int(timeout_seconds * 3 + 1),
    )


def _load_json_object(path: Path, *, backup_on_error: bool = True) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        if backup_on_error:
            backup = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup)
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_codex_hooks(existing: dict[str, Any], managed_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    hooks = _copy_mapping(merged.get("hooks"))
    session_start = _copy_list(hooks.get("SessionStart"))
    session_start = [
        entry
        for entry in session_start
        if not (isinstance(entry, dict) and _is_managed_codex_entry(entry))
    ]
    session_start.extend(managed_payload["hooks"]["SessionStart"])
    hooks["SessionStart"] = session_start
    merged["hooks"] = hooks
    return merged


def _merge_claude_settings(
    existing: dict[str, Any],
    managed_payload: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(existing)
    hooks = _copy_mapping(merged.get("hooks"))
    session_start = _copy_list(hooks.get("SessionStart"))
    session_start = [
        entry
        for entry in session_start
        if not (isinstance(entry, dict) and _is_managed_claude_entry(entry))
    ]
    session_start.extend(managed_payload["hooks"]["SessionStart"])
    hooks["SessionStart"] = session_start
    merged["hooks"] = hooks
    return merged


def _remove_codex_hook(existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    hooks = _copy_mapping(merged.get("hooks"))
    session_start = _copy_list(hooks.get("SessionStart"))
    hooks["SessionStart"] = [
        entry
        for entry in session_start
        if not (isinstance(entry, dict) and _is_managed_codex_entry(entry))
    ]
    merged["hooks"] = hooks
    return merged


def _remove_claude_hook(existing: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    hooks = _copy_mapping(merged.get("hooks"))
    session_start = _copy_list(hooks.get("SessionStart"))
    hooks["SessionStart"] = [
        entry
        for entry in session_start
        if not (isinstance(entry, dict) and _is_managed_claude_entry(entry))
    ]
    merged["hooks"] = hooks
    return merged


def _copy_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _copy_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _find_codex_hook(existing: dict[str, Any]) -> dict[str, Any] | None:
    hooks = _copy_mapping(existing.get("hooks"))
    for entry in _copy_list(hooks.get("SessionStart")):
        if not isinstance(entry, dict):
            continue
        if _is_managed_hook(entry):
            return entry
        for nested in _copy_list(entry.get("hooks")):
            if isinstance(nested, dict) and _is_managed_hook(nested):
                return nested
    return None


def _find_claude_hook(existing: dict[str, Any]) -> dict[str, Any] | None:
    hooks = _copy_mapping(existing.get("hooks"))
    for entry in _copy_list(hooks.get("SessionStart")):
        if not isinstance(entry, dict):
            continue
        for nested in _copy_list(entry.get("hooks")):
            if isinstance(nested, dict) and _is_managed_hook(nested):
                return nested
    return None


def _is_managed_claude_entry(entry: dict[str, Any]) -> bool:
    nested = entry.get("hooks")
    if not isinstance(nested, list):
        return False
    return any(isinstance(item, dict) and _is_managed_hook(item) for item in nested)


def _is_managed_codex_entry(entry: dict[str, Any]) -> bool:
    if _is_managed_hook(entry):
        return True
    nested = entry.get("hooks")
    if not isinstance(nested, list):
        return False
    return any(isinstance(item, dict) and _is_managed_hook(item) for item in nested)


def _is_managed_hook(entry: dict[str, Any]) -> bool:
    if entry.get("id") == MANAGED_HOOK_ID:
        return True
    return entry.get("managed_by") == MANAGED_BY and "engram axi" in str(entry.get("command", ""))


def _hook_status_issues(hook: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    command = str(hook.get("command") or "")
    if "engram axi" not in command:
        issues.append("command_does_not_run_engram_axi")
    if hook.get("read_only") is not True:
        issues.append("not_marked_read_only")
    if hook.get("capture") is True:
        issues.append("capture_enabled")
    if not (hook.get("timeout_ms") or hook.get("timeout")):
        issues.append("missing_timeout")
    if not _hook_trace_file(hook):
        issues.append("missing_trace_file")
    if _trace_origin_from_command(command) != HOOK_TRACE_ORIGIN:
        issues.append("missing_trace_origin")
    return issues


def _hook_trace_file(hook: dict[str, Any]) -> Path | None:
    raw = hook.get("trace_file")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    parsed = _trace_file_from_command(str(hook.get("command") or ""))
    return Path(parsed).expanduser() if parsed else None


def _trace_file_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part == "--trace-file" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _trace_origin_from_command(command: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts):
        if part == "--trace-origin" and index + 1 < len(parts):
            return parts[index + 1]
    return None


def _last_hook_trace(
    path: Path,
    *,
    client: str,
    operations: set[str] | None = None,
    origins: set[str] | None = None,
    successful: bool = False,
) -> dict[str, Any] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-200:]):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(record, dict)
            and record.get("hookId") == MANAGED_HOOK_ID
            and record.get("client") == client
            and (operations is None or record.get("operation") in operations)
            and (origins is None or record.get("origin") in origins)
            and (not successful or record.get("exitCode") == 0)
        ):
            return {
                "timestamp": record.get("timestamp"),
                "operation": record.get("operation"),
                "status": record.get("status"),
                "exit_code": record.get("exitCode"),
                "duration_ms": record.get("durationMs"),
                "origin": record.get("origin"),
                "project": record.get("project"),
            }
    return None


def _hook_repair_actions(client: str) -> list[dict[str, str]]:
    return [
        {
            "cmd": f"engram axi hooks install {client}",
            "reason": "Rewrite Engram's managed startup hook safely",
        },
        {
            "cmd": f"engram axi hooks uninstall {client}",
            "reason": "Remove Engram's managed startup hook",
        },
    ]


def _hook_next_actions(client: str) -> list[dict[str, str]]:
    if client == "codex":
        return [
            {
                "cmd": "start a new Codex session",
                "reason": "Codex reads hooks at session startup",
            },
            {
                "cmd": "engram axi --project \"$PWD\" --budget 800 --timeout 3",
                "reason": "Manually inspect the same read-only packet",
            },
        ]
    return [
        {
            "cmd": "start a new Claude Code session",
            "reason": "Claude Code reads SessionStart hooks at session startup",
        },
        {
            "cmd": "engram axi --project \"$PWD\" --budget 800 --timeout 3",
            "reason": "Manually inspect the same read-only packet",
        },
    ]
