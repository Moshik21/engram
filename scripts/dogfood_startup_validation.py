#!/usr/bin/env python3
"""Non-destructive Engram dogfood startup validation.

This runner inspects the installed local Engram path used during dogfooding. It
does not start, stop, delete, or rewrite anything by default.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

EXPECTED_MCP_TOOLS = {
    "remember",
    "observe",
    "recall",
    "get_context",
    "bootstrap_project",
    "claim_authority",
    "route_question",
}
EXPECTED_OPENCLAW_TRANSPORT = "streamable-http"

MANAGED_AXI_HOOK_ID = "engram-axi-context"
TRACE_FILE_RELATIVE = ".engram/axi-hook-runs.jsonl"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str
    timed_out: bool = False
    duration_seconds: float = 0.0


@dataclass
class Check:
    name: str
    status: str
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo).expanduser().resolve()
    home = Path(args.home).expanduser().resolve()
    env_path = home / ".engram/.env"
    env = parse_env_file(env_path)
    port = int(env.get("ENGRAM_API_PORT") or args.port)
    server_url = args.server_url or f"http://127.0.0.1:{port}"
    mcp_url = server_url.rstrip("/") + "/mcp"

    context = {
        "repo_root": str(repo_root),
        "home": str(home),
        "env_path": str(env_path),
        "port": port,
        "server_url": server_url,
        "mcp_url": mcp_url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    checks: list[Check] = []
    checks.append(check_local_config(env_path, env))
    health = fetch_json(f"{server_url.rstrip('/')}/health", timeout=args.http_timeout)
    checks.append(check_health(health, server_url))
    listener = inspect_port_listener(port)
    checks.append(check_port_listener(listener, port))
    launch = inspect_launch_agent(args.launch_agent_label)
    pid_state = inspect_pid_file(home / ".engram/engram.pid")
    checks.append(check_supervisor_parity(health, listener, launch, pid_state))
    checks.append(check_project_mcp(repo_root / ".mcp.json", mcp_url))
    checks.append(check_storage_path(env, home))
    checks.append(check_engramctl_status(timeout=args.command_timeout))
    checks.append(check_engramctl_storage(timeout=args.storage_timeout))
    if args.skip_slow or args.skip_doctor:
        checks.append(Check("engramctl doctor", "skip", "Skipped by flag."))
    else:
        checks.append(check_engramctl_doctor(timeout=args.doctor_timeout))
    if args.skip_slow or args.skip_mcp_live:
        checks.append(Check("MCP live tool catalog", "skip", "Skipped by flag."))
    else:
        checks.append(check_mcp_catalog(repo_root, mcp_url, timeout=args.mcp_timeout))
    checks.append(check_codex_config(home, mcp_url))
    checks.append(check_claude_code_config(mcp_url, timeout=args.claude_timeout))
    checks.append(check_axi_hooks_and_traces(home, repo_root))
    checks.append(check_openclaw(home, mcp_url, require_openclaw=args.require_openclaw))

    report = {"context": context, "checks": [asdict(check) for check in checks]}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_text_report(report)

    return 1 if any(check.status == "fail" for check in checks) else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the local Engram dogfood startup/adoption state.",
    )
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--home", default=str(Path.home()))
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--server-url")
    parser.add_argument("--launch-agent-label", default="dev.engram.local")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument("--skip-doctor", action="store_true")
    parser.add_argument("--skip-mcp-live", action="store_true")
    parser.add_argument("--require-openclaw", action="store_true")
    parser.add_argument("--http-timeout", type=float, default=3.0)
    parser.add_argument("--command-timeout", type=float, default=25.0)
    parser.add_argument("--storage-timeout", type=float, default=45.0)
    parser.add_argument("--doctor-timeout", type=float, default=180.0)
    parser.add_argument("--mcp-timeout", type=float, default=90.0)
    parser.add_argument("--claude-timeout", type=float, default=60.0)
    return parser.parse_args(argv)


def parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8", errors="replace"))


def parse_env_text(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        env[key] = value
    return env


def check_local_config(env_path: Path, env: dict[str, str]) -> Check:
    if not env_path.exists():
        return Check(
            "local native config",
            "fail",
            f"Missing local config at {env_path}.",
            next_actions=["Run: engramctl quickstart --mode helix"],
        )

    mode = env.get("ENGRAM_MODE", "")
    transport = env.get("ENGRAM_HELIX__TRANSPORT", "")
    data_dir = env.get("ENGRAM_HELIX__DATA_DIR", "")
    variant = env.get("ENGRAM_INSTALL_VARIANT", "")
    evidence = {
        "path": str(env_path),
        "install_variant": variant,
        "mode": mode,
        "helix_transport": transport,
        "helix_data_dir": data_dir,
        "api_port": env.get("ENGRAM_API_PORT", "8100"),
    }
    if mode == "helix" and transport == "native":
        return Check("local native config", "pass", "Native PyO3 Helix is configured.", evidence)
    if mode == "helix":
        return Check(
            "local native config",
            "warn",
            "Helix mode is configured, but the native transport is not explicit.",
            evidence,
            ["Run: engramctl setup --mode helix"],
        )
    return Check(
        "local native config",
        "fail",
        f"Expected ENGRAM_MODE=helix, found {mode or 'unset'}.",
        evidence,
        ["Run: engramctl quickstart --mode helix"],
    )


def fetch_json(url: str, *, timeout: float) -> dict[str, Any]:
    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8", errors="replace")
        data = json.loads(payload)
        return {"ok": True, "data": data, "url": url}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def check_health(result: dict[str, Any], server_url: str) -> Check:
    if result.get("ok") and (result.get("data") or {}).get("status") == "healthy":
        data = result["data"]
        return Check(
            "HTTP health",
            "pass",
            f"{server_url} is healthy in {data.get('mode', 'unknown')} mode.",
            {"payload": data},
        )
    if result.get("ok"):
        return Check(
            "HTTP health",
            "warn",
            f"{server_url} responded but did not report healthy.",
            {"payload": result.get("data")},
        )
    return Check(
        "HTTP health",
        "fail",
        f"{server_url} did not respond to /health.",
        {"error": result.get("error")},
        ["Check: engramctl status", "Check logs: tail -f ~/.engram/logs/engram.log"],
    )


def inspect_port_listener(port: int) -> dict[str, Any]:
    if not shutil.which("lsof"):
        return {"available": False, "rows": [], "error": "lsof not found"}
    result = run_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], timeout=5)
    rows = [line for line in strip_ansi(result.stdout).splitlines() if line.strip()]
    return {
        "available": True,
        "returncode": result.returncode,
        "rows": rows,
        "command": result.command,
        "timed_out": result.timed_out,
    }


def check_port_listener(listener: dict[str, Any], port: int) -> Check:
    if not listener.get("available"):
        return Check("port listener", "skip", "lsof is not available.", {"port": port})
    rows = listener.get("rows") or []
    data_rows = rows[1:] if len(rows) > 1 and rows[0].startswith("COMMAND") else rows
    if data_rows:
        return Check(
            "port listener",
            "pass",
            f"Port {port} has a listener.",
            {"rows": data_rows[:3]},
        )
    return Check(
        "port listener",
        "fail",
        f"No process is listening on port {port}.",
        {"rows": rows},
        ["Run: engramctl start", f"Check: lsof -nP -iTCP:{port} -sTCP:LISTEN"],
    )


def inspect_launch_agent(label: str) -> dict[str, Any]:
    if sys.platform != "darwin":
        return {"platform": sys.platform, "available": False, "label": label}
    if not shutil.which("launchctl"):
        return {"platform": sys.platform, "available": False, "label": label}
    command = ["launchctl", "print", f"gui/{os.getuid()}/{label}"]
    result = run_command(command, timeout=10)
    text = strip_ansi(result.stdout)
    return {
        "available": True,
        "label": label,
        "returncode": result.returncode,
        "running": "state = running" in text,
        "pid": extract_regex_int(text, r"\bpid = (\d+)"),
        "path": extract_regex_text(text, r"\bpath = (.+)"),
        "excerpt": "\n".join(text.splitlines()[:16]),
    }


def inspect_pid_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "alive": False}
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    try:
        pid = int(raw)
    except ValueError:
        return {"path": str(path), "exists": True, "alive": False, "raw": raw}
    return {"path": str(path), "exists": True, "pid": pid, "alive": process_exists(pid)}


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_supervisor_parity(
    health: dict[str, Any],
    listener: dict[str, Any],
    launch: dict[str, Any],
    pid_state: dict[str, Any],
) -> Check:
    return classify_supervisor_state(
        health_ok=bool(health.get("ok") and (health.get("data") or {}).get("status") == "healthy"),
        listener_rows=listener.get("rows") or [],
        launch_running=bool(launch.get("running")),
        launch_pid=launch.get("pid"),
        launch_available=bool(launch.get("available")),
        pid_exists=bool(pid_state.get("exists")),
        pid_alive=bool(pid_state.get("alive")),
        pid=pid_state.get("pid"),
        evidence={"launch_agent": launch, "pid_file": pid_state},
    )


def classify_supervisor_state(
    *,
    health_ok: bool,
    listener_rows: list[str],
    launch_running: bool,
    launch_pid: int | None,
    launch_available: bool,
    pid_exists: bool,
    pid_alive: bool,
    pid: int | None,
    evidence: dict[str, Any] | None = None,
) -> Check:
    data_rows = (
        listener_rows[1:]
        if listener_rows and listener_rows[0].startswith("COMMAND")
        else listener_rows
    )
    listener_present = bool(data_rows)
    evidence = evidence or {}
    evidence.update(
        {
            "health_ok": health_ok,
            "listener_present": listener_present,
            "launch_running": launch_running,
            "launch_pid": launch_pid,
            "pid_exists": pid_exists,
            "pid_alive": pid_alive,
            "pid": pid,
        }
    )

    if not health_ok and (listener_present or launch_running or pid_alive):
        return Check(
            "supervisor parity",
            "fail",
            "A process/supervisor is present, but HTTP health is not ready.",
            evidence,
            [
                "Inspect logs: tail -f ~/.engram/logs/engram.log",
                "Inspect listener: lsof -nP -iTCP:8100 -sTCP:LISTEN",
            ],
        )
    if not health_ok and pid_exists and not pid_alive:
        return Check(
            "supervisor parity",
            "fail",
            "A stale PID file exists while the API is offline.",
            evidence,
            ["Remove the stale PID only after confirming no Engram process owns the port."],
        )
    if health_ok and pid_alive:
        return Check("supervisor parity", "pass", "Health and PID file agree.", evidence)
    if health_ok and pid_exists and not pid_alive:
        return Check(
            "supervisor parity",
            "warn",
            "Health is green, but a stale PID file is present.",
            evidence,
            ["Remove the stale PID only after confirming the active supervisor owns the runtime."],
        )
    if health_ok and launch_running and not pid_alive:
        return Check(
            "supervisor parity",
            "pass",
            "Health and LaunchAgent supervisor agree.",
            evidence,
        )
    if health_ok and listener_present:
        return Check(
            "supervisor parity",
            "warn",
            "Health is green, but no known PID file or LaunchAgent owns the listener.",
            evidence,
            ["Inspect listener ownership before using engramctl stop."],
        )
    if not launch_available and not pid_exists:
        return Check(
            "supervisor parity",
            "warn",
            "No local supervisor evidence was available.",
            evidence,
        )
    return Check("supervisor parity", "pass", "No supervisor drift detected.", evidence)


def check_project_mcp(path: Path, expected_url: str) -> Check:
    if not path.exists():
        return Check(
            "project .mcp.json",
            "warn",
            f"Missing project MCP config at {path}.",
            next_actions=[f"Run: engramctl connect claude-code --project {path.parent}"],
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return Check("project .mcp.json", "fail", f"Invalid JSON: {exc}", {"path": str(path)})
    server = (data.get("mcpServers") or {}).get("engram") or {}
    url = server.get("url")
    evidence = {"path": str(path), "server": server}
    if url == expected_url:
        return Check(
            "project .mcp.json",
            "pass",
            "Project MCP config points at the local Engram MCP endpoint.",
            evidence,
        )
    return Check(
        "project .mcp.json",
        "fail",
        f"Expected {expected_url}, found {url or 'missing'}.",
        evidence,
        [f"Run: engramctl connect claude-code --project {path.parent}"],
    )


def check_storage_path(env: dict[str, str], home: Path) -> Check:
    mode = env.get("ENGRAM_MODE", "lite")
    if mode == "helix":
        path = Path(env.get("ENGRAM_HELIX__DATA_DIR") or home / ".helix/engram-native")
        label = "Helix native data"
    else:
        path = Path(env.get("ENGRAM_SQLITE__PATH") or home / ".engram/engram.db")
        label = "SQLite data"
    evidence = {"mode": mode, "path": str(path), "exists": path.exists()}
    if path.exists():
        evidence["size_bytes"] = path_size_bytes(path)
        return Check("storage path", "pass", f"{label} exists.", evidence)
    return Check(
        "storage path",
        "warn",
        f"{label} path does not exist yet.",
        evidence,
        ["Run: engramctl storage"],
    )


def check_engramctl_status(*, timeout: float) -> Check:
    if not shutil.which("engramctl"):
        return Check("engramctl status", "fail", "engramctl is not on PATH.")
    result = run_command(["engramctl", "status"], timeout=timeout)
    clean = strip_ansi(result.stdout)
    evidence = command_evidence(result, clean)
    if result.returncode == 0 and ("Server: running" in clean or '"status":"healthy"' in clean):
        return Check("engramctl status", "pass", "engramctl reports a running runtime.", evidence)
    return Check(
        "engramctl status",
        "fail",
        "engramctl status did not confirm a running runtime.",
        evidence,
        ["Run: engramctl status", "Check logs: tail -f ~/.engram/logs/engram.log"],
    )


def check_engramctl_storage(*, timeout: float) -> Check:
    if not shutil.which("engramctl"):
        return Check("engramctl storage", "fail", "engramctl is not on PATH.")
    result = run_command(["engramctl", "storage"], timeout=timeout)
    clean = strip_ansi(result.stdout)
    evidence = command_evidence(result, clean)
    if result.returncode == 0 and "Engram Storage" in clean and "Paths:" in clean:
        return Check("engramctl storage", "pass", "Storage paths and counts are visible.", evidence)
    if result.returncode == 0 and "Engram Storage (offline)" in clean:
        return Check(
            "engramctl storage",
            "warn",
            "Storage paths are visible only through offline fallback.",
            evidence,
        )
    return Check(
        "engramctl storage",
        "fail",
        "engramctl storage did not produce storage visibility.",
        evidence,
        ["Run: engramctl storage", "Check /api/storage after the runtime is healthy."],
    )


def check_engramctl_doctor(*, timeout: float) -> Check:
    if not shutil.which("engramctl"):
        return Check("engramctl doctor", "fail", "engramctl is not on PATH.")
    result = run_command(["engramctl", "doctor", "--format", "json"], timeout=timeout)
    clean = strip_ansi(result.stdout)
    evidence = command_evidence(result, clean)
    payload = extract_last_json(clean)
    if payload is not None:
        evidence["json"] = payload
    if result.returncode == 0:
        return Check("engramctl doctor", "pass", "Doctor completed successfully.", evidence)
    if result.timed_out:
        detail = "Doctor timed out; native warmup or storage checks may still be running."
    else:
        detail = "Doctor failed."
    return Check(
        "engramctl doctor", "fail", detail, evidence, ["Run: engramctl doctor --format json"]
    )


def check_mcp_catalog(repo_root: Path, mcp_url: str, *, timeout: float) -> Check:
    runner = ["uv", "run", "python"] if shutil.which("uv") else [sys.executable]
    cwd = repo_root / "server"
    if runner[0] == "uv" and not cwd.exists():
        return Check("MCP live tool catalog", "skip", "server/ is missing; cannot use uv run.")
    snippet = """
import asyncio
import json
import sys
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

url = sys.argv[1]
expected = set(json.loads(sys.argv[2]))
project_path = sys.argv[3]

async def main():
    async with streamablehttp_client(
        url,
        timeout=30,
        sse_read_timeout=30,
        terminate_on_close=False,
    ) as (read, write, get_session_id):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            names = sorted(tool.name for tool in result.tools)
            tool_payloads = {tool.name: tool.model_dump() for tool in result.tools}
            recall_schema = (
                tool_payloads.get("recall", {})
                .get("inputSchema", {})
                .get("properties", {})
            )
            recall_has_project_path = "project_path" in recall_schema
            context_probe = None
            recall_probe = None
            if recall_has_project_path:
                context_result = await session.call_tool(
                    "get_context",
                    {
                        "topic_hint": (
                            "Engram dogfood startup validation project_path "
                            "schema probe"
                        ),
                        "project_path": project_path,
                        "max_tokens": 1000,
                        "format": "structured",
                    },
                )
                context_text = (
                    context_result.content[0].text if context_result.content else "{}"
                )
                context_payload = json.loads(context_text)
                context_budget = context_payload.get("budget") or {}
                context_lifecycle = context_payload.get("lifecycle") or {}
                context_packets = (
                    context_payload.get("cached_packets")
                    or (context_payload.get("recalled_context") or {}).get("packets")
                    or []
                )
                context_probe = {
                    "status": context_payload.get("status"),
                    "budget_miss": context_budget.get("budget_miss"),
                    "degraded": (
                        context_lifecycle.get("degraded")
                        or context_budget.get("degraded")
                    ),
                    "timeout": (
                        context_lifecycle.get("timeout")
                        or context_budget.get("timeout")
                    ),
                    "skip_reason": (
                        context_lifecycle.get("skip_reason")
                        or context_budget.get("skip_reason")
                    ),
                    "packet_count": len(context_packets),
                }
                probe_result = await session.call_tool(
                    "recall",
                    {
                        "query": (
                            "Engram dogfood startup validation project_path "
                            "schema probe"
                        ),
                        "project_path": project_path,
                        "limit": 3,
                    },
                )
                text = probe_result.content[0].text if probe_result.content else "{}"
                probe_payload = json.loads(text)
                budget = probe_payload.get("budget") or {}
                lifecycle = probe_payload.get("lifecycle") or {}
                recall_probe = {
                    "status": probe_payload.get("status"),
                    "query_time_ms": probe_payload.get("query_time_ms"),
                    "budget_miss": budget.get("budget_miss"),
                    "degraded": lifecycle.get("degraded") or budget.get("degraded"),
                    "timeout": lifecycle.get("timeout") or budget.get("timeout"),
                    "skip_reason": lifecycle.get("skip_reason") or budget.get("skip_reason"),
                    "fallback_status": lifecycle.get("fallback_status"),
                    "packet_count": len(probe_payload.get("packets") or []),
                    "result_count": len(probe_payload.get("results") or []),
                }
            print(json.dumps({
                "count": len(names),
                "has_remember": "remember" in names,
                "missing": sorted(expected.difference(names)),
                "names": names,
                "recall_has_project_path": recall_has_project_path,
                "context_probe": context_probe,
                "recall_probe": recall_probe,
            }))

asyncio.run(main())
""".strip()
    result = run_command(
        [
            *runner,
            "-c",
            snippet,
            mcp_url,
            json.dumps(sorted(EXPECTED_MCP_TOOLS)),
            str(repo_root),
        ],
        timeout=timeout,
        cwd=cwd if runner[0] == "uv" else repo_root,
    )
    clean = strip_ansi(result.stdout)
    evidence = command_evidence(result, clean)
    payload = extract_last_json(clean)
    if payload:
        evidence["catalog"] = {
            "count": payload.get("count"),
            "has_remember": payload.get("has_remember"),
            "missing": payload.get("missing"),
            "recall_has_project_path": payload.get("recall_has_project_path"),
            "context_probe": payload.get("context_probe"),
            "recall_probe": payload.get("recall_probe"),
        }
    context_probe = payload.get("context_probe") if isinstance(payload, dict) else None
    context_probe_ok = (
        isinstance(context_probe, dict)
        and context_probe.get("status") == "ok"
        and not context_probe.get("budget_miss")
        and not context_probe.get("degraded")
        and not context_probe.get("timeout")
        and int(context_probe.get("packet_count") or 0) > 0
    )
    recall_probe = payload.get("recall_probe") if isinstance(payload, dict) else None
    recall_probe_ok = (
        isinstance(recall_probe, dict)
        and recall_probe.get("status") == "ok"
        and not recall_probe.get("budget_miss")
        and not recall_probe.get("degraded")
        and not recall_probe.get("timeout")
    )
    if (
        result.returncode == 0
        and payload
        and not payload.get("missing")
        and payload.get("recall_has_project_path") is True
        and context_probe_ok
        and recall_probe_ok
    ):
        return Check(
            "MCP live tool catalog",
            "pass",
            (
                "Live MCP catalog exposes expected tools, including remember, "
                "and recall accepts project_path."
            ),
            evidence,
        )
    if result.timed_out:
        detail = "MCP catalog probe timed out."
    elif payload:
        missing = ", ".join(payload.get("missing") or [])
        if missing:
            detail = f"MCP catalog is missing: {missing}."
        elif payload.get("recall_has_project_path") is not True:
            detail = "MCP recall schema is missing project_path."
        elif not context_probe_ok:
            detail = "MCP get_context(project_path=...) probe was degraded or failed."
        elif not recall_probe_ok:
            detail = "MCP recall(project_path=...) probe was degraded or failed."
        else:
            detail = "MCP catalog probe returned unexpected payload."
    else:
        detail = "MCP catalog probe failed before returning JSON."
    return Check(
        "MCP live tool catalog",
        "fail",
        detail,
        evidence,
        [f"Probe manually with an MCP client against {mcp_url}"],
    )


def check_codex_config(home: Path, expected_mcp_url: str) -> Check:
    path = home / ".codex/config.toml"
    if not path.exists():
        return Check(
            "Codex MCP config",
            "warn",
            f"Missing {path}.",
            next_actions=["Run: engramctl connect codex"],
        )
    text = path.read_text(encoding="utf-8", errors="replace")
    has_remote = re.search(r"(?m)^remote_mcp_client_enabled\s*=\s*true\b", text) is not None
    has_server = re.search(r"(?m)^\[mcp_servers\.engram\]\s*$", text) is not None
    url_match = re.search(
        r'(?m)^url\s*=\s*"([^"]+)"\s*$', section_text(text, "[mcp_servers.engram]")
    )
    url = url_match.group(1) if url_match else None
    evidence = {"path": str(path), "remote_mcp_client_enabled": has_remote, "url": url}
    if has_remote and has_server and url == expected_mcp_url:
        return Check(
            "Codex MCP config", "pass", "Codex global MCP config points at Engram.", evidence
        )
    return Check(
        "Codex MCP config",
        "fail",
        "Codex MCP config is missing the Engram HTTP endpoint or remote MCP enablement.",
        evidence,
        ["Run: engramctl connect codex"],
    )


def check_claude_code_config(expected_mcp_url: str, *, timeout: float) -> Check:
    if not shutil.which("claude"):
        return Check("Claude Code MCP config", "skip", "Claude Code CLI is not on PATH.")
    result_list = run_command(["claude", "mcp", "list"], timeout=timeout)
    result_get = run_command(["claude", "mcp", "get", "engram"], timeout=timeout)
    combined = strip_ansi(result_list.stdout + "\n" + result_get.stdout)
    evidence = {
        "list": command_evidence(result_list, strip_ansi(result_list.stdout)),
        "get": command_evidence(result_get, strip_ansi(result_get.stdout)),
        "connected": "Status: \u2713 Connected" in combined or " - \u2713 Connected" in combined,
        "project_scope": "Scope: Project config" in combined,
        "url": expected_mcp_url,
        "conflicting_scopes": "[Conflicting scopes]" in combined,
    }
    if (
        result_get.returncode == 0
        and evidence["connected"]
        and evidence["project_scope"]
        and not evidence["conflicting_scopes"]
    ):
        return Check(
            "Claude Code MCP config",
            "pass",
            "Claude Code sees one connected project-scoped Engram server.",
            evidence,
        )
    return Check(
        "Claude Code MCP config",
        "fail",
        "Claude Code did not confirm a clean connected project-scoped Engram MCP config.",
        evidence,
        [
            "Run: claude mcp get engram",
            "If scopes conflict, remove stale user scope with: claude mcp remove engram -s user",
        ],
    )


def check_axi_hooks_and_traces(home: Path, repo_root: Path) -> Check:
    codex_hook = load_hook_file(home / ".codex/hooks.json", client="codex")
    claude_hook = load_hook_file(home / ".claude/settings.json", client="claude-code")
    trace_path = home / TRACE_FILE_RELATIVE
    traces = read_jsonl(trace_path)
    trace_summary = summarize_axi_traces(traces, repo_root=repo_root)
    evidence = {
        "codex_hook": codex_hook,
        "claude_code_hook": claude_hook,
        "trace_file": str(trace_path),
        "trace_summary": trace_summary,
    }
    issues = []
    next_actions = []
    for client, hook in (("codex", codex_hook), ("claude-code", claude_hook)):
        if not hook.get("installed"):
            issues.append(f"{client} hook is not installed")
            next_actions.append(f"Run: engramctl connect {client} --axi")
        elif not hook.get("read_only") or hook.get("capture"):
            issues.append(f"{client} hook is not read-only")
            next_actions.append(f"Run: engram axi hooks install {client}")
    for client, summary in trace_summary.items():
        session_start = summary.get("session_start")
        if not session_start:
            issues.append(f"{client} has no session-start trace")
            next_actions.append(
                f"Restart {client} or start a new session after installing the hook."
            )
        else:
            hook = codex_hook if client == "codex" else claude_hook
            freshness_issue = startup_trace_freshness_issue(client, hook, session_start)
            if freshness_issue:
                issues.append(freshness_issue)
                next_actions.append(_session_trace_refresh_action(client, trace_path))
            startup_issue = startup_project_issue(client, session_start)
            if startup_issue:
                issues.append(startup_issue)
                next_actions.append(_session_trace_refresh_action(client, trace_path))
        if not summary.get("followup"):
            issues.append(f"{client} has no follow-up context/recall trace")
            next_actions.append(
                "Run from that client: " + _followup_trace_command(client, trace_path)
            )
    if issues:
        return Check(
            "AXI hooks and traces", "warn", "; ".join(issues), evidence, dedupe(next_actions)
        )
    latest_projects = [
        f"{client}: {summary['session_start'].get('project')}"
        for client, summary in trace_summary.items()
        if summary.get("session_start") and summary["session_start"].get("project")
    ]
    detail = "Codex and Claude Code have read-only AXI hooks plus startup/follow-up evidence."
    if latest_projects:
        detail += " Latest startup projects: " + "; ".join(latest_projects) + "."
    return Check("AXI hooks and traces", "pass", detail, evidence)


def _followup_trace_command(client: str, trace_path: Path) -> str:
    """Return the exact AXI follow-up command that refreshes trace evidence."""
    return (
        'engram axi context --project "$PWD" '
        f"--trace-file {shlex.quote(str(trace_path))} "
        f"--trace-client {shlex.quote(client)} "
        "--trace-origin agent-followup"
    )


def _session_trace_refresh_action(client: str, trace_path: Path) -> str:
    """Return the operator action for refreshing startup-adjacent trace evidence."""
    del trace_path
    return (
        f"Start a new interactive {client} session from the target project after "
        "installing the current hook-run command. Manual agent-followup traces do "
        "not refresh SessionStart evidence."
    )


def startup_trace_freshness_issue(
    client: str,
    hook: dict[str, Any],
    session_start: dict[str, Any],
) -> str | None:
    hook_modified = hook.get("modified_epoch")
    trace_time = parse_trace_timestamp_epoch(session_start.get("timestamp"))
    if not isinstance(hook_modified, (int, float)) or trace_time is None:
        return None
    if float(hook_modified) > trace_time + 1:
        return f"{client} session-start trace predates current hook config"
    return None


def startup_project_issue(client: str, session_start: dict[str, Any]) -> str | None:
    project = str(session_start.get("project") or "").strip()
    if not project:
        return f"{client} session-start trace has no project path"
    path = Path(project).expanduser()
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if resolved.parent == resolved:
        return f"{client} session-start project is filesystem root ({project})"
    return None


def parse_trace_timestamp_epoch(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def load_hook_file(path: Path, *, client: str) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "installed": False, "reason": "missing file"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"path": str(path), "installed": False, "reason": f"invalid JSON: {exc}"}
    hook = find_managed_hook(data)
    if not hook:
        return {"path": str(path), "installed": False, "reason": "managed hook not found"}
    command = hook.get("command", "")
    try:
        modified_epoch = path.stat().st_mtime
    except OSError:
        modified_epoch = None
    return {
        "path": str(path),
        "client": client,
        "installed": True,
        "read_only": hook.get("read_only") is True,
        "capture": hook.get("capture") is True,
        "trace_file": hook.get("trace_file"),
        "has_session_origin": "--trace-origin session-start-hook" in command,
        "has_trace_client": f"--trace-client {client}" in command,
        "modified_epoch": modified_epoch,
    }


def find_managed_hook(data: dict[str, Any]) -> dict[str, Any] | None:
    for event_entries in (data.get("hooks") or {}).values():
        if not isinstance(event_entries, list):
            continue
        for event_entry in event_entries:
            for hook in event_entry.get("hooks") or []:
                if hook.get("id") == MANAGED_AXI_HOOK_ID:
                    return hook
    return None


def summarize_axi_traces(
    records: list[dict[str, Any]], *, repo_root: Path
) -> dict[str, dict[str, Any]]:
    del repo_root
    summary: dict[str, dict[str, Any]] = {}
    for client in ("codex", "claude-code"):
        client_records = [record for record in records if record.get("client") == client]
        session = latest_record(
            client_records,
            origin="session-start-hook",
            operations={"home", "hook-run"},
            acceptable_statuses={"healthy", "ok", "degraded"},
        )
        followup = latest_record(
            client_records,
            origin="agent-followup",
            operations={"context", "recall"},
            acceptable_statuses={"ok", "healthy"},
        )
        summary[client] = {
            "session_start": compact_trace_record(session),
            "followup": compact_trace_record(followup),
        }
    return summary


def latest_record(
    records: list[dict[str, Any]],
    *,
    origin: str,
    operations: set[str],
    acceptable_statuses: set[str],
) -> dict[str, Any] | None:
    matches = [
        record
        for record in records
        if record.get("origin") == origin
        and record.get("operation") in operations
        and record.get("status") in acceptable_statuses
    ]
    return matches[-1] if matches else None


def compact_trace_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if not record:
        return None
    compact = {
        "timestamp": record.get("timestamp"),
        "operation": record.get("operation"),
        "origin": record.get("origin"),
        "project": record.get("project"),
        "status": record.get("status"),
        "durationMs": record.get("durationMs"),
        "exitCode": record.get("exitCode"),
    }
    for key in (
        "cacheHit",
        "packetCount",
        "resultCount",
        "fallbackStatus",
        "skipReason",
        "budgetMiss",
        "degraded",
    ):
        if key in record:
            compact[key] = record.get(key)
    return compact


def check_openclaw(home: Path, expected_mcp_url: str, *, require_openclaw: bool) -> Check:
    skill_path = home / ".openclaw/skills/engram-brain/SKILL.md"
    evidence: dict[str, Any] = {
        "skill_path": str(skill_path),
        "skill_installed": skill_path.exists(),
    }
    manual_command = openclaw_mcp_set_command(expected_mcp_url)
    openclaw_command = resolve_openclaw_command()
    evidence["command_prefix"] = openclaw_command
    if not openclaw_command:
        status = "fail" if require_openclaw else "warn"
        return Check(
            "OpenClaw MCP config",
            status,
            (
                "OpenClaw CLI is not installed and npx is unavailable; skill files "
                "can be checked but MCP config cannot be verified."
            ),
            evidence,
            [
                "Install OpenClaw or npm/npx, then run:",
                manual_command,
            ],
        )
    result = run_command([*openclaw_command, "mcp", "show", "engram", "--json"], timeout=60)
    clean = strip_ansi(result.stdout)
    evidence["command"] = command_evidence(result, clean)
    payload = extract_last_json(clean)
    evidence["payload"] = payload
    if (
        result.returncode == 0
        and payload
        and payload.get("url") == expected_mcp_url
        and payload.get("transport") == EXPECTED_OPENCLAW_TRANSPORT
    ):
        return Check(
            "OpenClaw MCP config", "pass", "OpenClaw MCP config points at Engram.", evidence
        )
    if result.returncode == 0 and payload and payload.get("url") == expected_mcp_url:
        return Check(
            "OpenClaw MCP config",
            "fail",
            "OpenClaw MCP config points at Engram but is missing the streamable-http transport.",
            evidence,
            [f"Run: {manual_command}"],
        )
    return Check(
        "OpenClaw MCP config",
        "fail",
        "OpenClaw CLI did not confirm the Engram MCP config.",
        evidence,
        [f"Run: {manual_command}"],
    )


def openclaw_mcp_set_command(expected_mcp_url: str) -> str:
    payload = json.dumps(
        {"url": expected_mcp_url, "transport": "streamable-http"},
        separators=(",", ":"),
    )
    prefix = " ".join(resolve_openclaw_command() or ["npx", "-y", "openclaw"])
    return f"{prefix} mcp set engram '{payload}'"


def resolve_openclaw_command() -> list[str] | None:
    configured = os.environ.get("ENGRAM_OPENCLAW_COMMAND", "").strip()
    if configured:
        return shlex.split(configured)
    if shutil.which("openclaw"):
        return ["openclaw"]
    if os.environ.get("ENGRAM_OPENCLAW_DISABLE_NPX") != "1" and shutil.which("npx"):
        return ["npx", "-y", "openclaw"]
    return None


def run_command(cmd: list[str], *, timeout: float, cwd: Path | None = None) -> CommandResult:
    started = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=cmd,
            returncode=result.returncode,
            stdout=result.stdout,
            duration_seconds=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        return CommandResult(
            command=cmd,
            returncode=None,
            stdout=stdout,
            timed_out=True,
            duration_seconds=time.monotonic() - started,
        )


def command_evidence(result: CommandResult, output: str) -> dict[str, Any]:
    return {
        "command": " ".join(result.command),
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "duration_seconds": round(result.duration_seconds, 3),
        "output_excerpt": excerpt(output),
    }


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def excerpt(text: str, *, max_lines: int = 30, max_chars: int = 5000) -> str:
    lines = text.strip().splitlines()
    if len(lines) > max_lines:
        lines = [*lines[: max_lines // 2], "...", *lines[-max_lines // 2 :]]
    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[: max_chars - 20] + "\n...<truncated>"
    return out


def extract_last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            return payload
    decoder = json.JSONDecoder()
    for match in reversed(list(re.finditer(r"{", text))):
        try:
            payload, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def extract_regex_int(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def extract_regex_text(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def path_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def section_text(text: str, header: str) -> str:
    pattern = re.compile(rf"(?ms)^{re.escape(header)}\s*\n(.*?)(?=^\[|\Z)")
    match = pattern.search(text)
    return match.group(1) if match else ""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def print_text_report(report: dict[str, Any]) -> None:
    context = report["context"]
    checks = [Check(**check) for check in report["checks"]]
    print("Engram Dogfood Startup Validation")
    print(f"repo:   {context['repo_root']}")
    print(f"api:    {context['server_url']}")
    print(f"mcp:    {context['mcp_url']}")
    print(f"env:    {context['env_path']}")
    print("")
    width = max(len(check.name) for check in checks) if checks else 0
    for check in checks:
        print(f"{check.status.upper():<5} {check.name:<{width}}  {check.detail}")
    print("")
    counts = {
        status: sum(1 for check in checks if check.status == status)
        for status in ("pass", "warn", "fail", "skip")
    }
    summary = (
        f"{counts['pass']} pass, {counts['warn']} warn, "
        f"{counts['fail']} fail, {counts['skip']} skip"
    )
    print(f"Summary: {summary}")
    next_actions = [
        f"{check.name}: {action}"
        for check in checks
        for action in check.next_actions
        if check.status in {"warn", "fail"}
    ]
    if next_actions:
        print("")
        print("Next actions:")
        for action in dedupe(next_actions):
            print(f"- {action}")


if __name__ == "__main__":
    raise SystemExit(main())
