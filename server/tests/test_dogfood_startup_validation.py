from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dogfood_startup_validation.py"
MATRIX_SCRIPT = ROOT / "scripts/dogfood_startup_matrix.py"


def _load_runner():
    return _load_module("dogfood_startup_validation", SCRIPT)


def _load_matrix():
    return _load_module("dogfood_startup_matrix", MATRIX_SCRIPT)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_env_text_handles_exports_and_quotes() -> None:
    runner = _load_runner()

    parsed = runner.parse_env_text(
        """
        # comment
        export ENGRAM_MODE=helix
        ENGRAM_HELIX__TRANSPORT="native"
        ENGRAM_HELIX__DATA_DIR='/tmp/engram native'
        """
    )

    assert parsed["ENGRAM_MODE"] == "helix"
    assert parsed["ENGRAM_HELIX__TRANSPORT"] == "native"
    assert parsed["ENGRAM_HELIX__DATA_DIR"] == "/tmp/engram native"


def test_supervisor_accepts_launchagent_as_runtime_owner() -> None:
    runner = _load_runner()

    check = runner.classify_supervisor_state(
        health_ok=True,
        listener_rows=["COMMAND PID USER", "python 123 user"],
        launch_running=True,
        launch_pid=123,
        launch_available=True,
        pid_exists=False,
        pid_alive=False,
        pid=None,
        evidence={},
    )

    assert check.status == "pass"
    assert "LaunchAgent" in check.detail


def test_supervisor_classifies_half_started_listener_as_failure() -> None:
    runner = _load_runner()

    check = runner.classify_supervisor_state(
        health_ok=False,
        listener_rows=["COMMAND PID USER", "python 123 user"],
        launch_running=False,
        launch_pid=None,
        launch_available=True,
        pid_exists=False,
        pid_alive=False,
        pid=None,
        evidence={},
    )

    assert check.status == "fail"
    assert "health is not ready" in check.detail


def test_find_managed_hook_and_trace_summary() -> None:
    runner = _load_runner()

    hook = runner.find_managed_hook(
        {
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"id": "other", "command": "echo nope"}]},
                    {
                        "hooks": [
                            {
                                "id": "engram-axi-context",
                                "command": "engram axi --trace-client codex",
                                "read_only": True,
                            }
                        ]
                    },
                ]
            }
        }
    )
    assert hook["id"] == "engram-axi-context"

    traces = [
        {
            "client": "codex",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": str(ROOT),
            "timestamp": "2026-05-20T00:00:00Z",
        },
        {
            "client": "codex",
            "origin": "agent-followup",
            "operation": "context",
            "status": "ok",
            "project": str(ROOT),
            "timestamp": "2026-05-20T00:01:00Z",
        },
    ]

    summary = runner.summarize_axi_traces(traces, repo_root=ROOT)

    assert summary["codex"]["session_start"]["status"] == "healthy"
    assert summary["codex"]["followup"]["operation"] == "context"
    assert summary["claude-code"]["session_start"] is None


def test_trace_summary_prefers_fresh_hook_run_session_start() -> None:
    runner = _load_runner()

    traces = [
        {
            "client": "codex",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": "/",
            "timestamp": "2026-05-27T20:12:28Z",
        },
        {
            "client": "codex",
            "origin": "session-start-hook",
            "operation": "hook-run",
            "status": "healthy",
            "project": str(ROOT),
            "timestamp": "2026-05-28T00:24:34Z",
        },
    ]

    summary = runner.summarize_axi_traces(traces, repo_root=ROOT)

    assert summary["codex"]["session_start"]["operation"] == "hook-run"
    assert summary["codex"]["session_start"]["project"] == str(ROOT)


def test_axi_hook_check_warns_on_root_session_start_project(tmp_path: Path) -> None:
    runner = _load_runner()
    home = tmp_path / "home"
    trace_path = home / ".engram/axi-hook-runs.jsonl"

    _write_managed_axi_hook(home / ".codex/hooks.json", client="codex")
    _write_managed_axi_hook(home / ".claude/settings.json", client="claude-code")
    trace_path.parent.mkdir(parents=True)
    records = [
        {
            "client": "codex",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": "/",
            "timestamp": "2026-05-27T20:12:28Z",
        },
        {
            "client": "codex",
            "origin": "agent-followup",
            "operation": "context",
            "status": "ok",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:13:00Z",
        },
        {
            "client": "claude-code",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:12:28Z",
        },
        {
            "client": "claude-code",
            "origin": "agent-followup",
            "operation": "context",
            "status": "ok",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:13:00Z",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    check = runner.check_axi_hooks_and_traces(home, ROOT)

    assert check.status == "warn"
    assert "codex session-start project is filesystem root (/)" in check.detail
    assert "trace_summary" in check.evidence
    assert any(
        "Start a new interactive codex session" in action
        and "Manual agent-followup traces do not refresh SessionStart evidence" in action
        for action in check.next_actions
    )


def test_axi_hook_check_warns_when_trace_predates_hook_config(tmp_path: Path) -> None:
    runner = _load_runner()
    home = tmp_path / "home"
    trace_path = home / ".engram/axi-hook-runs.jsonl"
    codex_hook = home / ".codex/hooks.json"
    claude_hook = home / ".claude/settings.json"

    _write_managed_axi_hook(codex_hook, client="codex")
    _write_managed_axi_hook(claude_hook, client="claude-code")
    trace_path.parent.mkdir(parents=True)
    records = [
        {
            "client": "codex",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:12:28Z",
        },
        {
            "client": "codex",
            "origin": "agent-followup",
            "operation": "context",
            "status": "ok",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:13:00Z",
        },
        {
            "client": "claude-code",
            "origin": "session-start-hook",
            "operation": "home",
            "status": "healthy",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:12:28Z",
        },
        {
            "client": "claude-code",
            "origin": "agent-followup",
            "operation": "context",
            "status": "ok",
            "project": str(ROOT),
            "timestamp": "2026-05-27T20:13:00Z",
        },
    ]
    trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    new_config_time = 1_779_916_000
    os.utime(codex_hook, (new_config_time, new_config_time))
    os.utime(claude_hook, (new_config_time, new_config_time))

    check = runner.check_axi_hooks_and_traces(home, ROOT)

    assert check.status == "warn"
    assert "codex session-start trace predates current hook config" in check.detail
    assert "claude-code session-start trace predates current hook config" in check.detail
    assert any(
        "Start a new interactive codex session" in action
        and "Manual agent-followup traces do not refresh SessionStart evidence" in action
        for action in check.next_actions
    )


def _write_managed_axi_hook(path: Path, *, client: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "id": "engram-axi-context",
                                    "command": (
                                        "engram axi hook-run "
                                        f"--trace-client {client} "
                                        "--trace-origin session-start-hook"
                                    ),
                                    "read_only": True,
                                    "capture": False,
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )


def test_openclaw_command_uses_streamable_http_transport(monkeypatch) -> None:
    runner = _load_runner()
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("ENGRAM_OPENCLAW_COMMAND", raising=False)

    command = runner.openclaw_mcp_set_command("http://127.0.0.1:8100/mcp")

    assert command == (
        "npx -y openclaw mcp set engram "
        '\'{"url":"http://127.0.0.1:8100/mcp","transport":"streamable-http"}\''
    )


def test_resolve_openclaw_command_prefers_configured_command(monkeypatch) -> None:
    runner = _load_runner()

    monkeypatch.setenv("ENGRAM_OPENCLAW_COMMAND", "npx -y openclaw")

    assert runner.resolve_openclaw_command() == ["npx", "-y", "openclaw"]


def test_openclaw_check_requires_streamable_http_transport(tmp_path: Path, monkeypatch) -> None:
    runner = _load_runner()
    bin_dir = tmp_path / "bin"
    home = tmp_path / "home"
    skill_path = home / ".openclaw/skills/engram-brain/SKILL.md"
    bin_dir.mkdir()
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("# Engram Brain\n")
    fake_openclaw = bin_dir / "openclaw"
    payload = {"url": "http://127.0.0.1:8100/mcp", "transport": "streamable-http"}
    fake_openclaw.write_text(
        f"""#!/usr/bin/env bash
if [ "$1 $2 $3 $4" = "mcp show engram --json" ]; then
  printf '%s\\n' '{json.dumps(payload)}'
  exit 0
fi
exit 2
"""
    )
    fake_openclaw.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ.get('PATH', '')}")

    check = runner.check_openclaw(home, "http://127.0.0.1:8100/mcp", require_openclaw=True)

    assert check.status == "pass"

    fake_openclaw.write_text(
        """#!/usr/bin/env bash
if [ "$1 $2 $3 $4" = "mcp show engram --json" ]; then
  printf '%s\\n' '{"url":"http://127.0.0.1:8100/mcp","type":"http"}'
  exit 0
fi
exit 2
"""
    )

    check = runner.check_openclaw(home, "http://127.0.0.1:8100/mcp", require_openclaw=True)

    assert check.status == "fail"
    assert "streamable-http transport" in check.detail


def test_extract_last_json_handles_warning_before_pretty_json() -> None:
    runner = _load_runner()

    payload = runner.extract_last_json(
        """npm warn deprecated node-domexception@1.0.0
{
  "url": "http://127.0.0.1:8100/mcp",
  "transport": "streamable-http"
}
"""
    )

    assert payload == {
        "url": "http://127.0.0.1:8100/mcp",
        "transport": "streamable-http",
    }


def test_mcp_catalog_check_requires_project_path_schema(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)

    def fake_run_command(command, *, timeout, cwd=None, env=None):
        assert command[-1] == str(tmp_path)
        return runner.CommandResult(
            command=list(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "count": 7,
                    "has_remember": True,
                    "missing": [],
                    "names": sorted(runner.EXPECTED_MCP_TOOLS),
                    "recall_has_project_path": True,
                    "context_probe": {
                        "status": "ok",
                        "budget_miss": False,
                        "degraded": False,
                        "timeout": False,
                        "packet_count": 1,
                    },
                    "recall_probe": {
                        "status": "ok",
                        "query_time_ms": 4.5,
                        "budget_miss": False,
                        "degraded": False,
                        "timeout": False,
                        "packet_count": 1,
                        "result_count": 0,
                    },
                }
            ),
        )

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    check = runner.check_mcp_catalog(tmp_path, "http://127.0.0.1:8100/mcp", timeout=5)

    assert check.status == "pass"
    assert check.evidence["catalog"]["recall_has_project_path"] is True
    assert check.evidence["catalog"]["context_probe"]["status"] == "ok"
    assert check.evidence["catalog"]["recall_probe"]["status"] == "ok"


def test_mcp_catalog_check_fails_without_project_path_schema(monkeypatch, tmp_path: Path) -> None:
    runner = _load_runner()
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)

    def fake_run_command(command, *, timeout, cwd=None, env=None):
        return runner.CommandResult(
            command=list(command),
            returncode=0,
            stdout=json.dumps(
                {
                    "count": 7,
                    "has_remember": True,
                    "missing": [],
                    "names": sorted(runner.EXPECTED_MCP_TOOLS),
                    "recall_has_project_path": False,
                    "context_probe": None,
                    "recall_probe": None,
                }
            ),
        )

    monkeypatch.setattr(runner, "run_command", fake_run_command)

    check = runner.check_mcp_catalog(tmp_path, "http://127.0.0.1:8100/mcp", timeout=5)

    assert check.status == "fail"
    assert "missing project_path" in check.detail


def test_matrix_detects_stopped_validation_payload() -> None:
    matrix = _load_matrix()

    payload = {
        "checks": [
            {"name": "HTTP health", "status": "fail"},
            {"name": "engramctl status", "status": "fail"},
            {"name": "engramctl storage", "status": "warn"},
        ]
    }

    assert matrix.stopped_state_detected(payload) is True
    assert matrix.validation_summary(payload) == {
        "pass": 0,
        "warn": 1,
        "fail": 2,
        "skip": 0,
    }


def test_matrix_read_json_object_handles_doctor_preamble(tmp_path: Path) -> None:
    matrix = _load_matrix()
    path = tmp_path / "doctor.json"
    path.write_text(
        """
Verifying vector integrity after migration...
All vectors verified successfully!
{
  "status": "warn",
  "checks": [
    {"name": "lifecycle_snapshot", "status": "warn"},
    {"name": "server", "status": "pass"}
  ]
}
"""
    )

    payload = matrix.read_json_object(path)

    assert payload["status"] == "warn"
    assert matrix.validation_summary(payload) == {
        "pass": 1,
        "warn": 1,
        "fail": 0,
        "skip": 0,
    }


def test_matrix_doctor_step_preserves_doctor_warnings(tmp_path: Path) -> None:
    matrix = _load_matrix()
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    script = tmp_path / "doctor.sh"
    script.write_text(
        """#!/usr/bin/env bash
cat <<'JSON'
setup log before json
{
  "status": "warn",
  "checks": [
    {"name": "lifecycle_snapshot", "status": "warn"},
    {"name": "server", "status": "pass"}
  ]
}
JSON
"""
    )
    script.chmod(0o755)

    step = matrix.run_doctor_step(
        "doctor",
        [str(script)],
        tmp_path,
        evidence_dir,
        "doctor.json",
    )

    assert step.status == "warn"
    assert step.detail == "Doctor completed with warnings."
    assert step.evidence["summary"] == {
        "status": "warn",
        "pass": 1,
        "warn": 1,
        "fail": 0,
        "skip": 0,
    }


def test_matrix_validation_step_preserves_validation_warnings(tmp_path: Path) -> None:
    matrix = _load_matrix()
    repo = tmp_path / "repo"
    evidence_dir = tmp_path / "evidence"
    script = repo / "scripts/dogfood_startup_validation.py"
    script.parent.mkdir(parents=True)
    evidence_dir.mkdir()
    script.write_text(
        """#!/usr/bin/env python3
import json
print(json.dumps({
    "checks": [
        {"name": "AXI hooks and traces", "status": "warn"},
        {"name": "HTTP health", "status": "pass"},
    ]
}))
"""
    )

    step = matrix.run_validation_step(
        "validation",
        repo,
        evidence_dir,
        "validation.json",
    )

    assert step.status == "warn"
    assert step.detail == "Validation JSON contains warning checks."
    assert step.evidence["summary"] == {
        "pass": 1,
        "warn": 1,
        "fail": 0,
        "skip": 0,
    }


def test_matrix_renders_commands_and_summary() -> None:
    matrix = _load_matrix()

    report = {
        "context": {
            "generated_at": "2026-05-20T00:00:00Z",
            "repo": str(ROOT),
            "evidence_dir": "/tmp/evidence",
            "confirm_lifecycle": True,
        },
        "summary": {"pass": 1, "warn": 0, "fail": 0, "skip": 0},
        "steps": [
            {
                "name": "status",
                "status": "pass",
                "command": ["engramctl", "status"],
                "output_path": "/tmp/evidence/status.log",
                "detail": "Command completed.",
            }
        ],
    }

    markdown = matrix.render_markdown(report)

    assert "Summary: 1 pass, 0 warn, 0 fail, 0 skip" in markdown
    assert "`engramctl status`" in markdown


def test_matrix_health_url_uses_configured_local_port(tmp_path: Path, monkeypatch) -> None:
    matrix = _load_matrix()
    home = tmp_path / "home"
    env_path = home / ".engram/.env"
    env_path.parent.mkdir(parents=True)
    env_path.write_text("ENGRAM_API_PORT=18100\n")
    monkeypatch.setenv("HOME", str(home))

    assert matrix.health_url_from_env() == "http://127.0.0.1:18100/health"
