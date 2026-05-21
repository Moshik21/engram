from __future__ import annotations

import importlib.util
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
