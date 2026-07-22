"""Mop watchdog: parent/child process split.

Hazard (measured 2026-07-21): a sync-blocked native call froze the runner's
event loop — the asyncio deadline could not fire and SIGTERM was dead (its
handler runs on the loop); only SIGKILL worked, and the shell stayed paused
~45min. The parent must therefore run the cycle in a killable child and
restore the shell in a finally the child cannot block.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from engram import brain_cli
from engram.brain_runtime import (
    brain_lock_path,
    brain_status_path,
    exclusive_brain_lock,
    read_pause_marker,
    write_pause_marker,
)


@pytest.fixture()
def engram_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("ENGRAM_HOME", str(tmp_path))
    return tmp_path


def _run_args(**overrides) -> argparse.Namespace:
    base = {
        "brain_command": "run",
        "tier": "mop",
        "profile": "quiet",
        "group_id": "default",
        "dry_run": True,
        "pause_shell": True,
        "format": "json",
        "budget": 10,
        "force": False,
        "deadline_seconds": 0.3,
        "child": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


async def _async_result(value):
    return value


_SLEEPER = [sys.executable, "-c", "import time; time.sleep(60)"]
_TERM_IGNORER = [
    sys.executable,
    "-c",
    "import signal, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "while True:\n"
    "    time.sleep(0.05)\n",
]
_HEALTHY_PAYLOAD = {
    "ok": True,
    "error": None,
    "result": {
        "status": "completed",
        "profile": "quiet",
        "tier": "mop",
        "cycle_id": None,
        "summary": {
            "total_processed": 3,
            "total_affected": 3,
            "deferred_before": 2,
            "deferred_after": 0,
        },
    },
    "paused_shell": False,
    "duration_s": 0.01,
}
_HEALTHY_CHILD = [
    sys.executable,
    "-c",
    f"import json; print(json.dumps({_HEALTHY_PAYLOAD!r}))",
]


def _parent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Small grace so expiry tests stay fast (wall = deadline + grace)."""
    monkeypatch.setenv("ENGRAM_BRAIN_WATCHDOG_GRACE_SECONDS", "0")


class TestWatchdogParent:
    def test_kills_frozen_child_and_resumes(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _parent_env(monkeypatch)
        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli, "_pause_shell", return_value=True) as pause,
            patch.object(brain_cli, "_resume_shell", return_value=True) as resume,
            patch.object(brain_cli, "_child_command", return_value=_SLEEPER),
            patch.object(
                brain_cli,
                "_run_cycle",
                side_effect=AssertionError("parent must never run the cycle in-process"),
            ),
        ):
            started = time.monotonic()
            rc = brain_cli.run_brain_command(args)
            elapsed = time.monotonic() - started
        assert rc == 1
        assert elapsed < 10
        pause.assert_called_once()
        resume.assert_called_once()
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is False
        assert "watchdog killed a frozen cycle" in payload["error"]
        assert "window skipped" in payload["error"]
        assert payload["paused_shell"] is True
        assert set(payload) == {"ok", "error", "result", "paused_shell", "duration_s"}
        status = json.loads(brain_status_path().read_text())
        assert status["ok"] is False
        assert "watchdog killed" in status["error"]

    def test_healthy_child_result_passes_through(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _parent_env(monkeypatch)
        args = _run_args(deadline_seconds=30)
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli, "_pause_shell", return_value=True),
            patch.object(brain_cli, "_resume_shell", return_value=True) as resume,
            patch.object(brain_cli, "_child_command", return_value=_HEALTHY_CHILD),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        resume.assert_called_once()
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True
        assert payload["error"] is None
        assert payload["paused_shell"] is True
        assert payload["result"]["summary"]["total_processed"] == 3
        status = json.loads(brain_status_path().read_text())
        assert status["ok"] is True
        assert status["summary"]["deferred_after"] == 0
        assert status["paused_shell"] is True

    def test_sigterm_ignoring_child_gets_sigkill(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _parent_env(monkeypatch)
        monkeypatch.setattr(brain_cli, "_WATCHDOG_TERM_GRACE_SECONDS", 0.5)
        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli, "_pause_shell", return_value=True),
            patch.object(brain_cli, "_resume_shell", return_value=True) as resume,
            patch.object(brain_cli, "_child_command", return_value=_TERM_IGNORER),
        ):
            started = time.monotonic()
            rc = brain_cli.run_brain_command(args)
            elapsed = time.monotonic() - started
        assert rc == 1
        assert elapsed < 10  # SIGKILL fired; the ignorer would sleep forever otherwise
        resume.assert_called_once()
        payload = json.loads(capsys.readouterr().out)
        assert "watchdog killed a frozen cycle" in payload["error"]

    def test_finally_resumes_and_writes_status_when_spawn_crashes(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _parent_env(monkeypatch)
        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli, "_pause_shell", return_value=True),
            patch.object(brain_cli, "_resume_shell", return_value=True) as resume,
            patch.object(
                brain_cli, "_spawn_child_and_watch", side_effect=RuntimeError("spawn exploded")
            ),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 1
        resume.assert_called_once()
        status = json.loads(brain_status_path().read_text())
        assert status["ok"] is False
        assert "spawn exploded" in status["error"]
        assert json.loads(capsys.readouterr().out)["ok"] is False

    def test_lock_held_skips_before_pausing(
        self, engram_home: Path, monkeypatch: pytest.MonkeyPatch, capsys
    ):
        _parent_env(monkeypatch)
        args = _run_args()
        with exclusive_brain_lock(brain_lock_path()):
            with (
                patch.object(brain_cli, "on_battery_power", return_value=False),
                patch.object(brain_cli, "_shell_healthy", return_value=False),
                patch.object(brain_cli, "serve_process_alive", return_value=False),
                patch.object(brain_cli, "_pause_shell") as pause,
                patch.object(brain_cli, "_spawn_child_and_watch") as spawn,
            ):
                rc = brain_cli.run_brain_command(args)
        assert rc == 1
        pause.assert_not_called()
        spawn.assert_not_called()
        assert not brain_status_path().exists()
        payload = json.loads(capsys.readouterr().out)
        assert payload["skipped"] is True


class TestWatchdogChildMode:
    def test_child_leaves_parent_marker_status_and_shell_alone(self, engram_home: Path, capsys):
        write_pause_marker()  # the PARENT's marker
        args = _run_args(pause_shell=False, child=True, deadline_seconds=5)
        with (
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "_resume_shell") as resume,
            patch.object(
                brain_cli,
                "_run_cycle",
                new=lambda a: _async_result(
                    {"status": "completed", "profile": "quiet", "summary": {}}
                ),
            ),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        resume.assert_not_called()
        assert read_pause_marker() is not None
        assert not brain_status_path().exists()
        payload = json.loads(capsys.readouterr().out)
        assert payload["result"]["status"] == "completed"

    def test_child_lock_contention_prints_skip_shape(self, engram_home: Path, capsys):
        args = _run_args(pause_shell=False, child=True)
        with exclusive_brain_lock(brain_lock_path()):
            with patch.object(brain_cli, "_shell_healthy", return_value=False):
                rc = brain_cli.run_brain_command(args)
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["skipped"] is True
        assert "Another brain process holds" in payload["error"]
        assert not brain_status_path().exists()


class TestDeadlinePlumbing:
    def test_child_command_carries_cycle_deadline_and_flags(self):
        args = _run_args(deadline_seconds=7, dry_run=True, force=True, profile="quiet")
        cmd = brain_cli._child_command(args, brain_cli._deadline_seconds(args))
        joined = " ".join(cmd)
        assert cmd[:3] == [sys.executable, "-m", "engram"]
        assert "--child" in cmd
        assert "--no-pause-shell" in cmd
        assert "--dry-run" in cmd
        assert "--force" in cmd
        assert "--tier mop" in joined
        assert "--deadline-seconds 7.0" in joined
        assert "--format json" in joined
        assert "--profile quiet" in joined

    def test_deadline_env_fallback_reaches_child(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ENGRAM_BRAIN_DEADLINE_SECONDS", "123")
        args = _run_args(deadline_seconds=None)
        cmd = brain_cli._child_command(args, brain_cli._deadline_seconds(args))
        assert "--deadline-seconds 123.0" in " ".join(cmd)

    def test_grace_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ENGRAM_BRAIN_WATCHDOG_GRACE_SECONDS", "5")
        assert brain_cli._watchdog_grace_seconds() == 5.0
        monkeypatch.setenv("ENGRAM_BRAIN_WATCHDOG_GRACE_SECONDS", "bogus")
        assert brain_cli._watchdog_grace_seconds() == 120.0
        monkeypatch.delenv("ENGRAM_BRAIN_WATCHDOG_GRACE_SECONDS")
        assert brain_cli._watchdog_grace_seconds() == 120.0


class TestParentImportAudit:
    def test_module_level_imports_construct_no_stores(self):
        """The parent path must never open the graph: store/extractor imports
        stay function-local to _run_mop/_run_cycle (child-only code)."""
        banned = (
            "engram.storage",
            "engram.graph_manager",
            "engram.hygiene_ops",
            "engram.extraction",
            "engram.consolidation.engine",
        )
        tree = ast.parse(inspect.getsource(brain_cli))
        for node in tree.body:
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not any(name == b or name.startswith(b + ".") for b in banned), name

    def test_parent_helpers_do_not_call_cycle_runners(self):
        """_run_watchdog_parent/_spawn_child_and_watch must not reach the
        in-process cycle runners (which construct stores)."""
        src = inspect.getsource(brain_cli)
        tree = ast.parse(src)
        parent_funcs = {"_run_watchdog_parent", "_spawn_child_and_watch", "_child_command"}
        banned_calls = {"_run_cycle", "_run_mop", "_run_cycle_with_deadline", "open_local_stores"}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in parent_funcs:
                for call in ast.walk(node):
                    if isinstance(call, ast.Call):
                        fn = call.func
                        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                        assert name not in banned_calls, f"{node.name} calls {name}"
