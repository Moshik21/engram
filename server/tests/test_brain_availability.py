"""Availability & exclusivity: pause marker, power gate, abort-on-pause-failure,
lock probes, preflight skip, deadline, and lock-contention status protection."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from engram.brain_runtime import (
    ExclusiveAccessError,
    brain_lock_is_held,
    clear_pause_marker,
    exclusive_brain_lock,
    on_battery_power,
    read_pause_marker,
    require_exclusive_local_access,
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
        "dry_run": None,
        "pause_shell": True,
        "format": "text",
        "budget": 10,
        "force": False,
        "deadline_seconds": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class TestPauseMarker:
    def test_lifecycle(self, engram_home: Path):
        assert read_pause_marker() is None
        write_pause_marker()
        marker = read_pause_marker()
        assert marker is not None and "paused_at" in marker
        clear_pause_marker()
        assert read_pause_marker() is None

    def test_clear_missing_is_noop(self, engram_home: Path):
        clear_pause_marker()  # must not raise


class TestLockProbe:
    def test_free_and_held(self, tmp_path: Path):
        lock = tmp_path / "brain.lock"
        assert brain_lock_is_held(lock) is False  # missing file
        lock.touch()
        assert brain_lock_is_held(lock) is False  # free
        with exclusive_brain_lock(lock):
            assert brain_lock_is_held(lock) is True
        assert brain_lock_is_held(lock) is False


class TestPowerGate:
    def test_battery_detected(self):
        assert on_battery_power("Now drawing from 'Battery Power'\n") is True

    def test_ac_power(self):
        assert on_battery_power("Now drawing from 'AC Power'\n") is False

    def test_empty_output_defaults_to_run(self):
        assert on_battery_power("") is False


class TestExclusiveLocalAccess:
    def test_refuses_when_shell_healthy(self, tmp_path: Path):
        with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
            with pytest.raises(ExclusiveAccessError, match="shell is running"):
                with require_exclusive_local_access(lock_path=tmp_path / "l"):
                    pass

    def test_refuses_when_serve_process_alive(self, tmp_path: Path):
        with (
            patch("engram.brain_runtime.shell_is_healthy", return_value=False),
            patch("engram.brain_runtime.serve_process_alive", return_value=True),
        ):
            with pytest.raises(ExclusiveAccessError, match="serve"):
                with require_exclusive_local_access(lock_path=tmp_path / "l"):
                    pass

    def test_grants_and_serializes(self, tmp_path: Path):
        lock = tmp_path / "l"
        with (
            patch("engram.brain_runtime.shell_is_healthy", return_value=False),
            patch("engram.brain_runtime.serve_process_alive", return_value=False),
        ):
            with require_exclusive_local_access(lock_path=lock):
                assert brain_lock_is_held(lock) is True
                with pytest.raises(ExclusiveAccessError):
                    with require_exclusive_local_access(lock_path=lock):
                        pass

    def test_force_skips_shell_checks_but_still_locks(self, tmp_path: Path):
        lock = tmp_path / "l"
        with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
            with require_exclusive_local_access(force=True, lock_path=lock):
                assert brain_lock_is_held(lock) is True


class TestPauseShell:
    def test_shell_down_no_process_returns_false(self, engram_home: Path):
        from engram import brain_cli

        with (
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
        ):
            assert brain_cli._pause_shell() is False
        assert read_pause_marker() is None

    def test_shell_down_but_process_alive_aborts(self, engram_home: Path):
        from engram import brain_cli

        with (
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=True),
        ):
            with pytest.raises(brain_cli.BrainPauseError, match="serve"):
                brain_cli._pause_shell()

    def test_engramctl_missing_aborts_and_clears_marker(self, engram_home: Path):
        from engram import brain_cli

        with (
            patch.object(brain_cli, "_shell_healthy", return_value=True),
            patch.object(brain_cli.subprocess, "run", side_effect=FileNotFoundError),
        ):
            with pytest.raises(brain_cli.BrainPauseError, match="engramctl not found"):
                brain_cli._pause_shell()
        assert read_pause_marker() is None

    def test_successful_stop_returns_true_and_keeps_marker(self, engram_home: Path):
        from engram import brain_cli

        healths = iter([True, False])  # healthy at probe, down after stop

        class _Ok:
            returncode = 0
            stdout = ""
            stderr = ""

        with (
            patch.object(brain_cli, "_shell_healthy", side_effect=lambda *_: next(healths, False)),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli.subprocess, "run", return_value=_Ok()),
        ):
            assert brain_cli._pause_shell() is True
        assert read_pause_marker() is not None

    def test_shell_never_drops_aborts_with_marker_kept(self, engram_home: Path):
        from engram import brain_cli

        class _Ok:
            returncode = 0
            stdout = ""
            stderr = ""

        clock = {"t": 0.0}

        def fake_monotonic():
            clock["t"] += 30.0
            return clock["t"]

        with (
            patch.object(brain_cli, "_shell_healthy", return_value=True),
            patch.object(brain_cli.subprocess, "run", return_value=_Ok()),
            patch.object(brain_cli.time, "monotonic", side_effect=fake_monotonic),
            patch.object(brain_cli.time, "sleep"),
        ):
            with pytest.raises(brain_cli.BrainPauseError, match="still up"):
                brain_cli._pause_shell()
        assert read_pause_marker() is not None


class TestResumeShell:
    def test_confirmed_resume_clears_marker(self, engram_home: Path):
        from engram import brain_cli

        write_pause_marker()

        class _Ok:
            returncode = 0
            stdout = ""
            stderr = ""

        with (
            patch.object(brain_cli.subprocess, "run", return_value=_Ok()),
            patch.object(brain_cli, "_shell_healthy", return_value=True),
        ):
            assert brain_cli._resume_shell() is True
        assert read_pause_marker() is None

    def test_failed_resume_keeps_marker(self, engram_home: Path):
        from engram import brain_cli

        write_pause_marker()
        with patch.object(brain_cli.subprocess, "run", side_effect=FileNotFoundError):
            assert brain_cli._resume_shell() is False
        assert read_pause_marker() is not None


class TestRunCommand:
    def test_lock_contention_does_not_write_status(self, engram_home: Path):
        from engram import brain_cli
        from engram.brain_runtime import brain_lock_path, brain_status_path

        args = _run_args()
        with exclusive_brain_lock(brain_lock_path()):
            with (
                patch.object(brain_cli, "on_battery_power", return_value=False),
                patch.object(brain_cli, "_shell_healthy", return_value=False),
                patch.object(brain_cli, "serve_process_alive", return_value=False),
            ):
                rc = brain_cli.run_brain_command(args)
        assert rc == 1
        assert not brain_status_path().exists()

    def test_battery_gate_skips_without_pause(self, engram_home: Path, capsys):
        from engram import brain_cli
        from engram.brain_runtime import brain_status_path

        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=True),
            patch.object(brain_cli, "_shell_healthy", return_value=True),
            patch.object(brain_cli, "_pause_shell") as pause,
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        pause.assert_not_called()
        assert not brain_status_path().exists()
        assert "battery" in capsys.readouterr().out

    def test_preflight_no_work_skips(self, engram_home: Path, capsys):
        from engram import brain_cli

        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_preflight_skip_no_work", return_value=True),
            patch.object(brain_cli, "_shell_healthy", return_value=True),
            patch.object(brain_cli, "_pause_shell") as pause,
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        pause.assert_not_called()
        assert "no actionable" in capsys.readouterr().out

    def test_pause_failure_aborts_without_cycle(self, engram_home: Path):
        from engram import brain_cli
        from engram.brain_runtime import brain_status_path

        args = _run_args()
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_preflight_skip_no_work", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=True),
            patch.object(
                brain_cli,
                "_pause_shell",
                side_effect=brain_cli.BrainPauseError("boom"),
            ),
            patch.object(brain_cli, "_run_cycle") as cycle,
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 1
        cycle.assert_not_called()
        status = json.loads(brain_status_path().read_text())
        assert status["ok"] is False
        assert "pause-shell failed" in status["error"]

    def test_stranded_marker_resumes_before_run(self, engram_home: Path):
        from engram import brain_cli

        write_pause_marker()
        args = _run_args(pause_shell=False, force=True)
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(brain_cli, "_resume_shell", return_value=True) as resume,
            patch.object(
                brain_cli,
                "_run_cycle",
                new=lambda a: _async_result({"status": "completed", "summary": {}}),
            ),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        assert resume.call_count >= 1

    def test_no_pause_with_healthy_shell_refused(self, engram_home: Path):
        from engram import brain_cli

        args = _run_args(pause_shell=False)
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=True),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 2

    def test_successful_run_records_monotonic(self, engram_home: Path):
        from engram import brain_cli
        from engram.brain_runtime import brain_status_path

        args = _run_args(pause_shell=False, force=True)
        with (
            patch.object(brain_cli, "on_battery_power", return_value=False),
            patch.object(brain_cli, "_shell_healthy", return_value=False),
            patch.object(brain_cli, "serve_process_alive", return_value=False),
            patch.object(
                brain_cli,
                "_run_cycle",
                new=lambda a: _async_result({"status": "completed", "summary": {}}),
            ),
        ):
            rc = brain_cli.run_brain_command(args)
        assert rc == 0
        status = json.loads(brain_status_path().read_text())
        assert status["ok"] is True
        assert status["duration_monotonic_s"] is not None
        assert status["system_slept"] is False


async def _async_result(value):
    return value


class TestDeadline:
    def test_deadline_cancels_slow_cycle(self, engram_home: Path):
        from engram import brain_cli

        async def slow(_args):
            await asyncio.sleep(30)

        args = _run_args(deadline_seconds=0.05)
        with patch.object(brain_cli, "_run_cycle", new=slow):
            with pytest.raises(RuntimeError, match="deadline"):
                asyncio.run(brain_cli._run_cycle_with_deadline(args))

    def test_zero_deadline_disables(self, engram_home: Path):
        from engram import brain_cli

        async def fast(_args):
            return {"status": "completed"}

        args = _run_args(deadline_seconds=0)
        with patch.object(brain_cli, "_run_cycle", new=fast):
            result = asyncio.run(brain_cli._run_cycle_with_deadline(args))
        assert result["status"] == "completed"


class TestCliExclusivityGuards:
    def test_hygiene_cli_refuses_with_shell_up(self, capsys):
        from engram.hygiene_cli import run_hygiene_command

        args = argparse.Namespace(
            action="report",
            group_id=None,
            mode="lite",
            helix_data_dir=None,
            dry_run=False,
            budget=10,
            format="json",
            force_local=False,
        )
        with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
            rc = asyncio.run(run_hygiene_command(args))
        assert rc == 2
        assert "must not run concurrently" in capsys.readouterr().err

    def test_index_cli_refuses_with_shell_up(self, capsys):
        from engram.index_cli import run_index_command

        args = argparse.Namespace(
            group_id=None,
            mode="lite",
            helix_data_dir=None,
            max_entities=10,
            batch_size=4,
            dry_run=True,
            remeasure=False,
            format="json",
            force_local=False,
        )
        with patch("engram.brain_runtime.shell_is_healthy", return_value=True):
            rc = asyncio.run(run_index_command(args))
        assert rc == 2

    def test_steward_debt_sense_uses_http_when_shell_up(self):
        from engram import loop_cli

        args = argparse.Namespace(
            mode=None,
            helix_data_dir=None,
            group_id=None,
        )
        payload = {"debt": {"deferred_evidence": 3}, "pressure": {"should_trigger_mop": False}}
        with (
            patch.object(loop_cli, "_server_reachable", return_value=True),
            patch.object(loop_cli, "_collect_live_debt", return_value=payload) as live,
        ):
            got = loop_cli._collect_live_or_offline_debt(args)
        assert got == payload
        live.assert_called_once()


class TestShutdownConsolidationGate:
    @pytest.mark.asyncio
    async def test_shell_role_skips_shutdown_cycle(self):
        from engram import main as engram_main
        from engram.config import EngramConfig

        class _Engine:
            is_running = False
            cancelled = False

            def cancel(self):
                self.cancelled = True

            def run_cycle(self, **kwargs):  # pragma: no cover - must not be called
                raise AssertionError("shutdown cycle ran in shell role")

        engine = _Engine()
        saved = dict(engram_main._app_state)
        engram_main._app_state.clear()
        engram_main._app_state.update(
            {
                "config": EngramConfig(runtime_role="shell"),
                "consolidation_engine": engine,
            }
        )
        try:
            await engram_main._shutdown()
        finally:
            engram_main._app_state.clear()
            engram_main._app_state.update(saved)
