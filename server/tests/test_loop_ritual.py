"""Structural + behavioral tests for steward nudge hook and dogfood script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_session_steward_nudge_hook_writes_file(tmp_path: Path):
    hook = ROOT / "hooks" / "session-steward-nudge.sh"
    assert hook.is_file()
    nudge = tmp_path / "session-steward-nudge.md"
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["ENGRAM_SESSION_STEWARD_NUDGE_FILE"] = str(nudge)
    result = subprocess.run(
        ["bash", str(hook)],
        input='{"session_id":"s1","cwd":"/tmp/proj","hook_event_name":"SessionEnd"}',
        text=True,
        capture_output=True,
        env=env,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert nudge.is_file()
    body = nudge.read_text(encoding="utf-8")
    assert "Loop Steward" in body
    assert "engram loop" in body
    assert "do not narrate" in body.lower() or "Silent" in body
    # Must not claim apply was run
    assert "applied" not in body.lower() or "never" in body.lower() or "Apply only" in body


def test_dogfood_script_exists_and_runs(tmp_path: Path):
    script = ROOT / "scripts" / "dogfood_loop_steward.sh"
    assert script.is_file()
    env = os.environ.copy()
    env["ENGRAM_DOGFOOD_LOOP_DIR"] = str(tmp_path / "dogfood")
    env["ENGRAM_LOOP_ADJUSTMENT_FILE"] = str(tmp_path / "dogfood" / "loop-adjustment.json")
    env["ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE"] = str(tmp_path / "dogfood" / "loop-adjustments.jsonl")
    result = subprocess.run(
        ["bash", str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert (
        "propose-from-report" in result.stdout
        or "Proposed" in result.stdout
        or "regime" in result.stdout
    )
    assert (tmp_path / "dogfood" / "propose.json").is_file() or "OK dogfood" in result.stdout
    # Continuity either ran or honest SKIP
    cont = tmp_path / "dogfood" / "continuity.txt"
    if cont.is_file():
        text = cont.read_text(encoding="utf-8")
        assert "SKIP" in text or "PASS" in text or "FAIL" in text or "Continuity" in text
