"""CLI tests for `engram loop status|apply|clear` (real entry path)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def loop_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # Ensure engram reads adjustment under this home
    adj = home / ".engram" / "loop-adjustment.json"
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(adj))
    monkeypatch.setenv(
        "ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE",
        str(home / ".engram" / "loop-adjustments.jsonl"),
    )
    return home


def _run_loop(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "engram", "loop", *args]
    return subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
        timeout=60,
    )


def test_loop_apply_status_clear_roundtrip(loop_home: Path, monkeypatch: pytest.MonkeyPatch):
    sample = {
        "version": 1,
        "group_id": "default",
        "regime": "debt_heavy",
        "reason": "hygiene should_mop; deferred multi-k; continuity pass",
        "ttl_hours": 12,
        "created_by": "harness:cli-test",
        "max_risk": "low",
        "budgets": {
            "evidence_drain": 2000,
            "already_exists": 500,
            "stale_reject": 500,
            "cue_hygiene": 500,
            "adjudication_limit": 400,
        },
        "phase_boost": ["evidence_adjudication", "prune"],
        "phase_defer": ["graph_embed", "dream"],
        "intake": {"auto_extract_min_score": 0.85, "pattern_junk_reject": True},
        "expected": {"continuity_must_pass": False},
    }
    adj_file = loop_home / "sample.json"
    adj_file.write_text(json.dumps(sample), encoding="utf-8")

    env = os.environ.copy()
    env["HOME"] = str(loop_home)
    env["ENGRAM_LOOP_ADJUSTMENT_FILE"] = str(loop_home / ".engram" / "loop-adjustment.json")
    env["ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE"] = str(loop_home / ".engram" / "loop-adjustments.jsonl")

    applied = _run_loop(
        "apply",
        "--file",
        str(adj_file),
        "--skip-continuity-check",
        "--format",
        "json",
        env=env,
    )
    assert applied.returncode == 0, applied.stderr + applied.stdout
    applied_payload = json.loads(applied.stdout)
    assert applied_payload["applied"] is True
    assert applied_payload["adjustment"]["regime"] == "debt_heavy"
    assert applied_payload["adjustment"]["budgets"]["evidence_drain"] == 2000

    status = _run_loop("status", "--format", "json", env=env)
    assert status.returncode == 0, status.stderr + status.stdout
    status_payload = json.loads(status.stdout)
    assert status_payload["active"] is True
    assert status_payload["remaining_ttl_seconds"] > 0
    assert status_payload["adjustment"]["phase_defer"] == ["graph_embed", "dream"]

    cleared = _run_loop("clear", "--format", "json", env=env)
    assert cleared.returncode == 0, cleared.stderr + cleared.stdout
    assert json.loads(cleared.stdout)["cleared"] is True

    status2 = _run_loop("status", "--format", "json", env=env)
    assert status2.returncode == 0
    assert json.loads(status2.stdout)["active"] is False


def test_loop_apply_rejects_empty_reason(loop_home: Path):
    env = os.environ.copy()
    env["HOME"] = str(loop_home)
    env["ENGRAM_LOOP_ADJUSTMENT_FILE"] = str(loop_home / ".engram" / "loop-adjustment.json")
    bad = loop_home / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "reason": "",
                "regime": "debt_heavy",
                "ttl_hours": 12,
                "max_risk": "low",
            }
        ),
        encoding="utf-8",
    )
    result = _run_loop(
        "apply",
        "--file",
        str(bad),
        "--skip-continuity-check",
        env=env,
    )
    assert result.returncode != 0
    assert "rejected" in (result.stderr + result.stdout).lower()
