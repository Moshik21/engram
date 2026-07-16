"""Unit + CLI tests for loop propose / propose-from-report."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engram.loop_adjustment import (
    classify_regime_from_debt,
    propose_from_report,
    status_payload,
)


def test_classify_and_propose_from_report_debt_heavy():
    debt = {
        "deferred_evidence": 2500,
        "cue_only_episodes": 100,
        "open_work": 3000,
        "should_trigger_mop": True,
    }
    assert classify_regime_from_debt(debt) == "debt_heavy"
    result = propose_from_report(debt, created_by="test")
    assert not result.rejected
    assert result.adjustment.regime == "debt_heavy"
    assert result.adjustment.budgets.get("evidence_drain", 0) >= 1000
    assert "dream" in result.adjustment.phase_defer


def test_propose_from_report_healthy_and_offline():
    healthy = propose_from_report({"deferred_evidence": 0, "cue_only_episodes": 0, "open_work": 0})
    assert healthy.adjustment.regime == "healthy"
    offline = propose_from_report(None, server_reachable=False)
    assert offline.adjustment.regime == "offline"


def test_propose_does_not_write_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(tmp_path / "loop-adjustment.json"))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(tmp_path / "a.jsonl"))
    result = propose_from_report(
        {"deferred_evidence": 900, "should_trigger_mop": True},
        group_id="default",
    )
    assert not result.rejected
    assert status_payload("default")["active"] is False


def test_propose_rejects_empty_reason_via_clamp():
    from engram.loop_adjustment import LoopAdjustment, clamp_loop_adjustment

    bad = clamp_loop_adjustment(LoopAdjustment(reason="", regime="debt_heavy", max_risk="low"))
    assert bad.rejected


@pytest.fixture
def loop_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(
        "ENGRAM_LOOP_ADJUSTMENT_FILE", str(home / ".engram" / "loop-adjustment.json")
    )
    monkeypatch.setenv(
        "ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE",
        str(home / ".engram" / "loop-adjustments.jsonl"),
    )
    return home


def _run_loop(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "engram", "loop", *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
        timeout=60,
    )


def test_cli_propose_from_report_no_write(loop_home: Path):
    env = os.environ.copy()
    env["HOME"] = str(loop_home)
    env["ENGRAM_LOOP_ADJUSTMENT_FILE"] = str(loop_home / ".engram" / "loop-adjustment.json")
    debt = loop_home / "debt.json"
    debt.write_text(
        json.dumps(
            {
                "debt": {"deferred_evidence": 2000, "open_work": 2200},
                "pressure": {"should_trigger_mop": True},
            }
        ),
        encoding="utf-8",
    )
    prop = _run_loop(
        "propose-from-report",
        "--debt-json",
        str(debt),
        "--format",
        "json",
        env=env,
    )
    assert prop.returncode == 0, prop.stderr + prop.stdout
    payload = json.loads(prop.stdout)
    assert payload["proposed"] is True
    assert payload["wrote_active"] is False
    assert payload["adjustment"]["regime"] == "debt_heavy"

    status = _run_loop("status", "--format", "json", env=env)
    assert status.returncode == 0
    assert json.loads(status.stdout)["active"] is False
