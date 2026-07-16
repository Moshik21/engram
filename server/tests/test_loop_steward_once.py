"""Unit + CLI tests for engram loop steward-once."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from engram.loop_adjustment import run_steward_once, status_payload


def test_steward_once_healthy_no_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "loop-adjustment.json"
    audit = tmp_path / "a.jsonl"
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_FILE", str(path))
    monkeypatch.setenv("ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE", str(audit))
    out = run_steward_once(
        {"deferred_evidence": 0, "cue_only_episodes": 0, "open_work": 0},
        path=path,
        audit_path=audit,
        created_by="test",
    )
    assert out["status"] == "ok"
    assert out["regime"] == "healthy"
    assert out["applied"] is False
    assert out["wrote_active"] is False
    assert out["healthy_noop"] is True
    assert status_payload("default", path=path)["active"] is False


def test_steward_once_debt_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "loop-adjustment.json"
    audit = tmp_path / "a.jsonl"
    debt = {
        "deferred_evidence": 2500,
        "open_work": 3000,
        "should_trigger_mop": True,
    }
    mop_calls: list[dict] = []

    def mop_fn(*, budget: int, dry_run: bool = False):
        mop_calls.append({"budget": budget, "dry_run": dry_run})
        return {"debt_after": {"deferred_evidence": 2000}, "ok": True}

    out = run_steward_once(
        debt,
        path=path,
        audit_path=audit,
        created_by="test",
        do_mop=True,
        mop_budget=150,
        mop_fn=mop_fn,
    )
    assert out["status"] == "ok"
    assert out["regime"] == "debt_heavy"
    assert out["applied"] is True
    assert out["wrote_active"] is True
    assert status_payload("default", path=path)["active"] is True
    assert status_payload("default", path=path)["regime"] == "debt_heavy"
    assert mop_calls == [{"budget": 150, "dry_run": False}]
    assert out["debt_after"]["deferred_evidence"] == 2000


def test_steward_once_dry_run_no_persist(tmp_path: Path):
    path = tmp_path / "loop-adjustment.json"
    out = run_steward_once(
        {"deferred_evidence": 2000, "should_trigger_mop": True},
        path=path,
        audit_path=tmp_path / "a.jsonl",
        dry_run=True,
    )
    assert out["regime"] == "debt_heavy"
    assert out["applied"] is False
    assert out.get("would_apply") is True
    assert status_payload("default", path=path)["active"] is False


def _run_loop(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "engram", "loop", *args],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
        env=env or os.environ.copy(),
        timeout=90,
    )


def test_cli_steward_once_healthy_and_debt(tmp_path: Path):
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["ENGRAM_LOOP_ADJUSTMENT_FILE"] = str(tmp_path / "loop-adjustment.json")
    env["ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE"] = str(tmp_path / "a.jsonl")

    healthy = tmp_path / "healthy.json"
    healthy.write_text(
        json.dumps({"debt": {"deferred_evidence": 0, "open_work": 0}}),
        encoding="utf-8",
    )
    r1 = _run_loop(
        "steward-once",
        "--debt-json",
        str(healthy),
        "--format",
        "json",
        env=env,
    )
    assert r1.returncode == 0, r1.stderr + r1.stdout
    p1 = json.loads(r1.stdout)
    assert p1["regime"] == "healthy"
    assert p1["applied"] is False
    assert p1["wrote_active"] is False

    status = _run_loop("status", "--format", "json", env=env)
    assert json.loads(status.stdout)["active"] is False

    debt = tmp_path / "debt.json"
    debt.write_text(
        json.dumps(
            {
                "debt": {"deferred_evidence": 2200, "open_work": 2500},
                "pressure": {"should_trigger_mop": True},
            }
        ),
        encoding="utf-8",
    )
    r2 = _run_loop(
        "steward-once",
        "--debt-json",
        str(debt),
        "--format",
        "json",
        env=env,
    )
    assert r2.returncode == 0, r2.stderr + r2.stdout
    p2 = json.loads(r2.stdout)
    assert p2["regime"] == "debt_heavy"
    assert p2["applied"] is True
    assert p2["wrote_active"] is True

    status2 = _run_loop("status", "--format", "json", env=env)
    assert json.loads(status2.stdout)["active"] is True
