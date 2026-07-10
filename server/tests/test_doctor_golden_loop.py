"""Doctor golden-loop checks (hooks, surface, continuity)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from engram.doctor import (
    _check_hooks_install,
    _check_mcp_surface,
    _check_promotion_window,
    build_doctor_report,
)


def test_check_hooks_install_warns_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    checks: list[dict] = []
    _check_hooks_install(checks)
    assert checks[0]["name"] == "hooks"
    assert checks[0]["status"] == "warn"


def test_check_hooks_install_pass_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    hooks = tmp_path / ".engram" / "hooks"
    hooks.mkdir(parents=True)
    for name in (
        "capture-prompt.sh",
        "capture-response.sh",
        "session-start.sh",
        "session-end.sh",
        "pre-compact.sh",
        "session-promote-nudge.sh",
    ):
        (hooks / name).write_text("#!/bin/bash\n", encoding="utf-8")
    checks: list[dict] = []
    _check_hooks_install(checks)
    assert checks[0]["status"] == "pass"


def test_check_mcp_surface_public(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "public")
    checks: list[dict] = []
    _check_mcp_surface(checks)
    assert checks[0]["status"] == "pass"
    assert checks[0]["metadata"]["surface"] == "public"


def test_check_mcp_surface_full_warns(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "full")
    checks: list[dict] = []
    _check_mcp_surface(checks)
    assert checks[0]["status"] == "warn"


def test_check_promotion_window_writable(tmp_path, monkeypatch):
    monkeypatch.setenv("ENGRAM_PROMOTION_WINDOW_FILE", str(tmp_path / "promotion-window.json"))
    checks: list[dict] = []
    _check_promotion_window(checks)
    assert checks[0]["status"] == "pass"


@pytest.mark.asyncio
async def test_build_doctor_report_skips_golden_by_default(monkeypatch):
    monkeypatch.setenv("ENGRAM_MCP_SURFACE", "public")
    args = argparse.Namespace(
        format="json",
        mode="lite",
        group_id=None,
        sqlite_path=None,
        helix_data_dir=None,
        replace=False,
        no_smoke=True,
        no_lifecycle=True,
        lifecycle_timeout=10.0,
        smoke_timeout=45.0,
        skip_server=True,
        server_url="http://localhost:8100",
        timeout=1.0,
        require_golden_loop=False,
        continuity_timeout=60.0,
    )
    report = await build_doctor_report(args)
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["hooks"]["status"] == "skipped"
    assert by_name["continuity_smoke"]["status"] == "skipped"
