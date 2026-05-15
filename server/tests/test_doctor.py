from __future__ import annotations

import argparse
import json

import pytest

from engram.doctor import build_doctor_report, configure_doctor_parser, format_doctor_report
from engram.storage.resolver import EngineMode


def _parse_doctor_args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    configure_doctor_parser(parser)
    return parser.parse_args(list(args))


@pytest.mark.asyncio
async def test_doctor_can_run_config_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))
    args = _parse_doctor_args("--mode", "lite", "--skip-server", "--no-smoke")

    report = await build_doctor_report(args)

    assert report["status"] == "pass", json.dumps(report["checks"], indent=2)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["config"]["status"] == "pass"
    assert checks["sqlite"]["status"] == "pass"
    assert checks["mode"]["metadata"]["resolved_mode"] == "lite"
    assert checks["lifecycle_snapshot"]["status"] == "pass"
    assert checks["server"]["status"] == "skipped"
    assert checks["brain_loop_smoke"]["status"] == "skipped"
    assert report["lifecycle_summary"]["groupId"] == "default"
    assert report["lifecycle_summary"]["loop"] == [
        "capture",
        "cue",
        "project",
        "recall",
        "consolidate",
    ]


@pytest.mark.asyncio
async def test_doctor_runs_brain_loop_smoke(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))
    args = _parse_doctor_args(
        "--mode",
        "lite",
        "--skip-server",
        "--sqlite-path",
        str(tmp_path / "doctor-smoke.db"),
        "--group-id",
        "doctor_brain",
    )

    report = await build_doctor_report(args)

    assert report["status"] == "pass", json.dumps(report["checks"], indent=2)
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["brain_loop_smoke"]["status"] == "pass"
    assert checks["brain_loop_smoke"]["metadata"]["group_id"] == "doctor_brain"
    assert checks["lifecycle_snapshot"]["status"] == "pass"
    assert checks["lifecycle_snapshot"]["metadata"]["group_id"] == "doctor_brain"
    assert report["lifecycle_summary"]["groupId"] == "doctor_brain"
    assert report["smoke_report"]["group_id"] == "doctor_brain"
    assert report["smoke_report"]["coverage_gaps"] == []


@pytest.mark.asyncio
async def test_doctor_smoke_uses_native_mode_when_resolved(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))
    native_dir = tmp_path / "native-data"
    requested: dict[str, object] = {}
    lifecycle_kwargs: dict[str, object] = {}

    async def fake_resolve_mode(mode: str) -> EngineMode:
        assert mode == "helix"
        return EngineMode.HELIX

    async def fake_lifecycle_summary(*args, **kwargs):
        lifecycle_kwargs.update(kwargs)
        return {
            "groupId": kwargs.get("group_id"),
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
            "totals": {"episodes": 0, "cues": 0, "projected": 0, "cycles": 0},
            "cue": {"coverage": 0.0},
            "project": {"status": "ready"},
            "consolidate": {"status": "ready"},
        }

    async def fake_smoke(**kwargs):
        requested.update(kwargs)
        return {
            "group_id": kwargs["group_id"],
            "coverage_gaps": [],
            "totals": {"episodes": 3},
            "project": {"projected_count": 3},
            "consolidate": {"cycle_count": 1},
            "smoke": {"mode": kwargs["mode"].value},
        }

    monkeypatch.setattr("engram.doctor.resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(
        "engram.doctor.build_lifecycle_summary_for_config",
        fake_lifecycle_summary,
    )
    monkeypatch.setattr("engram.doctor.run_projected_consolidated_smoke_for_args", fake_smoke)

    args = _parse_doctor_args(
        "--mode",
        "helix",
        "--skip-server",
        "--group-id",
        "native_brain",
        "--helix-data-dir",
        str(native_dir),
    )

    report = await build_doctor_report(args)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "pass", json.dumps(report["checks"], indent=2)
    assert requested["mode"] == EngineMode.HELIX
    assert requested["group_id"] == "native_brain"
    assert requested.get("helix_data_dir") is None
    assert lifecycle_kwargs["helix_data_dir"] == native_dir
    assert checks["brain_loop_smoke"]["metadata"]["mode"] == "helix"
    assert report["smoke_report"]["smoke"]["mode"] == "helix"


@pytest.mark.asyncio
async def test_doctor_can_skip_lifecycle_snapshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))
    args = _parse_doctor_args("--mode", "lite", "--skip-server", "--no-smoke", "--no-lifecycle")

    report = await build_doctor_report(args)

    assert report["status"] == "pass"
    checks = {check["name"]: check for check in report["checks"]}
    assert checks["lifecycle_snapshot"]["status"] == "skipped"
    assert report["lifecycle_summary"] is None


@pytest.mark.asyncio
async def test_doctor_lifecycle_snapshot_supports_helix_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))

    async def fake_resolve_mode(mode: str) -> EngineMode:
        assert mode == "helix"
        return EngineMode.HELIX

    async def fake_lifecycle_summary(*args, **kwargs):
        return {
            "groupId": kwargs.get("group_id"),
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
            "totals": {"episodes": 0, "cues": 0, "projected": 0, "cycles": 0},
            "cue": {"coverage": 0.0},
            "project": {"status": "ready"},
            "consolidate": {"status": "ready"},
        }

    monkeypatch.setattr("engram.doctor.resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(
        "engram.doctor.build_lifecycle_summary_for_config",
        fake_lifecycle_summary,
    )
    args = _parse_doctor_args(
        "--mode",
        "helix",
        "--skip-server",
        "--no-smoke",
        "--group-id",
        "native_brain",
    )

    report = await build_doctor_report(args)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "pass"
    assert checks["lifecycle_snapshot"]["status"] == "pass"
    assert checks["lifecycle_snapshot"]["metadata"]["resolved_mode"] == "helix"
    assert checks["lifecycle_snapshot"]["metadata"]["group_id"] == "native_brain"
    assert report["lifecycle_summary"]["groupId"] == "native_brain"


@pytest.mark.asyncio
async def test_doctor_warns_when_lifecycle_snapshot_has_attention(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))

    async def fake_lifecycle_summary(*args, **kwargs):
        return {
            "groupId": kwargs.get("group_id"),
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
            "totals": {"episodes": 2, "cues": 2, "projected": 1, "cycles": 1},
            "capture": {"status": "ready"},
            "cue": {"status": "ready", "coverage": 1.0},
            "project": {"status": "ready"},
            "recall": {"status": "ready"},
            "consolidate": {
                "status": "attention",
                "latestCycle": {
                    "id": "cyc_phase_error",
                    "status": "completed",
                    "error": None,
                    "phases": [
                        {
                            "phase": "graph_embed",
                            "status": "error",
                            "error": "optional vector index unavailable",
                        }
                    ],
                },
            },
        }

    monkeypatch.setattr(
        "engram.doctor.build_lifecycle_summary_for_config",
        fake_lifecycle_summary,
    )
    args = _parse_doctor_args("--mode", "lite", "--skip-server", "--no-smoke")

    report = await build_doctor_report(args)

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "warn"
    assert checks["lifecycle_snapshot"]["status"] == "warn"
    assert (
        checks["lifecycle_snapshot"]["detail"]
        == "lifecycle snapshot loaded with attention: "
        "consolidate (graph_embed: optional vector index unavailable)"
    )
    assert (
        checks["lifecycle_snapshot"]["metadata"]["consolidate_issue"]
        == "graph_embed: optional vector index unavailable"
    )


def test_doctor_markdown_includes_status_and_checks() -> None:
    rendered = format_doctor_report(
        {
            "status": "pass",
            "checks": [
                {
                    "name": "config",
                    "status": "pass",
                    "detail": "config loaded",
                    "metadata": {},
                }
            ],
            "lifecycle_summary": {
                "groupId": "doctor_brain",
                "totals": {"episodes": 2, "cues": 2, "projected": 1, "cycles": 1},
                "cue": {"coverage": 1.0},
                "project": {"status": "attention"},
                "consolidate": {"status": "ready"},
            },
        }
    )

    assert "Engram Doctor" in rendered
    assert "Overall: `pass`" in rendered
    assert "- config: `pass` - config loaded" in rendered
    assert "## Lifecycle Snapshot" in rendered
    assert "Group: `doctor_brain`" in rendered
    assert "Cue coverage: 100.0%" in rendered


def test_doctor_markdown_includes_lifecycle_phase_issue() -> None:
    rendered = format_doctor_report(
        {
            "status": "warn",
            "checks": [],
            "lifecycle_summary": {
                "groupId": "doctor_brain",
                "totals": {"episodes": 2, "cues": 2, "projected": 1, "cycles": 1},
                "cue": {"coverage": 1.0},
                "project": {"status": "ready"},
                "consolidate": {
                    "status": "attention",
                    "latestCycle": {
                        "id": "cyc_phase_error",
                        "status": "completed",
                        "error": None,
                        "phases": [
                            {
                                "phase": "graph_embed",
                                "status": "error",
                                "error": "optional vector index unavailable",
                            }
                        ],
                    },
                },
            },
        }
    )

    assert (
        "Consolidate: `attention` | error "
        "`graph_embed: optional vector index unavailable`"
    ) in rendered


def test_doctor_markdown_lists_brain_loop_smoke_coverage_gaps() -> None:
    rendered = format_doctor_report(
        {
            "status": "warn",
            "checks": [],
            "smoke_report": {
                "group_id": "doctor_brain",
                "totals": {"episodes": 1},
                "project": {"projected_count": 1},
                "consolidate": {"cycle_count": 1},
                "coverage_gaps": [
                    "recall gate needs runtime analyses",
                    "consolidation calibration quality needs labeled decision outcomes",
                ],
            },
        }
    )

    assert "- Coverage gaps: 2" in rendered
    assert "  - recall gate needs runtime analyses" in rendered
    assert (
        "  - consolidation calibration quality needs labeled decision outcomes"
        in rendered
    )
