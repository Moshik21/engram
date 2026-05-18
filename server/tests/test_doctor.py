from __future__ import annotations

import argparse
import json

import pytest

from engram.doctor import (
    build_doctor_report,
    configure_doctor_parser,
    format_doctor_report,
    run_doctor_command,
)
from engram.evaluation.brain_loop_report import EVALUATION_SIGNAL_ORDER
from engram.storage.resolver import EngineMode


def _parse_doctor_args(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    configure_doctor_parser(parser)
    return parser.parse_args(list(args))


def _measured_signal_payloads() -> dict[str, dict[str, object]]:
    return {
        signal: {
            "status": "measured",
            "evidence_count": 1,
            "metric": 1.0,
            "gap": None,
        }
        for signal in EVALUATION_SIGNAL_ORDER
    }


def test_doctor_help_mentions_evaluation_signal_readiness() -> None:
    parser = argparse.ArgumentParser()
    configure_doctor_parser(parser)

    help_text = parser.format_help()

    assert "evaluation-signal readiness" in help_text
    assert "Skip the Capture -> Cue -> Project -> Recall ->" in help_text
    assert "Consolidate smoke and evaluation-signal readiness" in help_text
    assert "summary." in help_text


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
    evaluation_signals = checks["brain_loop_smoke"]["metadata"]["evaluation_signals"]
    assert evaluation_signals["ready"] is True
    assert evaluation_signals["measured"] == len(EVALUATION_SIGNAL_ORDER)
    assert evaluation_signals["required"] == len(EVALUATION_SIGNAL_ORDER)
    assert evaluation_signals["unmeasured"] == []
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
async def test_doctor_smoke_metadata_reports_unmeasured_evaluation_signals(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(tmp_path / "configured.db"))
    signals = _measured_signal_payloads()
    signals.pop("projection_yield")
    signals["false_recall"] = {
        "status": "needs_labels",
        "evidence_count": 0,
        "metric": None,
        "gap": "recall quality needs labeled recall_samples input",
    }

    async def fake_lifecycle_summary(*args, **kwargs):
        return {
            "groupId": kwargs.get("group_id"),
            "loop": ["capture", "cue", "project", "recall", "consolidate"],
            "totals": {"episodes": 0, "cues": 0, "projected": 0, "cycles": 0},
            "cue": {"coverage": 0.0},
            "project": {"status": "ready"},
            "consolidate": {"status": "ready"},
        }

    async def fake_smoke(**kwargs):
        return {
            "group_id": kwargs["group_id"],
            "coverage_gaps": [],
            "totals": {"episodes": 3},
            "project": {"projected_count": 3},
            "consolidate": {"cycle_count": 1},
            "evaluation_signals": signals,
        }

    monkeypatch.setattr(
        "engram.doctor.build_lifecycle_summary_for_config",
        fake_lifecycle_summary,
    )
    monkeypatch.setattr("engram.doctor.run_projected_consolidated_smoke_for_args", fake_smoke)

    report = await build_doctor_report(
        _parse_doctor_args("--mode", "lite", "--skip-server")
    )

    checks = {check["name"]: check for check in report["checks"]}
    assert report["status"] == "fail"
    assert checks["brain_loop_smoke"]["status"] == "fail"
    assert (
        checks["brain_loop_smoke"]["detail"]
        == "disposable lite Capture -> Cue -> Project -> Recall -> Consolidate "
        "smoke has unmeasured evaluation signals"
    )
    evaluation = checks["brain_loop_smoke"]["metadata"]["evaluation_signals"]
    assert evaluation["ready"] is False
    assert evaluation["measured"] == len(EVALUATION_SIGNAL_ORDER) - 2
    assert evaluation["unmeasured"] == [
        "projection_yield:missing",
        "false_recall:needs_labels",
    ]
    assert evaluation["statuses"]["projection_yield"] == "missing"
    assert evaluation["statuses"]["false_recall"] == "needs_labels"


@pytest.mark.asyncio
async def test_doctor_json_output_includes_evaluation_signal_metadata(
    monkeypatch,
    capsys,
) -> None:
    async def fake_report(args):
        assert args.format == "json"
        return {
            "status": "pass",
            "checks": [
                {
                    "name": "brain_loop_smoke",
                    "status": "pass",
                    "detail": "brain-loop smoke passed",
                    "metadata": {
                        "evaluation_signals": {
                            "required": len(EVALUATION_SIGNAL_ORDER),
                            "measured": len(EVALUATION_SIGNAL_ORDER),
                            "ready": True,
                            "unmeasured": [],
                            "statuses": {
                                signal: "measured"
                                for signal in EVALUATION_SIGNAL_ORDER
                            },
                        }
                    },
                }
            ],
            "lifecycle_summary": None,
            "smoke_report": {
                "group_id": "doctor_brain",
                "evaluation_signals": _measured_signal_payloads(),
            },
        }

    monkeypatch.setattr("engram.doctor.build_doctor_report", fake_report)

    await run_doctor_command(argparse.Namespace(format="json"))

    payload = json.loads(capsys.readouterr().out)
    metadata = payload["checks"][0]["metadata"]["evaluation_signals"]
    assert metadata["ready"] is True
    assert metadata["measured"] == len(EVALUATION_SIGNAL_ORDER)
    assert metadata["unmeasured"] == []


@pytest.mark.asyncio
async def test_doctor_command_exits_nonzero_for_failed_report(
    monkeypatch,
    capsys,
) -> None:
    async def fake_report(args):
        assert args.format == "json"
        return {
            "status": "fail",
            "checks": [
                {
                    "name": "brain_loop_smoke",
                    "status": "fail",
                    "detail": "brain-loop smoke has unmeasured evaluation signals",
                    "metadata": {
                        "evaluation_signals": {
                            "required": len(EVALUATION_SIGNAL_ORDER),
                            "measured": len(EVALUATION_SIGNAL_ORDER) - 1,
                            "ready": False,
                            "unmeasured": ["false_recall:needs_labels"],
                            "statuses": {
                                **{
                                    signal: "measured"
                                    for signal in EVALUATION_SIGNAL_ORDER
                                },
                                "false_recall": "needs_labels",
                            },
                        }
                    },
                }
            ],
            "lifecycle_summary": None,
            "smoke_report": None,
        }

    monkeypatch.setattr("engram.doctor.build_doctor_report", fake_report)

    with pytest.raises(SystemExit) as exc:
        await run_doctor_command(argparse.Namespace(format="json"))

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert payload["checks"][0]["metadata"]["evaluation_signals"]["ready"] is False


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


def test_doctor_markdown_lists_brain_loop_smoke_evaluation_signal_summary() -> None:
    signals = _measured_signal_payloads()
    signals["false_recall"] = {
        "status": "needs_labels",
        "evidence_count": 0,
        "metric": None,
        "gap": "recall quality needs labeled recall_samples input",
    }

    rendered = format_doctor_report(
        {
            "status": "warn",
            "checks": [],
            "smoke_report": {
                "group_id": "doctor_brain",
                "totals": {"episodes": 1},
                "project": {"projected_count": 1},
                "consolidate": {"cycle_count": 1},
                "coverage_gaps": [],
                "evaluation_signals": signals,
            },
        }
    )

    assert "- Evaluation signals: 5/6 measured" in rendered
    assert "  - false_recall:needs_labels" in rendered
