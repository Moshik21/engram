from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from engram.evaluation.adoption_evidence import (
    adoption_client_set_failure_message,
    adoption_evidence_failure_message,
    build_adoption_client_set_evidence,
    build_adoption_evidence,
    link_adoption_to_human_label_evidence,
    load_adoption_evidence,
)
from engram.evaluation.brain_loop_report import (
    EVALUATION_SIGNAL_ORDER,
    format_brain_loop_report_markdown,
)
from engram.evaluation.cli import configure_evaluate_parser, run_evaluate_command


def test_adoption_evidence_accepts_passed_live_report() -> None:
    evidence = build_adoption_evidence(
        _adoption_report(),
        artifact_path=Path("adoption-report.json"),
        artifact_sha256="abc123",
    )

    assert evidence["status"] == "measured"
    assert evidence["artifact_path"] == "adoption-report.json"
    assert evidence["artifact_sha256"] == "abc123"
    assert evidence["client"] == "Cursor"
    assert evidence["session_id"] == "cursor-thread-1"
    assert evidence["call_count"] == 4
    assert adoption_evidence_failure_message(evidence, prefix="Adoption") is None


def test_adoption_evidence_preserves_live_harness_blockers() -> None:
    report = _adoption_report(client="Claude Code", session_id="claude-session-blocked")
    report["status"] = "failed"
    report["callCount"] = 0
    report["evidence"]["blockers"] = [
        "mcp_server_failed",
        "authentication_failed",
    ]
    report["evidence"]["blocker_details"] = [
        "system:error: Not logged in - Please run /login",
    ]
    report["evidence"]["mcp_server_failures"] = ["engram"]
    report["validation"]["failures"] = [
        "live_harness_authentication_failed",
        "live_harness_mcp_server_failed",
    ]

    evidence = build_adoption_evidence(report)
    client_set = build_adoption_client_set_evidence(
        [evidence],
        required_clients=["Claude Code"],
    )

    assert evidence["status"] == "failed"
    assert evidence["blockers"] == [
        "mcp_server_failed",
        "authentication_failed",
    ]
    assert evidence["blocker_details"] == [
        "system:error: Not logged in - Please run /login",
    ]
    assert evidence["mcp_server_failures"] == ["engram"]
    assert evidence["failures"] == [
        "adoption_status_not_passed",
        (
            "adoption_validation_failures("
            "live_harness_authentication_failed,live_harness_mcp_server_failed)"
        ),
    ]
    assert client_set["blockers"] == [
        "mcp_server_failed",
        "authentication_failed",
    ]
    assert client_set["mcp_server_failures"] == ["engram"]
    assert client_set["reports"][0]["blockers"] == [
        "mcp_server_failed",
        "authentication_failed",
    ]


def test_adoption_evidence_rejects_failed_or_sessionless_report() -> None:
    report = _adoption_report()
    report["status"] = "failed"
    report["evidence"]["client"] = None
    report["validation"]["file_memory"]["substituted_for_engram"] = True
    report["validation"]["failures"] = ["missing_before_answer_tool"]

    evidence = build_adoption_evidence(report)

    assert evidence["status"] == "failed"
    assert evidence["failures"] == [
        "adoption_status_not_passed",
        "missing_adoption_client",
        "file_memory_substituted_for_engram",
        "adoption_validation_failures(missing_before_answer_tool)",
    ]


def test_adoption_evidence_requires_live_evidence_gate() -> None:
    report = _adoption_report()
    report["evidence"]["required"] = False

    evidence = build_adoption_evidence(report)

    assert evidence["status"] == "failed"
    assert evidence["required_live_evidence"] is False
    assert evidence["failures"] == ["missing_required_live_evidence_gate"]


def test_adoption_evidence_requires_expected_client_gate() -> None:
    report = _adoption_report()
    report["evidence"].pop("required_client")

    evidence = build_adoption_evidence(report, required_client="Cursor")

    assert evidence["status"] == "failed"
    assert evidence["client"] == "Cursor"
    assert evidence["gate_required_client"] == "Cursor"
    assert evidence["failures"] == ["missing_required_adoption_client_gate"]


def test_adoption_evidence_rejects_wrong_expected_client() -> None:
    evidence = build_adoption_evidence(_adoption_report(), required_client="Windsurf")

    assert evidence["status"] == "failed"
    assert evidence["failures"] == [
        "adoption_client_mismatch(Cursor!=Windsurf)",
        "required_adoption_client_mismatch(Cursor!=Windsurf)",
    ]


def test_adoption_client_set_requires_each_expected_client_gate() -> None:
    cursor = build_adoption_evidence(_adoption_report(client="Cursor"))
    windsurf_report = _adoption_report(client="Windsurf")
    windsurf_report["evidence"].pop("required_client")
    windsurf = build_adoption_evidence(windsurf_report)

    evidence = build_adoption_client_set_evidence(
        [cursor, windsurf],
        required_clients=["Cursor", "Windsurf"],
    )

    assert evidence["status"] == "failed"
    assert evidence["required_clients"] == ["Cursor", "Windsurf"]
    assert evidence["observed_clients"] == ["Cursor", "Windsurf"]
    assert evidence["failures"] == [
        "missing_required_adoption_client_gate(Windsurf)"
    ]
    assert adoption_client_set_failure_message(
        evidence,
        prefix="Adoption clients",
    ) == (
        "Adoption clients: ['missing_required_adoption_client_gate(Windsurf)']"
    )


def test_load_adoption_evidence_records_report_hash(tmp_path: Path) -> None:
    report_path = tmp_path / "adoption-report.json"
    report_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")

    evidence = load_adoption_evidence(report_path, required_client="Cursor")

    assert evidence["status"] == "measured"
    assert evidence["artifact_path"] == str(report_path)
    assert evidence["gate_required_client"] == "Cursor"
    assert evidence["artifact_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()


def test_adoption_human_label_link_rejects_different_client_or_session() -> None:
    report = {
        "adoption_evidence": build_adoption_evidence(_adoption_report()),
        "human_label_evidence": {
            "client": "Claude Code",
            "captured_at": "2026-05-18T23:00:00Z",
            "session_id": "claude-session-1",
        },
    }

    linked = link_adoption_to_human_label_evidence(report)

    assert linked["adoption_evidence"]["status"] == "failed"
    assert "human_label_client_mismatch(Claude Code!=Cursor)" in linked[
        "adoption_evidence"
    ]["failures"]
    assert "human_label_session_id_mismatch(claude-session-1!=cursor-thread-1)" in linked[
        "adoption_evidence"
    ]["failures"]


@pytest.mark.asyncio
async def test_evaluate_cli_attaches_and_gates_adoption_report(
    capsys,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    windsurf_path = tmp_path / "windsurf-adoption-report.json"
    human_path = tmp_path / "human-labels.json"
    bundle_path = tmp_path / "brain-loop-release-evidence.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    windsurf_path.write_text(
        json.dumps(_adoption_report(client="Windsurf")),
        encoding="utf-8",
    )
    human_path.write_text(json.dumps(_human_label_artifact()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-client",
            "Cursor",
            "--additional-adoption-report",
            str(windsurf_path),
            "--require-adoption-clients",
            "Cursor",
            "Windsurf",
            "--human-label-artifact",
            str(human_path),
            "--evidence-bundle",
            str(bundle_path),
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert report["adoption_evidence"]["status"] == "measured"
    assert report["adoption_evidence"]["client"] == "Cursor"
    assert report["adoption_client_evidence"]["status"] == "measured"
    assert report["adoption_client_evidence"]["required_clients"] == [
        "Cursor",
        "Windsurf",
    ]
    assert report["adoption_client_evidence"]["observed_clients"] == [
        "Cursor",
        "Windsurf",
    ]
    assert report["additional_adoption_evidence"][0]["client"] == "Windsurf"
    assert report["adoption_evidence"]["artifact_sha256"] == hashlib.sha256(
        adoption_path.read_bytes()
    ).hexdigest()
    assert bundle["sources"]["adoption_report"] == str(adoption_path)
    assert bundle["sources"]["additional_adoption_reports"] == [str(windsurf_path)]
    assert bundle["source_sha256"]["adoption_report"] == hashlib.sha256(
        adoption_path.read_bytes()
    ).hexdigest()
    assert bundle["source_sha256"]["additional_adoption_reports"] == [
        hashlib.sha256(windsurf_path.read_bytes()).hexdigest()
    ]
    assert bundle["source_sha256"]["human_label_artifact"] == hashlib.sha256(
        human_path.read_bytes()
    ).hexdigest()
    assert bundle["gates"]["require_release_evidence"] is True
    assert bundle["status"] == "passed"
    assert bundle["gate_profile"] == "release"
    assert bundle["release_ready"] is True
    assert bundle["gates"]["require_adoption_evidence"] is False
    assert bundle["gates"]["require_adoption_client"] == "Cursor"
    assert bundle["gates"]["require_adoption_clients"] == ["Cursor", "Windsurf"]
    assert bundle["gates"]["require_human_label_evidence"] is False
    assert bundle["gates"]["min_human_recall_samples"] == 10
    assert bundle["gates"]["min_human_session_samples"] == 3
    assert bundle["report"]["adoption_evidence"] == report["adoption_evidence"]


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_missing_required_adoption_client_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    human_path = tmp_path / "human-labels.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    human_path.write_text(json.dumps(_human_label_artifact()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-clients",
            "Cursor",
            "Windsurf",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption client evidence failed gates: "
        "['missing_required_adoption_client(Windsurf)']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_release_gate_rejects_failed_additional_adoption_report(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    blocked_path = tmp_path / "blocked-adoption-report.json"
    human_path = tmp_path / "human-labels.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    blocked = _adoption_report(client="Windsurf")
    blocked["status"] = "failed"
    blocked_path.write_text(json.dumps(blocked), encoding="utf-8")
    human_path.write_text(json.dumps(_human_label_artifact()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--adoption-report",
            str(adoption_path),
            "--additional-adoption-report",
            str(blocked_path),
            "--human-label-artifact",
            str(human_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption client evidence failed gates: "
        "['adoption_report_failed(Windsurf)']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_wrong_required_adoption_client(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    human_path = tmp_path / "human-labels.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    human_path.write_text(json.dumps(_human_label_artifact()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-client",
            "Windsurf",
            "--human-label-artifact",
            str(human_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption evidence failed gates: "
        "['adoption_client_mismatch(Cursor!=Windsurf)', "
        "'required_adoption_client_mismatch(Cursor!=Windsurf)']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_release_gate_requires_production_human_label_counts(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    human_path = tmp_path / "human-labels.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    human_path.write_text(
        json.dumps(_human_label_artifact(recall_count=1, session_count=1)),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--adoption-report",
            str(adoption_path),
            "--human-label-artifact",
            str(human_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Human label evidence failed gates: "
        "['insufficient_human_recall_samples(1<10)', "
        "'insufficient_human_session_samples(1<3)']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_missing_adoption_report_when_required(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-adoption-evidence",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption evidence failed gates: ['missing_adoption_evidence']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_requires_adoption_report_for_client_gate(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-adoption-client",
            "Cursor",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption evidence failed gates: ['missing_adoption_evidence']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_prefills_adoption_metadata(
    capsys,
    tmp_path: Path,
) -> None:
    adoption_path = tmp_path / "adoption-report.json"
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-client",
            "Cursor",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    template = json.loads(capsys.readouterr().out)
    assert template["client"] == "Cursor"
    assert template["capturedAt"] == "2026-05-18T23:00:00Z"
    assert template["sessionId"] == "cursor-thread-1"
    assert template["adoptionReport"]["path"] == str(adoption_path)
    assert template["adoptionReport"]["status"] == "measured"
    assert template["adoptionReport"]["sha256"] == hashlib.sha256(
        adoption_path.read_bytes()
    ).hexdigest()
    assert f"--adoption-report {adoption_path}" in template["validationCommand"]


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_preserves_required_client(
    capsys,
    tmp_path: Path,
) -> None:
    adoption_path = tmp_path / "adoption-report.json"
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-client",
            "Cursor",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    template = json.loads(capsys.readouterr().out)
    assert template["requiredAdoptionClient"] == "Cursor"
    assert template["adoptionReport"]["requiredClient"] == "Cursor"
    assert template["adoptionReport"]["gateRequiredClient"] == "Cursor"
    assert "--require-adoption-client Cursor" in template["validationCommand"]


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_preserves_multi_client_gate(
    capsys,
    tmp_path: Path,
) -> None:
    adoption_path = tmp_path / "cursor-adoption-report.json"
    windsurf_path = tmp_path / "windsurf-adoption-report.json"
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    windsurf_path.write_text(
        json.dumps(_adoption_report(client="Windsurf")),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-client",
            "Cursor",
            "--additional-adoption-report",
            str(windsurf_path),
            "--require-adoption-clients",
            "Cursor",
            "Windsurf",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    template = json.loads(capsys.readouterr().out)
    assert template["requiredAdoptionClients"] == ["Cursor", "Windsurf"]
    assert template["additionalAdoptionReports"][0]["client"] == "Windsurf"
    assert f"--additional-adoption-report {windsurf_path}" in template[
        "validationCommand"
    ]
    assert "--require-adoption-clients Cursor Windsurf" in template[
        "validationCommand"
    ]


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_requires_primary_adoption_report(
    tmp_path: Path,
) -> None:
    windsurf_path = tmp_path / "windsurf-adoption-report.json"
    windsurf_path.write_text(
        json.dumps(_adoption_report(client="Windsurf")),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--additional-adoption-report",
            str(windsurf_path),
            "--require-adoption-clients",
            "Windsurf",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "--additional-adoption-report requires --adoption-report "
        "when generating a human-label template"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_rejects_blocked_additional_report(
    tmp_path: Path,
) -> None:
    adoption_path = tmp_path / "cursor-adoption-report.json"
    windsurf_path = tmp_path / "windsurf-adoption-report.json"
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    windsurf = _adoption_report(client="Windsurf")
    windsurf["evidence"].pop("required_client")
    windsurf_path.write_text(json.dumps(windsurf), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--adoption-report",
            str(adoption_path),
            "--additional-adoption-report",
            str(windsurf_path),
            "--require-adoption-clients",
            "Cursor",
            "Windsurf",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption client evidence failed gates: "
        "['missing_required_adoption_client_gate(Windsurf)']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_human_label_template_rejects_blocked_adoption_report(
    tmp_path: Path,
) -> None:
    adoption_path = tmp_path / "adoption-report.json"
    adoption = _adoption_report()
    adoption["evidence"]["required"] = False
    adoption_path.write_text(json.dumps(adoption), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--adoption-report",
            str(adoption_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Adoption evidence failed gates: ['missing_required_live_evidence_gate']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_release_gate_rejects_missing_human_labels(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    adoption_path = tmp_path / "adoption-report.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    adoption_path.write_text(json.dumps(_adoption_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--adoption-report",
            str(adoption_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Human label evidence failed gates: ['missing_human_label_evidence']"
    )


def test_markdown_includes_adoption_evidence() -> None:
    report = {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T23:10:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "adoption_evidence": build_adoption_evidence(
            _adoption_report(),
            artifact_path=Path("adoption-report.json"),
            artifact_sha256="abc123",
        ),
        "adoption_client_evidence": build_adoption_client_set_evidence(
            [
                build_adoption_evidence(
                    _adoption_report(client="Cursor"),
                    artifact_path=Path("cursor-adoption-report.json"),
                    artifact_sha256="cursor123",
                ),
                build_adoption_evidence(
                    _adoption_report(client="Windsurf"),
                    artifact_path=Path("windsurf-adoption-report.json"),
                    artifact_sha256="windsurf123",
                ),
            ],
            required_clients=["Cursor", "Windsurf"],
        ),
    }

    markdown = format_brain_loop_report_markdown(report)

    assert "## Adoption Evidence" in markdown
    assert "Cursor: measured" in markdown
    assert "## Adoption Client Evidence" in markdown
    assert "Required clients: Cursor, Windsurf | observed clients: Cursor, Windsurf" in markdown
    assert (
        "Report: Cursor measured | cursor-adoption-report.json | sha256 cursor123"
        in markdown
    )
    assert (
        "Report: Windsurf measured | windsurf-adoption-report.json | sha256 windsurf123"
        in markdown
    )
    assert "Calls: 4 | captured 2026-05-18T23:00:00Z | session cursor-thread-1" in markdown
    assert "Artifact: adoption-report.json | sha256 abc123" in markdown


def test_markdown_includes_blocked_adoption_evidence() -> None:
    adoption_report = _adoption_report(
        client="Claude Code",
        session_id="claude-session-blocked",
    )
    adoption_report["status"] = "failed"
    adoption_report["callCount"] = 0
    adoption_report["evidence"]["blockers"] = [
        "mcp_server_failed",
        "authentication_failed",
    ]
    adoption_report["evidence"]["blocker_details"] = [
        "system:error: Not logged in - Please run /login",
    ]
    adoption_report["evidence"]["mcp_server_failures"] = ["engram"]
    adoption_report["validation"]["failures"] = [
        "live_harness_authentication_failed",
        "live_harness_mcp_server_failed",
    ]
    adoption_evidence = build_adoption_evidence(
        adoption_report,
        artifact_path=Path("claude-adoption-report.json"),
        artifact_sha256="blocked123",
    )
    report = {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T23:10:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "adoption_evidence": adoption_evidence,
        "adoption_client_evidence": build_adoption_client_set_evidence(
            [adoption_evidence],
            required_clients=["Claude Code"],
        ),
    }

    markdown = format_brain_loop_report_markdown(report)

    assert "Claude Code: failed" in markdown
    assert "Blockers: mcp_server_failed, authentication_failed" in markdown
    assert "MCP server failures: engram" in markdown
    assert "Blocker details: system:error: Not logged in - Please run /login" in markdown
    assert (
        "Report: Claude Code failed | claude-adoption-report.json | "
        "sha256 blocked123 | blockers mcp_server_failed, authentication_failed"
    ) in markdown


def _adoption_report(
    *,
    client: str = "Cursor",
    session_id: str | None = None,
) -> dict:
    normalized = "-".join(client.lower().split())
    session_id = session_id or f"{normalized}-thread-1"
    return {
        "status": "passed",
        "authorityPath": "claim-authority.json",
        "callsPath": "live-harness-transcript.json",
        "callCount": 4,
        "evidence": {
            "required": True,
            "client": client,
            "required_client": client,
            "captured_at": "2026-05-18T23:00:00Z",
            "session_id": session_id,
            "session_filter": session_id,
            "source": "copied_mcp_log",
            "missing": [],
        },
        "validation": {
            "required_tools_before_answer": {
                "expected": ["claim_authority", "get_context", "recall"],
                "observed": ["claim_authority", "get_context", "recall"],
                "missing": [],
                "in_order": True,
            },
            "capture": {
                "destination": "engram",
                "expected_tool": "remember",
                "observed_tools": ["remember"],
                "missing": False,
            },
            "file_memory": {
                "present": True,
                "observed_tools": [],
                "substituted_for_engram": False,
            },
            "failures": [],
        },
    }


def _human_label_artifact(*, recall_count: int = 10, session_count: int = 3) -> dict:
    return {
        "kind": "engram_human_label_evidence",
        "humanLabeled": True,
        "source": "staging_harness",
        "client": "Cursor",
        "capturedAt": "2026-05-18T23:00:00Z",
        "sessionId": "cursor-thread-1",
        "labeler": "operator",
        "recallSamples": [
            {
                "source": "staging_harness",
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 2,
                "packetsUsed": 1,
                "falseRecalls": 0,
            }
            for _index in range(recall_count)
        ],
        "sessionSamples": [
            {
                "source": "staging_harness",
                "baselineScore": 0.2,
                "memoryScore": 0.8,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": True,
            }
            for _index in range(session_count)
        ],
    }


def _measured_report() -> dict:
    return {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T23:01:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "evaluation_signals": {
            signal: {
                "status": "measured",
                "evidence_count": 2,
                "metric": 1.0,
                "gap": None,
            }
            for signal in EVALUATION_SIGNAL_ORDER
        },
        "coverage_gaps": [],
    }
