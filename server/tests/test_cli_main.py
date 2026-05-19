from __future__ import annotations

import importlib
import os

import pytest


def test_serve_mode_helix_sets_runtime_env(monkeypatch) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[tuple[str, dict]] = []

    def fake_run(app: str, **kwargs) -> None:
        calls.append((app, kwargs))

    monkeypatch.delenv("ENGRAM_MODE", raising=False)
    monkeypatch.setattr("sys.argv", ["engram", "serve", "--mode", "helix"])
    monkeypatch.setattr("uvicorn.run", fake_run)

    cli.main()

    assert calls == [
        ("engram.main:app", {"host": "0.0.0.0", "port": 8100, "log_level": "info"})
    ]
    assert calls[0][1]["port"] == 8100
    assert calls[0][1]["host"] == "0.0.0.0"
    assert calls[0][1]["log_level"] == "info"
    assert os.environ["ENGRAM_MODE"] == "helix"


def test_serve_native_data_dir_sets_runtime_env(monkeypatch, tmp_path) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[tuple[str, dict]] = []
    native_dir = tmp_path / "native-data"

    def fake_run(app: str, **kwargs) -> None:
        calls.append((app, kwargs))

    monkeypatch.delenv("ENGRAM_MODE", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__TRANSPORT", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__DATA_DIR", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "serve",
            "--mode",
            "helix",
            "--helix-data-dir",
            str(native_dir),
        ],
    )
    monkeypatch.setattr("uvicorn.run", fake_run)

    cli.main()

    assert calls == [
        ("engram.main:app", {"host": "0.0.0.0", "port": 8100, "log_level": "info"})
    ]
    assert os.environ["ENGRAM_MODE"] == "helix"
    assert os.environ["ENGRAM_HELIX__TRANSPORT"] == "native"
    assert os.environ["ENGRAM_HELIX__DATA_DIR"] == str(native_dir)


def test_mcp_native_data_dir_sets_runtime_env(monkeypatch, tmp_path) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[dict] = []
    native_dir = tmp_path / "native-data"

    def fake_mcp_main(**kwargs) -> None:
        calls.append(kwargs)

    monkeypatch.delenv("ENGRAM_MODE", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__TRANSPORT", raising=False)
    monkeypatch.delenv("ENGRAM_HELIX__DATA_DIR", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "mcp",
            "--mode",
            "helix",
            "--helix-data-dir",
            str(native_dir),
            "--transport",
            "stdio",
        ],
    )
    monkeypatch.setattr("engram.mcp.server.main", fake_mcp_main)

    cli.main()

    assert calls == [{"transport": "stdio", "host": "127.0.0.1", "port": 8200}]
    assert os.environ["ENGRAM_MODE"] == "helix"
    assert os.environ["ENGRAM_HELIX__TRANSPORT"] == "native"
    assert os.environ["ENGRAM_HELIX__DATA_DIR"] == str(native_dir)


def test_evaluate_require_evaluation_signals_dispatches_to_command(
    monkeypatch,
    tmp_path,
) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[dict] = []
    report_path = tmp_path / "brain-loop-report.json"
    benchmark_path = tmp_path / "showcase-results.json"
    human_path = tmp_path / "human-labels.json"
    adoption_path = tmp_path / "adoption-report.json"
    windsurf_adoption_path = tmp_path / "windsurf-adoption-report.json"
    bundle_path = tmp_path / "brain-loop-evidence.json"

    async def fake_run_evaluate_command(args) -> None:
        calls.append(
            {
                "command": args.command,
                "from_json": args.from_json,
                "require_evaluation_signals": args.require_evaluation_signals,
                "min_evaluation_signal_evidence": args.min_evaluation_signal_evidence,
                "benchmark_artifact": args.benchmark_artifact,
                "require_benchmark_evidence": args.require_benchmark_evidence,
                "min_benchmark_scenarios": args.min_benchmark_scenarios,
                "min_benchmark_pass_rate": args.min_benchmark_pass_rate,
                "human_label_artifact": args.human_label_artifact,
                "human_label_template": args.human_label_template,
                "require_human_label_evidence": args.require_human_label_evidence,
                "adoption_report": args.adoption_report,
                "require_adoption_evidence": args.require_adoption_evidence,
                "require_adoption_client": args.require_adoption_client,
                "additional_adoption_report": args.additional_adoption_report,
                "require_adoption_clients": args.require_adoption_clients,
                "min_human_recall_samples": args.min_human_recall_samples,
                "min_human_session_samples": args.min_human_session_samples,
                "evidence_bundle": args.evidence_bundle,
                "format": args.format,
                "require_release_evidence": args.require_release_evidence,
            }
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "evaluate",
            "--from-json",
            str(report_path),
            "--require-release-evidence",
            "--min-evaluation-signal-evidence",
            "4",
            "--benchmark-artifact",
            str(benchmark_path),
            "--require-benchmark-evidence",
            "--min-benchmark-scenarios",
            "6",
            "--min-benchmark-pass-rate",
            "0.8",
            "--human-label-artifact",
            str(human_path),
            "--require-human-label-evidence",
            "--adoption-report",
            str(adoption_path),
            "--require-adoption-evidence",
            "--require-adoption-client",
            "Cursor",
            "--additional-adoption-report",
            str(windsurf_adoption_path),
            "--require-adoption-clients",
            "Cursor",
            "Windsurf",
            "--min-human-recall-samples",
            "5",
            "--min-human-session-samples",
            "2",
            "--evidence-bundle",
            str(bundle_path),
            "--format",
            "json",
        ],
    )
    monkeypatch.setattr(
        "engram.evaluation.cli.run_evaluate_command",
        fake_run_evaluate_command,
    )

    cli.main()

    assert calls == [
        {
            "command": "evaluate",
            "from_json": report_path,
            "require_evaluation_signals": False,
            "require_release_evidence": True,
            "min_evaluation_signal_evidence": 4,
            "benchmark_artifact": benchmark_path,
            "require_benchmark_evidence": True,
            "min_benchmark_scenarios": 6,
            "min_benchmark_pass_rate": 0.8,
            "human_label_artifact": human_path,
            "human_label_template": False,
            "require_human_label_evidence": True,
            "adoption_report": adoption_path,
            "require_adoption_evidence": True,
            "require_adoption_client": "Cursor",
            "additional_adoption_report": [windsurf_adoption_path],
            "require_adoption_clients": ["Cursor", "Windsurf"],
            "min_human_recall_samples": 5,
            "min_human_session_samples": 2,
            "evidence_bundle": bundle_path,
            "format": "json",
        }
    ]


def test_adoption_dispatches_to_transcript_validator(monkeypatch, tmp_path) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[dict] = []
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "mcp-calls.jsonl"
    report_path = tmp_path / "adoption-report.json"

    def fake_run_adoption_command(args) -> int:
        calls.append(
            {
                "command": args.command,
                "authority": args.authority,
                "calls": args.calls,
                "template": args.template,
                "format": args.format,
                "report_out": args.report_out,
                "require_live_evidence": args.require_live_evidence,
                "require_client": args.require_client,
            }
        )
        return 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "adoption",
            "--authority",
            str(authority_path),
            "--calls",
            str(calls_path),
            "--require-live-evidence",
            "--require-client",
            "Cursor",
            "--report-out",
            str(report_path),
            "--format",
            "markdown",
        ],
    )
    monkeypatch.setattr(
        "engram.mcp.adoption_cli.run_adoption_command",
        fake_run_adoption_command,
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == [
        {
            "command": "adoption",
            "authority": authority_path,
            "calls": [calls_path],
            "template": False,
            "format": "markdown",
            "report_out": report_path,
            "require_live_evidence": True,
            "require_client": "Cursor",
        }
    ]


def test_adoption_template_dispatches_to_command(monkeypatch, tmp_path) -> None:
    cli = importlib.import_module("engram.__main__")
    calls: list[dict] = []
    authority_path = tmp_path / "claim-authority.json"

    def fake_run_adoption_command(args) -> int:
        calls.append(
            {
                "command": args.command,
                "authority": args.authority,
                "calls": args.calls,
                "template": args.template,
                "client": args.client,
                "captured_at": args.captured_at,
                "format": args.format,
            }
        )
        return 0

    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "adoption",
            "--authority",
            str(authority_path),
            "--template",
            "--client",
            "Claude Code",
            "--captured-at",
            "2026-05-18T22:31:00Z",
            "--format",
            "markdown",
        ],
    )
    monkeypatch.setattr(
        "engram.mcp.adoption_cli.run_adoption_command",
        fake_run_adoption_command,
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    assert calls == [
        {
            "command": "adoption",
            "authority": authority_path,
            "calls": None,
            "template": True,
            "client": "Claude Code",
            "captured_at": "2026-05-18T22:31:00Z",
            "format": "markdown",
        }
    ]


def test_top_level_help_mentions_doctor_readiness_smoke(monkeypatch, capsys) -> None:
    cli = importlib.import_module("engram.__main__")
    monkeypatch.setattr("sys.argv", ["engram", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "engram doctor" in output
    assert "engram adoption" in output
    assert "readiness smoke" in output


def test_doctor_help_mentions_evaluation_signal_readiness(monkeypatch, capsys) -> None:
    cli = importlib.import_module("engram.__main__")
    monkeypatch.setattr("sys.argv", ["engram", "doctor", "--help"])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "Run local diagnostics, lifecycle snapshot, and brain-loop smoke" in output
    assert "evaluation-signal readiness" in output
