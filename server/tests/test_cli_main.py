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

    async def fake_run_evaluate_command(args) -> None:
        calls.append(
            {
                "command": args.command,
                "from_json": args.from_json,
                "require_evaluation_signals": args.require_evaluation_signals,
                "format": args.format,
            }
        )

    monkeypatch.setattr(
        "sys.argv",
        [
            "engram",
            "evaluate",
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
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
            "require_evaluation_signals": True,
            "format": "json",
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
