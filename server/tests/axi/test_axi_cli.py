from __future__ import annotations

import argparse
import json

from engram.axi.cli import configure_axi_parser, run_axi_command


class HealthyClient:
    server_url = "http://127.0.0.1:8100"

    def health(self) -> dict:
        return {"status": "healthy", "mode": "helix"}

    def runtime(self, *, project_path: str | None = None) -> dict:
        return {
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {"projectPath": project_path, "artifactCount": 1},
            "agentAdoption": {"status": "ready", "requiredNextTools": ["get_context"]},
        }

    def storage(self, **_kwargs) -> dict:
        return {
            "mode": "helix",
            "backend": "helix_native",
            "counts": {"episodes": 1, "entities": 1, "relationships": 0, "cues": 1},
            "disk": {"humanSize": "1.0 MB"},
            "paths": [
                {
                    "label": "Helix native data",
                    "path": "/tmp/helix",
                    "exists": True,
                    "humanSize": "1.0 MB",
                }
            ],
        }

    def context(self, **_kwargs) -> dict:
        return {"context": "Engram context", "entityCount": 1, "factCount": 1}

    def clear_packet_cache(self) -> dict:
        return {
            "status": "cleared",
            "clearedCount": 2,
            "packetCache": {
                "entryCount": 0,
                "freshCount": 0,
                "hitCount": 4,
                "persistent": True,
                "path": "/tmp/engram-packet-cache.sqlite3",
            },
        }


def _parse_axi_args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    axi_parser = subparsers.add_parser("axi")
    configure_axi_parser(axi_parser)
    return parser.parse_args(["axi", *argv])


def test_axi_parser_preserves_global_flags_before_subcommand() -> None:
    args = _parse_axi_args("--json", "--budget", "500", "context", "--topic", "Engram")

    assert args.command == "axi"
    assert args.axi_command == "context"
    assert args.json is True
    assert args.budget == 500
    assert args.topic_hint == "Engram"


def test_axi_parser_accepts_common_flags_after_subcommand() -> None:
    args = _parse_axi_args("recall", "Engram", "--json", "--limit", "3")

    assert args.axi_command == "recall"
    assert args.query == "Engram"
    assert args.limit == 3
    assert args.json is True


def test_axi_value_uses_report_timeout_default_without_overriding_explicit_timeout() -> None:
    value_args = _parse_axi_args("value")
    explicit_subcommand_args = _parse_axi_args("value", "--timeout", "7")
    explicit_global_args = _parse_axi_args("--timeout", "3", "value")

    assert value_args.timeout is None
    assert value_args._axi_timeout_default == 20.0
    assert explicit_subcommand_args.timeout == 7.0
    assert explicit_global_args.timeout == 3.0


def test_run_axi_value_uses_report_timeout_default(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    class ValueClient:
        server_url = "http://127.0.0.1:8100"

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def evaluation_report(self) -> dict:
            return {
                "memory_value": {
                    "status": "measured",
                    "cost": {
                        "operation_count": 1,
                        "p95_added_latency_ms": 12,
                    },
                    "benefit": {
                        "memory_need_precision": 1,
                        "useful_packet_rate": 1,
                        "session_continuity_lift": 0.5,
                    },
                }
            }

    monkeypatch.setattr("engram.axi.cli.AxiRestClient", ValueClient)
    args = _parse_axi_args("value", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    assert captured["timeout_seconds"] == 20.0
    assert json.loads(capsys.readouterr().out)["operation"] == "value"


def test_axi_parser_accepts_packet_cache_clear() -> None:
    args = _parse_axi_args("packet-cache", "clear", "--json")

    assert args.axi_command == "packet-cache"
    assert args.packet_cache_command == "clear"
    assert args.json is True


def test_run_axi_command_prints_json_home(monkeypatch, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    args = _parse_axi_args("--json", "--project", "/tmp/project")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "healthy"
    assert payload["mode"] == "helix"
    assert payload["brain"]["project"] == "/tmp/project"


def test_run_axi_packet_cache_clear_prints_json_and_traces(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    trace_file = tmp_path / "axi-runs.jsonl"
    args = _parse_axi_args(
        "packet-cache",
        "clear",
        "--json",
        "--trace-file",
        str(trace_file),
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "packet-cache.clear"
    assert payload["cleared_count"] == 2
    trace = json.loads(trace_file.read_text().splitlines()[0])
    assert trace["operation"] == "packet-cache.clear"
    assert trace["status"] == "cleared"


def test_run_axi_command_writes_metadata_only_trace(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    trace_file = tmp_path / "axi-runs.jsonl"
    args = _parse_axi_args(
        "--json",
        "--project",
        "/tmp/project",
        "--trace-file",
        str(trace_file),
        "--trace-client",
        "codex",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    trace = json.loads(trace_file.read_text().splitlines()[0])
    assert trace["hookId"] == "engram-axi-context"
    assert trace["client"] == "codex"
    assert trace["origin"] == "manual"
    assert trace["operation"] == "home"
    assert trace["status"] == output["status"]
    assert trace["project"] == "/tmp/project"
    assert "context" not in trace
    assert "brain" not in trace


def test_run_axi_hooks_install_dry_run_prints_json(tmp_path, capsys) -> None:
    args = _parse_axi_args(
        "hooks",
        "install",
        "codex",
        "--home",
        str(tmp_path),
        "--dry-run",
        "--json",
        "--engram-command",
        "/tmp/engram",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "hooks.install"
    assert payload["client"] == "codex"
    assert payload["dry_run"] is True
    assert payload["capture"] is False
    assert payload["command"].startswith("/tmp/engram axi ")
    assert "--timeout 3" in payload["command"]
    assert not (tmp_path / ".codex/hooks.json").exists()


def test_run_axi_hooks_status_prints_json(tmp_path, capsys) -> None:
    args = _parse_axi_args(
        "hooks",
        "status",
        "codex",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "hooks.status"
    assert payload["status"] == "missing"
    assert payload["ready"] is False


def test_run_axi_doctor_can_verify_ready_hooks(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args(
        "hooks",
        "install",
        "codex",
        "--home",
        str(tmp_path),
    )
    assert run_axi_command(install_args) == 0
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "doctor"
    assert payload["status"] == "pass"
    assert payload["hooks"] == [
        {
            "client": "codex",
            "status": "installed",
            "ready": True,
            "read_only": True,
            "capture": False,
            "last_run": None,
            "last_observed_run": None,
            "last_followup": None,
            "issues": [],
        }
    ]
    assert {"name": "hook:codex", "status": "pass"} in payload["checks"]


def test_run_axi_doctor_can_require_hook_run_evidence(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-20T22:34:44Z",
                "hookId": "engram-axi-context",
                "client": "codex",
                "operation": "home",
                "status": "healthy",
                "exitCode": 0,
                "durationMs": 120,
                "origin": "session-start-hook",
                "project": "/tmp/project",
            }
        )
        + "\n"
    )
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--require-hook-run",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"name": "hook:codex", "status": "pass"} in payload["checks"]
    assert payload["hooks"][0]["last_run"]["duration_ms"] == 120
    assert payload["hooks"][0]["last_run"]["origin"] == "session-start-hook"
    assert payload["hooks"][0]["last_observed_run"]["origin"] == "session-start-hook"
    assert payload["hooks"][0]["last_followup"] is None


def test_run_axi_doctor_can_require_followup_evidence(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-20T22:34:44Z",
                        "hookId": "engram-axi-context",
                        "client": "codex",
                        "operation": "home",
                        "status": "healthy",
                        "exitCode": 0,
                        "durationMs": 120,
                        "origin": "session-start-hook",
                        "project": "/tmp/project",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T22:35:01Z",
                        "hookId": "engram-axi-context",
                        "client": "codex",
                        "operation": "context",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 92,
                        "origin": "agent-followup",
                        "project": "/tmp/project",
                    }
                ),
            ]
        )
        + "\n"
    )
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--require-hook-run",
        "--require-followup",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert {"name": "hook:codex", "status": "pass"} in payload["checks"]
    assert payload["hooks"][0]["last_run"]["origin"] == "session-start-hook"
    assert payload["hooks"][0]["last_observed_run"]["origin"] == "session-start-hook"
    assert payload["hooks"][0]["last_followup"]["operation"] == "context"
    assert payload["hooks"][0]["last_followup"]["origin"] == "agent-followup"


def test_run_axi_doctor_fails_when_required_followup_is_missing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-20T22:34:44Z",
                        "hookId": "engram-axi-context",
                        "client": "codex",
                        "origin": "session-start-hook",
                        "operation": "home",
                        "status": "healthy",
                        "exitCode": 0,
                        "durationMs": 120,
                        "project": "/tmp/project",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T22:35:01Z",
                        "hookId": "engram-axi-context",
                        "client": "codex",
                        "origin": "agent-followup",
                        "operation": "context",
                        "status": "error",
                        "exitCode": 1,
                        "durationMs": 10000,
                        "project": "/tmp/project",
                    }
                ),
            ]
        )
        + "\n"
    )
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--require-hook-run",
        "--require-followup",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert {
        "name": "hook:codex",
        "status": "fail",
        "detail": "missing_followup",
    } in payload["checks"]
    assert payload["hooks"][0]["last_followup"] is None


def test_run_axi_doctor_rejects_manual_trace_when_hook_run_required(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-20T22:34:44Z",
                "hookId": "engram-axi-context",
                "client": "codex",
                "origin": "manual",
                "operation": "home",
                "status": "healthy",
                "exitCode": 0,
                "durationMs": 120,
                "project": "/tmp/project",
            }
        )
        + "\n"
    )
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--require-hook-run",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert {
        "name": "hook:codex",
        "status": "fail",
        "detail": "missing_session_start_origin",
    } in payload["checks"]
    assert payload["hooks"][0]["last_run"] is None
    assert payload["hooks"][0]["last_observed_run"]["origin"] == "manual"


def test_run_axi_doctor_fails_when_required_hook_run_is_missing(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    capsys.readouterr()
    doctor_args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--require-hook-run",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(doctor_args)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert {
        "name": "hook:codex",
        "status": "fail",
        "detail": "missing_last_run",
    } in payload["checks"]


def test_run_axi_doctor_fails_when_requested_hook_is_missing(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    args = _parse_axi_args(
        "doctor",
        "--hooks",
        "codex",
        "--home",
        str(tmp_path),
        "--json",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "fail"
    assert {"name": "hook:codex", "status": "fail", "detail": "missing"} in payload["checks"]
