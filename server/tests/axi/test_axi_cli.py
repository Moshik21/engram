from __future__ import annotations

import argparse
import io
import json
import os

from engram.axi.cli import _normalize_project_path, configure_axi_parser, run_axi_command


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

    def packet_cache(self) -> dict:
        return {
            "status": "ok",
            "packetCache": {
                "entryCount": 2,
                "freshCount": 2,
                "invalidatedCount": 0,
                "expiredCount": 0,
                "hitCount": 4,
                "scopes": {"project_home": 2},
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


def test_normalize_project_path_resolves_tmp_symlink() -> None:
    left = _normalize_project_path("/tmp/engram-followup-test")
    right = _normalize_project_path("/private/tmp/engram-followup-test")
    assert left == right


def test_run_axi_hook_run_resolves_tmp_symlink_from_stdin(monkeypatch, capsys) -> None:
    captured: dict[str, str | None] = {}

    class CapturingClient(HealthyClient):
        def runtime(self, *, project_path: str | None = None, **kwargs: object) -> dict:
            captured["project_path"] = project_path
            return super().runtime(project_path=project_path)

        def runtime_fast(self, *, project_path: str | None = None) -> dict:
            captured["project_path"] = project_path
            return {
                "runtime": {"mode": "helix"},
                "artifactBootstrap": {
                    "projectPath": project_path,
                    "artifactCount": 1,
                },
                "agentAdoption": {"status": "ready", "requiredNextTools": ["get_context"]},
            }

    monkeypatch.setattr(
        "engram.axi.cli.sys.stdin",
        io.StringIO(json.dumps({"cwd": "/tmp/engram-followup-test"})),
    )
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: CapturingClient())
    args = _parse_axi_args("hook-run", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    expected = _normalize_project_path("/tmp/engram-followup-test")
    assert captured["project_path"] == expected
    payload = json.loads(capsys.readouterr().out)
    assert payload["brain"]["project"] == expected


def test_axi_parser_preserves_global_flags_before_subcommand() -> None:
    args = _parse_axi_args("--json", "--budget", "500", "context", "--topic", "Engram")

    assert args.command == "axi"
    assert args.axi_command == "context"
    assert args.json is True
    assert args.budget == 500
    assert args.topic_hint == "Engram"


def test_axi_parser_preserves_global_project_and_topic_before_context() -> None:
    args = _parse_axi_args(
        "--project",
        "/tmp/global-project",
        "--topic",
        "global topic",
        "context",
    )

    assert args.axi_command == "context"
    assert args.project_path == "/tmp/global-project"
    assert args.topic_hint == "global topic"


def test_axi_parser_preserves_global_project_before_duplicate_subcommands() -> None:
    recall_args = _parse_axi_args("--project", "/tmp/global-project", "recall", "query")
    doctor_args = _parse_axi_args("--project", "/tmp/global-project", "doctor")

    assert recall_args.project_path == "/tmp/global-project"
    assert doctor_args.project_path == "/tmp/global-project"


def test_axi_parser_subcommand_project_and_topic_override_global_values() -> None:
    args = _parse_axi_args(
        "--project",
        "/tmp/global-project",
        "--topic",
        "global topic",
        "context",
        "--project",
        "/tmp/subcommand-project",
        "--topic",
        "subcommand topic",
    )

    assert args.project_path == "/tmp/subcommand-project"
    assert args.topic_hint == "subcommand topic"


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

        def evaluation_report(self, *, live_cost: bool = False) -> dict:
            captured["live_cost"] = live_cost
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
    assert captured["live_cost"] is True
    assert json.loads(capsys.readouterr().out)["operation"] == "value"


def test_axi_parser_accepts_packet_cache_clear() -> None:
    args = _parse_axi_args("packet-cache", "clear", "--json")

    assert args.axi_command == "packet-cache"
    assert args.packet_cache_command == "clear"
    assert args.json is True


def test_axi_parser_defaults_packet_cache_to_summary() -> None:
    args = _parse_axi_args("packet-cache", "--json")

    assert args.axi_command == "packet-cache"
    assert args.packet_cache_command == "summary"
    assert args.json is True


def test_run_axi_command_prints_json_home(monkeypatch, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    args = _parse_axi_args("--json", "--project", "/tmp/project")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "healthy"
    assert payload["mode"] == "helix"
    assert payload["brain"]["project"] == _normalize_project_path("/tmp/project")


def test_run_axi_context_infers_project_from_cwd(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}

    class ContextClient(HealthyClient):
        def context(self, **kwargs: object) -> dict:
            captured.update(kwargs)
            return {"context": "Engram context", "entityCount": 1, "factCount": 1}

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sample'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: ContextClient())
    args = _parse_axi_args("context", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured["project_path"] == str(tmp_path)


def test_run_axi_recall_infers_project_from_cwd(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}

    class RecallClient(HealthyClient):
        def recall(
            self,
            query_text: str,
            *,
            limit: int,
            project_path: str | None = None,
        ) -> dict:
            captured["query"] = query_text
            captured["limit"] = limit
            captured["project_path"] = project_path
            return {"status": "ok", "results": []}

    (tmp_path / "README.md").write_text("# Sample\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: RecallClient())
    args = _parse_axi_args("recall", "native PyO3 recall", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured == {
        "query": "native PyO3 recall",
        "limit": 5,
        "project_path": str(tmp_path),
    }


def test_run_axi_context_leaves_project_empty_outside_project(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    captured: dict[str, object] = {}

    class ContextClient(HealthyClient):
        def context(self, **kwargs: object) -> dict:
            captured.update(kwargs)
            return {"context": "Engram context", "entityCount": 1, "factCount": 1}

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: ContextClient())
    args = _parse_axi_args("context", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    json.loads(capsys.readouterr().out)
    assert captured["project_path"] is None


def test_run_axi_hook_run_uses_hook_stdin_cwd(monkeypatch, tmp_path, capsys) -> None:
    (tmp_path / "README.md").write_text("# Hook project\n")
    monkeypatch.setattr(
        "engram.axi.cli.sys.stdin",
        io.StringIO(json.dumps({"cwd": str(tmp_path)})),
    )
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    args = _parse_axi_args("hook-run", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["brain"]["project"] == str(tmp_path)


def test_run_axi_hook_run_ignores_filesystem_root_cwd(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "engram.axi.cli.sys.stdin",
        io.StringIO(json.dumps({"cwd": "/"})),
    )
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    args = _parse_axi_args("hook-run", "--json")

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["brain"]["project"] is None


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


def test_run_axi_packet_cache_summary_prints_json_and_traces(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    trace_file = tmp_path / "axi-runs.jsonl"
    args = _parse_axi_args(
        "packet-cache",
        "--json",
        "--trace-file",
        str(trace_file),
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "packet-cache.summary"
    assert payload["packet_cache"]["entry_count"] == 2
    trace = json.loads(trace_file.read_text().splitlines()[0])
    assert trace["operation"] == "packet-cache.summary"
    assert trace["status"] == "ok"


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
    assert trace["project"] == _normalize_project_path("/tmp/project")
    assert "context" not in trace
    assert "brain" not in trace


def test_run_axi_context_trace_records_redaction_safe_usefulness_metadata(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class ContextTraceClient(HealthyClient):
        def context(self, **_kwargs: object) -> dict:
            return {
                "context": "Engram context",
                "entityCount": 2,
                "factCount": 3,
                "packet_cache": {"hit": True, "packet_count": 2},
                "cached_packets": [
                    {"title": "Project", "summary": "project packet"},
                    {"title": "Identity", "summary": "identity packet"},
                ],
            }

    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: ContextTraceClient())
    trace_file = tmp_path / "axi-runs.jsonl"
    args = _parse_axi_args(
        "context",
        "--json",
        "--trace-file",
        str(trace_file),
        "--trace-client",
        "codex",
        "--trace-origin",
        "agent-followup",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    capsys.readouterr()
    trace = json.loads(trace_file.read_text().splitlines()[0])
    assert trace["operation"] == "context"
    assert trace["origin"] == "agent-followup"
    assert trace["cacheHit"] is True
    assert trace["packetCount"] == 2
    assert trace["entityCount"] == 2
    assert trace["factCount"] == 3
    assert trace["degraded"] is False
    assert "context" not in trace
    assert "packets" not in trace


def test_run_axi_recall_trace_records_redaction_safe_lifecycle_metadata(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    class RecallTraceClient(HealthyClient):
        def recall(
            self,
            _query: str,
            *,
            limit: int,
            project_path: str | None = None,
        ) -> dict:
            del limit, project_path
            return {
                "status": "ok",
                "results": [],
                "packets": [
                    {"title": "Project", "summary": "project packet"},
                    {"title": "Plan", "summary": "plan packet"},
                    {"title": "Install", "summary": "install packet"},
                ],
                "lifecycle": {
                    "resultCount": 0,
                    "packetCount": 3,
                    "fallbackStatus": "cache_satisfied",
                    "skipReason": "cache_satisfied",
                    "degraded": False,
                },
                "budget": {"budgetMiss": False, "degraded": False},
            }

    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: RecallTraceClient())
    trace_file = tmp_path / "axi-runs.jsonl"
    args = _parse_axi_args(
        "recall",
        "Engram performance",
        "--json",
        "--trace-file",
        str(trace_file),
        "--trace-client",
        "codex",
        "--trace-origin",
        "agent-followup",
    )

    exit_code = run_axi_command(args)

    assert exit_code == 0
    capsys.readouterr()
    trace = json.loads(trace_file.read_text().splitlines()[0])
    assert trace["operation"] == "recall"
    assert trace["cacheHit"] is True
    assert trace["packetCount"] == 3
    assert trace["resultCount"] == 0
    assert trace["fallbackStatus"] == "cache_satisfied"
    assert trace["skipReason"] == "cache_satisfied"
    assert trace["budgetMiss"] is False
    assert trace["degraded"] is False
    assert "results" not in trace
    assert "packets" not in trace


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
            "last_run_stale_after_config_change": False,
            "last_run_project_root": False,
            "last_observed_run": None,
            "last_followup": None,
            "followup_summary": {
                "status": "missing",
                "trace_count": 0,
                "sample_limit": 20,
            },
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
                "timestamp": "2099-05-20T22:34:44Z",
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


def test_run_axi_doctor_rejects_stale_hook_run_evidence(monkeypatch, tmp_path, capsys) -> None:
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
    hook_path = tmp_path / ".codex/hooks.json"
    new_config_time = 1_779_916_000
    hook_path.touch()

    os.utime(hook_path, (new_config_time, new_config_time))
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
    assert {"name": "hook:codex", "status": "fail", "detail": "stale_session_start_run"} in payload[
        "checks"
    ]
    assert payload["hooks"][0]["last_run_stale_after_config_change"] is True


def test_run_axi_doctor_rejects_root_hook_run_project(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("engram.axi.cli.AxiRestClient", lambda **_kwargs: HealthyClient())
    install_args = _parse_axi_args("hooks", "install", "codex", "--home", str(tmp_path))
    assert run_axi_command(install_args) == 0
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        json.dumps(
            {
                "timestamp": "2099-05-20T22:34:44Z",
                "hookId": "engram-axi-context",
                "client": "codex",
                "operation": "home",
                "status": "healthy",
                "exitCode": 0,
                "durationMs": 120,
                "origin": "session-start-hook",
                "project": "/",
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
        "detail": "session_start_project_root",
    } in payload["checks"]
    assert payload["hooks"][0]["last_run_project_root"] is True


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
                        "timestamp": "2099-05-20T22:34:44Z",
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
                        "timestamp": "2099-05-20T22:35:01Z",
                        "hookId": "engram-axi-context",
                        "client": "codex",
                        "operation": "context",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 92,
                        "origin": "agent-followup",
                        "project": "/tmp/project",
                        "cacheHit": True,
                        "packetCount": 2,
                        "resultCount": 0,
                        "fallbackStatus": "cache_satisfied",
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
    assert payload["hooks"][0]["last_followup"]["cacheHit"] is True
    assert payload["hooks"][0]["last_followup"]["packetCount"] == 2
    assert payload["hooks"][0]["last_followup"]["resultCount"] == 0
    assert payload["hooks"][0]["last_followup"]["fallbackStatus"] == "cache_satisfied"


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
                        "timestamp": "2099-05-20T22:34:44Z",
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
                        "timestamp": "2099-05-20T22:35:01Z",
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
