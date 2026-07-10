from __future__ import annotations

import json

from engram.axi.hooks import (
    HOOK_TRACE_ORIGIN,
    MANAGED_HOOK_ID,
    build_hook_command,
    build_hook_print_payload,
    build_hook_status_payload,
    install_hook,
    uninstall_hook,
)


def test_hook_command_is_read_only_session_start_packet() -> None:
    command = build_hook_command(
        server_url="http://127.0.0.1:8100",
        budget=500,
        timeout_seconds=3,
    )

    assert command == (
        "engram axi hook-run --server-url http://127.0.0.1:8100 --budget 500 --timeout 3"
    )
    assert "observe" not in command
    assert "remember" not in command


def test_hook_command_can_pin_engram_executable_path() -> None:
    command = build_hook_command(
        engram_command="/opt/Engram Tools/engram",
        server_url="http://127.0.0.1:8100",
        budget=500,
        timeout_seconds=3,
    )

    assert command == (
        "'/opt/Engram Tools/engram' axi hook-run "
        "--server-url http://127.0.0.1:8100 --budget 500 --timeout 3"
    )


def test_hook_command_can_write_metadata_only_trace() -> None:
    command = build_hook_command(
        server_url="http://127.0.0.1:8100",
        budget=500,
        timeout_seconds=3,
        trace_file="/tmp/axi-runs.jsonl",
        trace_client="codex",
        trace_origin=HOOK_TRACE_ORIGIN,
    )

    assert command == (
        "engram axi hook-run --server-url http://127.0.0.1:8100 "
        "--budget 500 --timeout 3 "
        "--trace-file /tmp/axi-runs.jsonl --trace-client codex "
        "--trace-origin session-start-hook"
    )


def test_hook_print_payload_does_not_write_config(tmp_path) -> None:
    result = build_hook_print_payload("codex", home=tmp_path)

    assert result.exit_code == 0
    assert result.payload["operation"] == "hooks.print"
    assert result.payload["capture"] is False
    assert result.payload["trace_file"] == str(tmp_path / ".engram/axi-hook-runs.jsonl")
    managed = result.payload["config"]["hooks"]["SessionStart"][0]["hooks"][0]
    assert managed["id"] == MANAGED_HOOK_ID
    assert managed["type"] == "command"
    assert not (tmp_path / ".codex/hooks.json").exists()


def test_hook_status_reports_missing_without_writing_config(tmp_path) -> None:
    result = build_hook_status_payload("codex", home=tmp_path)

    assert result.exit_code == 0
    assert result.payload["operation"] == "hooks.status"
    assert result.payload["status"] == "missing"
    assert result.payload["installed"] is False
    assert result.payload["ready"] is False
    assert result.payload["next"][0]["cmd"] == "engram axi hooks install codex"
    assert not (tmp_path / ".codex/hooks.json").exists()


def test_install_codex_hook_preserves_existing_config_and_is_idempotent(tmp_path) -> None:
    target = tmp_path / ".codex/hooks.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "other": "preserved",
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "",
                            "hooks": [
                                {
                                    "id": "user-hook",
                                    "type": "command",
                                    "command": "echo user",
                                }
                            ],
                        },
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "id": "stop-hook",
                                    "type": "command",
                                    "command": "echo stop",
                                }
                            ]
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    first = install_hook("codex", home=tmp_path)
    second = install_hook("codex", home=tmp_path)
    payload = json.loads(target.read_text())

    assert first.payload["changed"] is True
    assert second.payload["changed"] is False
    assert payload["other"] == "preserved"
    assert payload["hooks"]["Stop"][0]["hooks"][0]["id"] == "stop-hook"
    assert [entry["hooks"][0]["id"] for entry in payload["hooks"]["SessionStart"]] == [
        "user-hook",
        MANAGED_HOOK_ID,
    ]
    managed_group = payload["hooks"]["SessionStart"][1]
    assert managed_group["matcher"] == "startup|resume|clear"
    managed = managed_group["hooks"][0]
    assert managed["type"] == "command"
    assert managed["read_only"] is True
    assert managed["capture"] is False
    assert managed["timeout"] == 10
    assert managed["statusMessage"] == "Loading Engram AXI context"
    assert "engram axi" in managed["command"]
    assert managed["trace_file"] == str(tmp_path / ".engram/axi-hook-runs.jsonl")
    assert "--trace-origin session-start-hook" in managed["command"]

    status = build_hook_status_payload("codex", home=tmp_path).payload
    assert status["status"] == "installed"
    assert status["installed"] is True
    assert status["ready"] is True
    assert status["read_only"] is True
    assert status["capture"] is False
    assert status["trace_file"] == str(tmp_path / ".engram/axi-hook-runs.jsonl")
    assert status["last_run"] is None
    assert status["last_observed_run"] is None
    assert status["last_followup"] is None
    assert status["issues"] == []


def test_install_claude_code_hook_preserves_existing_settings(tmp_path) -> None:
    target = tmp_path / ".claude/settings.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(git status:*)"]},
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "",
                            "hooks": [{"type": "command", "command": "echo user"}],
                        }
                    ],
                    "UserPromptSubmit": [
                        {
                            "matcher": "",
                            "hooks": [{"type": "command", "command": "echo prompt"}],
                        }
                    ],
                },
            }
        )
        + "\n"
    )

    result = install_hook("claude-code", home=tmp_path)
    payload = json.loads(target.read_text())

    assert result.payload["changed"] is True
    assert payload["permissions"] == {"allow": ["Bash(git status:*)"]}
    assert len(payload["hooks"]["SessionStart"]) == 2
    managed = payload["hooks"]["SessionStart"][1]["hooks"][0]
    assert managed["id"] == MANAGED_HOOK_ID
    assert managed["read_only"] is True
    assert managed["capture"] is False
    assert managed["timeout"] == 10000
    assert "observe" not in managed["command"]
    assert payload["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == "echo prompt"

    status = build_hook_status_payload("claude-code", home=tmp_path).payload
    assert status["status"] == "installed"
    assert status["ready"] is True
    assert status["capture"] is False
    assert status["command"] == managed["command"]


def test_install_hook_dry_run_does_not_write(tmp_path) -> None:
    result = install_hook("codex", home=tmp_path, dry_run=True)

    assert result.payload["dry_run"] is True
    assert result.payload["changed"] is True
    assert not (tmp_path / ".codex/hooks.json").exists()


def test_install_hook_capture_is_opt_in_and_visible(tmp_path) -> None:
    result = install_hook("claude", home=tmp_path, capture=True)
    payload = json.loads((tmp_path / ".claude/settings.json").read_text())
    managed = payload["hooks"]["SessionStart"][0]["hooks"][0]

    assert result.payload["capture"] is True
    assert managed["capture"] is True
    assert "explicit-opt-in" in managed["capture_policy"]

    status = build_hook_status_payload("claude", home=tmp_path).payload
    assert status["status"] == "attention"
    assert status["ready"] is False
    assert "capture_enabled" in status["issues"]


def test_hook_status_flags_drifted_managed_hook(tmp_path) -> None:
    target = tmp_path / ".codex/hooks.json"
    target.parent.mkdir(parents=True)
    target.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "",
                            "hooks": [
                                {
                                    "id": MANAGED_HOOK_ID,
                                    "type": "command",
                                    "command": 'engram axi --project "$PWD"',
                                }
                            ],
                        }
                    ]
                }
            }
        )
        + "\n"
    )

    result = build_hook_status_payload("codex", home=tmp_path)

    assert result.payload["status"] == "attention"
    assert result.payload["ready"] is False
    assert result.payload["issues"] == [
        "not_marked_read_only",
        "missing_timeout",
        "missing_trace_file",
        "missing_trace_origin",
    ]


def test_hook_status_reports_last_metadata_trace(tmp_path) -> None:
    install_hook("codex", home=tmp_path)
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:00Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "home",
                        "status": "healthy",
                        "exitCode": 0,
                        "durationMs": 142,
                        "origin": HOOK_TRACE_ORIGIN,
                        "project": "/repo",
                        "context": "must not be copied into status",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:12Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "context",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 93,
                        "origin": "agent-followup",
                        "project": "/repo",
                        "context": "must not be copied into status",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:20Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "home",
                        "status": "healthy",
                        "exitCode": 0,
                        "durationMs": 81,
                        "origin": "manual",
                        "project": "/repo",
                    }
                ),
            ]
        )
        + "\n"
    )

    result = build_hook_status_payload("codex", home=tmp_path)

    assert result.payload["last_run"] == {
        "timestamp": "2026-05-20T20:00:00Z",
        "operation": "home",
        "status": "healthy",
        "exit_code": 0,
        "duration_ms": 142,
        "origin": HOOK_TRACE_ORIGIN,
        "project": "/repo",
    }
    assert result.payload["last_observed_run"] == {
        "timestamp": "2026-05-20T20:00:20Z",
        "operation": "home",
        "status": "healthy",
        "exit_code": 0,
        "duration_ms": 81,
        "origin": "manual",
        "project": "/repo",
    }
    assert result.payload["last_followup"] == {
        "timestamp": "2026-05-20T20:00:12Z",
        "operation": "context",
        "status": "ok",
        "exit_code": 0,
        "duration_ms": 93,
        "origin": "agent-followup",
        "project": "/repo",
    }
    assert result.payload["followup_summary"]["status"] == "measured"
    assert result.payload["followup_summary"]["trace_count"] == 1
    assert result.payload["followup_summary"]["duration_ms"]["avg"] == 93.0
    assert result.payload["followup_summary"]["recent"][0] == result.payload["last_followup"]


def test_hook_status_summarizes_recent_followup_trace_metadata(tmp_path) -> None:
    install_hook("codex", home=tmp_path)
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:00Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "other",
                        "operation": "recall",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 1,
                        "origin": "agent-followup",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:01Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "context",
                        "status": "degraded",
                        "exitCode": 0,
                        "durationMs": 500,
                        "origin": "agent-followup",
                        "project": "/repo",
                        "cacheHit": False,
                        "packetCount": 1,
                        "resultCount": 0,
                        "fallbackStatus": "context_packet_fallback",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:02Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "recall",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 8,
                        "origin": "agent-followup",
                        "project": "/repo",
                        "cacheHit": True,
                        "packetCount": 2,
                        "resultCount": 0,
                        "fallbackStatus": "cache_satisfied",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:03Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "recall",
                        "status": "timeout",
                        "exitCode": 1,
                        "durationMs": 3000,
                        "timeoutSeconds": 3,
                        "origin": "agent-followup",
                        "project": "/repo",
                        "budgetMiss": True,
                        "degraded": True,
                        "fallbackStatus": "recall_timeout",
                    }
                ),
            ]
        )
        + "\n"
    )

    summary = build_hook_status_payload("codex", home=tmp_path).payload["followup_summary"]

    assert summary["status"] == "measured"
    assert summary["trace_count"] == 3
    assert summary["operation_counts"] == {"context": 1, "recall": 2}
    assert summary["status_counts"] == {"degraded": 1, "ok": 1, "timeout": 1}
    assert summary["duration_ms"] == {
        "count": 3,
        "avg": 1169.3333,
        "p95": 3000.0,
        "max": 3000.0,
    }
    assert summary["cache_hit_count"] == 1
    assert summary["cache_hit_rate"] == 0.3333
    assert summary["packet_count"] == 3
    assert summary["result_count"] == 0
    assert summary["fallback_status_counts"] == {
        "cache_satisfied": 1,
        "context_packet_fallback": 1,
        "recall_timeout": 1,
    }
    assert summary["budget_miss_count"] == 1
    assert summary["degraded_count"] == 2
    assert summary["timeout_count"] == 1
    assert [record["status"] for record in summary["recent"]] == [
        "timeout",
        "ok",
        "degraded",
    ]
    assert summary["latest_healthy_streak"] == {
        "status": "missing",
        "trace_count": 0,
        "sample_limit": 3,
    }


def test_hook_status_reports_latest_healthy_followup_streak(tmp_path) -> None:
    install_hook("codex", home=tmp_path)
    trace_file = tmp_path / ".engram/axi-hook-runs.jsonl"
    trace_file.parent.mkdir(parents=True)
    trace_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:00Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "recall",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 500,
                        "origin": "agent-followup",
                        "fallbackStatus": "project_file_recall_fallback",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:01Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "recall",
                        "status": "timeout",
                        "exitCode": 1,
                        "durationMs": 3000,
                        "origin": "agent-followup",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:02Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "context",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 20,
                        "origin": "agent-followup",
                        "cacheHit": True,
                        "packetCount": 1,
                    }
                ),
                json.dumps(
                    {
                        "timestamp": "2026-05-20T20:00:03Z",
                        "hookId": MANAGED_HOOK_ID,
                        "client": "codex",
                        "operation": "recall",
                        "status": "ok",
                        "exitCode": 0,
                        "durationMs": 8,
                        "origin": "agent-followup",
                        "cacheHit": True,
                        "packetCount": 2,
                        "fallbackStatus": "cache_satisfied",
                    }
                ),
            ]
        )
        + "\n"
    )

    streak = build_hook_status_payload("codex", home=tmp_path).payload["followup_summary"][
        "latest_healthy_streak"
    ]

    assert streak["status"] == "measured"
    assert streak["trace_count"] == 2
    assert streak["duration_ms"] == {
        "count": 2,
        "avg": 14.0,
        "p95": 20.0,
        "max": 20.0,
    }
    assert streak["cache_hit_count"] == 2
    assert streak["cache_hit_rate"] == 1.0
    assert streak["fallback_status_counts"] == {"cache_satisfied": 1}
    assert [record["duration_ms"] for record in streak["recent"]] == [8, 20]


def test_uninstall_hook_removes_only_managed_codex_entry(tmp_path) -> None:
    install_hook("codex", home=tmp_path)
    target = tmp_path / ".codex/hooks.json"
    payload = json.loads(target.read_text())
    payload["hooks"]["SessionStart"].insert(0, {"id": "user-hook", "command": "echo user"})
    target.write_text(json.dumps(payload, indent=2) + "\n")

    result = uninstall_hook("codex", home=tmp_path)
    updated = json.loads(target.read_text())

    assert result.payload["changed"] is True
    assert updated["hooks"]["SessionStart"] == [{"id": "user-hook", "command": "echo user"}]
