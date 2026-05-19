from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from engram.mcp.adoption_cli import (
    build_adoption_validation_report,
    build_live_adoption_transcript_template,
    render_adoption_validation_markdown,
    run_adoption_command,
)
from engram.setup import install_hooks


def test_adoption_validation_report_accepts_followed_jsonl_transcript(tmp_path: Path) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "calls.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps({"phase": "before_answer", "tool": "bootstrap_project"}),
                json.dumps({"phase": "before_answer", "tool": "get_context"}),
                json.dumps({"phase": "before_answer", "tool": "recall"}),
                json.dumps({"phase": "capture", "tool": "remember"}),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "passed"
    assert report["callCount"] == 4
    assert report["evidence"]["required"] is False
    assert report["validation"]["failures"] == []


def test_adoption_validation_report_flags_file_memory_bypass(tmp_path: Path) -> None:
    authority_path = tmp_path / "protocol.json"
    calls_path = tmp_path / "calls.json"
    authority_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "calls": [
                    {
                        "phase": "before_answer",
                        "tool": "read_file_memory",
                        "source": "project_local_memory",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "failed"
    assert "file_memory_used_as_substitute" in report["validation"]["failures"]
    assert report["validation"]["capture"]["missing"] is True


def test_adoption_validation_report_normalizes_real_mcp_log_shapes(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "calls.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "transcript": [
                    {
                        "stage": "before_answer",
                        "name": "mcp__engram__bootstrap_project",
                    },
                    {
                        "phase": "before_answer",
                        "tool": {"name": "mcp__engram__get_context"},
                    },
                    {
                        "phase": "before_answer",
                        "function": {"name": "mcp__engram__recall"},
                    },
                    {
                        "phase": "capture",
                        "tool_call": {"toolName": "mcp__engram__remember"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "passed"
    assert report["validation"]["required_tools_before_answer"]["observed"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert report["validation"]["capture"]["observed_tools"] == ["remember"]


def test_adoption_validation_report_accepts_claude_stream_json_tool_use(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "claude-stream.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "claude-session-123",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__claim_authority",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__bootstrap_project",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__get_context",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__recall",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__remember",
                                    "input": {},
                                },
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "{\"adoption_protocol_followed\": true}",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "passed"
    assert report["callCount"] == 5
    assert report["validation"]["required_tools_before_answer"]["observed"] == [
        "claim_authority",
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert report["validation"]["capture"]["observed_tools"] == ["remember"]


def test_adoption_validation_report_extracts_claude_stream_json_live_evidence(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "claude-stream.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "claude-session-123",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__claim_authority",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__bootstrap_project",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__get_context",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__recall",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__remember",
                                    "input": {},
                                },
                            ]
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "user",
                        "timestamp": "2026-05-18T21:58:47.456Z",
                        "message": {"content": []},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "passed"
    assert report["evidence"] == {
        "required": True,
        "client": "Claude Code",
        "captured_at": "2026-05-18T21:58:47.456Z",
        "session_id": "claude-session-123",
        "source": "claude_stream_json",
        "missing": [],
    }


def test_adoption_validation_report_classifies_blocked_claude_stream(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "claude-blocked.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "claude-session-blocked",
                        "mcp_servers": [{"name": "engram", "status": "failed"}],
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "session_id": "claude-session-blocked",
                        "error": "authentication_failed",
                        "message": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Not logged in · Please run /login",
                                }
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )
    markdown = render_adoption_validation_markdown(report)

    assert report["status"] == "failed"
    assert report["callCount"] == 0
    assert report["evidence"]["client"] == "Claude Code"
    assert report["evidence"]["session_id"] == "claude-session-blocked"
    assert report["evidence"]["blockers"] == [
        "mcp_server_failed",
        "authentication_failed",
    ]
    assert report["evidence"]["mcp_server_failures"] == ["engram"]
    assert "live_harness_authentication_failed" in report["validation"]["failures"]
    assert "live_harness_mcp_server_failed" in report["validation"]["failures"]
    assert "missing_required_before_answer_tools" in report["validation"]["failures"]
    assert "Blockers: `['mcp_server_failed', 'authentication_failed']`" in markdown
    assert "MCP server failures: `['engram']`" in markdown


def test_adoption_validation_report_accepts_plaintext_harness_notes(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-harness-notes.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                "## Before answer",
                "- mcp__engram__bootstrap_project",
                "- tool: mcp__engram__get_context",
                "| before_answer | function=mcp__engram__recall |",
                "## Capture",
                "- mcp__engram__remember",
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "passed"
    assert report["validation"]["required_tools_before_answer"]["observed"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert report["validation"]["capture"]["observed_tools"] == ["remember"]


def test_adoption_validation_accepts_rest_auto_observe_for_observe_capture(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "live-rest-hook-transcript.json"
    protocol = _protocol()
    protocol["capture"] = {"destination": "engram", "tool": "observe"}
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Claude Code",
                    "capturedAt": "2026-05-18T23:04:00Z",
                    "source": "rest_hook_trace",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {
                        "phase": "capture",
                        "method": "POST",
                        "path": "/api/knowledge/auto-observe",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "passed"
    assert report["validation"]["capture"]["expected_tool"] == "observe"
    assert report["validation"]["capture"]["observed_tools"] == ["auto_observe"]
    assert report["evidence"]["source"] == "rest_hook_trace"


def test_adoption_validation_does_not_treat_rest_auto_observe_as_remember(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "rest-hook-only-transcript.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps({"phase": "before_answer", "tool": "bootstrap_project"}),
                json.dumps({"phase": "before_answer", "tool": "get_context"}),
                json.dumps({"phase": "before_answer", "tool": "recall"}),
                json.dumps(
                    {
                        "phase": "capture",
                        "method": "POST",
                        "path": "/api/knowledge/auto-observe",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "failed"
    assert report["validation"]["capture"]["expected_tool"] == "remember"
    assert report["validation"]["capture"]["observed_tools"] == ["auto_observe"]
    assert "missing_required_capture_tool" in report["validation"]["failures"]


def test_adoption_validation_accepts_plaintext_rest_auto_observe_capture(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-rest-hook-notes.md"
    protocol = _protocol()
    protocol["capture"] = {"destination": "engram", "tool": "observe"}
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                "Client: Claude Code",
                "Captured at: 2026-05-18T23:06:00Z",
                "## Before answer",
                "- mcp__engram__bootstrap_project",
                "- mcp__engram__get_context",
                "- mcp__engram__recall",
                "## Capture",
                "- POST /api/knowledge/auto-observe",
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "passed"
    assert report["validation"]["capture"]["observed_tools"] == ["auto_observe"]


def test_adoption_command_merges_stream_and_hook_trace_files(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    stream_path = tmp_path / "claude-stream.jsonl"
    trace_path = tmp_path / "adoption-trace.jsonl"
    protocol = _protocol()
    protocol["capture"] = {"destination": "engram", "tool": "observe"}
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")
    stream_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "claude-session-merge",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__claim_authority",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__bootstrap_project",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__get_context",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__recall",
                                    "input": {},
                                },
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(
            {
                "phase": "capture",
                "tool": "auto_observe",
                "client": "Claude Code",
                "capturedAt": "2026-05-18T23:18:00Z",
                "session_id": "claude-session-merge",
                "source": "rest_hook_prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=[stream_path, trace_path],
            format="json",
            require_live_evidence=True,
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["callsPath"] == [str(stream_path), str(trace_path)]
    assert payload["callCount"] == 5
    assert payload["evidence"]["client"] == "Claude Code"
    assert payload["evidence"]["captured_at"] == "2026-05-18T23:18:00Z"
    assert payload["evidence"]["session_id"] == "claude-session-merge"
    assert payload["validation"]["capture"]["observed_tools"] == ["auto_observe"]


def test_adoption_command_rejects_mismatched_live_session_evidence(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    stream_path = tmp_path / "claude-stream.jsonl"
    trace_path = tmp_path / "stale-adoption-trace.jsonl"
    protocol = _protocol()
    protocol["capture"] = {"destination": "engram", "tool": "observe"}
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")
    stream_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "current-session",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__bootstrap_project",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__get_context",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__recall",
                                    "input": {},
                                },
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        json.dumps(
            {
                "phase": "capture",
                "tool": "auto_observe",
                "client": "Claude Code",
                "capturedAt": "2026-05-18T23:24:00Z",
                "session_id": "stale-session",
                "source": "rest_hook_prompt",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=[stream_path, trace_path],
            format="json",
            require_live_evidence=True,
        )
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["evidence"]["session_mismatch"] is True
    assert payload["evidence"]["session_ids"] == ["current-session", "stale-session"]
    assert "inconsistent_live_harness_session" in payload["validation"]["failures"]


def test_adoption_command_filters_cumulative_trace_by_session(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    stream_path = tmp_path / "claude-stream.jsonl"
    trace_path = tmp_path / "cumulative-adoption-trace.jsonl"
    protocol = _protocol()
    protocol["capture"] = {"destination": "engram", "tool": "observe"}
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")
    stream_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "type": "system",
                        "subtype": "init",
                        "session_id": "current-session",
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__bootstrap_project",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__get_context",
                                    "input": {},
                                },
                                {
                                    "type": "tool_use",
                                    "name": "mcp__engram__recall",
                                    "input": {},
                                },
                            ]
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phase": "capture",
                        "tool": "auto_observe",
                        "client": "Claude Code",
                        "capturedAt": "2026-05-18T23:20:00Z",
                        "session_id": "stale-session",
                        "source": "rest_hook_prompt",
                    }
                ),
                json.dumps(
                    {
                        "phase": "capture",
                        "tool": "auto_observe",
                        "client": "Claude Code",
                        "capturedAt": "2026-05-18T23:30:00Z",
                        "session_id": "current-session",
                        "source": "rest_hook_response",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=[stream_path, trace_path],
            format="json",
            require_live_evidence=True,
            session_id="current-session",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["callCount"] == 4
    assert payload["evidence"]["session_filter"] == "current-session"
    assert payload["evidence"]["session_id"] == "current-session"
    assert payload["validation"]["capture"]["observed_tools"] == ["auto_observe"]


def test_generated_prompt_hook_trace_validates_with_session_filtered_stream(
    tmp_path: Path,
) -> None:
    """Generated hooks should emit trace JSONL that adoption validation consumes."""

    posted_payloads: list[dict[str, object]] = []

    class AutoObserveHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            posted_payloads.append(
                {
                    "path": self.path,
                    "body": json.loads(raw_body.decode("utf-8")),
                }
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')

        def log_message(self, _format: str, *args: object) -> None:
            return

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), AutoObserveHandler)
    except OSError as exc:
        pytest.skip(f"local auto-observe test server unavailable: {exc}")

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        hooks_dir = tmp_path / "hooks"
        settings_path = tmp_path / "settings.json"
        trace_path = tmp_path / "adoption-trace.jsonl"
        authority_path = tmp_path / "claim-authority.json"
        stream_path = tmp_path / "claude-stream.jsonl"

        install_hooks(hooks_dir=hooks_dir, settings_path=settings_path)
        prompt_script = hooks_dir / "capture-prompt.sh"
        protocol = _protocol()
        protocol["capture"] = {"destination": "engram", "tool": "observe"}
        authority_path.write_text(
            json.dumps({"agent_protocol": protocol}),
            encoding="utf-8",
        )
        stream_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "system",
                            "subtype": "init",
                            "session_id": "live-session-1",
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "name": "mcp__engram__bootstrap_project",
                                        "input": {},
                                    },
                                    {
                                        "type": "tool_use",
                                        "name": "mcp__engram__get_context",
                                        "input": {},
                                    },
                                    {
                                        "type": "tool_use",
                                        "name": "mcp__engram__recall",
                                        "input": {},
                                    },
                                ]
                            },
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        hook_input = {
            "prompt": "Please remember that Engram should be cross-context memory.",
            "cwd": "/Users/konnermoshier/Engram",
            "session_id": "live-session-1",
        }
        result = subprocess.run(
            [str(prompt_script)],
            input=json.dumps(hook_input),
            text=True,
            capture_output=True,
            check=False,
            env={
                "ENGRAM_URL": f"http://127.0.0.1:{server.server_address[1]}",
                "ENGRAM_ADOPTION_TRACE_FILE": str(trace_path),
                "HOME": str(tmp_path),
                "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            },
            timeout=10,
        )

        assert result.returncode == 0, result.stderr
        assert posted_payloads == [
            {
                "path": "/api/knowledge/auto-observe",
                "body": {
                    "content": (
                        "[user|Engram] Please remember that Engram should be "
                        "cross-context memory."
                    ),
                    "source": "auto:prompt",
                    "project": "Engram",
                    "role": "user",
                    "session_id": "live-session-1",
                },
            }
        ]

        trace_record = json.loads(trace_path.read_text(encoding="utf-8"))
        assert trace_record["phase"] == "capture"
        assert trace_record["tool"] == "auto_observe"
        assert trace_record["source"] == "rest_hook_prompt"
        assert trace_record["client"] == "Claude Code"
        assert trace_record["session_id"] == "live-session-1"
        assert trace_record["capturedAt"].endswith("Z")

        report = build_adoption_validation_report(
            authority_path=authority_path,
            calls_path=[stream_path, trace_path],
            require_live_evidence=True,
            session_id_filter="live-session-1",
        )

        assert report["status"] == "passed"
        assert report["callCount"] == 4
        assert report["evidence"]["session_filter"] == "live-session-1"
        assert report["evidence"]["session_id"] == "live-session-1"
        assert report["validation"]["capture"]["observed_tools"] == ["auto_observe"]
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def test_adoption_validation_report_accepts_live_evidence_metadata(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "live-client-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Claude Code",
                    "capturedAt": "2026-05-18T21:42:00Z",
                    "sessionId": "claude-session-123",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "passed"
    assert report["evidence"] == {
        "required": True,
        "client": "Claude Code",
        "captured_at": "2026-05-18T21:42:00Z",
        "session_id": "claude-session-123",
        "source": "copied_mcp_log",
        "missing": [],
    }
    assert report["validation"]["failures"] == []
    assert report["release_evidence"]["status"] == "ready_for_human_labels"
    assert report["release_evidence"]["human_label_metadata"] == {
        "client": "Claude Code",
        "capturedAt": "2026-05-18T21:42:00Z",
        "sessionId": "claude-session-123",
    }
    assert report["release_evidence"]["commands"]["human_label_template"] == (
        "engram evaluate --human-label-template "
        "--adoption-report adoption-report.json "
        "--human-label-template-out human-label-template.json --format json"
    )


def test_adoption_release_handoff_requires_live_evidence_gate(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "live-client-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Claude Code",
                    "capturedAt": "2026-05-18T21:42:00Z",
                    "sessionId": "claude-session-123",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=False,
    )
    markdown = render_adoption_validation_markdown(report)

    assert report["status"] == "passed"
    assert report["evidence"]["required"] is False
    assert report["release_evidence"]["status"] == "blocked"
    assert (
        "Re-run adoption validation with --require-live-evidence before using this "
        "report as release evidence."
    ) in report["release_evidence"]["notes"]
    assert "Status: `blocked`" in markdown
    assert "--require-live-evidence" in markdown


def test_adoption_command_writes_release_report_artifact(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "live-client-transcript.json"
    report_path = tmp_path / "release" / "adoption-report.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Cursor",
                    "capturedAt": "2026-05-18T22:31:00Z",
                    "sessionId": "cursor-thread-4",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=[calls_path],
            template=False,
            client=None,
            captured_at=None,
            session_id=None,
            source=None,
            report_out=report_path,
            require_live_evidence=True,
            require_client="Cursor",
            format="markdown",
        )
    )

    markdown = capsys.readouterr().out
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert code == 0
    assert saved_report["status"] == "passed"
    assert saved_report["release_evidence"]["adoption_report_path"] == str(report_path)
    assert (
        f"--adoption-report {report_path}"
        in saved_report["release_evidence"]["commands"]["human_label_template"]
    )
    assert (
        "--require-adoption-client Cursor"
        in saved_report["release_evidence"]["commands"]["human_label_template"]
    )
    assert (
        "--require-adoption-client Cursor"
        in saved_report["release_evidence"]["commands"]["release_gate"]
    )
    assert (
        "--human-label-template-out human-label-template.json"
        in saved_report["release_evidence"]["commands"]["human_label_template"]
    )
    assert f"Adoption report path: `{report_path}`" in markdown


def test_adoption_validation_report_accepts_required_client_case_insensitive(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "cursor-live-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Cursor",
                    "capturedAt": "2026-05-18T21:42:00Z",
                    "sessionId": "cursor-thread-1",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
        required_client="cursor",
    )
    markdown = render_adoption_validation_markdown(report)

    assert report["status"] == "passed"
    assert report["evidence"]["required_client"] == "cursor"
    assert report["evidence"]["client_mismatch"] is False
    assert report["validation"]["failures"] == []
    assert report["release_evidence"]["human_label_metadata"]["requiredClient"] == "cursor"
    assert (
        "--require-adoption-client cursor"
        in report["release_evidence"]["commands"]["release_gate"]
    )
    assert "Required adoption client: `cursor`" in markdown


def test_adoption_validation_report_rejects_wrong_live_client(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "claude-live-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Claude Code",
                    "capturedAt": "2026-05-18T21:42:00Z",
                    "sessionId": "claude-session-123",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
        required_client="Cursor",
    )
    markdown = render_adoption_validation_markdown(report)

    assert report["status"] == "failed"
    assert report["evidence"]["required_client"] == "Cursor"
    assert report["evidence"]["client_mismatch"] is True
    assert "live_harness_client_mismatch" in report["validation"]["failures"]
    assert "Required client: `Cursor`" in markdown
    assert "Client mismatch: `True`" in markdown
    assert "Release Evidence Next Steps" in markdown
    assert "Status: `blocked`" in markdown


def test_adoption_validation_report_requires_session_when_session_filter_is_live_gate(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "sessionless-live-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "Cursor",
                    "capturedAt": "2026-05-18T21:42:00Z",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
        session_id_filter="cursor-thread-1",
    )
    markdown = render_adoption_validation_markdown(report)

    assert report["status"] == "failed"
    assert report["evidence"]["session_filter"] == "cursor-thread-1"
    assert report["evidence"]["session_id"] is None
    assert "missing_live_harness_session" in report["validation"]["failures"]
    assert "Session filter: `cursor-thread-1`" in markdown


def test_adoption_validation_report_rejects_live_evidence_placeholders(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "live-client-transcript.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "client": "<Claude Code | Cursor | Windsurf | other MCP client>",
                    "capturedAt": "<ISO-8601 timestamp from live run>",
                    "source": "copied_mcp_log",
                },
                "calls": [
                    {"phase": "before_answer", "tool": "mcp__engram__bootstrap_project"},
                    {"phase": "before_answer", "tool": "mcp__engram__get_context"},
                    {"phase": "before_answer", "tool": "mcp__engram__recall"},
                    {"phase": "capture", "tool": "mcp__engram__remember"},
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "failed"
    assert report["evidence"]["missing"] == ["client", "captured_at"]
    assert "missing_live_harness_evidence" in report["validation"]["failures"]


def test_adoption_validation_report_requires_live_evidence_metadata(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "calls.jsonl"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                json.dumps({"phase": "before_answer", "tool": "bootstrap_project"}),
                json.dumps({"phase": "before_answer", "tool": "get_context"}),
                json.dumps({"phase": "before_answer", "tool": "recall"}),
                json.dumps({"phase": "capture", "tool": "remember"}),
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "failed"
    assert report["evidence"]["required"] is True
    assert report["evidence"]["missing"] == ["client", "captured_at"]
    assert "missing_live_harness_evidence" in report["validation"]["failures"]


def test_adoption_validation_report_extracts_plaintext_live_evidence(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-harness-notes.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                "Client: Cursor",
                "Captured at: 2026-05-18T22:03:00Z",
                "Session ID: cursor-thread-9",
                "## Before answer",
                "- mcp__engram__bootstrap_project",
                "- mcp__engram__get_context",
                "- mcp__engram__recall",
                "## Capture",
                "- mcp__engram__remember",
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
        require_live_evidence=True,
    )

    assert report["status"] == "passed"
    assert report["evidence"]["client"] == "Cursor"
    assert report["evidence"]["captured_at"] == "2026-05-18T22:03:00Z"
    assert report["evidence"]["session_id"] == "cursor-thread-9"


def test_live_adoption_template_uses_authority_protocol_example(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    protocol = _protocol()
    protocol["verification"] = {
        "transcript_schema": {
            "example": [
                {"phase": "before_answer", "tool": "bootstrap_project"},
                {"phase": "before_answer", "tool": "get_context"},
                {"phase": "capture", "tool": "remember"},
            ]
        }
    }
    authority_path.write_text(json.dumps({"agent_protocol": protocol}), encoding="utf-8")

    template = build_live_adoption_transcript_template(
        authority_path=authority_path,
        client="Claude Code",
        captured_at="2026-05-18T22:30:00Z",
        session_id="claude-thread-1",
    )

    assert template["metadata"] == {
        "client": "Claude Code",
        "capturedAt": "2026-05-18T22:30:00Z",
        "sessionId": "claude-thread-1",
        "source": "copied_mcp_log",
    }
    assert template["calls"] == [
        {"phase": "before_answer", "tool": "bootstrap_project"},
        {"phase": "before_answer", "tool": "get_context"},
        {"phase": "capture", "tool": "remember"},
    ]
    assert template["capture_commands"] == [
        {
            "label": "claude_code_stream_json",
            "command": (
                "claude -p --verbose --output-format stream-json "
                "--include-hook-events --mcp-config .mcp.json --strict-mcp-config "
                "--allowedTools "
                "mcp__engram__claim_authority,mcp__engram__bootstrap_project,"
                "mcp__engram__get_context,mcp__engram__recall,"
                "mcp__engram__remember "
                "'<prompt instructing Claude to call claim_authority and follow "
                "agent_protocol>' > claude-stream.jsonl"
            ),
        }
    ]
    assert template["validation_commands"] == [
        {
            "label": "single_transcript",
            "command": (
                "engram adoption --authority claim-authority.json "
                "--calls live-harness-transcript.json "
                "--require-client 'Claude Code' --require-live-evidence "
                "--report-out adoption-report.json"
            ),
        },
        {
            "label": "claude_stream_with_autocapture_trace",
            "command": (
                "engram adoption --authority claim-authority.json "
                "--calls claude-stream.jsonl ~/.engram/adoption-trace.jsonl "
                "--session-id claude-thread-1 "
                "--require-client 'Claude Code' --require-live-evidence "
                "--report-out adoption-report.json"
            ),
        },
    ]
    assert template["manual_transcript_markdown"] == "\n".join(
        [
            "Client: Claude Code",
            "Captured at: 2026-05-18T22:30:00Z",
            "Session ID: claude-thread-1",
            "Source: copied_mcp_log",
            "",
            "## Before answer",
            "- bootstrap_project",
            "- get_context",
            "",
            "## Capture",
            "- remember",
        ]
    )
    assert "manual_transcript_markdown" in template["instructions"][2]
    assert "AutoCapture hooks" in template["instructions"][-1]


def test_adoption_template_command_outputs_markdown(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=None,
            template=True,
            client="Cursor",
            captured_at="2026-05-18T22:31:00Z",
            session_id="cursor-thread-4",
            source=None,
            require_live_evidence=False,
            format="markdown",
        )
    )

    assert code == 0
    markdown = capsys.readouterr().out
    assert "# Engram Live Adoption Transcript Template" in markdown
    assert "Client: `Cursor`" in markdown
    assert "`before_answer`: `bootstrap_project`" in markdown
    assert "## Capture" in markdown
    assert "claude -p --verbose --output-format stream-json" in markdown
    assert "mcp__engram__claim_authority,mcp__engram__bootstrap_project" in markdown
    assert "## Manual Transcript" in markdown
    assert "Use this for Cursor, Windsurf, or copied MCP UI logs" in markdown
    assert "Client: Cursor" in markdown
    assert "Session ID: cursor-thread-4" in markdown
    assert "- bootstrap_project" in markdown
    assert "- remember" in markdown
    assert (
        "--calls live-harness-transcript.json "
        "--require-client Cursor --require-live-evidence "
        "--report-out adoption-report.json"
    ) in markdown
    assert "--calls claude-stream.jsonl ~/.engram/adoption-trace.jsonl" in markdown
    assert "--session-id cursor-thread-4 --require-client Cursor" in markdown
    assert "--report-out adoption-report.json" in markdown


def test_adoption_command_returns_nonzero_for_failed_validation(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "protocol.json"
    calls_path = tmp_path / "calls.json"
    authority_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    calls_path.write_text(
        json.dumps([{"phase": "before_answer", "tool": "read_file_memory"}]),
        encoding="utf-8",
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=calls_path,
            format="json",
        )
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"


def test_adoption_command_reports_malformed_plaintext_transcript(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-notes.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text("- mcp__engram__recall\n", encoding="utf-8")

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=calls_path,
            format="json",
        )
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["validation"]["failures"] == ["invalid_calls_transcript"]
    assert "before_answer/capture phase" in payload["validation"]["error"]


def test_adoption_validation_report_classifies_self_reported_file_memory_bypass(
    tmp_path: Path,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-agent-chat.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(
        "\n".join(
            [
                "Honest answer: in this conversation I haven't touched Engram at all",
                "until you asked me to ping it.",
                "You have two memory systems in play:",
                "1. File-based auto-memory at MEMORY.md. It's already my primary",
                "knowledge base for this project.",
                "I'm essentially ignoring that protocol because the file-based system",
                "is doing the same job and is transparent.",
                "Otherwise I'll keep file-based as primary.",
            ]
        ),
        encoding="utf-8",
    )

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    assert report["status"] == "failed"
    assert report["callCount"] == 1
    assert report["validation"]["required_tools_before_answer"]["observed"] == [
        "read_file_memory"
    ]
    assert report["validation"]["file_memory"]["observed_tools"] == [
        "read_file_memory"
    ]
    assert "missing_required_before_answer_tools" in report["validation"]["failures"]
    assert "file_memory_used_as_substitute" in report["validation"]["failures"]
    assert "missing_required_capture_tool" in report["validation"]["failures"]


def test_adoption_command_classifies_copied_claude_file_memory_bypass_transcript(
    tmp_path: Path,
    capsys,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-claude-chat.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text(_claude_file_memory_bypass_transcript(), encoding="utf-8")

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=calls_path,
            format="json",
        )
    )

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["callCount"] == 1
    assert payload["validation"]["required_tools_before_answer"]["observed"] == [
        "read_file_memory"
    ]
    assert payload["validation"]["file_memory"]["observed_tools"] == [
        "read_file_memory"
    ]
    assert "missing_required_before_answer_tools" in payload["validation"]["failures"]
    assert "file_memory_used_as_substitute" in payload["validation"]["failures"]
    assert "missing_required_capture_tool" in payload["validation"]["failures"]


def test_adoption_command_accepts_calls_from_stdin(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    authority_path = tmp_path / "claim-authority.json"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "\n".join(
                [
                    "## Before answer",
                    "- mcp__engram__bootstrap_project",
                    "- mcp__engram__get_context",
                    "- mcp__engram__recall",
                    "## Capture",
                    "- mcp__engram__remember",
                ]
            )
        ),
    )

    code = run_adoption_command(
        argparse.Namespace(
            authority=authority_path,
            calls=Path("-"),
            format="json",
        )
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "passed"
    assert payload["callsPath"] == "-"
    assert payload["callCount"] == 4


def test_adoption_markdown_report_summarizes_failures(tmp_path: Path) -> None:
    authority_path = tmp_path / "protocol.json"
    calls_path = tmp_path / "calls.json"
    authority_path.write_text(json.dumps(_protocol()), encoding="utf-8")
    calls_path.write_text(
        json.dumps([{"phase": "before_answer", "tool": "read_file_memory"}]),
        encoding="utf-8",
    )
    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )

    markdown = render_adoption_validation_markdown(report)

    assert "# Engram MCP Adoption Validation: failed" in markdown
    assert "Live Harness Evidence" in markdown
    assert "missing_required_before_answer_tools" in markdown
    assert "missing_required_capture_tool" in markdown


def test_adoption_markdown_report_includes_parse_error(tmp_path: Path) -> None:
    authority_path = tmp_path / "claim-authority.json"
    calls_path = tmp_path / "copied-notes.md"
    authority_path.write_text(json.dumps({"agent_protocol": _protocol()}), encoding="utf-8")
    calls_path.write_text("- mcp__engram__recall\n", encoding="utf-8")

    report = build_adoption_validation_report(
        authority_path=authority_path,
        calls_path=calls_path,
    )
    markdown = render_adoption_validation_markdown(report)

    assert "invalid_calls_transcript" in markdown
    assert "## Error" in markdown
    assert "before_answer/capture phase" in markdown


def _protocol() -> dict:
    return {
        "file_memory_present": True,
        "file_memory_is_substitute": False,
        "required_tools_before_answer": [
            "bootstrap_project",
            "get_context",
            "recall",
        ],
        "capture": {
            "destination": "engram",
            "tool": "remember",
        },
    }


def _claude_file_memory_bypass_transcript() -> str:
    return "\n".join(
        [
            "User: Is Engram up and running? I ran mcp and saw it connected.",
            "",
            "Assistant: Let me poke it with a low-impact call to confirm it's actually responding.",
            "",
            "Called engram (ctrl+o to expand)",
            "",
            (
                "Assistant: Yes. Engram responded cleanly. Connected, runtime mode helix, "
                "all features enabled."
            ),
            (
                "One observable: artifactCount: 0, lastObservedAt: null, and all "
                "recall/epistemic stats at zero."
            ),
            (
                "So the server is up, but it hasn't yet ingested any artifacts or recorded "
                "any recalls for this project."
            ),
            "",
            "User: Interesting. What is your path for communicating, recording, etc with Engram?",
            "",
            (
                "Assistant: Honest answer: in this conversation I haven't touched Engram "
                "at all until you asked me to ping it."
            ),
            "What I actually do right now",
            "You have two memory systems in play:",
            (
                "1. File-based auto-memory at MEMORY.md. It's already my primary knowledge "
                "base for this project."
            ),
            (
                "2. Engram - connected but artifactCount: 0. I'd have to call get_context, "
                "observe, remember, recall deliberately."
            ),
            (
                "I'm essentially ignoring that protocol because the file-based system is "
                "doing the same job and is transparent."
            ),
            "Otherwise I'll keep file-based as primary.",
        ]
    )
