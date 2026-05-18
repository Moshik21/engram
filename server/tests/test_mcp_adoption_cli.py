from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

from engram.mcp.adoption_cli import (
    build_adoption_validation_report,
    build_live_adoption_transcript_template,
    render_adoption_validation_markdown,
    run_adoption_command,
)


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
    assert "require-live-evidence" in template["instructions"][-1]


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
    assert "require-live-evidence" in markdown


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
