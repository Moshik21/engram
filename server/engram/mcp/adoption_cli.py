"""CLI helpers for validating MCP client adoption transcripts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from engram.retrieval.memory_authority import (
    ENGRAM_CAPTURE_TOOLS,
    validate_agent_protocol_calls,
)

_PHASE_VALUES = frozenset({"before_answer", "capture"})
_PHASE_ALIASES = {
    "before answer": "before_answer",
    "before-answer": "before_answer",
    "before_answer": "before_answer",
    "pre answer": "before_answer",
    "pre-answer": "before_answer",
    "pre_answer": "before_answer",
    "capture": "capture",
}
_PHASE_FIELD_RE = re.compile(
    r"\b(?:phase|stage)\s*[:=]\s*`?([A-Za-z_-]+(?:\s+[A-Za-z_-]+)?)`?",
    re.IGNORECASE,
)
_PHASE_PREFIX_RE = re.compile(
    r"^[\s>#*\-|\[]*([A-Za-z_-]+(?:\s+[A-Za-z_-]+)?)\]?\s*:?\b",
    re.IGNORECASE,
)
_TOOL_FIELD_RE = re.compile(
    r"\b(?:tool|name|tool_name|toolName|function)\s*[:=]\s*`?([A-Za-z0-9_.\-/]+)`?",
)
_ENGRAM_TOOL_RE = re.compile(
    r"\b(?:mcp__engram__|engram[./])([A-Za-z][A-Za-z0-9_]*)\b",
)
_ENGRAM_API_CAPTURE_RE = re.compile(r"/api/knowledge/auto-observe\b")
_SELF_REPORTED_ENGRAM_IGNORED_RE = re.compile(
    r"\b(?:have not|haven't|had not|hadn't|did not|didn't)\s+"
    r"(?:touch|touched|use|used|call|called)\s+engram\b"
    r"|ignoring\s+(?:that\s+)?protocol"
    r"|rout(?:e|ed|ing)\s+around\s+(?:it|engram)"
    r"|skip(?:ped|ping)?\s+engram",
    re.IGNORECASE,
)
_SELF_REPORTED_FILE_MEMORY_RE = re.compile(
    r"\bfile[- ](?:based|local)\b"
    r"|project[- ]local"
    r"|\bMEMORY\.md\b"
    r"|auto[- ]injected\s+memory",
    re.IGNORECASE,
)
_SELF_REPORTED_FILE_MEMORY_PRIMARY_RE = re.compile(
    r"\bprimary\b"
    r"|\bsubstitute\b"
    r"|\bsame\s+job\b"
    r"|\boverlap"
    r"|keep\s+file[- ]based"
    r"|file[- ]based\s+as\s+primary",
    re.IGNORECASE,
)
_LIVE_EVIDENCE_FIELD_RE = re.compile(
    r"^\s*(?:[-*]\s*)?"
    r"(client|harness|captured at|captured_at|capturedAt|timestamp|recorded at|"
    r"recorded_at|session|session id|session_id|thread|thread id|thread_id|source)"
    r"\s*[:=]\s*(.+?)\s*$",
    re.IGNORECASE,
)
_LIVE_EVIDENCE_REQUIRED_FIELDS = ("client", "captured_at")


def configure_adoption_parser(parser: argparse.ArgumentParser) -> None:
    """Configure `engram adoption` arguments."""
    parser.add_argument(
        "--authority",
        type=Path,
        required=True,
        help="Path to a claim_authority JSON payload or agent_protocol JSON object.",
    )
    parser.add_argument(
        "--calls",
        type=Path,
        required=False,
        help=(
            "Path to tool-call transcript JSON/JSONL, or plaintext/Markdown "
            "with explicit before_answer/capture phase and Engram tool lines; "
            "use '-' to read the transcript from stdin. Not required with --template."
        ),
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help=(
            "Print a live-harness transcript template from the authority payload "
            "instead of validating calls."
        ),
    )
    parser.add_argument(
        "--client",
        default=None,
        help="Client label to place in --template metadata, for example 'Claude Code'.",
    )
    parser.add_argument(
        "--captured-at",
        default=None,
        help="Live capture timestamp to place in --template metadata.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional live client session/thread id to place in --template metadata.",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Optional transcript source label to place in --template metadata.",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json).",
    )
    parser.add_argument(
        "--require-live-evidence",
        action="store_true",
        help=(
            "Fail unless the transcript includes live harness evidence metadata "
            "(client/harness and captured_at/timestamp)."
        ),
    )


def run_adoption_command(args: argparse.Namespace) -> int:
    """Validate a recorded client transcript against a claim_authority protocol."""
    output_format = getattr(args, "format", "json")
    if getattr(args, "template", False):
        try:
            template = build_live_adoption_transcript_template(
                authority_path=args.authority,
                client=getattr(args, "client", None),
                captured_at=getattr(args, "captured_at", None),
                session_id=getattr(args, "session_id", None),
                source=getattr(args, "source", None),
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            report = _build_adoption_error_report(
                authority_path=args.authority,
                calls_path=Path("<template>"),
                failure="invalid_authority_payload",
                error=str(exc),
            )
            if output_format == "markdown":
                print(render_adoption_validation_markdown(report))
            else:
                print(json.dumps(report, indent=2, sort_keys=True))
            return 1
        if output_format == "markdown":
            print(render_live_adoption_template_markdown(template))
        else:
            print(json.dumps(template, indent=2, sort_keys=True))
        return 0

    calls_path = getattr(args, "calls", None)
    if calls_path is None:
        print(
            "engram adoption requires --calls unless --template is used",
            file=sys.stderr,
        )
        return 2

    calls_text = sys.stdin.read() if calls_path == Path("-") else None
    report = build_adoption_validation_report(
        authority_path=args.authority,
        calls_path=calls_path,
        calls_text=calls_text,
        require_live_evidence=getattr(args, "require_live_evidence", False),
    )
    if output_format == "markdown":
        print(render_adoption_validation_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def build_live_adoption_transcript_template(
    *,
    authority_path: Path,
    client: str | None = None,
    captured_at: str | None = None,
    session_id: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """Build a live-harness transcript template from a claim_authority payload."""
    authority_payload = _load_json(authority_path)
    protocol = _extract_protocol(authority_payload)
    return {
        "metadata": {
            "client": client or "<Claude Code | Cursor | Windsurf | other MCP client>",
            "capturedAt": captured_at or "<ISO-8601 timestamp from live run>",
            "sessionId": session_id or "<client session/thread id if available>",
            "source": source or "copied_mcp_log",
        },
        "calls": _protocol_example_calls(protocol),
        "instructions": [
            "Run the real MCP client after saving the matching claim_authority response.",
            "Replace metadata placeholders with observed live client metadata.",
            "Replace or confirm calls with the actual Engram tool calls from the client log.",
            (
                "Validate with engram adoption --authority claim-authority.json "
                "--calls live-harness-transcript.json --require-live-evidence."
            ),
        ],
    }


def build_adoption_validation_report(
    *,
    authority_path: Path,
    calls_path: Path,
    calls_text: str | None = None,
    require_live_evidence: bool = False,
) -> dict[str, Any]:
    """Build a validation report for a saved claim_authority payload and calls."""
    try:
        authority_payload = _load_json(authority_path)
        protocol = _extract_protocol(authority_payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _build_adoption_error_report(
            authority_path=authority_path,
            calls_path=calls_path,
            failure="invalid_authority_payload",
            error=str(exc),
        )

    try:
        calls, evidence = (
            _load_calls_bundle_text(calls_text)
            if calls_text is not None
            else _load_calls_bundle(calls_path)
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _build_adoption_error_report(
            authority_path=authority_path,
            calls_path=calls_path,
            failure="invalid_calls_transcript",
            error=str(exc),
        )

    validation = validate_agent_protocol_calls(protocol, calls)
    evidence_report = _build_live_evidence_report(
        evidence,
        require_live_evidence=require_live_evidence,
    )
    if evidence_report["required"] and evidence_report["missing"]:
        validation = dict(validation)
        failures = list(validation.get("failures") or [])
        failures.append("missing_live_harness_evidence")
        validation["failures"] = failures
        validation["status"] = "failed"
    return {
        "status": validation["status"],
        "authorityPath": str(authority_path),
        "callsPath": str(calls_path),
        "callCount": len(calls),
        "evidence": evidence_report,
        "validation": validation,
    }


def _build_adoption_error_report(
    *,
    authority_path: Path,
    calls_path: Path,
    failure: str,
    error: str,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "authorityPath": str(authority_path),
        "callsPath": str(calls_path),
        "callCount": 0,
        "evidence": {
            "required": False,
            "client": None,
            "captured_at": None,
            "session_id": None,
            "source": None,
            "missing": [],
        },
        "validation": {
            "required_tools_before_answer": {
                "expected": [],
                "observed": [],
                "missing": [],
                "in_order": False,
            },
            "capture": {
                "destination": None,
                "expected_tool": None,
                "observed_tools": [],
                "missing": False,
                "unexpected_engram_capture_tools": [],
            },
            "file_memory": {
                "present": None,
                "observed_tools": [],
                "substituted_for_engram": False,
            },
            "failures": [failure],
            "error": error,
        },
    }


def render_adoption_validation_markdown(report: dict[str, Any]) -> str:
    """Render an adoption validation report for humans."""
    validation = report["validation"]
    lines = [
        f"# Engram MCP Adoption Validation: {report['status']}",
        "",
        f"- Authority payload: `{report['authorityPath']}`",
        f"- Calls transcript: `{report['callsPath']}`",
        f"- Calls observed: `{report['callCount']}`",
        "",
        "## Live Harness Evidence",
        f"- Required: `{report.get('evidence', {}).get('required', False)}`",
        f"- Client: `{report.get('evidence', {}).get('client')}`",
        f"- Captured at: `{report.get('evidence', {}).get('captured_at')}`",
        f"- Session ID: `{report.get('evidence', {}).get('session_id')}`",
        f"- Source: `{report.get('evidence', {}).get('source')}`",
        f"- Missing: `{report.get('evidence', {}).get('missing', [])}`",
        "",
        "## Required Before-Answer Tools",
        f"- Expected: `{validation['required_tools_before_answer']['expected']}`",
        f"- Observed: `{validation['required_tools_before_answer']['observed']}`",
        f"- Missing: `{validation['required_tools_before_answer']['missing']}`",
        f"- In order: `{validation['required_tools_before_answer']['in_order']}`",
        "",
        "## Capture",
        f"- Destination: `{validation['capture']['destination']}`",
        f"- Expected tool: `{validation['capture']['expected_tool']}`",
        f"- Observed tools: `{validation['capture']['observed_tools']}`",
        f"- Missing: `{validation['capture']['missing']}`",
        "",
        "## File Memory",
        f"- Present: `{validation['file_memory']['present']}`",
        f"- Observed tools: `{validation['file_memory']['observed_tools']}`",
        f"- Substituted for Engram: `{validation['file_memory']['substituted_for_engram']}`",
    ]
    if validation["failures"]:
        lines.extend(["", "## Failures"])
        lines.extend(f"- `{failure}`" for failure in validation["failures"])
    if validation.get("error"):
        lines.extend(["", "## Error", validation["error"]])
    return "\n".join(lines)


def render_live_adoption_template_markdown(template: dict[str, Any]) -> str:
    """Render a live-harness transcript template for operators."""
    metadata = template.get("metadata") or {}
    calls = template.get("calls") or []
    lines = [
        "# Engram Live Adoption Transcript Template",
        "",
        "## Metadata",
        f"- Client: `{metadata.get('client')}`",
        f"- Captured at: `{metadata.get('capturedAt')}`",
        f"- Session ID: `{metadata.get('sessionId')}`",
        f"- Source: `{metadata.get('source')}`",
        "",
        "## Calls",
    ]
    for call in calls:
        if not isinstance(call, dict):
            continue
        lines.append(f"- `{call.get('phase')}`: `{call.get('tool')}`")
    lines.extend(
        [
            "",
            "## Validate",
            (
                "`engram adoption --authority claim-authority.json --calls "
                "live-harness-transcript.json --require-live-evidence`"
            ),
            "",
            "Replace placeholders with observed live client metadata and actual tool calls.",
        ]
    )
    return "\n".join(lines)


def _extract_protocol(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("authority payload must be a JSON object")
    protocol = payload.get("agent_protocol", payload)
    if not isinstance(protocol, dict):
        raise ValueError("agent_protocol must be a JSON object")
    if "required_tools_before_answer" not in protocol or "capture" not in protocol:
        raise ValueError(
            "authority payload must contain an agent_protocol with "
            "required_tools_before_answer and capture"
        )
    return protocol


def _protocol_example_calls(protocol: dict[str, Any]) -> list[dict[str, str]]:
    verification = protocol.get("verification")
    if isinstance(verification, dict):
        schema = verification.get("transcript_schema")
        if isinstance(schema, dict) and isinstance(schema.get("example"), list):
            return [_normalize_template_call(call) for call in schema["example"]]

    calls = [
        {"phase": "before_answer", "tool": str(tool)}
        for tool in protocol.get("required_tools_before_answer") or []
        if tool
    ]
    capture = protocol.get("capture")
    if isinstance(capture, dict):
        expected_tool = capture.get("tool")
        if capture.get("destination") == "engram" and expected_tool:
            calls.append({"phase": "capture", "tool": str(expected_tool)})
    return calls


def _normalize_template_call(call: Any) -> dict[str, str]:
    if not isinstance(call, dict):
        return {"phase": "before_answer", "tool": str(call)}
    phase = str(call.get("phase") or call.get("stage") or "before_answer")
    tool = _normalize_tool_name(str(call.get("tool") or call.get("name") or ""))
    return {"phase": phase, "tool": tool}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_calls(path: Path) -> list[dict[str, Any]]:
    return _load_calls_text(path.read_text(encoding="utf-8"))


def _load_calls_text(text: str) -> list[dict[str, Any]]:
    calls, _evidence = _load_calls_bundle_text(text)
    return calls


def _load_calls_bundle(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    return _load_calls_bundle_text(path.read_text(encoding="utf-8"))


def _load_calls_bundle_text(text: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    stripped = text.strip()
    if not stripped:
        return [], {}
    evidence: dict[str, str] = {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            calls = [json.loads(line) for line in text.splitlines() if line.strip()]
            evidence = _extract_record_live_evidence(calls)
        except json.JSONDecodeError:
            calls = _parse_plaintext_calls(text)
            evidence = _extract_plaintext_live_evidence(text)
    else:
        if isinstance(payload, dict):
            evidence = _extract_json_live_evidence(payload)
            calls = payload.get("calls", payload.get("transcript"))
        else:
            calls = payload
            evidence = _extract_record_live_evidence(calls)
    if not isinstance(calls, list):
        raise ValueError("calls payload must be a JSON array or JSONL records")
    return _normalize_calls_payload(calls), evidence


def _normalize_calls_payload(calls: list[Any]) -> list[dict[str, Any]]:
    try:
        return [_normalize_call(call) for call in calls]
    except ValueError as exc:
        stream_calls = _parse_claude_stream_json_calls(calls)
        if stream_calls:
            return [_normalize_call(call) for call in stream_calls]
        raise exc


def _parse_claude_stream_json_calls(records: list[Any]) -> list[dict[str, Any]]:
    """Extract Engram tool calls from Claude Code `--output-format stream-json`.

    Claude stream JSON is an event log, not a compact call transcript. Tool-use
    blocks do not include Engram's adoption phases, so capture tools are mapped
    to `capture` and other Engram tools are mapped to `before_answer`.
    """
    calls: list[dict[str, Any]] = []
    saw_claude_stream_shape = False
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("type") in {"system", "assistant", "user", "result"}:
            saw_claude_stream_shape = True
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            tool_name = _extract_tool_name(block)
            if not tool_name:
                continue
            normalized_tool = _normalize_tool_name(tool_name)
            if not _is_engram_tool_name(tool_name, normalized_tool):
                continue
            calls.append(
                {
                    "phase": (
                        "capture"
                        if normalized_tool in ENGRAM_CAPTURE_TOOLS
                        else "before_answer"
                    ),
                    "tool": normalized_tool,
                    "source": "claude_stream_json",
                }
            )
    return calls if saw_claude_stream_shape else []


def _parse_plaintext_calls(text: str) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []
    current_phase: str | None = None
    phase_less_tool_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        phase = _extract_plaintext_phase(line)
        if phase:
            current_phase = phase
        tool = _extract_plaintext_tool(line)
        if not tool:
            continue
        call_phase = phase or current_phase
        if not call_phase:
            phase_less_tool_lines.append(line)
            continue
        calls.append({"phase": call_phase, "tool": tool})

    if phase_less_tool_lines:
        raise ValueError(
            "plaintext call transcript tool lines must include or follow an explicit "
            "before_answer/capture phase"
        )
    if not calls:
        calls = _parse_self_reported_adoption_failure(text)
    if not calls:
        raise ValueError(
            "calls payload must be JSON/JSONL records or plaintext lines with "
            "explicit before_answer/capture phase and Engram tool names"
        )
    return calls


def _extract_plaintext_phase(line: str) -> str | None:
    field_match = _PHASE_FIELD_RE.search(line)
    if field_match:
        return _normalize_phase_name(field_match.group(1))
    prefix_match = _PHASE_PREFIX_RE.match(line)
    if prefix_match:
        phase = _normalize_phase_name(prefix_match.group(1))
        if phase:
            return phase
    normalized = line.strip(" #>*-|`:\t").lower()
    return _normalize_phase_name(normalized)


def _normalize_phase_name(phase_name: str) -> str | None:
    normalized = phase_name.strip(" #>*-|`:\t[]").lower().replace("_", " ")
    canonical = _PHASE_ALIASES.get(normalized)
    if canonical in _PHASE_VALUES:
        return canonical
    return None


def _extract_plaintext_tool(line: str) -> str | None:
    field_match = _TOOL_FIELD_RE.search(line)
    if field_match:
        return _normalize_tool_name(field_match.group(1))
    tool_match = _ENGRAM_TOOL_RE.search(line)
    if tool_match:
        return _normalize_tool_name(tool_match.group(0))
    if _ENGRAM_API_CAPTURE_RE.search(line):
        return "auto_observe"
    return None


def _parse_self_reported_adoption_failure(text: str) -> list[dict[str, str]]:
    """Extract a failed-adoption signal from copied agent prose.

    Real AI-harness transcripts sometimes do not expose raw MCP tool records.
    If the copied notes include the agent's own admission that it routed around
    Engram, turn that into a minimal failed call transcript so operators get the
    adoption failure they need to fix instead of a parse error.
    """
    ignored_engram = bool(_SELF_REPORTED_ENGRAM_IGNORED_RE.search(text))
    file_memory_primary = bool(
        _SELF_REPORTED_FILE_MEMORY_RE.search(text)
        and _SELF_REPORTED_FILE_MEMORY_PRIMARY_RE.search(text)
    )
    if ignored_engram and file_memory_primary:
        return [
            {
                "phase": "before_answer",
                "tool": "read_file_memory",
                "source": "project_local_memory",
                "evidence": "self_reported_file_memory_substituted_for_engram",
            }
        ]
    if ignored_engram:
        return [
            {
                "phase": "before_answer",
                "tool": "self_reported_no_engram_use",
                "source": "agent_self_report",
                "evidence": "self_reported_engram_ignored",
            }
        ]
    return []


def _build_live_evidence_report(
    evidence: dict[str, str],
    *,
    require_live_evidence: bool,
) -> dict[str, Any]:
    missing = [
        field
        for field in _LIVE_EVIDENCE_REQUIRED_FIELDS
        if not evidence.get(field) or _looks_like_placeholder(evidence.get(field))
    ]
    return {
        "required": require_live_evidence,
        "client": evidence.get("client"),
        "captured_at": evidence.get("captured_at"),
        "session_id": evidence.get("session_id"),
        "source": evidence.get("source"),
        "missing": missing if require_live_evidence else [],
    }


def _looks_like_placeholder(value: str | None) -> bool:
    if value is None:
        return True
    stripped = value.strip()
    if not stripped:
        return True
    return (
        stripped.startswith("<")
        and stripped.endswith(">")
        or stripped.lower() in {"todo", "tbd", "replace me", "placeholder"}
    )


def _extract_json_live_evidence(payload: dict[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key in (
        "metadata",
        "evidence",
        "adoption_evidence",
        "adoptionEvidence",
        "live_evidence",
        "liveEvidence",
    ):
        nested = payload.get(key)
        if isinstance(nested, dict):
            evidence.update(_normalize_live_evidence(nested))
    evidence.update(_normalize_live_evidence(payload))
    return evidence


def _extract_record_live_evidence(records: list[Any]) -> dict[str, str]:
    """Infer live client metadata from structured transcript records."""
    evidence: dict[str, str] = {}
    saw_claude_stream = False

    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("type") in {"system", "assistant", "user", "result"}:
            saw_claude_stream = True
        evidence.update(_normalize_live_evidence(record))
        if not evidence.get("session_id") and isinstance(record.get("session_id"), str):
            evidence["session_id"] = str(record["session_id"])
        if not evidence.get("captured_at") and isinstance(record.get("timestamp"), str):
            evidence["captured_at"] = str(record["timestamp"])
        if not evidence.get("captured_at") and isinstance(record.get("capturedAt"), str):
            evidence["captured_at"] = str(record["capturedAt"])

    if saw_claude_stream:
        evidence.setdefault("client", "Claude Code")
        evidence.setdefault("source", "claude_stream_json")
    return evidence


def _extract_plaintext_live_evidence(text: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for line in text.splitlines():
        match = _LIVE_EVIDENCE_FIELD_RE.match(line)
        if not match:
            continue
        key = _normalize_live_evidence_key(match.group(1))
        if key:
            evidence[key] = match.group(2).strip().strip("`")
    return evidence


def _normalize_live_evidence(payload: dict[str, Any]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for key, value in payload.items():
        normalized_key = _normalize_live_evidence_key(str(key))
        if not normalized_key:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            evidence[normalized_key] = text
    return evidence


def _normalize_live_evidence_key(key: str) -> str | None:
    normalized = key.strip().replace("-", "_").replace(" ", "_").lower()
    if normalized in {"client", "harness", "agent_client", "mcp_client"}:
        return "client"
    if normalized in {"captured_at", "capturedat", "timestamp", "recorded_at"}:
        return "captured_at"
    if normalized in {
        "session",
        "session_id",
        "sessionid",
        "thread",
        "thread_id",
        "threadid",
        "conversation_id",
        "conversationid",
    }:
        return "session_id"
    if normalized == "source":
        return "source"
    return None


def _normalize_call(call: Any) -> dict[str, Any]:
    if not isinstance(call, dict):
        raise ValueError("each call transcript record must be a JSON object")
    normalized = dict(call)
    if "phase" not in normalized and "stage" in normalized:
        normalized["phase"] = normalized["stage"]
    tool_name = _extract_tool_name(normalized)
    if tool_name:
        normalized["tool"] = _normalize_tool_name(tool_name)
    if "phase" not in normalized:
        raise ValueError("each call transcript record must include phase")
    if "tool" not in normalized:
        raise ValueError("each call transcript record must include tool")
    return normalized


def _extract_tool_name(call: dict[str, Any]) -> str | None:
    for key in ("tool", "name", "tool_name", "toolName", "endpoint", "path", "route", "url"):
        value = call.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = _extract_tool_name(value)
            if nested:
                return nested
    for key in ("function", "call", "tool_call", "toolCall"):
        value = call.get(key)
        if isinstance(value, dict):
            nested = _extract_tool_name(value)
            if nested:
                return nested
    return None


def _normalize_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip()
    for separator in ("__", ".", "/"):
        if separator in normalized:
            normalized = normalized.split(separator)[-1]
    normalized = normalized.replace("-", "_")
    return normalized


def _is_engram_tool_name(raw_tool_name: str, normalized_tool_name: str) -> bool:
    raw = raw_tool_name.strip()
    return (
        raw.startswith("mcp__engram__")
        or raw.startswith("engram.")
        or raw.startswith("engram/")
        or normalized_tool_name
        in {
            "claim_authority",
            "bootstrap_project",
            "get_context",
            "recall",
            "observe",
            "remember",
            "auto_observe",
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Engram MCP adoption transcripts")
    configure_adoption_parser(parser)
    return run_adoption_command(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
