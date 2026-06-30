"""MCP-facing memory authority and onboarding contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any

from engram.retrieval.runtime_state import build_runtime_state_surface

ENGRAM_OWNS = [
    "cross-project user facts",
    "identity and preferences",
    "corrections",
    "durable decisions",
    "personal and work relationships",
    "ongoing goals and commitments",
    "long-tail recall that should survive switching tools or projects",
]

PROJECT_LOCAL_OWNS = [
    "repo-specific coding conventions",
    "current-task scratch notes",
    "temporary implementation details that should not follow the user",
]

ENGRAM_CAPTURE_TOOLS = {"observe", "remember", "auto_observe"}
ENGRAM_CAPTURE_TOOL_EQUIVALENTS = {
    "observe": {"observe", "auto_observe"},
    "remember": {"remember"},
}
FILE_MEMORY_TOOL_MARKERS = {
    "file_memory",
    "local_memory",
    "project_local_memory",
    "read_file_memory",
    "read_local_memory",
    "read_project_memory",
}


async def build_mcp_memory_authority_surface(
    manager: Any,
    *,
    group_id: str,
    project_path: str | None = None,
    user_message: str | None = None,
    file_memory_present: bool = False,
) -> dict[str, Any]:
    """Return the contract an agent should use when deciding how to use Engram."""
    runtime_state = await build_runtime_state_surface(
        manager,
        group_id=group_id,
        project_path=project_path,
    )
    onboarding = _build_onboarding(runtime_state, project_path=project_path)
    return {
        "authority": {
            "source_of_truth": "portable_cross_context_memory",
            "summary": (
                "Engram owns memory that should follow the user across coding agents, "
                "general chat, project work, and other AI harnesses."
            ),
            "engram_owns": ENGRAM_OWNS,
            "project_local_files_own": PROJECT_LOCAL_OWNS,
            "conflict_policy": (
                "Prefer the user's latest statement. If Engram is stale or wrong, "
                "repair it with forget/remember instead of routing around it."
            ),
        },
        "routing": {
            "use_engram_for": ENGRAM_OWNS,
            "use_project_local_files_for": PROJECT_LOCAL_OWNS,
            "do_not_skip_engram_when": [
                "a file-based memory system is also present",
                "the runtime is connected but has not been bootstrapped yet",
                (
                    "the user asks about people, preferences, prior decisions, "
                    "or cross-project context"
                ),
            ],
        },
        "onboarding": onboarding,
        "agent_protocol": _build_agent_protocol(
            onboarding,
            user_message=user_message,
            file_memory_present=file_memory_present,
        ),
        "runtime": runtime_state,
        "lifecycle": ["Capture", "Cue", "Project", "Recall", "Consolidate"],
    }


def _build_onboarding(runtime_state: Mapping[str, Any], *, project_path: str | None) -> dict:
    artifacts = _mapping(runtime_state.get("artifactBootstrap"))
    stats = _mapping(runtime_state.get("stats"))
    recall_metrics = _mapping(stats.get("recallMetrics"))
    epistemic_metrics = _mapping(stats.get("epistemicMetrics"))
    artifact_count = int(artifacts.get("artifactCount") or 0)
    last_observed = artifacts.get("lastObservedAt")
    stale_count = int(artifacts.get("staleArtifactCount") or 0)
    artifact_gap = artifact_count == 0 or last_observed is None
    metrics_gap = _metrics_are_empty_or_zero(recall_metrics) and _metrics_are_empty_or_zero(
        epistemic_metrics
    )
    fresh_runtime = (
        artifact_gap
        and metrics_gap
    )
    needs_project_bootstrap = artifact_gap or stale_count > 0
    should_bootstrap = bool(
        project_path
        and artifacts.get("enabled", True)
        and needs_project_bootstrap
    )
    actions = []
    if should_bootstrap:
        actions.append(
            {
                "tool": "bootstrap_project",
                "args": {"project_path": project_path},
                "reason": "Seed project artifacts before judging recall usefulness.",
            }
        )
    elif artifacts.get("enabled", True) and needs_project_bootstrap:
        actions.append(
            {
                "tool": "bootstrap_project",
                "args": {"project_path": "<current_project_path>"},
                "reason": (
                    "Runtime is fresh; provide a project path before assuming "
                    "Engram is empty."
                ),
            }
        )
    actions.extend(
        [
            {
                "tool": "get_context",
                "args": {"format": "structured"},
                "reason": "Load portable user context before a substantive response.",
            },
            {
                "tool": "recall",
                "args": {"query": "<people, project, decision, or prior-context reference>"},
                "reason": "Look up prior context whenever it could change the answer.",
            },
        ]
    )
    state = (
        "fresh_runtime"
        if fresh_runtime
        else "needs_project_bootstrap"
        if needs_project_bootstrap
        else "ready"
    )
    return {
        "state": state,
        "should_bootstrap": should_bootstrap,
        "reason": (
            "Connected but empty/fresh Engram runtime; bootstrap is onboarding, not failure."
            if fresh_runtime
            else (
                "Project artifacts are missing or stale; bootstrap before judging "
                "recall usefulness."
            )
            if needs_project_bootstrap
            else "Engram has runtime evidence available; use recall/context normally."
        ),
        "recommended_actions": actions,
    }


def _build_agent_protocol(
    onboarding: Mapping[str, Any],
    *,
    user_message: str | None,
    file_memory_present: bool,
) -> dict[str, Any]:
    text = (user_message or "").strip()
    before_answer = [
        action
        for action in onboarding.get("recommended_actions", [])
        if isinstance(action, Mapping)
        and action.get("tool") in {"bootstrap_project", "get_context"}
    ]
    if text and _message_needs_recall(text):
        before_answer.append(
            {
                "tool": "recall",
                "args": {"query": text},
                "reason": "Prior context could change the answer.",
            }
        )

    capture = _capture_decision(text)
    return {
        "file_memory_present": file_memory_present,
        "file_memory_is_substitute": False,
        "rule": (
            "Use file-local memory as visible project context, not as a substitute "
            "for Engram's portable memory authority."
        ),
        "before_answer": before_answer,
        "capture": capture,
        "required_tools_before_answer": [
            str(action["tool"])
            for action in before_answer
            if isinstance(action, Mapping) and action.get("tool")
        ],
        "verification": _build_protocol_verification(before_answer, capture),
    }


def _build_protocol_verification(
    before_answer: list[Mapping[str, Any]],
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    capture_required = capture.get("destination") == "engram" and bool(capture.get("tool"))
    example = [
        {"phase": "before_answer", "tool": str(action["tool"])}
        for action in before_answer
        if isinstance(action, Mapping) and action.get("tool")
    ]
    if capture_required:
        example.append({"phase": "capture", "tool": str(capture["tool"])})
    return {
        "command": (
            "engram adoption --authority claim-authority.json "
            "--calls mcp-calls.jsonl"
        ),
        "live_evidence_command": (
            "engram adoption --authority claim-authority.json "
            "--calls live-harness-transcript.json --require-live-evidence"
        ),
        "capture_required": capture_required,
        "transcript_schema": {
            "format": "jsonl",
            "required_fields": ["phase", "tool"],
            "phase_values": ["before_answer", "capture"],
            "example": example,
        },
        "live_evidence_schema": {
            "format": "json",
            "required_metadata_fields": ["client", "capturedAt", "source"],
            "optional_metadata_fields": ["sessionId"],
            "example": {
                "metadata": {
                    "client": "<Claude Code | Cursor | Windsurf | other MCP client>",
                    "capturedAt": "<ISO-8601 timestamp>",
                    "sessionId": "<client session/thread id if available>",
                    "source": "<copied_mcp_log | client_trace | harness_export>",
                },
                "calls": example,
            },
        },
    }


def _capture_decision(text: str) -> dict[str, Any]:
    if not text:
        return {
            "destination": "none",
            "tool": None,
            "reason": "No user message was provided for capture routing.",
        }
    if _project_local_only(text):
        return {
            "destination": "project_local",
            "tool": None,
            "reason": "Treat as current-task scratch unless it becomes cross-context.",
        }
    if _message_is_high_signal(text):
        return {
            "destination": "engram",
            "tool": "remember",
            "reason": "High-signal cross-context user fact or durable decision.",
        }
    return {
        "destination": "engram",
        "tool": None,
        "reason": (
            "Harness auto-capture handles routine turns when installed; "
            "use observe only for explicit store requests or harness-invisible context."
        ),
    }


def _message_needs_recall(text: str) -> bool:
    normalized = text.lower()
    triggers = (
        "engram",
        "project",
        "decision",
        "remember",
        "preference",
        "goal",
        "prior",
        "previous",
        "last time",
        "what did we",
        "who is",
        "what is my",
    )
    return any(trigger in normalized for trigger in triggers)


def _message_is_high_signal(text: str) -> bool:
    normalized = text.lower()
    triggers = (
        "i am ",
        "i'm ",
        "i prefer",
        "i want",
        "my goal",
        "the goal",
        "remember",
        "correction",
        "always",
        "never",
        "cross-context",
        "source of truth",
        "building engram",
    )
    return any(trigger in normalized for trigger in triggers)


def _project_local_only(text: str) -> bool:
    normalized = text.lower()
    local_markers = (
        "scratch",
        "current task",
        "temporary implementation",
        "local fixture",
    )
    return (
        any(marker in normalized for marker in local_markers)
        and not _message_is_high_signal(text)
    )


def validate_agent_protocol_calls(
    protocol: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate that an observed client transcript followed `agent_protocol`.

    This is intentionally side-effect-free: real MCP clients can emit a compact
    call transcript and use this validator to prove they did not treat
    file-based memory as a substitute for Engram.
    """
    required = [
        str(tool)
        for tool in protocol.get("required_tools_before_answer") or []
        if tool
    ]
    before_answer_tools = _tools_for_phase(calls, "before_answer")
    missing_required = [
        tool
        for tool in required
        if tool not in before_answer_tools
    ]
    required_sequence_satisfied = _sequence_in_order(before_answer_tools, required)

    capture = _mapping(protocol.get("capture"))
    capture_validation = _validate_capture(capture, calls)
    local_memory_tools = _local_memory_tools(calls)
    file_memory_substituted = bool(
        protocol.get("file_memory_present")
        and local_memory_tools
        and (missing_required or not required_sequence_satisfied)
    )

    failures: list[str] = []
    if missing_required:
        failures.append("missing_required_before_answer_tools")
    if not required_sequence_satisfied:
        failures.append("required_before_answer_tools_out_of_order")
    if capture_validation["missing"]:
        failures.append("missing_required_capture_tool")
    if capture_validation["unexpected_engram_capture_tools"]:
        failures.append("unexpected_engram_capture")
    if file_memory_substituted:
        failures.append("file_memory_used_as_substitute")

    return {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "required_tools_before_answer": {
            "expected": required,
            "observed": before_answer_tools,
            "missing": missing_required,
            "in_order": required_sequence_satisfied,
        },
        "capture": capture_validation,
        "file_memory": {
            "present": bool(protocol.get("file_memory_present")),
            "observed_tools": local_memory_tools,
            "substituted_for_engram": file_memory_substituted,
        },
    }


def _validate_capture(
    capture: Mapping[str, Any],
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_tool = capture.get("tool")
    destination = capture.get("destination")
    capture_tools = _tools_for_phase(calls, "capture")
    unexpected_engram_capture_tools: list[str] = []
    missing = False

    if destination == "engram" and expected_tool:
        acceptable_tools = ENGRAM_CAPTURE_TOOL_EQUIVALENTS.get(
            str(expected_tool),
            {str(expected_tool)},
        )
        missing = not any(tool in acceptable_tools for tool in capture_tools)
    elif destination == "engram" and not expected_tool:
        # Harness-first protocol: routine capture is infrastructure; any harness
        # or agent capture tool in the transcript is acceptable evidence.
        missing = not any(tool in ENGRAM_CAPTURE_TOOLS for tool in capture_tools)
    elif destination in {"none", "project_local"}:
        unexpected_engram_capture_tools = [
            tool
            for tool in capture_tools
            if tool in ENGRAM_CAPTURE_TOOLS
        ]

    return {
        "destination": destination,
        "expected_tool": expected_tool,
        "observed_tools": capture_tools,
        "missing": missing,
        "unexpected_engram_capture_tools": unexpected_engram_capture_tools,
    }


def _tools_for_phase(
    calls: Sequence[Mapping[str, Any]],
    phase: str,
) -> list[str]:
    return [
        str(call["tool"])
        for call in calls
        if call.get("phase") == phase and call.get("tool")
    ]


def _sequence_in_order(observed: Sequence[str], expected: Sequence[str]) -> bool:
    if not expected:
        return True
    observed_iter = iter(observed)
    return all(any(tool == expected_tool for tool in observed_iter) for expected_tool in expected)


def _local_memory_tools(calls: Sequence[Mapping[str, Any]]) -> list[str]:
    tools = []
    for call in calls:
        tool = str(call.get("tool") or "")
        source = str(call.get("source") or "")
        if tool in FILE_MEMORY_TOOL_MARKERS or source in FILE_MEMORY_TOOL_MARKERS:
            tools.append(tool or source)
    return tools


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _metrics_are_empty_or_zero(metrics: Mapping[str, Any]) -> bool:
    values = list(_numeric_values(metrics))
    return not values or all(value == 0 for value in values)


def _numeric_values(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, Number):
        yield float(value)
        return
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _numeric_values(nested)
        return
    if isinstance(value, list | tuple):
        for nested in value:
            yield from _numeric_values(nested)
