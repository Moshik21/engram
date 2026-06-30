"""Adoption-debt signals for agent Engram usage."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

_IDENTITY_QUERY_MARKERS = (
    "who is",
    "what is my",
    "my name",
    "my preference",
    "remember me",
    "do you know",
    "what did we",
    "prior decision",
    "last time",
)


def build_adoption_debt(
    memory_operation_metrics: Mapping[str, Any] | None,
    *,
    last_context_load: str | None = None,
    context_loaded_this_session: bool = False,
    turns_since_context: int | None = None,
    project_path: str | None = None,
    last_context_project_path: str | None = None,
    session_tool_calls: int = 0,
    query_text: str | None = None,
    personal_entity_count: int | None = None,
) -> dict[str, Any]:
    """Summarize capture-vs-recall imbalance for agent adoption nudges."""
    metrics = memory_operation_metrics if isinstance(memory_operation_metrics, Mapping) else {}
    operation_counts = metrics.get("operation_counts") or {}
    source_counts = metrics.get("source_counts") or {}

    episodes_captured = _int(operation_counts.get("observe"))
    harness_captured = _int(source_counts.get("api_auto_observe"))
    agent_context_loads = _int(source_counts.get("mcp_context"))
    agent_recalls = _int(source_counts.get("mcp_recall")) + _int(
        source_counts.get("mcp_session_prime")
    )
    agent_recall_total = agent_context_loads + agent_recalls
    harness_capture_active = harness_captured > 0

    if turns_since_context is not None:
        turns_without_recall = max(0, int(turns_since_context))
    else:
        turns_without_recall = max(0, episodes_captured - agent_recall_total)

    if context_loaded_this_session:
        turns_without_recall = 0

    wake_reason = _resolve_wake_reason(
        context_loaded_this_session=context_loaded_this_session,
        project_path=project_path,
        last_context_project_path=last_context_project_path,
        session_tool_calls=session_tool_calls,
        harness_capture_active=harness_capture_active,
        harness_captured=harness_captured,
        agent_recall_total=agent_recall_total,
        turns_without_recall=turns_without_recall,
        query_text=query_text,
        personal_entity_count=personal_entity_count,
    )

    consequence = _consequence_text(
        episodes_captured=episodes_captured,
        harness_captured=harness_captured,
        agent_recall_total=agent_recall_total,
        turns_without_recall=turns_without_recall,
        wake_reason=wake_reason,
    )

    payload: dict[str, Any] = {
        "turnsWithoutRecall": turns_without_recall,
        "lastContextLoad": last_context_load,
        "consequence": consequence,
        "episodesCaptured": episodes_captured,
        "harnessEpisodesCaptured": harness_captured,
        "agentRecallCount": agent_recall_total,
        "contextLoadedThisSession": context_loaded_this_session,
        "harnessCaptureActive": harness_capture_active,
    }
    if wake_reason:
        payload["wakeReason"] = wake_reason
        payload["suggestedAction"] = _suggested_action(wake_reason)
    return payload


def adoption_debt_is_actionable(debt: Mapping[str, Any] | None) -> bool:
    """Return whether debt should be piggybacked on tool responses."""
    if not isinstance(debt, Mapping):
        return False
    return bool(debt.get("wakeReason"))


def harness_capture_active_from_metrics(
    memory_operation_metrics: Mapping[str, Any] | None,
) -> bool:
    metrics = memory_operation_metrics if isinstance(memory_operation_metrics, Mapping) else {}
    source_counts = metrics.get("source_counts") or {}
    return _int(source_counts.get("api_auto_observe")) > 0


def _resolve_wake_reason(
    *,
    context_loaded_this_session: bool,
    project_path: str | None,
    last_context_project_path: str | None,
    session_tool_calls: int,
    harness_capture_active: bool,
    harness_captured: int,
    agent_recall_total: int,
    turns_without_recall: int,
    query_text: str | None,
    personal_entity_count: int | None,
) -> str | None:
    if _project_switched(project_path, last_context_project_path):
        return "project_switched"

    if context_loaded_this_session:
        if harness_capture_active and agent_recall_total > 0:
            return None
        return None

    if session_tool_calls <= 1:
        return "session_unprimed"

    if _query_hints_identity(query_text) and (
        personal_entity_count is None or personal_entity_count < 3
    ):
        return "identity_query"

    if harness_captured > 0 and agent_recall_total == 0 and turns_without_recall > 2:
        return "capture_without_recall"

    if turns_without_recall > 2 and agent_recall_total == 0:
        return "capture_without_recall"

    return None


def _suggested_action(wake_reason: str) -> str:
    if wake_reason == "project_switched":
        return "get_context + search_artifacts"
    if wake_reason == "identity_query":
        return "recall"
    if wake_reason in {"session_unprimed", "capture_without_recall"}:
        return "get_context"
    return "get_context"


def _project_switched(
    project_path: str | None,
    last_context_project_path: str | None,
) -> bool:
    current = _normalize_project_path(project_path)
    previous = _normalize_project_path(last_context_project_path)
    if not current or not previous:
        return False
    return current != previous


def _normalize_project_path(path: str | None) -> str | None:
    if not path or not str(path).strip():
        return None
    try:
        return str(Path(path).expanduser().resolve())
    except OSError:
        return str(path).strip()


def _query_hints_identity(query_text: str | None) -> bool:
    if not query_text:
        return False
    normalized = query_text.lower()
    return any(marker in normalized for marker in _IDENTITY_QUERY_MARKERS)


def _consequence_text(
    *,
    episodes_captured: int,
    harness_captured: int,
    agent_recall_total: int,
    turns_without_recall: int,
    wake_reason: str | None,
) -> str:
    if wake_reason == "session_unprimed":
        return "Session not primed — call get_context before the first substantive answer."
    if wake_reason == "project_switched":
        return "Project changed since last context load — reload get_context and search_artifacts."
    if wake_reason == "identity_query":
        return "Identity or preference query with thin personal graph — call recall."
    if wake_reason == "capture_without_recall":
        captured = harness_captured or episodes_captured
        return (
            f"{captured} episodes captured by harness, {agent_recall_total} agent recalls — "
            "memory is compounding but invisible until you call get_context/recall."
        )
    if episodes_captured <= 0 and harness_captured <= 0:
        return "No episodes captured yet — call get_context to start the compounding loop."
    if agent_recall_total <= 0:
        captured = harness_captured or episodes_captured
        return (
            f"{captured} episodes captured, 0 recalled — "
            "memory is compounding in storage but invisible until you call "
            "get_context/recall/search_artifacts."
        )
    if turns_without_recall > 0:
        return (
            f"{episodes_captured} episodes captured, {agent_recall_total} agent recalls — "
            f"{turns_without_recall} turns since last context load."
        )
    return (
        f"{episodes_captured} episodes captured, {agent_recall_total} agent recalls — "
        "context loop active."
    )


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
