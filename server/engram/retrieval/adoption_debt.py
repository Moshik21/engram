"""Adoption-debt signals for agent Engram usage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def build_adoption_debt(
    memory_operation_metrics: Mapping[str, Any] | None,
    *,
    last_context_load: str | None = None,
    context_loaded_this_session: bool = False,
    turns_since_context: int | None = None,
) -> dict[str, Any]:
    """Summarize capture-vs-recall imbalance for agent adoption nudges."""
    metrics = memory_operation_metrics if isinstance(memory_operation_metrics, Mapping) else {}
    operation_counts = metrics.get("operation_counts") or {}
    source_counts = metrics.get("source_counts") or {}

    episodes_captured = _int(operation_counts.get("observe"))
    agent_context_loads = _int(source_counts.get("mcp_context"))
    agent_recalls = _int(source_counts.get("mcp_recall")) + _int(
        source_counts.get("mcp_session_prime")
    )
    agent_recall_total = agent_context_loads + agent_recalls

    if turns_since_context is not None:
        turns_without_recall = max(0, int(turns_since_context))
    else:
        turns_without_recall = max(0, episodes_captured - agent_recall_total)

    if context_loaded_this_session:
        turns_without_recall = 0

    consequence = _consequence_text(
        episodes_captured=episodes_captured,
        agent_recall_total=agent_recall_total,
        turns_without_recall=turns_without_recall,
    )

    return {
        "turnsWithoutRecall": turns_without_recall,
        "lastContextLoad": last_context_load,
        "consequence": consequence,
        "episodesCaptured": episodes_captured,
        "agentRecallCount": agent_recall_total,
        "contextLoadedThisSession": context_loaded_this_session,
    }


def adoption_debt_is_actionable(debt: Mapping[str, Any] | None) -> bool:
    """Return whether debt should be piggybacked on tool responses."""
    if not isinstance(debt, Mapping):
        return False
    if debt.get("contextLoadedThisSession"):
        return False
    turns = _int(debt.get("turnsWithoutRecall"))
    episodes = _int(debt.get("episodesCaptured"))
    return turns > 0 or (episodes > 0 and _int(debt.get("agentRecallCount")) == 0)


def _consequence_text(
    *,
    episodes_captured: int,
    agent_recall_total: int,
    turns_without_recall: int,
) -> str:
    if episodes_captured <= 0:
        return "No episodes captured yet — call get_context to start the compounding loop."
    if agent_recall_total <= 0:
        return (
            f"{episodes_captured} episodes captured, 0 recalled — "
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
