"""Shared Capture -> Cue -> Project -> Recall -> Consolidate summary contract."""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from typing import Any

from engram.config import ActivationConfig
from engram.consolidation.audit_reader import ConsolidationAuditReader
from engram.consolidation.presenter import serialize_cycle_summary

LOOP = ["capture", "cue", "project", "recall", "consolidate"]
CAPTURE_ACTIVE_STATUSES = {"queued", "pending", "processing", "extracting"}
PROJECT_ACTIVE_STATES = {"queued", "cued", "scheduled", "projecting"}
EMPTY_INTENTION_SUMMARY = {
    "activeCount": 0,
    "refreshContextCount": 0,
    "afterConsolidationCount": 0,
    "pinnedResultCount": 0,
    "needsRefreshCount": 0,
    "latestRefreshedAt": None,
}


def _enum_value(value: object) -> str | None:
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        return enum_value
    return value if isinstance(value, str) else None


def _iso_z(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return f"{value.isoformat()}Z"


def _int_metric(source: dict[str, Any], key: str) -> int:
    value = source.get(key, 0)
    return int(value or 0)


def _float_metric(source: dict[str, Any], key: str) -> float:
    value = source.get(key, 0.0)
    return float(value or 0.0)


def _cycle_has_phase_issue(cycle: dict[str, Any] | None) -> bool:
    if not cycle:
        return False
    phase_issue = cycle.get("phase_issue")
    if isinstance(phase_issue, str) and phase_issue.strip():
        return True
    phases = cycle.get("phases") or []
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        if phase.get("status") == "error":
            return True
        error = phase.get("error")
        if isinstance(error, str) and error.strip():
            return True
    return False


def _state_counts(projection_metrics: dict[str, Any]) -> dict[str, int]:
    raw = projection_metrics.get("state_counts") or {}
    return {
        "queued": _int_metric(raw, "queued"),
        "cued": _int_metric(raw, "cued"),
        "cueOnly": _int_metric(raw, "cue_only"),
        "scheduled": _int_metric(raw, "scheduled"),
        "projecting": _int_metric(raw, "projecting"),
        "projected": _int_metric(raw, "projected"),
        "merged": _int_metric(raw, "merged"),
        "failed": _int_metric(raw, "failed"),
        "deadLetter": _int_metric(raw, "dead_letter"),
    }


def _serialize_episode(episode: Any, cue: Any | None = None) -> dict[str, Any]:
    return {
        "episodeId": episode.id,
        "content": episode.content[:200] if episode.content else None,
        "source": episode.source or "unknown",
        "status": _enum_value(episode.status),
        "projectionState": _enum_value(getattr(episode, "projection_state", None)),
        "lastProjectionReason": getattr(episode, "last_projection_reason", None),
        "lastProjectedAt": _iso_z(getattr(episode, "last_projected_at", None)),
        "conversationDate": _iso_z(getattr(episode, "conversation_date", None)),
        "createdAt": _iso_z(episode.created_at),
        "updatedAt": _iso_z(episode.updated_at),
        "error": episode.error,
        "retryCount": episode.retry_count,
        "processingDurationMs": episode.processing_duration_ms,
        "entities": [],
        "factsCount": 0,
        "cue": None
        if cue is None
        else {
            "cueText": cue.cue_text[:240] if cue.cue_text else None,
            "projectionState": _enum_value(cue.projection_state),
            "routeReason": cue.route_reason,
            "hitCount": cue.hit_count,
            "surfacedCount": cue.surfaced_count,
            "selectedCount": cue.selected_count,
            "usedCount": cue.used_count,
            "nearMissCount": cue.near_miss_count,
            "policyScore": cue.policy_score,
            "projectionAttempts": cue.projection_attempts,
            "lastHitAt": _iso_z(cue.last_hit_at),
            "lastFeedbackAt": _iso_z(cue.last_feedback_at),
            "lastProjectedAt": _iso_z(cue.last_projected_at),
        },
    }


async def _recent_cycles(
    *,
    graph_store: Any | None,
    consolidation_engine: Any | None,
    consolidation_reader: ConsolidationAuditReader | None,
    group_id: str,
    limit: int,
) -> list[Any]:
    if consolidation_reader is not None:
        return await consolidation_reader.recent_cycles(group_id, limit=limit)

    get_recent_cycles = getattr(consolidation_engine, "get_recent_cycles", None)
    if callable(get_recent_cycles):
        return await get_recent_cycles(group_id, limit=limit)

    db = getattr(graph_store, "_db", None)
    if db is None:
        return []

    from engram.consolidation.store import SQLiteConsolidationStore

    store = SQLiteConsolidationStore(":memory:")
    await store.initialize(db=db)
    return await store.get_recent_cycles(group_id, limit=limit)


async def _intention_summary(manager: Any, group_id: str) -> dict[str, Any]:
    """Summarize active prospective-memory state for the Recall stage."""
    list_intentions = getattr(manager, "list_intentions", None)
    if not callable(list_intentions):
        return dict(EMPTY_INTENTION_SUMMARY)

    result = list_intentions(group_id=group_id, enabled_only=True)
    intentions = await result if inspect.isawaitable(result) else result

    active_count = 0
    refresh_context_count = 0
    after_consolidation_count = 0
    pinned_result_count = 0
    needs_refresh_count = 0
    latest_refreshed_at: str | None = None
    for entity in intentions or []:
        active_count += 1
        attrs = getattr(entity, "attributes", None)
        if not isinstance(attrs, dict):
            continue
        if attrs.get("trigger_type") != "refresh_context":
            continue
        refresh_context_count += 1
        refresh_trigger = attrs.get("refresh_trigger") or "manual"
        pinned_result = attrs.get("pinned_result")
        last_refreshed = attrs.get("last_refreshed")
        if pinned_result:
            pinned_result_count += 1
        if isinstance(last_refreshed, str) and (
            latest_refreshed_at is None or last_refreshed > latest_refreshed_at
        ):
            latest_refreshed_at = last_refreshed
        if refresh_trigger == "after_consolidation":
            after_consolidation_count += 1
            if not pinned_result:
                needs_refresh_count += 1

    return {
        "activeCount": active_count,
        "refreshContextCount": refresh_context_count,
        "afterConsolidationCount": after_consolidation_count,
        "pinnedResultCount": pinned_result_count,
        "needsRefreshCount": needs_refresh_count,
        "latestRefreshedAt": latest_refreshed_at,
    }


async def build_lifecycle_summary(
    *,
    group_id: str,
    manager: Any,
    graph_store: Any | None = None,
    consolidation_engine: Any | None = None,
    consolidation_reader: ConsolidationAuditReader | None = None,
    consolidation_scheduler: Any | None = None,
    pressure_accumulator: Any | None = None,
    activation_config: ActivationConfig | None = None,
    top_n: int = 10,
    episode_limit: int = 5,
    cycle_limit: int = 10,
) -> dict[str, Any]:
    """Build the shared semantic brain-loop summary for REST, MCP, and UI."""
    if graph_store is None:
        get_lifecycle_graph_store = getattr(manager, "get_lifecycle_graph_store", None)
        if get_lifecycle_graph_store is not None:
            graph_store = get_lifecycle_graph_store()
    graph_state = await manager.get_graph_state(
        group_id=group_id,
        top_n=top_n,
        include_edges=False,
    )
    stats = graph_state.get("stats", {})
    cue_metrics = stats.get("cue_metrics") or {}
    projection_metrics = stats.get("projection_metrics") or {}
    state_counts = _state_counts(projection_metrics)

    serialized_episodes = []
    if graph_store is not None:
        get_episodes_paginated = getattr(graph_store, "get_episodes_paginated", None)
        if get_episodes_paginated is not None:
            recent_episodes, _cursor = await get_episodes_paginated(
                group_id=group_id,
                limit=max(0, episode_limit),
            )
            get_episode_cue = getattr(graph_store, "get_episode_cue", None)
            for episode in recent_episodes:
                cue = None
                if get_episode_cue is not None:
                    cue_result = get_episode_cue(episode.id, group_id)
                    cue = await cue_result if inspect.isawaitable(cue_result) else cue_result
                serialized_episodes.append(_serialize_episode(episode, cue))

    latest_episode = serialized_episodes[0] if serialized_episodes else None
    capture_active_count = sum(
        1 for episode in serialized_episodes if episode["status"] in CAPTURE_ACTIVE_STATUSES
    )
    project_active_count = sum(state_counts[state] for state in PROJECT_ACTIVE_STATES)
    project_failed_count = state_counts["failed"] + state_counts["deadLetter"]

    recent_cycles = await _recent_cycles(
        graph_store=graph_store,
        consolidation_engine=consolidation_engine,
        consolidation_reader=consolidation_reader,
        group_id=group_id,
        limit=max(1, cycle_limit),
    )
    latest_cycle = serialize_cycle_summary(recent_cycles[0]) if recent_cycles else None

    pressure_payload = None
    if pressure_accumulator is not None:
        snapshot = pressure_accumulator.get_snapshot(group_id)
        if snapshot and activation_config is not None:
            pressure_payload = {
                "value": round(pressure_accumulator.get_pressure(group_id, activation_config), 2),
                "threshold": activation_config.consolidation_pressure_threshold,
                "episodesSinceLast": snapshot.episodes_since_last,
                "entitiesCreated": snapshot.entities_created,
                "lastCycleTime": snapshot.last_cycle_time,
            }

    top_activated = [
        {
            "id": item["id"],
            "name": item["name"],
            "entityType": item["entity_type"],
            "summary": item.get("summary"),
            "activation": item.get("activation", 0),
            "accessCount": item.get("access_count", 0),
        }
        for item in graph_state.get("top_activated", [])
    ]

    recall_metrics = stats.get("recall_metrics") or {}
    intention_summary = await _intention_summary(manager, group_id)
    total_episodes = _int_metric(stats, "episodes")
    cycle_count = len(recent_cycles)
    engine_running = bool(getattr(consolidation_engine, "is_running", False))

    return {
        "groupId": group_id,
        "generatedAt": _iso_z(datetime.now(timezone.utc)),
        "loop": LOOP,
        "totals": {
            "episodes": total_episodes,
            "cues": _int_metric(cue_metrics, "cue_count"),
            "projected": state_counts["projected"],
            "cycles": cycle_count,
            "entities": _int_metric(stats, "entities"),
            "relationships": _int_metric(stats, "relationships"),
        },
        "capture": {
            "status": "active" if capture_active_count else "ready",
            "episodeCount": total_episodes,
            "activeCount": capture_active_count,
            "latestEpisode": latest_episode,
        },
        "cue": {
            "status": "attention"
            if _int_metric(cue_metrics, "episodes_without_cues") > 0
            else "ready",
            "cueCount": _int_metric(cue_metrics, "cue_count"),
            "episodesWithoutCues": _int_metric(cue_metrics, "episodes_without_cues"),
            "coverage": _float_metric(cue_metrics, "cue_coverage"),
            "hitCount": _int_metric(cue_metrics, "cue_hit_count"),
            "surfacedCount": _int_metric(cue_metrics, "cue_surfaced_count"),
            "selectedCount": _int_metric(cue_metrics, "cue_selected_count"),
            "usedCount": _int_metric(cue_metrics, "cue_used_count"),
            "nearMissCount": _int_metric(cue_metrics, "cue_near_miss_count"),
            "avgPolicyScore": _float_metric(cue_metrics, "avg_policy_score"),
            "projectionConversionRate": _float_metric(
                cue_metrics,
                "cue_to_projection_conversion_rate",
            ),
        },
        "project": {
            "status": "attention"
            if project_failed_count
            else "active"
            if project_active_count
            else "ready",
            "projectedCount": state_counts["projected"],
            "activeCount": project_active_count,
            "failedCount": state_counts["failed"],
            "deadLetterCount": state_counts["deadLetter"],
            "failureRate": _float_metric(projection_metrics, "failure_rate"),
            "stateCounts": state_counts,
        },
        "recall": {
            "status": "active" if top_activated or intention_summary["activeCount"] else "ready",
            "activeEntityCount": _int_metric(stats, "active_entities"),
            "topScore": top_activated[0]["activation"] if top_activated else 0,
            "triggerCount": _int_metric(recall_metrics, "trigger_count"),
            "topActivated": top_activated,
            "intentions": intention_summary,
        },
        "consolidate": {
            "status": "attention"
            if latest_cycle
            and (
                latest_cycle.get("status") == "failed"
                or _cycle_has_phase_issue(latest_cycle)
            )
            else "active"
            if engine_running
            else "ready",
            "isRunning": engine_running,
            "schedulerActive": bool(getattr(consolidation_scheduler, "is_active", False)),
            "cycleCount": cycle_count,
            "pressure": pressure_payload,
            "latestCycle": latest_cycle,
        },
        "recentEpisodes": serialized_episodes,
    }


class LifecycleSummaryService:
    """Build the shared brain-loop summary behind route-facing manager facades."""

    def __init__(
        self,
        *,
        manager: Any,
        activation_config: ActivationConfig,
    ) -> None:
        self._manager = manager
        self._activation_config = activation_config

    async def get_lifecycle_summary(
        self,
        *,
        group_id: str,
        consolidation_engine: Any | None = None,
        consolidation_reader: ConsolidationAuditReader | None = None,
        consolidation_scheduler: Any | None = None,
        pressure_accumulator: Any | None = None,
        activation_config: ActivationConfig | None = None,
        episode_limit: int = 5,
        cycle_limit: int = 10,
    ) -> dict[str, Any]:
        """Return the shared semantic Capture -> Cue -> Project -> Recall -> Consolidate view."""
        return await build_lifecycle_summary(
            group_id=group_id,
            manager=self._manager,
            consolidation_engine=consolidation_engine,
            consolidation_reader=consolidation_reader,
            consolidation_scheduler=consolidation_scheduler,
            pressure_accumulator=pressure_accumulator,
            activation_config=activation_config or self._activation_config,
            episode_limit=episode_limit,
            cycle_limit=cycle_limit,
        )
