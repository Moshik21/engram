"""Runtime control loop for recall-need thresholds and metrics."""

from __future__ import annotations

import inspect
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, cast


@dataclass(frozen=True)
class RecallNeedThresholds:
    """Decision thresholds used by the memory-need analyzer."""

    linguistic_score: float = 0.30
    borderline_score: float = 0.15
    resonance_score: float = 0.45

    def to_dict(self) -> dict[str, float]:
        """Serialize thresholds for telemetry and stats."""
        return {
            "linguistic": round(self.linguistic_score, 4),
            "borderline": round(self.borderline_score, 4),
            "resonance": round(self.resonance_score, 4),
        }


async def resolve_manager_recall_need_thresholds(
    manager: Any,
    group_id: str,
) -> RecallNeedThresholds:
    """Return manager-provided recall-need thresholds with sync/async compatibility."""
    thresholds = cast(Any, manager).get_recall_need_thresholds(group_id)
    if inspect.isawaitable(thresholds):
        thresholds = await thresholds
    if isinstance(thresholds, RecallNeedThresholds):
        return thresholds
    return RecallNeedThresholds()


async def record_manager_memory_need_analysis(
    manager: Any,
    group_id: str,
    need: Any,
) -> None:
    """Record a memory-need analysis through sync or async manager facades."""
    result = cast(Any, manager).record_memory_need_analysis(group_id, need)
    if inspect.isawaitable(result):
        await result


@dataclass
class _AnalysisSample:
    should_recall: bool
    trigger_family: str | None
    analyzer_latency_ms: float
    probe_triggered: bool
    probe_latency_ms: float
    decision_path: str | None
    graph_override_used: bool
    mode_requested: str | None
    mode_executed: str | None
    skip_reason: str | None
    budget_profile: str | None
    cache_hit: bool | None
    cache_satisfied: bool
    budget_skipped: bool


@dataclass
class _InteractionSample:
    interaction_type: str
    result_type: str = "entity"
    memory_id: str | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class _MemoryFeedbackState:
    memory_id: str
    result_type: str = "entity"
    confirmed_count: int = 0
    corrected_count: int = 0
    dismissed_count: int = 0
    used_count: int = 0
    surfaced_count: int = 0
    selected_count: int = 0
    last_confirmed_at: float | None = None
    last_corrected_at: float | None = None
    last_dismissed_at: float | None = None
    last_used_at: float | None = None
    last_interaction_at: float | None = None


@dataclass
class _GroupState:
    analyses: deque[_AnalysisSample]
    interactions: deque[_InteractionSample]
    feedback_by_memory: dict[str, _MemoryFeedbackState] = field(default_factory=dict)
    thresholds: RecallNeedThresholds = field(default_factory=RecallNeedThresholds)


class RecallNeedController:
    """In-memory controller for recall-need metrics and adaptive thresholds."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._states: dict[str, _GroupState] = {}

    def get_thresholds(self, group_id: str) -> RecallNeedThresholds:
        """Return active thresholds for a group."""
        state = self._get_state(group_id)
        if not getattr(self._cfg, "recall_need_adaptive_thresholds_enabled", False):
            return RecallNeedThresholds()
        return state.thresholds

    def record_analysis(self, group_id: str, need) -> None:
        """Record a memory-need analysis decision."""
        state = self._get_state(group_id)
        state.analyses.append(
            _AnalysisSample(
                should_recall=bool(getattr(need, "should_recall", False)),
                trigger_family=getattr(need, "trigger_family", None),
                analyzer_latency_ms=float(getattr(need, "analyzer_latency_ms", 0.0) or 0.0),
                probe_triggered=bool(getattr(need, "probe_triggered", False)),
                probe_latency_ms=float(getattr(need, "probe_latency_ms", 0.0) or 0.0),
                decision_path=getattr(need, "decision_path", None),
                graph_override_used=bool(getattr(need, "graph_override_used", False)),
                mode_requested=getattr(need, "mode_requested", None),
                mode_executed=getattr(need, "mode_executed", None),
                skip_reason=getattr(need, "skip_reason", None),
                budget_profile=getattr(need, "budget_profile", None),
                cache_hit=getattr(need, "cache_hit", None),
                cache_satisfied=bool(getattr(need, "cache_satisfied", False)),
                budget_skipped=bool(getattr(need, "budget_skipped", False)),
            )
        )
        self._update_thresholds(state)

    def record_interaction(
        self,
        group_id: str,
        interaction_type: str,
        *,
        result_type: str = "entity",
        memory_id: str | None = None,
        timestamp: float | None = None,
    ) -> None:
        """Record a recall interaction outcome."""
        state = self._get_state(group_id)
        occurred_at = time.time() if timestamp is None else timestamp
        state.interactions.append(
            _InteractionSample(
                interaction_type=interaction_type,
                result_type=result_type,
                memory_id=memory_id,
                timestamp=occurred_at,
            )
        )
        if memory_id:
            self._record_memory_feedback(
                state,
                memory_id=memory_id,
                result_type=result_type,
                interaction_type=interaction_type,
                timestamp=occurred_at,
            )
        self._update_thresholds(state)

    def memory_feedback_summary(
        self,
        group_id: str,
        memory_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Return compact per-memory feedback summaries for trust surfaces."""
        state = self._get_state(group_id)
        summaries: dict[str, dict[str, Any]] = {}
        for memory_id in memory_ids:
            feedback = state.feedback_by_memory.get(memory_id)
            if feedback is None:
                continue
            summaries[memory_id] = {
                "memory_id": feedback.memory_id,
                "result_type": feedback.result_type,
                "confirmed_count": feedback.confirmed_count,
                "corrected_count": feedback.corrected_count,
                "dismissed_count": feedback.dismissed_count,
                "used_count": feedback.used_count,
                "surfaced_count": feedback.surfaced_count,
                "selected_count": feedback.selected_count,
                "last_confirmed_at": _iso_z(feedback.last_confirmed_at),
                "last_corrected_at": _iso_z(feedback.last_corrected_at),
                "last_dismissed_at": _iso_z(feedback.last_dismissed_at),
                "last_used_at": _iso_z(feedback.last_used_at),
                "last_interaction_at": _iso_z(feedback.last_interaction_at),
            }
        return summaries

    def snapshot(self, group_id: str) -> dict:
        """Return rolling recall-need metrics for a group."""
        state = self._get_state(group_id)
        analyses = list(state.analyses)
        interactions = list(state.interactions)

        total_analyses = len(analyses)
        trigger_count = sum(1 for sample in analyses if sample.should_recall)
        family_counts = Counter(
            sample.trigger_family or "unknown" for sample in analyses if sample.should_recall
        )
        interaction_counts = Counter(sample.interaction_type for sample in interactions)
        analyzer_latencies = [
            sample.analyzer_latency_ms for sample in analyses if sample.analyzer_latency_ms > 0
        ]
        probe_latencies = [
            sample.probe_latency_ms for sample in analyses if sample.probe_latency_ms > 0
        ]
        probe_triggered = sum(1 for sample in analyses if sample.probe_triggered)
        graph_lift_count = sum(1 for sample in analyses if sample.decision_path == "graph_lift")
        graph_override_count = sum(1 for sample in analyses if sample.graph_override_used)
        cache_hit_count = sum(1 for sample in analyses if sample.cache_hit is True)
        cache_miss_count = sum(1 for sample in analyses if sample.cache_hit is False)
        used_count = interaction_counts["used"] + interaction_counts["confirmed"]
        surfaced_count = interaction_counts["surfaced"] + interaction_counts["selected"]
        dismissed_count = interaction_counts["dismissed"]

        surfaced_to_used = None
        if used_count > 0:
            surfaced_to_used = round(surfaced_count / used_count, 4)

        return {
            "total_analyses": total_analyses,
            "trigger_count": trigger_count,
            "used_count": used_count,
            "dismissed_count": dismissed_count,
            "surfaced_count": surfaced_count,
            "selected_count": interaction_counts["selected"],
            "confirmed_count": interaction_counts["confirmed"],
            "corrected_count": interaction_counts["corrected"],
            "surfaced_to_used_ratio": surfaced_to_used,
            "false_recall_rate": (
                round(dismissed_count / trigger_count, 4) if trigger_count else 0.0
            ),
            "graph_lift_rate": round(graph_lift_count / trigger_count, 4) if trigger_count else 0.0,
            "probe_trigger_rate": (
                round(probe_triggered / total_analyses, 4) if total_analyses else 0.0
            ),
            "graph_override_count": graph_override_count,
            "family_contributions": dict(family_counts),
            "mode_requested_counts": _counter_dict(
                sample.mode_requested for sample in analyses if sample.mode_requested
            ),
            "mode_executed_counts": _counter_dict(
                sample.mode_executed for sample in analyses if sample.mode_executed
            ),
            "skip_reason_counts": _counter_dict(
                sample.skip_reason for sample in analyses if sample.skip_reason
            ),
            "budget_profile_counts": _counter_dict(
                sample.budget_profile for sample in analyses if sample.budget_profile
            ),
            "cache_hit_count": cache_hit_count,
            "cache_miss_count": cache_miss_count,
            "cache_satisfied_count": sum(1 for sample in analyses if sample.cache_satisfied),
            "budget_skipped_count": sum(1 for sample in analyses if sample.budget_skipped),
            "thresholds": state.thresholds.to_dict(),
            "adaptive_thresholds_enabled": bool(
                getattr(self._cfg, "recall_need_adaptive_thresholds_enabled", False)
            ),
            "analyzer_latency_ms": self._latency_summary(analyzer_latencies),
            "probe_latency_ms": self._latency_summary(probe_latencies),
        }

    def _get_state(self, group_id: str) -> _GroupState:
        state = self._states.get(group_id)
        window = max(10, int(getattr(self._cfg, "recall_need_threshold_window", 100)))
        if state is not None:
            if state.analyses.maxlen != window:
                state.analyses = deque(state.analyses, maxlen=window)
            if state.interactions.maxlen != window:
                state.interactions = deque(state.interactions, maxlen=window)
            return state
        state = _GroupState(
            analyses=deque(maxlen=window),
            interactions=deque(maxlen=window),
        )
        self._states[group_id] = state
        return state

    def _record_memory_feedback(
        self,
        state: _GroupState,
        *,
        memory_id: str,
        result_type: str,
        interaction_type: str,
        timestamp: float,
    ) -> None:
        feedback = state.feedback_by_memory.get(memory_id)
        if feedback is None:
            feedback = _MemoryFeedbackState(memory_id=memory_id, result_type=result_type)
            state.feedback_by_memory[memory_id] = feedback
        feedback.result_type = result_type or feedback.result_type
        feedback.last_interaction_at = timestamp
        if interaction_type == "confirmed":
            feedback.confirmed_count += 1
            feedback.last_confirmed_at = timestamp
        elif interaction_type == "corrected":
            feedback.corrected_count += 1
            feedback.last_corrected_at = timestamp
        elif interaction_type == "dismissed":
            feedback.dismissed_count += 1
            feedback.last_dismissed_at = timestamp
        elif interaction_type == "used":
            feedback.used_count += 1
            feedback.last_used_at = timestamp
        elif interaction_type == "surfaced":
            feedback.surfaced_count += 1
        elif interaction_type == "selected":
            feedback.selected_count += 1
        self._prune_memory_feedback(state)

    def _prune_memory_feedback(self, state: _GroupState) -> None:
        max_entries = max(20, (state.interactions.maxlen or 100) * 4)
        if len(state.feedback_by_memory) <= max_entries:
            return
        ordered = sorted(
            state.feedback_by_memory.items(),
            key=lambda item: item[1].last_interaction_at or 0.0,
        )
        for memory_id, _feedback in ordered[: len(state.feedback_by_memory) - max_entries]:
            state.feedback_by_memory.pop(memory_id, None)

    def _update_thresholds(self, state: _GroupState) -> None:
        if not getattr(self._cfg, "recall_need_adaptive_thresholds_enabled", False):
            state.thresholds = RecallNeedThresholds()
            return

        trigger_count = sum(1 for sample in state.analyses if sample.should_recall)
        min_samples = max(1, int(getattr(self._cfg, "recall_need_adaptive_min_samples", 30)))
        if trigger_count < min_samples:
            state.thresholds = RecallNeedThresholds()
            return

        interaction_counts = Counter(sample.interaction_type for sample in state.interactions)
        used_count = interaction_counts["used"] + interaction_counts["confirmed"]
        use_rate = min(1.0, used_count / max(trigger_count, 1))
        target_use_rate = float(getattr(self._cfg, "recall_need_target_use_rate", 0.55))
        step = 0.02

        direction = 0
        if use_rate < (target_use_rate - 0.05):
            direction = 1
        elif use_rate > min(1.0, target_use_rate + 0.10):
            direction = -1

        if direction == 0:
            return

        state.thresholds = RecallNeedThresholds(
            linguistic_score=self._clamp(
                state.thresholds.linguistic_score + (direction * step),
                0.25,
                0.45,
            ),
            borderline_score=self._clamp(
                state.thresholds.borderline_score + (direction * step),
                0.10,
                0.25,
            ),
            resonance_score=self._clamp(
                state.thresholds.resonance_score + (direction * step),
                0.40,
                0.60,
            ),
        )

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, round(value, 4)))

    @staticmethod
    def _latency_summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"avg": 0.0, "p95": 0.0}
        ordered = sorted(values)
        return {
            "avg": round(sum(ordered) / len(ordered), 4),
            "p95": round(ordered[min(len(ordered) - 1, int((len(ordered) - 1) * 0.95))], 4),
        }


def _counter_dict(values) -> dict[str, int]:
    return dict(Counter(str(value) for value in values))


def _iso_z(timestamp: float | None) -> str | None:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")
