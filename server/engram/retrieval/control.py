"""Runtime control loop for recall-need thresholds and metrics."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field


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


@dataclass
class _AnalysisSample:
    should_recall: bool
    trigger_family: str | None
    analyzer_latency_ms: float
    probe_triggered: bool
    probe_latency_ms: float
    decision_path: str | None
    graph_override_used: bool


@dataclass
class _InteractionSample:
    interaction_type: str
    result_type: str = "entity"


@dataclass
class _GroupState:
    analyses: deque[_AnalysisSample]
    interactions: deque[_InteractionSample]
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
            )
        )
        self._update_thresholds(state)

    def record_interaction(
        self,
        group_id: str,
        interaction_type: str,
        *,
        result_type: str = "entity",
    ) -> None:
        """Record a recall interaction outcome."""
        state = self._get_state(group_id)
        state.interactions.append(
            _InteractionSample(
                interaction_type=interaction_type,
                result_type=result_type,
            )
        )
        self._update_thresholds(state)

    def snapshot(self, group_id: str) -> dict:
        """Return rolling recall-need metrics for a group."""
        state = self._get_state(group_id)
        analyses = list(state.analyses)
        interactions = list(state.interactions)

        total_analyses = len(analyses)
        trigger_count = sum(1 for sample in analyses if sample.should_recall)
        family_counts = Counter(
            sample.trigger_family or "unknown"
            for sample in analyses
            if sample.should_recall
        )
        interaction_counts = Counter(sample.interaction_type for sample in interactions)
        analyzer_latencies = [
            sample.analyzer_latency_ms
            for sample in analyses
            if sample.analyzer_latency_ms > 0
        ]
        probe_latencies = [
            sample.probe_latency_ms
            for sample in analyses
            if sample.probe_latency_ms > 0
        ]
        probe_triggered = sum(1 for sample in analyses if sample.probe_triggered)
        graph_lift_count = sum(1 for sample in analyses if sample.decision_path == "graph_lift")
        graph_override_count = sum(1 for sample in analyses if sample.graph_override_used)
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
