"""Projection policy for cue routing and feedback-driven escalation."""

from __future__ import annotations

from dataclasses import dataclass

from engram.config import ActivationConfig
from engram.models.episode import Episode, EpisodeProjectionState
from engram.models.episode_cue import EpisodeCue


@dataclass
class CuePolicyDecision:
    """Initial routing decision for a freshly built cue."""

    projection_state: EpisodeProjectionState
    route_reason: str
    projection_priority: float
    policy_score: float


@dataclass
class CueFeedbackDecision:
    """Feedback-driven cue updates after recall interaction."""

    updates: dict[str, object]
    should_promote: bool
    promotion_reason: str | None = None


class ProjectionPolicy:
    """Encapsulate deterministic projection routing and lightweight learning."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    def decide_initial(
        self,
        *,
        episode: Episode,
        discourse_class: str,
        route_reason: str,
        projection_priority: float,
    ) -> CuePolicyDecision:
        """Decide initial cue state and seed its policy score."""
        original_priority = projection_priority
        priority = projection_priority
        policy_score = projection_priority

        if self._cfg.cue_policy_learning_enabled:
            source_boost, source_reason = self._source_boost(episode.source)
            discourse_boost = self._cfg.cue_policy_discourse_boosts.get(
                discourse_class,
                0.0,
            )
            priority = min(1.0, priority + source_boost + discourse_boost)
            policy_score = min(
                self._cfg.cue_policy_score_cap,
                priority + (source_boost * 0.5) + discourse_boost,
            )
            boosted_into_schedule = original_priority < 0.55 and priority >= 0.55
            if source_reason is not None and (
                route_reason == "default" or boosted_into_schedule
            ):
                route_reason = source_reason

        state = self._projection_state(
            discourse_class=discourse_class,
            route_reason=route_reason,
            priority=priority,
        )
        return CuePolicyDecision(
            projection_state=state,
            route_reason=route_reason,
            projection_priority=round(priority, 4),
            policy_score=round(policy_score, 4),
        )

    def apply_feedback(
        self,
        cue: EpisodeCue,
        *,
        interaction_type: str | None,
        score: float,
        count_hit: bool = True,
    ) -> CueFeedbackDecision:
        """Update cue counters and decide whether feedback should escalate projection."""
        event_type = interaction_type or "surfaced"
        if event_type == "confirmed":
            event_type = "used"
        elif event_type == "corrected":
            event_type = "dismissed"
        normalized_score = min(max(score, 0.0), 1.0)
        updates: dict[str, object] = {}

        if count_hit and event_type in {"surfaced", "selected", "used"}:
            updates["hit_count"] = (cue.hit_count or 0) + 1

        count_field = {
            "surfaced": "surfaced_count",
            "selected": "selected_count",
            "used": "used_count",
            "near_miss": "near_miss_count",
        }.get(event_type)
        if count_field is not None:
            updates[count_field] = getattr(cue, count_field, 0) + 1

        if not self._cfg.cue_policy_learning_enabled:
            return CueFeedbackDecision(
                updates=updates,
                should_promote=False,
                promotion_reason=None,
            )

        base_score = cue.policy_score or cue.projection_priority or cue.cue_score
        delta = self._feedback_delta(event_type, normalized_score)
        policy_score = min(
            self._cfg.cue_policy_score_cap,
            max(0.0, base_score + delta),
        )
        updates["policy_score"] = round(policy_score, 4)
        updates["projection_priority"] = round(
            min(1.0, max(cue.projection_priority, policy_score)),
            4,
        )

        should_promote = False
        promotion_reason = None
        if (
            event_type not in {"dismissed"}
            and policy_score >= self._cfg.cue_policy_schedule_threshold
        ):
            should_promote = True
            promotion_reason = f"cue_policy_{event_type}"

        return CueFeedbackDecision(
            updates=updates,
            should_promote=should_promote,
            promotion_reason=promotion_reason,
        )

    def _projection_state(
        self,
        *,
        discourse_class: str,
        route_reason: str,
        priority: float,
    ) -> EpisodeProjectionState:
        if discourse_class == "system":
            return EpisodeProjectionState.CUE_ONLY
        if route_reason in {"contradiction_hint", "identity_hint"}:
            return EpisodeProjectionState.SCHEDULED
        if priority >= 0.55:
            return EpisodeProjectionState.SCHEDULED
        return EpisodeProjectionState.CUED

    def _feedback_delta(self, event_type: str, score: float) -> float:
        if event_type == "used":
            return self._cfg.cue_policy_use_weight * max(score, 0.5)
        if event_type == "selected":
            return self._cfg.cue_policy_select_weight * max(score, 0.5)
        if event_type == "near_miss":
            return self._cfg.cue_policy_near_miss_weight * max(score, 0.3)
        if event_type == "dismissed":
            return -self._cfg.cue_policy_select_weight * max(score, 0.5)
        return self._cfg.cue_policy_surface_weight * max(score, 0.2)

    def _source_boost(self, source: str | None) -> tuple[float, str | None]:
        if not source:
            return 0.0, None
        exact = self._cfg.cue_policy_source_boosts.get(source)
        if exact is not None:
            return exact, f"source_priority:{source}"

        for prefix, boost in self._cfg.cue_policy_source_boosts.items():
            if source.startswith(prefix):
                return boost, f"source_priority:{prefix}"
        return 0.0, None
