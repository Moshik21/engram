"""Tests for cue projection policy."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.extraction.policy import ProjectionPolicy
from engram.models.episode import EpisodeProjectionState
from engram.models.episode_cue import EpisodeCue


def test_projection_policy_promotes_used_feedback():
    cfg = ActivationConfig(
        cue_policy_learning_enabled=True,
        cue_policy_use_weight=0.5,
        cue_policy_schedule_threshold=0.8,
    )
    cue = EpisodeCue(
        episode_id="ep_policy",
        group_id="default",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        projection_priority=0.42,
        policy_score=0.42,
        cue_text="mentions: Phoenix",
    )

    decision = ProjectionPolicy(cfg).apply_feedback(
        cue,
        interaction_type="used",
        score=0.9,
    )

    assert decision.updates["used_count"] == 1
    assert decision.updates["hit_count"] == 1
    assert decision.updates["policy_score"] > cue.policy_score
    assert decision.should_promote is True
    assert decision.promotion_reason == "cue_policy_used"


def test_projection_policy_near_miss_does_not_increment_hit_count():
    cue = EpisodeCue(
        episode_id="ep_policy_nm",
        group_id="default",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        projection_priority=0.3,
        policy_score=0.3,
        cue_text="mentions: Engram",
    )
    cfg = ActivationConfig(cue_policy_learning_enabled=True)

    decision = ProjectionPolicy(cfg).apply_feedback(
        cue,
        interaction_type="near_miss",
        score=0.7,
    )

    assert "hit_count" not in decision.updates
    assert decision.updates["near_miss_count"] == 1
    assert decision.should_promote is False


def test_projection_policy_dismissed_feedback_does_not_promote():
    cue = EpisodeCue(
        episode_id="ep_policy_dismissed",
        group_id="default",
        projection_state=EpisodeProjectionState.CUE_ONLY,
        projection_priority=0.72,
        policy_score=0.79,
        cue_text="mentions: Engram",
    )
    cfg = ActivationConfig(
        cue_policy_learning_enabled=True,
        cue_policy_select_weight=0.4,
        cue_policy_schedule_threshold=0.8,
    )

    decision = ProjectionPolicy(cfg).apply_feedback(
        cue,
        interaction_type="dismissed",
        score=0.9,
        count_hit=False,
    )

    assert "hit_count" not in decision.updates
    assert decision.updates["policy_score"] < cue.policy_score
    assert decision.should_promote is False
