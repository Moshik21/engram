"""Tests for deterministic projection planning."""

from __future__ import annotations

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.extraction.extractor import MAX_EXTRACTION_INPUT_CHARS
from engram.extraction.planner import ProjectionPlanner
from engram.models.episode import Episode


def test_projection_planner_prefers_late_correction_span():
    correction = (
        "Correction: I actually moved to Phoenix in 2024 and no longer live in Mesa."
    )
    filler = "Earlier note: I lived in Mesa and commuted to Tempe. "
    content = (filler * 180) + correction
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        projection_planner_enabled=True,
    )
    episode = Episode(id="ep_long", content=content, group_id="default")
    cue = build_episode_cue(episode, cfg)

    plan = ProjectionPlanner(cfg).plan(episode, cue)

    assert correction not in content[:MAX_EXTRACTION_INPUT_CHARS]
    assert plan.strategy in {"focused_span", "targeted_spans"}
    assert plan.was_truncated is True
    assert plan.selected_chars <= MAX_EXTRACTION_INPUT_CHARS
    assert correction in plan.selected_text


def test_projection_planner_keeps_short_episode_intact():
    content = "Konner moved to Phoenix and is redesigning Engram extraction."
    cfg = ActivationConfig(projection_planner_enabled=True)
    episode = Episode(id="ep_short", content=content, group_id="default")

    plan = ProjectionPlanner(cfg).plan(episode)

    assert plan.strategy == "full_episode"
    assert plan.selected_text == content
    assert plan.was_truncated is False


def test_projection_planner_can_disable_targeted_projection():
    correction = (
        "Correction: I actually moved to Phoenix in 2024 and no longer live in Mesa."
    )
    filler = "Earlier note: I lived in Mesa and commuted to Tempe. "
    content = (filler * 180) + correction
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        projection_planner_enabled=True,
        targeted_projection_enabled=False,
    )
    episode = Episode(id="ep_long_disabled", content=content, group_id="default")
    cue = build_episode_cue(episode, cfg)

    plan = ProjectionPlanner(cfg).plan(episode, cue)

    assert plan.strategy == "full_episode"
    assert plan.was_truncated is True
    assert correction not in plan.selected_text


def test_projection_planner_can_disable_projector_v2():
    correction = (
        "Correction: I actually moved to Phoenix in 2024 and no longer live in Mesa."
    )
    filler = "Earlier note: I lived in Mesa and commuted to Tempe. "
    content = (filler * 180) + correction
    cfg = ActivationConfig(
        cue_layer_enabled=True,
        projection_planner_enabled=True,
        projector_v2_enabled=False,
    )
    episode = Episode(id="ep_long_v1", content=content, group_id="default")
    cue = build_episode_cue(episode, cfg)

    plan = ProjectionPlanner(cfg).plan(episode, cue)

    assert plan.strategy == "full_episode"
    assert plan.was_truncated is True
    assert correction not in plan.selected_text
