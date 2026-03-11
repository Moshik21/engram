"""Tests for state-dependent retrieval (Brain Architecture 1C)."""

from __future__ import annotations

import time

import pytest

from engram.config import ActivationConfig
from engram.retrieval.context import ConversationContext
from engram.retrieval.state import (
    CognitiveState,
    compute_state_bias,
    get_time_bucket,
    infer_cognitive_mode,
)

# --- Mode inference ---


class TestInferCognitiveMode:
    def test_task_mode(self):
        assert infer_cognitive_mode("how do I fix this error?") == "task"

    def test_exploratory_mode(self):
        assert infer_cognitive_mode("tell me about machine learning") == "exploratory"

    def test_reflective_mode(self):
        assert infer_cognitive_mode("I feel grateful for what happened") == "reflective"

    def test_neutral_mode(self):
        assert infer_cognitive_mode("hello") == "neutral"

    def test_empty_string(self):
        assert infer_cognitive_mode("") == "neutral"


# --- Time bucket ---


class TestGetTimeBucket:
    def test_morning(self):
        assert get_time_bucket(8) == "morning"

    def test_afternoon(self):
        assert get_time_bucket(14) == "afternoon"

    def test_evening(self):
        assert get_time_bucket(19) == "evening"

    def test_night(self):
        assert get_time_bucket(23) == "night"

    def test_early_morning_is_night(self):
        assert get_time_bucket(3) == "night"

    def test_defaults_to_current(self):
        result = get_time_bucket()
        assert result in ("morning", "afternoon", "evening", "night")


# --- CognitiveState ---


class TestCognitiveState:
    def test_defaults(self):
        state = CognitiveState()
        assert state.arousal_level == 0.0
        assert state.mode == "neutral"
        assert state.domain_weights == {}
        assert state.time_bucket == "afternoon"


# --- compute_state_bias ---


class TestComputeStateBias:
    def test_domain_match_boosts(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        state = CognitiveState(
            arousal_level=0.5,
            mode="task",
        )
        bias = compute_state_bias(
            state=state,
            entity_attrs={"emo_composite": 0.5},
            entity_type="Software",
            cfg=cfg,
            domain_groups={"technical": ["Software"]},
        )
        # domain: 0.06 * 0.8 = 0.048 (task -> technical = 0.8)
        # arousal: 0.04 * (1.0 - 0) = 0.04 (perfect match)
        assert bias == pytest.approx(0.088)

    def test_reflective_personal_boost(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        state = CognitiveState(mode="reflective", arousal_level=0.7)
        bias = compute_state_bias(
            state=state,
            entity_attrs={"emo_composite": 0.7},
            entity_type="Person",
            cfg=cfg,
            domain_groups={"personal": ["Person"]},
        )
        # domain: 0.06 * 0.9 (reflective -> personal) = 0.054
        # arousal: 0.04 * (1.0 - 0) = 0.04
        assert bias == pytest.approx(0.094)

    def test_returns_zero_when_disabled(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=False)
        state = CognitiveState(mode="task", arousal_level=0.5)
        bias = compute_state_bias(
            state=state,
            entity_attrs={"emo_composite": 0.5},
            entity_type="Software",
            cfg=cfg,
            domain_groups={"technical": ["Software"]},
        )
        assert bias == 0.0

    def test_arousal_mismatch_reduces_boost(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        state = CognitiveState(arousal_level=0.9, mode="neutral")
        bias = compute_state_bias(
            state=state,
            entity_attrs={"emo_composite": 0.1},
            entity_type="Other",
            cfg=cfg,
            domain_groups={},
        )
        # No domain match, arousal diff = 0.8
        # match_score = max(0, 1.0 - 0.8 * 2.0) = max(0, -0.6) = 0.0
        assert bias == 0.0

    def test_no_emo_composite_no_arousal_boost(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        state = CognitiveState(arousal_level=0.5, mode="task")
        bias = compute_state_bias(
            state=state,
            entity_attrs={},
            entity_type="Software",
            cfg=cfg,
            domain_groups={"technical": ["Software"]},
        )
        # domain only: 0.06 * 0.8 = 0.048
        assert bias == pytest.approx(0.048)

    def test_neutral_mode_no_domain_affinity(self):
        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        state = CognitiveState(mode="neutral", arousal_level=0.5)
        bias = compute_state_bias(
            state=state,
            entity_attrs={"emo_composite": 0.5},
            entity_type="Person",
            cfg=cfg,
            domain_groups={"personal": ["Person"]},
        )
        # Neutral has empty affinity map, so default 0.3 applies
        # domain: 0.06 * 0.3 = 0.018
        # arousal: 0.04 * 1.0 = 0.04
        assert bias == pytest.approx(0.058)


# --- ConversationContext integration ---


class TestConversationContextCogState:
    def test_cognitive_state_none_before_update(self):
        ctx = ConversationContext()
        assert ctx.cognitive_state is None

    def test_update_cognitive_state(self):
        ctx = ConversationContext()
        ctx.update_cognitive_state(
            "how do I fix this bug?",
            salience_composite=0.3,
        )
        state = ctx.cognitive_state
        assert state is not None
        assert state.mode == "task"
        assert state.arousal_level > 0.0

    def test_arousal_ema_tracks(self):
        ctx = ConversationContext()
        ctx.update_cognitive_state("hello", salience_composite=1.0, alpha=0.5)
        # EMA: 0.5 * 1.0 + 0.5 * 0.0 (init was 0.3 per context init)
        # Actually init is 0.3, so: 0.5 * 1.0 + 0.5 * 0.3 = 0.65
        s1 = ctx.cognitive_state.arousal_level
        assert s1 > 0.0

        ctx.update_cognitive_state("hello", salience_composite=0.0, alpha=0.5)
        s2 = ctx.cognitive_state.arousal_level
        assert s2 < s1  # Decayed toward 0

    def test_clear_resets_cognitive_state(self):
        ctx = ConversationContext()
        ctx.update_cognitive_state("test", salience_composite=0.5)
        assert ctx.cognitive_state is not None
        ctx.clear()
        assert ctx.cognitive_state is None


# --- Config ---


class TestStateConfig:
    def test_defaults(self):
        cfg = ActivationConfig()
        assert cfg.state_dependent_retrieval_enabled is False
        assert cfg.state_domain_weight == 0.06
        assert cfg.state_arousal_match_weight == 0.04
        assert cfg.state_arousal_ema_alpha == 0.3

    def test_conservative_enables(self):
        cfg = ActivationConfig(consolidation_profile="conservative")
        assert cfg.state_dependent_retrieval_enabled is True

    def test_standard_enables(self):
        cfg = ActivationConfig(consolidation_profile="standard")
        assert cfg.state_dependent_retrieval_enabled is True


# --- Scorer integration ---


class TestScorerStateBias:
    def test_state_boost_field_exists(self):
        from engram.retrieval.scorer import ScoredResult

        sr = ScoredResult(
            node_id="e1",
            score=1.0,
            semantic_similarity=0.5,
            activation=0.3,
            spreading=0.1,
            edge_proximity=0.0,
        )
        assert sr.state_boost == 0.0

    def test_scorer_uses_state_biases(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        candidates = [("ent1", 0.8)]
        state_biases = {"ent1": 0.05}

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            state_biases=state_biases,
        )
        assert results[0].state_boost == pytest.approx(0.05)

    def test_thompson_scorer_uses_state_biases(self):
        from engram.retrieval.scorer import score_candidates_thompson

        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        candidates = [("ent1", 0.8)]
        state_biases = {"ent1": 0.03}

        results = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            rng_seed=42,
            state_biases=state_biases,
        )
        assert results[0].state_boost == pytest.approx(0.03)
