"""Tests for state-dependent retrieval (Brain Architecture 1C)."""

from __future__ import annotations

import time
from datetime import datetime

import pytest

from engram.config import ActivationConfig
from engram.retrieval.context import ConversationContext
from engram.retrieval.state import (
    CognitiveState,
    compute_domain_weights,
    compute_state_boost,
    entity_type_to_domain,
    infer_cognitive_state,
    infer_mode,
    infer_time_bucket,
    update_arousal_ema,
)

# --- Time bucket tests ---


class TestInferTimeBucket:
    def test_morning(self):
        assert infer_time_bucket(datetime(2026, 1, 1, 8, 0)) == "morning"

    def test_afternoon(self):
        assert infer_time_bucket(datetime(2026, 1, 1, 14, 0)) == "afternoon"

    def test_evening(self):
        assert infer_time_bucket(datetime(2026, 1, 1, 19, 0)) == "evening"

    def test_night(self):
        assert infer_time_bucket(datetime(2026, 1, 1, 23, 0)) == "night"

    def test_early_morning_is_night(self):
        assert infer_time_bucket(datetime(2026, 1, 1, 3, 0)) == "night"

    def test_defaults_to_now(self):
        result = infer_time_bucket()
        assert result in ("morning", "afternoon", "evening", "night")


# --- Mode inference tests ---


class TestInferMode:
    def test_task_mode(self):
        assert infer_mode("how do I fix this error?") == "task"

    def test_exploratory_mode(self):
        assert infer_mode("tell me about machine learning") == "exploratory"

    def test_reflective_mode(self):
        assert infer_mode("I feel grateful for what happened") == "reflective"

    def test_neutral_mode(self):
        assert infer_mode("hello") == "neutral"

    def test_empty_string(self):
        assert infer_mode("") == "neutral"


# --- Domain weights tests ---


class TestComputeDomainWeights:
    def test_single_domain(self):
        types = ["Person", "Person", "Event"]
        groups = {"personal": ["Person", "Event"], "technical": ["Software"]}
        weights = compute_domain_weights(types, groups)
        assert weights["personal"] == pytest.approx(1.0)

    def test_mixed_domains(self):
        types = ["Person", "Software", "Person"]
        groups = {"personal": ["Person"], "technical": ["Software"]}
        weights = compute_domain_weights(types, groups)
        assert weights["personal"] == pytest.approx(2 / 3)
        assert weights["technical"] == pytest.approx(1 / 3)

    def test_empty_types(self):
        assert compute_domain_weights([], {"personal": ["Person"]}) == {}

    def test_unknown_type_falls_to_knowledge(self):
        types = ["Unknown"]
        groups = {"personal": ["Person"]}
        weights = compute_domain_weights(types, groups)
        assert "knowledge" in weights


# --- Arousal EMA tests ---


class TestUpdateArousalEMA:
    def test_ema_update(self):
        result = update_arousal_ema(0.3, 0.8, alpha=0.3)
        # 0.3 * 0.8 + 0.7 * 0.3 = 0.24 + 0.21 = 0.45
        assert result == pytest.approx(0.45)

    def test_ema_with_zero_alpha(self):
        result = update_arousal_ema(0.5, 0.9, alpha=0.0)
        assert result == pytest.approx(0.5)

    def test_ema_with_full_alpha(self):
        result = update_arousal_ema(0.5, 0.9, alpha=1.0)
        assert result == pytest.approx(0.9)


# --- State boost tests ---


class TestComputeStateBoost:
    def test_domain_match_boost(self):
        state = CognitiveState(
            arousal_level=0.5,
            domain_weights={"personal": 0.8, "technical": 0.2},
        )
        boost = compute_state_boost(
            entity_domain="personal",
            entity_arousal=0.5,
            state=state,
            domain_weight=0.06,
            arousal_match_weight=0.04,
        )
        # domain: 0.06 * 0.8 = 0.048
        # arousal: 0.04 * (1.0 - 0.0) = 0.04
        assert boost == pytest.approx(0.088)

    def test_arousal_mismatch_reduces_boost(self):
        state = CognitiveState(arousal_level=0.9)
        boost = compute_state_boost(
            entity_domain=None,
            entity_arousal=0.1,
            state=state,
            domain_weight=0.06,
            arousal_match_weight=0.04,
        )
        # domain: 0 (no domain match)
        # arousal: 0.04 * (1.0 - 0.8) = 0.008
        assert boost == pytest.approx(0.008)

    def test_perfect_arousal_match(self):
        state = CognitiveState(arousal_level=0.7)
        boost = compute_state_boost(
            entity_domain=None,
            entity_arousal=0.7,
            state=state,
            domain_weight=0.06,
            arousal_match_weight=0.04,
        )
        # arousal: 0.04 * 1.0 = 0.04
        assert boost == pytest.approx(0.04)

    def test_no_domain_weights(self):
        state = CognitiveState(arousal_level=0.5, domain_weights={})
        boost = compute_state_boost(
            entity_domain="personal",
            entity_arousal=0.5,
            state=state,
            domain_weight=0.06,
            arousal_match_weight=0.04,
        )
        # domain affinity = 0 (empty weights)
        assert boost == pytest.approx(0.04)


# --- Entity type to domain ---


class TestEntityTypeToDomain:
    def test_known_type(self):
        groups = {"personal": ["Person", "Event"], "technical": ["Software"]}
        assert entity_type_to_domain("Person", groups) == "personal"

    def test_unknown_defaults_to_knowledge(self):
        groups = {"personal": ["Person"]}
        assert entity_type_to_domain("Unknown", groups) == "knowledge"


# --- Full cognitive state inference ---


class TestInferCognitiveState:
    def test_infer_state(self):
        state = infer_cognitive_state(
            query="how do I fix this bug?",
            recent_entity_types=["Software", "Software", "Person"],
            domain_groups={
                "personal": ["Person"],
                "technical": ["Software"],
            },
            current_arousal=0.4,
            session_start=time.time() - 600,
        )
        assert state.mode == "task"
        assert state.arousal_level == 0.4
        assert "technical" in state.domain_weights


# --- ConversationContext integration ---


class TestConversationContextState:
    def test_arousal_tracking(self):
        ctx = ConversationContext()
        assert ctx.arousal_level == 0.3
        ctx.update_arousal(0.8, alpha=0.3)
        assert ctx.arousal_level == pytest.approx(0.45)

    def test_entity_type_tracking(self):
        ctx = ConversationContext()
        ctx.track_entity_type("Person")
        ctx.track_entity_type("Software")
        assert ctx.recent_entity_types == ["Person", "Software"]

    def test_entity_type_capped(self):
        ctx = ConversationContext()
        for i in range(120):
            ctx.track_entity_type(f"Type{i}")
        assert len(ctx.recent_entity_types) == 100

    def test_clear_resets_state(self):
        ctx = ConversationContext()
        ctx.update_arousal(0.9)
        ctx.track_entity_type("Person")
        ctx.clear()
        assert ctx.arousal_level == 0.3
        assert ctx.recent_entity_types == []


# --- Config tests ---


class TestStateConfig:
    def test_config_fields_exist(self):
        cfg = ActivationConfig()
        assert cfg.state_dependent_retrieval_enabled is False  # default off
        assert cfg.state_domain_weight == 0.06
        assert cfg.state_arousal_match_weight == 0.04
        assert cfg.state_arousal_ema_alpha == 0.3

    def test_config_enabled_in_standard_profile(self):
        cfg = ActivationConfig(consolidation_profile="standard")
        assert cfg.state_dependent_retrieval_enabled is True


# --- Scorer integration ---


class TestScorerStateBoost:
    def test_state_boost_in_scorer(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        candidates = [("ent1", 0.8)]
        # Pre-computed state biases (as pipeline would compute)
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
        assert len(results) == 1
        assert results[0].state_boost == pytest.approx(0.05)

    def test_no_state_boost_when_no_biases(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        candidates = [("ent1", 0.8)]

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            state_biases=None,
        )
        assert results[0].state_boost == 0.0

    def test_thompson_scorer_state_boost(self):
        from engram.retrieval.scorer import score_candidates_thompson

        cfg = ActivationConfig(state_dependent_retrieval_enabled=True)
        candidates = [("ent1", 0.8)]
        state_biases = {"ent1": 0.07}

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
        assert results[0].state_boost == pytest.approx(0.07)
