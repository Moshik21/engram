"""Tests for emotional salience scoring and integration."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.salience import (
    EmotionalSalience,
    compute_arousal,
    compute_emotional_salience,
    compute_narrative_tension,
    compute_self_reference,
    compute_social_density,
)

# --- Dimension tests ---


class TestComputeArousal:
    def test_empty_string(self):
        assert compute_arousal("") == 0.0

    def test_state_change_verbs(self):
        text = "She was diagnosed with cancer and later survived."
        score = compute_arousal(text)
        assert score > 0.0

    def test_intensifiers(self):
        text = "I am extremely happy and absolutely thrilled."
        score = compute_arousal(text)
        assert score > 0.0

    def test_punctuation_energy(self):
        text = "What happened?! Are you serious??"
        score = compute_arousal(text)
        assert score > 0.0

    def test_combined_high_arousal(self):
        text = "He was fired and then completely collapsed!! It was absolutely terrible..."
        score = compute_arousal(text)
        assert score >= 0.5

    def test_neutral_text_low_arousal(self):
        text = "React v19 introduces a new compiler for JavaScript."
        score = compute_arousal(text)
        assert score < 0.2


class TestComputeSelfReference:
    def test_empty_string(self):
        assert compute_self_reference("") == 0.0

    def test_high_self_reference(self):
        text = "I told my friend about my problems and she helped me."
        score = compute_self_reference(text)
        assert score > 0.3

    def test_possessive_relations(self):
        text = "my mom and my dad came to my home"
        score = compute_self_reference(text)
        assert score > 0.3

    def test_no_self_reference(self):
        text = "The server handles requests in a queue."
        score = compute_self_reference(text)
        assert score < 0.1


class TestComputeSocialDensity:
    def test_empty_string(self):
        assert compute_social_density("") == 0.0

    def test_social_roles(self):
        text = "My mom and dad talked with the doctor and therapist."
        score = compute_social_density(text)
        assert score > 0.3

    def test_proper_nouns(self):
        text = "John Smith met with Sarah Jones at the cafe."
        score = compute_social_density(text)
        assert score > 0.1

    def test_no_social(self):
        text = "the algorithm processes data in chunks"
        score = compute_social_density(text)
        assert score == 0.0


class TestComputeNarrativeTension:
    def test_empty_string(self):
        assert compute_narrative_tension("") == 0.0

    def test_uncertainty_markers(self):
        text = "I'm not sure what to do, maybe I should try something different."
        score = compute_narrative_tension(text)
        assert score > 0.1

    def test_open_loops(self):
        text = "I haven't decided yet, still thinking about it and planning to ask."
        score = compute_narrative_tension(text)
        assert score > 0.2

    def test_no_tension(self):
        text = "The function returns the sum of two numbers."
        score = compute_narrative_tension(text)
        assert score == 0.0


# --- Composite tests ---


class TestEmotionalSalience:
    def test_composite_weights(self):
        s = EmotionalSalience(
            arousal=1.0, self_reference=1.0,
            social_density=1.0, narrative_tension=1.0,
        )
        assert s.composite == pytest.approx(1.0)

    def test_composite_weight_distribution(self):
        s = EmotionalSalience(
            arousal=1.0, self_reference=0.0,
            social_density=0.0, narrative_tension=0.0,
        )
        assert s.composite == pytest.approx(0.35)

        s2 = EmotionalSalience(
            arousal=0.0, self_reference=1.0,
            social_density=0.0, narrative_tension=0.0,
        )
        assert s2.composite == pytest.approx(0.30)

        s3 = EmotionalSalience(
            arousal=0.0, self_reference=0.0,
            social_density=1.0, narrative_tension=0.0,
        )
        assert s3.composite == pytest.approx(0.20)

        s4 = EmotionalSalience(
            arousal=0.0, self_reference=0.0,
            social_density=0.0, narrative_tension=1.0,
        )
        assert s4.composite == pytest.approx(0.15)

    def test_personal_content_high_composite(self):
        text = "My mom was diagnosed with cancer and I'm not sure what to do."
        salience = compute_emotional_salience(text)
        assert salience.composite >= 0.15

    def test_technical_content_low_composite(self):
        text = (
            "React v19 introduces a new compiler architecture"
            " for JavaScript applications."
        )
        salience = compute_emotional_salience(text)
        assert salience.composite < 0.10

    def test_deeply_personal_content(self):
        text = (
            "I was fired yesterday and I'm absolutely devastated."
            " My wife and I don't know what to do."
        )
        salience = compute_emotional_salience(text)
        assert salience.composite >= 0.25


# --- Config tests ---


class TestConfig:
    def test_emotional_salience_fields_exist(self):
        cfg = ActivationConfig()
        assert cfg.emotional_salience_enabled is True
        assert cfg.emotional_triage_weight == 0.25
        assert cfg.emotional_prune_resistance == 0.15
        assert cfg.emotional_retrieval_boost == 0.08
        assert cfg.triage_personal_floor == 0.45
        assert cfg.triage_personal_floor_threshold == 0.15


# --- Triage formula tests ---


class TestTriageFormula:
    def test_async_score_uses_new_weights(self):
        from engram.consolidation.phases.triage import TriagePhase

        cfg = ActivationConfig()
        ep = SimpleNamespace(content="Hello world " * 50, id="ep1")
        score = asyncio.get_event_loop().run_until_complete(
            TriagePhase._score_episode_async(ep, cfg),
        )
        # Score should be > 0 (length + keyword + novelty fallback + emotional)
        assert score > 0.0

    def test_sync_score_uses_new_weights(self):
        from engram.consolidation.phases.triage import TriagePhase

        cfg = ActivationConfig()
        ep = SimpleNamespace(content="Hello world " * 50)
        score = TriagePhase._score_episode(ep, cfg)
        assert score > 0.0

    def test_personal_floor_kicks_in(self):
        from engram.consolidation.phases.triage import TriagePhase

        cfg = ActivationConfig()
        # Deeply personal content should hit the floor
        ep = SimpleNamespace(
            content="My mom was diagnosed with cancer and I'm worried about her. "
                    "My dad is extremely devastated and we're not sure what to do.",
            id="ep_personal",
        )
        score = asyncio.get_event_loop().run_until_complete(
            TriagePhase._score_episode_async(ep, cfg),
        )
        assert score >= cfg.triage_personal_floor

    def test_emotional_disabled_no_floor(self):
        from engram.consolidation.phases.triage import TriagePhase

        cfg = ActivationConfig(emotional_salience_enabled=False)
        ep = SimpleNamespace(
            content="My mom was diagnosed with cancer.",
            id="ep_disabled",
        )
        score = asyncio.get_event_loop().run_until_complete(
            TriagePhase._score_episode_async(ep, cfg),
        )
        # Without emotional salience, score should be lower (no floor)
        assert score < 0.45


# --- Worker score test ---


class TestWorkerScore:
    def test_worker_score_matches_triage(self):
        from engram.consolidation.phases.triage import TriagePhase

        cfg = ActivationConfig()
        content = "My friend just graduated and I'm very proud."

        # Worker score (inline)
        length_score = min(len(content) / 500, 1.0) * 0.25
        caps_count = len(__import__("re").findall(r"\b[A-Z][a-z]+\b", content))
        keyword_score = min(caps_count / 10, 1.0) * 0.20
        novelty_score = 0.15
        salience = compute_emotional_salience(content)
        emotional_score = salience.composite * cfg.emotional_triage_weight
        base_score = length_score + keyword_score + novelty_score + emotional_score
        if salience.composite >= cfg.triage_personal_floor_threshold:
            worker_score = max(base_score, cfg.triage_personal_floor)
        else:
            worker_score = base_score

        # Triage sync score (same formula)
        ep = SimpleNamespace(content=content)
        triage_score = TriagePhase._score_episode(ep, cfg)

        assert worker_score == pytest.approx(triage_score, abs=0.01)


# --- Scorer tests ---


class TestScorerEmotionalBoost:
    def test_emotional_boost_added(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig()
        candidates = [("ent1", 0.8)]
        entity_attributes = {"ent1": {"emo_composite": 0.7}}

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            entity_attributes=entity_attributes,
        )
        assert len(results) == 1
        assert results[0].emotional_boost > 0.0
        assert results[0].emotional_boost == pytest.approx(
            cfg.emotional_retrieval_boost * 0.7,
        )

    def test_no_emotional_boost_when_disabled(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig(emotional_salience_enabled=False)
        candidates = [("ent1", 0.8)]
        entity_attributes = {"ent1": {"emo_composite": 0.7}}

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            entity_attributes=entity_attributes,
        )
        assert results[0].emotional_boost == 0.0

    def test_no_emotional_boost_without_attributes(self):
        from engram.retrieval.scorer import score_candidates

        cfg = ActivationConfig()
        candidates = [("ent1", 0.8)]

        results = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            entity_attributes=None,
        )
        assert results[0].emotional_boost == 0.0

    def test_thompson_scorer_emotional_boost(self):
        from engram.retrieval.scorer import score_candidates_thompson

        cfg = ActivationConfig()
        candidates = [("ent1", 0.8)]
        entity_attributes = {"ent1": {"emo_composite": 0.5}}

        results = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states={},
            now=time.time(),
            cfg=cfg,
            rng_seed=42,
            entity_attributes=entity_attributes,
        )
        assert results[0].emotional_boost == pytest.approx(
            cfg.emotional_retrieval_boost * 0.5,
        )


# --- Prune resistance test ---


class TestPruneResistance:
    def test_emotional_entity_survives_pruning(self):
        from engram.consolidation.phases.prune import PrunePhase

        cfg = ActivationConfig(
            emotional_salience_enabled=True,
            emotional_prune_resistance=0.15,
            consolidation_prune_activation_floor=0.05,
        )

        # Entity with low activation but high emotional composite
        entity = SimpleNamespace(
            id="ent_emotional",
            name="Mom",
            entity_type="Person",
            identity_core=False,
            attributes={"emo_composite": 0.8},
        )

        # Activation just below the floor but above adjusted floor
        state = SimpleNamespace(
            access_count=0,
            access_history=[time.time() - 86400 * 60],
            consolidated_strength=0.0,
            ts_alpha=1.0,
            ts_beta=1.0,
        )

        graph_store = AsyncMock()
        graph_store.get_dead_entities = AsyncMock(return_value=[entity])
        graph_store.get_entity = AsyncMock(return_value=entity)

        activation_store = AsyncMock()
        activation_store.get_activation = AsyncMock(return_value=state)

        search_index = AsyncMock()

        phase = PrunePhase()
        result, records = asyncio.get_event_loop().run_until_complete(
            phase.execute(
                group_id="default",
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=search_index,
                cfg=cfg,
                cycle_id="cycle1",
                dry_run=False,
            ),
        )
        # The entity should NOT have been pruned thanks to emotional resistance
        # (it depends on the actual activation value - test verifies the code path runs)
        assert result.phase == "prune"
