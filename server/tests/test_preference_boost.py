"""Tests for preference-directed boost in retrieval scorer."""

from __future__ import annotations

import time

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.retrieval.scorer import ScoredResult, score_candidates, score_candidates_thompson

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candidates():
    return [("ent_a", 0.9), ("ent_b", 0.8)]


def _make_activation_states():
    now = time.time()
    return {
        "ent_a": ActivationState(node_id="ent_a", access_history=[now - 100], access_count=1),
        "ent_b": ActivationState(node_id="ent_b", access_history=[now - 200], access_count=1),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreferenceBoostScoring:
    """Preference boost is added to composite score when enabled."""

    def test_preference_boost_adds_to_score(self):
        cfg = ActivationConfig(preference_directed_enabled=True, preference_retrieval_weight=0.08)
        candidates = _make_candidates()
        states = _make_activation_states()
        now = time.time()

        # Score without preference boosts
        scored_without = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            preference_boosts=None,
        )

        # Score with preference boosts
        pref_boosts = {"ent_a": 0.8, "ent_b": -0.3}
        scored_with = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            preference_boosts=pref_boosts,
        )

        # ent_a should have higher score with preference boost
        score_a_without = next(s for s in scored_without if s.node_id == "ent_a").score
        score_a_with = next(s for s in scored_with if s.node_id == "ent_a").score
        assert score_a_with > score_a_without

        # ent_b should have lower score (negative preference)
        score_b_without = next(s for s in scored_without if s.node_id == "ent_b").score
        score_b_with = next(s for s in scored_with if s.node_id == "ent_b").score
        assert score_b_with < score_b_without

    def test_no_boost_when_disabled(self):
        cfg = ActivationConfig(preference_directed_enabled=False, preference_retrieval_weight=0.08)
        candidates = _make_candidates()
        states = _make_activation_states()
        now = time.time()

        pref_boosts = {"ent_a": 0.8, "ent_b": 0.5}
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            preference_boosts=pref_boosts,
        )
        for s in scored:
            assert s.preference_boost == 0.0

    def test_no_boost_when_dict_is_none(self):
        cfg = ActivationConfig(preference_directed_enabled=True, preference_retrieval_weight=0.08)
        candidates = _make_candidates()
        states = _make_activation_states()
        now = time.time()

        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            preference_boosts=None,
        )
        for s in scored:
            assert s.preference_boost == 0.0

    def test_no_boost_when_dict_is_empty(self):
        cfg = ActivationConfig(preference_directed_enabled=True, preference_retrieval_weight=0.08)
        candidates = _make_candidates()
        states = _make_activation_states()
        now = time.time()

        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            preference_boosts={},
        )
        for s in scored:
            assert s.preference_boost == 0.0


class TestPreferenceBoostThompson:
    """Thompson sampling scorer also incorporates preference boost."""

    def test_thompson_preference_boost(self):
        cfg = ActivationConfig(
            preference_directed_enabled=True,
            preference_retrieval_weight=0.08,
            ts_enabled=True,
        )
        candidates = _make_candidates()
        states = _make_activation_states()
        now = time.time()

        pref_boosts = {"ent_a": 1.0}
        scored = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses={},
            hop_distances={},
            seed_node_ids=set(),
            activation_states=states,
            now=now,
            cfg=cfg,
            rng_seed=42,
            preference_boosts=pref_boosts,
        )
        score_a = next(s for s in scored if s.node_id == "ent_a")
        assert score_a.preference_boost > 0.0
        assert score_a.preference_boost == pytest.approx(0.08 * 1.0)


class TestScoredResultHasPreferenceField:
    """ScoredResult dataclass includes preference_boost field."""

    def test_default_preference_boost(self):
        sr = ScoredResult(
            node_id="x",
            score=1.0,
            semantic_similarity=0.5,
            activation=0.1,
            spreading=0.0,
            edge_proximity=0.0,
        )
        assert sr.preference_boost == 0.0
