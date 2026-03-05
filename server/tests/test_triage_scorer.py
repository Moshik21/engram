"""Tests for multi-signal triage scorer."""

from __future__ import annotations

import numpy as np
import pytest

from engram.config import ActivationConfig
from engram.retrieval.triage_scorer import (
    CalibrationState,
    EmbeddingSurpriseState,
    TriageScorer,
    TriageSignals,
    _compute_structural_extractability,
)


# --- Structural extractability ---


def test_structural_empty():
    assert _compute_structural_extractability("") == 0.0


def test_structural_rich_content():
    content = (
        "Alice works at Google in San Francisco since January 2024. "
        "Bob married Charlie and moved to Berlin."
    )
    score = _compute_structural_extractability(content)
    assert score > 0.3, f"Rich content should score >0.3, got {score}"


def test_structural_no_entities():
    content = "the quick brown fox jumps over the lazy dog"
    score = _compute_structural_extractability(content)
    assert score < 0.1, f"No entities should score <0.1, got {score}"


def test_structural_dates_boost():
    content = "Meeting scheduled for January 15, 2024 at noon"
    score_with_date = _compute_structural_extractability(content)
    score_without = _compute_structural_extractability("Meeting scheduled at noon")
    assert score_with_date > score_without


def test_structural_relationship_verbs():
    content = "Alice works at Anthropic and collaborates with Bob"
    score = _compute_structural_extractability(content)
    assert score > 0.2, f"Relationship verbs should boost score, got {score}"


# --- Embedding surprise ---


def test_surprise_first_episode_neutral():
    state = EmbeddingSurpriseState()
    emb = np.random.randn(64).astype(np.float32)
    distance = state.update(emb)
    assert distance == 0.5, "First episode should return neutral 0.5"


def test_surprise_similar_episodes_low():
    state = EmbeddingSurpriseState()
    base = np.random.randn(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Feed several similar episodes to build centroid
    for _ in range(20):
        noise = np.random.randn(64).astype(np.float32) * 0.01
        state.update(base + noise)

    # A very similar episode should have low surprise
    dist = state.update(base)
    z = state.z_score(dist)
    assert z < 0.6, f"Similar episode should have low z-score, got {z}"


def test_surprise_novel_episode_high():
    state = EmbeddingSurpriseState()
    base = np.random.randn(64).astype(np.float32)
    base /= np.linalg.norm(base)

    # Feed several similar episodes
    for _ in range(20):
        noise = np.random.randn(64).astype(np.float32) * 0.01
        state.update(base + noise)

    # A very different episode should have high surprise
    novel = -base  # Opposite direction
    dist = state.update(novel)
    z = state.z_score(dist)
    assert z > 0.6, f"Novel episode should have high z-score, got {z}"


def test_surprise_centroid_tracks_ema():
    state = EmbeddingSurpriseState()
    v1 = np.ones(64, dtype=np.float32)
    v2 = -np.ones(64, dtype=np.float32)

    state.update(v1)
    # Centroid should be v1
    assert np.allclose(state.centroid, v1)

    # After updating with v2, centroid should shift toward v2
    state.update(v2)
    # EMA: centroid = 0.05 * v2 + 0.95 * v1
    expected = 0.05 * v2 + 0.95 * v1
    assert np.allclose(state.centroid, expected, atol=1e-5)


# --- Calibration state ---


def test_calibration_cold_start():
    cal = CalibrationState()
    features = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    pred = cal.predict(features)
    assert pred == 0.5, "Cold start should predict 0.5"
    assert not cal.is_mature
    assert cal.blend_factor == 0.0


def test_calibration_learns():
    cal = CalibrationState()

    # Feed positive examples (high features → extracted)
    for _ in range(60):
        features = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32)
        cal.update(features, extracted=True)

    # Feed negative examples (low features → not extracted)
    for _ in range(60):
        features = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        cal.update(features, extracted=False)

    assert cal.is_mature
    assert cal.blend_factor > 0.0

    # High-signal features should predict higher extraction probability
    high_pred = cal.predict(np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3], dtype=np.float32))
    low_pred = cal.predict(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32))
    assert high_pred > low_pred, f"High {high_pred} should > Low {low_pred}"


def test_calibration_blend_factor_ramp():
    cal = CalibrationState()
    features = np.array([0.5] * 6, dtype=np.float32)

    # Before 30 samples: 0.0
    for _ in range(29):
        cal.update(features, True)
    assert cal.blend_factor == 0.0

    # At 31+: starts blending
    cal.update(features, True)
    cal.update(features, True)
    assert cal.blend_factor > 0.0

    # At 200: fully calibrated
    for _ in range(170):
        cal.update(features, True)
    assert cal.blend_factor == 1.0


# --- TriageScorer (integration) ---


@pytest.mark.asyncio
async def test_scorer_empty_content():
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)
    signals = await scorer.score("")
    assert signals.composite == 0.0


@pytest.mark.asyncio
async def test_scorer_rich_content():
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)

    content = (
        "Alice works at Google in San Francisco since January 2024. "
        "Bob married Charlie and moved to Berlin. "
        "She was extremely excited about the promotion."
    )
    signals = await scorer.score(content)

    assert signals.composite > 0.0
    assert signals.structural_extractability > 0.0
    assert signals.emotional_salience > 0.0
    assert signals.compute_ms < 100  # Should be fast


@pytest.mark.asyncio
async def test_scorer_personal_floor():
    cfg = ActivationConfig(
        emotional_salience_enabled=True,
        triage_personal_floor=0.45,
        triage_personal_floor_threshold=0.05,  # Low threshold for test
    )
    scorer = TriageScorer(cfg)

    content = (
        "My mom was diagnosed with cancer yesterday. "
        "I'm extremely worried and afraid. "
        "My dad is devastated."
    )
    signals = await scorer.score(content)
    assert signals.composite >= 0.45, (
        f"Personal content should hit floor, got {signals.composite}"
    )


@pytest.mark.asyncio
async def test_scorer_with_embedding():
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)

    embedding = list(np.random.randn(64).astype(float))
    signals = await scorer.score(
        "Alice works at Google",
        embedding=embedding,
    )
    # First episode: neutral surprise
    assert signals.embedding_surprise == 0.5


@pytest.mark.asyncio
async def test_scorer_record_outcome():
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)

    signals = await scorer.score("Alice works at Google in San Francisco")
    scorer.record_outcome(signals, extracted_entities=3)

    assert scorer._calibration.n_samples == 1


def test_scorer_calibration_maturity():
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)
    assert scorer.calibration_maturity == "cold_start"

    # Feed enough samples
    features = np.array([0.5] * 6, dtype=np.float32)
    for _ in range(50):
        scorer._calibration.update(features, True)
    assert scorer.calibration_maturity == "blending"

    for _ in range(150):
        scorer._calibration.update(features, True)
    assert scorer.calibration_maturity == "mature"


@pytest.mark.asyncio
async def test_scorer_signals_dataclass():
    """TriageSignals fields are properly rounded."""
    cfg = ActivationConfig()
    scorer = TriageScorer(cfg)
    signals = await scorer.score("Alice and Bob went to Google headquarters in 2024")

    for field_name in [
        "embedding_surprise", "structural_extractability",
        "entity_candidate_count", "knowledge_gap",
        "yield_prediction", "emotional_salience",
        "novelty", "goal_boost",
    ]:
        val = getattr(signals, field_name)
        assert isinstance(val, float)
        # Check rounded to 4 decimal places
        assert val == round(val, 4)
