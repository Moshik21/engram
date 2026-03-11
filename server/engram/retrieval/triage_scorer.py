"""Multi-signal triage scorer — LLM-quality episode scoring with zero API calls.

Replaces `_llm_judge_score()` with 8 deterministic signals that directly measure
what the LLM actually evaluates: entity extractability, novelty, emotional salience,
and knowledge-gap potential.

Signals:
  1. Embedding surprise     (0.25) — cosine distance from EMA corpus centroid
  2. Structural extractability (0.20) — regex: names, relationships, dates
  3. Entity candidate count  (0.15) — FTS5 probe against entity index
  4. Knowledge gap           (0.10) — new candidates NOT in graph
  5. Yield prediction        (0.10) — calibrated from extraction outcomes
  6. Emotional salience      (0.10) — existing compute_emotional_salience()
  7. Novelty (FTS5)          (0.05) — existing episode search
  8. Goal boost              (0.05) — existing goal priming
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engram.config import ActivationConfig

logger = logging.getLogger(__name__)

_SHARED_TRIAGE_SCORERS: dict[str, TriageScorer] = {}


# --- Regex patterns for structural extractability ---

_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
_RELATIONSHIP_VERBS = re.compile(
    r"\b(?:works?\s+(?:at|for|with)|lives?\s+in|moved?\s+to|"
    r"married|divorced|uses?|created?|built|leads?|manages?|"
    r"studies|researches?|knows?|loves?|hates?|likes?|dislikes?|"
    r"prefers?|avoids?|depends?\s+on|integrates?\s+with|"
    r"teaches|mentors?|collaborates?\s+with|recovering\s+from|"
    r"diagnosed\s+with|suffering\s+from|started|founded|"
    r"hired|fired|promoted|graduated|born|died)\b",
    re.IGNORECASE,
)
_DATES = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?|"
    r"(?:yesterday|today|last\s+(?:week|month|year)|next\s+(?:week|month|year)))\b",
    re.IGNORECASE,
)
_QUOTED_STRINGS = re.compile(r'"[^"]{3,}"')
_URLS = re.compile(r"https?://\S+")
_NUMBERS_WITH_CONTEXT = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|dollars?|years?|months?|hours?|"
    r"minutes?|GB|MB|TB|k|K|M)\b"
)


@dataclass
class TriageSignals:
    """Individual signal scores for a triage decision."""

    embedding_surprise: float = 0.0
    structural_extractability: float = 0.0
    entity_candidate_count: float = 0.0
    knowledge_gap: float = 0.0
    yield_prediction: float = 0.0
    emotional_salience: float = 0.0
    novelty: float = 0.0
    goal_boost: float = 0.0
    composite: float = 0.0
    compute_ms: float = 0.0


@dataclass
class EmbeddingSurpriseState:
    """EMA centroid tracker for embedding surprise detection."""

    centroid: np.ndarray | None = None
    count: int = 0
    alpha: float = 0.05  # EMA decay — slow adaptation
    # Running stats for z-score normalization
    _distance_sum: float = 0.0
    _distance_sq_sum: float = 0.0

    @property
    def mean_distance(self) -> float:
        if self.count < 2:
            return 0.5
        return self._distance_sum / self.count

    @property
    def std_distance(self) -> float:
        if self.count < 3:
            return 0.2
        var = (self._distance_sq_sum / self.count) - self.mean_distance ** 2
        return float(max(var, 0.0) ** 0.5)

    def update(self, embedding: np.ndarray) -> float:
        """Update centroid with new embedding, return raw cosine distance."""
        if self.centroid is None:
            self.centroid = embedding.copy()
            self.count = 1
            return 0.5  # First episode — neutral score

        # Cosine distance from centroid
        norm_e = np.linalg.norm(embedding)
        norm_c = np.linalg.norm(self.centroid)
        if norm_e == 0 or norm_c == 0:
            return 0.5

        cosine_sim = float(np.dot(embedding, self.centroid) / (norm_e * norm_c))
        distance = 1.0 - cosine_sim

        # Update running stats
        self.count += 1
        self._distance_sum += distance
        self._distance_sq_sum += distance ** 2

        # Update centroid via EMA
        self.centroid = self.alpha * embedding + (1.0 - self.alpha) * self.centroid

        return distance

    def z_score(self, distance: float) -> float:
        """Normalize distance to z-score, then sigmoid to [0, 1]."""
        std = self.std_distance
        if std < 1e-6:
            return 0.5
        z = (distance - self.mean_distance) / std
        # Sigmoid: z=0 → 0.5, z=2 → 0.88, z=-2 → 0.12
        return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class CalibrationState:
    """Online logistic regression for yield prediction.

    Uses sufficient statistics accumulator (X^T X + X^T y) with
    exponential decay for concept drift adaptation.
    """

    n_features: int = 6  # signals fed to calibrator
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0))
    xtx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    xty: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_samples: int = 0
    decay: float = 0.995  # Exponential decay per sample
    _initialized: bool = False

    def _init_arrays(self) -> None:
        if not self._initialized:
            self.weights = np.zeros(self.n_features)
            self.xtx = np.eye(self.n_features) * 0.01  # Ridge regularization
            self.xty = np.zeros(self.n_features)
            self._initialized = True

    def predict(self, features: np.ndarray) -> float:
        """Predict extraction probability given signal features."""
        self._init_arrays()
        if self.n_samples < 30:
            return 0.5  # Cold start — return neutral
        logit = float(np.dot(self.weights, features))
        return float(1.0 / (1.0 + np.exp(-np.clip(logit, -10, 10))))

    def update(self, features: np.ndarray, extracted: bool) -> None:
        """Update sufficient statistics with new observation."""
        self._init_arrays()
        y = 1.0 if extracted else 0.0
        x = features.astype(np.float64)

        # Decay existing statistics
        self.xtx *= self.decay
        self.xty *= self.decay

        # Accumulate
        self.xtx += np.outer(x, x)
        self.xty += x * y
        self.n_samples += 1

        # Solve for weights (ridge regression)
        try:
            self.weights = np.linalg.solve(self.xtx, self.xty).astype(np.float32)
        except np.linalg.LinAlgError:
            pass  # Keep current weights on singular matrix

    @property
    def is_mature(self) -> bool:
        return self.n_samples >= 50

    @property
    def blend_factor(self) -> float:
        """How much to trust calibrated prediction vs default (0=default, 1=calibrated)."""
        if self.n_samples < 30:
            return 0.0
        if self.n_samples >= 200:
            return 1.0
        return (self.n_samples - 30) / 170.0


class TriageScorer:
    """Multi-signal episode scorer. Zero LLM calls, <5ms per episode."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg
        self._surprise_state = EmbeddingSurpriseState()
        self._calibration = CalibrationState()

    async def score(
        self,
        content: str,
        search_index: Any = None,
        graph_store: Any = None,
        activation_store: Any = None,
        group_id: str = "default",
        embedding: list[float] | None = None,
        goals: list | None = None,
    ) -> TriageSignals:
        """Compute all signals and weighted composite for an episode."""
        t0 = time.perf_counter()
        cfg = self._cfg

        if not content:
            return TriageSignals()

        # 1. Embedding surprise (0.25)
        surprise_score = 0.0
        if embedding is not None:
            emb_array = np.asarray(embedding, dtype=np.float32)
            raw_distance = self._surprise_state.update(emb_array)
            surprise_score = self._surprise_state.z_score(raw_distance)

        # 2. Structural extractability (0.20)
        struct_score = _compute_structural_extractability(content)

        # 3. Entity candidate count (0.15) — FTS5 probe
        candidate_count = 0
        candidate_ids: set[str] = set()
        if graph_store and hasattr(graph_store, "find_entity_candidates"):
            candidate_count, candidate_ids = await _probe_entity_candidates(
                content, graph_store, group_id,
            )

        entity_candidate_score = min(candidate_count / 8.0, 1.0)

        # 4. Knowledge gap (0.10) — candidates NOT in existing graph
        gap_score = 0.0
        if candidate_count > 0:
            # Names found in text that DON'T match existing entities = new knowledge
            names_in_text = set(_PROPER_NAMES.findall(content))
            new_names = len(names_in_text) - len(candidate_ids)
            gap_score = min(max(new_names, 0) / 5.0, 1.0)
        elif len(_PROPER_NAMES.findall(content)) > 0:
            # Names found but no graph store — assume all are new
            gap_score = min(len(_PROPER_NAMES.findall(content)) / 5.0, 1.0)

        # 5. Yield prediction (0.10) — calibrated from past outcomes
        calib_features = np.array([
            surprise_score,
            struct_score,
            entity_candidate_score,
            gap_score,
            0.0,  # placeholder for emotional
            0.0,  # placeholder for novelty
        ], dtype=np.float32)
        yield_score = self._calibration.predict(calib_features)
        # Blend with default based on maturity
        blend = self._calibration.blend_factor
        yield_score = blend * yield_score + (1.0 - blend) * 0.5

        # 6. Emotional salience (0.10)
        emotional_score = 0.0
        if cfg.emotional_salience_enabled:
            from engram.extraction.salience import compute_emotional_salience
            salience = compute_emotional_salience(content)
            emotional_score = salience.composite

        # Update calibration features with emotional score
        calib_features[4] = emotional_score

        # 7. Novelty via FTS5 (0.05)
        novelty_score = 0.5  # Default
        if search_index and hasattr(search_index, "search_episodes"):
            try:
                query = content[:200].strip()
                if query:
                    matches = await search_index.search_episodes(
                        query, group_id=group_id, limit=3,
                    )
                    if matches:
                        top_score = matches[0][1]
                        similarity = min(top_score / 10.0, 1.0)
                        novelty_score = 1.0 - similarity
                    else:
                        novelty_score = 1.0
            except Exception:
                novelty_score = 0.5

        calib_features[5] = novelty_score

        # 8. Goal boost (0.05)
        goal_score = 0.0
        if goals and cfg.goal_priming_enabled:
            from engram.retrieval.goals import compute_goal_triage_boost
            goal_score = compute_goal_triage_boost(content, goals, cfg)

        # --- Weighted composite ---
        w = cfg.triage_scorer_weights
        composite = (
            w.get("embedding_surprise", 0.25) * surprise_score
            + w.get("structural_extractability", 0.20) * struct_score
            + w.get("entity_candidate_count", 0.15) * entity_candidate_score
            + w.get("knowledge_gap", 0.10) * gap_score
            + w.get("yield_prediction", 0.10) * yield_score
            + w.get("emotional_salience", 0.10) * emotional_score
            + w.get("novelty", 0.05) * novelty_score
            + w.get("goal_boost", 0.05) * goal_score
        )

        # Personal floor: guarantee extraction for emotionally rich content
        if (cfg.emotional_salience_enabled
                and emotional_score >= cfg.triage_personal_floor_threshold):
            composite = max(composite, cfg.triage_personal_floor)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return TriageSignals(
            embedding_surprise=round(surprise_score, 4),
            structural_extractability=round(struct_score, 4),
            entity_candidate_count=round(entity_candidate_score, 4),
            knowledge_gap=round(gap_score, 4),
            yield_prediction=round(yield_score, 4),
            emotional_salience=round(emotional_score, 4),
            novelty=round(novelty_score, 4),
            goal_boost=round(goal_score, 4),
            composite=round(composite, 4),
            compute_ms=round(elapsed_ms, 2),
        )

    def record_outcome(self, signals: TriageSignals, extracted_entities: int) -> None:
        """Feed extraction outcome back to calibrator for self-improvement."""
        features = np.array([
            signals.embedding_surprise,
            signals.structural_extractability,
            signals.entity_candidate_count,
            signals.knowledge_gap,
            signals.emotional_salience,
            signals.novelty,
        ], dtype=np.float32)
        self._calibration.update(features, extracted_entities > 0)

    @property
    def calibration_maturity(self) -> str:
        if self._calibration.n_samples < 30:
            return "cold_start"
        if self._calibration.n_samples < 200:
            return "blending"
        return "mature"


def get_shared_triage_scorer(cfg: ActivationConfig) -> TriageScorer:
    """Reuse the same scorer state across worker and consolidation in-process."""
    key_payload = {
        "weights": cfg.triage_scorer_weights,
        "emotional_salience_enabled": cfg.emotional_salience_enabled,
        "goal_priming_enabled": cfg.goal_priming_enabled,
        "triage_personal_floor": cfg.triage_personal_floor,
        "triage_personal_floor_threshold": cfg.triage_personal_floor_threshold,
        "triage_personal_boost_enabled": cfg.triage_personal_boost_enabled,
        "triage_personal_boost": cfg.triage_personal_boost,
        "triage_personal_min_matches": cfg.triage_personal_min_matches,
    }
    key = hashlib.sha1(
        json.dumps(key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    scorer = _SHARED_TRIAGE_SCORERS.get(key)
    if scorer is None:
        scorer = TriageScorer(cfg)
        _SHARED_TRIAGE_SCORERS[key] = scorer
    return scorer


def _compute_structural_extractability(content: str) -> float:
    """Score content for structural signals that indicate extractable entities/relationships."""
    if not content:
        return 0.0

    proper_names = len(_PROPER_NAMES.findall(content))
    rel_verbs = len(_RELATIONSHIP_VERBS.findall(content))
    dates = len(_DATES.findall(content))
    quoted = len(_QUOTED_STRINGS.findall(content))
    urls = len(_URLS.findall(content))
    numbers = len(_NUMBERS_WITH_CONTEXT.findall(content))

    # Weighted combination normalized to [0, 1]
    raw = (
        proper_names * 0.25  # Named entities
        + rel_verbs * 0.30  # Relationship indicators
        + dates * 0.15  # Temporal anchors
        + quoted * 0.10  # Specific references
        + urls * 0.05  # External references
        + numbers * 0.15  # Quantitative context
    )
    return min(raw / 3.0, 1.0)


async def _probe_entity_candidates(
    content: str,
    graph_store: Any,
    group_id: str,
) -> tuple[int, set[str]]:
    """Use FTS5 to probe how many named entities in content already exist in graph.

    Returns (total_names_found, set_of_matching_entity_ids).
    """
    names = _PROPER_NAMES.findall(content)
    if not names:
        return 0, set()

    # Deduplicate and take top 10 unique names
    unique_names = list(dict.fromkeys(names))[:10]
    matched_ids: set[str] = set()

    for name in unique_names:
        try:
            candidates = await graph_store.find_entity_candidates(
                name, group_id=group_id, limit=2,
            )
            for c in candidates:
                entity_id = c[0] if isinstance(c, (list, tuple)) else getattr(c, "id", None)
                if entity_id:
                    matched_ids.add(entity_id)
        except Exception:
            pass

    return len(unique_names), matched_ids
