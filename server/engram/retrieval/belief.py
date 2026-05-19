"""Bayesian Belief Map implementation for Project Synapse."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig

logger = logging.getLogger(__name__)


@dataclass
class BeliefMap:
    """A probabilistic 'degree of belief' for a retrieved memory."""

    composite_confidence: float
    evidence_density: float
    temporal_stability: float
    frequency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_confidence": self.composite_confidence,
            "evidence_density": self.evidence_density,
            "temporal_stability": self.temporal_stability,
            "frequency": self.frequency,
        }


class BeliefMapScorer:
    """Calculates belief maps based on frequency, recency, and temporal consistency."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    def calculate_belief(
        self,
        entity_data: dict[str, Any],
        relevance: float,
        activation: float,
    ) -> BeliefMap:
        """Compute the belief map for a given entity result."""
        # Frequency of corroboration
        freq = entity_data.get("evidence_count", 1)
        if isinstance(freq, str):
            try:
                freq = int(freq)
            except ValueError:
                freq = 1

        # Evidence Density: 0.1 to 1.0 based on corroboration count
        density = min(freq / 10.0, 1.0)

        # Temporal Stability:
        # Check for contradictory relationships or 'moved from' signals.
        stability = 1.0
        relationships = entity_data.get("relationships", [])
        negated_count = sum(1 for r in relationships if r.get("polarity") == "negative")
        if negated_count > 0:
            stability *= max(0.1, 1.0 - (negated_count * 0.3))

        if density > 0 and activation > 0:
            # If it's highly active but poorly corroborated, it's less stable
            if activation > 0.8 and density < 0.3:
                stability *= 0.6
            # If it's old (low activation) but highly corroborated, it's very stable
            elif activation < 0.2 and density > 0.7:
                stability *= 1.2  # Bonus for stable, long-term facts

        stability = min(1.0, stability)

        # Composite Bayesian Confidence
        # We weight relevance (how well it matches the query) against
        # cognitive factors (activation and density).
        composite = (relevance * 0.5) + (activation * 0.25) + (density * 0.25)

        # Apply stability penalty
        composite *= stability

        return BeliefMap(
            composite_confidence=round(composite, 4),
            evidence_density=round(density, 4),
            temporal_stability=round(stability, 4),
            frequency=freq,
        )
