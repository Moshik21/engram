"""Adaptive commit policy for evidence-based extraction."""

from __future__ import annotations

from dataclasses import dataclass

from engram.extraction.evidence import CommitDecision, EvidenceBundle, EvidenceCandidate

# Signals that warrant cold-start threshold relaxation
_HIGH_CONFIDENCE_SIGNALS = frozenset(
    {
        "identity_pattern",
        "name_declaration",
        "self_introduction",
        "family_declaration",
        "workplace_declaration",
        "residence_declaration",
        "technical_token",
    }
)


@dataclass
class CommitThresholds:
    """Per-fact-class confidence thresholds for committing evidence."""

    entity: float = 0.70
    relationship: float = 0.75
    attribute: float = 0.65
    temporal: float = 0.60

    def for_class(self, fact_class: str) -> float:
        return getattr(self, fact_class, 0.70)


class AdaptiveCommitPolicy:
    """Decides which evidence to commit, defer, or reject.

    Adapts thresholds based on graph density:
    - Cold start (<50 entities): lower thresholds by 0.15, but only for high-confidence signals
    - Dense graph (>500 entities): raise thresholds by 0.05
    """

    def __init__(
        self,
        thresholds: CommitThresholds | None = None,
        *,
        adaptive: bool = True,
        defer_band: float = 0.15,
    ) -> None:
        self._base = thresholds or CommitThresholds()
        self._adaptive = adaptive
        self._defer_band = defer_band

    def evaluate(
        self,
        bundle: EvidenceBundle,
        entity_count: int = 0,
    ) -> list[CommitDecision]:
        """Evaluate each candidate in the bundle and return commit decisions."""
        decisions: list[CommitDecision] = []
        for candidate in bundle.candidates:
            threshold = self._effective_threshold(
                candidate.fact_class,
                entity_count,
                signals=candidate.corroborating_signals,
            )
            decision = self._decide(candidate, threshold)
            decisions.append(decision)
        return decisions

    def _effective_threshold(
        self,
        fact_class: str,
        entity_count: int,
        signals: list[str] | None = None,
    ) -> float:
        """Compute adaptive threshold based on graph density and signal quality."""
        base = self._base.for_class(fact_class)
        if not self._adaptive:
            return base
        if entity_count < 50:
            # Only relax for high-confidence signals (identity, tech tokens)
            # Bare proper_name candidates get no cold-start relaxation
            if signals and any(s in _HIGH_CONFIDENCE_SIGNALS for s in signals):
                return max(0.0, base - 0.15)
            return base
        if entity_count > 500:
            return min(1.0, base + 0.05)
        return base

    def _decide(
        self,
        candidate: EvidenceCandidate,
        threshold: float,
    ) -> CommitDecision:
        """Decide commit/defer/reject for a single candidate."""
        conf = candidate.confidence

        if conf >= threshold:
            return CommitDecision(
                evidence_id=candidate.evidence_id,
                action="commit",
                reason="above_threshold",
                effective_confidence=conf,
            )

        # Defer if within the defer band below threshold
        if conf >= threshold - self._defer_band:
            return CommitDecision(
                evidence_id=candidate.evidence_id,
                action="defer",
                reason="borderline",
                effective_confidence=conf,
            )

        # Reject if too far below threshold
        return CommitDecision(
            evidence_id=candidate.evidence_id,
            action="reject",
            reason="below_threshold",
            effective_confidence=conf,
        )
