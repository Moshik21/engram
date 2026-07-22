"""Adaptive commit policy for evidence-based extraction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from engram.extraction.evidence import CommitDecision, EvidenceBundle, EvidenceCandidate
from engram.extraction.promotion import is_high_signal_entity_type
from engram.ingestion.salience import is_observation_source

logger = logging.getLogger(__name__)

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
        # span_verified / high_signal_type are handled in _decide for client
        # proposals; do not use them alone to relax cold-start thresholds for
        # unverified claims.
    }
)

# M1.4 squatter guard (P4: names are identifiers, not content). An entity
# name longer than this is a summary in disguise — the excess folds into the
# summary and the name is capped.
_MAX_ENTITY_NAME_TOKENS = 6

# Identity-class signals exempt from the observation corroboration hold —
# "my name is Konner" captured via observe must not wait for a second episode.
_IDENTITY_SIGNALS = _HIGH_CONFIDENCE_SIGNALS - {"technical_token"}


def _cap_entity_name(candidate: EvidenceCandidate) -> None:
    """Cap sentence-length entity names; fold the excess into the summary.

    The battery's squatter class: a milestone observation extracted into a
    sentence-long entity name that scored 0.99 on unrelated ranking queries.
    """
    payload = candidate.payload if isinstance(candidate.payload, dict) else {}
    name = str(payload.get("name") or "")
    tokens = name.split()
    if len(tokens) <= _MAX_ENTITY_NAME_TOKENS:
        return
    capped = " ".join(tokens[:_MAX_ENTITY_NAME_TOKENS])
    summary = str(payload.get("summary") or "")
    payload["name"] = capped
    if not summary:
        payload["summary"] = name
    elif not summary.startswith(name):
        payload["summary"] = f"{name}. {summary}"
    candidate.payload = payload
    if candidate.corroborating_signals is None:
        candidate.corroborating_signals = []
    if "name_capped" not in candidate.corroborating_signals:
        candidate.corroborating_signals.append("name_capped")
    logger.warning(
        "Squatter guard: entity name exceeds %d tokens (%d); capped %r -> %r "
        "(excess folded into summary)",
        _MAX_ENTITY_NAME_TOKENS,
        len(tokens),
        name,
        capped,
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
        *,
        episode_source: str | None = None,
    ) -> list[CommitDecision]:
        """Evaluate each candidate in the bundle and return commit decisions.

        ``episode_source`` is the source of the episode being projected; for
        observation-class sources (mcp_observe / api_auto_observe / axi /
        auto:* hooks) entity candidates require >=2-episode corroboration
        before full commit (M1.4, extends the bare-proper-name gate).
        """
        observation_sourced = is_observation_source(episode_source)
        decisions: list[CommitDecision] = []
        for candidate in bundle.candidates:
            if candidate.fact_class == "entity":
                _cap_entity_name(candidate)
                if observation_sourced:
                    if candidate.corroborating_signals is None:
                        candidate.corroborating_signals = []
                    if "observation_sourced" not in candidate.corroborating_signals:
                        candidate.corroborating_signals.append("observation_sourced")
            threshold = self._effective_threshold(
                candidate.fact_class,
                entity_count,
                signals=candidate.corroborating_signals,
            )
            decision = self._decide(candidate, threshold)
            if (
                decision.action == "commit"
                and candidate.fact_class == "entity"
                and "observation_sourced" in (candidate.corroborating_signals or [])
                and not (_IDENTITY_SIGNALS & set(candidate.corroborating_signals or []))
            ):
                # Squatter guard: one-shot observation entities defer until a
                # second episode corroborates (evidence adjudication releases
                # the hold at group count >= 2).
                decision = CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="defer",
                    reason="observation_needs_corroboration",
                    effective_confidence=decision.effective_confidence,
                )
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
        signals = set(candidate.corroborating_signals or [])

        # Hot-path junk gate: never store pattern scrap as deferred debt.
        # (Client proposals still pass through their own trust gates below.)
        if candidate.source_type != "client_proposal":
            try:
                from engram.consolidation.evidence_drain import (
                    classify_extraction_candidate,
                )

                junk = classify_extraction_candidate(candidate)
                if junk.disposition == "reject_junk":
                    return CommitDecision(
                        evidence_id=candidate.evidence_id,
                        action="reject",
                        reason=f"junk:{junk.reason or 'unspecified'}",
                        effective_confidence=conf,
                    )
            except Exception:
                # Classification must never break extraction; fall through.
                pass

        # Hard trust gates for harness proposals (no silent fallthrough).
        if candidate.source_type == "client_proposal":
            if "predicate_not_allowed" in signals:
                return CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="reject",
                    reason="predicate_not_allowed",
                    effective_confidence=conf,
                )
            if "identity_core_conflict" in signals:
                return CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="defer",
                    reason="identity_core_conflict",
                    effective_confidence=conf,
                )
            if "span_unverified" in signals:
                # Span fail → defer (corroboration later). Never auto-upgrade to LLM.
                return CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="defer",
                    reason="span_unverified",
                    effective_confidence=conf,
                )
            if "date_conflict" in signals:
                return CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="defer",
                    reason="date_conflict",
                    effective_confidence=conf,
                )

        # Agent-promoted facts with verified spans are the product write path.
        # Do not bury them in the deferred evidence swamp.
        if (
            candidate.source_type == "client_proposal"
            and "span_verified" in signals
            and "date_conflict" not in signals
        ):
            entity_type = ""
            if candidate.fact_class == "entity":
                from engram.entity_dedup_policy import canonicalize_entity_type_case

                entity_type = canonicalize_entity_type_case(
                    str((candidate.payload or {}).get("entity_type") or "")
                )
            high_signal = (
                "high_signal_type" in signals
                or is_high_signal_entity_type(entity_type)
                or candidate.fact_class == "relationship"
            )
            if high_signal or conf >= threshold:
                return CommitDecision(
                    evidence_id=candidate.evidence_id,
                    action="commit",
                    reason="client_proposal_span_verified",
                    effective_confidence=conf,
                )

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
