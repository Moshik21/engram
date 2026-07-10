"""Convert client-supplied entity/relationship proposals to evidence candidates."""

from __future__ import annotations

from datetime import datetime

from engram.extraction.evidence import EvidenceCandidate
from engram.extraction.promotion import is_high_signal_entity_type
from engram.extraction.temporal import resolve_temporal_hint

# Confidence by calling model tier
MODEL_TIER_CONFIDENCE: dict[str, float] = {
    "opus": 0.92,
    "sonnet": 0.85,
    "haiku": 0.75,
    "default": 0.70,
}

# Trust hardening: an unverified, single-source client proposal must not commit
# on first sight. Caller-supplied model_tier (opus -> 0.92) otherwise exceeds the
# entity (0.70) and relationship (0.75) commit thresholds, so an unverified claim
# would weaponize confidence into an immediate commit. We cap unverified proposals
# into the defer band so they require span verification OR cross-episode
# corroboration before promotion. The cap stays inside the defer band (threshold -
# 0.15) so the commit policy defers (never rejects) the claim.
UNVERIFIED_PROPOSAL_CONFIDENCE_CAP = 0.62
# A span-verified proposal carries concrete textual evidence, so it earns a
# commit-worthy floor (above the relationship commit threshold of 0.75) even when
# the caller's model tier is low. Higher tiers keep their (higher) confidence.
SPAN_VERIFIED_PROPOSAL_CONFIDENCE = 0.80
# High-signal durable types (Decision/Person/…) with a verified span get a
# slightly higher floor so dense-graph threshold bumps cannot defer them.
HIGH_SIGNAL_VERIFIED_CONFIDENCE = 0.88


def _normalize_for_span(text: str) -> str:
    """Whitespace-normalize and casefold text for deterministic span matching."""
    return " ".join(text.split()).casefold()


def span_is_verified(source_span: str | None, episode_content: str | None) -> bool:
    """Return True when source_span is a whitespace-normalized casefold substring.

    Deterministic span validator: a claim's cited span must actually appear in the
    episode content. On a miss the caller defers the claim and tags it
    'span_unverified' (never rejects — the claim may still corroborate later).
    """
    if not source_span or not episode_content:
        return False
    needle = _normalize_for_span(source_span)
    if not needle:
        return False
    return needle in _normalize_for_span(episode_content)


def _reanchor_temporal(
    payload: dict,
    reference_date: datetime | None,
) -> bool:
    """Re-anchor a relative temporal phrase against an absolute proposed date.

    When an annotation supplies both an absolute ``valid_from`` and a relative
    ``temporal_hint``, re-resolve the phrase against ``reference_date`` (the
    conversation date) and flag a conflict when the two disagree. Returns True when
    a date conflict was detected (caller defers + tags 'date_conflict').
    """
    valid_from = payload.get("valid_from")
    temporal_hint = payload.get("temporal_hint")
    if not valid_from or not temporal_hint or reference_date is None:
        return False
    try:
        absolute = datetime.fromisoformat(str(valid_from))
    except (ValueError, TypeError):
        return False
    resolved = resolve_temporal_hint(str(temporal_hint), reference_date=reference_date)
    if resolved is None:
        return False
    # Disagreement when the relative phrase resolves to a different calendar day.
    return resolved.date() != absolute.date()


def _effective_proposal_confidence(
    base_confidence: float,
    *,
    span_verified: bool,
    extra_signal_count: int,
    high_signal: bool = False,
) -> float:
    """Cap unverified single-source proposals below commit thresholds.

    A proposal earns commit-worthy confidence only when its cited span is verified
    or it already carries additional corroborating signals beyond the bare
    source_type/model_tier markers. Otherwise it is capped into the defer band.

    High-signal durable types (Decision, Preference, Person, …) with a verified
    span get a higher floor so dense-graph adaptive thresholds cannot defer them.
    """
    if span_verified:
        floor = (
            HIGH_SIGNAL_VERIFIED_CONFIDENCE if high_signal else SPAN_VERIFIED_PROPOSAL_CONFIDENCE
        )
        # Floor verified claims at a commit-worthy confidence; keep higher tiers.
        return max(base_confidence, floor)
    if extra_signal_count > 0:
        # Multi-signal proposals keep their tier confidence (real corroboration).
        return base_confidence
    return min(base_confidence, UNVERIFIED_PROPOSAL_CONFIDENCE_CAP)


def proposals_to_evidence(
    entities: list[dict] | None,
    relationships: list[dict] | None,
    episode_id: str,
    group_id: str,
    model_tier: str = "default",
    *,
    source_type: str = "client_proposal",
    confidence_bonus: float = 0.0,
    adjudication_request_id: str | None = None,
    rationale: str | None = None,
    source_span: str | None = None,
    episode_content: str | None = None,
    reference_date: datetime | None = None,
    verify_spans: bool = False,
) -> list[EvidenceCandidate]:
    """Convert client-supplied proposals to EvidenceCandidate list.

    Each proposal becomes an EvidenceCandidate with source_type="client_proposal".
    Confidence is based on the calling model's tier, then de-weaponized: an
    unverified single-source proposal is capped into the defer band so span
    verification or cross-episode corroboration is required before it can commit.

    Args:
        entities: List of {"name": ..., "entity_type": ..., "source_span": ...} dicts
        relationships: List of
            {"subject": ..., "predicate": ..., "object": ..., "source_span": ...} dicts
        episode_id: The episode these proposals belong to
        group_id: The group context
        model_tier: The calling model tier (opus/sonnet/haiku/default)
        source_span: Legacy bundle-level span fallback when a claim omits its own.
        episode_content: Episode text used to validate each claim's source_span.
        reference_date: Conversation date used to re-anchor relative temporal hints.
        verify_spans: When True, run the deterministic span validator and apply
            trust-hardening confidence caps. The adjudication-resolution path passes
            curated spans and opts out.
    """
    base_confidence = min(
        0.97,
        MODEL_TIER_CONFIDENCE.get(model_tier, 0.70) + confidence_bonus,
    )
    candidates: list[EvidenceCandidate] = []
    extractor_prefix = "client" if source_type == "client_proposal" else source_type

    def _build_candidate(
        *,
        fact_class: str,
        payload: dict,
        claim: dict,
        high_signal: bool = False,
    ) -> EvidenceCandidate:
        claim_span = claim.get("source_span") or source_span
        # Auto-derive span from name/subject when agent omitted it but text contains it.
        if not claim_span and episode_content:
            for key in ("name", "subject", "object"):
                piece = str(payload.get(key) or claim.get(key) or "").strip()
                if piece and span_is_verified(piece, episode_content):
                    claim_span = piece
                    break
        signals = [source_type, f"model_{model_tier}"]
        confidence = base_confidence
        if verify_spans:
            verified = span_is_verified(claim_span, episode_content)
            if verified:
                signals.append("span_verified")
                # Only mark high-signal when the claim is grounded in the text.
                if high_signal:
                    signals.append("high_signal_type")
            else:
                signals.append("span_unverified")
            date_conflict = fact_class == "relationship" and _reanchor_temporal(
                payload,
                reference_date,
            )
            if date_conflict:
                signals.append("date_conflict")
            # extra_signal_count: corroboration beyond the bare source/model markers.
            confidence = _effective_proposal_confidence(
                base_confidence,
                span_verified=verified and not date_conflict,
                extra_signal_count=0,
                high_signal=high_signal and verified and not date_conflict,
            )
        elif high_signal:
            # Adjudication path (no verify_spans): still tag high-signal types.
            signals.append("high_signal_type")
        if rationale:
            payload = {**payload, "_adjudication_rationale": rationale}
        return EvidenceCandidate(
            episode_id=episode_id,
            group_id=group_id,
            fact_class=fact_class,
            confidence=confidence,
            source_type=source_type,
            extractor_name=f"{extractor_prefix}_{model_tier}",
            payload=payload,
            source_span=claim_span,
            adjudication_request_id=adjudication_request_id,
            corroborating_signals=signals,
        )

    for ent in entities or []:
        name = ent.get("name", "").strip()
        if not name:
            continue
        entity_type = ent.get("entity_type", "Concept")
        high_signal = is_high_signal_entity_type(str(entity_type))
        payload = {
            "name": name,
            "entity_type": entity_type,
            **({"summary": ent["summary"]} if ent.get("summary") else {}),
            **({"attributes": ent["attributes"]} if ent.get("attributes") else {}),
        }
        candidates.append(
            _build_candidate(
                fact_class="entity",
                payload=payload,
                claim=ent,
                high_signal=high_signal,
            ),
        )

    for rel in relationships or []:
        subject = rel.get("subject", "").strip()
        predicate = rel.get("predicate", "").strip()
        obj = rel.get("object", "").strip()
        if not subject or not predicate:
            continue
        payload = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "polarity": rel.get("polarity", "positive"),
            **({"temporal_hint": rel["temporal_hint"]} if rel.get("temporal_hint") else {}),
            **({"valid_from": rel["valid_from"]} if rel.get("valid_from") else {}),
            **({"valid_to": rel["valid_to"]} if rel.get("valid_to") else {}),
        }
        candidates.append(
            _build_candidate(fact_class="relationship", payload=payload, claim=rel),
        )

    return candidates


def events_to_proposals(
    events: list[dict] | None,
) -> tuple[list[dict], list[dict]]:
    """Materialize agent event annotations into Event entities + OCCURRED_ON edges.

    Each event ``{name, date, source_span}`` becomes:
      - an ``entity_type="Event"`` proposal for the event name, and
      - an ``OCCURRED_ON`` relationship from the event to a dated marker entity
        (``entity_type="Date"``) with ``valid_from=date`` so the event lands as a
        first-class dated node flowing through the normal evidence pipeline.

    Returns (entity_proposals, relationship_proposals) ready for
    ``proposals_to_evidence`` so events ride the existing commit/defer machinery.
    """
    entity_proposals: list[dict] = []
    relationship_proposals: list[dict] = []
    for event in events or []:
        if not isinstance(event, dict):
            continue
        name = str(event.get("name", "")).strip()
        if not name:
            continue
        date = event.get("date")
        span = event.get("source_span")
        entity_proposals.append(
            {
                "name": name,
                "entity_type": "Event",
                **({"source_span": span} if span else {}),
            },
        )
        if date:
            date_str = str(date).strip()
            entity_proposals.append(
                {
                    "name": date_str,
                    "entity_type": "Date",
                    **({"source_span": span} if span else {}),
                },
            )
            relationship_proposals.append(
                {
                    "subject": name,
                    "predicate": "OCCURRED_ON",
                    "object": date_str,
                    "valid_from": date_str,
                    **({"source_span": span} if span else {}),
                },
            )
    return entity_proposals, relationship_proposals
