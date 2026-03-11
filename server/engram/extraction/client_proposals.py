"""Convert client-supplied entity/relationship proposals to evidence candidates."""

from __future__ import annotations

from engram.extraction.evidence import EvidenceCandidate

# Confidence by calling model tier
MODEL_TIER_CONFIDENCE: dict[str, float] = {
    "opus": 0.92,
    "sonnet": 0.85,
    "haiku": 0.75,
    "default": 0.70,
}


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
) -> list[EvidenceCandidate]:
    """Convert client-supplied proposals to EvidenceCandidate list.

    Each proposal becomes an EvidenceCandidate with source_type="client_proposal".
    Confidence is based on the calling model's tier.

    Args:
        entities: List of {"name": ..., "entity_type": ...} dicts
        relationships: List of {"subject": ..., "predicate": ..., "object": ...} dicts
        episode_id: The episode these proposals belong to
        group_id: The group context
        model_tier: The calling model tier (opus/sonnet/haiku/default)
    """
    base_confidence = min(
        0.97,
        MODEL_TIER_CONFIDENCE.get(model_tier, 0.70) + confidence_bonus,
    )
    candidates: list[EvidenceCandidate] = []
    extractor_prefix = "client" if source_type == "client_proposal" else source_type

    for ent in entities or []:
        name = ent.get("name", "").strip()
        if not name:
            continue
        payload = {
            "name": name,
            "entity_type": ent.get("entity_type", "Concept"),
            **({"summary": ent["summary"]} if ent.get("summary") else {}),
            **({"attributes": ent["attributes"]} if ent.get("attributes") else {}),
        }
        if rationale:
            payload["_adjudication_rationale"] = rationale
        candidates.append(
            EvidenceCandidate(
                episode_id=episode_id,
                group_id=group_id,
                fact_class="entity",
                confidence=base_confidence,
                source_type=source_type,
                extractor_name=f"{extractor_prefix}_{model_tier}",
                payload=payload,
                source_span=source_span,
                adjudication_request_id=adjudication_request_id,
                corroborating_signals=[source_type, f"model_{model_tier}"],
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
        if rationale:
            payload["_adjudication_rationale"] = rationale
        candidates.append(
            EvidenceCandidate(
                episode_id=episode_id,
                group_id=group_id,
                fact_class="relationship",
                confidence=base_confidence,
                source_type=source_type,
                extractor_name=f"{extractor_prefix}_{model_tier}",
                payload=payload,
                source_span=source_span,
                adjudication_request_id=adjudication_request_id,
                corroborating_signals=[source_type, f"model_{model_tier}"],
            ),
        )

    return candidates
