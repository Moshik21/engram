"""Bridge from evidence candidates to EntityCandidate/ClaimCandidate for ApplyEngine."""

from __future__ import annotations

import re

from engram.extraction.evidence import CommitDecision, EvidenceCandidate
from engram.extraction.models import ClaimCandidate, EntityCandidate


def _extractive_summary(source_span: str | None, name: str) -> str | None:
    """Generate extractive summary from the source span containing the entity name."""
    if not source_span:
        return None
    # Find first sentence containing the entity name
    sentences = re.split(r"(?<=[.!?])\s+", source_span)
    for sent in sentences:
        if name.lower() in sent.lower():
            return sent[:200].strip()
    # Fallback: use the whole span
    return source_span[:200].strip()


class EvidenceBridge:
    """Converts committed evidence into EntityCandidate/ClaimCandidate for ApplyEngine.

    This is the compatibility layer that allows the evidence pipeline to feed
    into the existing ApplyEngine without any changes to the apply path.
    """

    def bridge(
        self,
        committed: list[tuple[EvidenceCandidate, CommitDecision]],
    ) -> tuple[list[EntityCandidate], list[ClaimCandidate]]:
        """Convert committed evidence pairs into entity and claim candidates."""
        entities: list[EntityCandidate] = []
        claims: list[ClaimCandidate] = []

        for evidence, _decision in committed:
            if evidence.fact_class == "entity":
                entities.append(self._to_entity(evidence))
            elif evidence.fact_class == "relationship":
                claims.append(self._to_claim(evidence))
            elif evidence.fact_class == "attribute":
                entity = self._attribute_to_entity(evidence)
                if entity:
                    entities.append(entity)
            # temporal evidence does not directly produce entities/claims;
            # it enriches relationships via temporal_hint

        # Attach temporal hints to claims
        self._attach_temporal_hints(committed, claims)

        return entities, claims

    def _to_entity(self, ev: EvidenceCandidate) -> EntityCandidate:
        """Convert entity evidence to EntityCandidate."""
        name = ev.payload.get("name", "")
        entity_type = ev.payload.get("entity_type", "Concept")
        summary = ev.payload.get("summary") or _extractive_summary(ev.source_span, name)
        return EntityCandidate(
            name=name,
            entity_type=entity_type,
            summary=summary,
            attributes=ev.payload.get("attributes"),
            raw_payload={
                "evidence_id": ev.evidence_id,
                "confidence": ev.confidence,
                "signals": ev.corroborating_signals,
                **(
                    {"adjudication_request_id": ev.adjudication_request_id}
                    if ev.adjudication_request_id
                    else {}
                ),
            },
        )

    def _to_claim(self, ev: EvidenceCandidate) -> ClaimCandidate:
        """Convert relationship evidence to ClaimCandidate."""
        return ClaimCandidate(
            subject_text=ev.payload.get("subject", ""),
            predicate=ev.payload.get("predicate", ""),
            object_text=ev.payload.get("object"),
            polarity=ev.payload.get("polarity", "positive"),
            confidence=ev.confidence,
            raw_payload={
                "evidence_id": ev.evidence_id,
                "source": ev.payload.get("subject", ""),
                "target": ev.payload.get("object"),
                "predicate": ev.payload.get("predicate", ""),
                "polarity": ev.payload.get("polarity", "positive"),
                **(
                    {"temporal_hint": ev.payload.get("temporal_hint")}
                    if ev.payload.get("temporal_hint")
                    else {}
                ),
                **(
                    {"valid_from": ev.payload.get("valid_from")}
                    if ev.payload.get("valid_from")
                    else {}
                ),
                **(
                    {"valid_to": ev.payload.get("valid_to")}
                    if ev.payload.get("valid_to")
                    else {}
                ),
                "confidence": ev.confidence,
                "signals": ev.corroborating_signals,
                "temporal_evidence_ids": [],
                **(
                    {"adjudication_request_id": ev.adjudication_request_id}
                    if ev.adjudication_request_id
                    else {}
                ),
            },
        )

    def _attribute_to_entity(
        self, ev: EvidenceCandidate,
    ) -> EntityCandidate | None:
        """Convert attribute evidence to an EntityCandidate with attributes dict."""
        entity_name = ev.payload.get("entity")
        if not entity_name:
            return None
        attr_type = ev.payload.get("attribute_type", "unknown")
        value = ev.payload.get("value", "")
        return EntityCandidate(
            name=entity_name,
            entity_type="Person" if entity_name == "User" else "Concept",
            attributes={attr_type: value},
            raw_payload={
                "evidence_id": ev.evidence_id,
                "confidence": ev.confidence,
                "attribute_type": attr_type,
            },
        )

    def _attach_temporal_hints(
        self,
        committed: list[tuple[EvidenceCandidate, CommitDecision]],
        claims: list[ClaimCandidate],
    ) -> None:
        """Attach temporal markers to nearby claims."""
        temporal_by_entity: dict[str, list[tuple[str, str]]] = {}
        for ev, _ in committed:
            if ev.fact_class == "temporal":
                nearby = ev.payload.get("nearby_entity")
                marker = ev.payload.get("temporal_marker")
                if nearby and marker:
                    temporal_by_entity.setdefault(nearby.lower(), []).append(
                        (marker, ev.evidence_id),
                    )

        for claim in claims:
            subj = (claim.subject_text or "").lower()
            obj = (claim.object_text or "").lower()
            if subj in temporal_by_entity and not claim.temporal_hint:
                marker, evidence_id = temporal_by_entity[subj][0]
                claim.temporal_hint = marker
                claim.raw_payload.setdefault("temporal_evidence_ids", []).append(
                    evidence_id,
                )
            elif obj in temporal_by_entity and not claim.temporal_hint:
                marker, evidence_id = temporal_by_entity[obj][0]
                claim.temporal_hint = marker
                claim.raw_payload.setdefault("temporal_evidence_ids", []).append(
                    evidence_id,
                )
