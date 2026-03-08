"""Projection bundle builder around the extractor."""

from __future__ import annotations

from typing import Any

from engram.extraction.extractor import ExtractionStatus
from engram.extraction.models import (
    ClaimCandidate,
    EntityCandidate,
    ProjectionBundle,
    ProjectionPlan,
)


class EpisodeProjector:
    """Run extraction against a plan and build typed intermediate candidates."""

    def __init__(self, extractor: Any) -> None:
        self._extractor = extractor

    async def project(self, plan: ProjectionPlan) -> ProjectionBundle:
        """Execute extraction for the selected plan text and type the results."""
        result = await self._extractor.extract(plan.selected_text)

        is_error = getattr(result, "is_error", False)
        if not isinstance(is_error, bool):
            is_error = False

        raw_status = getattr(result, "status", ExtractionStatus.OK)
        status_value = getattr(raw_status, "value", None)
        if not isinstance(status_value, str):
            if isinstance(raw_status, str):
                status_value = raw_status
            else:
                status_value = ExtractionStatus.OK.value

        entities = [
            self._to_entity_candidate(ent_data, plan)
            for ent_data in getattr(result, "entities", [])
        ]
        claims = [
            self._to_claim_candidate(rel_data, plan)
            for rel_data in getattr(result, "relationships", [])
        ]

        warnings = list(getattr(plan, "warnings", []))
        if plan.was_truncated:
            warnings.append("planned_subset_only")

        return ProjectionBundle(
            episode_id=plan.episode_id,
            plan=plan,
            entities=entities,
            claims=claims,
            warnings=warnings,
            extractor_status=status_value if is_error or entities or claims else "empty",
            extractor_error=getattr(result, "error", None),
            retryable=bool(getattr(result, "retryable", False)),
        )

    @staticmethod
    def _to_entity_candidate(
        ent_data: dict,
        plan: ProjectionPlan,
    ) -> EntityCandidate:
        name = ent_data.get("name", "")
        source_span_ids = _match_span_ids(
            plan,
            [name],
        )
        return EntityCandidate(
            name=name,
            entity_type=ent_data.get("entity_type", "Other"),
            summary=ent_data.get("summary"),
            attributes=ent_data.get("attributes") or None,
            pii_detected=bool(ent_data.get("pii_detected", False)),
            pii_categories=ent_data.get("pii_categories"),
            epistemic_mode=ent_data.get("epistemic_mode"),
            source_span_ids=source_span_ids,
            raw_payload=dict(ent_data),
        )

    @staticmethod
    def _to_claim_candidate(
        rel_data: dict,
        plan: ProjectionPlan,
    ) -> ClaimCandidate:
        subject_text = (
            rel_data.get("source")
            or rel_data.get("source_entity")
            or rel_data.get("source_name")
            or ""
        )
        object_text = (
            rel_data.get("target")
            or rel_data.get("target_entity")
            or rel_data.get("target_name")
        )
        predicate = (
            rel_data.get("predicate")
            or rel_data.get("relationship_type")
            or rel_data.get("type")
            or "RELATES_TO"
        )
        source_span_ids = _match_span_ids(
            plan,
            [subject_text, object_text or "", predicate.replace("_", " ")],
        )
        confidence = rel_data.get("confidence", 1.0)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 1.0
        return ClaimCandidate(
            subject_text=subject_text,
            predicate=predicate,
            object_text=object_text,
            object_value=rel_data.get("object_value"),
            polarity=rel_data.get("polarity", "positive"),
            temporal_hint=(
                rel_data.get("temporal_hint")
                or rel_data.get("valid_from")
                or rel_data.get("valid_to")
            ),
            confidence=confidence,
            source_span_ids=source_span_ids,
            raw_payload=dict(rel_data),
        )


def _match_span_ids(
    plan: ProjectionPlan,
    terms: list[str],
) -> list[str]:
    lowered_terms = [term.strip().lower() for term in terms if term and term.strip()]
    if not lowered_terms:
        return [plan.spans[0].span_id] if plan.spans else []

    matches: list[str] = []
    for span in plan.spans:
        lowered = span.text.lower()
        if any(term in lowered for term in lowered_terms):
            matches.append(span.span_id)

    if matches:
        return matches
    return [plan.spans[0].span_id] if plan.spans else []
