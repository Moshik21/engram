"""Deterministic ambiguity detection for edge-case adjudication."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from engram.extraction.evidence import EvidenceBundle, EvidenceCandidate

_THIRD_PERSON_PRONOUNS = re.compile(
    r"\b(?:he|she|they|them|their|theirs|his|her|hers)\b",
    re.IGNORECASE,
)
_HEDGE_MARKERS = re.compile(
    r"\b(?:think|guess|maybe|probably|might|seems?|unsure|not sure|anymore)\b",
    re.IGNORECASE,
)
_NEGATION_MARKERS = re.compile(
    r"\b(?:not|never|no longer|doesn't|don't|didn't|stopped|quit|left|ended)\b",
    re.IGNORECASE,
)
_EXCLUSIVE_PREDICATES = {
    "WORKS_AT",
    "LIVES_IN",
    "STUDIES_AT",
    "PARTNER_OF",
    "MARRIED_TO",
}
_ENTITY_FIELDS = ("name", "entity", "subject", "object", "nearby_entity")


@dataclass
class AmbiguityGroup:
    """Episode-local cluster of ambiguous evidence rows."""

    candidates: list[EvidenceCandidate] = field(default_factory=list)
    ambiguity_tags: set[str] = field(default_factory=set)
    selected_text: str = ""
    request_reason: str = ""


@dataclass
class AmbiguityAnalysis:
    """Partition of extracted evidence into clean and ambiguous subsets."""

    clean_candidates: list[EvidenceCandidate] = field(default_factory=list)
    ambiguous_groups: list[AmbiguityGroup] = field(default_factory=list)


class AmbiguityAnalyzer:
    """Tag extraction edge cases that should bypass normal hot-path commit."""

    def __init__(self, graph_store) -> None:
        self._graph = graph_store

    async def analyze(
        self,
        *,
        text: str,
        bundle: EvidenceBundle,
        group_id: str,
    ) -> AmbiguityAnalysis:
        """Split a bundle into clean candidates and grouped ambiguous work items."""
        if not bundle.candidates:
            return AmbiguityAnalysis()

        tagged: list[EvidenceCandidate] = []
        clean: list[EvidenceCandidate] = []

        for candidate in bundle.candidates:
            tags = set(await self._candidate_tags(candidate, bundle, group_id))
            if tags:
                candidate.ambiguity_tags = sorted(tags)
                candidate.ambiguity_score = min(
                    1.0,
                    max(candidate.confidence, 0.45) + (0.08 * len(tags)),
                )
                tagged.append(candidate)
            else:
                clean.append(candidate)

        # If a temporal marker is ambiguous, group related relationships with it.
        temporal_groups = [
            candidate
            for candidate in tagged
            if "temporal_attachment" in candidate.ambiguity_tags
            and candidate.fact_class == "temporal"
        ]
        for temporal in temporal_groups:
            for candidate in bundle.candidates:
                if candidate in tagged or candidate.fact_class != "relationship":
                    continue
                if self._shares_context(temporal, candidate):
                    candidate.ambiguity_tags = sorted(
                        set(candidate.ambiguity_tags) | {"temporal_attachment"},
                    )
                    candidate.ambiguity_score = max(
                        candidate.ambiguity_score,
                        max(candidate.confidence, 0.45) + 0.08,
                    )
                    tagged.append(candidate)
                    if candidate in clean:
                        clean.remove(candidate)

        return AmbiguityAnalysis(
            clean_candidates=clean,
            ambiguous_groups=self._group(tagged, text),
        )

    async def _candidate_tags(
        self,
        candidate: EvidenceCandidate,
        bundle: EvidenceBundle,
        group_id: str,
    ) -> list[str]:
        tags: list[str] = []
        source_text = candidate.source_span or ""

        if (
            candidate.fact_class in {"relationship", "temporal", "attribute"}
            and source_text
            and _THIRD_PERSON_PRONOUNS.search(source_text)
        ):
            tags.append("coreference")

        if (
            candidate.fact_class == "relationship"
            and (
                candidate.payload.get("polarity") == "negative"
                or (
                    source_text
                    and _NEGATION_MARKERS.search(source_text)
                    and _HEDGE_MARKERS.search(source_text)
                )
            )
        ):
            tags.append("negation_scope")

        if candidate.fact_class == "temporal":
            attachment_count = self._temporal_attachment_count(candidate, bundle)
            if attachment_count != 1:
                tags.append("temporal_attachment")

        if await self._conflicts_with_existing(candidate, group_id):
            tags.append("conflict_with_existing")

        return tags

    def _temporal_attachment_count(
        self,
        candidate: EvidenceCandidate,
        bundle: EvidenceBundle,
    ) -> int:
        nearby = (candidate.payload.get("nearby_entity") or "").strip().lower()
        if not nearby:
            return 0
        matches = 0
        for other in bundle.candidates:
            if other.fact_class == "relationship":
                subject = (other.payload.get("subject") or "").strip().lower()
                obj = (other.payload.get("object") or "").strip().lower()
                if nearby in {subject, obj}:
                    matches += 1
            elif other.fact_class in {"entity", "attribute"}:
                name = (
                    other.payload.get("name")
                    or other.payload.get("entity")
                    or ""
                ).strip().lower()
                if name == nearby:
                    matches += 1
        return matches

    async def _conflicts_with_existing(
        self,
        candidate: EvidenceCandidate,
        group_id: str,
    ) -> bool:
        if candidate.fact_class != "relationship":
            return False
        predicate = (candidate.payload.get("predicate") or "").upper()
        polarity = candidate.payload.get("polarity", "positive")
        if polarity != "negative" and predicate not in _EXCLUSIVE_PREDICATES:
            return False

        subject = (candidate.payload.get("subject") or "").strip()
        if not subject or subject == "User":
            return False

        subject_matches = await self._graph.find_entity_candidates(
            subject,
            group_id,
            limit=5,
        )
        if not subject_matches:
            return False
        if len(subject_matches) > 1:
            return True

        existing = await self._graph.get_relationships(
            subject_matches[0].id,
            direction="outgoing",
            group_id=group_id,
        )
        active = [
            rel for rel in existing if rel.predicate.upper() == predicate and rel.valid_to is None
        ]
        if not active:
            return False
        if polarity == "negative":
            return candidate.confidence < 0.85

        obj = (candidate.payload.get("object") or "").strip()
        if not obj:
            return True
        object_matches = await self._graph.find_entity_candidates(obj, group_id, limit=5)
        if len(object_matches) != 1:
            return True
        return any(rel.target_id != object_matches[0].id for rel in active)

    def _group(
        self,
        candidates: list[EvidenceCandidate],
        text: str,
    ) -> list[AmbiguityGroup]:
        groups: list[AmbiguityGroup] = []
        for candidate in candidates:
            if not candidate.ambiguity_tags:
                continue
            target = None
            for group in groups:
                if any(
                    self._shares_context(candidate, existing)
                    for existing in group.candidates
                ):
                    target = group
                    break
            if target is None:
                target = AmbiguityGroup()
                groups.append(target)
            target.candidates.append(candidate)
            target.ambiguity_tags.update(candidate.ambiguity_tags)
            if candidate.source_span and len(candidate.source_span) > len(target.selected_text):
                target.selected_text = candidate.source_span.strip()

        for group in groups:
            if not group.selected_text:
                group.selected_text = text[:240]
            group.request_reason = (
                "needs_adjudication:" + ",".join(sorted(group.ambiguity_tags))
            )
        return groups

    def _shares_context(
        self,
        left: EvidenceCandidate,
        right: EvidenceCandidate,
    ) -> bool:
        left_span = (left.source_span or "").strip().lower()
        right_span = (right.source_span or "").strip().lower()
        if left_span and right_span and (left_span in right_span or right_span in left_span):
            return True
        return bool(self._entity_names(left) & self._entity_names(right))

    def _entity_names(self, candidate: EvidenceCandidate) -> set[str]:
        names: set[str] = set()
        for key in _ENTITY_FIELDS:
            value = candidate.payload.get(key)
            if not value:
                continue
            normalized = str(value).strip().lower()
            if normalized and normalized not in {"user", "none"}:
                names.add(normalized)
        return names
