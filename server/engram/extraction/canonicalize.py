"""Predicate canonicalization — maps synonyms to canonical forms."""

from __future__ import annotations

CANONICAL_MAP: dict[str, str] = {
    # Employment
    "EMPLOYED_BY": "WORKS_AT",
    "WORKS_FOR": "WORKS_AT",
    "EMPLOYED_AT": "WORKS_AT",
    "HIRED_BY": "WORKS_AT",
    # Expertise
    "SKILLED_IN": "EXPERT_IN",
    "SPECIALIZES_IN": "EXPERT_IN",
    "PROFICIENT_IN": "EXPERT_IN",
    "KNOWLEDGEABLE_IN": "EXPERT_IN",
    # Location
    "LIVES_IN": "LOCATED_IN",
    "RESIDES_IN": "LOCATED_IN",
    "BASED_IN": "LOCATED_IN",
    "SITUATED_IN": "LOCATED_IN",
    # Social
    "IS_ACQUAINTED_WITH": "KNOWS",
    "MET": "KNOWS",
    "FRIENDS_WITH": "KNOWS",
    "CONNECTED_TO": "KNOWS",
    # Creation
    "BUILT": "CREATED",
    "DEVELOPED": "CREATED",
    "AUTHORED": "CREATED",
    "DESIGNED": "CREATED",
    "WROTE": "CREATED",
    "PUBLISHED": "CREATED",
    "COMPOSED": "CREATED",
    "PRODUCED": "CREATED",
    "FOUNDED": "CREATED",
    # Membership
    "BELONGS_TO": "MEMBER_OF",
    "JOINED": "MEMBER_OF",
    "AFFILIATED_WITH": "MEMBER_OF",
    # Usage
    "UTILIZES": "USES",
    "EMPLOYS": "USES",
    # Leadership
    "MANAGES": "LEADS",
    "DIRECTS": "LEADS",
    "HEADS": "LEADS",
    # Collaboration
    "WORKS_WITH": "COLLABORATES_WITH",
    "PARTNERS_WITH": "COLLABORATES_WITH",
    "COOPERATES_WITH": "COLLABORATES_WITH",
    # Research
    "STUDIES": "RESEARCHES",
    "INVESTIGATES": "RESEARCHES",
    # Mentorship
    "TEACHES": "MENTORS",
    "COACHES": "MENTORS",
    "TRAINS": "MENTORS",
}


class PredicateCanonicalizer:
    """Maps predicate synonyms to canonical forms.

    Unknown predicates pass through unchanged.
    """

    def __init__(self, extra_mappings: dict[str, str] | None = None) -> None:
        self._map = dict(CANONICAL_MAP)
        if extra_mappings:
            self._map.update(extra_mappings)

    def canonicalize(self, predicate: str) -> str:
        """Return the canonical form of a predicate.

        Normalizes to uppercase with underscores before lookup.
        Unknown predicates pass through unchanged.
        """
        normalized = predicate.upper().replace(" ", "_")
        return self._map.get(normalized, normalized)
