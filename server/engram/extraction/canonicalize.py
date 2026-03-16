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
    # Health / Physical
    "INJURED": "RECOVERING_FROM",
    "HURT": "RECOVERING_FROM",
    "DIAGNOSED_WITH": "HAS_CONDITION",
    "SUFFERS_FROM": "HAS_CONDITION",
    "TREATING": "TREATS",
    "PRESCRIBED": "TREATS",
    # Sentiment / Preference
    "ENJOYS": "LIKES",
    "LOVES": "LIKES",
    "APPRECIATES": "LIKES",
    "HATES": "DISLIKES",
    "AVOIDS": "DISLIKES",
    "PREFERS": "PREFERS",
    "FAVORS": "PREFERS",
    # Goals / Aspirations
    "WANTS_TO": "AIMS_FOR",
    "PLANS_TO": "AIMS_FOR",
    "INTENDS_TO": "AIMS_FOR",
    "ASPIRES_TO": "AIMS_FOR",
    "WORKING_TOWARD": "AIMS_FOR",
    # Causation
    "CAUSED": "CAUSED_BY",
    "RESULTED_IN": "LED_TO",
    "TRIGGERED": "LED_TO",
    "DEPENDS_ON": "REQUIRES",
    "NEEDS": "REQUIRES",
    "BLOCKED_BY": "REQUIRES",
    # Conditional / Causal links
    "ALLOWS": "ENABLES",
    "FACILITATES": "ENABLES",
    "MAKES_POSSIBLE": "ENABLES",
    "SUPPORTS": "ENABLES",
    "BLOCKS": "PREVENTS",
    "INHIBITS": "PREVENTS",
    "STOPS": "PREVENTS",
    "RESTRICTS": "PREVENTS",
    # Hierarchy / Containment
    "CONTAINS": "HAS_PART",
    "INCLUDES": "HAS_PART",
    "COMPOSED_OF": "HAS_PART",
    "PARENT_OF": "PARENT_OF",
    "CHILD_OF": "CHILD_OF",
    # Learning
    "LEARNING": "STUDYING",
    "READING": "STUDYING",
    "PRACTICING": "STUDYING",
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
