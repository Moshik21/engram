"""Narrow extractor for relationship evidence from verb patterns."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.evidence import EvidenceCandidate
from engram.models.episode_cue import EpisodeCue

# Relationship verb patterns
_RELATIONSHIP_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+works?\s+"
            r"(?:at|for)\s+([A-Z][a-zA-Z0-9 ]+?)(?:\.|,|$)",
            re.I,
        ),
        "WORKS_AT",
        "works_at_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:lives?|resides?)\s+"
            r"in\s+([A-Z][a-zA-Z ]+?)(?:\.|,|$)",
            re.I,
        ),
        "LIVES_IN",
        "lives_in_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was)\s+"
            r"(?:married|engaged)\s+to\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "MARRIED_TO",
        "married_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:uses?|is using)\s+"
            r"([A-Z][a-zA-Z0-9.]+)",
            re.I,
        ),
        "USES",
        "uses_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:likes?|loves?|enjoys?)"
            r"\s+([A-Za-z][A-Za-z ]+?)(?:\.|,|$)",
            re.I,
        ),
        "LIKES",
        "preference_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
            r"(?:hates?|dislikes?|avoids?)\s+"
            r"([A-Za-z][A-Za-z ]+?)(?:\.|,|$)",
            re.I,
        ),
        "DISLIKES",
        "preference_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
            r"(?:knows?|is friends? with)\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "KNOWS",
        "social_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
            r"(?:studies?|is studying|studied)\s+(?:at\s+)?"
            r"([A-Z][a-zA-Z ]+?)(?:\.|,|$)",
            re.I,
        ),
        "STUDIES_AT",
        "education_pattern",
    ),
    (
        re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+"
            r"(?:manages?|leads?|runs?)\s+"
            r"([A-Z][a-zA-Z0-9 ]+?)(?:\.|,|$)",
            re.I,
        ),
        "MANAGES",
        "management_pattern",
    ),
]

# First-person relationship patterns (subject = "User" implied)
_FIRST_PERSON_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\bi\s+work\s+(?:at|for)\s+([A-Z][a-zA-Z0-9 ]+?)"
            r"(?:\.|,|\band\b|$)",
            re.I,
        ),
        "WORKS_AT",
        "first_person_works",
    ),
    (
        re.compile(
            r"\bi\s+(?:live|reside)\s+in\s+([A-Z][a-zA-Z ]+?)"
            r"(?:\.|,|\band\b|$)",
            re.I,
        ),
        "LIVES_IN",
        "first_person_lives",
    ),
    (
        re.compile(
            r"\bi\s+(?:use|am using)\s+([A-Z][a-zA-Z0-9.]+)",
            re.I,
        ),
        "USES",
        "first_person_uses",
    ),
    (
        re.compile(
            r"\bi\s+(?:like|love|enjoy)\s+([A-Za-z][A-Za-z ]+?)"
            r"(?:\.|,|$)",
            re.I,
        ),
        "LIKES",
        "first_person_pref",
    ),
    (
        re.compile(
            r"\bi\s+(?:hate|dislike|avoid)\s+([A-Za-z][A-Za-z ]+?)"
            r"(?:\.|,|$)",
            re.I,
        ),
        "DISLIKES",
        "first_person_pref",
    ),
    (
        re.compile(
            r"\bi\s+(?:study|am studying|studied)\s+(?:at\s+)?"
            r"([A-Z][a-zA-Z ]+?)(?:\.|,|$)",
            re.I,
        ),
        "STUDIES_AT",
        "first_person_edu",
    ),
    (
        re.compile(
            r"\bi\s+(?:manage|lead|run)\s+"
            r"([A-Z][a-zA-Z0-9 ]+?)(?:\.|,|$)",
            re.I,
        ),
        "MANAGES",
        "first_person_mgmt",
    ),
]

# Family relationship patterns
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\bmy\s+(wife|husband|partner|spouse)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "PARTNER_OF",
        "family_partner",
    ),
    (
        re.compile(
            r"\bmy\s+(son|daughter|child)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "PARENT_OF",
        "family_child",
    ),
    (
        re.compile(
            r"\bmy\s+(mom|dad|mother|father)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "CHILD_OF",
        "family_parent",
    ),
    (
        re.compile(
            r"\bmy\s+(brother|sister|sibling)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "SIBLING_OF",
        "family_sibling",
    ),
]

# Contradiction patterns (from cues.py)
_CONTRADICTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "negation",
        re.compile(
            r"\b(?:no longer|doesn't|don't|didn't|not anymore|never)\b",
            re.I,
        ),
    ),
    (
        "ended",
        re.compile(
            r"\b(?:stopped|quit|left|ended|cancelled|canceled)\b",
            re.I,
        ),
    ),
    (
        "correction",
        re.compile(
            r"\b(?:actually|correction|instead|updated|changed)\b",
            re.I,
        ),
    ),
    (
        "move",
        re.compile(r"\b(?:moved to|moved from)\b", re.I),
    ),
]

_canonicalizer = PredicateCanonicalizer()


def _has_contradiction_near(
    text: str,
    start: int,
    end: int,
    window: int = 80,
) -> bool:
    """Check if any contradiction marker appears near the match span."""
    context = text[max(0, start - window) : min(len(text), end + window)]
    return any(pat.search(context) for _, pat in _CONTRADICTION_PATTERNS)


class RelationshipPatternExtractor:
    """Extracts relationship evidence from verb patterns."""

    name = "relationship_pattern"

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]:
        candidates: list[EvidenceCandidate] = []
        seen: set[tuple[str, str, str]] = set()

        # 1. Third-person relationship patterns
        for pattern, predicate, signal in _RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                canonical = _canonicalizer.canonicalize(predicate)
                key = (subject.lower(), canonical, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                negated = _has_contradiction_near(
                    text,
                    match.start(),
                    match.end(),
                )
                polarity = "negative" if negated else "positive"
                signals = [signal]
                if negated:
                    signals.append("contradiction_nearby")
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="relationship",
                        confidence=0.75 if not negated else 0.60,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "subject": subject,
                            "predicate": canonical,
                            "object": obj,
                            "polarity": polarity,
                        },
                        source_span=text[
                            max(0, match.start() - 20) : min(
                                len(text),
                                match.end() + 20,
                            )
                        ],
                        corroborating_signals=signals,
                    )
                )

        # 2. First-person patterns (User is implicit subject)
        for pattern, predicate, signal in _FIRST_PERSON_PATTERNS:
            for match in pattern.finditer(text):
                obj = match.group(1).strip()
                canonical = _canonicalizer.canonicalize(predicate)
                key = ("user", canonical, obj.lower())
                if key in seen:
                    continue
                seen.add(key)
                negated = _has_contradiction_near(
                    text,
                    match.start(),
                    match.end(),
                )
                polarity = "negative" if negated else "positive"
                signals = [signal, "first_person"]
                if negated:
                    signals.append("contradiction_nearby")
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="relationship",
                        confidence=0.80 if not negated else 0.60,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "subject": "User",
                            "predicate": canonical,
                            "object": obj,
                            "polarity": polarity,
                        },
                        source_span=text[
                            max(0, match.start() - 20) : min(
                                len(text),
                                match.end() + 20,
                            )
                        ],
                        corroborating_signals=signals,
                    )
                )

        # 3. Family patterns
        for pattern, predicate, signal in _FAMILY_PATTERNS:
            for match in pattern.finditer(text):
                role = match.group(1).strip().lower()
                fname = match.group(2).strip()
                canonical = _canonicalizer.canonicalize(predicate)
                key = ("user", canonical, fname.lower())
                if key in seen:
                    continue
                seen.add(key)
                signals = [signal, "family_relation", f"role_{role}"]
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="relationship",
                        confidence=0.85,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "subject": "User",
                            "predicate": canonical,
                            "object": fname,
                            "polarity": "positive",
                        },
                        source_span=text[
                            max(0, match.start() - 20) : min(
                                len(text),
                                match.end() + 20,
                            )
                        ],
                        corroborating_signals=signals,
                    )
                )

        return candidates
