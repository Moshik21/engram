"""Emotional salience scoring — pure regex, ~0.1ms per call."""

from __future__ import annotations

import re
from dataclasses import dataclass

# State-change verbs indicating emotional arousal
_AROUSAL_VERBS = re.compile(
    r"\b(?:diagnosed|fired|hired|promoted|married|divorced|born|died|"
    r"crashed|broke|won|lost|failed|passed|moved|graduated|retired|"
    r"attacked|escaped|survived|discovered|confessed|betrayed|"
    r"quit|started|ended|collapsed|exploded|panicked)\b",
    re.IGNORECASE,
)

_INTENSIFIERS = re.compile(
    r"\b(?:very|extremely|incredibly|absolutely|completely|totally|"
    r"utterly|deeply|profoundly|terribly|desperately|seriously|"
    r"really|so much|never|always|worst|best)\b",
    re.IGNORECASE,
)

_PUNCTUATION_ENERGY = re.compile(r"[!?]{2,}|\.{3,}")

_SELF_PRONOUNS = re.compile(
    r"\b(?:I|me|my|mine|myself|we|our|ours|ourselves)\b",
)

_POSSESSIVE_RELATIONS = re.compile(
    r"\b(?:my\s+(?:mom|dad|mother|father|brother|sister|wife|husband|"
    r"partner|friend|son|daughter|child|family|boss|doctor|therapist|"
    r"teacher|dog|cat|home|car|job|life))\b",
    re.IGNORECASE,
)

_SOCIAL_ROLES = re.compile(
    r"\b(?:mom|dad|mother|father|brother|sister|wife|husband|partner|"
    r"friend|colleague|boss|teacher|doctor|therapist|mentor|coach|"
    r"neighbor|roommate|classmate|teammate)\b",
    re.IGNORECASE,
)

_PROPER_NOUNS = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")

_UNCERTAINTY_MARKERS = re.compile(
    r"\b(?:maybe|perhaps|might|could|wondering|unsure|confused|"
    r"not sure|don't know|worried about|afraid that|hoping|"
    r"what if|should I|trying to figure)\b",
    re.IGNORECASE,
)

_OPEN_LOOPS = re.compile(
    r"\b(?:haven't decided|still thinking|waiting for|need to figure|"
    r"not yet|planning to|want to|going to|considering|debating)\b",
    re.IGNORECASE,
)


@dataclass
class EmotionalSalience:
    """Emotional salience scores for a text."""

    arousal: float  # 0-1
    self_reference: float  # 0-1
    social_density: float  # 0-1
    narrative_tension: float  # 0-1

    @property
    def composite(self) -> float:
        """Weighted composite: arousal 0.35, self_ref 0.30, social 0.20, tension 0.15."""
        return (
            0.35 * self.arousal
            + 0.30 * self.self_reference
            + 0.20 * self.social_density
            + 0.15 * self.narrative_tension
        )


def compute_arousal(content: str) -> float:
    """State-change verbs, intensifiers, punctuation energy -> [0, 1]."""
    if not content:
        return 0.0
    verbs = len(_AROUSAL_VERBS.findall(content))
    intensifiers = len(_INTENSIFIERS.findall(content))
    punctuation = len(_PUNCTUATION_ENERGY.findall(content))
    raw = verbs * 0.3 + intensifiers * 0.15 + punctuation * 0.2
    return min(1.0, raw)


def compute_self_reference(content: str) -> float:
    """Pronoun ratio + possessive relations -> [0, 1]."""
    if not content:
        return 0.0
    words = content.split()
    if not words:
        return 0.0
    pronouns = len(_SELF_PRONOUNS.findall(content))
    possessives = len(_POSSESSIVE_RELATIONS.findall(content))
    ratio = pronouns / len(words)
    raw = ratio * 3.0 + possessives * 0.25
    return min(1.0, raw)


def compute_social_density(content: str) -> float:
    """Role terms + proper nouns -> [0, 1]."""
    if not content:
        return 0.0
    roles = len(_SOCIAL_ROLES.findall(content))
    proper = len(_PROPER_NOUNS.findall(content))
    raw = roles * 0.2 + proper * 0.15
    return min(1.0, raw)


def compute_narrative_tension(content: str) -> float:
    """Uncertainty markers, open loops -> [0, 1]."""
    if not content:
        return 0.0
    uncertainty = len(_UNCERTAINTY_MARKERS.findall(content))
    loops = len(_OPEN_LOOPS.findall(content))
    raw = uncertainty * 0.2 + loops * 0.25
    return min(1.0, raw)


def compute_emotional_salience(content: str) -> EmotionalSalience:
    """Compute all emotional salience dimensions for content."""
    return EmotionalSalience(
        arousal=compute_arousal(content),
        self_reference=compute_self_reference(content),
        social_density=compute_social_density(content),
        narrative_tension=compute_narrative_tension(content),
    )
