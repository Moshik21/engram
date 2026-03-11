"""Narrow extractor for entity evidence from proper names, tech tokens, identity patterns."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceCandidate
from engram.models.episode_cue import EpisodeCue

# Reuse patterns from cues.py
_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
_TECHNICAL_TOKENS = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|"
    r"[A-Za-z][A-Za-z0-9_-]*\.(?:py|ts|tsx|js|jsx|json|toml|md|yaml|yml)|"
    r"(?:API|SDK|CLI|MCP|LLM|SQL|FTS5|Redis|SQLite|FalkorDB|FastAPI|React"
    r"|Next\.js|TypeScript))\b"
)
_IDENTITY_PATTERNS = re.compile(
    r"\b(?:my name is|i am|i'm|my wife|my husband|my partner|my mom|my dad|"
    r"i work at|i live in|we live in)\b",
    re.IGNORECASE,
)

# More specific identity captures that extract the *value*
_IDENTITY_CAPTURES: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, entity_type, signal_name)
    (
        re.compile(
            r"\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re.I,
        ),
        "Person",
        "name_declaration",
    ),
    (
        re.compile(
            r"\bi(?:'m| am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ),
        "Person",
        "self_introduction",
    ),
    (
        re.compile(
            r"\bi work(?:s)? (?:at|for)\s+([A-Z][a-zA-Z0-9 ]+?)"
            r"(?:\.|,|\band\b|$)",
            re.I,
        ),
        "Organization",
        "workplace_declaration",
    ),
    (
        re.compile(
            r"\bi live in\s+([A-Z][a-zA-Z ]+?)(?:\.|,|\band\b|$)", re.I,
        ),
        "Location",
        "residence_declaration",
    ),
    (
        re.compile(
            r"\bwe live in\s+([A-Z][a-zA-Z ]+?)(?:\.|,|\band\b|$)", re.I,
        ),
        "Location",
        "residence_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:wife|husband|partner)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:mom|dad|mother|father)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:son|daughter|brother|sister)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
]

# Type inference heuristics
_TECH_KEYWORDS = frozenset({
    "api", "sdk", "cli", "mcp", "llm", "sql", "fts5", "redis", "sqlite",
    "falkordb", "fastapi", "react", "next.js", "typescript", "python",
    "javascript", "node", "docker", "kubernetes", "aws", "gcp", "azure",
})

_LOCATION_SUFFIXES = frozenset({
    "city", "town", "village", "state", "county", "island",
    "creek", "valley", "heights", "hills", "beach", "springs",
})

# Common non-entity proper nouns to skip
_STOPWORDS = frozenset({
    "The", "This", "That", "These", "Those", "Here", "There",
    "What", "Which", "Where", "When", "Who", "How", "Why",
    "But", "And", "However", "Also", "Just", "Very",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
})


def _infer_entity_type(name: str, signal: str | None = None) -> str:
    """Infer entity type from name and context."""
    lower = name.lower()
    if signal in (
        "name_declaration", "self_introduction", "family_declaration",
    ):
        return "Person"
    if signal == "workplace_declaration":
        return "Organization"
    if signal == "residence_declaration":
        return "Location"
    if lower in _TECH_KEYWORDS or "." in name or "/" in name:
        return "Technology"
    tokens = lower.split()
    if tokens and tokens[-1] in _LOCATION_SUFFIXES:
        return "Location"
    # Default to Person for proper names, Concept for others
    if name[0].isupper() and name.isalpha():
        return "Person"
    return "Concept"


def _get_source_span(text: str, name: str) -> str | None:
    """Extract the sentence containing the entity name."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if name in sent:
            return sent[:200]
    return None


class IdentityEntityExtractor:
    """Extracts entity evidence from proper names, tech tokens, and identity patterns."""

    name = "identity_entity"

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]:
        candidates: list[EvidenceCandidate] = []
        seen_names: set[str] = set()

        # 1. Identity captures (highest confidence -- explicit declarations)
        for pattern, entity_type, signal in _IDENTITY_CAPTURES:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="entity",
                        confidence=0.90,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "name": name, "entity_type": entity_type,
                        },
                        source_span=_get_source_span(text, name),
                        corroborating_signals=[
                            signal, "identity_pattern",
                        ],
                    )
                )

        # 2. Proper names (medium confidence)
        for match in _PROPER_NAMES.finditer(text):
            name = match.group()
            if (
                not name
                or name in _STOPWORDS
                or name.lower() in seen_names
            ):
                continue
            seen_names.add(name.lower())
            entity_type = _infer_entity_type(name)
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="entity",
                    confidence=0.65,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload={
                        "name": name, "entity_type": entity_type,
                    },
                    source_span=_get_source_span(text, name),
                    corroborating_signals=["proper_name"],
                )
            )

        # 3. Technical tokens (medium confidence)
        for match in _TECHNICAL_TOKENS.finditer(text):
            token = match.group()
            if not token or token.lower() in seen_names:
                continue
            seen_names.add(token.lower())
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="entity",
                    confidence=0.70,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload={
                        "name": token, "entity_type": "Technology",
                    },
                    source_span=_get_source_span(text, token),
                    corroborating_signals=["technical_token"],
                )
            )

        return candidates
