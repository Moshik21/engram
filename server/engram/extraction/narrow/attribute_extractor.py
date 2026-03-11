"""Narrow extractor for attribute evidence (preferences, quantities, states)."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceCandidate
from engram.models.episode_cue import EpisodeCue

# Preference patterns
_PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(
            r"\bi\s+(?:prefer|always use|default to)\s+"
            r"([A-Za-z][A-Za-z0-9. ]+?)"
            r"(?:\s+(?:over|instead|rather)|[.,;]|$)",
            re.I,
        ),
        "preference",
        "explicit_preference",
    ),
    (
        re.compile(
            r"\bmy\s+(?:favorite|preferred|go-to)\s+(?:\w+\s+)?"
            r"(?:is|are)\s+([A-Za-z][A-Za-z0-9. ]+?)(?:[.,;]|$)",
            re.I,
        ),
        "preference",
        "favorite_declaration",
    ),
    (
        re.compile(
            r"\bi\s+(?:always|usually|typically)\s+"
            r"([a-z][a-z ]+?)"
            r"(?:\s+(?:when|for|because)|[.,;]|$)",
            re.I,
        ),
        "habit",
        "habitual_pattern",
    ),
]

# Number with context (reused from cues.py)
_NUMBERS_WITH_CONTEXT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*"
    r"(percent|%|dollars?|USD|EUR|years?|months?|hours?|minutes?"
    r"|GB|MB|TB|k|K|M)\b"
)

# State/status patterns
_STATE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\bi(?:'m| am)\s+(?:currently\s+)?(?:a|an)\s+"
            r"([a-zA-Z ]+?)(?:\s+at|\s+for|[.,;]|$)",
            re.I,
        ),
        "role_state",
    ),
    (
        re.compile(
            r"\bi(?:'m| am)\s+(?:currently\s+)?"
            r"(?:learning|studying|working on|building|developing)\s+"
            r"([A-Za-z][A-Za-z0-9. ]+?)(?:[.,;]|$)",
            re.I,
        ),
        "activity_state",
    ),
    (
        re.compile(
            r"\bi(?:'m| am)\s+(?:based|located)\s+in\s+"
            r"([A-Z][a-zA-Z ]+?)(?:[.,;]|$)",
            re.I,
        ),
        "location_state",
    ),
]

# Nearby entity for attaching attributes
_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")


class AttributeEvidenceExtractor:
    """Extracts preference, quantity, and state attribute evidence."""

    name = "attribute"

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]:
        candidates: list[EvidenceCandidate] = []
        seen: set[str] = set()

        # 1. Preference patterns
        for pattern, attr_type, signal in _PREFERENCE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value or value.lower() in seen:
                    continue
                seen.add(value.lower())
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="attribute",
                        confidence=0.70,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "entity": "User",
                            "attribute_type": attr_type,
                            "value": value,
                        },
                        source_span=text[
                            max(0, match.start() - 20):min(
                                len(text), match.end() + 20,
                            )
                        ],
                        corroborating_signals=[signal],
                    )
                )

        # 2. Quantitative attributes
        for match in _NUMBERS_WITH_CONTEXT.finditer(text):
            amount = match.group(1)
            unit = match.group(2).strip()
            span_key = f"{amount}_{unit}".lower()
            if span_key in seen:
                continue
            seen.add(span_key)
            # Find nearby entity to attach to
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            nearby_matches = list(
                _PROPER_NAMES.finditer(text[start:end]),
            )
            nearby = (
                nearby_matches[0].group() if nearby_matches else None
            )
            payload: dict = {
                "attribute_type": "quantity",
                "value": float(amount),
                "unit": unit,
            }
            if nearby:
                payload["entity"] = nearby
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="attribute",
                    confidence=0.60,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload=payload,
                    source_span=text[
                        max(0, match.start() - 30):min(
                            len(text), match.end() + 30,
                        )
                    ],
                    corroborating_signals=["number_with_context"],
                )
            )

        # 3. State patterns
        for pattern, signal in _STATE_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value or value.lower() in seen:
                    continue
                seen.add(value.lower())
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="attribute",
                        confidence=0.72,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "entity": "User",
                            "attribute_type": "state",
                            "value": value,
                            "state_signal": signal,
                        },
                        source_span=text[
                            max(0, match.start() - 20):min(
                                len(text), match.end() + 20,
                            )
                        ],
                        corroborating_signals=[signal],
                    )
                )

        return candidates
