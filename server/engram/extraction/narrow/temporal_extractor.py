"""Narrow extractor for temporal evidence markers."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceCandidate
from engram.models.episode_cue import EpisodeCue

# Reuse from cues.py
_DATES = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|"
    r"(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|"
    r"next\s+(?:week|month|year)|since\s+[A-Z][a-z]+))\b",
    re.IGNORECASE,
)

# Duration patterns
_DURATIONS = re.compile(
    r"\b(?:for\s+)?(\d+)\s*(years?|months?|weeks?|days?|hours?)\b",
    re.IGNORECASE,
)

# Deadline/event patterns
_EVENT_TEMPORAL = re.compile(
    r"\b(?:by|before|after|until|starting|ending|due)\s+"
    r"(?:(\d{4}[-/]\d{1,2}[-/]\d{1,2})|"
    r"(next\s+(?:week|month|year|Monday|Tuesday|Wednesday|"
    r"Thursday|Friday))|"
    r"(tomorrow|today))\b",
    re.IGNORECASE,
)

# Nearby entity detection for attaching temporal markers
_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")


def _find_nearby_entity(
    text: str, pos: int, window: int = 100,
) -> str | None:
    """Find the closest proper name entity near a temporal marker."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]
    matches = list(_PROPER_NAMES.finditer(context))
    if not matches:
        return None
    # Return the match closest to the center of the window
    center = min(window, pos)
    best = min(matches, key=lambda m: abs(m.start() - center))
    return best.group()


class TemporalEvidenceExtractor:
    """Extracts temporal markers and attaches them to nearby entities."""

    name = "temporal"

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]:
        candidates: list[EvidenceCandidate] = []
        seen_spans: set[str] = set()

        # 1. Date markers
        for match in _DATES.finditer(text):
            marker = match.group().strip()
            if marker.lower() in seen_spans:
                continue
            seen_spans.add(marker.lower())
            nearby = _find_nearby_entity(text, match.start())
            payload: dict = {
                "temporal_marker": marker,
                "temporal_type": "date",
            }
            if nearby:
                payload["nearby_entity"] = nearby
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="temporal",
                    confidence=0.70,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload=payload,
                    source_span=text[
                        max(0, match.start() - 30):min(
                            len(text), match.end() + 30,
                        )
                    ],
                    corroborating_signals=["date_pattern"],
                )
            )

        # 2. Duration markers
        for match in _DURATIONS.finditer(text):
            span_text = match.group().strip()
            if span_text.lower() in seen_spans:
                continue
            seen_spans.add(span_text.lower())
            nearby = _find_nearby_entity(text, match.start())
            payload = {
                "temporal_marker": span_text,
                "temporal_type": "duration",
                "amount": int(match.group(1)),
                "unit": match.group(2).lower().rstrip("s"),
            }
            if nearby:
                payload["nearby_entity"] = nearby
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="temporal",
                    confidence=0.60,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload=payload,
                    source_span=text[
                        max(0, match.start() - 30):min(
                            len(text), match.end() + 30,
                        )
                    ],
                    corroborating_signals=["duration_pattern"],
                )
            )

        # 3. Event temporal markers
        for match in _EVENT_TEMPORAL.finditer(text):
            span_text = match.group().strip()
            if span_text.lower() in seen_spans:
                continue
            seen_spans.add(span_text.lower())
            nearby = _find_nearby_entity(text, match.start())
            payload = {
                "temporal_marker": span_text,
                "temporal_type": "event_deadline",
            }
            if nearby:
                payload["nearby_entity"] = nearby
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="temporal",
                    confidence=0.65,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload=payload,
                    source_span=text[
                        max(0, match.start() - 30):min(
                            len(text), match.end() + 30,
                        )
                    ],
                    corroborating_signals=["event_temporal_pattern"],
                )
            )

        return candidates
