"""Deterministic extractor for showcase seed episodes."""

from __future__ import annotations

from engram.extraction.extractor import ExtractionResult
from engram.showcase.beats import SHOWCASE_SEED_EPISODES


def _extraction_for_label(label: str) -> ExtractionResult:
    if label == "liam_soccer":
        return ExtractionResult(
            entities=[
                {
                    "name": "Liam",
                    "entity_type": "Person",
                    "summary": "Son who plays soccer on Tuesdays",
                    "identity_core": True,
                },
                {
                    "name": "Soccer",
                    "entity_type": "Concept",
                    "summary": "Sport Liam played on Tuesdays",
                },
                {
                    "name": "Family calendar",
                    "entity_type": "Concept",
                    "summary": "Tracks Liam's games and practices",
                },
            ],
            relationships=[
                {
                    "source": "Liam",
                    "target": "Soccer",
                    "predicate": "PLAYS",
                    "weight": 1.0,
                },
                {
                    "source": "Liam",
                    "target": "Family calendar",
                    "predicate": "TRACKED_IN",
                    "weight": 0.8,
                },
            ],
        )
    if label == "liam_correction":
        return ExtractionResult(
            entities=[
                {
                    "name": "Liam",
                    "entity_type": "Person",
                    "summary": "Switched from soccer to baseball in spring",
                    "identity_core": True,
                },
                {
                    "name": "Baseball",
                    "entity_type": "Concept",
                    "summary": "Sport Liam plays on Tuesdays after spring switch",
                },
                {
                    "name": "Soccer",
                    "entity_type": "Concept",
                    "summary": "Former Tuesday sport for Liam",
                },
            ],
            relationships=[
                {
                    "source": "Liam",
                    "target": "Baseball",
                    "predicate": "PLAYS",
                    "weight": 1.0,
                    "temporal_hint": "this spring",
                },
                {
                    "source": "Liam",
                    "target": "Soccer",
                    "predicate": "PLAYED",
                    "weight": 0.4,
                },
            ],
        )
    if label == "family_calendar":
        return ExtractionResult(
            entities=[
                {
                    "name": "Liam",
                    "entity_type": "Person",
                    "summary": "Ask about practice and games after Tuesday sessions",
                    "identity_core": True,
                },
                {
                    "name": "Family calendar",
                    "entity_type": "Concept",
                    "summary": "Holds Liam's practice schedule",
                },
            ],
            relationships=[
                {
                    "source": "Liam",
                    "target": "Family calendar",
                    "predicate": "TRACKED_IN",
                    "weight": 1.0,
                },
            ],
        )
    return ExtractionResult(entities=[], relationships=[])


class ShowcaseExtractor:
    """Returns canned extraction for each showcase seed episode in order."""

    def __init__(self) -> None:
        self._labels = [episode.label for episode in SHOWCASE_SEED_EPISODES]
        self._index = 0

    async def extract(self, text: str) -> ExtractionResult:
        label = self._labels[min(self._index, len(self._labels) - 1)]
        self._index += 1
        return _extraction_for_label(label)