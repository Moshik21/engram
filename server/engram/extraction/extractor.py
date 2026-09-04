"""Extraction result types shared by every extractor.

The Anthropic ``EntityExtractor`` that lived here was deleted 2026-09-04:
Engram never calls an external model in operation. The resident agent
proposes structure through the MCP surface and the deterministic narrow
adapter (``engram.extraction.narrow_adapter``) is the only internal rung.
What remains is the result contract those callers share, plus the
``EntityExtractor`` protocol that names it.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol

_MAX_INPUT_CHARS = 8000  # Truncate very long episodes to save input tokens
MAX_EXTRACTION_INPUT_CHARS = _MAX_INPUT_CHARS


class ExtractionResult:
    """Parsed extraction output."""

    def __init__(
        self,
        entities: list[dict],
        relationships: list[dict],
        status: ExtractionStatus | None = None,
        error: str | None = None,
    ) -> None:
        self.entities = entities
        self.relationships = relationships
        self.status = status or (
            ExtractionStatus.OK if entities or relationships else ExtractionStatus.EMPTY
        )
        self.error = error

    @property
    def is_error(self) -> bool:
        return self.status in {
            ExtractionStatus.PARSE_ERROR,
            ExtractionStatus.API_ERROR,
            ExtractionStatus.TRUNCATED,
        }

    @property
    def retryable(self) -> bool:
        return self.status in {
            ExtractionStatus.PARSE_ERROR,
            ExtractionStatus.API_ERROR,
            ExtractionStatus.TRUNCATED,
        }


class ExtractionStatus(str, Enum):
    OK = "ok"
    EMPTY = "empty"
    PARSE_ERROR = "parse_error"
    API_ERROR = "api_error"
    TRUNCATED = "truncated"


class EntityExtractor(Protocol):
    """Interface every extractor handed to GraphManager satisfies.

    Concrete implementations: ``NarrowExtractorAdapter`` (production), the
    canned-result stubs in the test suite, and the benchmark adapters.
    """

    async def extract(self, text: str) -> ExtractionResult: ...
