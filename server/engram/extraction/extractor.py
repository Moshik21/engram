"""Entity extraction using Anthropic Claude API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any

from engram.extraction.prompts import EXTRACTION_SYSTEM_CACHED

logger = logging.getLogger(__name__)

# Rough chars-to-tokens ratio for estimating output budget
_CHARS_PER_TOKEN = 4
_MIN_OUTPUT_TOKENS = 2048
_MAX_OUTPUT_TOKENS = 8192
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


class EntityExtractor:
    """Extracts entities and relationships from text using Claude Haiku."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self._model = model
        self._client: Any | None = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        return self._client

    @staticmethod
    def _extract_message_text(blocks: object) -> str:
        if not isinstance(blocks, list):
            return ""
        parts: list[str] = []
        for block in blocks:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "".join(parts)

    def _estimate_max_tokens(self, text: str) -> int:
        """Estimate output tokens needed based on input size.

        Longer inputs tend to produce more entities/relationships,
        requiring more output tokens for the JSON response.
        """
        input_tokens_est = len(text) // _CHARS_PER_TOKEN
        # Output is roughly proportional to input for extraction tasks
        budget = max(_MIN_OUTPUT_TOKENS, input_tokens_est)
        return min(budget, _MAX_OUTPUT_TOKENS)

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text."""
        response_text = ""
        try:
            client = self._get_client()
            # Truncate very long input to save tokens
            if len(text) > _MAX_INPUT_CHARS:
                logger.info(
                    "Truncating extraction input from %d to %d chars",
                    len(text), _MAX_INPUT_CHARS,
                )
                text = text[:_MAX_INPUT_CHARS]
            max_tokens = self._estimate_max_tokens(text)
            message = await asyncio.to_thread(
                client.messages.create,
                model=self._model,
                max_tokens=max_tokens,
                system=EXTRACTION_SYSTEM_CACHED,
                messages=[{"role": "user", "content": text}],
            )

            response_text = self._extract_message_text(message.content)

            # Check for truncation before parsing
            if message.stop_reason == "max_tokens":
                logger.warning(
                    "Extraction response truncated (max_tokens=%d, input_chars=%d). "
                    "Retrying with max budget.",
                    max_tokens,
                    len(text),
                )
                # Retry once with maximum budget
                if max_tokens < _MAX_OUTPUT_TOKENS:
                    message = await asyncio.to_thread(
                        client.messages.create,
                        model=self._model,
                        max_tokens=_MAX_OUTPUT_TOKENS,
                        system=EXTRACTION_SYSTEM_CACHED,
                        messages=[{"role": "user", "content": text}],
                    )
                    response_text = self._extract_message_text(message.content)
                    if message.stop_reason == "max_tokens":
                        logger.error(
                            "Extraction still truncated at max budget (%d tokens). "
                            "Input too large (%d chars).",
                            _MAX_OUTPUT_TOKENS,
                            len(text),
                        )
                        return ExtractionResult(
                            entities=[],
                            relationships=[],
                            status=ExtractionStatus.TRUNCATED,
                            error="response_truncated",
                        )

            response_text = self._strip_markdown_fences(response_text)
            data = self._parse_json_lenient(response_text)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])
            logger.info(
                "Extracted %d entities, %d relationships from %d chars",
                len(entities),
                len(relationships),
                len(text),
            )

            status = ExtractionStatus.OK if entities or relationships else ExtractionStatus.EMPTY
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                status=status,
            )
        except json.JSONDecodeError as e:
            logger.error("Failed to parse extraction response as JSON: %s\n%s", e, response_text)
            return ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.PARSE_ERROR,
                error=str(e),
            )
        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.API_ERROR,
                error=str(e),
            )

    @staticmethod
    def _parse_json_lenient(text: str) -> dict[str, Any]:
        """Parse JSON, tolerating trailing data from LLM responses.

        Falls back to raw_decode when json.loads fails with 'Extra data',
        which happens when the model emits two JSON blocks or appends
        commentary after the closing brace.
        """
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise json.JSONDecodeError("expected object", text, 0)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text)
                logger.warning(
                    "Extraction response had trailing data at char %d; "
                    "parsed first JSON object only.",
                    e.pos,
                )
                if isinstance(obj, dict):
                    return obj
                raise
            raise

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
