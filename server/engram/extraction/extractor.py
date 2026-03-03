"""Entity extraction using Anthropic Claude API."""

from __future__ import annotations

import json
import logging
import os

from engram.extraction.prompts import EXTRACTION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Parsed extraction output."""

    def __init__(
        self,
        entities: list[dict],
        relationships: list[dict],
    ) -> None:
        self.entities = entities
        self.relationships = relationships


class EntityExtractor:
    """Extracts entities and relationships from text using Claude Haiku."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self._model = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
        return self._client

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text."""
        try:
            client = self._get_client()
            message = client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text}],
            )

            response_text = message.content[0].text
            response_text = self._strip_markdown_fences(response_text)
            data = json.loads(response_text)

            return ExtractionResult(
                entities=data.get("entities", []),
                relationships=data.get("relationships", []),
            )
        except json.JSONDecodeError as e:
            logger.error("Failed to parse extraction response as JSON: %s\n%s", e, response_text)
            return ExtractionResult(entities=[], relationships=[])
        except Exception as e:
            logger.error("Entity extraction failed: %s", e)
            return ExtractionResult(entities=[], relationships=[])

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
