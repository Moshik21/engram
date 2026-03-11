"""Entity extraction using a local Ollama LLM."""

from __future__ import annotations

import json
import logging
from typing import Any

from engram.extraction.extractor import ExtractionResult, ExtractionStatus
from engram.extraction.prompts import EXTRACTION_SYSTEM_CACHED

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 8000


class OllamaExtractor:
    """Extracts entities and relationships using a local Ollama model."""

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")

    @staticmethod
    async def is_available(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama is reachable."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{base_url.rstrip('/')}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text via Ollama."""
        import httpx

        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.EMPTY,
            )

        if len(text) > _MAX_INPUT_CHARS:
            text = text[:_MAX_INPUT_CHARS]

        # Build the system prompt text (strip cache_control for Ollama)
        system_text = ""
        if isinstance(EXTRACTION_SYSTEM_CACHED, list):
            for block in EXTRACTION_SYSTEM_CACHED:
                if isinstance(block, dict):
                    system_text += block.get("text", "")
                elif isinstance(block, str):
                    system_text += block
        elif isinstance(EXTRACTION_SYSTEM_CACHED, str):
            system_text = EXTRACTION_SYSTEM_CACHED

        payload = {
            "model": self._model,
            "system": system_text,
            "prompt": text,
            "stream": False,
            "format": "json",
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self._base_url}/api/generate",
                    json=payload,
                )
                resp.raise_for_status()

            body = resp.json()
            response_text = body.get("response", "")
            response_text = self._strip_markdown_fences(response_text)
            data = self._parse_json_lenient(response_text)

            entities = data.get("entities", [])
            relationships = data.get("relationships", [])

            logger.info(
                "Ollama extraction (%s): %d entities, %d relationships from %d chars",
                self._model,
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
            logger.error("Failed to parse Ollama response as JSON: %s", e)
            return ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.PARSE_ERROR,
                error=str(e),
            )
        except Exception as e:
            logger.error("Ollama extraction failed: %s", e)
            return ExtractionResult(
                entities=[],
                relationships=[],
                status=ExtractionStatus.API_ERROR,
                error=str(e),
            )

    @staticmethod
    def _parse_json_lenient(text: str) -> dict[str, Any]:
        """Parse JSON, tolerating trailing data."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
            raise json.JSONDecodeError("expected object", text, 0)
        except json.JSONDecodeError as e:
            if "Extra data" in str(e):
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text)
                if isinstance(obj, dict):
                    return obj
                raise
            raise

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """Strip markdown code fences from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
