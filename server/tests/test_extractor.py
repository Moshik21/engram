"""Tests for EntityExtractor — parsing logic and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from engram.extraction.extractor import EntityExtractor


class TestStripMarkdownFences:
    """Tests for _strip_markdown_fences (pure function)."""

    def test_strips_json_fences(self):
        raw = '```json\n{"entities": []}\n```'
        assert EntityExtractor._strip_markdown_fences(raw) == '{"entities": []}'

    def test_strips_bare_fences(self):
        raw = '```\n{"entities": []}\n```'
        assert EntityExtractor._strip_markdown_fences(raw) == '{"entities": []}'

    def test_no_fences_passthrough(self):
        raw = '{"entities": []}'
        assert EntityExtractor._strip_markdown_fences(raw) == '{"entities": []}'

    def test_empty_string(self):
        assert EntityExtractor._strip_markdown_fences("") == ""


@pytest.mark.asyncio
class TestExtract:
    """Tests for extract() with mocked Anthropic client."""

    def _make_extractor_with_response(self, response_text: str) -> EntityExtractor:
        """Create an extractor with a mocked client returning response_text."""
        extractor = EntityExtractor()
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_content = MagicMock()
        mock_content.text = response_text
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        extractor._client = mock_client
        return extractor

    async def test_happy_path(self):
        data = {
            "entities": [{"name": "Python", "entity_type": "Technology"}],
            "relationships": [{"source": "A", "target": "B", "predicate": "USES"}],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Python"
        assert len(result.relationships) == 1
        assert result.relationships[0]["predicate"] == "USES"

    async def test_markdown_wrapped_json(self):
        data = {"entities": [{"name": "Go", "entity_type": "Technology"}], "relationships": []}
        wrapped = f"```json\n{json.dumps(data)}\n```"
        extractor = self._make_extractor_with_response(wrapped)
        result = await extractor.extract("test text")
        assert len(result.entities) == 1
        assert result.entities[0]["name"] == "Go"

    async def test_invalid_json_returns_empty(self):
        extractor = self._make_extractor_with_response("not valid json {{{")
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == []

    async def test_api_exception_returns_empty(self):
        extractor = EntityExtractor()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        extractor._client = mock_client
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == []

    async def test_missing_relationships_key_defaults_empty(self):
        data = {"entities": [{"name": "X", "entity_type": "Thing"}]}
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert result.entities == [{"name": "X", "entity_type": "Thing"}]
        assert result.relationships == []

    async def test_missing_entities_key_defaults_empty(self):
        data = {"relationships": [{"source": "A", "target": "B", "predicate": "KNOWS"}]}
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == [{"source": "A", "target": "B", "predicate": "KNOWS"}]

    async def test_empty_response_returns_empty(self):
        data = {"entities": [], "relationships": []}
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == []
