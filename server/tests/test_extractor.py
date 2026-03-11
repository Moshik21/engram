"""Tests for EntityExtractor — parsing logic and error handling."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from engram.extraction.extractor import EntityExtractor, ExtractionStatus


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
        assert result.status == ExtractionStatus.PARSE_ERROR

    async def test_api_exception_returns_empty(self):
        extractor = EntityExtractor()
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API down")
        extractor._client = mock_client
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == []
        assert result.status == ExtractionStatus.API_ERROR

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

    async def test_creative_work_entity_type_in_response(self):
        """CreativeWork entity type parses correctly from extraction response."""
        data = {
            "entities": [
                {"name": "The Agent of Fate", "entity_type": "CreativeWork"},
            ],
            "relationships": [
                {"source": "Alex", "target": "The Agent of Fate", "predicate": "CREATED"},
            ],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("Alex wrote a fantasy book called The Agent of Fate")
        assert len(result.entities) == 1
        assert result.entities[0]["entity_type"] == "CreativeWork"
        assert result.relationships[0]["predicate"] == "CREATED"

    async def test_epistemic_mode_field_in_response(self):
        """epistemic_mode field is preserved from extraction response."""
        data = {
            "entities": [
                {
                    "name": "Alice",
                    "entity_type": "Person",
                    "summary": "Data scientist",
                    "epistemic_mode": "direct",
                },
                {
                    "name": "Retrieval Pipeline",
                    "entity_type": "Technology",
                    "summary": "Scored 0.9 on Alice",
                    "epistemic_mode": "meta",
                },
            ],
            "relationships": [],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert len(result.entities) == 2
        assert result.entities[0]["epistemic_mode"] == "direct"
        assert result.entities[1]["epistemic_mode"] == "meta"

    async def test_empty_response_returns_empty(self):
        data = {"entities": [], "relationships": []}
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("test text")
        assert result.entities == []
        assert result.relationships == []
        assert result.status == ExtractionStatus.EMPTY

    async def test_uses_cached_system_prompt(self):
        """System kwarg should be a list with cache_control for prompt caching."""
        data = {"entities": [], "relationships": []}
        extractor = self._make_extractor_with_response(json.dumps(data))
        await extractor.extract("test text")

        call_kwargs = extractor._client.messages.create.call_args[1]
        assert isinstance(call_kwargs["system"], list)
        assert len(call_kwargs["system"]) == 1
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    async def test_health_condition_entity_type(self):
        """HealthCondition entity type parses correctly from extraction response."""
        data = {
            "entities": [
                {
                    "name": "Type 2 Diabetes",
                    "entity_type": "HealthCondition",
                    "summary": "Chronic metabolic condition",
                },
                {
                    "name": "Pancreas",
                    "entity_type": "BodyPart",
                    "summary": "Organ that produces insulin",
                },
            ],
            "relationships": [
                {"source": "Type 2 Diabetes", "target": "Pancreas", "predicate": "AFFECTS"},
            ],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract(
            "He was diagnosed with type 2 diabetes which affects the pancreas"
        )
        assert len(result.entities) == 2
        assert result.entities[0]["entity_type"] == "HealthCondition"
        assert result.entities[1]["entity_type"] == "BodyPart"

    async def test_emotion_entity_type(self):
        """Emotion entity type parses correctly from extraction response."""
        data = {
            "entities": [
                {
                    "name": "Anxiety",
                    "entity_type": "Emotion",
                    "summary": "Feeling of worry and unease",
                },
            ],
            "relationships": [],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract("I have been feeling a lot of anxiety about the move")
        assert len(result.entities) == 1
        assert result.entities[0]["entity_type"] == "Emotion"

    async def test_goal_preference_habit_entity_types(self):
        """Goal, Preference, and Habit entity types parse correctly."""
        data = {
            "entities": [
                {
                    "name": "Run A Marathon",
                    "entity_type": "Goal",
                    "summary": "Aspiration to complete a marathon",
                },
                {
                    "name": "Morning Running",
                    "entity_type": "Habit",
                    "summary": "Daily morning running routine",
                },
                {
                    "name": "Plant-Based Diet",
                    "entity_type": "Preference",
                    "summary": "Prefers plant-based foods",
                },
            ],
            "relationships": [
                {"source": "Morning Running", "target": "Run A Marathon", "predicate": "SUPPORTS"},
            ],
        }
        extractor = self._make_extractor_with_response(json.dumps(data))
        result = await extractor.extract(
            "My goal is to run a marathon. I run every morning and prefer a plant-based diet."
        )
        assert len(result.entities) == 3
        types = {e["entity_type"] for e in result.entities}
        assert types == {"Goal", "Habit", "Preference"}
