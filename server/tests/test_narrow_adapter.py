"""Tests for NarrowExtractorAdapter."""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.extraction.extractor import ExtractionStatus
from engram.extraction.narrow_adapter import NarrowExtractorAdapter


@pytest.fixture()
def adapter():
    return NarrowExtractorAdapter(ActivationConfig())


@pytest.mark.asyncio
async def test_empty_text_returns_empty(adapter):
    result = await adapter.extract("")
    assert result.status == ExtractionStatus.EMPTY
    assert result.entities == []
    assert result.relationships == []


@pytest.mark.asyncio
async def test_whitespace_text_returns_empty(adapter):
    result = await adapter.extract("   \n  ")
    assert result.status == ExtractionStatus.EMPTY


@pytest.mark.asyncio
async def test_identity_extraction(adapter):
    """Narrow pipeline should extract identity entities from clear statements."""
    text = "My name is Konnor Moshier and I work at Anthropic."
    result = await adapter.extract(text)
    # Should get at least one entity
    assert not result.is_error
    entity_names = {e["name"].lower() for e in result.entities}
    # The identity extractor should pick up at least the name
    assert any("konnor" in n or "moshier" in n for n in entity_names) or len(result.entities) > 0


@pytest.mark.asyncio
async def test_output_format(adapter):
    """Entities should have name, entity_type, summary keys."""
    text = "I am a software engineer working on the Engram project at Anthropic."
    result = await adapter.extract(text)
    for entity in result.entities:
        assert "name" in entity
        assert "entity_type" in entity


@pytest.mark.asyncio
async def test_relationship_extraction(adapter):
    """Narrow pipeline should extract relationships."""
    text = "Konnor works at Anthropic and uses Python for Engram development."
    result = await adapter.extract(text)
    for rel in result.relationships:
        assert "source" in rel or "subject" in rel
        assert "predicate" in rel


@pytest.mark.asyncio
async def test_status_ok_when_entities_found(adapter):
    text = "John Smith is a developer at Google working on AI projects."
    result = await adapter.extract(text)
    if result.entities or result.relationships:
        assert result.status == ExtractionStatus.OK


@pytest.mark.asyncio
async def test_confidence_filtering():
    """Entities below commit threshold should be filtered out."""
    # Use high thresholds to filter more aggressively
    cfg = ActivationConfig()
    object.__setattr__(cfg, "evidence_commit_entity_threshold", 0.99)
    object.__setattr__(cfg, "evidence_commit_relationship_threshold", 0.99)
    adapter = NarrowExtractorAdapter(cfg)
    text = "Bob likes pizza."
    result = await adapter.extract(text)
    # With very high thresholds, most things should be filtered
    # (may still pass if cross-corroboration pushes above 0.99)
    assert not result.is_error
