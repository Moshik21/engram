"""Tests for fuzzy entity deduplication resolver."""

from unittest.mock import AsyncMock

import pytest

from engram.extraction.resolver import (
    compute_similarity,
    normalize_name,
    resolve_entity,
    resolve_entity_fast,
    validate_entity_name,
)
from engram.models.entity import Entity


class TestNormalizeName:
    def test_strip_whitespace(self):
        assert normalize_name("  Python  ") == "python"

    def test_lowercase(self):
        assert normalize_name("FastAPI") == "fastapi"

    def test_replace_hyphens(self):
        assert normalize_name("ACT-R") == "act r"

    def test_replace_underscores(self):
        assert normalize_name("some_name") == "some name"


class TestComputeSimilarity:
    def test_exact_match(self):
        assert compute_similarity("Python", "Python") == 1.0

    def test_case_insensitive(self):
        assert compute_similarity("python", "Python") == 1.0

    def test_hyphen_underscore_normalization(self):
        assert compute_similarity("ACT-R", "ACT_R") == 1.0

    def test_fuzzy_above_threshold(self):
        # "ReactJS" vs "React.js" — moderate similarity (0.77)
        # With type boost (+0.05) this crosses the 0.85 threshold in resolve_entity
        score = compute_similarity("ReactJS", "React.js")
        assert score >= 0.75

    def test_fuzzy_below_threshold(self):
        # Completely different strings
        score = compute_similarity("Redis", "Anthropic")
        assert score < 0.50

    def test_substring_containment(self):
        # "ACT-R" should partially match "ACT-R Spreading Activation"
        score = compute_similarity("ACT-R", "ACT-R Spreading Activation")
        assert score >= 0.85

    def test_token_reorder(self):
        # Token sort handles word reordering
        score = compute_similarity("ACT-R Spreading Activation", "Spreading Activation ACT-R")
        assert score >= 0.95

    def test_mesa_arizona_match(self):
        # "Mesa, Arizona" should match "Mesa" via partial ratio
        score = compute_similarity("Mesa, Arizona", "Mesa")
        assert score >= 0.85

    def test_no_false_positive_redis_react(self):
        score = compute_similarity("Redis", "React")
        assert score < 0.85

    def test_symmetry(self):
        score_ab = compute_similarity("FastAPI", "Fast API")
        score_ba = compute_similarity("Fast API", "FastAPI")
        assert score_ab == score_ba


@pytest.mark.asyncio
class TestResolveEntity:
    async def test_exact_match(self):
        existing = [
            Entity(id="ent_1", name="Python", entity_type="Technology"),
        ]
        result = await resolve_entity("Python", "Technology", existing)
        assert result is not None
        assert result.id == "ent_1"

    async def test_case_insensitive_match(self):
        existing = [
            Entity(id="ent_1", name="Python", entity_type="Technology"),
        ]
        result = await resolve_entity("python", "Technology", existing)
        assert result is not None

    async def test_fuzzy_match(self):
        existing = [
            Entity(id="ent_1", name="React", entity_type="Technology"),
        ]
        result = await resolve_entity("ReactJS", "Technology", existing)
        assert result is not None
        assert result.id == "ent_1"

    async def test_no_match_below_threshold(self):
        existing = [
            Entity(id="ent_1", name="Redis", entity_type="Technology"),
        ]
        result = await resolve_entity("Python", "Technology", existing)
        assert result is None

    async def test_type_priority_boost(self):
        existing = [
            Entity(id="ent_1", name="Mesa", entity_type="Location"),
            Entity(id="ent_2", name="Mesa", entity_type="Concept"),
        ]
        # Same-type should be preferred
        result = await resolve_entity("Mesa", "Location", existing)
        assert result is not None
        assert result.id == "ent_1"

    async def test_empty_existing(self):
        result = await resolve_entity("Python", "Technology", [])
        assert result is None

    async def test_actr_fuzzy_dedup(self):
        """ACTR should match ACT-R (hyphen normalization + fuzzy)."""
        existing = [
            Entity(id="ent_1", name="ACT-R", entity_type="Concept"),
        ]
        result = await resolve_entity("ACTR", "Concept", existing)
        assert result is not None
        assert result.id == "ent_1"

    async def test_numeric_identifier_near_match_rejected(self):
        existing = [
            Entity(id="ent_1", name="1712018", entity_type="Thing"),
        ]
        result = await resolve_entity("1712061", "Thing", existing)
        assert result is None

    async def test_labeled_identifier_alias_matches(self):
        existing = [
            Entity(id="ent_1", name="1712061", entity_type="Thing"),
        ]
        result = await resolve_entity("SKU 1712061", "Thing", existing)
        assert result is not None
        assert result.id == "ent_1"

    async def test_resolve_entity_fast_blocks_numeric_near_match(self):
        candidate = Entity(id="ent_1", name="1712018", entity_type="Thing")
        get_candidates = AsyncMock(return_value=[candidate])

        result = await resolve_entity_fast("1712061", "Thing", get_candidates, "test")
        assert result is None

    async def test_resolve_entity_fast_allows_labeled_identifier_alias(self):
        candidate = Entity(id="ent_1", name="1712061", entity_type="Thing")
        get_candidates = AsyncMock(return_value=[candidate])

        result = await resolve_entity_fast("Part #1712061", "Thing", get_candidates, "test")
        assert result is not None
        assert result.id == "ent_1"


class TestValidateEntityName:
    def test_rejects_short_names(self):
        assert validate_entity_name("A") is False
        assert validate_entity_name("") is False

    def test_accepts_two_char_names(self):
        assert validate_entity_name("Al") is True

    def test_rejects_long_names(self):
        assert validate_entity_name("This is a really long sentence fragment name") is False

    def test_accepts_normal_length(self):
        assert validate_entity_name("San Francisco Bay Area") is True  # 4 words

    def test_rejects_all_lowercase(self):
        assert validate_entity_name("something") is False
        assert validate_entity_name("just a word") is False

    def test_allows_tech_lowercase_with_dot(self):
        assert validate_entity_name("next.js") is True
        assert validate_entity_name("vue.config.js") is True

    def test_allows_tech_lowercase_with_slash(self):
        assert validate_entity_name("src/utils") is True

    def test_allows_proper_names(self):
        assert validate_entity_name("Alice") is True
        assert validate_entity_name("San Francisco") is True
        assert validate_entity_name("Anthropic") is True
