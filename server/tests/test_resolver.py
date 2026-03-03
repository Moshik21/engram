"""Tests for fuzzy entity deduplication resolver."""

import pytest

from engram.extraction.resolver import compute_similarity, normalize_name, resolve_entity
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
