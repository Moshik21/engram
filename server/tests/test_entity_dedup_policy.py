"""Tests for identifier-aware entity dedup policy."""

from engram.entity_dedup_policy import (
    NameRegime,
    analyze_name,
    dedup_policy,
    normalize_extracted_entity_type,
)


class TestAnalyzeName:
    def test_numeric_identifier(self):
        form = analyze_name("1712061")
        assert form.regime == NameRegime.IDENTIFIER
        assert form.canonical_code == "1712061"

    def test_labeled_identifier(self):
        form = analyze_name("SKU 1712061")
        assert form.regime == NameRegime.IDENTIFIER
        assert form.canonical_code == "1712061"
        assert form.has_identifier_label is True

    def test_hybrid_identifier_phrase(self):
        form = analyze_name("Model AB-1712061-C bracket")
        assert form.regime == NameRegime.HYBRID
        assert form.canonical_code == "ab1712061c"

    def test_year_like_phrase_stays_natural_language(self):
        form = analyze_name("2024 roadmap")
        assert form.regime == NameRegime.NATURAL_LANGUAGE
        assert form.canonical_code is None

    def test_numeronym_stays_natural_language(self):
        form = analyze_name("k8s")
        assert form.regime == NameRegime.NATURAL_LANGUAGE
        assert form.canonical_code is None


class TestDedupPolicy:
    def test_distinct_numeric_identifiers_rejected(self):
        decision = dedup_policy("1712061", "1712018")
        assert decision.allowed is False
        assert decision.reason == "identifier_mismatch"

    def test_labeled_identifier_alias_allowed(self):
        decision = dedup_policy("1712061", "SKU 1712061")
        assert decision.allowed is True
        assert decision.exact_identifier_match is True
        assert decision.reason == "identifier_exact_match"

    def test_hybrid_model_codes_require_exact_match(self):
        decision = dedup_policy("RTX 4090", "RTX 4080")
        assert decision.allowed is False
        assert decision.reason == "hybrid_code_mismatch"

    def test_natural_language_aliases_fall_back(self):
        decision = dedup_policy("k8s", "kubernetes")
        assert decision.allowed is True
        assert decision.exact_identifier_match is False
        assert decision.reason == "natural_language_fallback"

    def test_leading_zeros_preserved(self):
        decision = dedup_policy("Part #001234", "1234")
        assert decision.allowed is False
        assert decision.reason == "identifier_mismatch"


class TestEntityTypeNormalization:
    def test_code_like_other_type_promotes_to_identifier(self):
        entity_type, form = normalize_extracted_entity_type("1712061", "Other")
        assert entity_type == "Identifier"
        assert form.regime == NameRegime.IDENTIFIER

    def test_natural_language_name_keeps_original_type(self):
        entity_type, form = normalize_extracted_entity_type("iPhone 15 Pro", "Technology")
        assert entity_type == "Technology"
        assert form.regime == NameRegime.NATURAL_LANGUAGE
