"""Tests for predicate canonicalization."""

from engram.extraction.canonicalize import CANONICAL_MAP, PredicateCanonicalizer


class TestPredicateCanonicalizer:
    def test_known_synonym_mapped(self):
        c = PredicateCanonicalizer()
        assert c.canonicalize("EMPLOYED_BY") == "WORKS_AT"
        assert c.canonicalize("WORKS_FOR") == "WORKS_AT"
        assert c.canonicalize("SKILLED_IN") == "EXPERT_IN"
        assert c.canonicalize("LIVES_IN") == "LOCATED_IN"
        assert c.canonicalize("IS_ACQUAINTED_WITH") == "KNOWS"

    def test_unknown_passthrough(self):
        c = PredicateCanonicalizer()
        assert c.canonicalize("UNKNOWN_PRED") == "UNKNOWN_PRED"
        assert c.canonicalize("CUSTOM_RELATION") == "CUSTOM_RELATION"

    def test_case_insensitivity(self):
        c = PredicateCanonicalizer()
        assert c.canonicalize("employed_by") == "WORKS_AT"
        assert c.canonicalize("Employed_By") == "WORKS_AT"
        assert c.canonicalize("EMPLOYED_BY") == "WORKS_AT"

    def test_space_normalization(self):
        c = PredicateCanonicalizer()
        assert c.canonicalize("employed by") == "WORKS_AT"
        assert c.canonicalize("works for") == "WORKS_AT"

    def test_canonical_forms_are_idempotent(self):
        """Canonicalizing a canonical form returns itself."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("WORKS_AT") == "WORKS_AT"
        assert c.canonicalize("EXPERT_IN") == "EXPERT_IN"
        assert c.canonicalize("KNOWS") == "KNOWS"

    def test_extra_mappings(self):
        c = PredicateCanonicalizer(extra_mappings={"LIKES": "PREFERS"})
        assert c.canonicalize("LIKES") == "PREFERS"
        assert c.canonicalize("EMPLOYED_BY") == "WORKS_AT"

    def test_extra_mappings_override(self):
        c = PredicateCanonicalizer(
            extra_mappings={"EMPLOYED_BY": "HIRED_AT"},
        )
        assert c.canonicalize("EMPLOYED_BY") == "HIRED_AT"

    def test_all_map_entries_valid(self):
        """All values in CANONICAL_MAP should be uppercase with underscores."""
        for key, value in CANONICAL_MAP.items():
            assert key == key.upper(), f"Key {key} not uppercase"
            assert value == value.upper(), f"Value {value} not uppercase"
            assert " " not in key, f"Key {key} contains spaces"
            assert " " not in value, f"Value {value} contains spaces"
