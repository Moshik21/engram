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

    def test_wrote_maps_to_created(self):
        """Creative-work predicates map to CREATED."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("WROTE") == "CREATED"
        assert c.canonicalize("PUBLISHED") == "CREATED"
        assert c.canonicalize("COMPOSED") == "CREATED"
        assert c.canonicalize("PRODUCED") == "CREATED"
        assert c.canonicalize("FOUNDED") == "CREATED"

    def test_membership_mappings(self):
        """Membership predicates map to MEMBER_OF."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("BELONGS_TO") == "MEMBER_OF"
        assert c.canonicalize("JOINED") == "MEMBER_OF"
        assert c.canonicalize("AFFILIATED_WITH") == "MEMBER_OF"

    def test_health_mappings(self):
        """Health predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("INJURED") == "RECOVERING_FROM"
        assert c.canonicalize("HURT") == "RECOVERING_FROM"
        assert c.canonicalize("DIAGNOSED_WITH") == "HAS_CONDITION"
        assert c.canonicalize("SUFFERS_FROM") == "HAS_CONDITION"
        assert c.canonicalize("TREATING") == "TREATS"
        assert c.canonicalize("PRESCRIBED") == "TREATS"

    def test_sentiment_mappings(self):
        """Sentiment predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("ENJOYS") == "LIKES"
        assert c.canonicalize("LOVES") == "LIKES"
        assert c.canonicalize("APPRECIATES") == "LIKES"
        assert c.canonicalize("HATES") == "DISLIKES"
        assert c.canonicalize("AVOIDS") == "DISLIKES"
        assert c.canonicalize("PREFERS") == "PREFERS"
        assert c.canonicalize("FAVORS") == "PREFERS"

    def test_goals_mappings(self):
        """Goal predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("WANTS_TO") == "AIMS_FOR"
        assert c.canonicalize("PLANS_TO") == "AIMS_FOR"
        assert c.canonicalize("INTENDS_TO") == "AIMS_FOR"
        assert c.canonicalize("ASPIRES_TO") == "AIMS_FOR"
        assert c.canonicalize("WORKING_TOWARD") == "AIMS_FOR"

    def test_causation_mappings(self):
        """Causation predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("CAUSED") == "CAUSED_BY"
        assert c.canonicalize("RESULTED_IN") == "LED_TO"
        assert c.canonicalize("TRIGGERED") == "LED_TO"
        assert c.canonicalize("DEPENDS_ON") == "REQUIRES"
        assert c.canonicalize("NEEDS") == "REQUIRES"
        assert c.canonicalize("BLOCKED_BY") == "REQUIRES"

    def test_hierarchy_mappings(self):
        """Hierarchy predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("CONTAINS") == "HAS_PART"
        assert c.canonicalize("INCLUDES") == "HAS_PART"
        assert c.canonicalize("COMPOSED_OF") == "HAS_PART"
        assert c.canonicalize("PARENT_OF") == "PARENT_OF"
        assert c.canonicalize("CHILD_OF") == "CHILD_OF"

    def test_learning_mappings(self):
        """Learning predicates map correctly."""
        c = PredicateCanonicalizer()
        assert c.canonicalize("LEARNING") == "STUDYING"
        assert c.canonicalize("READING") == "STUDYING"
        assert c.canonicalize("PRACTICING") == "STUDYING"

    def test_all_map_entries_valid(self):
        """All values in CANONICAL_MAP should be uppercase with underscores."""
        for key, value in CANONICAL_MAP.items():
            assert key == key.upper(), f"Key {key} not uppercase"
            assert value == value.upper(), f"Value {value} not uppercase"
            assert " " not in key, f"Key {key} contains spaces"
            assert " " not in value, f"Value {value} contains spaces"
