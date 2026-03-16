"""Tests for LLM-free query decomposition."""

from __future__ import annotations

import asyncio
from collections import OrderedDict

from engram.retrieval.decomposer import (
    _decompose_deterministic,
    _extract_clauses,
    _extract_noun_phrases,
    decompose_query,
    needs_decomposition,
)

# ---------------------------------------------------------------------------
# needs_decomposition
# ---------------------------------------------------------------------------

class TestNeedsDecomposition:
    def test_simple_query(self):
        assert not needs_decomposition("What is my favorite color?")

    def test_simple_who_query(self):
        assert not needs_decomposition("Who is my manager?")

    def test_duration_between(self):
        assert needs_decomposition("How many days between starting the project and finishing it?")

    def test_how_long_between(self):
        assert needs_decomposition("How long between my trip to Paris and my trip to London?")

    def test_which_came_first(self):
        assert needs_decomposition("Which came first, learning Python or learning Rust?")

    def test_before_pattern(self):
        assert needs_decomposition("Before I moved to NYC, what was I working on?")

    def test_after_pattern(self):
        assert needs_decomposition("After my promotion, did I change teams?")

    def test_versus_question(self):
        assert needs_decomposition("Python or Rust?")

    def test_what_changed(self):
        assert needs_decomposition("What changed about my workout routine?")

    def test_compared_to(self):
        assert needs_decomposition("My Python skills compared to my Rust skills")

    def test_both_and(self):
        assert needs_decomposition("Both my morning routine and my evening routine")


# ---------------------------------------------------------------------------
# Strategy 1: Template matching
# ---------------------------------------------------------------------------

class TestTemplateDecomposition:
    def test_days_between(self):
        result = _decompose_deterministic(
            "How many days between starting the project and finishing it?"
        )
        assert len(result) == 2
        assert "starting the project" in result[0]
        assert "finishing it" in result[1]

    def test_how_long_between(self):
        result = _decompose_deterministic(
            "How long between my trip to Paris and my trip to London?"
        )
        assert len(result) == 2
        assert "Paris" in result[0]
        assert "London" in result[1]

    def test_how_many_months_between(self):
        result = _decompose_deterministic(
            "How many months between starting at Google and leaving?"
        )
        assert len(result) == 2

    def test_how_long_since(self):
        result = _decompose_deterministic(
            "How long since I started learning Rust?"
        )
        assert len(result) == 1
        assert "Rust" in result[0]

    def test_how_many_days_since(self):
        result = _decompose_deterministic(
            "How many days since my last workout?"
        )
        assert len(result) == 1
        assert "workout" in result[0]

    def test_which_came_first(self):
        result = _decompose_deterministic(
            "Which came first, learning Python or learning Rust?"
        )
        assert len(result) == 2
        assert "Python" in result[0]
        assert "Rust" in result[1]

    def test_which_happened_earlier(self):
        result = _decompose_deterministic(
            "Which happened earlier, the merger or the acquisition?"
        )
        assert len(result) == 2

    def test_did_i_before(self):
        result = _decompose_deterministic(
            "Did I start running before joining the gym?"
        )
        assert len(result) == 2
        assert "running" in result[0].lower() or "running" in result[1].lower()

    def test_before_comma_clause(self):
        result = _decompose_deterministic(
            "Before I moved to NYC, what was I working on?"
        )
        assert len(result) == 2
        assert "NYC" in result[0]
        assert "working" in result[1]

    def test_after_comma_clause(self):
        result = _decompose_deterministic(
            "After the project ended, did I start something new?"
        )
        assert len(result) == 2

    def test_what_changed_from_to(self):
        result = _decompose_deterministic(
            "What changed from version 1 to version 2?"
        )
        assert len(result) == 2
        assert "version 1" in result[0]
        assert "version 2" in result[1]

    def test_what_changed_about(self):
        result = _decompose_deterministic(
            "What changed about my diet?"
        )
        assert len(result) == 1
        assert "diet" in result[0].lower()

    def test_versus(self):
        result = _decompose_deterministic(
            "Python versus Rust for backend development"
        )
        assert len(result) == 2
        assert "Python" in result[0]
        assert "Rust" in result[1]

    def test_vs_dot(self):
        result = _decompose_deterministic(
            "React vs. Vue for the frontend?"
        )
        assert len(result) == 2

    def test_compared_to(self):
        result = _decompose_deterministic(
            "My old apartment compared to my new one"
        )
        assert len(result) == 2

    def test_both_and(self):
        result = _decompose_deterministic(
            "Both my morning routine and my evening routine"
        )
        assert len(result) == 2
        assert "morning" in result[0]
        assert "evening" in result[1]


# ---------------------------------------------------------------------------
# Strategy 2: Clause extraction
# ---------------------------------------------------------------------------

class TestClauseExtraction:
    def test_and_what(self):
        result = _extract_clauses(
            "What is my favorite programming language, and what projects am I working on?"
        )
        assert len(result) == 2

    def test_but_did(self):
        result = _extract_clauses(
            "I mentioned liking Python, but did I ever talk about Rust?"
        )
        assert len(result) >= 2

    def test_single_clause_no_split(self):
        result = _extract_clauses("What is my favorite color?")
        assert result == []

    def test_too_short_clause_filtered(self):
        # "x, and y" where x or y is too short
        result = _extract_clauses("yes, and what is my name?")
        assert result == []  # "yes" is only 1 word, filtered


# ---------------------------------------------------------------------------
# Strategy 3: Noun-phrase extraction
# ---------------------------------------------------------------------------

class TestNounPhraseExtraction:
    def test_two_noun_phrases(self):
        result = _extract_noun_phrases(
            "Tell me about my Python projects and my Rust experiments"
        )
        assert len(result) >= 2

    def test_proper_nouns(self):
        result = _extract_noun_phrases(
            "What is the connection between New York and San Francisco in my notes?"
        )
        # Should find "New York" and "San Francisco" as proper noun sequences
        found_ny = any("New York" in np for np in result)
        found_sf = any("San Francisco" in np for np in result)
        assert found_ny or found_sf or len(result) >= 2

    def test_single_np_not_enough(self):
        result = _extract_noun_phrases("What is my favorite color?")
        # Only one NP ("my favorite color") — not enough to decompose
        assert len(result) < 2

    def test_stop_phrases_filtered(self):
        result = _extract_noun_phrases("The first time and the last time")
        # "the first" and "the last" are stop phrases
        for np in result:
            assert np.lower() not in {"the first", "the last"}


# ---------------------------------------------------------------------------
# Full decompose_query (async entry point)
# ---------------------------------------------------------------------------

class TestDecomposeQuery:
    def test_simple_query_returns_original(self):
        result = asyncio.get_event_loop().run_until_complete(
            decompose_query("What is my favorite color?")
        )
        assert result == ["What is my favorite color?"]

    def test_complex_query_decomposed(self):
        result = asyncio.get_event_loop().run_until_complete(
            decompose_query("How many days between starting Python and starting Rust?")
        )
        assert len(result) == 2

    def test_cache_hit(self):
        cache: OrderedDict = OrderedDict()
        query = "How many days between event A and event B?"

        # First call populates cache
        result1 = asyncio.get_event_loop().run_until_complete(
            decompose_query(query, cache=cache)
        )
        assert query in cache

        # Second call hits cache
        result2 = asyncio.get_event_loop().run_until_complete(
            decompose_query(query, cache=cache)
        )
        assert result1 == result2

    def test_cache_eviction(self):
        from engram.retrieval.decomposer import DECOMPOSE_CACHE_SIZE

        cache: OrderedDict = OrderedDict()
        # Fill cache beyond limit
        for i in range(DECOMPOSE_CACHE_SIZE + 5):
            q = f"How many days between event {i} and event {i + 1}?"
            asyncio.get_event_loop().run_until_complete(
                decompose_query(q, cache=cache)
            )
        assert len(cache) <= DECOMPOSE_CACHE_SIZE

    def test_model_param_ignored(self):
        """model= is accepted for backward compatibility but ignored."""
        result = asyncio.get_event_loop().run_until_complete(
            decompose_query(
                "How many days between X and Y?",
                model="some-model-that-doesnt-exist",
            )
        )
        assert len(result) == 2

    def test_no_llm_import(self):
        """Verify the decomposer never imports anthropic."""
        import inspect

        import engram.retrieval.decomposer as mod

        source = inspect.getsource(mod)
        # The old _DECOMPOSE_PROMPT is kept for compat but no "import anthropic"
        assert "import anthropic" not in source


# ---------------------------------------------------------------------------
# End-to-end: realistic queries
# ---------------------------------------------------------------------------

class TestRealisticQueries:
    """Test with realistic user queries to verify decomposition quality."""

    def test_temporal_gap(self):
        result = _decompose_deterministic(
            "How many days between my dentist appointment and my doctor visit?"
        )
        assert len(result) == 2
        assert any("dentist" in r for r in result)
        assert any("doctor" in r for r in result)

    def test_ordering_question(self):
        result = _decompose_deterministic(
            "Which happened first, moving to San Francisco or starting at Anthropic?"
        )
        assert len(result) == 2

    def test_conditional_temporal(self):
        result = _decompose_deterministic(
            "After I finished the Engram project, what did I start working on?"
        )
        assert len(result) == 2

    def test_simple_direct_lookup_passthrough(self):
        result = _decompose_deterministic("What is my dog's name?")
        assert result == ["What is my dog's name?"]

    def test_how_long_since_event(self):
        result = _decompose_deterministic(
            "How long since I switched to HelixDB?"
        )
        assert len(result) >= 1
        assert any("HelixDB" in r for r in result)
