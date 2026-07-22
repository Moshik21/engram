"""Durable types (Decision/Preference/Commitment/Correction/Goal) reachable by local extraction."""

import re

import pytest

from engram.extraction.narrow.entity_extractor import IdentityEntityExtractor
from engram.extraction.narrow_adapter import NarrowExtractorAdapter
from engram.extraction.prompts import EXTRACTION_SYSTEM_PROMPT
from engram.extraction.resolver import validate_entity_name

DURABLE_TYPES = {"Decision", "Preference", "Commitment", "Correction", "Goal"}


@pytest.fixture
def extractor():
    return IdentityEntityExtractor()


def _by_type(candidates, entity_type):
    return [c for c in candidates if c.payload.get("entity_type") == entity_type]


class TestPromptVocabulary:
    def test_entity_type_union_contains_durable_types(self):
        match = re.search(r'"entity_type": "([^"]+)"', EXTRACTION_SYSTEM_PROMPT)
        assert match is not None
        vocabulary = set(match.group(1).split("|"))
        assert DURABLE_TYPES <= vocabulary

    def test_guidance_line_per_durable_type(self):
        for entity_type in DURABLE_TYPES:
            assert f"- {entity_type}:" in EXTRACTION_SYSTEM_PROMPT


class TestDecisionExtraction:
    def test_we_decided(self, extractor):
        results = extractor.extract(
            "We decided to use PostgreSQL for the new reporting service.",
            "ep1",
            "default",
        )
        decisions = _by_type(results, "Decision")
        assert len(decisions) == 1
        assert decisions[0].payload["name"] == "use PostgreSQL for the new reporting service"
        assert decisions[0].confidence == 0.60
        assert "decision_statement" in decisions[0].corroborating_signals

    def test_decision_prefix(self, extractor):
        results = extractor.extract(
            "Decision: migrate the billing cron to the new runner.",
            "ep1",
            "default",
        )
        decisions = _by_type(results, "Decision")
        assert len(decisions) == 1
        assert decisions[0].payload["name"] == "migrate the billing cron to the new runner"

    def test_question_does_not_fire(self, extractor):
        results = extractor.extract(
            "Should we decide on the database this week?",
            "ep1",
            "default",
        )
        assert _by_type(results, "Decision") == []

    def test_question_with_trigger_does_not_fire(self, extractor):
        results = extractor.extract(
            "Do you think we decided to use tabs everywhere?",
            "ep1",
            "default",
        )
        assert _by_type(results, "Decision") == []

    def test_negation_does_not_fire(self, extractor):
        results = extractor.extract(
            "We never decided to migrate the database.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Decision") == []

    def test_overlong_clause_does_not_fire(self, extractor):
        clause = " ".join(f"word{i}" for i in range(30))
        results = extractor.extract(f"We decided to {clause}.", "ep1", "default")
        assert _by_type(results, "Decision") == []


class TestPreferenceExtraction:
    def test_i_prefer(self, extractor):
        results = extractor.extract(
            "I prefer tabs over spaces for indentation.",
            "ep1",
            "default",
        )
        prefs = _by_type(results, "Preference")
        assert len(prefs) == 1
        assert prefs[0].payload["name"] == "tabs over spaces for indentation"
        assert prefs[0].confidence == 0.60
        assert "preference_statement" in prefs[0].corroborating_signals

    def test_i_like_x_over_y(self, extractor):
        results = extractor.extract(
            "I like Vim better than Emacs for quick edits.",
            "ep1",
            "default",
        )
        prefs = _by_type(results, "Preference")
        assert len(prefs) == 1
        assert "Vim better than Emacs" in prefs[0].payload["name"]

    def test_always_use_requires_first_person(self, extractor):
        # First-person anchored: fires.
        results = extractor.extract(
            "We always use absolute paths in the agent scripts.",
            "ep1",
            "default",
        )
        prefs = _by_type(results, "Preference")
        assert len(prefs) == 1
        assert prefs[0].payload["name"] == "absolute paths in the agent scripts"
        # Generic advice without a first-person anchor: must NOT fire.
        results = extractor.extract(
            "Always use parameterized queries to avoid SQL injection.",
            "ep2",
            "default",
        )
        assert _by_type(results, "Preference") == []

    def test_negated_prefer_does_not_fire(self, extractor):
        results = extractor.extract(
            "I don't prefer tabs over spaces.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Preference") == []

    def test_negated_always_use_does_not_fire(self, extractor):
        results = extractor.extract(
            "You shouldn't always use mocks in tests.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Preference") == []


class TestCommitmentExtraction:
    def test_ill_make_sure(self, extractor):
        results = extractor.extract(
            "I'll make sure the deploy scripts get updated tomorrow.",
            "ep1",
            "default",
        )
        commitments = _by_type(results, "Commitment")
        assert len(commitments) == 1
        assert commitments[0].payload["name"] == "the deploy scripts get updated tomorrow"
        assert commitments[0].confidence == 0.60
        assert "commitment_statement" in commitments[0].corroborating_signals

    def test_i_promise(self, extractor):
        results = extractor.extract(
            "I promise to review the pull request tonight.",
            "ep1",
            "default",
        )
        commitments = _by_type(results, "Commitment")
        assert len(commitments) == 1
        assert commitments[0].payload["name"] == "review the pull request tonight"

    def test_negated_will_does_not_fire(self, extractor):
        results = extractor.extract(
            "I will not forget the standup.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Commitment") == []

    def test_second_person_promise_question_does_not_fire(self, extractor):
        results = extractor.extract(
            "Will you make sure the tests pass?",
            "ep1",
            "default",
        )
        assert _by_type(results, "Commitment") == []


class TestCorrectionExtraction:
    def test_actually_its_x_not_y(self, extractor):
        results = extractor.extract(
            "Actually, it's Denver not Mesa where we live now.",
            "ep1",
            "default",
        )
        corrections = _by_type(results, "Correction")
        assert len(corrections) == 1
        assert corrections[0].payload["name"] == "Denver not Mesa where we live now"
        assert corrections[0].confidence == 0.60
        assert "correction_statement" in corrections[0].corroborating_signals

    def test_correction_prefix(self, extractor):
        results = extractor.extract(
            "Correction: the launch is on Tuesday not Monday.",
            "ep1",
            "default",
        )
        corrections = _by_type(results, "Correction")
        assert len(corrections) == 1
        assert corrections[0].payload["name"] == "the launch is on Tuesday not Monday"

    def test_actually_without_contrast_does_not_fire(self, extractor):
        results = extractor.extract(
            "Actually, it's a lovely day outside.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Correction") == []


class TestGoalExtraction:
    def test_my_goal_is(self, extractor):
        results = extractor.extract(
            "My goal is to ship the native backend by August.",
            "ep1",
            "default",
        )
        goals = _by_type(results, "Goal")
        assert len(goals) == 1
        assert goals[0].payload["name"] == "ship the native backend by August"
        assert goals[0].confidence == 0.60
        assert "goal_statement" in goals[0].corroborating_signals

    def test_were_aiming(self, extractor):
        results = extractor.extract(
            "We're aiming for a beta release next month.",
            "ep1",
            "default",
        )
        goals = _by_type(results, "Goal")
        assert len(goals) == 1
        assert goals[0].payload["name"] == "a beta release next month"

    def test_negated_goal_does_not_fire(self, extractor):
        results = extractor.extract(
            "My goal is not to rewrite everything.",
            "ep1",
            "default",
        )
        assert _by_type(results, "Goal") == []

    def test_goal_question_does_not_fire(self, extractor):
        results = extractor.extract(
            "What do you think my goal is?",
            "ep1",
            "default",
        )
        assert _by_type(results, "Goal") == []


class TestDurableNameValidation:
    def test_long_statement_name_allowed_for_durable_types(self):
        name = "Use PostgreSQL for the new reporting service instead of SQLite"
        assert validate_entity_name(name, entity_type="Decision")
        # Non-durable types keep the 5-word fragment cap
        assert not validate_entity_name(name)


class TestEndToEndNarrowPipeline:
    async def test_natural_sentence_yields_decision_entity(self):
        """Organic capture: a Decision entity with zero client proposals."""
        adapter = NarrowExtractorAdapter()
        bundle = adapter._pipeline.extract(
            "We decided to use PostgreSQL for the new reporting service.",
            episode_id="ep1",
            group_id="default",
        )
        assert all(c.source_type == "narrow_extractor" for c in bundle.candidates)

        result = await adapter.extract(
            "We decided to use PostgreSQL for the new reporting service."
        )
        assert not result.is_error
        decisions = [e for e in result.entities if e["entity_type"] == "Decision"]
        assert len(decisions) == 1
        # M1.4 squatter guard (P4: names are identifiers, not content): the
        # commit policy caps entity names at 6 tokens; the full clause folds
        # into the summary, so nothing is lost — just relocated.
        assert decisions[0]["name"] == "use PostgreSQL for the new reporting"
        assert "use PostgreSQL for the new reporting service" in decisions[0]["summary"]
