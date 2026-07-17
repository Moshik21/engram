"""Commit-time entity_type case normalization (M2.2).

entity_type is an open case-sensitive string; every typed behavior (durable
boost, packets, merge same-type gate, prune identity checks) compares exactly.
These tests pin the commit choke point: lowercase extractor output gets the
canonical TitleCase vocabulary before storage, unknown types are
preserved-but-cased, and stored rows are never rewritten.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.entity_dedup_policy import (
    canonicalize_entity_type_case,
    normalize_extracted_entity_type,
)
from engram.extraction.apply import ApplyEngine
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.models import EntityCandidate
from engram.extraction.promotion import is_high_signal_entity_type
from engram.models.episode import Episode


class TestCanonicalizeEntityTypeCase:
    def test_lowercase_known_type_maps_to_canonical(self):
        assert canonicalize_entity_type_case("person") == "Person"
        assert canonicalize_entity_type_case("decision") == "Decision"

    def test_uppercase_known_type_maps_to_canonical(self):
        assert canonicalize_entity_type_case("ORGANIZATION") == "Organization"

    def test_multiword_canonical_type_matches_case_insensitively(self):
        assert canonicalize_entity_type_case("preferenceprofile") == "PreferenceProfile"
        assert canonicalize_entity_type_case("clarificationintent") == "ClarificationIntent"

    def test_canonical_input_is_untouched(self):
        assert canonicalize_entity_type_case("Person") == "Person"

    def test_unknown_type_first_letter_cased_rest_preserved(self):
        assert canonicalize_entity_type_case("quantumWidget") == "QuantumWidget"
        assert canonicalize_entity_type_case("WidgetFOO") == "WidgetFOO"

    def test_missing_type_defaults_to_other(self):
        assert canonicalize_entity_type_case(None) == "Other"
        assert canonicalize_entity_type_case("  ") == "Other"


class TestNormalizeExtractedEntityType:
    def test_lowercase_type_is_canonicalized(self):
        entity_type, _ = normalize_extracted_entity_type("Konner Moshier", "person")
        assert entity_type == "Person"

    def test_lowercase_coercible_type_still_reaches_identifier_coercion(self):
        # "technology" only matches the coercible set after case normalization.
        entity_type, _ = normalize_extracted_entity_type("SKU-4471-A", "technology")
        assert entity_type == "Identifier"


@pytest.mark.asyncio
async def test_lowercase_person_from_extractor_gets_person_semantics():
    """Lowercase 'person' commits as canonical Person with durable semantics."""
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_case", content="Met Konner Moshier.", group_id="default")

    outcome = await engine.apply_entities(
        [EntityCandidate(name="Konner Moshier", entity_type="person")],
        episode,
        "default",
    )

    assert outcome.new_entity_names == ["Konner Moshier"]
    created = graph.create_entity.call_args[0][0]
    assert created.entity_type == "Person"
    # Typed semantics the raw lowercase string never had.
    assert is_high_signal_entity_type(created.entity_type)
    assert not is_high_signal_entity_type("person")


@pytest.mark.asyncio
async def test_lowercase_decision_gets_durable_word_limit_semantics():
    """A statement-length 'decision' name passes the durable-type word limit."""
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_dec", content="Decision recorded.", group_id="default")

    # 8 words — over the default 5-word limit, allowed only for durable types.
    name = "Use SQLite for the disposable smoke demo backend"
    outcome = await engine.apply_entities(
        [EntityCandidate(name=name, entity_type="decision")],
        episode,
        "default",
    )

    assert outcome.new_entity_names == [name]
    created = graph.create_entity.call_args[0][0]
    assert created.entity_type == "Decision"


@pytest.mark.asyncio
async def test_mixed_case_unknown_type_preserved_but_cased():
    graph = AsyncMock()
    graph.find_entity_candidates = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_entity = AsyncMock()

    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )
    episode = Episode(id="ep_unknown", content="Gizmo spotted.", group_id="default")

    outcome = await engine.apply_entities(
        [EntityCandidate(name="Gizmo", entity_type="quantumWidget")],
        episode,
        "default",
    )

    assert outcome.new_entity_names == ["Gizmo"]
    created = graph.create_entity.call_args[0][0]
    assert created.entity_type == "QuantumWidget"
