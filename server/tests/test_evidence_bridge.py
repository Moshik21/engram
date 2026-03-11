"""Tests for EvidenceBridge."""

import pytest

from engram.extraction.evidence import CommitDecision, EvidenceCandidate
from engram.extraction.evidence_bridge import EvidenceBridge


def _ev(
    fact_class: str,
    payload: dict,
    confidence: float = 0.85,
    source_span: str | None = None,
) -> EvidenceCandidate:
    return EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class=fact_class,
        confidence=confidence,
        source_type="narrow_extractor",
        extractor_name="test",
        payload=payload,
        source_span=source_span,
        corroborating_signals=["test_signal"],
    )


def _decision(ev: EvidenceCandidate) -> CommitDecision:
    return CommitDecision(
        evidence_id=ev.evidence_id,
        action="commit",
        reason="test",
        effective_confidence=ev.confidence,
    )


@pytest.fixture
def bridge():
    return EvidenceBridge()


class TestEvidenceBridge:
    def test_entity_to_entity_candidate(self, bridge):
        ev = _ev(
            "entity",
            {"name": "Alex", "entity_type": "Person"},
            source_span="My name is Alex.",
        )
        entities, claims = bridge.bridge([(ev, _decision(ev))])
        assert len(entities) == 1
        assert entities[0].name == "Alex"
        assert entities[0].entity_type == "Person"
        assert entities[0].summary is not None

    def test_relationship_to_claim(self, bridge):
        ev = _ev(
            "relationship",
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
        )
        entities, claims = bridge.bridge([(ev, _decision(ev))])
        assert len(claims) == 1
        assert claims[0].subject_text == "Alice"
        assert claims[0].predicate == "WORKS_AT"
        assert claims[0].object_text == "Google"
        assert claims[0].raw_payload["source"] == "Alice"
        assert claims[0].raw_payload["target"] == "Google"
        assert claims[0].raw_payload["predicate"] == "WORKS_AT"
        assert claims[0].raw_payload["confidence"] == pytest.approx(0.85)
        assert claims[0].raw_payload["temporal_evidence_ids"] == []

    def test_attribute_to_entity_with_attributes(self, bridge):
        ev = _ev(
            "attribute",
            {"entity": "User", "attribute_type": "preference", "value": "Python"},
        )
        entities, claims = bridge.bridge([(ev, _decision(ev))])
        assert len(entities) == 1
        assert entities[0].name == "User"
        assert entities[0].attributes == {"preference": "Python"}

    def test_attribute_without_entity_skipped(self, bridge):
        ev = _ev("attribute", {"attribute_type": "quantity", "value": 500})
        entities, claims = bridge.bridge([(ev, _decision(ev))])
        assert len(entities) == 0

    def test_temporal_does_not_produce_entity(self, bridge):
        ev = _ev(
            "temporal",
            {"temporal_marker": "yesterday", "nearby_entity": "Alice"},
        )
        entities, claims = bridge.bridge([(ev, _decision(ev))])
        assert len(entities) == 0
        assert len(claims) == 0

    def test_temporal_hint_attached_to_claim(self, bridge):
        ev_rel = _ev(
            "relationship",
            {"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
        )
        ev_temp = _ev(
            "temporal",
            {"temporal_marker": "2026-01-15", "nearby_entity": "Alice"},
        )
        entities, claims = bridge.bridge([
            (ev_rel, _decision(ev_rel)),
            (ev_temp, _decision(ev_temp)),
        ])
        assert len(claims) == 1
        assert claims[0].temporal_hint == "2026-01-15"
        assert ev_temp.evidence_id in claims[0].raw_payload["temporal_evidence_ids"]

    def test_empty_input(self, bridge):
        entities, claims = bridge.bridge([])
        assert entities == []
        assert claims == []

    def test_mixed_evidence(self, bridge):
        ev1 = _ev("entity", {"name": "Alex", "entity_type": "Person"})
        ev2 = _ev(
            "relationship",
            {"subject": "Alex", "predicate": "WORKS_AT", "object": "Anthropic"},
        )
        ev3 = _ev(
            "attribute",
            {"entity": "Alex", "attribute_type": "role", "value": "engineer"},
        )
        committed = [(ev, _decision(ev)) for ev in [ev1, ev2, ev3]]
        entities, claims = bridge.bridge(committed)
        assert len(entities) == 2  # entity + attribute
        assert len(claims) == 1

    def test_entity_raw_payload_has_evidence_id(self, bridge):
        ev = _ev("entity", {"name": "Test", "entity_type": "Concept"})
        entities, _ = bridge.bridge([(ev, _decision(ev))])
        assert "evidence_id" in entities[0].raw_payload

    def test_claim_polarity_preserved(self, bridge):
        ev = _ev(
            "relationship",
            {
                "subject": "User",
                "predicate": "WORKS_AT",
                "object": "Google",
                "polarity": "negative",
            },
        )
        _, claims = bridge.bridge([(ev, _decision(ev))])
        assert claims[0].polarity == "negative"

    def test_extractive_summary(self, bridge):
        ev = _ev(
            "entity",
            {"name": "Alice", "entity_type": "Person"},
            source_span="Alice is a software engineer. She works at Google.",
        )
        entities, _ = bridge.bridge([(ev, _decision(ev))])
        assert "Alice" in (entities[0].summary or "")
