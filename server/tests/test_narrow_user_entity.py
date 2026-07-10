"""Regression tests: narrow first-person facts must seed a 'User' entity.

Narrow extractors emit first-person/family relationships with subject="User"
but never emit a "User" entity candidate. The EvidenceBridge must synthesize a
canonical "User" Person entity so the relationship endpoint resolves instead of
being dropped as a missing entity.
"""

from engram.extraction.evidence import CommitDecision, EvidenceCandidate
from engram.extraction.evidence_bridge import EvidenceBridge
from engram.extraction.narrow.pipeline import NarrowExtractionPipeline


def _decision(ev: EvidenceCandidate) -> CommitDecision:
    return CommitDecision(
        evidence_id=ev.evidence_id,
        action="commit",
        reason="test",
        effective_confidence=ev.confidence,
    )


def test_first_person_relationship_through_pipeline_seeds_user_entity():
    """'I work at TechCorp.' -> User WORKS_AT TechCorp with a User entity."""
    pipeline = NarrowExtractionPipeline()
    bundle = pipeline.extract("I work at TechCorp.", "ep1", "default")

    # All committed -> bridge to entities/claims.
    committed = [(c, _decision(c)) for c in bundle.candidates]
    entities, claims = EvidenceBridge().bridge(committed)

    works_at = [c for c in claims if c.subject_text == "User" and c.predicate == "WORKS_AT"]
    assert works_at, "expected a User WORKS_AT claim"
    assert works_at[0].object_text == "TechCorp"

    user_entities = [e for e in entities if e.name == "User"]
    assert len(user_entities) == 1
    assert user_entities[0].entity_type == "Person"


def test_user_subject_relationship_emits_user_entity():
    """A bare User-subject relationship gets a synthesized User entity."""
    ev = EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class="relationship",
        confidence=0.8,
        source_type="narrow_extractor",
        extractor_name="relationship_pattern",
        payload={"subject": "User", "predicate": "WORKS_AT", "object": "TechCorp"},
    )
    entities, claims = EvidenceBridge().bridge([(ev, _decision(ev))])

    assert any(e.name == "User" and e.entity_type == "Person" for e in entities)
    assert claims[0].subject_text == "User"


def test_user_entity_is_idempotent_across_multiple_user_claims():
    """Multiple User-subject relationships produce exactly one User entity."""
    evs = [
        EvidenceCandidate(
            episode_id="ep1",
            group_id="default",
            fact_class="relationship",
            confidence=0.85,
            source_type="narrow_extractor",
            extractor_name="relationship_pattern",
            payload={"subject": "User", "predicate": predicate, "object": obj},
        )
        for predicate, obj in (
            ("WORKS_AT", "TechCorp"),
            ("LIVES_IN", "Berlin"),
            ("MARRIED_TO", "Jane"),
        )
    ]
    entities, _ = EvidenceBridge().bridge([(ev, _decision(ev)) for ev in evs])
    assert [e.name for e in entities].count("User") == 1


def test_existing_user_attribute_entity_not_duplicated():
    """If an attribute already produced a User entity, do not add another."""
    attr = EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class="attribute",
        confidence=0.7,
        source_type="narrow_extractor",
        extractor_name="attribute",
        payload={"entity": "User", "attribute_type": "preference", "value": "Python"},
    )
    rel = EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class="relationship",
        confidence=0.8,
        source_type="narrow_extractor",
        extractor_name="relationship_pattern",
        payload={"subject": "User", "predicate": "WORKS_AT", "object": "TechCorp"},
    )
    entities, _ = EvidenceBridge().bridge([(attr, _decision(attr)), (rel, _decision(rel))])
    assert [e.name for e in entities].count("User") == 1


def test_no_user_entity_when_no_user_subject():
    """Non-User claims must not spuriously create a User entity."""
    ev = EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class="relationship",
        confidence=0.85,
        source_type="narrow_extractor",
        extractor_name="relationship_pattern",
        payload={"subject": "Alice", "predicate": "WORKS_AT", "object": "Google"},
    )
    entities, _ = EvidenceBridge().bridge([(ev, _decision(ev))])
    assert not any(e.name == "User" for e in entities)
