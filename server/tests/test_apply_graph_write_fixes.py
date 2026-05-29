"""Regression tests for graph write-path fixes in the apply engine.

Covers two confirmed bugs:
  1. D1 auto-create endpoints — drop->create swap gated on
     cfg.graph_auto_create_endpoints instead of silently dropping the edge.
  2. committed_id_map positional zip — apply_relationships now appends a
     sentinel for meta-skipped claims so evidence_ids map to the right
     relationship_id instead of being shifted onto the wrong row.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.apply import ApplyEngine, apply_relationship_fact
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.evidence import CommitDecision, EvidenceCandidate
from engram.extraction.models import ClaimCandidate
from engram.ingestion.adjudication_service import EvidenceAdjudicationService
from engram.models.entity import Entity


def _make_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.find_conflicting_relationships = AsyncMock(return_value=[])
    graph.find_existing_relationship = AsyncMock(return_value=None)
    graph.create_relationship = AsyncMock(return_value="rel_persisted")
    graph.create_entity = AsyncMock()
    graph.update_entity = AsyncMock()
    return graph


@pytest.mark.asyncio
async def test_auto_create_endpoint_creates_provisional_entity_when_flag_enabled():
    graph = _make_graph()
    entity_map = {"User": "ent_user"}

    result = await apply_relationship_fact(
        graph_store=graph,
        canonicalizer=PredicateCanonicalizer(),
        cfg=ActivationConfig(graph_auto_create_endpoints=True),
        rel_data={
            "source": "User",
            "target": "Acme",
            "predicate": "WORKS_AT",
        },
        entity_map=entity_map,
        group_id="default",
        source_episode="ep_d1",
    )

    assert result.action == "created"
    # The missing endpoint is now provisioned and threaded back into entity_map.
    assert "Acme" in entity_map
    created = graph.create_entity.await_args.args[0]
    assert isinstance(created, Entity)
    assert created.name == "Acme"
    assert created.entity_type == "Concept"
    assert created.evidence_count == 1
    assert (created.attributes or {}).get("provisional_endpoint") is True
    # The created endpoint is reported honestly so the drop->create swap shows up.
    assert result.metadata.get("auto_created_endpoints") == ["Acme"]


@pytest.mark.asyncio
async def test_missing_endpoint_dropped_when_flag_disabled():
    graph = _make_graph()
    entity_map = {"User": "ent_user"}

    result = await apply_relationship_fact(
        graph_store=graph,
        canonicalizer=PredicateCanonicalizer(),
        cfg=ActivationConfig(graph_auto_create_endpoints=False),
        rel_data={
            "source": "User",
            "target": "Acme",
            "predicate": "WORKS_AT",
        },
        entity_map=entity_map,
        group_id="default",
        source_episode="ep_d1",
    )

    assert result.action == "missing_entities"
    assert "Acme" not in entity_map
    graph.create_entity.assert_not_called()
    graph.create_relationship.assert_not_called()


@pytest.mark.asyncio
async def test_committed_id_map_aligns_after_meta_skip():
    graph = _make_graph()
    engine = ApplyEngine(
        graph_store=graph,
        activation_store=AsyncMock(),
        cfg=ActivationConfig(),
        canonicalizer=PredicateCanonicalizer(),
    )

    claim_a = ClaimCandidate(
        subject_text="Alex",
        object_text="Phoenix",
        predicate="lives in",
        raw_payload={
            "source": "Alex",
            "target": "Phoenix",
            "predicate": "lives in",
            "evidence_id": "ev_a",
        },
    )
    # B's subject is a meta entity -> apply_relationships skips it. The sentinel
    # keeps results positionally aligned with claims.
    claim_b = ClaimCandidate(
        subject_text="MetaThing",
        object_text="Phoenix",
        predicate="relates to",
        raw_payload={
            "source": "MetaThing",
            "target": "Phoenix",
            "predicate": "relates to",
            "evidence_id": "ev_b",
        },
    )

    results = await engine.apply_relationships(
        [claim_a, claim_b],
        entity_map={"Alex": "ent_alex", "Phoenix": "ent_phoenix"},
        meta_entity_names={"MetaThing"},
        group_id="default",
        source_episode="ep_align",
    )

    # One result per claim, positionally aligned.
    assert len(results) == 2
    assert results[0].created is True
    assert results[1].action == "skipped_meta"

    a_rel_id = results[0].metadata["relationship_id"]

    evidence_pairs: list[tuple[EvidenceCandidate, CommitDecision]] = []
    committed = EvidenceAdjudicationService.committed_id_map(
        evidence_pairs,
        entity_map={"Alex": "ent_alex", "Phoenix": "ent_phoenix"},
        claims=[claim_a, claim_b],
        relationship_results=results,
    )

    # ev_a must map to A's relationship, not B's (which was skipped).
    assert committed["ev_a"] == a_rel_id
    assert "ev_b" not in committed
