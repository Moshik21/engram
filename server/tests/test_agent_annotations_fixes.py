"""Regression tests for cluster A1: client-proposal foundation (B1) + agent
annotation enrichment / trust hardening (feature #8).

Covered defects:
- B1: remember()/observe() with proposed entities + a relationship between them
  must persist both the entities AND the edge (previously the edge was silently
  dropped/deferred while metered as success).
- Trust de-weaponization: an unverified single-source proposal must NOT
  first-sight-commit, even at a high caller model_tier.
- Span validator: a claim whose source_span is not in the episode content is
  tagged 'span_unverified' and deferred (never rejected).
- Date re-anchoring: relative temporal hints resolve against conversation_date,
  not wall-clock ingest time; absolute+relative disagreement is tagged
  'date_conflict' and deferred.
- Events: dated annotations materialize as first-class Event nodes with an
  OCCURRED_ON edge dated to the event date (remember path); observe persists them
  cheaply as deferred evidence without projecting.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from types import SimpleNamespace

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.extraction.apply import apply_relationship_fact
from engram.extraction.canonicalize import PredicateCanonicalizer
from engram.extraction.client_proposals import (
    SPAN_VERIFIED_PROPOSAL_CONFIDENCE,
    UNVERIFIED_PROPOSAL_CONFIDENCE_CAP,
    events_to_proposals,
    proposals_to_evidence,
    span_is_verified,
)
from engram.extraction.commit_policy import AdaptiveCommitPolicy, CommitThresholds
from engram.extraction.evidence import EvidenceBundle
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.ingestion.capture_surface import (
    build_mcp_observe_write_surface,
    build_mcp_remember_write_surface,
    merge_event_proposals,
)
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex


class _EmptyExtractor(EntityExtractor):
    """Extractor that returns nothing — proposals are the only structure source."""

    def __init__(self) -> None:
        self._result = ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


@pytest_asyncio.fixture
async def proposal_manager():
    """Lite GraphManager with client proposals enabled."""
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "annotations.db")
    graph_store = SQLiteGraphStore(db_path)
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    search_index = FTS5SearchIndex(db_path)
    await search_index.initialize(db=graph_store._db)
    cfg = ActivationConfig(evidence_client_proposals_enabled=True)
    manager = GraphManager(
        graph_store,
        activation_store,
        search_index,
        _EmptyExtractor(),
        cfg=cfg,
    )
    yield manager, graph_store
    await graph_store.close()


async def _noop(*_args, **_kwargs):
    return None


def _session(session_id: str) -> SimpleNamespace:
    return SimpleNamespace(session_id=session_id, episode_count=0, last_activity=None)


# ── Span validator (deterministic) ──────────────────────────────────────────


class TestSpanValidator:
    def test_whitespace_normalized_casefold_substring(self):
        content = "Aurelia   works at\nNimbus   Corp."
        assert span_is_verified("aurelia works at nimbus corp.", content)

    def test_missing_span_is_unverified(self):
        assert not span_is_verified("never said this", "totally different text")

    def test_none_span_unverified(self):
        assert not span_is_verified(None, "some content")
        assert not span_is_verified("x", None)


# ── B1 foundation: proposals persist entities AND the edge ───────────────────


@pytest.mark.asyncio
async def test_remember_proposals_persist_entities_and_edge(proposal_manager):
    manager, graph_store = proposal_manager
    content = "Aurelia works at Nimbus Corp on the platform team."
    await manager.ingest_episode(
        content=content,
        group_id="default",
        source="test",
        proposed_entities=[
            {"name": "Aurelia", "entity_type": "Person", "source_span": content},
            {"name": "Nimbus Corp", "entity_type": "Organization", "source_span": content},
        ],
        proposed_relationships=[
            {
                "subject": "Aurelia",
                "predicate": "WORKS_AT",
                "object": "Nimbus Corp",
                "source_span": content,
            },
        ],
        model_tier="default",
    )

    aurelia = [
        e
        for e in await graph_store.find_entity_candidates("Aurelia", "default")
        if e.name == "Aurelia"
    ]
    nimbus = [
        e
        for e in await graph_store.find_entity_candidates("Nimbus Corp", "default")
        if e.name == "Nimbus Corp"
    ]
    assert aurelia, "proposed subject entity must persist"
    assert nimbus, "proposed object entity must persist"

    rels = await graph_store.get_relationships(
        aurelia[0].id,
        direction="outgoing",
        group_id="default",
    )
    assert any(r.predicate == "WORKS_AT" and r.target_id == nimbus[0].id for r in rels), (
        "the proposed edge between two committed entities must persist (B1)"
    )


@pytest.mark.asyncio
async def test_unverified_proposal_does_not_first_sight_commit(proposal_manager):
    manager, graph_store = proposal_manager
    # opus tier (0.92) would normally exceed every commit threshold; the cited
    # span is absent from the content, so it must defer instead of committing.
    await manager.ingest_episode(
        content="Some entirely unrelated content here.",
        group_id="default",
        source="test",
        proposed_entities=[
            {
                "name": "Ghostington",
                "entity_type": "Person",
                "source_span": "Ghostington founded Phantom Inc",
            },
        ],
        proposed_relationships=[
            {
                "subject": "Ghostington",
                "predicate": "FOUNDED",
                "object": "Phantom Inc",
                "source_span": "Ghostington founded Phantom Inc",
            },
        ],
        model_tier="opus",
    )

    committed = [
        e
        for e in await graph_store.find_entity_candidates("Ghostington", "default")
        if e.name == "Ghostington"
    ]
    assert not committed, "unverified single-source proposal must not commit on first sight"

    pending = await graph_store.get_pending_evidence(group_id="default", limit=50)
    deferred_names = [
        p["payload"].get("name")
        for p in pending
        if p["fact_class"] == "entity" and p["status"] == "deferred"
    ]
    assert "Ghostington" in deferred_names, "unverified proposal must be deferred, not dropped"


# ── Confidence de-weaponization (unit) ───────────────────────────────────────


class TestConfidenceDeWeaponization:
    def test_unverified_proposal_capped_into_defer_band(self):
        cands = proposals_to_evidence(
            [{"name": "Mystery", "entity_type": "Person", "source_span": "not in text"}],
            None,
            "ep1",
            "default",
            episode_content="completely different episode body",
            verify_spans=True,
        )
        assert cands[0].confidence <= UNVERIFIED_PROPOSAL_CONFIDENCE_CAP
        assert "span_unverified" in cands[0].corroborating_signals

    def test_verified_relationship_commits_at_default_tier(self):
        content = "Aurelia works at Nimbus Corp."
        cands = proposals_to_evidence(
            None,
            [
                {
                    "subject": "Aurelia",
                    "predicate": "WORKS_AT",
                    "object": "Nimbus Corp",
                    "source_span": content,
                }
            ],
            "ep1",
            "default",
            episode_content=content,
            verify_spans=True,
        )
        rel = cands[0]
        assert rel.confidence >= SPAN_VERIFIED_PROPOSAL_CONFIDENCE
        policy = AdaptiveCommitPolicy(CommitThresholds())
        bundle = EvidenceBundle(candidates=cands)
        decision = policy.evaluate(bundle, entity_count=0)[0]
        assert decision.action == "commit", "span-verified rel must clear the 0.75 threshold"

    def test_opus_unverified_still_capped(self):
        cands = proposals_to_evidence(
            [{"name": "Mystery", "entity_type": "Person", "source_span": "absent"}],
            None,
            "ep1",
            "default",
            model_tier="opus",
            episode_content="other text",
            verify_spans=True,
        )
        # 0.92 tier confidence must be capped below the entity threshold (0.70).
        assert cands[0].confidence <= UNVERIFIED_PROPOSAL_CONFIDENCE_CAP


# ── Per-claim source_span ────────────────────────────────────────────────────


class TestPerClaimSourceSpan:
    def test_each_claim_carries_its_own_span(self):
        content = "Alpha started the project. Beta joined later."
        cands = proposals_to_evidence(
            [
                {"name": "Alpha", "entity_type": "Person", "source_span": "Alpha started"},
                {"name": "Beta", "entity_type": "Person", "source_span": "Beta joined later"},
            ],
            None,
            "ep1",
            "default",
            episode_content=content,
            verify_spans=True,
        )
        spans = {c.payload["name"]: c.source_span for c in cands}
        assert spans["Alpha"] == "Alpha started"
        assert spans["Beta"] == "Beta joined later"
        # both spans are substrings -> both verified
        assert all("span_verified" in c.corroborating_signals for c in cands)


# ── Date re-anchoring ────────────────────────────────────────────────────────


class TestDateReanchoring:
    @pytest.mark.asyncio
    async def test_relative_hint_anchored_to_conversation_date(self):
        from unittest.mock import AsyncMock

        graph_store = AsyncMock()
        graph_store.get_relationships.return_value = []
        graph_store.find_conflicting_relationships.return_value = []
        graph_store.find_existing_relationship.return_value = None
        created: dict = {}

        async def _create(rel):
            created["rel"] = rel
            return rel.id

        graph_store.create_relationship.side_effect = _create

        await apply_relationship_fact(
            graph_store=graph_store,
            canonicalizer=PredicateCanonicalizer(),
            cfg=ActivationConfig(),
            rel_data={
                "source": "A",
                "target": "B",
                "predicate": "WORKS_AT",
                "temporal_hint": "last month",
            },
            entity_map={"A": "ent_a", "B": "ent_b"},
            group_id="default",
            source_episode="ep1",
            conversation_date=datetime(2024, 6, 15),
        )
        valid_from = created["rel"].valid_from
        assert (valid_from.year, valid_from.month) == (2024, 5), (
            "relative hint must anchor to conversation_date, not wall clock"
        )

    def test_absolute_relative_disagreement_tagged_and_deferred(self):
        content = "It started recently."
        cands = proposals_to_evidence(
            None,
            [
                {
                    "subject": "Project",
                    "predicate": "STARTED",
                    "object": "2024-01-01",
                    "valid_from": "2024-01-01",
                    "temporal_hint": "last month",
                    "source_span": content,
                }
            ],
            "ep1",
            "default",
            episode_content=content,
            reference_date=datetime(2024, 6, 15),
            verify_spans=True,
        )
        rel = cands[0]
        assert "date_conflict" in rel.corroborating_signals
        # conflict forces it back into the defer band despite a verified span.
        assert rel.confidence <= UNVERIFIED_PROPOSAL_CONFIDENCE_CAP


# ── Events: materialization + cheap observe ──────────────────────────────────


class TestEventsToProposals:
    def test_event_becomes_dated_node_and_edge(self):
        ents, rels = events_to_proposals(
            [{"name": "Launch", "date": "2026-04-01", "source_span": "Launch on 2026-04-01"}],
        )
        types = {e["name"]: e["entity_type"] for e in ents}
        assert types["Launch"] == "Event"
        assert types["2026-04-01"] == "Date"
        assert rels[0]["subject"] == "Launch"
        assert rels[0]["predicate"] == "OCCURRED_ON"
        assert rels[0]["object"] == "2026-04-01"
        assert rels[0]["valid_from"] == "2026-04-01"

    def test_merge_event_proposals_appends(self):
        ents, rels = merge_event_proposals(
            [{"name": "Launch", "date": "2026-04-01"}],
            [{"name": "Existing", "entity_type": "Concept"}],
            None,
        )
        names = {e["name"] for e in ents}
        assert {"Existing", "Launch", "2026-04-01"} <= names
        assert any(r["predicate"] == "OCCURRED_ON" for r in rels)

    def test_merge_with_no_events_is_passthrough(self):
        ents, rels = merge_event_proposals(None, [{"name": "X"}], [{"subject": "X"}])
        assert ents == [{"name": "X"}]
        assert rels == [{"subject": "X"}]


@pytest.mark.asyncio
async def test_remember_events_materialize_dated_event_node(proposal_manager):
    manager, graph_store = proposal_manager
    content = "Big day. Product Launch happened on 2026-04-01 with the whole team."
    session = _session("s_events")
    await build_mcp_remember_write_surface(
        manager,
        content=content,
        group_id="default",
        session=session,
        source="mcp",
        events=[
            {
                "name": "Product Launch",
                "date": "2026-04-01",
                "source_span": "Product Launch happened on 2026-04-01",
            }
        ],
        activation_cfg=manager._cfg,
        ingest_live_turn=_noop,
        recall_middleware=_noop,
    )

    event_nodes = [
        e
        for e in await graph_store.find_entity_candidates("Product Launch", "default")
        if e.name == "Product Launch"
    ]
    assert event_nodes, "remember(events) must create a first-class Event node"
    assert event_nodes[0].entity_type == "Event"

    rels = await graph_store.get_relationships(
        event_nodes[0].id,
        direction="outgoing",
        group_id="default",
    )
    occurred = [r for r in rels if r.predicate == "OCCURRED_ON"]
    assert occurred, "Event must have an OCCURRED_ON edge"
    assert occurred[0].valid_from is not None
    assert occurred[0].valid_from.year == 2026 and occurred[0].valid_from.month == 4


@pytest.mark.asyncio
async def test_observe_events_persist_deferred_without_projection(proposal_manager):
    manager, graph_store = proposal_manager
    session = _session("s_observe_events")
    await build_mcp_observe_write_surface(
        manager,
        content="Meeting recap. Kickoff was on 2026-05-01 in Denver.",
        group_id="g_obs",
        session=session,
        source="mcp",
        events=[
            {
                "name": "Kickoff",
                "date": "2026-05-01",
                "source_span": "Kickoff was on 2026-05-01",
            }
        ],
        ingest_live_turn=_noop,
        recall_middleware=_noop,
    )

    # observe stays cheap: no graph node materialized at observe time.
    kickoff_nodes = await graph_store.find_entity_candidates("Kickoff", "g_obs")
    assert not [e for e in kickoff_nodes if e.name == "Kickoff"], (
        "observe must not project events into the graph"
    )

    pending = await graph_store.get_pending_evidence(group_id="g_obs", limit=50)
    statuses = {p["status"] for p in pending}
    assert pending, "observe(events) must persist deferred evidence"
    assert statuses == {"deferred"}
    event_names = {p["payload"].get("name") for p in pending if p["fact_class"] == "entity"}
    assert "Kickoff" in event_names
