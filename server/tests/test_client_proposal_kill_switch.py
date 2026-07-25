"""`evidence_client_proposals_enabled` must actually gate client proposals.

The flag shipped with a six-line docstring promising suppression semantics, a
profile that sets it True with a written safety rationale, a continuity harness
that sets it, and seven tests asserting it round-trips -- and **zero read
sites**. Setting it False changed nothing: the agent-as-extractor path ran
regardless. That is the most strategically important extraction path in the
product (no Ollama, no external API -- the harness agent is the only
intelligence source), shipped with a safety switch that did nothing.

These tests assert the flag GATES. Every "off" case here passed with the
opposite assertion before the fix, which is the assertion nobody wrote.

Regimes: isolated, in-process. A lite/SQLite store for the end-to-end cases;
stub objects for the routing/scoreboard cases. No live measurement.
"""

from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.extraction.harness_metrics import get_harness_metrics, reset_harness_metrics
from engram.graph_manager import GraphManager
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.search import FTS5SearchIndex

# A Decision statement that appears verbatim in the episode, so the span
# validator verifies it and the commit policy commits it -- but that the narrow
# regex extractor cannot produce (it is a lowercase declarative clause, not a
# proper-noun fragment). Its presence in the graph is therefore proof that the
# CLIENT PROPOSAL committed, not that internal extraction happened to agree.
_CONTENT = (
    "Decision recorded today: keep Engram fully local, with zero external keys "
    "for operation. Nimbus Corp signed off."
)
_PROPOSED_NAME = "keep Engram fully local"


class _EmptyExtractor(EntityExtractor):
    """Extractor that returns nothing -- proposals are the only structure source."""

    def __init__(self) -> None:
        self._result = ExtractionResult(entities=[], relationships=[])

    async def extract(self, text: str) -> ExtractionResult:
        return self._result


def _proposals() -> tuple[list[dict], list[dict]]:
    return (
        [
            {
                "name": _PROPOSED_NAME,
                "entity_type": "Decision",
                "source_span": _PROPOSED_NAME,
            },
            {
                "name": "Nimbus Corp",
                "entity_type": "Organization",
                "source_span": "Nimbus Corp",
            },
        ],
        [
            {
                "subject": "Nimbus Corp",
                "predicate": "DECIDED",
                "object": _PROPOSED_NAME,
                "source_span": _CONTENT,
            },
        ],
    )


def _manager(*, proposals_enabled: bool) -> GraphManager:
    """In-memory manager with only the flag under test varied."""
    cfg = ActivationConfig(
        evidence_extraction_enabled=True,
        evidence_client_proposals_enabled=proposals_enabled,
    )
    manager = GraphManager(
        graph_store=MagicMock(),
        activation_store=MagicMock(),
        search_index=MagicMock(),
        extractor=MagicMock(),
        cfg=cfg,
    )
    return manager


@pytest.fixture(autouse=True)
def isolated_harness_scoreboard(tmp_path, monkeypatch):
    """Keep the scoreboard off the operator's real one.

    `harness_metrics_path()` falls back to `~/.engram/harness-metrics.json` and
    nothing in the suite overrides it, so every pytest run that touches an
    extraction path has been incrementing the LIVE dogfood counters that
    `engram harness` and the graph-thesis investigation read.
    """
    monkeypatch.setenv("ENGRAM_HARNESS_METRICS_PATH", str(tmp_path / "harness-metrics.json"))
    reset_harness_metrics()
    yield
    reset_harness_metrics()


@pytest_asyncio.fixture
async def lite_stores():
    """Yield a factory that builds lite managers sharing nothing but the tmpdir."""
    opened: list[SQLiteGraphStore] = []
    tmpdir = tempfile.mkdtemp()

    async def _build(*, proposals_enabled: bool, tag: str) -> tuple[GraphManager, SQLiteGraphStore]:
        db_path = os.path.join(tmpdir, f"killswitch-{tag}.db")
        graph_store = SQLiteGraphStore(db_path)
        await graph_store.initialize()
        search_index = FTS5SearchIndex(db_path)
        await search_index.initialize(db=graph_store._db)
        cfg = ActivationConfig(
            evidence_client_proposals_enabled=proposals_enabled,
        )
        manager = GraphManager(
            graph_store,
            MemoryActivationStore(cfg=cfg),
            search_index,
            _EmptyExtractor(),
            cfg=cfg,
        )
        opened.append(graph_store)
        return manager, graph_store

    yield _build
    for store in opened:
        await store.close()


# ── The bundle builder: proposals are the sole evidence source, or they are not ──


class TestBundleGate:
    def test_flag_on_proposals_are_the_sole_evidence_source(self):
        """Control. Unchanged shipped behaviour at the default (True)."""
        manager = _manager(proposals_enabled=True)
        manager._evidence_pipeline = MagicMock()
        manager._evidence_pipeline.extract.side_effect = AssertionError(
            "narrow pipeline must not run when proposals are accepted"
        )
        entities, relationships = _proposals()

        bundle = manager._build_evidence_bundle(
            text=_CONTENT,
            episode_id="ep-on",
            group_id="default",
            proposed_entities=entities,
            proposed_relationships=relationships,
            model_tier="sonnet",
        )

        assert bundle.extractor_stats["extraction_path"] == "client_proposals"
        assert bundle.candidates
        assert all(c.source_type == "client_proposal" for c in bundle.candidates)

    def test_flag_off_proposals_are_ignored_and_internal_extraction_runs(self):
        """THE ASSERTION NOBODY WROTE. Failed before the fix: the bundle came back
        `client_proposals` with the flag False and the narrow pipeline never ran."""
        manager = _manager(proposals_enabled=False)
        narrow_bundle = SimpleNamespace(
            episode_id="ep-off",
            group_id="default",
            candidates=[],
            extractor_stats={},
            total_ms=0.0,
        )
        manager._evidence_pipeline = MagicMock()
        manager._evidence_pipeline.extract.return_value = narrow_bundle
        entities, relationships = _proposals()

        bundle = manager._build_evidence_bundle(
            text=_CONTENT,
            episode_id="ep-off",
            group_id="default",
            proposed_entities=entities,
            proposed_relationships=relationships,
            model_tier="sonnet",
        )

        manager._evidence_pipeline.extract.assert_called_once()
        assert bundle.extractor_stats["extraction_path"] == "narrow"
        assert not [c for c in bundle.candidates if c.source_type == "client_proposal"]

    def test_flag_off_does_not_claim_the_external_extractor_was_skipped(self):
        """`external_extractor_skipped` means "proposals present, so we did not
        call an external model". With the flag off that sentence is false, and a
        counter that says otherwise is a lying gauge (STANDING_GOAL 2.1)."""
        manager = _manager(proposals_enabled=False)
        manager._evidence_pipeline = MagicMock()
        manager._evidence_pipeline.extract.return_value = SimpleNamespace(
            episode_id="ep-off",
            group_id="default",
            candidates=[],
            extractor_stats={},
            total_ms=0.0,
        )
        entities, relationships = _proposals()

        before = get_harness_metrics().external_extractor_skipped
        narrow_before = get_harness_metrics().narrow_extractions

        manager._build_evidence_bundle(
            text=_CONTENT,
            episode_id="ep-off",
            group_id="default",
            proposed_entities=entities,
            proposed_relationships=relationships,
            model_tier="sonnet",
        )

        after = get_harness_metrics()
        assert after.external_extractor_skipped == before
        assert after.narrow_extractions == narrow_before + 1


# ── The routing gate: proposals force the v2 path, or they do not ──────────────


class TestRoutingGate:
    def test_flag_on_proposals_force_the_evidence_path(self):
        """Control: shipped hard-path behaviour survives at the default."""
        cfg = ActivationConfig(
            evidence_extraction_enabled=False,
            evidence_client_proposals_enabled=True,
        )
        manager = GraphManager(
            graph_store=MagicMock(),
            activation_store=MagicMock(),
            search_index=MagicMock(),
            extractor=MagicMock(),
            cfg=cfg,
        )
        assert manager._should_use_evidence_pipeline(
            proposed_entities=[{"name": "X", "entity_type": "Decision"}],
        )

    def test_flag_off_proposals_no_longer_force_the_evidence_path(self):
        """Failed before the fix: `has_client_proposals` short-circuited to True
        regardless of the flag, so the hard path ran with the switch off."""
        cfg = ActivationConfig(
            evidence_extraction_enabled=False,
            evidence_client_proposals_enabled=False,
        )
        manager = GraphManager(
            graph_store=MagicMock(),
            activation_store=MagicMock(),
            search_index=MagicMock(),
            extractor=MagicMock(),
            cfg=cfg,
        )
        assert not manager._should_use_evidence_pipeline(
            proposed_entities=[{"name": "X", "entity_type": "Decision"}],
        )


# ── End to end on a lite store: the fact lands in the graph, or it does not ────


@pytest.mark.asyncio
async def test_flag_on_commits_the_proposed_decision(lite_stores):
    """Control. The agent's clean atomic fact reaches the graph."""
    manager, graph_store = await lite_stores(proposals_enabled=True, tag="on")
    entities, relationships = _proposals()

    await manager.ingest_episode(
        content=_CONTENT,
        group_id="default",
        source="test",
        proposed_entities=entities,
        proposed_relationships=relationships,
        model_tier="default",
    )

    matches = [
        e
        for e in await graph_store.find_entity_candidates(_PROPOSED_NAME, "default")
        if e.name == _PROPOSED_NAME
    ]
    assert matches, "flag ON must commit the agent-proposed Decision"
    # Control for the scoreboard assertion below: the counter DOES move here.
    assert get_harness_metrics().client_proposal_commits > 0


@pytest.mark.asyncio
async def test_flag_off_does_not_commit_the_proposed_decision(lite_stores):
    """THE KILL SWITCH. Failed before the fix: the Decision committed anyway."""
    manager, graph_store = await lite_stores(proposals_enabled=False, tag="off")
    entities, relationships = _proposals()

    await manager.ingest_episode(
        content=_CONTENT,
        group_id="default",
        source="test",
        proposed_entities=entities,
        proposed_relationships=relationships,
        model_tier="default",
    )

    matches = [
        e
        for e in await graph_store.find_entity_candidates(_PROPOSED_NAME, "default")
        if e.name == _PROPOSED_NAME
    ]
    assert not matches, (
        f"flag OFF must not commit an agent-proposed fact; found {[e.name for e in matches]}"
    )
    # The scoreboard must not attribute the narrow extractor's work to the
    # harness. `is_proposal_path` used to fall back to "were proposals
    # supplied?", which is a different question from "did the proposal path
    # run" the moment the flag can decline them.
    snapshot = get_harness_metrics()
    assert snapshot.client_proposal_commits == 0, (
        "flag OFF must not credit the harness scoreboard with commits it did "
        f"not make; got {snapshot.to_dict()}"
    )
    assert snapshot.client_proposal_defers == 0
    assert snapshot.client_proposal_rejects == 0
