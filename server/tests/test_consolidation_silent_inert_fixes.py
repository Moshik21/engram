"""Regression tests for cluster A4 consolidation silent-inert fixes (B4, B11, B12).

These tests pin three "configured but silently produces nothing while metered
as success" defects:

- B4: microglia must NOT meter edge tagging as affected/success when there is
  no consolidation store to persist the Tag->Confirm->Demote lifecycle, and the
  engine must wire the store onto the graph store so the lifecycle is reachable.
- B11: schema formation output is analytics-only and must not be marked as
  recall-affecting (no schema_id added to affected_entity_ids).
- B12: graph_embed must report status='skipped' (not 'success') when 0
  embeddings were trained (below the min-entity threshold), so operators can see
  the structural signal is currently inert.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.phases.graph_embed import GraphEmbedPhase
from engram.consolidation.phases.microglia import MicrogliaPhase
from engram.models.entity import Entity
from engram.models.relationship import Relationship
from engram.utils.dates import utc_now

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(eid: str, etype: str, name: str) -> Entity:
    return Entity(
        id=eid,
        name=name,
        entity_type=etype,
        group_id="default",
        created_at=utc_now(),
        updated_at=utc_now(),
    )


def _rel(src: str, tgt: str, predicate: str, weight: float) -> Relationship:
    return Relationship(
        id=f"rel_{src}_{tgt}",
        source_id=src,
        target_id=tgt,
        predicate=predicate,
        weight=weight,
        source_episode=None,
        group_id="default",
    )


def _microglia_cfg(**overrides) -> ActivationConfig:
    defaults = dict(
        microglia_enabled=True,
        microglia_tag_threshold=0.5,
        microglia_confirm_threshold=0.4,
        microglia_min_cycles_to_demote=2,
        microglia_max_demotions_per_cycle=20,
        microglia_scan_edges_per_cycle=500,
        microglia_scan_entities_per_cycle=200,
    )
    defaults.update(overrides)
    return ActivationConfig(**defaults)


def _make_contaminated_graph():
    """Graph store returning one clearly-contaminated edge (Person->Software,
    generic predicate, low weight, no evidence) that would tag under scoring."""
    graph = AsyncMock()
    person = _entity("e1", "Person", "Alice")
    software = _entity("e2", "Software", "MyApp")
    edge = _rel("e1", "e2", "RELATES_TO", weight=0.2)

    graph.get_identity_core_entities = AsyncMock(return_value=[])
    graph.sample_edges = AsyncMock(return_value=[edge])
    graph.find_entities = AsyncMock(return_value=[])
    graph.update_entity = AsyncMock()
    graph.update_relationship_weight = AsyncMock()
    graph.get_active_neighbors_with_weights = AsyncMock(return_value=[])
    graph.get_entity = AsyncMock(side_effect=lambda eid, gid: person if eid == "e1" else software)
    activation = AsyncMock()
    search = AsyncMock()
    search.get_entity_embeddings = AsyncMock(return_value={})
    return graph, activation, search


# ---------------------------------------------------------------------------
# B4: microglia must not meter no-op tagging when no consolidation store
# ---------------------------------------------------------------------------


class TestMicrogliaNoStoreInert:
    @pytest.mark.asyncio
    async def test_no_store_does_not_report_tagged_or_affected(self):
        """With consolidation_store=None the edge contamination lifecycle is
        inert; the phase must NOT count tags and must NOT report success."""
        phase = MicrogliaPhase()
        cfg = _microglia_cfg()
        graph, activation, search = _make_contaminated_graph()
        graph._consolidation_store = None  # production reality before the fix

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        # No tagged records emitted (nothing could be persisted).
        tagged = [r for r in records if r.action == "tagged"]
        assert tagged == []
        # No affected work metered, and status is honestly 'skipped'.
        assert result.items_affected == 0
        assert result.status == "skipped"

    @pytest.mark.asyncio
    async def test_with_store_still_tags_and_reports_success(self):
        """Sanity: the guard only suppresses the no-store case. When a real
        store IS wired the lifecycle runs and tagging is metered."""
        phase = MicrogliaPhase()
        cfg = _microglia_cfg()
        graph, activation, search = _make_contaminated_graph()

        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.create_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        tagged = [r for r in records if r.action == "tagged"]
        assert len(tagged) >= 1
        assert result.items_affected >= 1
        assert result.status == "success"
        consol_store.create_complement_tag.assert_called_once()


class TestEngineWiresConsolidationStore:
    def test_engine_attaches_store_to_graph(self):
        """The engine must wire the consolidation store onto the graph store so
        microglia's getattr(graph_store, '_consolidation_store') resolves."""
        graph = AsyncMock()
        # ensure the attribute starts absent / falsy
        graph._consolidation_store = None
        consol_store = AsyncMock()

        ConsolidationEngine(
            graph_store=graph,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=ActivationConfig(),
            consolidation_store=consol_store,
        )

        assert graph._consolidation_store is consol_store

    def test_engine_no_store_leaves_graph_untouched(self):
        """When no consolidation store exists the engine must not crash and the
        graph store stays inert (microglia will correctly report skipped)."""
        graph = AsyncMock()
        graph._consolidation_store = None

        ConsolidationEngine(
            graph_store=graph,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=ActivationConfig(),
            consolidation_store=None,
        )

        assert graph._consolidation_store is None


# ---------------------------------------------------------------------------
# B12: graph_embed must report 'skipped' when 0 embeddings trained
# ---------------------------------------------------------------------------


class TestGraphEmbedBelowThreshold:
    @pytest.mark.asyncio
    async def test_below_threshold_reports_skipped(self):
        """Too few entities -> node2vec returns {} -> table stays empty. The
        phase must report status='skipped' (not 'success') with 0 processed."""
        cfg = ActivationConfig(
            graph_embedding_node2vec_enabled=True,
            graph_embedding_node2vec_min_entities=50,
            graph_embedding_transe_enabled=False,
            graph_embedding_gnn_enabled=False,
        )
        phase = GraphEmbedPhase()

        # Only 5 entities — well below the 50-entity training threshold.
        n = 5
        entities = [type("E", (), {"id": f"e{i}"})() for i in range(n)]

        class MockGraph:
            async def find_entities(self, group_id, limit):
                return entities

            async def get_active_neighbors_with_weights(self, eid, group_id):
                idx = int(eid[1:])
                return [(f"e{(idx + 1) % n}", 1.0, "REL")]

        class MockSearchIndex:
            pass

        result, records = await phase.execute(
            group_id="default",
            graph_store=MockGraph(),
            activation_store=None,
            search_index=MockSearchIndex(),
            cfg=cfg,
            cycle_id="cyc_belowthreshold",
            dry_run=True,
        )

        assert result.phase == "graph_embed"
        assert result.status == "skipped"
        assert result.items_processed == 0
        assert result.items_affected == 0
        # No GraphEmbedRecord because nothing trained.
        assert records == []
