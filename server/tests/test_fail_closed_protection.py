"""Fail-closed protection tests (M1.3): dream/microglia destructive passes.

When identity-core protection cannot be loaded, or a connectivity check
errors, the destructive pass must be SKIPPED for the cycle — never run
unprotected (LTD sweep, microglia demotion) or speculate (dream
associations treating unknown connectivity as maximal surprise).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.dream import DreamSpreadingPhase
from engram.consolidation.phases.microglia import MicrogliaPhase
from engram.models.consolidation import CycleContext, DreamAssociationRecord
from engram.models.entity import Entity

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entity(
    eid: str,
    name: str,
    etype: str,
    summary: str = "A sufficiently long summary",
) -> Entity:
    return Entity(
        id=eid,
        name=name,
        entity_type=etype,
        summary=summary,
        group_id="default",
    )


def _assoc_cfg(**overrides) -> ActivationConfig:
    defaults = {
        "consolidation_dream_enabled": True,
        "consolidation_dream_associations_enabled": True,
        "consolidation_dream_assoc_min_surprise": 0.2,
        "consolidation_dream_assoc_min_summary_len": 5,
        "consolidation_dream_assoc_max_per_cycle": 10,
        "consolidation_dream_assoc_max_per_domain_pair": 5,
    }
    defaults.update(overrides)
    return ActivationConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. Dream LTD sweep: identity-core fetch failure skips the sweep
# ---------------------------------------------------------------------------


class TestDreamLtdSweepFailClosed:
    @pytest.mark.asyncio
    async def test_raising_identity_store_skips_sweep_helper(self):
        """Helper returns None (skipped) and applies zero weight decay."""
        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(
            consolidation_dream_ltd_sweep_enabled=True,
            consolidation_dream_ltd_sweep_size=50,
            consolidation_dream_ltd_sweep_decay=0.002,
        )
        graph = AsyncMock()
        activation = AsyncMock()

        state = SimpleNamespace(access_history=[])
        activation.get_top_activated = AsyncMock(return_value=[("low_ent", state)])
        graph.get_identity_core_entities = AsyncMock(side_effect=RuntimeError("store down"))
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("neighbor_1", 0.5, "KNOWS", "Person")]
        )
        graph.update_relationship_weight = AsyncMock()

        with patch("engram.consolidation.phases.dream.compute_activation", return_value=0.01):
            decayed = await phase._apply_ltd_low_activation_sweep(
                activation_store=activation,
                graph_store=graph,
                group_id="default",
                cfg=cfg,
                now=1_000_000.0,
            )

        assert decayed is None
        graph.update_relationship_weight.assert_not_called()

    @pytest.mark.asyncio
    async def test_raising_identity_store_recorded_in_phase_result(self):
        """execute() records the skipped sweep and decays nothing."""
        phase = DreamSpreadingPhase()
        cfg = ActivationConfig(
            consolidation_dream_enabled=True,
            consolidation_dream_ltd_sweep_enabled=True,
        )
        graph = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()

        state = SimpleNamespace(access_history=[])
        activation.get_top_activated = AsyncMock(return_value=[("low_ent", state)])
        graph.get_identity_core_entities = AsyncMock(side_effect=RuntimeError("store down"))
        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("neighbor_1", 0.5, "KNOWS", "Person")]
        )
        graph.update_relationship_weight = AsyncMock()

        # Activation 0.01 is below the seed floor (0.15): no seeds, but the
        # entity qualifies for the low-activation sweep.
        with patch("engram.consolidation.phases.dream.compute_activation", return_value=0.01):
            result, _records = await phase.execute(
                group_id="default",
                graph_store=graph,
                activation_store=activation,
                search_index=search,
                cfg=cfg,
                cycle_id="cyc_failclosed",
                dry_run=False,
            )

        assert result.status == "success"
        assert result.error is not None
        assert "ltd_sweep_skipped:identity_core_unavailable" in result.error
        graph.update_relationship_weight.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Dream associations: connectivity-check failure skips the candidate pair
# ---------------------------------------------------------------------------


class TestDreamAssociationsFailClosed:
    @pytest.mark.asyncio
    async def test_raising_path_check_creates_no_association(self):
        """A failed connectivity check must not become a speculative edge."""
        phase = DreamSpreadingPhase()
        cfg = _assoc_cfg()

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]
        graph = AsyncMock()
        graph.find_entities.return_value = entities
        graph.path_exists_within_hops = AsyncMock(side_effect=RuntimeError("storage error"))
        graph.get_identity_core_entities = AsyncMock(return_value=[])

        activation = AsyncMock()
        activation.get_top_activated.return_value = []

        search = AsyncMock()
        search.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        context = CycleContext()
        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_failclosed",
            dry_run=False,
            context=context,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert assoc_records == []
        graph.create_relationship.assert_not_called()
        assert context.dream_association_ids == set()

        assert result.status == "success"
        assert result.error is not None
        assert "assoc_pairs_skipped:path_check_failed=1" in result.error

    @pytest.mark.asyncio
    async def test_only_failing_pair_is_skipped(self):
        """Pairs whose connectivity check succeeds are still created."""
        phase = DreamSpreadingPhase()
        cfg = _assoc_cfg()

        entities = [
            _make_entity("t1", "Python", "Technology"),
            _make_entity("t2", "React", "Technology"),
            _make_entity("p1", "John", "Person"),
        ]

        async def _path_check(src_id, tgt_id, **kwargs):
            if "t1" in (src_id, tgt_id):
                raise RuntimeError("storage error")
            return False

        graph = AsyncMock()
        graph.find_entities.return_value = entities
        graph.path_exists_within_hops = AsyncMock(side_effect=_path_check)
        graph.get_identity_core_entities = AsyncMock(return_value=[])

        activation = AsyncMock()
        activation.get_top_activated.return_value = []

        search = AsyncMock()
        search.get_entity_embeddings.return_value = {
            "t1": [1.0, 0.0, 0.0, 0.0],
            "t2": [0.9, 0.1, 0.0, 0.0],
            "p1": [0.95, 0.05, 0.0, 0.0],
        }

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_failclosed",
            dry_run=False,
        )

        assoc_records = [r for r in records if isinstance(r, DreamAssociationRecord)]
        assert len(assoc_records) == 1
        pair = {assoc_records[0].source_entity_id, assoc_records[0].target_entity_id}
        assert pair == {"t2", "p1"}
        assert result.error is not None
        assert "assoc_pairs_skipped:path_check_failed=1" in result.error


# ---------------------------------------------------------------------------
# 3. Microglia: identity-core fetch failure skips the demotion pass
# ---------------------------------------------------------------------------


class TestMicrogliaDemotionFailClosed:
    @pytest.mark.asyncio
    async def test_raising_identity_store_skips_demotion(self):
        phase = MicrogliaPhase()
        cfg = ActivationConfig(
            microglia_enabled=True,
            microglia_min_cycles_to_demote=2,
        )
        graph = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()

        graph.get_identity_core_entities = AsyncMock(side_effect=RuntimeError("store down"))
        graph.sample_edges = AsyncMock(return_value=[])
        graph.find_entities = AsyncMock(return_value=[])
        graph.get_entity = AsyncMock(return_value=None)
        graph.update_entity = AsyncMock()
        graph.update_relationship_weight = AsyncMock()
        # Confirmed tag that WOULD be demoted if the pass ran
        confirmed_tag = {
            "id": "tag_1",
            "target_type": "edge",
            "target_id": "e1:e2:RELATES_TO",
            "tag_type": "c1q_domain",
            "score": 0.8,
        }
        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[confirmed_tag])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.clear_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("e2", 1.0, "RELATES_TO")]
        )

        context = CycleContext()
        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_failclosed",
            dry_run=False,
            context=context,
        )

        # Demotion pass never ran: no confirmed tags read, no weight change
        consol_store.get_confirmed_tags.assert_not_called()
        graph.update_relationship_weight.assert_not_called()
        assert context.microglia_demoted_edge_ids == set()
        assert [r for r in records if r.action == "demoted"] == []

        # Skip is recorded in the phase result
        assert result.status == "success"
        assert result.error == "demotion_skipped:identity_core_unavailable"

    @pytest.mark.asyncio
    async def test_working_identity_store_still_demotes(self):
        """Sanity: with a healthy identity store the demotion pass runs."""
        phase = MicrogliaPhase()
        cfg = ActivationConfig(
            microglia_enabled=True,
            microglia_min_cycles_to_demote=2,
        )
        graph = AsyncMock()
        activation = AsyncMock()
        search = AsyncMock()

        graph.get_identity_core_entities = AsyncMock(return_value=[])
        graph.sample_edges = AsyncMock(return_value=[])
        graph.find_entities = AsyncMock(return_value=[])
        graph.get_entity = AsyncMock(return_value=None)
        graph.update_entity = AsyncMock()
        graph.update_relationship_weight = AsyncMock()
        confirmed_tag = {
            "id": "tag_1",
            "target_type": "edge",
            "target_id": "e1:e2:RELATES_TO",
            "tag_type": "c1q_domain",
            "score": 0.8,
        }
        consol_store = AsyncMock()
        consol_store.get_active_complement_tags = AsyncMock(return_value=[])
        consol_store.get_confirmed_tags = AsyncMock(return_value=[confirmed_tag])
        consol_store.get_unconfirmed_tags = AsyncMock(return_value=[])
        consol_store.clear_complement_tag = AsyncMock()
        graph._consolidation_store = consol_store

        graph.get_active_neighbors_with_weights = AsyncMock(
            return_value=[("e2", 1.0, "RELATES_TO")]
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph,
            activation_store=activation,
            search_index=search,
            cfg=cfg,
            cycle_id="cyc_abc123",
            dry_run=False,
        )

        graph.update_relationship_weight.assert_called_once()
        assert result.error is None
        assert len([r for r in records if r.action == "demoted"]) == 1
