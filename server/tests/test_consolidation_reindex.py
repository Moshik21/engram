"""Tests for the ReindexPhase."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.reindex import ReindexPhase
from engram.models.consolidation import CycleContext


def _make_entity(entity_id: str, name: str = "Test Entity"):
    """Create a mock entity with given id and name."""
    e = MagicMock()
    e.id = entity_id
    e.name = name
    return e


@pytest.fixture
def phase():
    return ReindexPhase()


@pytest.fixture
def cfg():
    return ActivationConfig()


@pytest.fixture
def graph_store():
    store = AsyncMock()
    return store


@pytest.fixture
def activation_store():
    return AsyncMock()


@pytest.fixture
def search_index():
    idx = AsyncMock()
    idx.index_entity = AsyncMock()
    idx.batch_index_entities = AsyncMock(
        side_effect=lambda entities: len(entities),
    )
    return idx


class TestReindexPhase:
    @pytest.mark.asyncio
    async def test_empty_context_returns_zero(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Empty context (no affected entities) should return 0 processed."""
        ctx = CycleContext()
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_processed == 0
        assert result.items_affected == 0
        assert records == []

    @pytest.mark.asyncio
    async def test_no_context_returns_zero(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """None context should return 0 processed."""
        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=None,
        )
        assert result.items_processed == 0
        assert result.items_affected == 0

    @pytest.mark.asyncio
    async def test_merge_survivors_reindexed(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Merge survivor entities should be re-embedded."""
        ctx = CycleContext()
        ctx.merge_survivor_ids.add("ent_1")
        ctx.affected_entity_ids.add("ent_1")

        entity = _make_entity("ent_1", "Alice")
        graph_store.get_entity = AsyncMock(return_value=entity)

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_affected == 1
        search_index.batch_index_entities.assert_called_once()
        batch_args = search_index.batch_index_entities.call_args[0][0]
        assert len(batch_args) == 1
        assert batch_args[0].id == "ent_1"
        assert records[0].source_phase == "merge"
        assert records[0].entity_name == "Alice"

    @pytest.mark.asyncio
    async def test_infer_endpoints_reindexed(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Inferred edge endpoint entities should be re-embedded."""
        ctx = CycleContext()
        ctx.inferred_edge_entity_ids.update({"ent_a", "ent_b"})
        ctx.affected_entity_ids.update({"ent_a", "ent_b"})

        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: _make_entity(eid, f"Entity_{eid}"),
        )

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_affected == 2
        search_index.batch_index_entities.assert_called_once()
        batch_args = search_index.batch_index_entities.call_args[0][0]
        assert len(batch_args) == 2
        phases = {r.source_phase for r in records}
        assert phases == {"infer"}

    @pytest.mark.asyncio
    async def test_pruned_entities_excluded(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Pruned entities should NOT be reindexed."""
        ctx = CycleContext()
        ctx.affected_entity_ids.add("ent_1")
        ctx.merge_survivor_ids.add("ent_1")
        ctx.pruned_entity_ids.add("ent_1")

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_processed == 0
        assert result.items_affected == 0
        search_index.index_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_no_index_calls(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Dry run should count but not call index_entity."""
        ctx = CycleContext()
        ctx.affected_entity_ids.add("ent_1")
        ctx.merge_survivor_ids.add("ent_1")

        entity = _make_entity("ent_1", "Alice")
        graph_store.get_entity = AsyncMock(return_value=entity)

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
            context=ctx,
        )
        assert result.items_processed == 1
        assert result.items_affected == 0
        search_index.index_entity.assert_not_called()
        assert len(records) == 1
        assert records[0].source_phase == "merge"

    @pytest.mark.asyncio
    async def test_max_per_cycle_respected(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
    ):
        """Should cap reindexing at consolidation_reindex_max_per_cycle."""
        cfg = ActivationConfig(consolidation_reindex_max_per_cycle=2)
        ctx = CycleContext()
        for i in range(5):
            eid = f"ent_{i}"
            ctx.affected_entity_ids.add(eid)
            ctx.merge_survivor_ids.add(eid)

        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: _make_entity(eid),
        )

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_processed == 2
        assert result.items_affected == 2
        search_index.batch_index_entities.assert_called_once()
        assert len(search_index.batch_index_entities.call_args[0][0]) == 2

    @pytest.mark.asyncio
    async def test_entity_not_found_skipped(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Entity that can't be fetched should be gracefully skipped."""
        ctx = CycleContext()
        ctx.affected_entity_ids.add("ent_gone")
        ctx.merge_survivor_ids.add("ent_gone")

        graph_store.get_entity = AsyncMock(return_value=None)

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_processed == 1
        assert result.items_affected == 0
        search_index.index_entity.assert_not_called()

    @pytest.mark.asyncio
    async def test_embedding_error_non_fatal(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Embedding failure for one entity should not halt the phase."""
        ctx = CycleContext()
        ctx.affected_entity_ids.update({"ent_ok", "ent_fail"})
        ctx.merge_survivor_ids.update({"ent_ok", "ent_fail"})

        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: _make_entity(eid),
        )

        # Force batch to fail so it falls back to one-at-a-time
        search_index.batch_index_entities = AsyncMock(
            side_effect=RuntimeError("Batch embedding failed"),
        )

        call_count = 0

        async def mock_index(entity):
            nonlocal call_count
            call_count += 1
            if entity.id == "ent_fail":
                raise RuntimeError("Embedding API error")

        search_index.index_entity = mock_index

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        # One should succeed, one should fail
        assert result.items_processed == 2
        assert result.items_affected == 1
        assert len(records) == 1
        assert records[0].entity_id == "ent_ok"

    @pytest.mark.asyncio
    async def test_deduplicates_entity_ids(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Same entity from merge AND infer should be reindexed once."""
        ctx = CycleContext()
        ctx.merge_survivor_ids.add("ent_shared")
        ctx.inferred_edge_entity_ids.add("ent_shared")
        ctx.affected_entity_ids.add("ent_shared")

        entity = _make_entity("ent_shared", "Shared Entity")
        graph_store.get_entity = AsyncMock(return_value=entity)

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        assert result.items_affected == 1
        search_index.batch_index_entities.assert_called_once()
        assert len(search_index.batch_index_entities.call_args[0][0]) == 1
        # Source should be "merge" since it's in merge_survivor_ids
        assert records[0].source_phase == "merge"

    @pytest.mark.asyncio
    async def test_audit_trail_correct_source_phase(
        self,
        phase,
        graph_store,
        activation_store,
        search_index,
        cfg,
    ):
        """Records should have correct source_phase based on context sets."""
        ctx = CycleContext()
        ctx.merge_survivor_ids.add("ent_merged")
        ctx.affected_entity_ids.add("ent_merged")
        ctx.inferred_edge_entity_ids.add("ent_inferred")
        ctx.affected_entity_ids.add("ent_inferred")

        graph_store.get_entity = AsyncMock(
            side_effect=lambda eid, gid: _make_entity(eid, f"Name_{eid}"),
        )

        result, records = await phase.execute(
            group_id="test",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_test",
            context=ctx,
        )
        by_id = {r.entity_id: r for r in records}
        assert by_id["ent_merged"].source_phase == "merge"
        assert by_id["ent_inferred"].source_phase == "infer"

    @pytest.mark.asyncio
    async def test_phase_name_is_reindex(self, phase):
        assert phase.name == "reindex"
