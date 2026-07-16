"""Tests for hybrid entity-vector index completeness and backfill."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engram.models.entity import Entity
from engram.storage.index_completeness import (
    backfill_missing_entity_vectors,
    is_indexable_entity,
    measure_entity_vector_coverage,
)


def _entity(
    eid: str,
    name: str = "Entity",
    *,
    deleted: bool = False,
) -> Entity:
    return Entity(
        id=eid,
        name=name,
        entity_type="Concept",
        summary="summary",
        group_id="default",
        deleted_at=datetime.now(timezone.utc) if deleted else None,
    )


def test_is_indexable_entity_filters_empty_and_deleted():
    assert is_indexable_entity(_entity("a", "Alpha"))
    assert not is_indexable_entity(_entity("b", "   "))
    assert not is_indexable_entity(_entity("c", "Gone", deleted=True))
    assert not is_indexable_entity(None)


@pytest.mark.asyncio
async def test_measure_reports_missing_vectors():
    entities = [
        _entity("e1", "Alpha"),
        _entity("e2", "Beta"),
        _entity("e3", "Gamma"),
        _entity("e4", ""),  # skipped empty
        _entity("e5", "Deleted", deleted=True),
    ]
    graph = AsyncMock()
    graph.find_entities = AsyncMock(return_value=entities)

    search = AsyncMock()
    # Only e1 has a vector
    search.get_entity_embeddings = AsyncMock(
        side_effect=lambda ids, group_id=None: {eid: [0.1, 0.2] for eid in ids if eid == "e1"}
    )

    report = await measure_entity_vector_coverage(graph, search, "default")
    assert report.indexable_count == 3
    assert report.vector_count == 1
    assert report.missing_count == 2
    assert set(report.missing_ids) == {"e2", "e3"}
    assert report.coverage == pytest.approx(1 / 3, rel=1e-3)
    assert report.skipped_empty_name == 1
    assert report.skipped_deleted == 1


@pytest.mark.asyncio
async def test_backfill_indexes_only_missing_via_batch():
    missing = [_entity("e2", "Beta"), _entity("e3", "Gamma")]
    graph = AsyncMock()
    graph.get_entity = AsyncMock(
        side_effect=lambda eid, gid: next((e for e in missing if e.id == eid), None)
    )
    search = AsyncMock()
    search.batch_index_entities = AsyncMock(side_effect=lambda ents: len(ents))
    search.index_entity = AsyncMock()

    result = await backfill_missing_entity_vectors(
        graph,
        search,
        "default",
        max_entities=10,
        missing_ids=["e2", "e3"],
        entities_by_id={e.id: e for e in missing},
    )
    assert result.attempted == 2
    assert result.indexed == 2
    assert result.failed == 0
    assert set(result.indexed_ids) == {"e2", "e3"}
    search.batch_index_entities.assert_awaited()
    search.index_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_backfill_falls_back_to_one_at_a_time():
    entity = _entity("e9", "Solo")
    graph = AsyncMock()
    search = SimpleNamespace(
        batch_index_entities=AsyncMock(side_effect=RuntimeError("batch boom")),
        index_entity=AsyncMock(),
    )
    result = await backfill_missing_entity_vectors(
        graph,
        search,
        "default",
        missing_ids=["e9"],
        entities_by_id={"e9": entity},
    )
    assert result.indexed == 1
    assert result.indexed_ids == ["e9"]
    search.index_entity.assert_awaited_once()


@pytest.mark.asyncio
async def test_backfill_dry_run_does_not_index():
    entity = _entity("e1", "Alpha")
    search = AsyncMock()
    result = await backfill_missing_entity_vectors(
        AsyncMock(),
        search,
        "default",
        missing_ids=["e1"],
        entities_by_id={"e1": entity},
        dry_run=True,
    )
    assert result.attempted == 1
    assert result.indexed == 0
    search.batch_index_entities.assert_not_awaited()
    search.index_entity.assert_not_awaited()


@pytest.mark.asyncio
async def test_reindex_phase_fills_missing_with_leftover_budget():
    from engram.config import ActivationConfig
    from engram.consolidation.phases.reindex import ReindexPhase
    from engram.models.consolidation import CycleContext

    phase = ReindexPhase()
    cfg = ActivationConfig(
        consolidation_reindex_max_per_cycle=5,
        consolidation_reindex_fill_missing_vectors=True,
    )
    graph = AsyncMock()
    # No affected entities needing reindex
    graph.get_entity = AsyncMock(return_value=None)
    search = AsyncMock()
    search.get_entity_embeddings = AsyncMock(return_value={})
    search.batch_index_entities = AsyncMock(side_effect=lambda ents: len(ents))

    missing = [_entity(f"m{i}", f"Miss{i}") for i in range(3)]
    graph.find_entities = AsyncMock(return_value=missing)
    graph.get_entity = AsyncMock(
        side_effect=lambda eid, gid: next((e for e in missing if e.id == eid), None)
    )

    result, records = await phase.execute(
        group_id="default",
        graph_store=graph,
        activation_store=AsyncMock(),
        search_index=search,
        cfg=cfg,
        cycle_id="cyc_fill",
        context=CycleContext(),
    )
    assert result.items_affected >= 3
    assert any(r.source_phase == "missing_vector" for r in records)
    search.batch_index_entities.assert_awaited()
