"""Entity hybrid-index completeness: measure missing vectors and backfill.

Product recall can win via exact-name, identity_core, and packet cache even when
HNSW/BM25 entity vectors lag. This module closes the permanent gap by:

1. Enumerating active (non-deleted, named) entities
2. Probing which lack stored entity vectors
3. Batch-indexing the missing set via the existing SearchIndex path

Designed for unit tests with fakes and for live dogfood via CLI.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from engram.models.entity import Entity

logger = logging.getLogger(__name__)

# Safety cap when listing entities for coverage scans.
DEFAULT_ENTITY_SCAN_LIMIT = 50_000
DEFAULT_EMBEDDING_PROBE_CHUNK = 64
DEFAULT_BACKFILL_BATCH = 32


@dataclass
class VectorCoverageReport:
    """Snapshot of entity-vector hybrid index completeness."""

    group_id: str
    entity_count: int = 0
    indexable_count: int = 0
    vector_count: int = 0
    missing_ids: list[str] = field(default_factory=list)
    skipped_empty_name: int = 0
    skipped_deleted: int = 0

    @property
    def missing_count(self) -> int:
        return len(self.missing_ids)

    @property
    def coverage(self) -> float:
        if self.indexable_count <= 0:
            return 1.0
        return round(self.vector_count / self.indexable_count, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "entity_count": self.entity_count,
            "indexable_count": self.indexable_count,
            "vector_count": self.vector_count,
            "missing_count": self.missing_count,
            "coverage": self.coverage,
            "missing_ids": list(self.missing_ids),
            "skipped_empty_name": self.skipped_empty_name,
            "skipped_deleted": self.skipped_deleted,
        }


@dataclass
class BackfillResult:
    """Outcome of a missing-vector backfill run."""

    group_id: str
    attempted: int = 0
    indexed: int = 0
    failed: int = 0
    missing_before: int = 0
    coverage_before: float = 0.0
    coverage_after: float | None = None
    indexed_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "attempted": self.attempted,
            "indexed": self.indexed,
            "failed": self.failed,
            "missing_before": self.missing_before,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "indexed_ids": list(self.indexed_ids),
        }


def is_indexable_entity(entity: Entity | Any) -> bool:
    """Return True when an entity should hold a hybrid search vector."""
    if entity is None:
        return False
    deleted = getattr(entity, "deleted_at", None)
    if deleted is not None:
        return False
    name = str(getattr(entity, "name", "") or "").strip()
    return bool(name)


async def list_indexable_entities(
    graph_store: Any,
    group_id: str,
    *,
    limit: int = DEFAULT_ENTITY_SCAN_LIMIT,
) -> tuple[list[Entity], int, int]:
    """Load active named entities for vector completeness.

    Returns (entities, skipped_empty_name, skipped_deleted).
    """
    # Prefer uncapped group listing when the store supports high limits.
    find = getattr(graph_store, "find_entities", None)
    if not callable(find):
        return [], 0, 0

    raw = await find(group_id=group_id, limit=max(1, int(limit)))
    entities: list[Entity] = []
    skipped_empty = 0
    skipped_deleted = 0
    for entity in raw or []:
        if getattr(entity, "deleted_at", None) is not None:
            skipped_deleted += 1
            continue
        name = str(getattr(entity, "name", "") or "").strip()
        if not name:
            skipped_empty += 1
            continue
        entities.append(entity)
    return entities, skipped_empty, skipped_deleted


async def measure_entity_vector_coverage(
    graph_store: Any,
    search_index: Any,
    group_id: str,
    *,
    limit: int = DEFAULT_ENTITY_SCAN_LIMIT,
    probe_chunk: int = DEFAULT_EMBEDDING_PROBE_CHUNK,
) -> VectorCoverageReport:
    """Report which indexable entities are missing hybrid entity vectors."""
    entities, skipped_empty, skipped_deleted = await list_indexable_entities(
        graph_store,
        group_id,
        limit=limit,
    )
    report = VectorCoverageReport(
        group_id=group_id,
        entity_count=len(entities) + skipped_empty + skipped_deleted,
        indexable_count=len(entities),
        skipped_empty_name=skipped_empty,
        skipped_deleted=skipped_deleted,
    )
    if not entities:
        return report

    get_embeddings = getattr(search_index, "get_entity_embeddings", None)
    if not callable(get_embeddings):
        # Without a probe API every indexable entity is treated as missing so
        # backfill still has a deterministic plan.
        report.missing_ids = [str(e.id) for e in entities]
        report.vector_count = 0
        return report

    present: set[str] = set()
    chunk = max(1, int(probe_chunk))
    ids = [str(e.id) for e in entities]
    for i in range(0, len(ids), chunk):
        batch_ids = ids[i : i + chunk]
        try:
            found = await get_embeddings(batch_ids, group_id=group_id)
        except TypeError:
            # Some fakes/impls only accept entity_ids.
            found = await get_embeddings(batch_ids)
        except Exception:
            logger.warning(
                "get_entity_embeddings failed for chunk starting at %d",
                i,
                exc_info=True,
            )
            found = {}
        if isinstance(found, dict):
            for eid, vec in found.items():
                if vec:
                    present.add(str(eid))

    report.vector_count = len(present)
    report.missing_ids = [eid for eid in ids if eid not in present]
    return report


async def backfill_missing_entity_vectors(
    graph_store: Any,
    search_index: Any,
    group_id: str,
    *,
    max_entities: int = 200,
    batch_size: int = DEFAULT_BACKFILL_BATCH,
    dry_run: bool = False,
    remeasure: bool = False,
    limit: int = DEFAULT_ENTITY_SCAN_LIMIT,
    probe_chunk: int = DEFAULT_EMBEDDING_PROBE_CHUNK,
    missing_ids: Sequence[str] | None = None,
    entities_by_id: dict[str, Entity] | None = None,
) -> BackfillResult:
    """Index entities that lack hybrid vectors. Returns a structured result.

    When *missing_ids* is provided, skips the coverage scan and only indexes
    those IDs (entities loaded from graph or *entities_by_id*).
    """
    coverage_before = 0.0
    missing_before = 0
    planned: list[Entity] = []

    if missing_ids is not None:
        id_list = [str(x) for x in missing_ids if str(x)]
        missing_before = len(id_list)
        lookup = dict(entities_by_id or {})
        for eid in id_list[: max(0, int(max_entities))]:
            entity = lookup.get(eid)
            if entity is None and hasattr(graph_store, "get_entity"):
                entity = await graph_store.get_entity(eid, group_id)
            if entity is not None and is_indexable_entity(entity):
                planned.append(entity)
    else:
        report = await measure_entity_vector_coverage(
            graph_store,
            search_index,
            group_id,
            limit=limit,
            probe_chunk=probe_chunk,
        )
        coverage_before = report.coverage
        missing_before = report.missing_count
        by_id = {
            str(e.id): e
            for e in (await list_indexable_entities(graph_store, group_id, limit=limit))[0]
        }
        for eid in report.missing_ids[: max(0, int(max_entities))]:
            entity = by_id.get(eid)
            if entity is not None:
                planned.append(entity)

    result = BackfillResult(
        group_id=group_id,
        attempted=len(planned),
        missing_before=missing_before,
        coverage_before=coverage_before,
    )
    if not planned or dry_run:
        return result

    batch_n = max(1, int(batch_size))
    use_batch = hasattr(search_index, "batch_index_entities")
    indexed_ids: list[str] = []
    failed = 0

    for i in range(0, len(planned), batch_n):
        chunk = planned[i : i + batch_n]
        if use_batch:
            try:
                count = await search_index.batch_index_entities(chunk)
                count_i = int(count or 0)
                # Best-effort: assume leading entities succeeded when count < len.
                for entity in chunk[:count_i]:
                    indexed_ids.append(str(entity.id))
                if count_i < len(chunk):
                    failed += len(chunk) - count_i
                continue
            except Exception:
                logger.warning(
                    "batch_index_entities failed; falling back to one-at-a-time",
                    exc_info=True,
                )
                use_batch = False

        for entity in chunk:
            try:
                await search_index.index_entity(entity)
                indexed_ids.append(str(entity.id))
            except Exception:
                failed += 1
                logger.warning(
                    "index_entity failed for %s",
                    getattr(entity, "id", "?"),
                    exc_info=True,
                )

    result.indexed = len(indexed_ids)
    result.failed = failed
    result.indexed_ids = indexed_ids

    if remeasure:
        after = await measure_entity_vector_coverage(
            graph_store,
            search_index,
            group_id,
            limit=limit,
            probe_chunk=probe_chunk,
        )
        result.coverage_after = after.coverage

    return result
