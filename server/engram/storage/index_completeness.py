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
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from engram.models.entity import Entity

logger = logging.getLogger(__name__)

# Safety cap when listing entities for coverage scans.
DEFAULT_ENTITY_SCAN_LIMIT = 50_000
DEFAULT_EMBEDDING_PROBE_CHUNK = 64
DEFAULT_BACKFILL_BATCH = 32
# Safety caps when listing episodes/cues for vector-debt drains.
DEFAULT_EPISODE_SCAN_LIMIT = 50_000
# ANN census sweep size when the index has no by-id embedding probe.
VECTOR_CENSUS_K = 50_000
# Cue drain: an episode younger than this with no cue may simply not have its
# cue written yet (cue capture is same-session, best-effort) — the cursor must
# not advance past it or the cue would be stranded forever. Older and cueless
# is final: nothing will ever write that cue.
CUE_CURSOR_YOUNG_EPISODE_GRACE_SECONDS = 3600.0


class EmbeddingProviderUnavailableError(RuntimeError):
    """The backfill has work to do but the embedding provider cannot embed.

    Raised LOUDLY instead of silently producing a vector-less "success" —
    the M2.6 outage happened because a broken provider was invisible.
    """


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
    # Episode/cue drains: durable progression cursor (created_ts, item_id) of
    # the newest successfully indexed item, set only when vector presence was
    # measured inexactly (ANN census / no probe). Not part of to_dict().
    cursor_next: tuple[float, str] | None = None

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


async def ensure_embedding_provider_healthy(search_index: Any) -> None:
    """Raise :class:`EmbeddingProviderUnavailableError` when embedding cannot work.

    Probes the search index's provider with a single embed call. When the
    index exposes no probe surface the check is skipped and per-item failure
    accounting takes over.
    """
    if getattr(search_index, "_embeddings_enabled", None) is False:
        raise EmbeddingProviderUnavailableError("embeddings disabled on search index")
    embed = getattr(search_index, "_embed_texts", None)
    if not callable(embed):
        return
    try:
        vecs = await embed(["engram vector backfill provider probe"])
    except Exception as exc:
        raise EmbeddingProviderUnavailableError(f"embedding probe raised: {exc}") from exc
    if not vecs or not vecs[0]:
        raise EmbeddingProviderUnavailableError("embedding probe returned empty")


def _created_ts(value: Any) -> float:
    """Best-effort epoch seconds from a created_at datetime/ISO string/number."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        try:
            return value.timestamp()
        except (OverflowError, OSError, ValueError):  # silent-ok: cursor key fallback, epoch 0
            return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:  # silent-ok: cursor key fallback, epoch 0
        return 0.0


async def _ids_with_vectors(
    search_index: Any,
    ids: list[str],
    group_id: str,
    *,
    probe_attr: str,
    census_attr: str,
    probe_chunk: int = DEFAULT_EMBEDDING_PROBE_CHUNK,
) -> tuple[set[str] | None, bool]:
    """Return (ids that already hold vectors, presence_is_exact).

    Prefers a by-id embedding probe (``probe_attr``, exact), falls back to a
    single ANN census sweep (``census_attr``, one embed call — INEXACT: on
    helix-native HNSW a single-probe sweep surfaces only a small reachable
    subset regardless of k, measured 39 visible of 3000 written). Returns
    ``(None, False)`` when the index exposes neither.
    """
    probe = getattr(search_index, probe_attr, None)
    if callable(probe):
        present: set[str] = set()
        chunk = max(1, int(probe_chunk))
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            try:
                found = await probe(batch, group_id=group_id)
            except TypeError:
                # Some fakes/impls only accept the id list.
                found = await probe(batch)
            except Exception:
                logger.warning("%s failed for chunk starting at %d", probe_attr, i, exc_info=True)
                found = {}
            if isinstance(found, dict):
                present.update(str(k) for k, v in found.items() if v)
        return present, True

    embed_one = getattr(search_index, "_embed_text", None)
    census = getattr(search_index, census_attr, None)
    if callable(embed_one) and callable(census):
        try:
            vec = await embed_one("engram vector census probe")
        except Exception as exc:
            raise EmbeddingProviderUnavailableError(f"census embed raised: {exc}") from exc
        if not vec:
            raise EmbeddingProviderUnavailableError("census embed returned empty vector")
        scored, _rows, _filtered = await census(vec, VECTOR_CENSUS_K, group_id)
        return {str(i) for i, _score in scored}, False

    return None, False


async def backfill_missing_episode_vectors(
    graph_store: Any,
    search_index: Any,
    group_id: str,
    *,
    max_episodes: int = 400,
    dry_run: bool = False,
    cursor: tuple[float, str] | None = None,
    limit: int = DEFAULT_EPISODE_SCAN_LIMIT,
    probe_chunk: int = DEFAULT_EMBEDDING_PROBE_CHUNK,
) -> BackfillResult:
    """Embed and index episodes that lack EpisodeVec vectors.

    Capture-time indexing (capture_service) is the primary episode-vector
    writer; this drain is the safety net for capture-time failures and
    pre-existing debt. It lists missing under budget, embeds via the provider
    (through ``search_index.index_episode``), writes vectors, returns counts.

    When presence is measured inexactly (ANN census undercounts on
    helix-native), *cursor* — a persisted ``(created_ts, episode_id)`` high
    water mark — guarantees progression: items are drained oldest-first,
    strictly after the cursor, and ``result.cursor_next`` reports the new mark
    (only successful indexing advances it). Without the cursor the drain would
    re-embed the same first budget-window forever and grow duplicate vectors.
    """
    result = BackfillResult(group_id=group_id)
    get_episodes = getattr(graph_store, "get_episodes", None)
    if not callable(get_episodes):
        return result

    episodes = await get_episodes(group_id=group_id, limit=max(1, int(limit))) or []
    indexable = [
        ep
        for ep in episodes
        if getattr(ep, "deleted_at", None) is None
        and str(getattr(ep, "id", "") or "")
        and str(getattr(ep, "content", "") or "").strip()
    ]
    if not indexable:
        return result

    ids = [str(ep.id) for ep in indexable]
    present, presence_exact = await _ids_with_vectors(
        search_index,
        ids,
        group_id,
        probe_attr="get_episode_embeddings",
        census_attr="_vector_search_episodes",
        probe_chunk=probe_chunk,
    )
    if present is None:
        # No presence probe at all: treat everything as missing so the drain
        # still has a deterministic plan (mirrors the entity backfill).
        missing = list(indexable)
    else:
        missing = [ep for ep in indexable if str(ep.id) not in present]
    result.missing_before = len(missing)

    keyed = sorted(
        ((_created_ts(getattr(ep, "created_at", None)), str(ep.id)), ep) for ep in missing
    )
    if not presence_exact and cursor is not None:
        keyed = [(key, ep) for key, ep in keyed if key > cursor]
    keyed = keyed[: max(0, int(max_episodes))]
    result.attempted = len(keyed)
    if not keyed or dry_run:
        return result

    await ensure_embedding_provider_healthy(search_index)

    # index_episode implementations swallow embed failures internally (log +
    # stat instead of raise); track the index's own failure counter so a
    # half-broken provider can never report a clean drain.
    stats = getattr(search_index, "_embed_stats", None)
    stat_failed_before = int(stats.get("episodes_failed", 0)) if isinstance(stats, dict) else None

    indexed_ids: list[str] = []
    indexed_keys: list[tuple[float, str]] = []
    failed = 0
    for key, ep in keyed:
        try:
            await search_index.index_episode(ep)
            indexed_ids.append(str(ep.id))
            indexed_keys.append(key)
        except Exception:
            failed += 1
            logger.warning(
                "index_episode failed for %s",
                getattr(ep, "id", "?"),
                exc_info=True,
            )

    if stat_failed_before is not None and isinstance(stats, dict):
        swallowed = max(0, int(stats.get("episodes_failed", 0)) - stat_failed_before)
        swallowed = min(swallowed, len(indexed_ids))
        if swallowed:
            # Best-effort: which ids failed is unknown; drop trailing ones.
            failed += swallowed
            indexed_ids = indexed_ids[: len(indexed_ids) - swallowed]
            indexed_keys = indexed_keys[: len(indexed_ids)]

    result.indexed = len(indexed_ids)
    result.failed = failed
    result.indexed_ids = indexed_ids
    if not presence_exact and indexed_keys:
        result.cursor_next = max(indexed_keys)
    return result


async def backfill_missing_cue_vectors(
    graph_store: Any,
    search_index: Any,
    group_id: str,
    *,
    max_cues: int = 400,
    dry_run: bool = False,
    cursor: tuple[float, str] | None = None,
    probe_chunk: int = DEFAULT_EMBEDDING_PROBE_CHUNK,
    limit: int = DEFAULT_EPISODE_SCAN_LIMIT,
) -> BackfillResult:
    """Embed and index episode cues that lack CueVec vectors.

    Cue indexing at capture is best-effort (outbox replay only covers shell
    uptime), so coverage decays without a drain. Vectors are keyed by
    episode_id.

    LISTING IS EPISODE-BASED AND BOUNDED. The native bulk cue listing
    (``find_cues_by_group``) takes only ``gid`` — no server-side k/limit —
    and measured 20s+ on an 8.7k-cue brain as a loop-blocking sync native
    call that ignores deadlines, so the drain must never issue it. Instead:
    list episodes (bounded, proven fast), walk them oldest-first from the
    persisted cursor, and probe each episode's cue by id
    (``get_episode_cue``, ~74ms native). ``max_cues`` bounds the PROBES per
    window, which also caps embeds, so a window cannot grind.

    Cursor: ``(created_ts, episode_id)`` of the newest episode with a FINAL
    outcome — cue indexed, cue already vectored, or old-and-cueless. It only
    advances over a contiguous prefix of final outcomes, so failures are
    retried next window and nothing is stranded; young cueless episodes
    (< :data:`CUE_CURSOR_YOUNG_EPISODE_GRACE_SECONDS`) stop advancement
    because their cue may still be written. NOTE: the key is the EPISODE's
    created_at (earlier versions keyed on the cue row's created_at); old
    persisted cursors stay structurally valid — at worst a few boundary
    episodes are re-probed once.

    ``missing_before`` is WINDOW-SCOPED (missing cues found by this window's
    probes); a global count would require the unbounded listing.
    """
    result = BackfillResult(group_id=group_id)
    get_episodes = getattr(graph_store, "get_episodes", None)
    get_cue = getattr(graph_store, "get_episode_cue", None)
    if not callable(get_episodes) or not callable(get_cue):
        return result

    episodes = await get_episodes(group_id=group_id, limit=max(1, int(limit))) or []
    candidates = sorted(
        ((_created_ts(getattr(ep, "created_at", None)), str(ep.id)), ep)
        for ep in episodes
        if getattr(ep, "deleted_at", None) is None and str(getattr(ep, "id", "") or "")
    )
    presence_probe_exact = callable(getattr(search_index, "get_cue_embeddings", None))
    if not presence_probe_exact and cursor is not None:
        candidates = [(key, ep) for key, ep in candidates if key > cursor]
    if not candidates:
        return result

    present, presence_exact = await _ids_with_vectors(
        search_index,
        [key[1] for key, _ep in candidates],
        group_id,
        probe_attr="get_cue_embeddings",
        census_attr="_vector_search_cues",
        probe_chunk=probe_chunk,
    )
    present = present or set()

    max_probes = max(0, int(max_cues))
    grace_cutoff = time.time() - CUE_CURSOR_YOUNG_EPISODE_GRACE_SECONDS
    provider_checked = False
    probes = 0
    indexed_ids: list[str] = []
    failed = 0
    missing_found = 0
    attempted = 0
    prefix_intact = True
    cursor_candidate: tuple[float, str] | None = None

    for key, _ep in candidates:
        episode_id = key[1]
        if episode_id in present:
            if prefix_intact:
                cursor_candidate = key
            continue
        if probes >= max_probes:
            break
        probes += 1
        try:
            cue = await get_cue(episode_id, group_id)
        except Exception:
            failed += 1
            prefix_intact = False
            logger.warning(
                "cue probe (get_episode_cue) failed for %s — cue window stopped early",
                episode_id,
                exc_info=True,
            )
            break
        if cue is None or not str(getattr(cue, "cue_text", "") or "").strip():
            if key[0] >= grace_cutoff:
                # Young and cueless: capture may still write this cue.
                # Everything after is younger (sorted) — stop the window.
                break
            if prefix_intact:
                cursor_candidate = key
            continue
        missing_found += 1
        attempted += 1
        if dry_run:
            continue
        if not provider_checked:
            await ensure_embedding_provider_healthy(search_index)
            provider_checked = True
        try:
            await search_index.index_episode_cue(cue)
            indexed_ids.append(episode_id)
            if prefix_intact:
                cursor_candidate = key
        except Exception:
            failed += 1
            prefix_intact = False
            logger.warning("index_episode_cue failed for %s", episode_id, exc_info=True)

    result.missing_before = missing_found
    result.attempted = attempted
    result.indexed = len(indexed_ids)
    result.failed = failed
    result.indexed_ids = indexed_ids
    if not presence_exact and not dry_run and cursor_candidate is not None:
        result.cursor_next = cursor_candidate
    return result
