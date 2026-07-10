"""Multi-pool candidate generation: search + activation + graph + working memory + entity query."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING, TypeVar

from engram.config import ActivationConfig
from engram.retrieval.router import QueryType

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer
    from engram.storage.protocols import ActivationStore, GraphStore, SearchIndex

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Dynamic pool sizing
# ---------------------------------------------------------------------------

# Query-type-specific pool multipliers: (search, activation, graph)
_POOL_MULTIPLIERS: dict[QueryType, tuple[float, float, float]] = {
    QueryType.TEMPORAL: (1.0, 3.0, 1.0),
    QueryType.FREQUENCY: (1.0, 3.0, 1.0),
    QueryType.DIRECT_LOOKUP: (2.0, 1.0, 1.0),
    QueryType.ASSOCIATIVE: (1.0, 1.0, 2.0),
    QueryType.CREATION: (1.0, 1.0, 2.0),
    QueryType.DEFAULT: (1.0, 1.0, 1.0),
}


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(value, hi))


def compute_dynamic_limits(
    total_entities: int,
    cfg: ActivationConfig,
    query_type: QueryType | None = None,
) -> dict[str, int]:
    """Scale pool limits with sqrt(total_entities / 1000).

    At 1k entities the limits equal cfg defaults. At 5k they grow ~2.2x.
    Query-type multipliers further adjust individual pools.
    All values are clamped to config field constraints.
    """
    scale = math.sqrt(max(total_entities, 1000) / 1000.0)

    search_mul, act_mul, graph_mul = _POOL_MULTIPLIERS.get(
        query_type or QueryType.DEFAULT,
        (1.0, 1.0, 1.0),
    )

    pool_search = _clamp(int(cfg.pool_search_limit * scale * search_mul), 5, 200)
    pool_activation = _clamp(int(cfg.pool_activation_limit * scale * act_mul), 5, 100)
    pool_graph_limit = _clamp(int(cfg.pool_graph_limit * scale * graph_mul), 5, 100)
    pool_graph_seed = _clamp(int(cfg.pool_graph_seed_count * scale), 1, 50)
    pool_graph_neighbors = _clamp(int(cfg.pool_graph_max_neighbors * scale), 1, 50)
    pool_wm = _clamp(int(cfg.pool_wm_limit * scale), 5, 50)
    pool_total = _clamp(
        pool_search + pool_activation + pool_graph_limit + pool_wm,
        20,
        1000,
    )

    return {
        "pool_search_limit": pool_search,
        "pool_activation_limit": pool_activation,
        "pool_graph_seed_count": pool_graph_seed,
        "pool_graph_max_neighbors": pool_graph_neighbors,
        "pool_graph_limit": pool_graph_limit,
        "pool_wm_limit": pool_wm,
        "pool_total_limit": pool_total,
    }


async def _search_pool(
    query: str,
    group_id: str,
    search_index: SearchIndex,
    limit: int,
    *,
    timeout_seconds: float | None = None,
    stage_timings_ms: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Pool 1: semantic/FTS search candidates."""
    started = time.perf_counter()
    try:
        search_call = search_index.search(
            query=query,
            group_id=group_id,
            limit=limit,
        )
        results = (
            await asyncio.wait_for(search_call, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await search_call
        )
        _add_stage_timing(stage_timings_ms, "recall_primary_search", started)
        _add_stage_timing(stage_timings_ms, "recall_embed", started)
        return results or []
    except asyncio.TimeoutError:
        _add_stage_timing(stage_timings_ms, "recall_primary_search_timeout", started)
        return []
    except asyncio.CancelledError:
        _add_stage_timing(stage_timings_ms, "recall_primary_search_cancelled", started)
        raise
    except Exception as e:
        logger.warning("Search pool failed (non-fatal): %s", e)
        return []


async def _activation_pool(
    group_id: str,
    activation_store: ActivationStore,
    limit: int,
    now: float,
) -> list[tuple[str, float]]:
    """Pool 2: top activated entities by ACT-R base-level activation.

    get_top_activated already sorts by computed activation internally.
    We re-compute here to get the actual score for RRF ranking.
    """
    try:
        from engram.activation.engine import compute_activation
        from engram.config import ActivationConfig

        top = await activation_store.get_top_activated(
            group_id=group_id,
            limit=limit,
            now=now,
        )
        cfg = ActivationConfig()
        results: list[tuple[str, float]] = []
        for eid, state in top:
            act = compute_activation(state.access_history, now, cfg)
            results.append((eid, act))
        return results
    except Exception as e:
        logger.warning("Activation pool failed (non-fatal): %s", e)
        return []


async def _graph_neighborhood_pool(
    seed_ids: list[str],
    group_id: str,
    graph_store: GraphStore,
    max_neighbors: int,
    pool_limit: int,
) -> list[tuple[str, float]]:
    """Pool 3: 1-hop neighbors of top search seeds, ranked by fan-in count."""
    try:
        if not seed_ids:
            return []

        # Count how many seeds each neighbor connects to (fan-in)
        fan_in: dict[str, int] = {}
        seed_set = set(seed_ids)

        for sid in seed_ids:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                entity_id=sid,
                group_id=group_id,
            )
            for neighbor in neighbors[:max_neighbors]:
                nid = neighbor[0]
                if nid not in seed_set:
                    fan_in[nid] = fan_in.get(nid, 0) + 1

        # Sort by fan-in descending, take top pool_limit
        ranked = sorted(fan_in.items(), key=lambda x: (-x[1], x[0]))
        return [(nid, float(count)) for nid, count in ranked[:pool_limit]]
    except Exception as e:
        logger.warning("Graph neighborhood pool failed (non-fatal): %s", e)
        return []


async def _working_memory_pool(
    working_memory: WorkingMemoryBuffer,
    group_id: str,
    graph_store: GraphStore,
    now: float,
    max_neighbors: int,
    pool_limit: int,
) -> list[tuple[str, float]]:
    """Pool 4: working memory entities + dampened 1-hop neighbors."""
    try:
        wm_candidates = working_memory.get_candidates(now)
        if not wm_candidates:
            return []

        results: dict[str, float] = {}
        wm_ids: set[str] = set()

        # Add WM entities with their recency score
        for item_id, recency_score, _item_type in wm_candidates:
            results[item_id] = recency_score
            wm_ids.add(item_id)

        # Expand 1-hop neighbors with 0.5x dampening
        for item_id, recency_score, _item_type in wm_candidates:
            neighbors = await graph_store.get_active_neighbors_with_weights(
                entity_id=item_id,
                group_id=group_id,
            )
            for neighbor in neighbors[:max_neighbors]:
                nid = neighbor[0]
                if nid not in wm_ids:
                    dampened = recency_score * 0.5
                    # Keep the higher score if already seen
                    if nid not in results or dampened > results[nid]:
                        results[nid] = dampened

        # Sort by score descending, take top pool_limit
        ranked = sorted(results.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:pool_limit]
    except Exception as e:
        logger.warning("Working memory pool failed (non-fatal): %s", e)
        return []


# ---------------------------------------------------------------------------
# Entity name patterns for query extraction
# ---------------------------------------------------------------------------

# Consecutive capitalized words (e.g., "Kansas City Masterpiece", "Dell XPS 13")
_TITLE_CASE_PHRASE = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|\d+)){1,4}\b")
# Single capitalized word that isn't sentence-initial (e.g., "Instagram")
_SINGLE_CAP_WORD = re.compile(r"(?<!\.\s)(?<!^)\b[A-Z][a-z]{2,}\b")
# Quoted strings
_QUOTED_STRING = re.compile(r'"([^"]{2,})"')
# Words after "my" or "the" (possessive/definite references)
_POSSESSIVE_NOUN = re.compile(
    r"\b(?:my|the)\s+([a-zA-Z][a-zA-Z0-9 ]{1,30}?)"
    r"(?:\s+(?:is|was|are|were|do|does|did|has|have|had|that|which|who)\b"
    r"|[?.!,;]|$)",
    re.IGNORECASE,
)
# Common stop words to filter out
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "that",
        "this",
        "these",
        "those",
        "it",
        "they",
        "them",
        "we",
        "you",
        "i",
        "me",
        "he",
        "she",
        "and",
        "or",
        "but",
        "not",
        "no",
        "if",
        "about",
        "with",
        "from",
        "for",
        "on",
        "in",
        "at",
        "to",
        "of",
        "by",
        "all",
        "some",
        "any",
        "most",
        "many",
        "much",
        "few",
        "more",
        "less",
        "very",
        "just",
        "also",
        "too",
        "so",
        "than",
        "then",
        "now",
        "like",
        "know",
        "think",
        "want",
        "need",
        "use",
        "tell",
        "say",
        "favorite",
        "favourite",
        "prefer",
        "preferred",
        "best",
        "worst",
    }
)


def _extract_entity_names_from_query(query: str) -> list[str]:
    """Extract potential entity names from a query using simple heuristics.

    Returns deduplicated candidate names ordered by extraction confidence:
    0. Full query (exact Decision-name lookup — product continuity path)
    1. Quoted strings (highest confidence)
    2. Title-case phrases (multi-word capitalized sequences)
    3. Single capitalized words (not sentence-initial)
    4. Noun phrases after "my"/"the" (possessive/definite references)
    """
    seen: set[str] = set()
    candidates: list[str] = []

    def _add(name: str) -> None:
        cleaned = name.strip().strip("?!.,;:'\"")
        key = cleaned.lower()
        if len(key) < 2 or key in seen or key in _STOP_WORDS:
            return
        # Skip if every word is a stop word
        words = key.split()
        if all(w in _STOP_WORDS for w in words):
            return
        seen.add(key)
        candidates.append(cleaned)

    # 0. Full query as name candidate (exact Decision / Preference recall).
    # Without this, multi-word Decision names never hit find_entity_candidates
    # when title-case heuristics miss lowercase content words.
    stripped = " ".join((query or "").split())
    if 3 <= len(stripped) <= 200:
        _add(stripped)

    # 1. Quoted strings (highest confidence)
    for match in _QUOTED_STRING.finditer(query):
        _add(match.group(1))

    # 2. Title-case phrases (multi-word capitalized sequences)
    for match in _TITLE_CASE_PHRASE.finditer(query):
        _add(match.group(0))

    # 3. Single capitalized words (not sentence-initial)
    for match in _SINGLE_CAP_WORD.finditer(query):
        _add(match.group(0))

    # 4. Noun phrases after "my"/"the"
    for match in _POSSESSIVE_NOUN.finditer(query):
        phrase = match.group(1).strip()
        # Take only meaningful words, drop trailing stop words
        words = phrase.split()
        while words and words[-1].lower() in _STOP_WORDS:
            words.pop()
        if words:
            _add(" ".join(words))

    return candidates


def _name_match_score(query_name: str, entity_name: str) -> float:
    """Score how well a query-extracted name matches an entity name.

    Returns a score in [0.0, 1.0] reflecting match quality.
    """
    q = query_name.lower().strip()
    e = entity_name.lower().strip()
    if not q or not e:
        return 0.0
    # Exact match
    if q == e:
        return 1.0
    # One contains the other
    if q in e or e in q:
        shorter = min(len(q), len(e))
        longer = max(len(q), len(e))
        return 0.6 + 0.3 * (shorter / longer)
    # Token overlap
    q_tokens = set(q.split())
    e_tokens = set(e.split())
    if q_tokens and e_tokens:
        overlap = len(q_tokens & e_tokens)
        if overlap > 0:
            precision = overlap / len(q_tokens)
            recall = overlap / len(e_tokens)
            return 0.4 + 0.3 * (2 * precision * recall / (precision + recall))
    return 0.0


async def _entity_query_pool(
    query: str,
    group_id: str,
    graph_store: GraphStore,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Pool 5: entities matched by extracting names from the query text.

    Extracts potential entity names using heuristics (title-case phrases,
    quoted strings, possessive nouns), looks them up via
    ``find_entity_candidates()``, and returns matched entity IDs with
    scores reflecting name match quality.
    """
    try:
        candidate_names = _extract_entity_names_from_query(query)
        if not candidate_names:
            return []

        # Collect matched entities with best score per entity
        entity_scores: dict[str, float] = {}

        for name in candidate_names:
            try:
                entities = await graph_store.find_entity_candidates(
                    name,
                    group_id,
                    limit=5,
                )
            except Exception:
                continue

            for entity in entities or []:
                score = _name_match_score(name, entity.name)
                if score > 0.0:
                    # Keep the best score if entity found via multiple query names
                    if entity.id not in entity_scores or score > entity_scores[entity.id]:
                        entity_scores[entity.id] = score

        if not entity_scores:
            return []

        # Sort by score descending, take top limit
        ranked = sorted(entity_scores.items(), key=lambda x: (-x[1], x[0]))
        return ranked[:limit]
    except Exception as e:
        logger.warning("Entity query pool failed (non-fatal): %s", e)
        return []


def _merge_pools_rrf(
    pools: list[list[tuple[str, float]]],
    rrf_k: int,
    limit: int,
) -> list[str]:
    """Merge ranked lists via Reciprocal Rank Fusion.

    score(d) = sum over pools: 1 / (rrf_k + rank_i(d))
    Returns entity IDs ordered by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}

    for pool in pools:
        for rank, (eid, _score) in enumerate(pool):
            rrf_scores[eid] = rrf_scores.get(eid, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    return [eid for eid, _ in ranked[:limit]]


def _add_stage_timing(
    stage_timings_ms: dict[str, float] | None,
    key: str,
    started: float,
) -> None:
    if stage_timings_ms is None:
        return
    elapsed = round((time.perf_counter() - started) * 1000, 4)
    stage_timings_ms[key] = round(stage_timings_ms.get(key, 0.0) + elapsed, 4)


def _set_stage_metric(
    stage_timings_ms: dict[str, float] | None,
    key: str,
    value: int | float,
) -> None:
    if stage_timings_ms is None:
        return
    stage_timings_ms[key] = round(float(value), 4)


def _primary_search_timeout_seconds(
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float] | None = None,
) -> float | None:
    """Timeout for primary hybrid search.

    Product rule: stats/graph preflight timeouts must **not** poison explicit
    search into a 100ms no-op on large native brains. Prefer the configured
    primary timeout, and when probes already timed out allow at least the
    explicit search budget so name/BM25 can still return candidates.
    """
    timeout_ms = int(getattr(cfg, "retrieval_primary_search_timeout_ms", 0) or 0)
    if timeout_ms <= 0:
        return None
    probe_timed_out = bool(
        stage_timings_ms
        and (
            "recall_stats_timeout" in stage_timings_ms or "graph_expand_timeout" in stage_timings_ms
        )
    )
    if probe_timed_out:
        # Floor (not ceiling): after a failed stats probe, give search at least
        # the explicit search budget so large brains can still hit BM25/name.
        explicit_floor_ms = int(
            getattr(cfg, "recall_budget_explicit_search_ms", 1500) or 1500
        )
        adaptive_ms = int(
            getattr(cfg, "retrieval_primary_search_timeout_after_probe_timeout_ms", 0) or 0
        )
        # Use the max of primary, explicit floor, and (if set) adaptive value —
        # never shrink primary below a workable product search budget.
        timeout_ms = max(timeout_ms, explicit_floor_ms, adaptive_ms)
    _set_stage_metric(
        stage_timings_ms,
        "recall_primary_search_effective_timeout_ms",
        timeout_ms,
    )
    return timeout_ms / 1000.0


def _graph_probe_timed_out(stage_timings_ms: dict[str, float] | None) -> bool:
    return bool(
        stage_timings_ms
        and (
            "recall_stats_timeout" in stage_timings_ms or "graph_expand_timeout" in stage_timings_ms
        )
    )


def _stage_timeout_seconds(cfg: ActivationConfig, field_name: str) -> float | None:
    timeout_ms = int(getattr(cfg, field_name, 0) or 0)
    if timeout_ms <= 0:
        return None
    return timeout_ms / 1000.0


def _graph_pool_timeout_seconds(
    cfg: ActivationConfig,
    *,
    budget_profile: str | None = None,
) -> float | None:
    """Return graph-pool timeout; auto profiles use the relaxed auto budget."""
    if budget_profile in {"auto_lite", "auto_deep", "startup"}:
        timeout_ms = int(getattr(cfg, "retrieval_graph_pool_timeout_auto_ms", 0) or 0)
    else:
        timeout_ms = int(getattr(cfg, "retrieval_graph_pool_timeout_ms", 0) or 0)
    if timeout_ms <= 0:
        return None
    return timeout_ms / 1000.0


async def _bounded_pool(
    awaitable: Awaitable[T],
    *,
    timeout_seconds: float | None,
    stage_timings_ms: dict[str, float] | None,
    stage_key: str,
    timeout_key: str,
    cancelled_key: str,
    fallback: T,
) -> T:
    started = time.perf_counter()
    try:
        result = (
            await asyncio.wait_for(awaitable, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await awaitable
        )
        _add_stage_timing(stage_timings_ms, stage_key, started)
        return result
    except asyncio.TimeoutError:
        _add_stage_timing(stage_timings_ms, timeout_key, started)
        return fallback
    except asyncio.CancelledError:
        _add_stage_timing(stage_timings_ms, cancelled_key, started)
        raise
    except Exception as e:
        logger.warning("%s failed (non-fatal): %s", stage_key, e)
        return fallback


async def generate_candidates(
    query: str,
    group_id: str,
    search_index: SearchIndex,
    activation_store: ActivationStore,
    graph_store: GraphStore,
    cfg: ActivationConfig,
    now: float | None = None,
    working_memory: WorkingMemoryBuffer | None = None,
    total_entities: int = 0,
    query_type: QueryType | None = None,
    stage_timings_ms: dict[str, float] | None = None,
    name_match_out: dict[str, float] | None = None,
    budget_profile: str | None = None,
) -> list[tuple[str, float]]:
    """Orchestrate multi-pool candidate generation.

    Returns (entity_id, real_semantic_similarity) tuples in RRF-merged order.
    Pool sizes scale with sqrt(total_entities / 1000) and query-type multipliers.

    If ``name_match_out`` is provided, it is populated with
    ``{entity_id: name_match_score}`` from the entity-query pool so the caller
    can seed graph traversal from deterministically name-resolved entities
    (the returned tuples carry only semantic similarity, which must not be
    polluted by name-match scores).
    """
    if now is None:
        now = time.time()

    # Compute dynamic pool limits based on corpus size and query type
    limits = compute_dynamic_limits(total_entities, cfg, query_type)

    # Step 1: Run search + activation + entity query pools concurrently
    gather_tasks: list = [
        _search_pool(
            query,
            group_id,
            search_index,
            limits["pool_search_limit"],
            timeout_seconds=_primary_search_timeout_seconds(cfg, stage_timings_ms),
            stage_timings_ms=stage_timings_ms,
        ),
        _bounded_pool(
            _activation_pool(
                group_id,
                activation_store,
                limits["pool_activation_limit"],
                now,
            ),
            timeout_seconds=_stage_timeout_seconds(
                cfg,
                "retrieval_activation_pool_timeout_ms",
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_activation_pool",
            timeout_key="recall_activation_pool_timeout",
            cancelled_key="recall_activation_pool_cancelled",
            fallback=[],
        ),
    ]
    run_entity_query = cfg.entity_query_retrieval_enabled and hasattr(
        graph_store, "find_entity_candidates"
    )
    if run_entity_query:
        gather_tasks.append(
            _entity_query_pool(
                query,
                group_id,
                graph_store,
                limit=limits.get("pool_entity_query_limit", cfg.pool_entity_query_limit),
            )
        )

    gathered = await asyncio.gather(*gather_tasks)
    search_results: list[tuple[str, float]] = gathered[0]
    activation_results: list[tuple[str, float]] = gathered[1]
    entity_query_results: list[tuple[str, float]] = gathered[2] if run_entity_query else []
    _set_stage_metric(stage_timings_ms, "recall_search_candidate_count", len(search_results))
    _set_stage_metric(
        stage_timings_ms,
        "recall_search_candidate_max_score",
        max((score for _eid, score in search_results), default=0.0),
    )
    _set_stage_metric(
        stage_timings_ms,
        "recall_activation_candidate_count",
        len(activation_results),
    )
    _set_stage_metric(
        stage_timings_ms,
        "recall_entity_query_candidate_count",
        len(entity_query_results),
    )
    if name_match_out is not None:
        for eid, name_score in entity_query_results:
            if name_score > name_match_out.get(eid, 0.0):
                name_match_out[eid] = name_score

    # Step 2: Graph neighborhood from top search seeds (sequential)
    seed_ids = [eid for eid, score in search_results if score >= cfg.seed_threshold][
        : limits["pool_graph_seed_count"]
    ]
    skip_secondary_graph = bool(
        search_results
        and cfg.retrieval_skip_secondary_graph_after_probe_timeout
        and _graph_probe_timed_out(stage_timings_ms)
    )
    if skip_secondary_graph:
        graph_results = []
        _set_stage_metric(stage_timings_ms, "recall_graph_pool_skipped_probe_timeout", 0.0)
    else:
        graph_results = await _bounded_pool(
            _graph_neighborhood_pool(
                seed_ids,
                group_id,
                graph_store,
                limits["pool_graph_max_neighbors"],
                limits["pool_graph_limit"],
            ),
            timeout_seconds=_graph_pool_timeout_seconds(
                cfg,
                budget_profile=budget_profile,
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_graph_pool",
            timeout_key="recall_graph_pool_timeout",
            cancelled_key="recall_graph_pool_cancelled",
            fallback=[],
        )
    _set_stage_metric(stage_timings_ms, "recall_graph_candidate_count", len(graph_results))

    # Step 3: Working memory pool (if provided)
    wm_results: list[tuple[str, float]] = []
    if working_memory is not None and cfg.working_memory_enabled:
        wm_results = await _working_memory_pool(
            working_memory,
            group_id,
            graph_store,
            now,
            cfg.pool_wm_max_neighbors,
            limits["pool_wm_limit"],
        )
    _set_stage_metric(
        stage_timings_ms,
        "recall_working_memory_candidate_count",
        len(wm_results),
    )
    primary_search_timed_out = bool(
        stage_timings_ms and "recall_primary_search_timeout" in stage_timings_ms
    )
    activation_only_after_primary_timeout = (
        primary_search_timed_out
        and bool(activation_results)
        and not search_results
        and not entity_query_results
        and not graph_results
        and not wm_results
        and (query_type or QueryType.DEFAULT) not in {QueryType.TEMPORAL, QueryType.FREQUENCY}
        and cfg.retrieval_activation_only_primary_timeout_short_circuit
    )
    if activation_only_after_primary_timeout:
        _set_stage_metric(
            stage_timings_ms,
            "recall_activation_only_primary_timeout_short_circuit",
            len(activation_results),
        )
        _set_stage_metric(stage_timings_ms, "recall_candidate_count", 0)
        _set_stage_metric(stage_timings_ms, "recall_candidate_max_score", 0.0)
        return []

    # Step 4: Merge non-empty pools via RRF
    pools = [
        p
        for p in [
            search_results,
            activation_results,
            graph_results,
            wm_results,
            entity_query_results,
        ]
        if p
    ]
    if not pools:
        return []

    merged_ids = _merge_pools_rrf(pools, cfg.rrf_k, limits["pool_total_limit"])

    # Step 5: Build semantic score map from search results
    search_scores: dict[str, float] = {eid: score for eid, score in search_results}

    # Backfill real semantic scores for non-search entities
    non_search_ids = [eid for eid in merged_ids if eid not in search_scores]
    backfilled: dict[str, float] = {}
    skip_similarity_backfill = bool(
        primary_search_timed_out
        and not search_results
        and cfg.retrieval_skip_similarity_backfill_after_primary_timeout
    )
    if non_search_ids and skip_similarity_backfill:
        _set_stage_metric(
            stage_timings_ms,
            "recall_similarity_backfill_skipped_primary_timeout",
            len(non_search_ids),
        )
    elif non_search_ids:
        backfilled = await _bounded_pool(
            search_index.compute_similarity(
                query=query,
                entity_ids=non_search_ids,
                group_id=group_id,
            ),
            timeout_seconds=_stage_timeout_seconds(
                cfg,
                "retrieval_similarity_backfill_timeout_ms",
            ),
            stage_timings_ms=stage_timings_ms,
            stage_key="recall_similarity_backfill",
            timeout_key="recall_similarity_backfill_timeout",
            cancelled_key="recall_similarity_backfill_cancelled",
            fallback={},
        )
    _set_stage_metric(stage_timings_ms, "recall_candidate_count", len(merged_ids))
    _set_stage_metric(
        stage_timings_ms,
        "recall_candidate_max_score",
        max(
            (search_scores.get(eid, backfilled.get(eid, 0.0)) for eid in merged_ids),
            default=0.0,
        ),
    )

    # Step 6: Return (entity_id, real_semantic_similarity) in RRF order
    return [(eid, search_scores.get(eid, backfilled.get(eid, 0.0))) for eid in merged_ids]
