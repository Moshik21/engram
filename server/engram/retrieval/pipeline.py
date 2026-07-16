"""Full retrieval pipeline: FTS5 → activation → spreading → scoring."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING

from engram.activation.spreading import (
    identify_actr_seeds,
    identify_seeds,
    spread_activation,
)
from engram.config import ActivationConfig
from engram.retrieval.plan import build_recall_plan, execute_recall_plan
from engram.retrieval.recall_graph_gate import (
    GatedGraphStore,
)
from engram.retrieval.recall_graph_gate import (
    skip_secondary_graph_after_probe_timeout as _skip_secondary_graph_after_probe_timeout,
)
from engram.retrieval.router import QueryType, apply_route, classify_query
from engram.retrieval.scorer import (
    ScoredResult,
    score_candidates,
    score_candidates_thompson,
)

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer

logger = logging.getLogger(__name__)


def _extract_chunk_score(chunk: dict) -> float:
    """Extract a score from a chunk result dict.

    Chunk results may carry a distance (lower = closer) or a score (higher = better).
    Convert distance to similarity when needed.
    """
    if "distance" in chunk:
        dist = float(chunk["distance"])
        return max(0.0, 1.0 - dist / 2.0)
    if "score" in chunk:
        return float(chunk["score"])
    return 0.5


def _detect_temporal_cues(query: str) -> dict:
    """Detect temporal signals in query text."""
    q = query.lower()
    cues = {
        "is_temporal": False,
        "wants_earliest": False,  # "first", "earliest", "initially"
        "wants_latest": False,  # "last", "most recent", "current", "now", "still"
        "wants_count": False,  # "how many days", "how long"
        "is_state_query": False,  # "how many", "what is my current"
    }

    earliest_words = {"first", "earliest", "initially", "original", "started"}
    latest_words = {
        "last",
        "latest",
        "most recent",
        "current",
        "now",
        "still",
        "currently",
        "today",
    }
    count_words = {"how many days", "how long", "how many weeks", "how many months"}
    state_words = {"how many", "what is my", "do i still", "am i still", "what's my current"}

    for w in earliest_words:
        if w in q:
            cues["is_temporal"] = True
            cues["wants_earliest"] = True
    for w in latest_words:
        if w in q:
            cues["is_temporal"] = True
            cues["wants_latest"] = True
    for w in count_words:
        if w in q:
            cues["is_temporal"] = True
            cues["wants_count"] = True
    for w in state_words:
        if w in q:
            cues["is_state_query"] = True

    return cues


async def _inject_entity_matches(
    query: str,
    group_id: str,
    graph_store,
    candidates: list[tuple[str, float]],
    max_inject: int = 10,
) -> list[tuple[str, float]]:
    """Fallback: find entities by name match and inject with 1-hop neighbors.

    When semantic search returns few candidates, this does a simple name-based
    lookup to ensure known entities become spreading seeds.
    """
    if not hasattr(graph_store, "find_entities"):
        return candidates

    existing_ids = {eid for eid, _ in candidates}
    injected = list(candidates)

    # Tokenize query, try each token as a name search
    tokens = [t.strip("?!.,;:'\"") for t in query.split() if len(t) > 2]

    for token in tokens:
        try:
            entities = await graph_store.find_entities(
                name=token,
                group_id=group_id,
                limit=3,
            )
        except Exception:
            continue

        for entity in entities:
            if entity.id not in existing_ids:
                # Inject with baseline similarity (will be re-scored)
                injected.append((entity.id, 0.15))
                existing_ids.add(entity.id)

            # Also inject 1-hop neighbors
            if not hasattr(graph_store, "get_active_neighbors_with_weights"):
                continue
            try:
                neighbors = await graph_store.get_active_neighbors_with_weights(
                    entity_id=entity.id,
                    group_id=group_id,
                )
                for neighbor_info in neighbors[:max_inject]:
                    nid = neighbor_info[0]
                    if nid not in existing_ids:
                        injected.append((nid, 0.10))
                        existing_ids.add(nid)
            except Exception:
                continue

    return injected


def _scale_limit(base: int, total_entities: int, lo: int, hi: int) -> int:
    """Scale a limit by sqrt(total_entities / 1000), clamped to [lo, hi]."""
    scale = math.sqrt(max(total_entities, 1000) / 1000.0)
    return max(lo, min(int(base * scale), hi))


def _stage_metric(stage_timings_ms: dict[str, float] | None, key: str) -> float:
    if not stage_timings_ms:
        return 0.0
    value = stage_timings_ms.get(key, 0.0)
    return float(value) if isinstance(value, int | float) else 0.0


def _candidate_pool_has_no_semantic_anchor(
    stage_timings_ms: dict[str, float] | None,
) -> bool:
    """Return true when non-semantic pools produced only zero-score candidates."""
    return (
        _stage_metric(stage_timings_ms, "recall_candidate_count") > 0
        and _stage_metric(stage_timings_ms, "recall_candidate_max_score") <= 0.0
        and _stage_metric(stage_timings_ms, "recall_search_candidate_count") <= 0
        and _stage_metric(stage_timings_ms, "recall_entity_query_candidate_count") <= 0
        and _stage_metric(stage_timings_ms, "recall_graph_candidate_count") <= 0
    )


def _candidate_pool_has_zero_semantic_score(
    stage_timings_ms: dict[str, float] | None,
) -> bool:
    """Return true when candidates exist but none carried a semantic score."""
    return (
        _stage_metric(stage_timings_ms, "recall_candidate_count") > 0
        and _stage_metric(stage_timings_ms, "recall_candidate_max_score") <= 0.0
        and _stage_metric(stage_timings_ms, "recall_search_candidate_count") <= 0
    )


def _merge_special_results(
    episode_candidates: list[ScoredResult],
    cue_candidates: list[ScoredResult],
    cfg: ActivationConfig,
    suppressed_cue_out: dict[str, float] | None = None,
) -> list[ScoredResult]:
    """Merge episode and cue-backed candidates, keeping the strongest per episode.

    When an episode-typed candidate outscores its colliding cue candidate, the
    survivor is surfaced as a plain episode (cue content stays unsurfaced). The
    cue hit must still be recorded for promotion, so the dropped cue's score is
    stashed in ``suppressed_cue_out`` (node_id -> cue score) for the caller to
    feed to ``record_cue_feedback`` without mutating what is surfaced (B13).
    """
    special_results: list[ScoredResult] = []

    if episode_candidates and cfg.episode_retrieval_enabled:
        # (-score, node_id): break tied scores by id so the core top-k is
        # deterministic across builds (mirrors scorer.py / candidate_pool).
        episode_candidates.sort(key=lambda r: (-r.score, r.node_id))
        special_results.extend(episode_candidates[: cfg.episode_retrieval_max])

    if cue_candidates and cfg.cue_recall_enabled:
        cue_candidates.sort(key=lambda r: (-r.score, r.node_id))
        best_by_episode = {result.node_id: result for result in special_results}
        for cue_result in cue_candidates[: cfg.cue_recall_max]:
            existing = best_by_episode.get(cue_result.node_id)
            if existing is None or cue_result.score > existing.score:
                best_by_episode[cue_result.node_id] = cue_result
            elif suppressed_cue_out is not None:
                # Episode candidate outscored its cue: the survivor is surfaced
                # as a plain episode, so the cue hit would otherwise be dropped.
                # Record the suppressed cue's score for decoupled feedback.
                suppressed_cue_out[cue_result.node_id] = cue_result.score
        special_results = list(best_by_episode.values())

    special_results.sort(key=lambda r: (-r.score, r.node_id))
    return special_results


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


def _stage_timeout_seconds(cfg: ActivationConfig, field_name: str) -> float | None:
    timeout_ms = int(getattr(cfg, field_name, 0) or 0)
    if timeout_ms <= 0:
        return None
    return timeout_ms / 1000.0


async def _get_graph_stats_for_recall(graph_store, group_id: str) -> dict:
    """Fetch graph stats while supporting stores that do not accept exact=False."""
    try:
        return await graph_store.get_stats(group_id, exact=False)
    except TypeError:
        return await graph_store.get_stats(group_id)


def _primary_search_timeout_seconds(
    cfg: ActivationConfig,
    stage_timings_ms: dict[str, float] | None = None,
) -> float | None:
    """Mirror candidate_pool: never poison primary search after probe timeouts."""
    from engram.retrieval.candidate_pool import (
        _primary_search_timeout_seconds as _shared_primary_search_timeout,
    )

    return _shared_primary_search_timeout(cfg, stage_timings_ms)


def _optional_recall_capability(
    search_index,
    method_name: str,
    feature_name: str,
    *,
    sibling_methods: tuple[str, ...] = (),
):
    """Return an optional recall method, or fail for partial implementations."""
    method = getattr(search_index, method_name, None)
    if callable(method):
        return method
    if any(callable(getattr(search_index, name, None)) for name in sibling_methods):
        raise RuntimeError(
            f"{type(search_index).__name__} is missing required recall capability "
            f"'{method_name}' for {feature_name}",
        )
    logger.debug(
        "%s missing optional recall capability '%s' for %s",
        type(search_index).__name__,
        method_name,
        feature_name,
    )
    return None


async def retrieve(
    query: str,
    group_id: str,
    graph_store,
    activation_store,
    search_index,
    cfg: ActivationConfig,
    limit: int = 10,
    enable_routing: bool = True,
    reranker=None,
    working_memory: WorkingMemoryBuffer | None = None,
    community_store=None,
    predicate_cache=None,
    conv_context=None,
    priming_buffer: dict[str, tuple[float, float]] | None = None,
    goal_cache=None,
    record_feedback: bool = True,
    memory_need=None,
    stage_timings_ms: dict[str, float] | None = None,
    suppressed_cue_out: dict[str, float] | None = None,
    budget_profile: str | None = None,
) -> list[ScoredResult]:
    """Full retrieval pipeline:

    1. FTS5 → top-K candidates (semantic similarity scores)
    1.5. Classify query and route weights (if enabled)
    2. Batch get activation states
    3. Identify seeds (sem >= seed_threshold)
    4. Spread activation BFS
    5. Score all candidates with composite formula
    5.5. Cross-encoder re-ranking (if enabled)
    5.6. MMR diversity (if enabled)
    6. Return top-N sorted by score
    """
    now = time.time()

    # Save original weight for episode scoring (before routing modifies cfg)
    original_weight_semantic = cfg.weight_semantic
    planner_trace = None
    primary_search_timed_out = False

    # Fetch entity count for dynamic pool sizing
    total_entities = 0
    stats_started = time.perf_counter()
    try:
        timeout_seconds = _stage_timeout_seconds(cfg, "retrieval_stats_timeout_ms")
        stats = (
            await asyncio.wait_for(
                _get_graph_stats_for_recall(graph_store, group_id),
                timeout=timeout_seconds,
            )
            if timeout_seconds is not None
            else await _get_graph_stats_for_recall(graph_store, group_id)
        )
        _add_stage_timing(stage_timings_ms, "recall_stats", stats_started)
        if isinstance(stats, dict):
            total_entities = int(stats.get("entity_count", stats.get("entities", 0)) or 0)
    except asyncio.TimeoutError:
        _add_stage_timing(stage_timings_ms, "recall_stats_timeout", stats_started)
    except asyncio.CancelledError:
        _add_stage_timing(stage_timings_ms, "recall_stats_cancelled", stats_started)
        raise
    except Exception:
        pass

    # Step 0.1a: Graph-anchored query expansion (LLM-free).
    # Expands the query using real entities, relationships, and summaries from
    # the knowledge graph.  Zero cost, ~3ms.  Used as vector search query when
    # HyDE is disabled (production default).
    graph_expanded_query = query  # default: unchanged
    if cfg.graph_query_expansion_enabled:
        stage_started = time.perf_counter()
        try:
            from engram.retrieval.graph_expansion import expand_query_from_graph

            expansion_call = expand_query_from_graph(query, graph_store, group_id)
            timeout_seconds = _stage_timeout_seconds(
                cfg,
                "graph_query_expansion_timeout_ms",
            )
            graph_expanded_query = (
                await asyncio.wait_for(expansion_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await expansion_call
            )
            _add_stage_timing(stage_timings_ms, "graph_expand", stage_started)
        except asyncio.TimeoutError:
            graph_expanded_query = query
            _add_stage_timing(stage_timings_ms, "graph_expand_timeout", stage_started)
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "graph_expand_cancelled",
                stage_started,
            )
            raise
        except Exception:
            pass  # Fall back to original query

    graph_store = GatedGraphStore(graph_store, cfg, stage_timings_ms)

    # Step 0.1b: Template reformulation — convert question to statement form
    # for better embedding match.  Zero cost, <1ms.  If it produces a result,
    # it is used as an additional search query merged via RRF later.
    reformulated_query: str | None = None
    if cfg.template_reformulation_enabled:
        from engram.retrieval.graph_expansion import reformulate_query

        reformulated_query = reformulate_query(query)

    # Step 0.1c: HyDE — generate hypothetical answer for better vector matching.
    # Uses LLM to produce a passage that looks like the stored content,
    # bridging the question-answer embedding asymmetry.  The expanded text
    # is used for the search call (both BM25 and vector benefit from
    # answer-like phrasing); the original *query* is preserved for entity
    # extraction, episode/cue search, spreading activation seeds, etc.
    # When HyDE is enabled (benchmark mode), it takes priority over graph
    # expansion.  When HyDE is disabled (production), graph expansion
    # provides the query improvement.
    hyde_query = query  # default: unchanged
    if cfg.hyde_enabled:
        try:
            from engram.retrieval.hyde import generate_hypothetical_document

            hypothesis = await generate_hypothetical_document(
                query,
                model=cfg.hyde_model,
            )
            if hypothesis:
                hyde_query = hypothesis
                logger.debug("HyDE expanded query: %s", hypothesis[:100])
        except Exception:
            pass  # Fall back to original query
    else:
        # Production path: graph expansion replaces HyDE
        hyde_query = graph_expanded_query

    # Step 0.2: Query decomposition for complex temporal/multi-hop queries
    # (deterministic — no LLM call, pure regex/template matching)
    sub_queries: list[str] = [query]
    if cfg.query_decomposition_enabled:
        from engram.retrieval.decomposer import decompose_query, needs_decomposition

        if needs_decomposition(query):
            try:
                sub_queries = await decompose_query(query)
            except Exception:
                pass  # Fall back to original query

    # Step 1: Generate candidates (multi-pool or single-pool)
    # Name-resolved entities seed graph traversal even when their embedding
    # similarity is below seed_threshold (populated by generate_candidates,
    # consumed by identify_seeds). Defined here so it is in scope regardless of
    # which candidate-generation branch runs.
    name_match_seeds: dict[str, float] = {}
    if cfg.multi_pool_enabled:
        from engram.retrieval.candidate_pool import generate_candidates

        # Pre-classify query for pool composition multipliers
        pre_query_type = await classify_query(query)

        candidates = await generate_candidates(
            query=hyde_query,
            group_id=group_id,
            search_index=search_index,
            activation_store=activation_store,
            graph_store=graph_store,
            cfg=cfg,
            now=now,
            working_memory=working_memory,
            total_entities=total_entities,
            query_type=pre_query_type,
            stage_timings_ms=stage_timings_ms,
            name_match_out=name_match_seeds,
            budget_profile=budget_profile,
        )
        primary_search_timed_out = bool(
            stage_timings_ms and "recall_primary_search_timeout" in stage_timings_ms
        )
        query_type = await classify_query(query, search_results=candidates)
        if enable_routing or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)
        temporal_mode = query_type == QueryType.TEMPORAL
    else:
        # Original single-pool path — scale retrieval_top_k
        top_k = _scale_limit(cfg.retrieval_top_k, total_entities, 5, 500)
        search_started = time.perf_counter()
        try:
            search_call = search_index.search(
                query=hyde_query,
                group_id=group_id,
                limit=top_k,
            )
            timeout_seconds = _primary_search_timeout_seconds(cfg, stage_timings_ms)
            search_results = (
                await asyncio.wait_for(search_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await search_call
            )
            _add_stage_timing(stage_timings_ms, "recall_primary_search", search_started)
            _add_stage_timing(stage_timings_ms, "recall_embed", search_started)
        except asyncio.TimeoutError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_primary_search_timeout",
                search_started,
            )
            search_results = []
            primary_search_timed_out = True
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_primary_search_cancelled",
                search_started,
            )
            raise
        candidates = search_results or []

        # Step 1.5: Classify query and override weights
        query_type = await classify_query(query, search_results=candidates)
        if enable_routing or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)

        # Step 1.6: Temporal bypass — merge activation-based candidates
        temporal_mode = False
        if query_type == QueryType.TEMPORAL:
            temporal_mode = True
            act_limit = _scale_limit(cfg.retrieval_top_k, total_entities, 5, 500)
            top_activated = await activation_store.get_top_activated(
                group_id=group_id,
                limit=act_limit,
                now=now,
            )
            existing_ids = {eid for eid, _ in candidates}
            activation_candidates = [
                (eid, 0.0) for eid, _state in top_activated if eid not in existing_ids
            ]
            candidates = candidates + activation_candidates

        # Step 1.7: Inject working memory candidates
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            existing_ids = {eid for eid, _ in candidates}
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in existing_ids:
                    candidates.append((item_id, 0.1 * recency_score))
                    existing_ids.add(item_id)

    # Step 0.25: Merge decomposed sub-query results into candidate pool
    if len(sub_queries) > 1:
        all_sub_candidates: list[tuple[str, float]] = []
        for sq in sub_queries:
            try:
                sq_results = await search_index.search(
                    query=sq,
                    group_id=group_id,
                    limit=cfg.retrieval_top_k,
                )
                if sq_results:
                    all_sub_candidates.extend(sq_results)
            except Exception:
                continue
        # Deduplicate, keep highest score per entity
        seen_ids: dict[str, tuple[str, float]] = {}
        for eid, score in all_sub_candidates:
            if eid not in seen_ids or score > seen_ids[eid][1]:
                seen_ids[eid] = (eid, score)
        # Merge into main candidate pool (keep higher score if duplicate)
        existing_scores = {eid: s for eid, s in candidates}
        for eid, score in seen_ids.values():
            if eid not in existing_scores or score > existing_scores[eid]:
                existing_scores[eid] = score
        candidates = list(existing_scores.items())

    # Step 0.3: Template reformulation — run reformulated query as a second
    # search and merge results via max-score (RRF-style) into candidates.
    if reformulated_query and reformulated_query != query:
        try:
            reform_results = await search_index.search(
                query=reformulated_query,
                group_id=group_id,
                limit=cfg.retrieval_top_k,
            )
            if reform_results:
                existing_scores = {eid: s for eid, s in candidates}
                for eid, score in reform_results:
                    if eid not in existing_scores or score > existing_scores[eid]:
                        existing_scores[eid] = score
                candidates = list(existing_scores.items())
                logger.debug(
                    "Reformulation merged %d results for: %s",
                    len(reform_results),
                    reformulated_query[:80],
                )
        except Exception:
            pass  # Non-fatal

    # Step 0.5: Planner-driven multi-intent recall (Phase 2)
    skip_planner_after_primary_timeout = bool(
        primary_search_timed_out
        and (not candidates or _candidate_pool_has_zero_semantic_score(stage_timings_ms))
    )
    if cfg.recall_planner_enabled and not skip_planner_after_primary_timeout:
        planner_started = time.perf_counter()
        try:
            planner_call = execute_recall_plan(
                build_recall_plan(
                    query,
                    cfg,
                    conv_context=conv_context,
                    memory_need=memory_need,
                ),
                group_id=group_id,
                search_index=search_index,
                base_candidates=candidates,
            )
            timeout_seconds = _stage_timeout_seconds(cfg, "recall_planner_timeout_ms")
            planner_trace = (
                await asyncio.wait_for(planner_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await planner_call
            )
            _add_stage_timing(stage_timings_ms, "recall_planner", planner_started)
            if planner_trace.merged_candidates:
                candidates = planner_trace.merged_candidates
        except asyncio.TimeoutError:
            _add_stage_timing(stage_timings_ms, "recall_planner_timeout", planner_started)
            planner_trace = None
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_planner_cancelled",
                planner_started,
            )
            raise
    elif cfg.recall_planner_enabled and skip_planner_after_primary_timeout:
        if stage_timings_ms is not None:
            stage_timings_ms["recall_planner_skipped_primary_timeout"] = 0.0

    # Step 0.6: Legacy multi-query decomposition (Wave 2)
    elif cfg.conv_multi_query_enabled and conv_context is not None:
        import asyncio as _asyncio

        turn_count = conv_context._turn_count
        # Weight schedule: early sessions query-dominant, later context-dominant
        if turn_count < 3:
            w_topic, w_entity = 0.25, 0.15
        else:
            w_topic, w_entity = 0.35, 0.30

        recent_turns = conv_context.get_recent_turns(cfg.conv_multi_query_turns)
        topic_query = " ".join(recent_turns).strip()
        top_ents = conv_context.get_top_entities(cfg.conv_multi_query_top_entities)
        entity_query = " ".join(e.name for e in top_ents).strip()

        sub_limit = min(cfg.retrieval_top_k // 2, 25)
        sub_queries: list[tuple[str, float]] = []
        if topic_query and topic_query != query:
            sub_queries.append((topic_query, w_topic))
        if entity_query and entity_query != query:
            sub_queries.append((entity_query, w_entity))

        if sub_queries:
            tasks = [
                search_index.search(query=sq, group_id=group_id, limit=sub_limit)
                for sq, _ in sub_queries
            ]
            sub_results = await _asyncio.gather(*tasks, return_exceptions=True)
            existing_scores = {eid: s for eid, s in candidates}
            for (sq, sw), result in zip(sub_queries, sub_results):
                if isinstance(result, list) and result:
                    for eid, score in result:
                        weighted = score * sw
                        if eid not in existing_scores or weighted > existing_scores[eid]:
                            existing_scores[eid] = weighted
            candidates = list(existing_scores.items())

    # Step 1.1: Episode search (runs in both modes)
    # passage_first strategy: 2x episode search budget, full scores (no discount)
    _passage_first = cfg.retrieval_strategy == "passage_first"
    _ep_budget_mult = 2 if _passage_first else 1
    _ep_score_weight = 1.0 if _passage_first else cfg.episode_retrieval_weight
    episode_candidates: list[ScoredResult] = []
    if cfg.episode_retrieval_enabled:
        episode_method_name = (
            "search_episodes_fast" if primary_search_timed_out else "search_episodes"
        )
        search_episodes = _optional_recall_capability(
            search_index,
            episode_method_name,
            "episode retrieval",
            sibling_methods=() if primary_search_timed_out else ("search_episode_cues",),
        )
        if search_episodes is not None:
            episode_search_started = time.perf_counter()
            try:
                ep_call = search_episodes(
                    query=query,
                    group_id=group_id,
                    limit=cfg.episode_retrieval_max * 3 * _ep_budget_mult,
                )
                timeout_field = (
                    "retrieval_fast_episode_search_timeout_ms"
                    if primary_search_timed_out
                    else "retrieval_episode_search_timeout_ms"
                )
                timeout_seconds = _stage_timeout_seconds(
                    cfg,
                    timeout_field,
                )
                ep_results = (
                    await asyncio.wait_for(ep_call, timeout=timeout_seconds)
                    if timeout_seconds is not None
                    else await ep_call
                )
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_episode_search",
                    episode_search_started,
                )
                _add_stage_timing(stage_timings_ms, "recall_embed", episode_search_started)
                for ep_id, sem_sim in ep_results:
                    ep_score = original_weight_semantic * sem_sim * _ep_score_weight
                    episode_candidates.append(
                        ScoredResult(
                            node_id=ep_id,
                            score=ep_score,
                            semantic_similarity=sem_sim,
                            activation=0.0,
                            spreading=0.0,
                            edge_proximity=0.0,
                            exploration_bonus=0.0,
                            result_type="episode",
                        )
                    )
            except asyncio.TimeoutError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_episode_search_timeout",
                    episode_search_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_episode_search_cancelled",
                    episode_search_started,
                )
                raise
            except Exception as e:
                logger.warning("Episode search failed (non-fatal): %s", e)

    # Step 1.2: Cue-backed episode search
    cue_candidates: list[ScoredResult] = []
    if cfg.cue_recall_enabled:
        cue_method_name = (
            "search_episode_cues_fast" if primary_search_timed_out else "search_episode_cues"
        )
        search_episode_cues = _optional_recall_capability(
            search_index,
            cue_method_name,
            "cue recall",
            sibling_methods=() if primary_search_timed_out else ("search_episodes",),
        )
        if search_episode_cues is not None:
            cue_search_started = time.perf_counter()
            try:
                cue_call = search_episode_cues(
                    query=query,
                    group_id=group_id,
                    limit=cfg.cue_recall_max * 3,
                )
                timeout_field = (
                    "retrieval_fast_cue_search_timeout_ms"
                    if primary_search_timed_out
                    else "retrieval_cue_search_timeout_ms"
                )
                timeout_seconds = _stage_timeout_seconds(
                    cfg,
                    timeout_field,
                )
                cue_results = (
                    await asyncio.wait_for(cue_call, timeout=timeout_seconds)
                    if timeout_seconds is not None
                    else await cue_call
                )
                _add_stage_timing(stage_timings_ms, "recall_cue_search", cue_search_started)
                _add_stage_timing(stage_timings_ms, "recall_embed", cue_search_started)
                for ep_id, sem_sim in cue_results:
                    cue_score = original_weight_semantic * sem_sim * cfg.cue_recall_weight
                    cue_candidates.append(
                        ScoredResult(
                            node_id=ep_id,
                            score=cue_score,
                            semantic_similarity=sem_sim,
                            activation=0.0,
                            spreading=0.0,
                            edge_proximity=0.0,
                            exploration_bonus=0.0,
                            result_type="cue_episode",
                        )
                    )
            except asyncio.TimeoutError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_cue_search_timeout",
                    cue_search_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_cue_search_cancelled",
                    cue_search_started,
                )
                raise
            except Exception as e:
                logger.warning("Cue search failed (non-fatal): %s", e)

    # Step 1.3: Chunk search — sub-episode precision
    chunk_hits: dict[str, dict] = {}
    if (
        cfg.chunk_search_enabled
        and not primary_search_timed_out
        and hasattr(search_index, "search_episode_chunks")
    ):
        chunk_started = time.perf_counter()
        try:
            chunk_call = search_index.search_episode_chunks(
                query=query,
                group_id=group_id,
                limit=cfg.episode_retrieval_max * 3 * _ep_budget_mult,
            )
            timeout_seconds = _stage_timeout_seconds(
                cfg,
                "retrieval_chunk_search_timeout_ms",
            )
            if timeout_seconds is not None:
                chunk_results = await asyncio.wait_for(
                    chunk_call,
                    timeout=timeout_seconds,
                )
            else:
                chunk_results = await chunk_call
            _add_stage_timing(stage_timings_ms, "recall_chunk_search", chunk_started)
            # Track chunk hits for downstream context enrichment
            seen_episode_ids_in_chunks: set[str] = set()
            for chunk in chunk_results:
                episode_id = chunk.get("episode_id", "")
                if not episode_id:
                    continue
                chunk_score = chunk.get("score", 0.0)
                if not chunk_score:
                    chunk_score = _extract_chunk_score(chunk)
                # Keep the best chunk per episode
                if episode_id not in chunk_hits or chunk_score > chunk_hits[episode_id].get(
                    "score", 0.0
                ):
                    chunk_hits[episode_id] = chunk
                    if "score" not in chunk or not chunk.get("score"):
                        chunk_hits[episode_id]["score"] = chunk_score
                # Add as episode candidate if not already found by episode search
                if episode_id not in seen_episode_ids_in_chunks:
                    seen_episode_ids_in_chunks.add(episode_id)
                    # Check if this episode is already in episode_candidates
                    existing_ep_ids = {ec.node_id for ec in episode_candidates}
                    chunk_text = chunk.get("chunk_text", "")
                    if episode_id not in existing_ep_ids:
                        ep_score = original_weight_semantic * chunk_score * _ep_score_weight
                        episode_candidates.append(
                            ScoredResult(
                                node_id=episode_id,
                                score=ep_score,
                                semantic_similarity=chunk_score,
                                activation=0.0,
                                spreading=0.0,
                                edge_proximity=0.0,
                                exploration_bonus=0.0,
                                result_type="episode",
                                source="chunk_search",
                                chunk_context=chunk_text or None,
                            )
                        )
                    else:
                        # Episode already exists — keep higher score
                        for ec in episode_candidates:
                            if ec.node_id == episode_id:
                                new_score = (
                                    original_weight_semantic * chunk_score * _ep_score_weight
                                )
                                if new_score > ec.score:
                                    ec.score = new_score
                                    ec.semantic_similarity = chunk_score
                                    ec.source = "chunk_search"
                                    ec.chunk_context = chunk_text or None
                                break
            # Attach chunk context to any episode candidates that matched chunks
            # (including those originally found by episode vector search)
            for ec in episode_candidates:
                if ec.chunk_context is None and ec.node_id in chunk_hits:
                    ct = chunk_hits[ec.node_id].get("chunk_text", "")
                    if ct:
                        ec.chunk_context = ct

            # Step 1.35: Session-level scoring — boost episodes with multiple
            # matching chunks (more chunks = more topically relevant session)
            episode_chunk_counts: dict[str, int] = {}
            for chunk in chunk_results:
                ep_id = chunk.get("episode_id", "")
                if ep_id:
                    episode_chunk_counts[ep_id] = episode_chunk_counts.get(ep_id, 0) + 1
            for ec in episode_candidates:
                chunk_count = episode_chunk_counts.get(ec.node_id, 0)
                if chunk_count > 1:
                    ec.score *= 1 + 0.2 * (chunk_count - 1)
        except asyncio.TimeoutError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_chunk_search_timeout",
                chunk_started,
            )
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_chunk_search_cancelled",
                chunk_started,
            )
            raise
        except Exception as e:
            logger.warning("Chunk search failed (non-fatal): %s", e)

    _set_stage_metric(
        stage_timings_ms,
        "recall_episode_candidate_count",
        len(episode_candidates),
    )
    _set_stage_metric(
        stage_timings_ms,
        "recall_episode_candidate_max_score",
        max((result.score for result in episode_candidates), default=0.0),
    )
    _set_stage_metric(
        stage_timings_ms,
        "recall_cue_candidate_count",
        len(cue_candidates),
    )
    _set_stage_metric(
        stage_timings_ms,
        "recall_cue_candidate_max_score",
        max((result.score for result in cue_candidates), default=0.0),
    )

    # Step 1.8: Entity-first fallback when search finds few candidates
    if not candidates and not primary_search_timed_out:
        entity_match_started = time.perf_counter()
        try:
            entity_match_call = _inject_entity_matches(
                query,
                group_id,
                graph_store,
                candidates,
            )
            timeout_seconds = _stage_timeout_seconds(
                cfg,
                "retrieval_entity_match_timeout_ms",
            )
            candidates = (
                await asyncio.wait_for(entity_match_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await entity_match_call
            )
            _add_stage_timing(stage_timings_ms, "recall_entity_match", entity_match_started)
        except asyncio.TimeoutError:
            candidates = []
            _add_stage_timing(
                stage_timings_ms,
                "recall_entity_match_timeout",
                entity_match_started,
            )
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_entity_match_cancelled",
                entity_match_started,
            )
            raise
    elif not candidates and primary_search_timed_out:
        if stage_timings_ms is not None:
            stage_timings_ms["recall_entity_match_skipped_primary_timeout"] = 0.0

    if not candidates:
        special_results = _merge_special_results(
            episode_candidates, cue_candidates, cfg, suppressed_cue_out
        )
        if special_results:
            return special_results[: min(limit, cfg.retrieval_top_n)]
        return []

    if primary_search_timed_out and _candidate_pool_has_zero_semantic_score(stage_timings_ms):
        special_results = _merge_special_results(
            episode_candidates, cue_candidates, cfg, suppressed_cue_out
        )
        if special_results:
            _set_stage_metric(
                stage_timings_ms,
                "recall_zero_semantic_special_deferred",
                len(special_results),
            )
            return []
        _set_stage_metric(
            stage_timings_ms,
            "recall_zero_semantic_short_circuit",
            0.0,
        )
        return []

    if _candidate_pool_has_no_semantic_anchor(stage_timings_ms):
        special_results = _merge_special_results(
            episode_candidates, cue_candidates, cfg, suppressed_cue_out
        )
        if special_results:
            _set_stage_metric(
                stage_timings_ms,
                "recall_zero_semantic_special_return",
                len(special_results),
            )
            return special_results[: min(limit, cfg.retrieval_top_n)]
        _set_stage_metric(
            stage_timings_ms,
            "recall_zero_semantic_short_circuit",
            0.0,
        )
        return []

    # Step 2: Batch get activation states
    entity_ids = [eid for eid, _ in candidates]
    activation_state_started = time.perf_counter()
    try:
        activation_state_call = activation_store.batch_get(entity_ids)
        timeout_seconds = _stage_timeout_seconds(
            cfg,
            "retrieval_activation_state_timeout_ms",
        )
        activation_states = (
            await asyncio.wait_for(activation_state_call, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await activation_state_call
        )
        activation_states = activation_states or {}
        _add_stage_timing(
            stage_timings_ms,
            "recall_activation_state",
            activation_state_started,
        )
    except asyncio.TimeoutError:
        activation_states = {}
        _add_stage_timing(
            stage_timings_ms,
            "recall_activation_state_timeout",
            activation_state_started,
        )
    except asyncio.CancelledError:
        _add_stage_timing(
            stage_timings_ms,
            "recall_activation_state_cancelled",
            activation_state_started,
        )
        raise
    except Exception:
        activation_states = {}

    # Step 2.5: Goal priming (Brain Architecture)
    goal_seeds: list[tuple[str, float]] = []
    if cfg.goal_priming_enabled:
        if _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
            _set_stage_metric(
                stage_timings_ms,
                "recall_goal_priming_skipped_probe_timeout",
                0.0,
            )
        else:
            goal_started = time.perf_counter()
            try:
                from engram.retrieval.goals import (
                    compute_goal_priming_seeds,
                    identify_active_goals,
                )

                goal_call = identify_active_goals(
                    graph_store,
                    activation_store,
                    group_id,
                    cfg,
                    cache=goal_cache,
                )
                timeout_seconds = _stage_timeout_seconds(
                    cfg,
                    "retrieval_goal_priming_timeout_ms",
                )
                active_goals = (
                    await asyncio.wait_for(goal_call, timeout=timeout_seconds)
                    if timeout_seconds is not None
                    else await goal_call
                )
                goal_seeds = compute_goal_priming_seeds(active_goals, cfg)
                _add_stage_timing(stage_timings_ms, "recall_goal_priming", goal_started)
            except asyncio.TimeoutError:
                goal_seeds = []
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_goal_priming_timeout",
                    goal_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_goal_priming_cancelled",
                    goal_started,
                )
                raise

    # Step 3: Identify seeds (strategy-dependent)
    if cfg.spreading_strategy == "actr":
        # ACT-R: seeds come from working memory only (not search results)
        if working_memory is not None and cfg.working_memory_enabled:
            seeds = identify_actr_seeds(working_memory, now, cfg)
        else:
            seeds = []
        seed_node_ids = {nid for nid, _ in seeds}
        context_gate = None  # context gate is a BFS/PPR concept
    else:
        seeds = identify_seeds(
            candidates,
            activation_states,
            now,
            cfg,
            temporal_mode=temporal_mode,
            name_match_scores=name_match_seeds,
        )
        seed_node_ids = {nid for nid, _ in seeds}

        # Step 3.5: Add working memory entities as additional seeds
        if working_memory is not None and cfg.working_memory_enabled:
            wm_candidates = working_memory.get_candidates(now)
            for item_id, recency_score, _item_type in wm_candidates:
                if item_id not in seed_node_ids:
                    energy = cfg.working_memory_seed_energy * recency_score
                    if energy > 0.0:
                        seeds.append((item_id, energy))
                        seed_node_ids.add(item_id)

        # Step 3.6: Session entity seed injection (Wave 2)
        if cfg.conv_session_entity_seeds_enabled and conv_context is not None:
            for entry in conv_context.get_top_entities(cfg.conv_multi_query_top_entities):
                if entry.entity_id not in seed_node_ids:
                    energy = cfg.conv_session_entity_seed_energy * min(
                        1.0,
                        entry.mention_weight / 5.0,
                    )
                    if energy > 0.0:
                        seeds.append((entry.entity_id, energy))
                        seed_node_ids.add(entry.entity_id)

        # Step 3.7: Build context gate (if enabled)
        context_gate = None
        if cfg.context_gating_enabled and predicate_cache is not None:
            query_emb = getattr(search_index, "_last_query_vec", None)
            if query_emb:
                from engram.activation.context_gate import build_context_gate

                context_gate = build_context_gate(query_emb, predicate_cache, cfg)

    # Step 3.9: Merge goal priming seeds
    if goal_seeds:
        for gid, energy in goal_seeds:
            if gid not in seed_node_ids:
                seeds.append((gid, energy))
                seed_node_ids.add(gid)

    # Step 3.8: Build seed entity types for cross-domain penalty
    seed_entity_types: dict[str, str] | None = None
    if cfg.cross_domain_penalty_enabled:
        if _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
            _set_stage_metric(
                stage_timings_ms,
                "recall_cross_domain_seed_skipped_probe_timeout",
                0.0,
            )
        else:
            cross_domain_started = time.perf_counter()
            try:

                async def _load_seed_entity_types() -> dict[str, str]:
                    loaded_types: dict[str, str] = {}
                    for seed_id, _ in seeds:
                        try:
                            ent = await graph_store.get_entity(seed_id, group_id)
                            if ent:
                                loaded_types[seed_id] = ent.entity_type
                        except Exception:
                            pass
                    return loaded_types

                seed_type_call = _load_seed_entity_types()
                timeout_seconds = _stage_timeout_seconds(
                    cfg,
                    "retrieval_cross_domain_seed_timeout_ms",
                )
                seed_entity_types = (
                    await asyncio.wait_for(seed_type_call, timeout=timeout_seconds)
                    if timeout_seconds is not None
                    else await seed_type_call
                )
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_cross_domain_seed",
                    cross_domain_started,
                )
            except asyncio.TimeoutError:
                seed_entity_types = None
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_cross_domain_seed_timeout",
                    cross_domain_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_cross_domain_seed_cancelled",
                    cross_domain_started,
                )
                raise

    # Step 4: Spread activation
    if seeds and _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
        bonuses, hop_distances = {}, {}
        _set_stage_metric(stage_timings_ms, "recall_spread_skipped_probe_timeout", 0.0)
    else:
        spread_started = time.perf_counter()
        try:
            spread_call = spread_activation(
                seeds,
                graph_store,
                cfg,
                group_id=group_id,
                community_store=community_store,
                context_gate=context_gate,
                seed_entity_types=seed_entity_types,
            )
            timeout_seconds = _stage_timeout_seconds(cfg, "retrieval_spread_timeout_ms")
            bonuses, hop_distances = (
                await asyncio.wait_for(spread_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await spread_call
            )
            _add_stage_timing(stage_timings_ms, "recall_spread", spread_started)
        except asyncio.TimeoutError:
            bonuses, hop_distances = {}, {}
            _add_stage_timing(stage_timings_ms, "recall_spread_timeout", spread_started)
        except asyncio.CancelledError:
            _add_stage_timing(stage_timings_ms, "recall_spread_cancelled", spread_started)
            raise

    # Step 4.1: Inhibitory spreading (Brain Architecture)
    if cfg.inhibitory_spreading_enabled:
        if _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
            _set_stage_metric(
                stage_timings_ms,
                "recall_inhibitory_spreading_skipped_probe_timeout",
                0.0,
            )
        else:
            from engram.retrieval.inhibition import apply_inhibition

            # Fetch seed relationships so predicate suppression (LIKES/DISLIKES etc.)
            # has the contradictory-edge data it needs to actually fire.
            seed_relationships: list[tuple[str, str, str, float]] = []
            if cfg.inhibition_predicate_suppression and seed_node_ids:
                for seed_id in seed_node_ids:
                    try:
                        rels = await graph_store.get_relationships(
                            seed_id,
                            group_id=group_id,
                        )
                    except Exception:
                        continue
                    for rel in rels:
                        seed_relationships.append(
                            (rel.source_id, rel.target_id, rel.predicate, rel.weight)
                        )

            bonuses = await apply_inhibition(
                bonuses=bonuses,
                hop_distances=hop_distances,
                seed_node_ids=seed_node_ids,
                graph_store=graph_store,
                search_index=search_index,
                group_id=group_id,
                cfg=cfg,
                relationships=seed_relationships or None,
            )

    # Step 4.5: Merge spreading-discovered entities with real semantic similarity
    existing_ids = {eid for eid, _ in candidates}
    new_ids = [nid for nid in bonuses if nid not in existing_ids and bonuses[nid] > 0.0]
    if new_ids:
        new_states = await activation_store.batch_get(new_ids)
        activation_states.update(new_states)
        similarity_started = time.perf_counter()
        discovered_sims = await search_index.compute_similarity(
            query=query,
            entity_ids=new_ids,
            group_id=group_id,
        )
        _add_stage_timing(stage_timings_ms, "recall_embed", similarity_started)
        candidates = candidates + [(nid, discovered_sims.get(nid, 0.0)) for nid in new_ids]

    # Step 4.6: Fingerprint similarity computation (Wave 2)
    conv_fingerprint_sim: dict[str, float] | None = None
    if (
        cfg.conv_fingerprint_enabled
        and conv_context is not None
        and cfg.conv_context_rerank_weight > 0.0
    ):
        fingerprint = conv_context.get_fingerprint()
        if fingerprint is not None:
            try:
                all_ids = [eid for eid, _ in candidates]
                conv_fingerprint_sim = await search_index.compute_similarity(
                    query="",
                    entity_ids=all_ids,
                    group_id=group_id,
                    query_embedding=fingerprint,
                )
            except Exception:
                pass

    # Step 4.7: Priming buffer boosts (Wave 3)
    priming_boosts: dict[str, float] | None = None
    if priming_buffer:
        priming_boosts = {}
        for eid, (boost, expiry) in priming_buffer.items():
            if now < expiry:
                priming_boosts[eid] = boost
        if not priming_boosts:
            priming_boosts = None

    # Step 4.8: Graph structural similarity (when weight > 0)
    # Compares graph embeddings between seed entities and candidates —
    # both vectors live in the same structural embedding space.
    graph_similarities: dict[str, float] | None = None
    if cfg.weight_graph_structural > 0:

        async def _compute_graph_similarities() -> dict[str, float] | None:
            all_candidate_ids = [eid for eid, _ in candidates]
            # Determine preferred method: try enabled methods that have data.
            methods_to_try = []
            if cfg.graph_embedding_node2vec_enabled:
                methods_to_try.append("node2vec")
            if cfg.graph_embedding_transe_enabled:
                methods_to_try.append("transe")
            if cfg.graph_embedding_gnn_enabled:
                methods_to_try.append("gnn")

            if not methods_to_try or not hasattr(search_index, "get_graph_embeddings"):
                return None

            # Get seed entity IDs (from spreading activation seeds).
            query_seed_ids = list(seed_node_ids) if seed_node_ids else []

            for method in methods_to_try:
                if not query_seed_ids:
                    # Fall back to top candidates as proxy seeds.
                    query_seed_ids = [
                        eid
                        for eid, _ in sorted(
                            candidates,
                            key=lambda x: x[1],
                            reverse=True,
                        )[:3]
                    ]
                    if not query_seed_ids:
                        break

                # Fetch graph embeddings for seeds + candidates.
                all_ids = list(set(query_seed_ids + all_candidate_ids))
                graph_embs = await search_index.get_graph_embeddings(
                    all_ids,
                    method=method,
                    group_id=group_id,
                )
                if not graph_embs:
                    continue

                # Find seed entities that have graph embeddings.
                seed_embs = {sid: graph_embs[sid] for sid in query_seed_ids if sid in graph_embs}
                if not seed_embs:
                    continue

                import numpy as np

                from engram.storage.sqlite.vectors import (
                    cosine_similarity as _cos_sim,
                )

                seed_vecs = list(seed_embs.values())
                query_graph_vec = np.mean(seed_vecs, axis=0).tolist()

                computed_similarities = {}
                for eid, g_emb in graph_embs.items():
                    if eid in seed_embs:
                        continue  # Don't score seeds against themselves.
                    computed_similarities[eid] = max(
                        0.0,
                        _cos_sim(query_graph_vec, g_emb),
                    )
                return computed_similarities or None
            return None

        graph_similarity_started = time.perf_counter()
        try:
            graph_similarity_call = _compute_graph_similarities()
            timeout_seconds = _stage_timeout_seconds(
                cfg,
                "retrieval_graph_similarity_timeout_ms",
            )
            graph_similarities = (
                await asyncio.wait_for(graph_similarity_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await graph_similarity_call
            )
            _add_stage_timing(
                stage_timings_ms,
                "recall_graph_similarity",
                graph_similarity_started,
            )
            if not graph_similarities and stage_timings_ms is not None:
                # weight_graph_structural > 0 but no graph embeddings exist for
                # any method/seed — the structural term is dead weight here.
                # Surface it instead of silently scoring 0 for every candidate.
                stage_timings_ms["recall_graph_structural_empty_source"] = 1.0
        except asyncio.TimeoutError:
            graph_similarities = None
            _add_stage_timing(
                stage_timings_ms,
                "recall_graph_similarity_timeout",
                graph_similarity_started,
            )
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_graph_similarity_cancelled",
                graph_similarity_started,
            )
            raise
        except Exception as e:
            logger.warning("Graph similarity computation failed (non-fatal): %s", e)

    # Step 4.9: Fetch entity attributes for emotional + state boosts
    entity_attributes: dict[str, dict] | None = None
    needs_attrs = cfg.emotional_salience_enabled or cfg.state_dependent_retrieval_enabled
    if needs_attrs and _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
        _set_stage_metric(
            stage_timings_ms,
            "recall_entity_attributes_skipped_probe_timeout",
            0.0,
        )
    elif needs_attrs:
        all_candidate_ids = [eid for eid, _ in candidates]

        async def _load_entity_attributes() -> dict[str, dict] | None:
            loaded_attributes: dict[str, dict] = {}
            for eid in all_candidate_ids:
                try:
                    ent = await graph_store.get_entity(eid, group_id)
                    if ent:
                        attrs = dict(ent.attributes) if isinstance(ent.attributes, dict) else {}
                        # Include entity_type for domain mapping.
                        if ent.entity_type:
                            attrs["entity_type"] = ent.entity_type
                        loaded_attributes[eid] = attrs
                except Exception:
                    pass
            return loaded_attributes or None

        attrs_started = time.perf_counter()
        try:
            attrs_call = _load_entity_attributes()
            timeout_seconds = _stage_timeout_seconds(
                cfg,
                "retrieval_entity_attributes_timeout_ms",
            )
            entity_attributes = (
                await asyncio.wait_for(attrs_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await attrs_call
            )
            _add_stage_timing(stage_timings_ms, "recall_entity_attributes", attrs_started)
        except asyncio.TimeoutError:
            entity_attributes = None
            _add_stage_timing(
                stage_timings_ms,
                "recall_entity_attributes_timeout",
                attrs_started,
            )
        except asyncio.CancelledError:
            _add_stage_timing(
                stage_timings_ms,
                "recall_entity_attributes_cancelled",
                attrs_started,
            )
            raise

    # Step 4.95: State-dependent retrieval biases (Brain Architecture)
    state_biases: dict[str, float] | None = None
    if cfg.state_dependent_retrieval_enabled and conv_context is not None:
        # Populate cognitive state once per recall turn so the bias is non-zero.
        # Without this the state source is never set and the feature is inert.
        update_state = getattr(conv_context, "update_cognitive_state", None)
        if callable(update_state):
            try:
                update_state(query)
            except Exception:
                pass
        cog_state = getattr(conv_context, "cognitive_state", None)
        if cog_state is not None and entity_attributes:
            from engram.retrieval.state import compute_state_bias

            state_biases = {}
            for eid, _ in candidates:
                attrs = entity_attributes.get(eid, {})
                etype = attrs.get("entity_type", "Other")
                bias = compute_state_bias(
                    cog_state,
                    attrs,
                    etype,
                    cfg,
                    domain_groups=cfg.domain_groups,
                )
                if bias > 0:
                    state_biases[eid] = bias
            if not state_biases:
                state_biases = None

    # Step 4.97: Preference-directed boosts
    preference_boosts: dict[str, float] | None = None
    if cfg.preference_directed_enabled:
        if _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
            _set_stage_metric(
                stage_timings_ms,
                "recall_preference_directed_skipped_probe_timeout",
                0.0,
            )
        else:
            try:
                pref_entities = await graph_store.find_entities(
                    name="UserPreference",
                    entity_type="PreferenceProfile",
                    group_id=group_id,
                    limit=1,
                )
                if pref_entities:
                    pref_attrs = pref_entities[0].attributes or {}
                    domain_scores = pref_attrs.get("domain_preference_scores", {})
                    if domain_scores and entity_attributes:
                        preference_boosts = {}
                        for eid, attrs in entity_attributes.items():
                            etype = attrs.get("entity_type", "Other")
                            for domain, types in cfg.domain_groups.items():
                                if etype in types:
                                    pref_score = domain_scores.get(domain, 0.0)
                                    if pref_score != 0.0:
                                        preference_boosts[eid] = pref_score
                                    break
            except Exception:
                preference_boosts = None

    # Step 5: Score all candidates
    if cfg.ts_enabled:
        scored = score_candidates_thompson(
            candidates=candidates,
            spreading_bonuses=bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
            conv_fingerprint_sim=conv_fingerprint_sim,
            priming_boosts=priming_boosts,
            graph_similarities=graph_similarities,
            entity_attributes=entity_attributes,
            state_biases=state_biases,
            preference_boosts=preference_boosts,
            name_match_scores=name_match_seeds,
        )
    else:
        scored = score_candidates(
            candidates=candidates,
            spreading_bonuses=bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            activation_states=activation_states,
            now=now,
            cfg=cfg,
            conv_fingerprint_sim=conv_fingerprint_sim,
            priming_boosts=priming_boosts,
            graph_similarities=graph_similarities,
            entity_attributes=entity_attributes,
            state_biases=state_biases,
            preference_boosts=preference_boosts,
            name_match_scores=name_match_seeds,
        )
    _set_stage_metric(stage_timings_ms, "recall_scored_count", len(scored))
    _set_stage_metric(
        stage_timings_ms,
        "recall_scored_max_score",
        max((result.score for result in scored), default=0.0),
    )

    if planner_trace is not None:
        for sr in scored:
            sr.planner_support = planner_trace.support_scores.get(sr.node_id, 0.0)
            sr.planner_intents = planner_trace.intent_types.get(sr.node_id, [])
            sr.recall_trace = planner_trace.support_details.get(sr.node_id, [])

    # Step 5.05: Temporal / recency scoring for episode results
    if cfg.temporal_retrieval_enabled:
        temporal_cues = _detect_temporal_cues(query)

        if temporal_cues["is_temporal"] or temporal_cues["is_state_query"]:
            halflife = cfg.recency_halflife_days
            for sr in scored:
                if sr.result_type not in {"episode", "cue_episode"}:
                    continue
                try:
                    ep = await graph_store.get_episode_by_id(sr.node_id, group_id)
                except Exception:
                    continue
                if not ep or not ep.conversation_date:
                    continue
                ep_ts = ep.conversation_date.timestamp()
                age_days = (now - ep_ts) / 86400

                if temporal_cues["wants_latest"] or temporal_cues["is_state_query"]:
                    # Boost recent episodes exponentially
                    recency_boost = math.exp(-age_days / halflife)
                    sr.score *= 1 + recency_boost
                elif temporal_cues["wants_earliest"]:
                    # Boost oldest episodes
                    oldness_boost = 1 - math.exp(-age_days / halflife)
                    sr.score *= 1 + oldness_boost

            # Also apply recency adjustment to episode_candidates (used in
            # _merge_special_results later) so the final mix reflects temporal
            # ordering.
            for sr in episode_candidates:
                try:
                    ep = await graph_store.get_episode_by_id(sr.node_id, group_id)
                except Exception:
                    continue
                if not ep or not ep.conversation_date:
                    continue
                ep_ts = ep.conversation_date.timestamp()
                age_days = (now - ep_ts) / 86400

                if temporal_cues["wants_latest"] or temporal_cues["is_state_query"]:
                    recency_boost = math.exp(-age_days / halflife)
                    sr.score *= 1 + recency_boost
                elif temporal_cues["wants_earliest"]:
                    oldness_boost = 1 - math.exp(-age_days / halflife)
                    sr.score *= 1 + oldness_boost

            # Current-value entity lift: a "what is X now" query needs the
            # role-bearing answer entity seated in the (small) entity budget so
            # its summary — which leads with the canonical current value
            # ("Current role: ...", see result_builder) — reaches the consumer.
            # The recency loops above are episode-only, so an answer entity at a
            # near-tie gets flipped out of the top-k by ranking noise. A small
            # boost on entities that actually hold a current value makes that
            # coverage reliable. Gated on wants_latest/is_state_query (multi-hop
            # queries never trip it) and on the role affordance (generic entities
            # untouched), so it only resolves near-ties for current-value queries.
            if (
                temporal_cues["wants_latest"] or temporal_cues["is_state_query"]
            ) and cfg.current_value_entity_boost != 1.0:
                for sr in scored:
                    if sr.result_type != "entity":
                        continue
                    try:
                        ent = await graph_store.get_entity(sr.node_id, group_id)
                    except Exception:
                        continue
                    if ent and (ent.attributes or {}).get("role"):
                        sr.score *= cfg.current_value_entity_boost

            # Re-sort after temporal adjustment ((-score, node_id) tie-break)
            scored.sort(key=lambda sr: (-sr.score, sr.node_id))

            # Step 5.06: Temporal date filtering — ensure temporally
            # relevant episodes survive into final results even if their
            # semantic scores are low.
            if temporal_cues["wants_earliest"] or temporal_cues["wants_latest"]:
                # Collect episode candidates that have dates
                dated_episodes: list[tuple[float, ScoredResult]] = []
                for sr in episode_candidates:
                    try:
                        ep = await graph_store.get_episode_by_id(sr.node_id, group_id)
                    except Exception:
                        continue
                    if ep and ep.conversation_date:
                        dated_episodes.append(
                            (ep.conversation_date.timestamp(), sr),
                        )

                if dated_episodes:
                    # Sort by date: ascending for earliest, descending for latest
                    dated_episodes.sort(
                        key=lambda x: x[0],
                        reverse=temporal_cues["wants_latest"],
                    )
                    # Guarantee at least top 3 date-sorted episodes survive
                    existing_ep_ids_in_scored = {
                        sr.node_id for sr in scored if sr.result_type in {"episode", "cue_episode"}
                    }
                    for ts, sr in dated_episodes[:3]:
                        if sr.node_id not in existing_ep_ids_in_scored:
                            scored.append(sr)
                            existing_ep_ids_in_scored.add(sr.node_id)
                    # Also ensure these survive in the episode_candidates
                    # list used by _merge_special_results.
                    existing_ec_ids = {ec.node_id for ec in episode_candidates}
                    for ts, sr in dated_episodes[:3]:
                        if sr.node_id not in existing_ec_ids:
                            episode_candidates.append(sr)

    # Step 5.5: Cross-encoder re-ranking (if enabled)
    if cfg.reranker_enabled and reranker is not None:
        if type(reranker).__name__ == "NoopReranker":
            _set_stage_metric(stage_timings_ms, "recall_reranker_skipped_noop", 0.0)
        else:
            reranker_started = time.perf_counter()
            try:

                async def _rerank_results() -> None:
                    if not cfg.reranker_rerank_episodes:
                        docs: list[tuple[str, str]] = []
                        for sr in scored[: cfg.reranker_top_n * 2]:
                            entity = await graph_store.get_entity(sr.node_id, group_id)
                            text = ""
                            if entity:
                                text = entity.name
                                if entity.summary:
                                    text = f"{entity.name}: {entity.summary}"
                            docs.append((sr.node_id, text))

                        if not docs:
                            return
                        reranked = await reranker.rerank(
                            query,
                            docs,
                            top_n=cfg.reranker_top_n,
                        )
                        rerank_order = {eid: i for i, (eid, _) in enumerate(reranked)}
                        scored.sort(key=lambda sr: rerank_order.get(sr.node_id, len(scored)))
                        return

                    # --- OFF-by-default episode rerank experiment ---
                    # Build entity docs (as above) plus episode docs from raw
                    # episode content (+ chunk_context when present), rerank the
                    # merged set, then write the rerank-derived relevance score
                    # back onto the surviving episode/cue candidates so the order
                    # reaches the passage-first top-k (Step 6 re-sorts special
                    # results by score, so list order alone would be discarded).
                    entity_srs = list(scored[: cfg.reranker_top_n * 2])
                    episode_srs: list[ScoredResult] = list(episode_candidates) + list(
                        cue_candidates
                    )

                    entities = await asyncio.gather(
                        *(graph_store.get_entity(sr.node_id, group_id) for sr in entity_srs)
                    )
                    episodes = await asyncio.gather(
                        *(graph_store.get_episode_by_id(sr.node_id, group_id) for sr in episode_srs)
                    )

                    docs = []
                    for sr, entity in zip(entity_srs, entities):
                        text = ""
                        if entity:
                            text = entity.name
                            if entity.summary:
                                text = f"{entity.name}: {entity.summary}"
                        docs.append((f"entity::{sr.node_id}", text))
                    for sr, ep in zip(episode_srs, episodes):
                        text = ep.content if ep else ""
                        if sr.chunk_context:
                            text = f"{text}\n{sr.chunk_context}" if text else sr.chunk_context
                        docs.append((f"episode::{id(sr)}", text))

                    if not docs:
                        return
                    reranked = await reranker.rerank(
                        query,
                        docs,
                        top_n=len(docs),
                    )
                    # Entities: re-sort scored by rerank rank (same as OFF path).
                    entity_rank = {
                        key.split("::", 1)[1]: i
                        for i, (key, _) in enumerate(reranked)
                        if key.startswith("entity::")
                    }
                    scored.sort(key=lambda sr: entity_rank.get(sr.node_id, len(scored)))
                    # Episodes/cues: overwrite score with the cross-encoder
                    # relevance score so Step 6's (-score, node_id) sort surfaces
                    # the reranked order. Keyed by id() to keep episode and cue
                    # candidates that share a node_id independently scored.
                    episode_score = {
                        int(key.split("::", 1)[1]): rscore
                        for key, rscore in reranked
                        if key.startswith("episode::")
                    }
                    for sr in episode_srs:
                        rscore = episode_score.get(id(sr))
                        if rscore is not None:
                            sr.score = rscore

                rerank_call = _rerank_results()
                timeout_seconds = _stage_timeout_seconds(cfg, "retrieval_reranker_timeout_ms")
                if timeout_seconds is not None:
                    await asyncio.wait_for(rerank_call, timeout=timeout_seconds)
                else:
                    await rerank_call
                _add_stage_timing(stage_timings_ms, "recall_reranker", reranker_started)
            except asyncio.TimeoutError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_reranker_timeout",
                    reranker_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_reranker_cancelled",
                    reranker_started,
                )
                raise
            except Exception as e:
                logger.warning("Reranking failed (non-fatal): %s", e)

    mmr_entity_embeddings: dict[str, list[float]] = {}

    # Step 5.6: MMR diversity (if enabled)
    if cfg.mmr_enabled:
        mmr_started = time.perf_counter()
        try:
            from engram.retrieval.mmr import apply_mmr

            async def _apply_mmr() -> list[ScoredResult]:
                nonlocal mmr_entity_embeddings
                entity_embeddings: dict[str, list[float]] = {}
                mmr_ids = [sr.node_id for sr in scored[: cfg.retrieval_top_n * 2]]
                if mmr_ids and hasattr(search_index, "get_entity_embeddings"):
                    entity_embeddings = await search_index.get_entity_embeddings(
                        mmr_ids,
                        group_id=group_id,
                    )
                if mmr_ids and not entity_embeddings:
                    _set_stage_metric(
                        stage_timings_ms,
                        "recall_mmr_empty_embeddings",
                        len(mmr_ids),
                    )
                mmr_entity_embeddings = entity_embeddings

                return apply_mmr(
                    scored,
                    entity_embeddings,
                    lambda_param=cfg.mmr_lambda,
                    top_n=min(limit, cfg.retrieval_top_n),
                )

            mmr_call = _apply_mmr()
            timeout_seconds = _stage_timeout_seconds(cfg, "retrieval_mmr_timeout_ms")
            scored = (
                await asyncio.wait_for(mmr_call, timeout=timeout_seconds)
                if timeout_seconds is not None
                else await mmr_call
            )
            _add_stage_timing(stage_timings_ms, "recall_mmr", mmr_started)
        except asyncio.TimeoutError:
            _add_stage_timing(stage_timings_ms, "recall_mmr_timeout", mmr_started)
        except asyncio.CancelledError:
            _add_stage_timing(stage_timings_ms, "recall_mmr_cancelled", mmr_started)
            raise
        except Exception as e:
            logger.warning("MMR diversity failed (non-fatal): %s", e)

    # Step 5.7: GC-MMR (if enabled, replaces standard MMR)
    if cfg.gc_mmr_enabled:
        if _skip_secondary_graph_after_probe_timeout(cfg, stage_timings_ms):
            _set_stage_metric(stage_timings_ms, "recall_gc_mmr_skipped_probe_timeout", 0.0)
        else:
            gc_mmr_started = time.perf_counter()
            try:
                from engram.retrieval.gc_mmr import apply_gc_mmr

                async def _apply_gc_mmr() -> list[ScoredResult]:
                    entity_embeddings: dict[str, list[float]] = dict(mmr_entity_embeddings)
                    if not cfg.mmr_enabled:
                        gc_ids = [sr.node_id for sr in scored[: cfg.retrieval_top_n * 2]]
                        if gc_ids and hasattr(search_index, "get_entity_embeddings"):
                            fetched = await search_index.get_entity_embeddings(
                                gc_ids,
                                group_id=group_id,
                            )
                            entity_embeddings.update(fetched)
                        if gc_ids and not entity_embeddings:
                            _set_stage_metric(
                                stage_timings_ms,
                                "recall_gc_mmr_empty_embeddings",
                                len(gc_ids),
                            )

                    return await apply_gc_mmr(
                        scored,
                        graph_store=graph_store,
                        group_id=group_id,
                        entity_embeddings=entity_embeddings,
                        lambda_rel=cfg.gc_mmr_lambda_relevance,
                        lambda_div=cfg.gc_mmr_lambda_diversity,
                        lambda_conn=cfg.gc_mmr_lambda_connectivity,
                        top_n=min(limit, cfg.retrieval_top_n),
                    )

                gc_mmr_call = _apply_gc_mmr()
                timeout_seconds = _stage_timeout_seconds(cfg, "retrieval_gc_mmr_timeout_ms")
                scored = (
                    await asyncio.wait_for(gc_mmr_call, timeout=timeout_seconds)
                    if timeout_seconds is not None
                    else await gc_mmr_call
                )
                _add_stage_timing(stage_timings_ms, "recall_gc_mmr", gc_mmr_started)
            except asyncio.TimeoutError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_gc_mmr_timeout",
                    gc_mmr_started,
                )
            except asyncio.CancelledError:
                _add_stage_timing(
                    stage_timings_ms,
                    "recall_gc_mmr_cancelled",
                    gc_mmr_started,
                )
                raise
            except Exception as e:
                logger.warning("GC-MMR failed (non-fatal): %s", e)

    async def _reserve_durable_entity_slots(
        assembled: list,
        top_n_slots: int,
    ) -> list:
        """Gate-G experiment: guarantee durable-type entities the TAIL slots.

        Additive-only — a durable entity may replace the lowest-ranked tail
        item but never a top episode, so the episode-vector core is untouched
        when the knob is off (default 0) and minimally displaced when on.
        """
        slots = int(getattr(cfg, "passage_first_durable_entity_slots", 0) or 0)
        if slots <= 0:
            return assembled
        from engram.extraction.promotion import is_durable_recall_entity_type

        present = {r.node_id for r in assembled}
        reserved = 0
        for sr in scored[:20]:
            if reserved >= slots:
                break
            if sr.result_type != "entity" or sr.node_id in present:
                continue
            try:
                ent = await graph_store.get_entity(sr.node_id, group_id)
            except Exception:
                continue
            if ent is None or not is_durable_recall_entity_type(ent.entity_type):
                continue
            if len(assembled) >= top_n_slots and assembled:
                assembled = assembled[:-1]
            assembled = assembled + [sr]
            present.add(sr.node_id)
            reserved += 1
        return assembled

    # Step 6: Return top-N, mixing entities, episodes, and cue-backed episodes
    top_n = min(limit, cfg.retrieval_top_n)
    special_budget = 0
    if episode_candidates and cfg.episode_retrieval_enabled:
        special_budget += cfg.episode_retrieval_max
    if cue_candidates and cfg.cue_recall_enabled:
        special_budget += cfg.cue_recall_max

    if special_budget > 0:
        if _passage_first:
            # Passage-first strategy: allocate MORE slots to episodes, fewer to
            # entities.  Entity budget is capped at 1/3 of top_n (min 3).
            # Config override: passage_first_entity_budget >= 0 forces exact limit
            # (0 = all slots to episodes, useful for benchmarks).
            if cfg.passage_first_entity_budget >= 0:
                entity_budget = cfg.passage_first_entity_budget
            else:
                entity_budget = min(3, top_n // 3)
            entity_limit = max(0, entity_budget)
            entity_results = scored[:entity_limit]
            special_results = _merge_special_results(
                episode_candidates,
                cue_candidates,
                cfg,
                suppressed_cue_out,
            )
            if cfg.passage_first_channel_separated:
                # Channel-separated (additive): episodes/cues take the top-k by
                # their own ranked order, then entities fill ONLY leftover slots —
                # an entity can never evict an answer episode. Tests whether the
                # graph's additive upside survives once displacement is removed.
                results = special_results[:top_n]
                remaining = top_n - len(results)
                if remaining > 0 and entity_results:
                    results = results + entity_results[:remaining]
            else:
                # Legacy blend: 2x-boost episodes then sort together (a high-
                # scoring entity can still displace a lower-scored episode).
                for sr in special_results:
                    sr.score *= 2.0
                results = entity_results + special_results
                results.sort(key=lambda r: (-r.score, r.node_id))
                results = results[:top_n]
                # Restore original scores after selection (undo the 2x boost)
                for sr in results:
                    if sr.result_type in {"episode", "cue_episode"}:
                        sr.score /= 2.0
        else:
            # Default: preserve room for special results, keep at least one entity slot.
            entity_limit = max(1, top_n - special_budget)
            entity_results = scored[:entity_limit]
            special_results = _merge_special_results(
                episode_candidates,
                cue_candidates,
                cfg,
                suppressed_cue_out,
            )
            results = entity_results + special_results
            results.sort(key=lambda r: (-r.score, r.node_id))
            results = results[:top_n]
    else:
        results = scored[:top_n]

    if _passage_first:
        results = await _reserve_durable_entity_slots(results, top_n)

    # Step 7: Record Thompson Sampling feedback only for true-usage recalls.
    if cfg.ts_enabled and record_feedback:
        from engram.activation.feedback import (
            record_negative_feedback,
            record_positive_feedback,
        )

        returned_ids = {r.node_id for r in results if r.result_type == "entity"}
        all_candidate_id_set = {eid for eid, _ in candidates}
        for eid in returned_ids:
            await record_positive_feedback(eid, activation_store, cfg)
        for eid in all_candidate_id_set - returned_ids:
            await record_negative_feedback(eid, activation_store, cfg)

    return results
