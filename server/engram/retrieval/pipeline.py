"""Full retrieval pipeline: FTS5 → activation → spreading → scoring."""

from __future__ import annotations

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
from engram.retrieval.router import QueryType, apply_route, classify_query
from engram.retrieval.scorer import (
    ScoredResult,
    score_candidates,
    score_candidates_thompson,
)

if TYPE_CHECKING:
    from engram.retrieval.working_memory import WorkingMemoryBuffer

logger = logging.getLogger(__name__)


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


def _merge_special_results(
    episode_candidates: list[ScoredResult],
    cue_candidates: list[ScoredResult],
    cfg: ActivationConfig,
) -> list[ScoredResult]:
    """Merge episode and cue-backed candidates, keeping the strongest per episode."""
    special_results: list[ScoredResult] = []

    if episode_candidates and cfg.episode_retrieval_enabled:
        episode_candidates.sort(key=lambda r: r.score, reverse=True)
        special_results.extend(episode_candidates[: cfg.episode_retrieval_max])

    if cue_candidates and cfg.cue_recall_enabled:
        cue_candidates.sort(key=lambda r: r.score, reverse=True)
        best_by_episode = {result.node_id: result for result in special_results}
        for cue_result in cue_candidates[: cfg.cue_recall_max]:
            existing = best_by_episode.get(cue_result.node_id)
            if existing is None or cue_result.score > existing.score:
                best_by_episode[cue_result.node_id] = cue_result
        special_results = list(best_by_episode.values())

    special_results.sort(key=lambda r: r.score, reverse=True)
    return special_results


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

    # Fetch entity count for dynamic pool sizing
    total_entities = 0
    try:
        stats = await graph_store.get_stats(group_id)
        if isinstance(stats, dict):
            total_entities = int(stats.get("entity_count", 0))
    except Exception:
        pass

    # Step 1: Generate candidates (multi-pool or single-pool)
    if cfg.multi_pool_enabled:
        from engram.retrieval.candidate_pool import generate_candidates

        # Pre-classify query for pool composition multipliers
        pre_query_type = await classify_query(query)

        candidates = await generate_candidates(
            query=query,
            group_id=group_id,
            search_index=search_index,
            activation_store=activation_store,
            graph_store=graph_store,
            cfg=cfg,
            now=now,
            working_memory=working_memory,
            total_entities=total_entities,
            query_type=pre_query_type,
        )
        query_type = await classify_query(query, search_results=candidates)
        if enable_routing or query_type == QueryType.TEMPORAL:
            cfg = apply_route(query_type, cfg)
        temporal_mode = query_type == QueryType.TEMPORAL
    else:
        # Original single-pool path — scale retrieval_top_k
        top_k = _scale_limit(cfg.retrieval_top_k, total_entities, 5, 500)
        search_results = await search_index.search(query=query, group_id=group_id, limit=top_k)
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

    # Step 0.5: Planner-driven multi-intent recall (Phase 2)
    if cfg.recall_planner_enabled:
        planner_trace = await execute_recall_plan(
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
        if planner_trace.merged_candidates:
            candidates = planner_trace.merged_candidates

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
    episode_candidates: list[ScoredResult] = []
    if cfg.episode_retrieval_enabled:
        search_episodes = _optional_recall_capability(
            search_index,
            "search_episodes",
            "episode retrieval",
            sibling_methods=("search_episode_cues",),
        )
        if search_episodes is not None:
            try:
                ep_results = await search_episodes(
                    query=query,
                    group_id=group_id,
                    limit=cfg.episode_retrieval_max * 3,
                )
                for ep_id, sem_sim in ep_results:
                    ep_score = original_weight_semantic * sem_sim * cfg.episode_retrieval_weight
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
            except Exception as e:
                logger.warning("Episode search failed (non-fatal): %s", e)

    # Step 1.2: Cue-backed episode search
    cue_candidates: list[ScoredResult] = []
    if cfg.cue_recall_enabled:
        search_episode_cues = _optional_recall_capability(
            search_index,
            "search_episode_cues",
            "cue recall",
            sibling_methods=("search_episodes",),
        )
        if search_episode_cues is not None:
            try:
                cue_results = await search_episode_cues(
                    query=query,
                    group_id=group_id,
                    limit=cfg.cue_recall_max * 3,
                )
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
            except Exception as e:
                logger.warning("Cue search failed (non-fatal): %s", e)

    # Step 1.8: Entity-first fallback when search finds few candidates
    if not candidates:
        candidates = await _inject_entity_matches(
            query,
            group_id,
            graph_store,
            candidates,
        )

    if not candidates:
        special_results = _merge_special_results(episode_candidates, cue_candidates, cfg)
        if special_results:
            return special_results[: min(limit, cfg.retrieval_top_n)]
        return []

    # Step 2: Batch get activation states
    entity_ids = [eid for eid, _ in candidates]
    activation_states = await activation_store.batch_get(entity_ids)

    # Step 2.5: Goal priming (Brain Architecture)
    goal_seeds: list[tuple[str, float]] = []
    if cfg.goal_priming_enabled:
        from engram.retrieval.goals import (
            compute_goal_priming_seeds,
            identify_active_goals,
        )

        active_goals = await identify_active_goals(
            graph_store, activation_store, group_id, cfg, cache=goal_cache,
        )
        goal_seeds = compute_goal_priming_seeds(active_goals, cfg)

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
        if (cfg.conv_session_entity_seeds_enabled and conv_context is not None):
            for entry in conv_context.get_top_entities(cfg.conv_multi_query_top_entities):
                if entry.entity_id not in seed_node_ids:
                    energy = cfg.conv_session_entity_seed_energy * min(
                        1.0, entry.mention_weight / 5.0,
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
        seed_entity_types = {}
        for seed_id, _ in seeds:
            try:
                ent = await graph_store.get_entity(seed_id, group_id)
                if ent:
                    seed_entity_types[seed_id] = ent.entity_type
            except Exception:
                pass

    # Step 4: Spread activation
    bonuses, hop_distances = await spread_activation(
        seeds,
        graph_store,
        cfg,
        group_id=group_id,
        community_store=community_store,
        context_gate=context_gate,
        seed_entity_types=seed_entity_types,
    )

    # Step 4.1: Inhibitory spreading (Brain Architecture)
    if cfg.inhibitory_spreading_enabled:
        from engram.retrieval.inhibition import apply_inhibition

        bonuses = await apply_inhibition(
            bonuses=bonuses,
            hop_distances=hop_distances,
            seed_node_ids=seed_node_ids,
            graph_store=graph_store,
            search_index=search_index,
            group_id=group_id,
            cfg=cfg,
        )

    # Step 4.5: Merge spreading-discovered entities with real semantic similarity
    existing_ids = {eid for eid, _ in candidates}
    new_ids = [nid for nid in bonuses if nid not in existing_ids and bonuses[nid] > 0.0]
    if new_ids:
        new_states = await activation_store.batch_get(new_ids)
        activation_states.update(new_states)
        discovered_sims = await search_index.compute_similarity(
            query=query,
            entity_ids=new_ids,
            group_id=group_id,
        )
        candidates = candidates + [(nid, discovered_sims.get(nid, 0.0)) for nid in new_ids]

    # Step 4.6: Fingerprint similarity computation (Wave 2)
    conv_fingerprint_sim: dict[str, float] | None = None
    if (cfg.conv_fingerprint_enabled and conv_context is not None
            and cfg.conv_context_rerank_weight > 0.0):
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
        try:
            all_candidate_ids = [eid for eid, _ in candidates]
            # Determine preferred method: try enabled methods that have data
            methods_to_try = []
            if cfg.graph_embedding_node2vec_enabled:
                methods_to_try.append("node2vec")
            if cfg.graph_embedding_transe_enabled:
                methods_to_try.append("transe")
            if cfg.graph_embedding_gnn_enabled:
                methods_to_try.append("gnn")

            if methods_to_try and hasattr(search_index, "get_graph_embeddings"):
                # Get seed entity IDs (from spreading activation seeds)
                query_seed_ids = list(seed_node_ids) if seed_node_ids else []

                for method in methods_to_try:
                    if not query_seed_ids:
                        # Fall back to top candidates as proxy seeds
                        query_seed_ids = [
                            eid for eid, _ in sorted(
                                candidates, key=lambda x: x[1], reverse=True,
                            )[:3]
                        ]
                        if not query_seed_ids:
                            break

                    # Fetch graph embeddings for seeds + candidates
                    all_ids = list(set(query_seed_ids + all_candidate_ids))
                    graph_embs = await search_index.get_graph_embeddings(
                        all_ids, method=method, group_id=group_id,
                    )
                    if not graph_embs:
                        continue

                    # Find seed entities that have graph embeddings
                    seed_embs = {
                        sid: graph_embs[sid]
                        for sid in query_seed_ids
                        if sid in graph_embs
                    }
                    if not seed_embs:
                        continue

                    import numpy as np

                    from engram.storage.sqlite.vectors import (
                        cosine_similarity as _cos_sim,
                    )

                    seed_vecs = list(seed_embs.values())
                    query_graph_vec = np.mean(seed_vecs, axis=0).tolist()

                    graph_similarities = {}
                    for eid, g_emb in graph_embs.items():
                        if eid in seed_embs:
                            continue  # Don't score seeds against themselves
                        graph_similarities[eid] = max(
                            0.0, _cos_sim(query_graph_vec, g_emb),
                        )
                    break  # Found a method with data

                if not graph_similarities:
                    graph_similarities = None
        except Exception as e:
            logger.warning("Graph similarity computation failed (non-fatal): %s", e)

    # Step 4.9: Fetch entity attributes for emotional + state boosts
    entity_attributes: dict[str, dict] | None = None
    needs_attrs = cfg.emotional_salience_enabled or cfg.state_dependent_retrieval_enabled
    if needs_attrs:
        entity_attributes = {}
        all_candidate_ids = [eid for eid, _ in candidates]
        for eid in all_candidate_ids:
            try:
                ent = await graph_store.get_entity(eid, group_id)
                if ent:
                    attrs = (
                        dict(ent.attributes)
                        if isinstance(ent.attributes, dict)
                        else {}
                    )
                    # Include entity_type for domain mapping
                    if ent.entity_type:
                        attrs["entity_type"] = ent.entity_type
                    entity_attributes[eid] = attrs
            except Exception:
                pass
        if not entity_attributes:
            entity_attributes = None

    # Step 4.95: State-dependent retrieval biases (Brain Architecture)
    state_biases: dict[str, float] | None = None
    if cfg.state_dependent_retrieval_enabled and conv_context is not None:
        cog_state = getattr(conv_context, "cognitive_state", None)
        if cog_state is not None and entity_attributes:
            from engram.retrieval.state import compute_state_bias

            state_biases = {}
            for eid, _ in candidates:
                attrs = entity_attributes.get(eid, {})
                etype = attrs.get("entity_type", "Other")
                bias = compute_state_bias(
                    cog_state, attrs, etype, cfg,
                    domain_groups=cfg.domain_groups,
                )
                if bias > 0:
                    state_biases[eid] = bias
            if not state_biases:
                state_biases = None

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
        )

    if planner_trace is not None:
        for sr in scored:
            sr.planner_support = planner_trace.support_scores.get(sr.node_id, 0.0)
            sr.planner_intents = planner_trace.intent_types.get(sr.node_id, [])
            sr.recall_trace = planner_trace.support_details.get(sr.node_id, [])

    # Step 5.5: Cross-encoder re-ranking (if enabled)
    if cfg.reranker_enabled and reranker is not None:
        try:
            # Fetch entity summaries for reranking
            docs: list[tuple[str, str]] = []
            for sr in scored[: cfg.reranker_top_n * 2]:
                entity = await graph_store.get_entity(sr.node_id, group_id)
                text = ""
                if entity:
                    text = entity.name
                    if entity.summary:
                        text = f"{entity.name}: {entity.summary}"
                docs.append((sr.node_id, text))

            if docs:
                reranked = await reranker.rerank(
                    query,
                    docs,
                    top_n=cfg.reranker_top_n,
                )
                # Build reranked order
                rerank_order = {eid: i for i, (eid, _) in enumerate(reranked)}
                scored.sort(
                    key=lambda sr: rerank_order.get(sr.node_id, len(scored)),
                )
        except Exception as e:
            logger.warning("Reranking failed (non-fatal): %s", e)

    # Step 5.6: MMR diversity (if enabled)
    if cfg.mmr_enabled:
        try:
            from engram.retrieval.mmr import apply_mmr

            # Fetch entity embeddings for MMR
            entity_embeddings: dict[str, list[float]] = {}
            mmr_ids = [sr.node_id for sr in scored[: cfg.retrieval_top_n * 2]]
            if hasattr(search_index, "_vectors") and hasattr(search_index, "_embeddings_enabled"):
                if search_index._embeddings_enabled:
                    from engram.storage.sqlite.vectors import unpack_vector

                    for eid in mmr_ids:
                        cursor = await search_index._vectors.db.execute(
                            "SELECT embedding, dimensions FROM embeddings "
                            "WHERE id = ? AND group_id = ?",
                            (eid, group_id),
                        )
                        row = await cursor.fetchone()
                        if row:
                            entity_embeddings[eid] = unpack_vector(
                                row["embedding"],
                                row["dimensions"],
                            )

            scored = apply_mmr(
                scored,
                entity_embeddings,
                lambda_param=cfg.mmr_lambda,
                top_n=min(limit, cfg.retrieval_top_n),
            )
        except Exception as e:
            logger.warning("MMR diversity failed (non-fatal): %s", e)

    # Step 5.7: GC-MMR (if enabled, replaces standard MMR)
    if cfg.gc_mmr_enabled:
        try:
            from engram.retrieval.gc_mmr import apply_gc_mmr

            # Reuse entity_embeddings from MMR block, or fetch if MMR was disabled
            if not cfg.mmr_enabled:
                entity_embeddings = {}
                gc_ids = [sr.node_id for sr in scored[: cfg.retrieval_top_n * 2]]
                has_vecs = hasattr(search_index, "_vectors")
                if has_vecs and hasattr(search_index, "_embeddings_enabled"):
                    if search_index._embeddings_enabled:
                        from engram.storage.sqlite.vectors import unpack_vector

                        for eid in gc_ids:
                            cursor = await search_index._vectors.db.execute(
                                "SELECT embedding, dimensions FROM embeddings "
                                "WHERE id = ? AND group_id = ?",
                                (eid, group_id),
                            )
                            row = await cursor.fetchone()
                            if row:
                                entity_embeddings[eid] = unpack_vector(
                                    row["embedding"],
                                    row["dimensions"],
                                )

            scored = await apply_gc_mmr(
                scored,
                graph_store=graph_store,
                group_id=group_id,
                entity_embeddings=entity_embeddings,
                lambda_rel=cfg.gc_mmr_lambda_relevance,
                lambda_div=cfg.gc_mmr_lambda_diversity,
                lambda_conn=cfg.gc_mmr_lambda_connectivity,
                top_n=min(limit, cfg.retrieval_top_n),
            )
        except Exception as e:
            logger.warning("GC-MMR failed (non-fatal): %s", e)

    # Step 6: Return top-N, mixing entities, episodes, and cue-backed episodes
    top_n = min(limit, cfg.retrieval_top_n)
    special_budget = 0
    if episode_candidates and cfg.episode_retrieval_enabled:
        special_budget += cfg.episode_retrieval_max
    if cue_candidates and cfg.cue_recall_enabled:
        special_budget += cfg.cue_recall_max

    if special_budget > 0:
        # Preserve room for special results while keeping at least one entity slot.
        entity_limit = max(1, top_n - special_budget)
        entity_results = scored[:entity_limit]
        special_results = _merge_special_results(episode_candidates, cue_candidates, cfg)
        results = entity_results + special_results
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:top_n]
    else:
        results = scored[:top_n]

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
