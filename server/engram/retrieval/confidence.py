"""Recall result relevance-confidence post-processing."""

from __future__ import annotations

import inspect
import logging
from typing import Any

from engram.config import ActivationConfig
from engram.retrieval.belief import BeliefMapScorer
from engram.retrieval.relevance import RelevanceScorer
from engram.retrieval.scorer import ScoredResult

logger = logging.getLogger(__name__)


class RecallConfidenceApplier:
    """Apply relevance-confidence scoring for raw recall results."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        search_index: object,
    ) -> None:
        self._cfg = cfg
        self._search_index = search_index

    async def apply(self, *, query: str, results: list[dict[str, Any]]) -> None:
        if not self._cfg.relevance_confidence_enabled or not results:
            return
        try:
            await apply_relevance_confidence(
                query=query,
                results=results,
                search_index=self._search_index,
                cfg=self._cfg,
            )
        except Exception:
            logger.debug("Relevance scoring failed, continuing without it", exc_info=True)


async def apply_relevance_confidence(
    *,
    query: str,
    results: list[dict[str, Any]],
    search_index: object,
    cfg: ActivationConfig,
) -> None:
    """Mutate raw recall results with embedding-based relevance confidence."""
    if not results:
        return

    provider = getattr(search_index, "_provider", None)
    if provider is None or not hasattr(provider, "dimension"):
        return
    dimension = provider.dimension
    if inspect.iscoroutinefunction(dimension):
        return
    dimension_value = dimension()
    if dimension_value <= 0:
        return

    scorer = RelevanceScorer(provider)
    entity_summaries: dict[str, str] = {}
    episode_contents: dict[str, str] = {}
    chunk_texts: dict[str, str] = {}
    should_embed_episode_text = bool(
        cfg.relevance_confidence_episode_text_embeddings_enabled
    )

    for result in results:
        result_type = result.get("result_type", "entity")
        if result_type == "entity":
            entity = result.get("entity", {})
            if isinstance(entity, dict) and entity.get("id") and entity.get("summary"):
                entity_summaries[entity["id"]] = entity["summary"]
            continue

        if result_type not in {"episode", "cue_episode"}:
            continue

        episode = result.get("episode", {})
        if not isinstance(episode, dict):
            continue
        episode_id = episode.get("id", "")
        if should_embed_episode_text and episode_id and episode.get("content"):
            episode_contents[episode_id] = episode["content"]

        chunk = result.get("chunk_context") or ""
        if not chunk and result_type == "cue_episode":
            cue = result.get("cue", {})
            if isinstance(cue, dict):
                chunk = cue.get("compressed_content", "")
        if should_embed_episode_text and episode_id and chunk:
            chunk_texts[episode_id] = chunk

    scored = [_scored_result_from_recall_item(result) for result in results]
    await scorer.score_results(
        query=query,
        results=scored,
        entity_summaries=entity_summaries,
        episode_contents=episode_contents,
        chunk_texts=chunk_texts,
        query_vec=getattr(search_index, "_last_query_vec", None),
    )

    for result, scored_result in zip(results, scored):
        breakdown = result.get("score_breakdown")
        if isinstance(breakdown, dict):
            breakdown["relevance_confidence"] = round(
                scored_result.relevance_confidence,
                4,
            )

        if cfg.belief_map_enabled and result.get("result_type") == "entity":
            belief_scorer = BeliefMapScorer(cfg)
            entity_data = result.get("entity", {})
            belief = belief_scorer.calculate_belief(
                entity_data=entity_data,
                relevance=scored_result.relevance_confidence,
                activation=scored_result.activation,
            )
            result["belief_map"] = belief.to_dict()


def _scored_result_from_recall_item(result: dict[str, Any]) -> ScoredResult:
    result_type = result.get("result_type", "entity")
    breakdown = result.get("score_breakdown", {})
    if not isinstance(breakdown, dict):
        breakdown = {}

    node_id = ""
    if result_type == "entity":
        entity = result.get("entity", {})
        if isinstance(entity, dict):
            node_id = entity.get("id", "")
    elif result_type in {"episode", "cue_episode"}:
        episode = result.get("episode", {})
        if isinstance(episode, dict):
            node_id = episode.get("id", "")

    return ScoredResult(
        node_id=node_id,
        score=result.get("score", 0.0),
        semantic_similarity=breakdown.get("semantic", 0.0),
        activation=breakdown.get("activation", 0.0),
        spreading=0.0,
        edge_proximity=breakdown.get("edge_proximity", 0.0),
        result_type=result_type,
    )
