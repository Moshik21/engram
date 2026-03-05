"""Cross-encoder refinement for uncertain merge/infer decisions.

Tier 1 in the tiered scoring architecture:
  Tier 0 — Multi-signal rules (85%)
  Tier 1 — Cross-encoder (10%)  ← this module
  Tier 2 — Numpy classifier (future)
  Tier 3 — LLM fallback (2%)

Uses the same FastEmbedReranker (Xenova/ms-marco-MiniLM-L-6-v2) already
loaded for retrieval reranking — zero new dependencies.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# Singleton cross-encoder (lazy-loaded)
_cross_encoder = None
_ce_lock = asyncio.Lock()


async def _get_cross_encoder():
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is not None:
        return _cross_encoder
    async with _ce_lock:
        if _cross_encoder is not None:
            return _cross_encoder
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            _cross_encoder = TextCrossEncoder(
                model_name="Xenova/ms-marco-MiniLM-L-6-v2",
            )
            logger.info("Cross-encoder loaded for consolidation scoring")
            return _cross_encoder
        except ImportError:
            logger.warning("fastembed not installed — cross-encoder unavailable")
            return None
        except Exception as exc:
            logger.warning("Cross-encoder init failed: %s", exc)
            return None


async def cross_encoder_score(text_a: str, text_b: str) -> float | None:
    """Score similarity between two texts using the cross-encoder.

    Returns a relevance score in [0, 1], or None if unavailable.
    The cross-encoder scores (query, document) pairs — we use text_a
    as query and text_b as document.
    """
    model = await _get_cross_encoder()
    if model is None:
        return None

    try:
        scores = await asyncio.to_thread(
            lambda: list(model.rerank(text_a, [text_b])),
        )
        if scores:
            # ms-marco scores can be negative; sigmoid-normalize to [0, 1]
            import math

            raw = float(scores[0])
            return 1.0 / (1.0 + math.exp(-raw))
        return None
    except Exception as exc:
        logger.warning("Cross-encoder scoring failed: %s", exc)
        return None


def _entity_description(entity) -> str:
    """Build a text description of an entity for cross-encoder input."""
    parts = [entity.name]
    if getattr(entity, "entity_type", None):
        parts.append(f"({entity.entity_type})")
    summary = getattr(entity, "summary", None)
    if summary:
        parts.append(f"— {summary}")
    return " ".join(parts)


async def refine_merge_verdict(
    ea,
    eb,
    current_confidence: float,
    merge_threshold: float = 0.82,
) -> tuple[str, float]:
    """Refine an uncertain merge verdict using cross-encoder.

    Returns (verdict, refined_confidence).
    """
    desc_a = _entity_description(ea)
    desc_b = _entity_description(eb)

    ce_score = await cross_encoder_score(desc_a, desc_b)
    if ce_score is None:
        return "keep_separate", current_confidence

    # Blend: 60% cross-encoder, 40% original multi-signal
    refined = 0.60 * ce_score + 0.40 * current_confidence

    if refined >= merge_threshold:
        return "merge", round(refined, 4)
    else:
        return "keep_separate", round(refined, 4)


async def refine_infer_verdict(
    entity_a,
    entity_b,
    predicate: str,
    current_score: float,
    approve_threshold: float = 0.65,
    reject_threshold: float = 0.40,
) -> tuple[str, float]:
    """Refine an uncertain infer verdict using cross-encoder.

    Returns (verdict, refined_score).
    """
    desc_a = _entity_description(entity_a)
    desc_b = _entity_description(entity_b)
    query = f"{desc_a} {predicate} {desc_b}"

    ce_score = await cross_encoder_score(query, f"{desc_a} and {desc_b}")
    if ce_score is None:
        return "uncertain", current_score

    # Blend: 50% cross-encoder, 50% original multi-signal
    refined = 0.50 * ce_score + 0.50 * current_score

    if refined >= approve_threshold:
        return "approved", round(refined, 4)
    elif refined < reject_threshold:
        return "rejected", round(refined, 4)
    else:
        return "uncertain", round(refined, 4)
