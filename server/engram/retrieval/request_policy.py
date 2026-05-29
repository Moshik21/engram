"""Recall request policy helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from engram.config import ActivationConfig

T = TypeVar("T")

_PASSIVE_INTERACTIONS = {"surfaced", "selected", "dismissed", "corrected"}
_TRUE_USAGE_INTERACTIONS = {"used", "confirmed"}


def recall_fetch_limit(
    cfg: ActivationConfig,
    limit: int,
    *,
    conv_context: object | None,
) -> int:
    """Return the retrieval fetch limit including any near-miss window."""
    if cfg.conv_near_miss_enabled and conv_context is not None:
        return limit + cfg.conv_near_miss_window
    return limit


def should_record_ranking_feedback(
    *,
    record_access: bool,
    interaction_type: str | None,
) -> bool:
    """Decide whether ranking feedback should learn from this recall turn."""
    if interaction_type in _PASSIVE_INTERACTIONS:
        return False
    if interaction_type in _TRUE_USAGE_INTERACTIONS:
        return True
    return record_access


_EPISODE_RESULT_TYPES = frozenset({"episode", "cue_episode"})


def split_primary_and_near_miss_results(
    scored_results: Sequence[T],
    limit: int,
    *,
    near_miss_enabled: bool,
    cfg: ActivationConfig | None = None,
) -> tuple[list[T], list[T]]:
    """Split retrieved candidates into primary results and near-miss tail.

    Under ``passage_first`` the entity count in the primary window is capped to a
    budget sized against the CONSUMER ``limit`` (not the inflated overfetch the
    pipeline assembled against). Without this, a few high-scoring entities survive
    the truncation and EVICT answer-bearing episodes from the consumer's top-k —
    the graph then hurts recall (entities displacing passages). Capping keeps
    episodes from being decimated while preserving a small entity lead for
    fact/"current value" queries that answer from an entity. Entities beyond the
    cap fall to the near-miss tail.
    """
    if (
        cfg is not None
        and getattr(cfg, "retrieval_strategy", None) == "passage_first"
        and limit > 0
    ):
        if cfg.passage_first_entity_budget >= 0:
            entity_budget = cfg.passage_first_entity_budget
        else:
            entity_budget = min(3, max(1, limit // 3))
        primary: list[T] = []
        overflow: list[T] = []
        entity_count = 0
        for result in scored_results:
            is_entity = getattr(result, "result_type", None) not in _EPISODE_RESULT_TYPES
            if len(primary) < limit and not (is_entity and entity_count >= entity_budget):
                primary.append(result)
                if is_entity:
                    entity_count += 1
            else:
                overflow.append(result)
        near_miss_results = overflow if near_miss_enabled else []
        return primary, near_miss_results

    primary_results = list(scored_results[:limit])
    near_miss_results = list(scored_results[limit:]) if near_miss_enabled else []
    return primary_results, near_miss_results
