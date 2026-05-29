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


def split_primary_and_near_miss_results(
    scored_results: Sequence[T],
    limit: int,
    *,
    near_miss_enabled: bool,
) -> tuple[list[T], list[T]]:
    """Split retrieved candidates into primary results and near-miss tail."""
    primary_results = list(scored_results[:limit])
    near_miss_results = list(scored_results[limit:]) if near_miss_enabled else []
    return primary_results, near_miss_results
