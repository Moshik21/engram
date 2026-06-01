"""Deterministic importance gate for episode clusters.

Reuses the same zero-cost heuristic signals as the background worker's fallback
scorer (content length, capitalized-token density, keyword density — see
``engram/ingestion/worker_scoring.py`` lines 88-90) so a cluster is only
synthesized when its aggregate content carries enough signal to be worth a
durable observation. Pure arithmetic over the cluster text: no LLM, no RNG, no
clock, so the verdict is byte-stable across runs.
"""

from __future__ import annotations

import re

_CAP_TOKEN = re.compile(r"\b[A-Z][a-z]+\b")


def episode_importance(content: str) -> float:
    """Score a single episode's content in [0, ~0.6] using worker heuristics.

    Mirrors the worker fallback: length (0.25), capitalized-token density
    (0.20), and a fixed novelty floor (0.15). No goal/emotional boosts — those
    require live stores and would break determinism for the offline gate.
    """
    if not content:
        return 0.0
    length_score = min(len(content) / 500, 1.0) * 0.25
    caps = len(_CAP_TOKEN.findall(content))
    keyword_score = min(caps / 10, 1.0) * 0.20
    novelty_score = 0.15
    return length_score + keyword_score + novelty_score


def cluster_importance(contents: list[str]) -> float:
    """Aggregate cluster importance as the mean per-episode importance.

    The mean (not sum) keeps the score on the same [0, ~0.6] scale as a single
    episode so ``observer_reflect_min_importance`` is comparable across cluster
    sizes and a large cluster of low-signal episodes does not pass trivially.
    """
    if not contents:
        return 0.0
    scores = [episode_importance(c) for c in contents]
    return sum(scores) / len(scores)
