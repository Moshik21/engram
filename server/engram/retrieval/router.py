"""Query-type router for adaptive retrieval weight selection."""

from __future__ import annotations

import re
from copy import deepcopy
from enum import Enum

from engram.config import ActivationConfig

# ---------------------------------------------------------------------------
# Query types & weight profiles
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS = re.compile(
    r"\b(recent|recently|lately|last|today|yesterday|this\s+week|this\s+month"
    r"|just\s+now|earlier|earlier\s+today|few\s+days|what's\s+new|what\s+is\s+new)\b",
    re.IGNORECASE,
)

_ASSOCIATIVE_KEYWORDS = re.compile(
    r"\b(connect|connected|connection|link|linked|relation|relationship"
    r"|between|bridge|associated|related\s+to|ties|ties\s+to)\b",
    re.IGNORECASE,
)

_FREQUENCY_KEYWORDS = re.compile(
    r"\b(most|frequently|focus|top|primary|important|engaged|referenced"
    r"|interact.*most|come.*back|key.*areas)\b",
    re.IGNORECASE,
)

_CREATION_KEYWORDS = re.compile(
    r"\b(wrote|written|authored|created|built|made|published|designed"
    r"|composed|produced|founded|invented)\b",
    re.IGNORECASE,
)


class QueryType(Enum):
    DIRECT_LOOKUP = "direct_lookup"
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    ASSOCIATIVE = "associative"
    CREATION = "creation"
    DEFAULT = "default"


# Legacy (flag-off) profiles: (semantic, activation, spreading, edge).
# Frozen for byte-identity while usage_ranking_enabled=False (the shipped
# default). Dies when M2.6 flips the default — the usage table below is the
# successor.
_WEIGHT_PROFILES: dict[QueryType, tuple[float, float, float, float]] = {
    QueryType.DIRECT_LOOKUP: (0.75, 0.10, 0.05, 0.10),
    QueryType.TEMPORAL: (0.20, 0.55, 0.15, 0.10),
    QueryType.FREQUENCY: (0.15, 0.60, 0.15, 0.10),
    QueryType.ASSOCIATIVE: (0.55, 0.10, 0.20, 0.15),
    QueryType.CREATION: (0.30, 0.10, 0.25, 0.30),
    QueryType.DEFAULT: (0.40, 0.25, 0.15, 0.15),
}

# Usage-ranking (flag-on) profiles per the D3/F5 table (RF_target_design §3):
# (semantic, spreading, edge, beta_route). The activation column is DROPPED
# and its share redistributed proportionally among the surviving terms
# (exact fractions; the design table shows them rounded to 2 decimals).
# beta_route is the bounded multiplicative usage tiebreaker
# final = composite_sem * (1 + beta_route * u), beta_max = 0.30.
_USAGE_WEIGHT_PROFILES: dict[QueryType, tuple[float, float, float, float]] = {
    QueryType.DIRECT_LOOKUP: (0.75 / 0.90, 0.05 / 0.90, 0.10 / 0.90, 0.05),
    QueryType.TEMPORAL: (0.20 / 0.45, 0.15 / 0.45, 0.10 / 0.45, 0.25),
    QueryType.FREQUENCY: (0.15 / 0.40, 0.15 / 0.40, 0.10 / 0.40, 0.30),
    QueryType.ASSOCIATIVE: (0.55 / 0.90, 0.20 / 0.90, 0.15 / 0.90, 0.10),
    QueryType.CREATION: (0.30 / 0.85, 0.25 / 0.85, 0.30 / 0.85, 0.10),
    QueryType.DEFAULT: (0.40 / 0.70, 0.15 / 0.70, 0.15 / 0.70, 0.10),
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


async def classify_query(
    query: str,
    search_results: list[tuple[str, float]] | None = None,
) -> QueryType:
    """Classify a query into a retrieval type.

    Detection rules (evaluated in order):
    1. Temporal keywords → TEMPORAL
    2. Frequency keywords → FREQUENCY
    3. Associative keywords → ASSOCIATIVE
    4. Top search result score > 0.8 → DIRECT_LOOKUP
    5. Default → DEFAULT
    """
    if _TEMPORAL_KEYWORDS.search(query):
        return QueryType.TEMPORAL

    if _FREQUENCY_KEYWORDS.search(query):
        return QueryType.FREQUENCY

    if _CREATION_KEYWORDS.search(query):
        return QueryType.CREATION

    if _ASSOCIATIVE_KEYWORDS.search(query):
        return QueryType.ASSOCIATIVE

    if search_results and search_results[0][1] > 0.8:
        return QueryType.DIRECT_LOOKUP

    return QueryType.DEFAULT


# ---------------------------------------------------------------------------
# Weight application
# ---------------------------------------------------------------------------


def apply_route(query_type: QueryType, cfg: ActivationConfig) -> ActivationConfig:
    """Return a copy of cfg with weights overridden for the given query type.

    Flag OFF (``usage_ranking_enabled=False``, shipped default) — byte-
    identical to the pre-M2.3 router: legacy 4-column profile with the
    activation column, writes exactly (semantic, activation, spreading,
    edge_proximity).

    Flag ON — D3/F5 profile: the activation column is gone; writes exactly
    (semantic, spreading, edge_proximity, usage_beta_route). The base
    ``weight_activation`` is left untouched because the flag-on scoring
    path never reads it.

    Explicit-zero kill-switch (M0.5 semantics, both shapes): any core weight
    the base cfg sets to exactly 0.0 stays 0.0 (zero = term disabled). The
    disabled term's profile share is redistributed proportionally among the
    enabled terms so the routed weights still sum to the profile total.
    beta_route is not part of the redistribution — its kill is the
    ``usage_ranking_enabled`` flag itself.
    """
    routed = deepcopy(cfg)
    if cfg.usage_ranking_enabled:
        sem_p, spread_p, edge_p, beta_route = _USAGE_WEIGHT_PROFILES[query_type]
        profile3 = (sem_p, spread_p, edge_p)
        base3 = (cfg.weight_semantic, cfg.weight_spreading, cfg.weight_edge_proximity)
        enabled_total = sum(p for p, b in zip(profile3, base3) if b != 0.0)
        if enabled_total > 0.0:
            scale = sum(profile3) / enabled_total
            sem, spread, edge = (p * scale if b != 0.0 else 0.0 for p, b in zip(profile3, base3))
        else:
            sem = spread = edge = 0.0
        routed.weight_semantic = sem
        routed.weight_spreading = spread
        routed.weight_edge_proximity = edge
        routed.usage_beta_route = beta_route
        return routed

    profile = _WEIGHT_PROFILES[query_type]
    base = (
        cfg.weight_semantic,
        cfg.weight_activation,
        cfg.weight_spreading,
        cfg.weight_edge_proximity,
    )
    enabled_total = sum(p for p, b in zip(profile, base) if b != 0.0)
    if enabled_total > 0.0:
        scale = sum(profile) / enabled_total
        sem, act, spread, edge = (p * scale if b != 0.0 else 0.0 for p, b in zip(profile, base))
    else:
        sem = act = spread = edge = 0.0
    routed.weight_semantic = sem
    routed.weight_activation = act
    routed.weight_spreading = spread
    routed.weight_edge_proximity = edge
    return routed
