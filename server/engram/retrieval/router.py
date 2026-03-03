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


class QueryType(Enum):
    DIRECT_LOOKUP = "direct_lookup"
    TEMPORAL = "temporal"
    FREQUENCY = "frequency"
    ASSOCIATIVE = "associative"
    DEFAULT = "default"


_WEIGHT_PROFILES: dict[QueryType, tuple[float, float, float, float]] = {
    QueryType.DIRECT_LOOKUP: (0.75, 0.10, 0.05, 0.10),
    QueryType.TEMPORAL: (0.20, 0.55, 0.15, 0.10),
    QueryType.FREQUENCY: (0.15, 0.60, 0.15, 0.10),
    QueryType.ASSOCIATIVE: (0.55, 0.10, 0.20, 0.15),
    QueryType.DEFAULT: (0.40, 0.25, 0.15, 0.15),
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

    if _ASSOCIATIVE_KEYWORDS.search(query):
        return QueryType.ASSOCIATIVE

    if search_results and search_results[0][1] > 0.8:
        return QueryType.DIRECT_LOOKUP

    return QueryType.DEFAULT


# ---------------------------------------------------------------------------
# Weight application
# ---------------------------------------------------------------------------


def apply_route(query_type: QueryType, cfg: ActivationConfig) -> ActivationConfig:
    """Return a copy of cfg with weights overridden for the given query type."""
    sem, act, spread, edge = _WEIGHT_PROFILES[query_type]
    routed = deepcopy(cfg)
    routed.weight_semantic = sem
    routed.weight_activation = act
    routed.weight_spreading = spread
    routed.weight_edge_proximity = edge
    return routed
