"""Discourse classifier — detects meta-commentary about the memory system itself.

Prevents system debugging discussions, activation scores, and pipeline telemetry
from contaminating the knowledge graph with non-real-world facts.
"""

from __future__ import annotations

import re

# Patterns that indicate system/meta discourse about Engram internals.
# Each pattern is compiled once at import time for performance.
_SYSTEM_PATTERNS: list[re.Pattern[str]] = [
    # Entity/graph IDs
    re.compile(r"\b(?:ent|ep|rel|cyc)_[a-f0-9]", re.IGNORECASE),
    # System metrics
    re.compile(
        r"activation[ _]?(?:score|current)|access[_ ]?count|spreading[_ ]?bonus",
        re.IGNORECASE,
    ),
    # Pipeline terms
    re.compile(
        r"extraction pipeline|entity resolution|consolidation (?:phase|cycle|engine)"
        r"|triage (?:phase|score)|project_episode|store_episode|ingest_episode",
        re.IGNORECASE,
    ),
    # Engram internals
    re.compile(
        r"knowledge graph (?:node|entity|store)|graph ?store|MCP tool|episode worker",
        re.IGNORECASE,
    ),
    # System operations
    re.compile(
        r"retrieval pipeline|embedding distance|FTS5|vector search|hybrid search",
        re.IGNORECASE,
    ),
    # Meta-testing vocabulary
    re.compile(
        r"cold session|indirect retrieval|test case for|example case for",
        re.IGNORECASE,
    ),
]


def classify_discourse(content: str) -> str:
    """Classify content as ``"world"``, ``"system"``, or ``"hybrid"``.

    Returns:
        ``"system"``  — 2+ system-pattern matches (pure meta-commentary)
        ``"hybrid"``  — exactly 1 match (mixed content)
        ``"world"``   — 0 matches (real-world facts)
    """
    if not content:
        return "world"

    matches = sum(1 for pat in _SYSTEM_PATTERNS if pat.search(content))

    if matches >= 2:
        return "system"
    if matches == 1:
        return "hybrid"
    return "world"
