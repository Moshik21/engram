"""Shared text guard utilities for meta-contamination detection."""

from __future__ import annotations

import re

# Patterns that indicate a summary contains system-internal meta-commentary
_META_SUMMARY_PATTERN = re.compile(
    r"activation[ _]?(?:score|current)|access[_ ]?count"
    r"|knowledge graph|graph (?:node|entity|store)"
    r"|retrieval|embedding|consolidation|triage"
    r"|entity (?:resolution|extraction|in the)"
    r"|cold session|test case|example case"
    r"|MCP tool|episode worker|spreading activation"
    r"|\b(?:ent|ep|rel|cyc)_[a-f0-9]",
    re.IGNORECASE,
)


def is_meta_summary(text: str) -> bool:
    """Check if a summary fragment contains system-internal patterns."""
    return bool(_META_SUMMARY_PATTERN.search(text))
