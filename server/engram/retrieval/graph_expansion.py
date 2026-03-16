"""Graph-anchored query expansion — LLM-free alternative to HyDE.

Expands search queries using real entities, relationships, and summaries
from the knowledge graph. Zero cost, ~3ms latency.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_query_terms(query: str) -> list[str]:
    """Extract potential entity names and key terms from a query.

    Uses multiple strategies:
    1. Title-case phrases (e.g., "Kansas City Masterpiece")
    2. Quoted strings
    3. Nouns after possessives ("my car", "my favorite")
    4. Key noun phrases
    """
    terms: list[str] = []

    # Quoted strings
    for match in re.finditer(r'"([^"]+)"', query):
        terms.append(match.group(1))
    for match in re.finditer(r"'([^']+)'", query):
        terms.append(match.group(1))

    # Title-case phrases (2+ consecutive capitalized words)
    for match in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query):
        terms.append(match.group(1))

    # Single capitalized words (not at sentence start, not common words)
    stop = {
        "What", "When", "Where", "Who", "How", "Which", "Did", "Do", "Does",
        "Is", "Are", "Was", "Were", "Have", "Has", "Can", "Could", "Would",
        "Should", "The", "My", "I", "And", "Or", "But", "Not", "This", "That",
    }
    words = query.split()
    for i, w in enumerate(words):
        if w and w[0].isupper() and w not in stop and i > 0:
            terms.append(w.rstrip("?.,!"))

    # Nouns after possessives: "my X", "my favorite X"
    for match in re.finditer(
        r"\bmy\s+(?:favorite\s+|preferred\s+|current\s+)?(\w+(?:\s+\w+)?)",
        query,
        re.I,
    ):
        term = match.group(1).rstrip("?.,!")
        if len(term) > 2 and term.lower() not in {"name", "self", "own"}:
            terms.append(term)

    # Object of verb: "do I like", "do I use", "did I attend"
    for match in re.finditer(
        r"(?:do|did|have)\s+I\s+\w+\s+(.+?)(?:\?|$)", query, re.I
    ):
        obj = match.group(1).strip().rstrip("?.,!")
        if obj and len(obj) > 2:
            terms.append(obj)

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in terms:
        t_lower = t.lower().strip()
        if t_lower and t_lower not in seen and len(t_lower) > 1:
            seen.add(t_lower)
            unique.append(t)

    return unique


async def expand_query_from_graph(
    query: str,
    graph_store: Any,
    group_id: str,
    *,
    max_entities: int = 5,
    include_relationships: bool = True,
    include_neighbors: bool = True,
    max_expansion_chars: int = 500,
) -> str:
    """Expand a query using knowledge graph context.

    Extracts entity references from the query, looks them up in the
    knowledge graph, and enriches the query with entity summaries,
    relationship predicates, and neighbor names.

    Returns the expanded query string. Falls back to the original
    query if no graph matches are found.
    """
    terms = _extract_query_terms(query)
    if not terms:
        return query

    expansion_parts: list[str] = []
    matched_entity_ids: set[str] = set()

    for term in terms[:max_entities]:
        try:
            # Look up entity candidates
            candidates = await graph_store.find_entity_candidates(
                term, group_id
            )
            if not candidates:
                continue

            for entity in candidates[:2]:  # Top 2 matches per term
                eid = entity.id
                if eid in matched_entity_ids:
                    continue
                matched_entity_ids.add(eid)

                # Add entity name and summary
                if entity.name:
                    expansion_parts.append(entity.name)
                if entity.summary and len(entity.summary) > 5:
                    expansion_parts.append(entity.summary)

                # Add relationship predicates and targets
                if include_relationships:
                    try:
                        rels = await graph_store.get_relationships(
                            eid, group_id=group_id
                        )
                        for rel in rels[:5]:
                            pred = rel.predicate or ""
                            target = rel.target_id or ""
                            source = rel.source_id or ""
                            # Format: "LIKES Kansas City Masterpiece"
                            if pred:
                                other = target if source == eid else source
                                expansion_parts.append(f"{pred} {other}")
                    except Exception:
                        pass

                # Add 1-hop neighbor names
                if include_neighbors:
                    try:
                        neighbors = await graph_store.get_relationships(
                            eid, group_id=group_id
                        )
                        for rel in neighbors[:3]:
                            other_id = (
                                rel.target_id
                                if rel.source_id == eid
                                else rel.source_id
                            )
                            if other_id and other_id != eid:
                                # Try to get neighbor entity name
                                try:
                                    neighbor = await graph_store.get_entity(
                                        other_id, group_id
                                    )
                                    if neighbor and neighbor.name:
                                        expansion_parts.append(neighbor.name)
                                except Exception:
                                    pass
                    except Exception:
                        pass

        except Exception:
            continue

    if not expansion_parts:
        return query

    # Build expanded query: original + graph context
    expansion = " ".join(expansion_parts)
    if len(expansion) > max_expansion_chars:
        expansion = expansion[:max_expansion_chars]

    expanded = f"{query} {expansion}"
    logger.debug("Graph expansion: '%s' -> '%s'", query[:50], expanded[:100])
    return expanded


# --- Template-based query reformulation ---

_REFORMULATION_PATTERNS = [
    # "What X do I like?" -> "I like X"
    (r"what\s+(.+?)\s+do\s+I\s+(\w+)\??", r"I \2 \1"),
    # "What is my X?" -> "my X is"
    (r"what\s+is\s+my\s+(.+?)\??", r"my \1 is"),
    # "What is my favorite X?" -> "my favorite X is"
    (r"what\s+is\s+my\s+favorite\s+(.+?)\??", r"my favorite \1 is"),
    # "Do I X?" -> "I X"
    (r"do\s+I\s+(.+?)\??", r"I \1"),
    # "Where do I X?" -> "I X at"
    (r"where\s+do\s+I\s+(.+?)\??", r"I \1 at"),
    # "When did I X?" -> "I X on"
    (r"when\s+did\s+I\s+(.+?)\??", r"I \1 on"),
    # "Who is my X?" -> "my X is"
    (r"who\s+is\s+my\s+(.+?)\??", r"my \1 is"),
    # "How many X do I have?" -> "I have X"
    (r"how\s+many\s+(.+?)\s+do\s+I\s+have\??", r"I have \1"),
]
_COMPILED_PATTERNS = [(re.compile(p, re.I), r) for p, r in _REFORMULATION_PATTERNS]


def reformulate_query(query: str) -> str | None:
    """Convert a question into a statement form for better embedding match.

    Returns the reformulated query, or None if no pattern matches.
    Zero cost, <1ms.
    """
    for pattern, replacement in _COMPILED_PATTERNS:
        match = pattern.search(query)
        if match:
            return pattern.sub(replacement, query).strip()
    return None
