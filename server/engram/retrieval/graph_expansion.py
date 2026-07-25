"""Graph-anchored query expansion — LLM-free alternative to HyDE.

Expands search queries using real entities, relationships, and summaries
from the knowledge graph.

**Cost, measured rather than asserted (2026-07-24, live helix shell, warm).**
The docstring used to say "zero cost, ~3ms". That is true only of the queries
that do NO graph reads at all: ``_extract_query_terms`` returns ``[]`` for an
all-lowercase question, so the stage returns in ~0.03 ms having touched
nothing. Put ONE capitalised token in the query — a proper noun, i.e. exactly
what a user asks about — and the same stage issues a serial cascade of up to
``max_entities x 2 x (1 relationship read + 3 neighbour reads)`` and blew its
75 ms substage cap on **3 of 4** live A/B pairs (arm A, all-lowercase: 0/4).

That overrun is not merely a slow stage. ``graph_expand_timeout`` is one of the
two probes that arm the recall graph gate
(``recall_graph_gate.PROBE_TIMEOUT_KEYS``), so the timeout REFUSES every
secondary graph read for the rest of that recall: the entity-query pool, the
graph pool, spreading activation and entity attributes all recorded
``*_skipped_probe_timeout`` on 4/4 of the arms that tripped it. The query class
most likely to need the graph is the one that loses it.

So the fan-out is bounded here, at the source (ticket 2):

* the stage carries its own **deadline** and returns the expansion built from
  the reads that finished, instead of being cancelled and discarding all of
  them (the cancel-and-discard bug class recorded against spreading);
* independent per-term lookups run **together under a bounded semaphore**
  (``_EXPANSION_FANOUT``), never unbounded — the native executor is
  ``max_workers=4`` and unbounded fan-out is actively harmful there;
* the duplicate relationship read is gone: ``include_relationships`` and
  ``include_neighbors`` used to issue the *identical* ``get_relationships``
  call twice per entity.

The probe still works for the case it was built for: if the deadline expires
with **zero** completed reads, the store really is over budget and the caller
arms the gate exactly as before.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# Bounded fan-out for the independent per-term lookups. Mirrors the
# asyncio.Semaphore(8) already used by _apply_episode_usage_tiebreaker and the
# helix transport: enough to overlap (lib.rs releases the GIL), small enough
# not to swamp a 4-worker executor and push every other stage behind it.
_EXPANSION_FANOUT = 8

# Fraction of the caller's substage budget the expansion will spend before it
# stops issuing reads. The headroom exists so the stage returns its partial
# expansion *itself* rather than being cancelled by the caller's wait_for —
# a cancelled coroutine never reaches its return, so every completed read is
# thrown away.
_DEADLINE_SAFETY = 0.85


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
        "What",
        "When",
        "Where",
        "Who",
        "How",
        "Which",
        "Did",
        "Do",
        "Does",
        "Is",
        "Are",
        "Was",
        "Were",
        "Have",
        "Has",
        "Can",
        "Could",
        "Would",
        "Should",
        "The",
        "My",
        "I",
        "And",
        "Or",
        "But",
        "Not",
        "This",
        "That",
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
    for match in re.finditer(r"(?:do|did|have)\s+I\s+\w+\s+(.+?)(?:\?|$)", query, re.I):
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


async def _expand_from_candidates(
    candidates: Any,
    *,
    graph_store: Any,
    group_id: str,
    bounded: Any,
    include_relationships: bool,
    include_neighbors: bool,
    matched_entity_ids: set[str],
    expansion_parts: list[str],
) -> None:
    """Append expansion text for one term's entity candidates, in place."""
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

        if not (include_relationships or include_neighbors):
            continue

        # ONE relationship read serves both blocks below. They used to issue
        # the identical call twice per entity — the predicates and the
        # neighbour ids are two slices of the same list.
        rels = await bounded(lambda e=eid: graph_store.get_relationships(e, group_id=group_id))
        if not rels:
            continue

        # Add relationship predicates and targets
        if include_relationships:
            for rel in rels[:5]:
                pred = rel.predicate or ""
                target = rel.target_id or ""
                source = rel.source_id or ""
                # Format: "LIKES Kansas City Masterpiece"
                if pred:
                    other = target if source == eid else source
                    expansion_parts.append(f"{pred} {other}")

        # Add 1-hop neighbor names. Independent reads -> issued together under
        # the same bound instead of one at a time.
        if include_neighbors:
            neighbor_ids = [
                (rel.target_id if rel.source_id == eid else rel.source_id) for rel in rels[:3]
            ]
            neighbors = await asyncio.gather(
                *(
                    bounded(lambda n=other_id: graph_store.get_entity(n, group_id))
                    for other_id in neighbor_ids
                    if other_id and other_id != eid
                )
            )
            for neighbor in neighbors:
                if neighbor and neighbor.name:
                    expansion_parts.append(neighbor.name)


async def expand_query_from_graph(
    query: str,
    graph_store: Any,
    group_id: str,
    *,
    max_entities: int = 5,
    include_relationships: bool = True,
    include_neighbors: bool = True,
    max_expansion_chars: int = 500,
    deadline_seconds: float | None = None,
    stats_out: dict[str, float] | None = None,
) -> str:
    """Expand a query using knowledge graph context.

    Extracts entity references from the query, looks them up in the
    knowledge graph, and enriches the query with entity summaries,
    relationship predicates, and neighbor names.

    Returns the expanded query string. Falls back to the original
    query if no graph matches are found.

    ``deadline_seconds`` bounds the wall clock the stage will spend issuing
    reads. When it is spent the expansion is built from whatever finished and
    returned — the reads are NOT discarded. ``stats_out`` reports
    ``reads`` (completed graph reads) and ``truncated`` (1.0 when the deadline
    stopped it early) so the caller can tell "ran out of fan-out budget" from
    "the store is over budget"; see the module docstring for why that
    distinction decides whether the recall graph gate arms.
    """
    started = time.perf_counter()
    attempts = 0
    reads = 0
    truncated = False
    if stats_out is not None:
        stats_out["attempts"] = 0.0
        stats_out["reads"] = 0.0
        stats_out["truncated"] = 0.0

    def _record() -> None:
        """Publish progress into the caller's dict AS IT HAPPENS.

        Deliberately not published once at the end: a cancelled coroutine never
        reaches its return, so an end-only write would leave the caller unable
        to tell "the store was slow" from "this stage never got to run" — the
        exact conflation that makes a starved event loop look like an
        over-budget graph.
        """
        if stats_out is None:
            return
        stats_out["attempts"] = float(attempts)
        stats_out["reads"] = float(reads)
        stats_out["truncated"] = 1.0 if truncated else 0.0

    def _spent() -> bool:
        if deadline_seconds is None:
            return False
        return (time.perf_counter() - started) >= deadline_seconds

    terms = _extract_query_terms(query)
    if not terms:
        _record()
        return query

    expansion_parts: list[str] = []
    matched_entity_ids: set[str] = set()
    fanout = asyncio.Semaphore(_EXPANSION_FANOUT)

    async def _bounded(coro_factory) -> Any:
        """Run one graph read under the shared fan-out bound, or skip it."""
        nonlocal attempts, reads, truncated
        async with fanout:
            if _spent():
                truncated = True
                _record()
                return None
            # Published BEFORE the await: a read that is issued and never comes
            # back is exactly the case where the store IS over budget, and the
            # caller has to be able to see it from a cancelled coroutine.
            attempts += 1
            _record()
            try:
                value = await coro_factory()
            except Exception:
                return None
            reads += 1
            _record()
            return value

    # The per-term candidate lookups are independent of each other, so they are
    # issued TOGETHER (bounded) instead of one after another. Results are
    # consumed in term order below, so the expansion text is unchanged.
    lookups = await asyncio.gather(
        *(
            _bounded(lambda t=term: graph_store.find_entity_candidates(t, group_id))
            for term in terms[:max_entities]
        )
    )

    for candidates in lookups:
        if not candidates:
            continue
        try:
            await _expand_from_candidates(
                candidates,
                graph_store=graph_store,
                group_id=group_id,
                bounded=_bounded,
                include_relationships=include_relationships,
                include_neighbors=include_neighbors,
                matched_entity_ids=matched_entity_ids,
                expansion_parts=expansion_parts,
            )
        except Exception:
            continue

    _record()
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

#
# Patterns are anchored (^ ... $) so the capture groups consume the *whole*
# query and the statement is rebuilt from the captured groups with
# ``match.expand``. Un-anchored ``re.sub`` previously spliced the replacement
# into the middle of the original query, producing garbled fragments
# ("What is my favorite language?" -> "my f isavorite language?").
#
# Order matters: more specific patterns must come before more general ones
# (e.g. "what is my favorite X" before "what is my X").
#
_REFORMULATION_PATTERNS = [
    # "How many X do I have?" -> "I have X"
    (r"^\s*how\s+many\s+(.+?)\s+do\s+I\s+have\s*[?.!]*\s*$", r"I have \1"),
    # "What is my favorite X?" -> "my favorite X is"
    (r"^\s*what\s+is\s+my\s+favorite\s+(.+?)\s*[?.!]*\s*$", r"my favorite \1 is"),
    # "What is my X?" -> "my X is"
    (r"^\s*what\s+is\s+my\s+(.+?)\s*[?.!]*\s*$", r"my \1 is"),
    # "What X do I like?" -> "I like X"
    (r"^\s*what\s+(.+?)\s+do\s+I\s+(\w+)\s*[?.!]*\s*$", r"I \2 \1"),
    # "Where do I X?" -> "I X at"
    (r"^\s*where\s+do\s+I\s+(.+?)\s*[?.!]*\s*$", r"I \1 at"),
    # "When did I X?" -> "I X on"
    (r"^\s*when\s+did\s+I\s+(.+?)\s*[?.!]*\s*$", r"I \1 on"),
    # "Who is my X?" -> "my X is"
    (r"^\s*who\s+is\s+my\s+(.+?)\s*[?.!]*\s*$", r"my \1 is"),
    # "Do I X?" -> "I X"
    (r"^\s*do\s+I\s+(.+?)\s*[?.!]*\s*$", r"I \1"),
]
_COMPILED_PATTERNS = [(re.compile(p, re.I), r) for p, r in _REFORMULATION_PATTERNS]

# A reformulated statement should read as normal whitespace-separated words.
# Reject any output that contains a token longer than this many characters,
# which signals two words were spliced together without a space.
_MAX_REFORMULATED_TOKEN = 30


def _is_clean_reformulation(text: str) -> bool:
    """Reject reformulations with whitespace-spliced word fragments."""
    if not text:
        return False
    return all(len(token) <= _MAX_REFORMULATED_TOKEN for token in text.split())


def reformulate_query(query: str) -> str | None:
    """Convert a question into a statement form for better embedding match.

    Rebuilds the statement from captured groups (anchored full-query match)
    rather than splicing a replacement into the original string. Returns the
    reformulated query, or None if no pattern matches or the result looks
    garbled (callers fall back to the original query).
    Zero cost, <1ms.
    """
    for pattern, replacement in _COMPILED_PATTERNS:
        match = pattern.match(query)
        if match:
            reformulated = match.expand(replacement).strip()
            if reformulated and _is_clean_reformulation(reformulated):
                return reformulated
            return None
    return None
