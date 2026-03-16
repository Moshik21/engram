"""LLM-free query decomposition for complex temporal and multi-hop questions.

Breaks questions like "How many days between X and Y?" into independent
sub-queries that each find one piece of information.

Three strategies, applied in priority order:

1. **Template matching** — regex templates for known query structures
   (temporal-duration, temporal-comparison, change-tracking, conditional).
   Highest precision, zero false positives.

2. **Clause extraction** — lightweight syntactic splitting on coordinating
   conjunctions, commas-before-question-words, and subordinate clauses.
   Catches multi-clause questions that don't match a specific template.

3. **Noun-phrase extraction** — regex-based NP chunking extracts all
   significant noun phrases as independent search terms.  Broadest recall,
   used as fallback when the query is complex but no template matches.

All three are pure Python, zero external dependencies, <1 ms per query.
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict

logger = logging.getLogger(__name__)

DECOMPOSE_CACHE_SIZE = 256

# ---------------------------------------------------------------------------
# Strategy 1: Template-based decomposition
# ---------------------------------------------------------------------------
# Each template is (compiled_regex, extractor_function).
# The extractor receives the Match object and returns list[str] of sub-queries.
# Templates are tried in order; first match wins.

# --- helpers ----------------------------------------------------------------


def _clean(s: str) -> str:
    """Strip leading/trailing whitespace and trailing punctuation."""
    s = s.strip()
    s = re.sub(r"[?.!,;]+$", "", s).strip()
    return s


def _to_query(fragment: str, prefix: str = "") -> str:
    """Turn a fragment into a searchable sub-query string."""
    fragment = _clean(fragment)
    if not fragment:
        return ""
    if prefix:
        return f"{prefix} {fragment}"
    return fragment


# --- template extractors ----------------------------------------------------


def _extract_duration_between(m: re.Match) -> list[str]:
    """'How many days between X and Y' → [X, Y].

    Uses heuristic: if both halves share a repeated phrase prefix (e.g.
    "my trip to Paris and my trip to London"), split on the last 'and'
    that precedes the repeated prefix.
    """
    raw_x = m.group("x")
    raw_y = m.group("y")

    # The non-greedy x might have split too early.  Re-split on the last
    # " and " (or " to ") in the combined capture to find the real boundary.
    combined = raw_x + " and " + raw_y if raw_y else raw_x
    # Find the best split: last " and " that produces two non-trivial halves
    best_x, best_y = raw_x, raw_y
    # Try splitting on every " and " occurrence, prefer the one that
    # produces the most balanced (longest-short-side) split
    parts = combined.split(" and ")
    if len(parts) >= 2:
        best_balance = -1
        for i in range(1, len(parts)):
            left = " and ".join(parts[:i])
            right = " and ".join(parts[i:])
            balance = min(len(left), len(right))
            if balance > best_balance:
                best_balance = balance
                best_x = left
                best_y = right

    return [q for q in [_clean(best_x), _clean(best_y)] if q]


def _extract_duration_since(m: re.Match) -> list[str]:
    """'How long since X' → [X]."""
    return [_clean(m.group("x"))]


def _extract_comparison(m: re.Match) -> list[str]:
    """'Which came first, X or Y' → [X, Y]."""
    x = _clean(m.group("x"))
    y = _clean(m.group("y"))
    return [q for q in [x, y] if q]


def _extract_before_after(m: re.Match) -> list[str]:
    """'Before X, did Y' → [X, Y]."""
    event = _clean(m.group("event"))
    action = _clean(m.group("action"))
    return [q for q in [event, action] if q]


def _extract_change(m: re.Match) -> list[str]:
    """'What changed from X to Y' → [X, Y]."""
    x = _clean(m.group("x"))
    y = _clean(m.group("y"))
    return [q for q in [x, y] if q]


def _extract_change_about(m: re.Match) -> list[str]:
    """'What changed about X' → [X]."""
    return [_clean(m.group("x"))]


def _extract_x_and_y(m: re.Match) -> list[str]:
    """'Tell me about X and Y' or 'X versus Y' → [X, Y]."""
    x = _clean(m.group("x"))
    y = _clean(m.group("y"))
    return [q for q in [x, y] if q]


def _extract_both(m: re.Match) -> list[str]:
    """'Both X and Y' → [X, Y]."""
    x = _clean(m.group("x"))
    y = _clean(m.group("y"))
    return [q for q in [x, y] if q]


def _extract_did_x_before_after_y(m: re.Match) -> list[str]:
    """'Did I X before/after Y' → [X, Y]."""
    x = _clean(m.group("x"))
    y = _clean(m.group("y"))
    return [q for q in [x, y] if q]


# --- compiled templates (order matters) ------------------------------------

_TEMPLATES: list[tuple[re.Pattern, callable]] = [
    # Duration between two events
    # "How many days between starting the project and finishing it"
    # "How long between X and Y"
    (
        re.compile(
            r"how\s+(?:many\s+(?:days?|weeks?|months?|years?|hours?)|long|much\s+time)"
            r"\s+(?:between|from)\s+(?P<x>.+?)\s+(?:and|to|until)\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_duration_between,
    ),
    # Duration since an event
    # "How long since I started learning Rust"
    (
        re.compile(
            r"how\s+(?:many\s+(?:days?|weeks?|months?|years?|hours?)|long|much\s+time)"
            r"\s+(?:since|after|ago\s+(?:was|did))\s+(?P<x>.+)",
            re.IGNORECASE,
        ),
        _extract_duration_since,
    ),
    # Temporal comparison
    # "Which came first, X or Y"
    # "Which happened earlier, X or Y"
    # "Did X happen before Y"
    (
        re.compile(
            r"(?:which|what)\s+(?:came|happened|was|occurred|started)\s+"
            r"(?:first|earlier|before|later|after|last)"
            r"[,:]?\s*(?P<x>.+?)\s+or\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_comparison,
    ),
    # "Did I X before/after Y" pattern
    (
        re.compile(
            r"(?:did\s+(?:I|we|they|he|she)\s+)(?P<x>.+?)\s+(?:before|after)\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_did_x_before_after_y,
    ),
    # Before/after conditional
    # "Before I moved to NYC, what was I working on"
    # "After the project ended, did I start something new"
    (
        re.compile(
            r"(?:before|after|prior\s+to|following|since)\s+(?P<event>.+?)"
            r"[,]\s*(?P<action>.+)",
            re.IGNORECASE,
        ),
        _extract_before_after,
    ),
    # Change tracking with from/to
    # "What changed from version 1 to version 2"
    # "How did X change from A to B"
    (
        re.compile(
            r"(?:what|how\s+did\s+\w+)\s+(?:changed?|evolved?|shifted?|updated?|differed?)"
            r"\s+(?:from|between)\s+(?P<x>.+?)\s+(?:to|and)\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_change,
    ),
    # Change tracking without from/to
    # "What changed about my diet"
    # "What is different about the architecture"
    (
        re.compile(
            r"(?:what|how)\s+(?:changed|is\s+different|was\s+updated|evolved)"
            r"\s+(?:about|with|in|for|regarding)\s+(?P<x>.+)",
            re.IGNORECASE,
        ),
        _extract_change_about,
    ),
    # "Both X and Y" pattern
    (
        re.compile(
            r"\bboth\s+(?P<x>.+?)\s+and\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_both,
    ),
    # Versus / comparison without temporal framing
    # "X versus Y", "X vs Y", "X or Y" (only in question context)
    (
        re.compile(
            r"(?P<x>.+?)\s+(?:versus|vs\.?|compared\s+(?:to|with))\s+(?P<y>.+)",
            re.IGNORECASE,
        ),
        _extract_x_and_y,
    ),
]

# ---------------------------------------------------------------------------
# Strategy 2: Clause-based extraction
# ---------------------------------------------------------------------------
# Splits on coordinating conjunctions and clause boundaries to extract
# independent searchable fragments from multi-clause questions.

# Clause boundary markers
_CLAUSE_SPLITTERS = re.compile(
    r",\s*(?:and|but|or|yet|while|whereas|although|though|however)\s+"
    r"|(?:^|\s)(?:and\s+also|but\s+also|as\s+well\s+as)\s+"
    r"|\s+(?:and|but)\s+(?=(?:what|when|where|who|how|why|did|do|does|is|are|was|were|can|could)\s)",
    re.IGNORECASE,
)

# Question-word starters that indicate an independent sub-question
_QUESTION_STARTERS = re.compile(
    r"^(?:what|when|where|who|whom|which|how|why|did|do|does|is|are|was|were|can|could|would|should)\b",
    re.IGNORECASE,
)

# Filler phrases to strip from clause starts
_FILLER_PREFIX = re.compile(
    r"^(?:also|then|additionally|furthermore|moreover|plus|and)\s+",
    re.IGNORECASE,
)


def _extract_clauses(query: str) -> list[str]:
    """Split a multi-clause query into independent sub-queries.

    Returns list of meaningful clauses (length >= 2 words).
    Returns empty list if the query cannot be meaningfully split.
    """
    # Split on clause boundaries
    parts = _CLAUSE_SPLITTERS.split(query)

    if len(parts) <= 1:
        return []

    clauses = []
    for part in parts:
        part = _clean(part)
        part = _FILLER_PREFIX.sub("", part).strip()
        # Only keep clauses with enough substance (>= 3 words)
        if part and len(part.split()) >= 3:
            clauses.append(part)

    # Only return if we got at least 2 meaningful clauses
    if len(clauses) >= 2:
        return clauses
    return []


# ---------------------------------------------------------------------------
# Strategy 3: Noun-phrase extraction
# ---------------------------------------------------------------------------
# Lightweight regex NP chunker.  Extracts noun phrases as independent
# search terms.  No spaCy, no models — just regex.

# Word that is NOT a conjunction/preposition (valid NP continuation)
_NP_WORD = (
    r"(?!(?:and|or|but|yet|so|nor|for|in|on|at|to|from"
    r"|with|by|of|about|between|into)\b)[\w-]+"
)

# Determiner + optional adjectives + noun(s), stopping at conjunctions
_NP_PATTERN = re.compile(
    r"\b(?:my|the|a|an|this|that|these|those|his|her|their|our|its)\s+"
    r"(?:(?:new|old|first|last|recent|current|previous|next|original|main|primary"
    r"|favorite|favourite|best|worst|biggest|smallest|latest|earliest)\s+)*"
    + _NP_WORD + r"(?:\s+" + _NP_WORD + r"){0,3}",
)

# Proper noun sequences (capitalized words not at sentence start)
_PROPER_NOUN = re.compile(
    r"(?<!^)(?<![.!?]\s)(?P<pn>[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)",
)

# Stop phrases — too generic to be useful search terms
_STOP_NPS = frozenset({
    "the first", "the last", "the time", "the same", "the other",
    "the one", "the way", "the thing", "the difference",
    "my life", "the question", "this question",
    "how many", "how long", "how much",
})


def _extract_noun_phrases(query: str) -> list[str]:
    """Extract significant noun phrases from a query.

    Returns deduplicated list of NPs suitable as search sub-queries.
    Returns empty list if fewer than 2 NPs are found.
    """
    nps: list[str] = []
    seen_lower: set[str] = set()

    # Collect determiner-led NPs
    for m in _NP_PATTERN.finditer(query):
        np_text = _clean(m.group(0))
        low = np_text.lower()
        if low not in seen_lower and low not in _STOP_NPS and len(np_text) > 3:
            nps.append(np_text)
            seen_lower.add(low)

    # Collect proper noun sequences
    for m in _PROPER_NOUN.finditer(query):
        np_text = _clean(m.group("pn"))
        low = np_text.lower()
        if low not in seen_lower and len(np_text) > 3:
            nps.append(np_text)
            seen_lower.add(low)

    if len(nps) >= 2:
        return nps
    return []


# ---------------------------------------------------------------------------
# Complexity detection (reused from original, expanded)
# ---------------------------------------------------------------------------

_COMPLEX_PATTERNS = [
    r"how many days?\s+(?:between|before|after|since|until)",
    r"which\s+(?:came|happened|was|did)\s+first",
    r"(?:before|after)\s+(?:I|my|the)\s+\w+",
    r"\b(?:or|versus|vs\.?)\b.*\?",
    r"how long\s+(?:between|before|after|since|until|did it take)",
    r"what (?:changed|is different|was updated)",
    # New patterns for clause and NP strategies
    r",\s*(?:and|but)\s+(?:what|when|where|who|how|why|did|do|does|is|are|was|were)\b",
    r"\bboth\s+.+?\s+and\s+",
    r"\bcompared?\s+(?:to|with)\b",
]
_COMPLEX_RE = re.compile("|".join(_COMPLEX_PATTERNS), re.IGNORECASE)


def needs_decomposition(query: str) -> bool:
    """Check if a query is complex enough to benefit from decomposition."""
    return bool(_COMPLEX_RE.search(query))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def decompose_query(
    query: str,
    *,
    model: str = "",  # Kept for API compatibility; ignored (no LLM used)
    cache: OrderedDict | None = None,
) -> list[str]:
    """Decompose a complex query into atomic sub-queries — LLM-free.

    Applies three strategies in priority order:

    1. Template matching (highest precision)
    2. Clause extraction (medium precision)
    3. Noun-phrase extraction (broadest recall)

    Returns list of sub-queries. For simple queries, returns [query].
    """
    if not needs_decomposition(query):
        return [query]

    # Check cache
    if cache is not None and query in cache:
        cache.move_to_end(query)
        return cache[query]

    sub_queries = _decompose_deterministic(query)

    # Cache result
    if cache is not None:
        cache[query] = sub_queries
        if len(cache) > DECOMPOSE_CACHE_SIZE:
            cache.popitem(last=False)

    if len(sub_queries) > 1:
        logger.debug(
            "Decomposed query into %d sub-queries: %s", len(sub_queries), sub_queries
        )

    return sub_queries


def _decompose_deterministic(query: str) -> list[str]:
    """Core deterministic decomposition logic (sync, no cache).

    Tries strategies in order; returns first successful decomposition.
    Falls back to [query] if nothing works.
    """
    # Strategy 1: Template matching
    for pattern, extractor in _TEMPLATES:
        m = pattern.search(query)
        if m:
            result = extractor(m)
            if result and len(result) >= 1:
                logger.debug("Template match decomposed: %s", result)
                return result

    # Strategy 2: Clause extraction
    clauses = _extract_clauses(query)
    if clauses:
        logger.debug("Clause extraction decomposed: %s", clauses)
        return clauses

    # Strategy 3: Noun-phrase extraction
    nps = _extract_noun_phrases(query)
    if nps:
        logger.debug("NP extraction decomposed: %s", nps)
        return nps

    return [query]


# ---------------------------------------------------------------------------
# Kept for backward compatibility — the LLM prompt constant
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = (
    "Break this question into independent search queries. "
    "Each search query should find ONE piece of information "
    "from a user's conversation history.\n\n"
    "Rules:\n"
    "- Each sub-query should be a simple, searchable statement\n"
    '- For temporal questions ("how many days between X and Y"), '
    "create one query for X and one for Y\n"
    '- For comparison questions ("which came first, X or Y"), '
    "create one query for X and one for Y\n"
    "- For simple questions that don't need decomposition, "
    "return just the original question\n"
    "- Return as a JSON array of strings\n\n"
    "Question: {query}\n\n"
    "JSON array of search queries:"
)
