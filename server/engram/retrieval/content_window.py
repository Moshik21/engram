"""Graduated content window for recall items.

A recall item hands the agent the episode's full content. Measured 2026-09-04/06
on the fresh store, the fresh-agent battery counted 59,165 chars for 14/14
answers against 13,335 chars for 4/14 from repo files, with single rows of
2,019-4,098 chars (the hook and bootstrap capture caps). The answer tokens sit
next to the query terms in almost every hit, so a row longer than the window
is cut to a span centred on the densest cluster of query-term matches (the
head when nothing matches). The cut is never silent: both edges carry a marker
and the item reports ``full_chars`` and ``windowed``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from engram.config import ActivationConfig

ELLIPSIS = "[…]"

# Slack, in characters, for moving a cut edge onto whitespace so the window
# never splits a word.
_SNAP_CHARS = 32

_TERM_RE = re.compile(r"[a-z0-9][a-z0-9_.'-]*")
_WS_RE = re.compile(r"\s")

# Question and glue words: they match everywhere, so they cannot locate the
# answer. Content words of three characters or more are kept.
_STOPWORDS = frozenset(
    {
        "about",
        "and",
        "any",
        "are",
        "but",
        "can",
        "did",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "into",
        "its",
        "not",
        "our",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "those",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def query_terms(query: str) -> list[str]:
    """Casefolded content terms of the query, in first-seen order."""
    terms: list[str] = []
    for token in _TERM_RE.findall(query.casefold()):
        token = token.strip(".'-")
        if len(token) < 3 or token in _STOPWORDS or token in terms:
            continue
        terms.append(token)
    return terms


@dataclass(frozen=True)
class ContentWindow:
    query: str
    window_chars: int

    def apply(self, content: str) -> tuple[str, bool]:
        return window_content(content, self.query, window_chars=self.window_chars)


def content_window_for(cfg: Any, query: str) -> ContentWindow:
    """Read the window size from the activation config (stubs get the default)."""
    default = ActivationConfig.model_fields["recall_content_window_chars"].default
    raw = getattr(cfg, "recall_content_window_chars", default)
    try:
        window_chars = int(raw)
    except (TypeError, ValueError):
        window_chars = int(default)
    return ContentWindow(query=query, window_chars=window_chars)


# A row whose distinct query terms are spread out gets a stretched window, up
# to this many base windows, so the answer is not cut in half; rows whose terms
# are farther apart than that fall back to the densest single window. Measured
# 2026-09-06: at 450 one rig question lost both its groups because they sat
# ~1,000 chars apart in every top row (14/14 unwindowed, 13/14 windowed).
STRETCH_CAP = 3


def window_content(content: str, query: str, *, window_chars: int) -> tuple[str, bool]:
    """Return ``(text, windowed)``; content at or under the window is untouched."""
    if window_chars <= 0 or len(content) <= window_chars:
        return content, False
    terms = query_terms(query)
    start = _densest_window_start(content, terms, window_chars)
    size = window_chars
    span = _covering_span(content, terms, start, window_chars)
    if span is not None:
        span_start, span_end = span
        span_len = span_end - span_start
        if span_len > window_chars and span_len <= window_chars * STRETCH_CAP:
            size = min(len(content), span_len + 2 * _SNAP_CHARS)
            start = max(0, min(span_start - _SNAP_CHARS, len(content) - size))
    end = min(len(content), start + size)
    if start > 0:
        gap = _WS_RE.search(content, start, min(end, start + _SNAP_CHARS))
        if gap:
            start = gap.end()
    if end < len(content):
        gaps = list(_WS_RE.finditer(content, max(start, end - _SNAP_CHARS), end))
        if gaps:
            end = gaps[-1].start()
    text = content[start:end].strip()
    if start > 0:
        text = f"{ELLIPSIS} {text}"
    if end < len(content):
        text = f"{text} {ELLIPSIS}"
    return text, True


def _term_matches(content: str, terms: list[str]) -> list[tuple[int, int, int]]:
    """(position, term index, length) for every case-insensitive match, sorted."""
    matches: list[tuple[int, int, int]] = []
    for index, term in enumerate(terms):
        for found in re.finditer(re.escape(term), content, re.IGNORECASE):
            matches.append((found.start(), index, found.end() - found.start()))
    matches.sort()
    return matches


def _covering_span(
    content: str, terms: list[str], anchor_start: int, window: int
) -> tuple[int, int] | None:
    """Smallest span holding one match of EVERY term that occurs in the content,
    choosing for each term the match nearest the densest window's centre."""
    matches = _term_matches(content, terms)
    if not matches:
        return None
    centre = anchor_start + window / 2
    nearest: dict[int, tuple[int, int]] = {}
    for position, index, length in matches:
        best = nearest.get(index)
        if best is None or abs(position - centre) < abs(best[0] - centre):
            nearest[index] = (position, length)
    if len(nearest) < 2:
        return None
    lo = min(p for p, _ in nearest.values())
    hi = max(p + n for p, n in nearest.values())
    return lo, hi


def _densest_window_start(content: str, terms: list[str], window: int) -> int:
    """Start of the window covering the most distinct query terms, then matches."""
    # Match on the ORIGINAL text (case-insensitive regex): a casefolded copy can
    # change length (ß -> ss, İ -> i + combining dot) and shift every position.
    matches = _term_matches(content, terms)
    if not matches:
        return 0
    best_score = (0, 0)
    best_low = best_high = 0
    counts: dict[int, int] = {}
    right = 0
    for left, (position, _, _) in enumerate(matches):
        while right < len(matches) and (
            right == left or matches[right][0] + matches[right][2] <= position + window
        ):
            counts[matches[right][1]] = counts.get(matches[right][1], 0) + 1
            right += 1
        score = (len(counts), right - left)
        if score > best_score:
            best_score = score
            best_low = position
            best_high = matches[right - 1][0] + matches[right - 1][2]
        term_index = matches[left][1]
        counts[term_index] -= 1
        if counts[term_index] == 0:
            del counts[term_index]
    start = int((best_low + best_high) / 2 - window / 2)
    return max(0, min(start, len(content) - window))
