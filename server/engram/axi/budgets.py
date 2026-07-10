"""Output budgeting helpers for AXI commands."""

from __future__ import annotations

from typing import Any

DEFAULT_TOKEN_BUDGET = 800
CHARS_PER_TOKEN = 4


def token_budget_to_chars(tokens: int | None) -> int:
    """Return a conservative character budget for an approximate token budget."""
    if tokens is None or tokens <= 0:
        tokens = DEFAULT_TOKEN_BUDGET
    return max(160, int(tokens) * CHARS_PER_TOKEN)


def truncate_text(
    text: str | None,
    *,
    budget_tokens: int | None,
    minimum_chars: int = 160,
) -> tuple[str, bool, int]:
    """Return text capped to an approximate token budget.

    The third return value is the original character count, so callers can expose
    an explicit truncation marker without retaining full content in output.
    """
    if not text:
        return "", False, 0
    original = str(text)
    original_len = len(original)
    max_chars = max(minimum_chars, token_budget_to_chars(budget_tokens))
    if original_len <= max_chars:
        return original, False, original_len
    suffix = f"... [truncated; {original_len} chars total]"
    keep = max(minimum_chars, max_chars - len(suffix))
    return original[:keep].rstrip() + suffix, True, original_len


def compact_whitespace(text: str | None) -> str:
    """Collapse whitespace for one-line summaries."""
    if not text:
        return ""
    return " ".join(str(text).split())


def first_present(mapping: dict[str, Any] | None, *keys: str) -> Any:
    """Return the first non-empty value from a mapping."""
    if not mapping:
        return None
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None
