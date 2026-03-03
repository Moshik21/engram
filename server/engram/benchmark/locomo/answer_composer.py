"""Compose answers from retrieved entity summaries."""

from __future__ import annotations


def compose_answer(
    entity_summaries: list[str],
    max_length: int = 200,
) -> str:
    """Compose an answer from top entity summaries.

    Concatenates up to 3 summaries, truncates to max_length characters.
    """
    if not entity_summaries:
        return ""

    parts = entity_summaries[:3]
    combined = " ".join(s.strip() for s in parts if s and s.strip())

    if len(combined) > max_length:
        # Truncate at word boundary
        truncated = combined[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > max_length // 2:
            truncated = truncated[:last_space]
        return truncated.rstrip()

    return combined
