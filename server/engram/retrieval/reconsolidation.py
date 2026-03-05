"""Reconsolidation: labile window tracking for recently recalled entities."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class LabileEntry:
    """A recently recalled entity in its labile reconsolidation window."""

    entity_id: str
    name: str
    entity_type: str
    summary: str
    query: str
    recalled_at: float
    modification_count: int = 0


class LabileWindowTracker:
    """In-memory TTL cache for recently recalled entities."""

    def __init__(self, ttl: float = 300.0, max_entries: int = 50) -> None:
        self._entries: dict[str, LabileEntry] = {}
        self._ttl = ttl
        self._max_entries = max_entries

    def mark_labile(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        query: str,
    ) -> None:
        """Mark entity as labile. Does NOT extend existing window."""
        self._evict_expired()
        if entity_id in self._entries:
            return  # Window not extended (biological accuracy)
        if len(self._entries) >= self._max_entries:
            self._evict_oldest()
        self._entries[entity_id] = LabileEntry(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            summary=summary,
            query=query,
            recalled_at=time.time(),
        )

    def get_labile(self, entity_id: str) -> LabileEntry | None:
        """Get labile entry if still within window."""
        self._evict_expired()
        return self._entries.get(entity_id)

    def record_modification(self, entity_id: str) -> bool:
        """Increment modification count. Returns False if not labile."""
        entry = self.get_labile(entity_id)
        if entry is None:
            return False
        entry.modification_count += 1
        return True

    def is_budget_exceeded(self, entity_id: str, max_mods: int) -> bool:
        """Check if modification budget is exceeded."""
        entry = self.get_labile(entity_id)
        return entry is not None and entry.modification_count >= max_mods

    def _evict_expired(self) -> None:
        """Remove entries past their TTL."""
        now = time.time()
        expired = [
            eid for eid, entry in self._entries.items()
            if now - entry.recalled_at > self._ttl
        ]
        for eid in expired:
            del self._entries[eid]

    def _evict_oldest(self) -> None:
        """Remove the oldest entry to make room."""
        if not self._entries:
            return
        oldest_id = min(self._entries, key=lambda eid: self._entries[eid].recalled_at)
        del self._entries[oldest_id]


def jaccard_token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between lowercased word token sets."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


def attempt_reconsolidation(
    entity,
    new_content: str,
    labile_entry: LabileEntry,
    cfg,
) -> dict | None:
    """Attempt to reconsolidate an entity with new information.

    Returns dict of updates or None if no reconsolidation should happen.
    """
    overlap = jaccard_token_overlap(new_content, labile_entry.summary)
    if overlap < cfg.reconsolidation_overlap_threshold:
        return None

    updates: dict = {}

    # Summary: append new info, cap at 500 chars
    existing_summary = labile_entry.summary or ""
    # Extract a brief snippet from new content
    snippet = new_content[:200].strip()
    if snippet and snippet != existing_summary:
        if existing_summary:
            merged = f"{existing_summary}; {snippet}"
            if len(merged) > 500:
                merged = merged[:497] + "..."
            updates["summary"] = merged
        else:
            updates["summary"] = snippet[:500]

    # Entity type and name: NEVER changed
    # Attributes: only merge for non-identity-core entities
    is_identity_core = getattr(entity, "identity_core", False)
    if not is_identity_core:
        # No attribute changes needed for basic reconsolidation
        pass

    return updates if updates else None
