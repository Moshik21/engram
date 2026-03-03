"""Working memory buffer — LRU cache of recently accessed entities/episodes."""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass


@dataclass
class WorkingMemoryEntry:
    """A single item in working memory."""

    item_id: str
    item_type: str  # "entity" or "episode"
    score: float
    timestamp: float
    query: str


class WorkingMemoryBuffer:
    """LRU-evicting buffer that tracks recently discussed entities and queries.

    Items decay linearly from 1.0 to 0.0 over ``ttl_seconds``.
    """

    def __init__(self, capacity: int = 20, ttl_seconds: float = 300.0) -> None:
        self._entries: OrderedDict[str, WorkingMemoryEntry] = OrderedDict()
        self._capacity = capacity
        self._ttl = ttl_seconds
        self._recent_queries: deque[tuple[str, float]] = deque(maxlen=5)

    # ── Mutators ─────────────────────────────────────────────────

    def add(
        self,
        item_id: str,
        item_type: str,
        score: float,
        query: str,
        now: float,
    ) -> None:
        """Add or update an entry, moving it to the end (most-recent)."""
        if item_id in self._entries:
            self._entries.move_to_end(item_id)
            entry = self._entries[item_id]
            entry.score = score
            entry.timestamp = now
            entry.query = query
            entry.item_type = item_type
        else:
            if len(self._entries) >= self._capacity:
                self._entries.popitem(last=False)  # evict oldest
            self._entries[item_id] = WorkingMemoryEntry(
                item_id=item_id,
                item_type=item_type,
                score=score,
                timestamp=now,
                query=query,
            )

    def add_query(self, query: str, now: float) -> None:
        """Track a recent query string."""
        self._recent_queries.append((query, now))

    def clear(self) -> None:
        """Empty both the entry buffer and the query deque."""
        self._entries.clear()
        self._recent_queries.clear()

    # ── Readers ──────────────────────────────────────────────────

    def get_candidates(self, now: float) -> list[tuple[str, float, str]]:
        """Return non-expired items as ``(item_id, recency_score, item_type)``.

        ``recency_score`` decays linearly from 1.0 to 0.0 over TTL.
        """
        results: list[tuple[str, float, str]] = []
        for entry in self._entries.values():
            elapsed = now - entry.timestamp
            if elapsed >= self._ttl:
                continue
            recency_score = max(0.0, 1.0 - elapsed / self._ttl)
            results.append((entry.item_id, recency_score, entry.item_type))
        return results

    def get_recent_queries(self, n: int = 3) -> list[str]:
        """Return the *n* most recent non-expired query strings (newest first)."""
        results: list[str] = []
        for query, _ts in reversed(self._recent_queries):
            if len(results) >= n:
                break
            results.append(query)
        return results

    @property
    def size(self) -> int:
        """Number of entries currently in the buffer."""
        return len(self._entries)
