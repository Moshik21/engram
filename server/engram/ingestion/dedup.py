"""In-memory capture deduplication for auto-observe and replay."""

from __future__ import annotations

import hashlib
import time


class CaptureDedupCache:
    """Track recently seen capture content hashes with a TTL."""

    def __init__(
        self,
        *,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.cache: dict[str, float] = {}

    def check(self, content: str) -> bool:
        """Return True when content was recently seen and should be skipped."""
        now = time.time()
        if len(self.cache) > self.max_entries:
            self.evict_stale(now=now)

        content_hash = self.content_hash(content)
        timestamp = self.cache.get(content_hash)
        if timestamp is not None and now - timestamp < self.ttl_seconds:
            return True

        self.cache[content_hash] = now
        return False

    def evict_stale(self, *, now: float | None = None) -> None:
        """Remove entries older than the configured TTL."""
        current = time.time() if now is None else now
        stale = [
            content_hash
            for content_hash, timestamp in self.cache.items()
            if current - timestamp > self.ttl_seconds
        ]
        for content_hash in stale:
            del self.cache[content_hash]

    @staticmethod
    def content_hash(content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:16]
