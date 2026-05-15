"""Tests for capture deduplication cache behavior."""

from __future__ import annotations

import time

from engram.ingestion.dedup import CaptureDedupCache


def test_capture_dedup_blocks_recent_duplicate():
    dedup = CaptureDedupCache(ttl_seconds=300)
    content = "Hello world, this is a test"

    assert dedup.check(content) is False
    assert dedup.check(content) is True


def test_capture_dedup_allows_expired_content():
    dedup = CaptureDedupCache(ttl_seconds=300)
    content = "Expiring content test"

    assert dedup.check(content) is False
    for key in list(dedup.cache):
        dedup.cache[key] = time.time() - 400

    assert dedup.check(content) is False


def test_capture_dedup_evicts_stale_entries_when_cache_is_large():
    dedup = CaptureDedupCache(ttl_seconds=300, max_entries=1)
    stale_hash = dedup.content_hash("stale content")
    dedup.cache[stale_hash] = time.time() - 400
    dedup.cache[dedup.content_hash("fresh content")] = time.time()

    assert dedup.check("new content") is False

    assert stale_hash not in dedup.cache
