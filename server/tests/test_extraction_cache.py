"""Unit tests for the persistent ExtractionCache (no network).

Verifies the determinism mechanism that makes the depth-tier eval repeatable:
  * a cache HIT on the second extract of the same text reuses the stored verdict
    and does NOT call the inner extractor again,
  * the rehydrated result is byte-identical to the original,
  * errors (PARSE_ERROR / API_ERROR) are NOT cached (so a transient failure can
    be retried rather than freezing a broken corpus),
  * caches are keyed per model + prompt_version (no cross-provider mixing),
  * the cache survives a reopen (persistent across processes).
"""

from __future__ import annotations

import pytest

from engram.extraction.extraction_cache import ExtractionCache, content_hash
from engram.extraction.extractor import ExtractionResult, ExtractionStatus


class _CountingExtractor:
    """Fake inner extractor: deterministic output, counts calls. No network."""

    _model = "fake-haiku-1"

    def __init__(self, result: ExtractionResult) -> None:
        self._result = result
        self.calls = 0

    async def extract(self, text: str, **kwargs) -> ExtractionResult:
        self.calls += 1
        return self._result


class _StrictExtractor:
    """Inner extractor that accepts ONLY (text) -- mirrors the real
    EntityExtractor.extract signature, which takes no episode_id/group_id."""

    _model = "fake-strict-1"

    def __init__(self, result: ExtractionResult) -> None:
        self._result = result
        self.calls = 0

    async def extract(self, text: str) -> ExtractionResult:
        self.calls += 1
        return self._result


def _ok_result() -> ExtractionResult:
    return ExtractionResult(
        entities=[{"name": "Dana", "type": "Person"}],
        relationships=[{"source": "Dana", "predicate": "works_on", "target": "Atlas"}],
        status=ExtractionStatus.OK,
    )


@pytest.mark.asyncio
async def test_second_extract_is_cache_hit_no_inner_call(tmp_path):
    inner = _CountingExtractor(_ok_result())
    cache = ExtractionCache(inner, tmp_path / "c.sqlite")

    r1 = await cache.extract("Dana works on Atlas")
    assert inner.calls == 1
    assert cache.misses == 1 and cache.hits == 0

    # Same text -> HIT, inner NOT called again.
    r2 = await cache.extract("Dana works on Atlas")
    assert inner.calls == 1  # unchanged
    assert cache.hits == 1
    assert cache.hit_rate == pytest.approx(0.5)

    # Rehydrated verdict is byte-identical in content.
    assert r2.entities == r1.entities
    assert r2.relationships == r1.relationships
    assert r2.status == r1.status
    cache.close()


@pytest.mark.asyncio
async def test_errors_are_not_cached(tmp_path):
    err = ExtractionResult(
        entities=[],
        relationships=[],
        status=ExtractionStatus.PARSE_ERROR,
        error="boom",
    )
    inner = _CountingExtractor(err)
    cache = ExtractionCache(inner, tmp_path / "c.sqlite")

    await cache.extract("flaky text")
    await cache.extract("flaky text")
    # No cache hit because errors are never stored -> inner called twice.
    assert inner.calls == 2
    assert cache.hits == 0
    assert cache.stores == 0
    cache.close()


@pytest.mark.asyncio
async def test_cache_persists_across_reopen(tmp_path):
    db = tmp_path / "persist.sqlite"
    inner1 = _CountingExtractor(_ok_result())
    cache1 = ExtractionCache(inner1, db)
    await cache1.extract("persist me")
    assert inner1.calls == 1
    cache1.close()

    # Reopen with a fresh inner extractor: the warm cache must serve the hit so
    # the second process makes ZERO inner calls (proves cross-run determinism).
    inner2 = _CountingExtractor(_ok_result())
    cache2 = ExtractionCache(inner2, db)
    await cache2.extract("persist me")
    assert inner2.calls == 0
    assert cache2.hits == 1
    cache2.close()


@pytest.mark.asyncio
async def test_cache_keyed_per_model(tmp_path):
    db = tmp_path / "model.sqlite"
    inner = _CountingExtractor(_ok_result())

    cache_a = ExtractionCache(inner, db, extractor_model="model-a")
    await cache_a.extract("same text")
    assert inner.calls == 1
    cache_a.close()

    # A different model identifier must NOT reuse model-a's verdict.
    cache_b = ExtractionCache(inner, db, extractor_model="model-b")
    await cache_b.extract("same text")
    assert inner.calls == 2  # miss for model-b
    assert cache_b.misses == 1
    cache_b.close()


@pytest.mark.asyncio
async def test_cache_keyed_per_prompt_version(tmp_path):
    db = tmp_path / "ver.sqlite"
    inner = _CountingExtractor(_ok_result())

    cache_a = ExtractionCache(inner, db, prompt_version="v1")
    await cache_a.extract("text")
    cache_a.close()

    cache_b = ExtractionCache(inner, db, prompt_version="v2")
    await cache_b.extract("text")
    assert inner.calls == 2  # different prompt version => miss
    cache_b.close()


def test_content_hash_is_stable():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


@pytest.mark.asyncio
async def test_strict_inner_signature_does_not_get_extra_kwargs(tmp_path):
    """Regression: the projector introspects the cache's (**kwargs) signature and
    passes episode_id/group_id. The cache must forward only what the inner
    extractor accepts, or a real EntityExtractor.extract(self, text) raises
    TypeError: unexpected keyword argument 'episode_id'."""
    inner = _StrictExtractor(_ok_result())
    cache = ExtractionCache(inner, tmp_path / "c.sqlite")

    # Projector-style call with extra kwargs the inner does NOT accept.
    r = await cache.extract("Dana works on Atlas", episode_id="ep1", group_id="g1")

    assert inner.calls == 1
    assert r.status == ExtractionStatus.OK
    # And a cache HIT still works with the same extra kwargs.
    await cache.extract("Dana works on Atlas", episode_id="ep1", group_id="g1")
    assert inner.calls == 1
    assert cache.hits == 1
    cache.close()
