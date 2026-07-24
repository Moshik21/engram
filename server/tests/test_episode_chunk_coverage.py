"""Regression: prose episodes must produce chunk vectors (answer locality).

The dogfood autopsy found ZERO chunk rows corpus-wide: index_episode ran
_chunk_by_rounds (empty for non-conversational prose) then _segment_by_topic,
which returns a single-segment list `[whole_text]` for real 2-4 KB documents.
That single element was truthy, so the `if not chunks` gate skipped the
size-based `_chunk_text` last resort, and `if len(chunks) > 1` was False —
no chunk vectors were ever written. Long observations then never surfaced for
their natural single-fact questions. The fix falls through to size-based
chunking whenever the smarter strategies yield <= 1 chunk.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.models.episode import Episode
from engram.storage.helix.search import HelixSearchIndex


class _FakeProvider(EmbeddingProvider):
    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self._dim

    def dimension(self) -> int:
        return self._dim


def _make_index() -> HelixSearchIndex:
    return HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969, transport="native"),
        provider=_FakeProvider(16),
        embed_config=EmbeddingConfig(),
        storage_dim=16,
        embed_provider="fake",
        embed_model="fake-16",
    )


# A ~2.6 KB single-topic prose document with no speaker labels — the exact
# shape that produced zero chunks (a milestone observation / design note).
_PROSE = (
    "The recency and frequency redesign replaces the ACT-R activation ranking "
    "term with a bounded usage tiebreaker. Activation from access history "
    "saturated at a single access and conflated surfaced with used, producing "
    "a rich-get-richer echo loop that collapsed held-out reachability. "
    "The new signal composes a log-saturating frequency factor with an "
    "exponential recency decay and a floor, so behavioral usage can only break "
    "near ties and never bury a semantically stronger result. "
) * 6


@pytest.mark.asyncio
async def test_prose_episode_writes_multiple_chunk_vectors(monkeypatch):
    index = _make_index()
    index._embeddings_enabled = True
    index._helix_client = SimpleNamespace()  # truthy: enter the chunking branch
    index._topic_segmentation = True

    calls: list[str] = []

    async def fake_query(endpoint, payload=None, *a, **k):
        calls.append(endpoint)
        return []

    async def fake_embed_texts(texts):
        return [[0.1] * 16 for _ in texts]

    # The bug trigger: topic segmentation yields a single (truthy) segment.
    async def single_segment(text, **k):
        return [text]

    monkeypatch.setattr(index, "_query", fake_query)
    monkeypatch.setattr(index, "_embed_texts", fake_embed_texts)
    monkeypatch.setattr(index, "_segment_by_topic", single_segment)

    ep = Episode(id="ep_prose", content=_PROSE, group_id="g1")
    assert len(_PROSE) > index.CHUNK_MIN_LENGTH
    await index.index_episode(ep)

    chunk_calls = [c for c in calls if "chunk" in c]
    assert index._embed_stats["chunks_indexed"] > 1
    assert len(chunk_calls) > 1
    assert any(c == "add_episode_vector" for c in calls)  # full-content vector too


@pytest.mark.asyncio
async def test_short_episode_writes_no_chunks(monkeypatch):
    """A genuinely short episode keeps its single full-content vector and no
    chunks — the fix must not over-chunk."""
    index = _make_index()
    index._embeddings_enabled = True
    index._helix_client = SimpleNamespace()
    index._topic_segmentation = True

    calls: list[str] = []

    async def fake_query(endpoint, payload=None, *a, **k):
        calls.append(endpoint)
        return []

    async def fake_embed_texts(texts):
        return [[0.1] * 16 for _ in texts]

    monkeypatch.setattr(index, "_query", fake_query)
    monkeypatch.setattr(index, "_embed_texts", fake_embed_texts)

    ep = Episode(id="ep_short", content="A single short fact about the project.", group_id="g1")
    await index.index_episode(ep)

    assert index._embed_stats["chunks_indexed"] == 0
    assert not [c for c in calls if "chunk" in c]


def test_chunk_search_runs_on_primary_timeout_by_default():
    """RECALL_PERFORMANCE_PLAN M3: chunk search (independent ~30ms HNSW) must
    run even when the ENTITY primary search timed out — the old
    'not primary_search_timed_out' gate left the answer-locality lane inert.
    Pins the config default + the gate expression."""
    from engram.config import ActivationConfig

    cfg = ActivationConfig()
    assert cfg.recall_chunk_search_on_timeout is True

    # The live gate: chunk runs if enabled AND (primary didn't time out OR the
    # on-timeout flag is set). With the flag on, a primary timeout no longer
    # skips chunk search.
    def _chunk_gate(primary_timed_out: bool, on_timeout: bool) -> bool:
        return cfg.chunk_search_enabled and (not primary_timed_out or on_timeout)

    assert _chunk_gate(primary_timed_out=True, on_timeout=True) is True
    assert _chunk_gate(primary_timed_out=True, on_timeout=False) is False
    assert _chunk_gate(primary_timed_out=False, on_timeout=False) is True
