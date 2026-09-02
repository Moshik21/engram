"""Native episode indexing embeds all chunks in ONE provider call.

An 80k-char episode is ~44 chunks of 2000 chars; embedding them one call at a
time was ~15 s per episode in the startup outbox drain (2026-09-02).
"""

from __future__ import annotations

import pytest

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider
from engram.models import Episode
from engram.storage.helix.search import HelixSearchIndex


class _CountingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.embed_calls: list[int] = []

    async def embed(self, texts):
        self.embed_calls.append(len(texts))
        return [[1.0] * 8 for _ in texts]

    async def embed_query(self, text):
        return [1.0] * 8

    def dimension(self) -> int:
        return 8


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def query(self, endpoint, payload):
        self.calls.append(endpoint)
        return []


@pytest.mark.asyncio
async def test_native_chunks_are_embedded_in_one_batch(tmp_path) -> None:
    provider = _CountingProvider()
    client = _Client()
    index = HelixSearchIndex(
        helix_config=HelixDBConfig(transport="native", data_dir=str(tmp_path / "d")),
        provider=provider,
        embed_config=EmbeddingConfig(),
        embed_provider="test",
        embed_model="counting",
        client=client,
        owns_client=False,
        topic_segmentation=False,
    )
    content = ("The recall pipeline burned the wall on durable probes. " * 40 + "\n\n") * 6
    await index.index_episode(Episode(id="ep_1", group_id="default", content=content))
    chunk_inserts = client.calls.count("create_episode_chunk_vec")
    assert chunk_inserts >= 3, f"content did not chunk ({chunk_inserts} inserts)"
    # one call for the episode vector, one batched call for every chunk
    assert provider.embed_calls == [1, chunk_inserts], provider.embed_calls
