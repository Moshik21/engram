from __future__ import annotations

import fnmatch

import pytest

from engram.config import EmbeddingConfig
from engram.models.entity import Entity
from engram.storage.vector.redis_search import RedisSearchIndex


class TinyEmbeddingProvider:
    def dimension(self) -> int:
        return 2

    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed_query(text) for text in texts]


class FakePipeline:
    def __init__(self, redis: FakeRedis) -> None:
        self._redis = redis
        self._ops: list[tuple[str | bytes, str]] = []

    def hget(self, key: str | bytes, field: str) -> None:
        self._ops.append((key, field))

    async def execute(self) -> list[object | None]:
        return [await self._redis.hget(key, field) for key, field in self._ops]


class FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, object]] = {}

    async def hset(self, key: str, mapping: dict[str, object]) -> None:
        self.hashes[key] = dict(mapping)

    async def hget(self, key: str | bytes, field: str) -> object | None:
        key_text = key.decode() if isinstance(key, bytes) else key
        return self.hashes.get(key_text, {}).get(field)

    async def scan_iter(self, match: str, count: int = 100):
        del count
        for key in self.hashes:
            if fnmatch.fnmatch(key, match):
                yield key

    def pipeline(self, transaction: bool = False) -> FakePipeline:
        del transaction
        return FakePipeline(self)


@pytest.mark.asyncio
async def test_redis_embedding_reads_use_group_or_all_groups() -> None:
    redis = FakeRedis()
    index = RedisSearchIndex(
        redis,
        TinyEmbeddingProvider(),
        EmbeddingConfig(dimensions=2),
        storage_dim=2,
    )
    await index.index_entity(
        Entity(
            id="ent-alpha",
            name="alpha",
            entity_type="Test",
            group_id="brain-a",
        )
    )

    scoped_embeddings = await index.get_entity_embeddings(
        ["ent-alpha"],
        group_id="brain-a",
    )
    all_group_embeddings = await index.get_entity_embeddings(
        ["ent-alpha"],
        group_id=None,
    )
    all_group_similarity = await index.compute_similarity(
        "alpha",
        ["ent-alpha"],
        group_id=None,
    )

    assert scoped_embeddings["ent-alpha"] == pytest.approx([1.0, 0.0])
    assert all_group_embeddings["ent-alpha"] == pytest.approx([1.0, 0.0])
    assert all_group_similarity["ent-alpha"] == pytest.approx(1.0)
