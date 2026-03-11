"""Embedding provider abstraction — Voyage AI for production, Noop for dev."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text. May use different input_type."""
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embeddings via raw httpx (already a dependency via anthropic)."""

    API_URL = "https://api.voyageai.com/v1/embeddings"
    CACHE_MAX_SIZE = 256

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4-lite",
        dimensions: int = 1024,
        batch_size: int = 64,
    ) -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=30.0)
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches. Uses input_type='document' for storage."""
        if not texts:
            return []
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result = await self._call_api(batch, input_type="document")
            all_embeddings.extend(result)
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query with LRU caching."""
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        results = await self._call_api([text], input_type="query")
        vec = results[0] if results else []
        if vec:
            self._query_cache[text] = vec
            if len(self._query_cache) > self.CACHE_MAX_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    def clear_cache(self) -> None:
        """Clear the query embedding cache."""
        self._query_cache.clear()

    def dimension(self) -> int:
        return self._dimensions

    async def _call_api(self, texts: list[str], input_type: str) -> list[list[float]]:
        """Make a single API call to Voyage AI."""
        response = await self._client.post(
            self.API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "input": texts,
                "input_type": input_type,
            },
        )
        response.raise_for_status()
        data = response.json()
        # Sort by index to ensure correct order
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()


class FastEmbedProvider(EmbeddingProvider):
    """Local embeddings via fastembed (ONNX runtime). No API key required."""

    CACHE_MAX_SIZE = 256

    def __init__(self, model: str = "nomic-ai/nomic-embed-text-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model)
        self._dimensions: int = self._model.embedding_size  # type: ignore[attr-defined]
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        logger.info(
            "FastEmbedProvider ready: model=%s, dim=%d",
            model,
            self._dimensions,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally via ONNX. Run in thread pool (CPU-bound)."""
        if not texts:
            return []
        import asyncio

        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    async def embed_query(self, text: str) -> list[float]:
        """Embed query with LRU cache."""
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        results = await self.embed([text])
        vec = results[0] if results else []
        if vec:
            self._query_cache[text] = vec
            if len(self._query_cache) > self.CACHE_MAX_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    def dimension(self) -> int:
        return self._dimensions


def truncate_vectors(vectors: list[list[float]], target_dim: int) -> list[list[float]]:
    """Truncate vectors to target dimension (Matryoshka truncation).

    Only safe for models trained with Matryoshka representation learning
    (e.g. Nomic Embed v1.5). Voyage vectors do NOT support this.
    """
    if not vectors or target_dim <= 0 or target_dim >= len(vectors[0]):
        return vectors
    return [v[:target_dim] for v in vectors]


class NoopProvider(EmbeddingProvider):
    """Fallback when no API key is configured. Returns empty lists → disables vector search."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    async def embed_query(self, text: str) -> list[float]:
        return []

    def dimension(self) -> int:
        return 0
