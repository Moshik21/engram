"""Voyage AI embeddings for BENCHMARKS ONLY.

2026-09-04: Engram never calls an external model or service in operation;
local FastEmbed is the only embedding provider (``engram.embeddings.provider``).
This class exists for the showcase / A-B benchmark lanes that compare against
a hosted embedder, and is reached only when a benchmark asks for
``--vector-provider voyage`` with ``VOYAGE_API_KEY`` set.
"""

from __future__ import annotations

from collections import OrderedDict

from engram.embeddings.provider import EmbeddingProvider


class VoyageProvider(EmbeddingProvider):
    """Voyage AI embeddings via raw httpx (already a dependency via anthropic)."""

    API_URL = "https://api.voyageai.com/v1/embeddings"
    CACHE_MAX_SIZE = 256

    def __init__(
        self,
        api_key: str,
        model: str = "voyage-4-lite",
        dimensions: int = 3072,
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
