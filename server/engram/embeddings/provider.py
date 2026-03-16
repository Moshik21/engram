"""Embedding provider abstraction — Voyage AI for production, Noop for dev."""

from __future__ import annotations

import logging
import os
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


class GeminiProvider(EmbeddingProvider):
    """Google Gemini embeddings — multimodal, task-aware, MRL-trained.

    Uses ``gemini-embedding-2-preview`` (8,192 token limit, 3072d max).
    Task types: ``RETRIEVAL_DOCUMENT`` for indexing, ``RETRIEVAL_QUERY``
    for search.  Supports Matryoshka (MRL) dimension reduction via
    ``output_dimensionality``.  L2 normalization applied for reduced dims.

    Multimodal limits: 6 images, 80s audio, 128s video, 6 PDF pages per request.

    Requires ``google-genai`` package:  ``pip install google-genai``
    """

    CACHE_MAX_SIZE = 256

    def __init__(
        self,
        api_key: str = "",
        model: str = "gemini-embedding-2-preview",
        dimensions: int = 3072,
        batch_size: int = 64,
    ) -> None:
        from google import genai

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self._client = genai.Client(api_key=resolved_key) if resolved_key else genai.Client()
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        logger.info(
            "GeminiProvider ready: model=%s, dim=%d",
            model,
            dimensions,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with RETRIEVAL_DOCUMENT task type for indexing."""
        if not texts:
            return []
        import asyncio

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vecs = await asyncio.to_thread(
                self._embed_sync, batch, "RETRIEVAL_DOCUMENT"
            )
            all_embeddings.extend(vecs)
        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a query with RETRIEVAL_QUERY task type and LRU cache."""
        if text in self._query_cache:
            self._query_cache.move_to_end(text)
            return self._query_cache[text]
        import asyncio

        vecs = await asyncio.to_thread(
            self._embed_sync, [text], "RETRIEVAL_QUERY"
        )
        vec = vecs[0] if vecs else []
        if vec:
            self._query_cache[text] = vec
            if len(self._query_cache) > self.CACHE_MAX_SIZE:
                self._query_cache.popitem(last=False)
        return vec

    async def embed_image(self, image_bytes: bytes, mime_type: str = "image/png") -> list[float]:
        """Embed an image (multimodal, requires gemini-embedding-2-preview)."""
        import asyncio

        return await asyncio.to_thread(self._embed_image_sync, image_bytes, mime_type)

    async def embed_multimodal(
        self,
        text: str = "",
        image_bytes: bytes | None = None,
        image_mime: str = "image/png",
    ) -> list[float]:
        """Embed text + image together into a single aggregated vector."""
        import asyncio

        return await asyncio.to_thread(
            self._embed_multimodal_sync, text, image_bytes, image_mime
        )

    # gemini-embedding-2-preview supports 8,192 tokens per input.
    # Approximate as 4 chars/token -> 32,768 chars.
    _MAX_INPUT_CHARS = 32_000  # conservative estimate for 8,192 tokens

    def _embed_sync(self, texts: list[str], task_type: str) -> list[list[float]]:
        """Synchronous batch embed with task type and dimension control.

        Pre-truncates inputs exceeding ~8,192 tokens to stay within the
        Gemini embedding limit.
        """
        from google.genai import types

        # Pre-flight: warn and pre-truncate over-limit inputs
        processed = []
        for t in texts:
            if len(t) > self._MAX_INPUT_CHARS:
                logger.warning(
                    "GeminiProvider: input text is ~%d tokens (%.0f chars), "
                    "exceeding 8,192-token limit. Pre-truncating to %d chars.",
                    len(t) // 4,
                    len(t),
                    self._MAX_INPUT_CHARS,
                )
                processed.append(t[: self._MAX_INPUT_CHARS])
            else:
                processed.append(t)

        result = self._client.models.embed_content(
            model=self._model,
            contents=processed,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self._dimensions,
            ),
        )
        vecs = [list(e.values) for e in result.embeddings]
        # Normalize for non-3072 dimensions (Gemini MRL requirement)
        if self._dimensions != 3072:
            vecs = self._normalize(vecs)
        return vecs

    def _embed_image_sync(self, image_bytes: bytes, mime_type: str) -> list[float]:
        """Synchronous image embedding."""
        from google.genai import types

        result = self._client.models.embed_content(
            model=self._model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            ],
            config=types.EmbedContentConfig(
                output_dimensionality=self._dimensions,
            ),
        )
        vec = list(result.embeddings[0].values) if result.embeddings else []
        if vec and self._dimensions != 3072:
            vec = self._normalize([vec])[0]
        return vec

    def _embed_multimodal_sync(
        self,
        text: str,
        image_bytes: bytes | None,
        image_mime: str,
    ) -> list[float]:
        """Synchronous text+image aggregated embedding."""
        from google.genai import types

        parts = []
        if text:
            parts.append(types.Part(text=text))
        if image_bytes:
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime))
        if not parts:
            return []

        result = self._client.models.embed_content(
            model=self._model,
            contents=[types.Content(parts=parts)],
            config=types.EmbedContentConfig(
                output_dimensionality=self._dimensions,
            ),
        )
        vec = list(result.embeddings[0].values) if result.embeddings else []
        if vec and self._dimensions != 3072:
            vec = self._normalize([vec])[0]
        return vec

    @staticmethod
    def _normalize(vecs: list[list[float]]) -> list[list[float]]:
        """L2-normalize vectors (required for MRL-truncated Gemini embeddings)."""
        import numpy as np

        out = []
        for v in vecs:
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            out.append(arr.tolist())
        return out

    def clear_cache(self) -> None:
        self._query_cache.clear()

    def dimension(self) -> int:
        return self._dimensions


def truncate_vectors(vectors: list[list[float]], target_dim: int) -> list[list[float]]:
    """Truncate vectors to target dimension (Matryoshka prefix slicing).

    Safe for models trained with Matryoshka representation learning
    (Gemini Embedding 2, Nomic Embed v1.5). Voyage vectors do NOT support this.

    At 3072d stored → slice to 256d for fast approximate comparisons,
    512d for medium, or use full 3072d for maximum quality.
    """
    if not vectors or target_dim <= 0 or target_dim >= len(vectors[0]):
        return vectors
    return [v[:target_dim] for v in vectors]


def prefix_cosine_similarity(
    a: list[float], b: list[float], prefix_dim: int = 256
) -> float:
    """Fast approximate cosine similarity using Matryoshka prefix slicing.

    Slices both vectors to ``prefix_dim`` before computing similarity.
    Use for bulk comparisons in consolidation (triage, merge, dream)
    where speed matters more than precision.
    """
    import numpy as np

    dim = min(prefix_dim, len(a), len(b))
    va = np.asarray(a[:dim], dtype=np.float32)
    vb = np.asarray(b[:dim], dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


class NoopProvider(EmbeddingProvider):
    """Fallback when no API key is configured. Returns empty lists → disables vector search."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return []

    async def embed_query(self, text: str) -> list[float]:
        return []

    def dimension(self) -> int:
        return 0
