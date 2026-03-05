"""Cross-encoder re-ranking provider abstraction."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RerankerProvider(ABC):
    """Abstract re-ranker provider."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        """Re-rank documents by relevance to query.

        Args:
            query: The search query.
            documents: List of (entity_id, text) pairs.
            top_n: Number of results to return.

        Returns:
            List of (entity_id, rerank_score) sorted by score descending.
        """
        ...


class CohereReranker(RerankerProvider):
    """Cohere re-ranker via raw httpx."""

    API_URL = "https://api.cohere.com/v2/rerank"

    def __init__(self, api_key: str, model: str = "rerank-v3.5") -> None:
        import httpx

        self._client = httpx.AsyncClient(timeout=30.0)
        self._api_key = api_key
        self._model = model

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        if not documents:
            return []

        top_n = min(top_n, len(documents))

        try:
            response = await self._client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "query": query,
                    "documents": [text for _, text in documents],
                    "top_n": top_n,
                },
            )
            response.raise_for_status()
            data = response.json()

            results: list[tuple[str, float]] = []
            for item in data.get("results", []):
                idx = item["index"]
                score = item["relevance_score"]
                entity_id = documents[idx][0]
                results.append((entity_id, score))

            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            logger.warning("Cohere rerank failed, returning original order: %s", e)
            return [(eid, 1.0 - i / len(documents)) for i, (eid, _) in enumerate(documents[:top_n])]

    async def close(self) -> None:
        """Close the httpx client."""
        await self._client.aclose()


class FastEmbedReranker(RerankerProvider):
    """Local cross-encoder reranker via fastembed (ONNX runtime)."""

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model)
        logger.info("FastEmbedReranker ready: model=%s", model)

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        if not documents:
            return []

        import asyncio

        top_n = min(top_n, len(documents))
        texts = [text for _, text in documents]

        scores = await asyncio.to_thread(
            lambda: list(self._model.rerank(query, texts))
        )

        paired = [(documents[i][0], scores[i]) for i in range(len(documents))]
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_n]


class NoopReranker(RerankerProvider):
    """Pass-through reranker that preserves original order."""

    async def rerank(
        self,
        query: str,
        documents: list[tuple[str, str]],
        top_n: int = 10,
    ) -> list[tuple[str, float]]:
        top_n = min(top_n, len(documents))
        return [
            (eid, 1.0 - i / max(len(documents), 1)) for i, (eid, _) in enumerate(documents[:top_n])
        ]


def create_reranker(
    api_key: str | None = None,
    model: str = "rerank-v3.5",
    provider: str = "cohere",
    local_model: str = "Xenova/ms-marco-MiniLM-L-6-v2",
) -> RerankerProvider:
    """Factory to create a reranker provider."""
    if provider == "local":
        try:
            return FastEmbedReranker(model=local_model)
        except ImportError:
            logger.warning("fastembed not installed — falling back to NoopReranker")
            return NoopReranker()
    if provider == "cohere" and api_key:
        return CohereReranker(api_key=api_key, model=model)
    return NoopReranker()
