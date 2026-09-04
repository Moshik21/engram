"""Embedding provider abstraction — local FastEmbed in operation, Noop to disable vectors."""

from __future__ import annotations

import logging
import os
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _dotenv_fastembed_cache_path() -> str:
    """``FASTEMBED_CACHE_PATH`` from Engram's own dotenv chain, last file wins.

    2026-09-04: the operator's ``~/.engram/.env`` pins the cache to a
    subdirectory, and the launchd shell/brain export that file as real env
    vars — but ``engram doctor``, ``engram brain run`` from a terminal, and
    every other non-launchd process resolved the parent directory, where the
    configured quantized model does not exist, and embedded nothing. The
    variable is not an ``ENGRAM_*`` setting, so pydantic never read it; this
    reads it from the same files, in the same precedence.
    """
    try:
        from dotenv import dotenv_values

        from engram.config import DEFAULT_ENV_FILES
    except Exception:  # silent-ok: no dotenv support means env/default only
        return ""
    found = ""
    for env_file in DEFAULT_ENV_FILES:
        try:
            values = dotenv_values(Path(env_file).expanduser())
        except Exception:
            continue
        value = (values.get("FASTEMBED_CACHE_PATH") or "").strip()
        if value:
            found = value
    return found


def default_fastembed_cache_dir() -> str:
    """Stable on-disk cache for local ONNX models (never system temp).

    Fastembed defaults to ``$TMPDIR/fastembed_cache``, which macOS can purge and
    which broke dogfood when a half-downloaded ``model.onnx`` was left incomplete.
    Prefer an explicit ``FASTEMBED_CACHE_PATH``, else ``~/.engram/models/fastembed``.
    """
    explicit = os.environ.get("FASTEMBED_CACHE_PATH", "").strip() or _dotenv_fastembed_cache_path()
    if explicit:
        path = Path(explicit).expanduser()
    else:
        path = Path.home() / ".engram" / "models" / "fastembed"
    path.mkdir(parents=True, exist_ok=True)
    # Ensure child TextEmbedding / TextCrossEncoder calls inherit the same root
    # when they only read the env var (cache_dir=None path).
    os.environ.setdefault("FASTEMBED_CACHE_PATH", str(path))
    return str(path)


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


class FastEmbedProvider(EmbeddingProvider):
    """Local embeddings via fastembed (ONNX runtime). No API key required."""

    CACHE_MAX_SIZE = 256
    KNOWN_MODEL_DIMENSIONS = {
        "nomic-ai/nomic-embed-text-v1.5": 768,
        "nomic-ai/nomic-embed-text-v1.5-Q": 768,  # quantized; same dim, smaller download
    }

    # Memory bound for the local ONNX model. nomic-embed has an 8192-token
    # window and attention memory grows with the square of the input, and the
    # ONNX runtime never returns its arena: measured 2026-09-02 on the dogfood
    # brain, ONE 40k-char episode cost 7.4 GB / 15 s, 64 x 4.5k chars at
    # batch 64 cost 11 GB / 76 s, and the startup outbox drain took the shell
    # to a 17 GB footprint on a 16 GB Mac (Jetsam SIGKILL). 2000 chars matches
    # the chunk window (HelixSearchIndex._chunk_text), so long content is still
    # covered by its chunk vectors; at 2000 chars / batch 16 the same 64 texts
    # cost 2.1 GB / 7.7 s.
    DEFAULT_MAX_CHARS = 2000
    DEFAULT_BATCH_SIZE = 16

    def __init__(
        self,
        model: str = "nomic-ai/nomic-embed-text-v1.5",
        dimensions: int | None = None,
        cache_dir: str | None = None,
        max_chars: int = DEFAULT_MAX_CHARS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        import importlib.util

        if importlib.util.find_spec("fastembed") is None:
            raise ImportError("fastembed not installed")

        self._max_chars = max(0, int(max_chars))
        self._batch_size = max(1, int(batch_size))
        self._model_name = model
        self._cache_dir = cache_dir or default_fastembed_cache_dir()
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._dimensions = (
            dimensions
            if dimensions and dimensions > 0
            else self.KNOWN_MODEL_DIMENSIONS.get(model, 0)
        )
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        if self._dimensions <= 0:
            self._ensure_model()
        else:
            logger.info(
                "FastEmbedProvider configured lazy: model=%s, dim=%d, cache=%s",
                model,
                self._dimensions,
                self._cache_dir,
            )

    # Broken-model retry backoff: without it a repaired cache required a
    # process restart ("until model is repaired" was a lie — the latch never
    # cleared).
    BROKEN_RETRY_SECONDS = 300.0

    def _ensure_model(self) -> Any:
        """Load the ONNX model only when embeddings are actually requested.

        On load failure, mark the provider broken for BROKEN_RETRY_SECONDS and
        return None so callers can fall back (BM25) instead of thrashing
        retries; after the backoff a repaired cache heals without a restart.
        """
        if self._broken_now():
            return None
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._broken_now():
                return None
            if self._model is None:
                try:
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(
                        model_name=self._model_name,
                        cache_dir=self._cache_dir,
                    )
                    if self._dimensions <= 0:
                        self._dimensions = self._model.embedding_size  # type: ignore[attr-defined]
                    self._model_broken_until = 0.0
                    logger.info(
                        "FastEmbedProvider ready: model=%s, dim=%d, cache=%s",
                        self._model_name,
                        self._dimensions,
                        self._cache_dir,
                    )
                except Exception as exc:
                    import time as _time

                    self._model_broken_until = _time.monotonic() + self.BROKEN_RETRY_SECONDS
                    logger.error(
                        "FastEmbedProvider model load failed (cache=%s model=%s): %s. "
                        "Vector embeds disabled; retrying in %.0fs.",
                        self._cache_dir,
                        self._model_name,
                        exc,
                        self.BROKEN_RETRY_SECONDS,
                    )
                    return None
        return self._model

    def _broken_now(self) -> bool:
        until = getattr(self, "_model_broken_until", 0.0)
        if not until:
            return False
        import time as _time

        return _time.monotonic() < until

    @property
    def is_materialized(self) -> bool:
        """Whether the underlying ONNX model has been loaded."""
        return self._model is not None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts locally via ONNX. Run in thread pool (CPU-bound)."""
        if not texts:
            return []
        import asyncio

        return await asyncio.to_thread(self._embed_sync, texts)

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        if model is None:
            # Whole-batch failure contract (same as NoopProvider): callers
            # check `if not vecs`. Returning [[] for _ in texts] here poisoned
            # stores with present-but-empty vectors that later reads treated
            # as real (truthy list of falsy vectors).
            return []
        if self._max_chars > 0:
            texts = [t[: self._max_chars] for t in texts]
        return [vec.tolist() for vec in model.embed(texts, batch_size=self._batch_size)]

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
    """Truncate vectors to target dimension (Matryoshka prefix slicing).

    Safe for models trained with Matryoshka representation learning
    (Gemini Embedding 2, Nomic Embed v1.5). Voyage vectors do NOT support this.

    At 3072d stored → slice to 256d for fast approximate comparisons,
    512d for medium, or use full 3072d for maximum quality.
    """
    if not vectors or target_dim <= 0 or target_dim >= len(vectors[0]):
        return vectors
    return [v[:target_dim] for v in vectors]


def prefix_cosine_similarity(a: list[float], b: list[float], prefix_dim: int = 256) -> float:
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
