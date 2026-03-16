"""HelixDB search index — hybrid vector + BM25 search with RRF fusion.

Implements the SearchIndex protocol using HelixDB's native vector search
(SearchV) and BM25 text search (SearchBM25) endpoints, fused via Reciprocal
Rank Fusion in Python.

When a group_id is provided, filtered vector search queries
(``search_entity_vectors_filtered``, etc.) push group filtering into HelixDB,
eliminating the need for Python-side overfetch + post-hoc filtering on the
vector path.  BM25 results still require post-hoc group filtering.
"""

from __future__ import annotations

import asyncio
import logging
from functools import partial
from typing import Any

import numpy as np

from engram.config import EmbeddingConfig, HelixDBConfig
from engram.embeddings.provider import EmbeddingProvider, truncate_vectors
from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue
from engram.utils.attachments import get_first_image_attachment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RRF_K = 60  # RRF smoothing constant (standard default)
_OVERFETCH_FACTOR = 3  # overfetch to compensate for post-hoc group filtering


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _rrf_fusion(
    fts_results: list[tuple[str, float]],
    vec_results: list[tuple[str, float]],
    fts_weight: float,
    vec_weight: float,
    k: int = _RRF_K,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)).

    Returns (id, score) pairs sorted descending, normalized to 0-1.
    """
    scores: dict[str, float] = {}
    for rank, (item_id, _) in enumerate(fts_results):
        scores[item_id] = scores.get(item_id, 0.0) + fts_weight / (k + rank + 1)
    for rank, (item_id, _) in enumerate(vec_results):
        scores[item_id] = scores.get(item_id, 0.0) + vec_weight / (k + rank + 1)

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Normalize to 0-1
    if merged:
        max_score = merged[0][1]
        if max_score > 0:
            merged = [(eid, score / max_score) for eid, score in merged]

    return merged


# ---------------------------------------------------------------------------
# HelixSearchIndex
# ---------------------------------------------------------------------------


class HelixSearchIndex:
    """Hybrid search index backed by HelixDB vector + BM25 search.

    Implements the SearchIndex protocol for the Helix storage backend.
    Combines vector similarity search (SearchV endpoints) with BM25 text
    search (SearchBM25 endpoints) using Reciprocal Rank Fusion.
    """

    # Chunking configuration
    CHUNK_MIN_LENGTH = 200  # Don't chunk content shorter than this
    CHUNK_MAX_TOKENS = 512  # Target tokens per chunk
    CHUNK_OVERLAP = 50  # Overlap tokens between chunks

    # Retry configuration for embedding API calls
    _EMBED_MAX_RETRIES = 3
    _EMBED_BASE_DELAY = 1.0  # seconds
    _EMBED_MAX_DELAY = 30.0  # seconds

    def __init__(
        self,
        helix_config: HelixDBConfig,
        provider: EmbeddingProvider,
        embed_config: EmbeddingConfig,
        storage_dim: int = 0,
        embed_provider: str = "auto",
        embed_model: str = "noop",
        client=None,
        topic_segmentation: bool = True,
        topic_threshold: float = 0.5,
    ) -> None:
        self._helix_config = helix_config
        self._provider = provider
        self._embed_config = embed_config
        self._embeddings_enabled = provider.dimension() > 0
        self._storage_dim = storage_dim
        self._embed_provider = embed_provider
        self._embed_model = embed_model
        self._fts_weight = embed_config.fts_weight
        self._vec_weight = embed_config.vec_weight
        self._topic_segmentation = topic_segmentation
        self._topic_threshold = topic_threshold
        self._client: Any = None  # helix.Client, created lazily in initialize()
        self._helix_client = client  # Shared HelixClient (async httpx)

        # Last query vector — reused by RelevanceScorer to avoid re-embedding
        self._last_query_vec: list[float] | None = None

        # Fallback embedding provider (lazily initialized on primary failure)
        self._fallback_provider: EmbeddingProvider | None = None
        self._primary_consecutive_failures: int = 0
        self._PRIMARY_FAILURE_THRESHOLD = 3  # switch to fallback after N failures

        # Embedding statistics for diagnostics
        self._embed_stats = {
            "episodes_indexed": 0,
            "episodes_failed": 0,
            "chunks_indexed": 0,
            "chunks_failed": 0,
            "entities_indexed": 0,
            "entities_failed": 0,
            "retries": 0,
            "fallback_used": 0,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the Helix client connection.

        The helix Client constructor is synchronous and attempts a socket
        connection, so we run it in a thread to avoid blocking the event loop.
        """
        if self._helix_client is None:
            from engram.storage.helix.client import HelixClient

            self._helix_client = HelixClient(self._helix_config)
            await self._helix_client.initialize()

        if self._client is not None:
            return
        # Skip legacy HTTP client for native transport
        transport = getattr(self._helix_config, "transport", "http")
        if transport == "native":
            logger.info(
                "HelixSearchIndex: native transport (embeddings=%s)",
                self._embeddings_enabled,
            )
            return
        try:
            from helix import Client

            is_local = not self._helix_config.api_endpoint
            self._client = await asyncio.to_thread(
                partial(
                    Client,
                    local=is_local,
                    port=self._helix_config.port,
                    api_endpoint=self._helix_config.api_endpoint or "",
                    api_key=self._helix_config.api_key or None,
                    verbose=self._helix_config.verbose,
                    max_workers=self._helix_config.max_workers,
                )
            )
            logger.info(
                "HelixSearchIndex: connected (local=%s, port=%d, embeddings=%s)",
                is_local,
                self._helix_config.port,
                self._embeddings_enabled,
            )
        except Exception as e:
            logger.error("HelixSearchIndex: failed to initialize client: %s", e)
            raise

    async def close(self) -> None:
        """Log final embedding stats on shutdown."""
        stats = self._embed_stats
        total = stats["episodes_indexed"] + stats["episodes_failed"]
        if total > 0:
            logger.info(
                "HelixSearchIndex closing — embedding stats: "
                "episodes=%d/%d indexed (%.1f%%), "
                "chunks=%d indexed/%d failed, "
                "entities=%d indexed/%d failed, "
                "retries=%d, fallback_uses=%d",
                stats["episodes_indexed"],
                total,
                100.0 * stats["episodes_indexed"] / total if total else 0,
                stats["chunks_indexed"],
                stats["chunks_failed"],
                stats["entities_indexed"],
                stats["entities_failed"],
                stats["retries"],
                stats["fallback_used"],
            )

    @property
    def embed_stats(self) -> dict[str, int]:
        """Return embedding statistics for external diagnostics."""
        return dict(self._embed_stats)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Return the client, raising if not initialized."""
        if self._client is None:
            raise RuntimeError("HelixSearchIndex not initialized — call initialize() first")
        return self._client

    async def _query(self, endpoint: str, payload: dict) -> list[dict]:
        """Execute a Helix query.

        Fast path: shared async HelixClient (httpx connection pool).
        Legacy fallback: synchronous helix-py SDK via thread pool.

        Returns a list of response dicts. On error returns empty list.
        Helix v2 wraps results in ``{"node": {...}}`` envelopes — unwrap them.
        """
        # Fast path: shared async client
        if self._helix_client is not None:
            return await self._helix_client.query(endpoint, payload)

        # Legacy fallback: synchronous helix-py SDK
        client = self._ensure_client()
        try:
            results = await asyncio.to_thread(client.query, endpoint, payload)
            if not results or not isinstance(results, list):
                return []

            from engram.storage.helix import unwrap_helix_results

            return unwrap_helix_results(results)
        except Exception as e:
            logger.warning("HelixSearchIndex._query(%s) failed: %s", endpoint, e)
            return []

    def _get_fallback_provider(self) -> EmbeddingProvider | None:
        """Lazily create a local FastEmbed fallback provider.

        Called when the primary provider (e.g. Gemini API) fails repeatedly.
        Returns None if FastEmbed is not installed.
        """
        if self._fallback_provider is not None:
            return self._fallback_provider
        try:
            from engram.embeddings.provider import FastEmbedProvider

            self._fallback_provider = FastEmbedProvider()
            logger.warning(
                "HelixSearchIndex: activated FastEmbed fallback provider "
                "(dim=%d) after %d primary failures",
                self._fallback_provider.dimension(),
                self._primary_consecutive_failures,
            )
            return self._fallback_provider
        except ImportError:
            logger.warning(
                "HelixSearchIndex: FastEmbed not installed, no fallback available. "
                "Install with: pip install fastembed"
            )
            return None
        except Exception as e:
            logger.warning("HelixSearchIndex: fallback provider init failed: %s", e)
            return None

    async def _embed_with_retry(
        self,
        provider: EmbeddingProvider,
        texts: list[str],
        *,
        is_query: bool = False,
    ) -> list[list[float]]:
        """Embed texts with exponential backoff retry.

        Retries on transient errors (connection resets, rate limits, timeouts).
        Returns empty list only after all retries are exhausted.
        """
        import random

        last_error: Exception | None = None
        for attempt in range(self._EMBED_MAX_RETRIES):
            try:
                if is_query and len(texts) == 1:
                    vec = await provider.embed_query(texts[0])
                    return [vec] if vec else []
                else:
                    vecs = await provider.embed(texts)
                    return vecs if vecs else []
            except Exception as e:
                last_error = e
                self._embed_stats["retries"] += 1

                # Check if error is retryable
                err_str = str(e).lower()
                retryable = any(
                    kw in err_str
                    for kw in (
                        "connection reset",
                        "connection refused",
                        "timeout",
                        "rate limit",
                        "429",
                        "resource_exhausted",
                        "too many requests",
                        "server error",
                        "500",
                        "502",
                        "503",
                        "504",
                        "errno 54",
                        "broken pipe",
                    )
                )
                if not retryable:
                    # Non-retryable error (bad input, auth failure, etc.)
                    logger.warning(
                        "HelixSearchIndex: non-retryable embedding error: %s", e
                    )
                    break

                delay = min(
                    self._EMBED_BASE_DELAY * (2 ** attempt) + random.uniform(0, 1),
                    self._EMBED_MAX_DELAY,
                )
                logger.warning(
                    "HelixSearchIndex: embedding attempt %d/%d failed: %s. "
                    "Retrying in %.1fs",
                    attempt + 1,
                    self._EMBED_MAX_RETRIES,
                    e,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        logger.error(
            "HelixSearchIndex: embedding failed after %d attempts: %s",
            self._EMBED_MAX_RETRIES,
            last_error,
        )
        return []

    async def _embed_text(self, text: str) -> list[float]:
        """Embed a single query string with retry and fallback.

        Applies storage dimension truncation. Falls back to local FastEmbed
        if the primary provider fails repeatedly.
        """
        if not self._embeddings_enabled:
            return []

        # Try primary provider with retry (no fallback — dimension mismatch risk)
        vecs = await self._embed_with_retry(self._provider, [text], is_query=True)

        if not vecs:
            logger.warning("Failed to embed query after retries: %s", text[:80])
            return []
        vec = vecs[0]
        if vec and self._storage_dim > 0:
            vec = truncate_vectors([vec], self._storage_dim)[0]
        return vec

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed texts with retry and fallback.

        Applies storage dimension truncation. Falls back to local FastEmbed
        if the primary provider fails repeatedly.
        """
        if not self._embeddings_enabled or not texts:
            return []

        # Try primary provider with retry (no fallback — dimension mismatch risk)
        vecs = await self._embed_with_retry(self._provider, texts)

        if not vecs:
            logger.warning("Failed to embed %d texts after retries", len(texts))
            return []
        if vecs and self._storage_dim > 0:
            vecs = truncate_vectors(vecs, self._storage_dim)
        return vecs

    @staticmethod
    def _extract_id_field(row: dict, field: str) -> str:
        """Safely extract an ID field from a Helix result row."""
        return str(row.get(field, ""))

    @staticmethod
    def _extract_score(row: dict) -> float:
        """Extract and normalize a score from a Helix result row.

        Helix vector search returns distance (lower = closer).
        We convert to similarity (higher = better) in the range [0, 1].
        BM25 search returns a relevance score (higher = better).
        """
        # Try vector distance first
        if "distance" in row:
            dist = float(row["distance"])
            # Cosine distance -> similarity: sim = 1 - dist/2 for cosine,
            # or sim = 1 / (1 + dist) for L2. Assume cosine by default.
            return max(0.0, 1.0 - dist / 2.0)
        # Try BM25 score
        if "score" in row:
            return float(row["score"])
        return 0.0

    @staticmethod
    def _chunk_by_rounds(text: str, min_chars: int = 50) -> list[str]:
        """Split conversation text into individual user-assistant rounds.

        Each round = one user message + one assistant response.
        This matches Naive RAG's chunking granularity, producing ~5-6 chunks
        per typical conversation session instead of 1 whole-session embedding.

        Returns an empty list if the text has no speaker labels (i.e. is not
        conversational), so callers can fall back to other strategies.
        """
        import re

        # Split at speaker-label boundaries (User:/Assistant:/Human:/AI:)
        turn_pattern = r"\n(?=(?:User|Assistant|Human|AI)\s*:)"
        turns = re.split(turn_pattern, text)

        # If we didn't find speaker labels, return empty so caller falls back
        if len(turns) <= 1:
            return []

        # Group into rounds: a round is complete once we see both a user turn
        # and an assistant turn.  We accumulate turns until we have a pair,
        # then start the next round.
        rounds: list[str] = []
        current_round = ""
        for turn in turns:
            turn = turn.strip()
            if not turn:
                continue
            current_round = (current_round + "\n" + turn).strip()
            # A round is complete when it contains both user and assistant
            has_user = bool(
                re.search(r"(?:User|Human)\s*:", current_round)
            )
            has_assistant = bool(
                re.search(r"(?:Assistant|AI)\s*:", current_round)
            )
            if has_user and has_assistant:
                if len(current_round) >= min_chars:
                    rounds.append(current_round)
                current_round = ""

        # Don't lose the last incomplete round (e.g. trailing user message)
        if current_round.strip() and len(current_round) >= min_chars:
            rounds.append(current_round.strip())

        return rounds

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2000, overlap_chars: int = 200) -> list[str]:
        """Split text into overlapping chunks at sentence boundaries."""
        if len(text) <= max_chars:
            return [text]

        import re

        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > max_chars and current:
                chunks.append(current.strip())
                # Overlap: keep last portion
                words = current.split()
                overlap_words = words[-min(len(words), overlap_chars // 5) :]
                current = " ".join(overlap_words) + " " + sent
            else:
                current = (current + " " + sent).strip()
        if current.strip():
            chunks.append(current.strip())
        return chunks

    async def _segment_by_topic(
        self,
        text: str,
        max_segment_chars: int = 3000,
        min_segment_chars: int = 100,
        similarity_threshold: float = 0.5,
    ) -> list[str]:
        """Split text into segments at topic boundaries using embedding similarity.

        Uses sliding-window cosine similarity between consecutive sentence groups
        to detect topic shifts. Also splits at speaker transitions (User:/Assistant:).

        Falls back to simple sentence-boundary splitting if embedding fails.
        """
        import re

        if len(text) <= min_segment_chars:
            return [text]

        # Step 1: Split at speaker transitions first
        # This handles conversation transcripts with User:/Assistant: patterns
        speaker_pattern = r"\n(?=(?:User|Assistant|Human|AI|System)\s*:)"
        turns = re.split(speaker_pattern, text)

        # If we got meaningful speaker splits, use those as base units
        if len(turns) > 1:
            # Group consecutive short turns and split long turns
            segments: list[str] = []
            current = ""
            for turn in turns:
                turn = turn.strip()
                if not turn:
                    continue
                if len(current) + len(turn) < min_segment_chars:
                    current = (current + "\n" + turn).strip()
                elif len(current) + len(turn) > max_segment_chars and current:
                    segments.append(current.strip())
                    current = turn
                else:
                    if current:
                        segments.append(current.strip())
                    current = turn
            if current.strip():
                segments.append(current.strip())

            if len(segments) > 1:
                return [s for s in segments if len(s) >= min_segment_chars]

        # Step 2: Sentence-level splitting with topic detection
        # Better sentence splitter that handles abbreviations
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(sentence_pattern, text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

        if len(sentences) <= 3:
            return [text]

        # Step 3: Try embedding-based topic detection
        try:
            if self._embeddings_enabled:
                # Create sentence windows (3 sentences each)
                window_size = 3
                windows: list[str] = []
                for i in range(0, len(sentences) - window_size + 1):
                    window_text = " ".join(sentences[i : i + window_size])
                    windows.append(window_text)

                if len(windows) >= 2:
                    # Batch-embed all windows using the async provider
                    embeddings = await self._embed_texts(windows)

                    # If embeddings failed, fall through to size-based
                    if not embeddings or len(embeddings) != len(windows):
                        raise ValueError("Embedding batch returned incomplete results")

                    # Compute consecutive similarities to detect topic boundaries
                    boundaries = [0]  # Always start at sentence 0
                    for i in range(len(embeddings) - 1):
                        a = np.asarray(embeddings[i], dtype=np.float32)
                        b = np.asarray(embeddings[i + 1], dtype=np.float32)
                        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
                        if na > 0 and nb > 0:
                            sim = float(np.dot(a, b) / (na * nb))
                            if sim < similarity_threshold:
                                # Topic boundary at sentence index i + window_size
                                boundaries.append(i + window_size)

                    boundaries.append(len(sentences))

                    # Build segments from boundaries
                    topic_segments: list[str] = []
                    for i in range(len(boundaries) - 1):
                        start = boundaries[i]
                        end = boundaries[i + 1]
                        segment = " ".join(sentences[start:end])
                        if len(segment) >= min_segment_chars:
                            topic_segments.append(segment)
                        elif topic_segments:
                            # Merge tiny segments with previous
                            topic_segments[-1] += " " + segment
                        else:
                            topic_segments.append(segment)

                    if topic_segments:
                        return topic_segments
        except Exception:
            pass  # Fall through to size-based

        # Step 4: Fallback — size-based splitting at sentence boundaries
        segments = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > max_segment_chars and current:
                segments.append(current.strip())
                # Small overlap
                words = current.split()
                overlap = " ".join(words[-min(len(words), 40) :])
                current = overlap + " " + sent
            else:
                current = (current + " " + sent).strip()
        if current.strip():
            segments.append(current.strip())

        return [s for s in segments if len(s) >= min_segment_chars] or [text]

    def _filter_by_group(
        self,
        results: list[tuple[str, float]],
        raw_rows: list[dict],
        group_id: str | None,
        group_field: str = "group_id",
    ) -> list[tuple[str, float]]:
        """Post-filter results by group_id.

        Builds a set of IDs whose group matches, then filters the scored list.
        """
        if not group_id:
            return results
        # Build set of IDs that match the group
        allowed_ids: set[str] = set()
        for row in raw_rows:
            if str(row.get(group_field, "")) == group_id:
                # Collect all ID fields that might be present
                for field in ("entity_id", "episode_id", "id"):
                    val = row.get(field)
                    if val:
                        allowed_ids.add(str(val))
        return [(item_id, score) for item_id, score in results if item_id in allowed_ids]

    def _filter_by_entity_type(
        self,
        results: list[tuple[str, float]],
        raw_rows: list[dict],
        entity_types: list[str] | None,
    ) -> list[tuple[str, float]]:
        """Post-filter results by entity_type."""
        if not entity_types:
            return results
        type_set = set(entity_types)
        allowed_ids: set[str] = set()
        for row in raw_rows:
            if str(row.get("entity_type", "")) in type_set:
                eid = row.get("entity_id") or row.get("id")
                if eid:
                    allowed_ids.add(str(eid))
        return [(item_id, score) for item_id, score in results if item_id in allowed_ids]

    # ------------------------------------------------------------------
    # Vector search helpers
    # ------------------------------------------------------------------

    async def _vector_search_entities(
        self,
        query_vec: list[float],
        limit: int,
        group_id: str | None = None,
    ) -> tuple[list[tuple[str, float]], list[dict], bool]:
        """Search EntityVec index. Returns (scored_results, raw_rows, was_filtered).

        When *group_id* is provided the optimized ``search_entity_vectors_filtered``
        query pushes group filtering into HelixDB so no Python-side overfetch
        or post-hoc filtering is needed.  Falls back to the unfiltered query
        when group_id is ``None`` or if the filtered endpoint is unavailable.
        """
        filtered = False
        rows: list[dict] = []

        if group_id:
            try:
                rows = await self._query(
                    "search_entity_vectors_filtered",
                    {"vec": query_vec, "k": limit, "gid": group_id},
                )
                filtered = True
            except Exception:
                logger.debug(
                    "search_entity_vectors_filtered unavailable, falling back to unfiltered"
                )
                rows = []

        if not filtered:
            rows = await self._query(
                "search_entity_vectors",
                {"vec": query_vec, "k": limit},
            )

        results: list[tuple[str, float]] = []
        for row in rows:
            eid = self._extract_id_field(row, "entity_id")
            if eid:
                results.append((eid, self._extract_score(row)))
        return results, rows, filtered

    async def _vector_search_episodes(
        self,
        query_vec: list[float],
        limit: int,
        group_id: str | None = None,
    ) -> tuple[list[tuple[str, float]], list[dict], bool]:
        """Search EpisodeVec index. Returns (scored_results, raw_rows, was_filtered).

        Uses ``search_episode_vectors_filtered`` when *group_id* is provided.
        """
        filtered = False
        rows: list[dict] = []

        if group_id:
            try:
                rows = await self._query(
                    "search_episode_vectors_filtered",
                    {"vec": query_vec, "k": limit, "gid": group_id},
                )
                filtered = True
            except Exception:
                logger.debug(
                    "search_episode_vectors_filtered unavailable, falling back to unfiltered"
                )
                rows = []

        if not filtered:
            rows = await self._query(
                "search_episode_vectors",
                {"vec": query_vec, "k": limit},
            )

        results: list[tuple[str, float]] = []
        for row in rows:
            eid = self._extract_id_field(row, "episode_id")
            if eid:
                results.append((eid, self._extract_score(row)))
        return results, rows, filtered

    async def _vector_search_cues(
        self,
        query_vec: list[float],
        limit: int,
        group_id: str | None = None,
    ) -> tuple[list[tuple[str, float]], list[dict], bool]:
        """Search CueVec index. Returns (scored_results, raw_rows, was_filtered).

        Uses ``search_cue_vectors_filtered`` when *group_id* is provided.
        """
        filtered = False
        rows: list[dict] = []

        if group_id:
            try:
                rows = await self._query(
                    "search_cue_vectors_filtered",
                    {"vec": query_vec, "k": limit, "gid": group_id},
                )
                filtered = True
            except Exception:
                logger.debug(
                    "search_cue_vectors_filtered unavailable, falling back to unfiltered"
                )
                rows = []

        if not filtered:
            rows = await self._query(
                "search_cue_vectors",
                {"vec": query_vec, "k": limit},
            )

        results: list[tuple[str, float]] = []
        for row in rows:
            eid = self._extract_id_field(row, "episode_id")
            if eid:
                results.append((eid, self._extract_score(row)))
        return results, rows, filtered

    # ------------------------------------------------------------------
    # BM25 search helpers
    # ------------------------------------------------------------------

    async def _bm25_search_entities(
        self,
        query: str,
        limit: int,
    ) -> tuple[list[tuple[str, float]], list[dict]]:
        """BM25 text search over Entity nodes. Returns (scored_results, raw_rows)."""
        rows = await self._query(
            "search_entities_bm25",
            {"query": query, "k": limit},
        )
        results: list[tuple[str, float]] = []
        for row in rows:
            eid = row.get("entity_id") or row.get("id", "")
            if eid:
                results.append((str(eid), self._extract_score(row)))
        return results, rows

    async def _bm25_search_episodes(
        self,
        query: str,
        limit: int,
    ) -> tuple[list[tuple[str, float]], list[dict]]:
        """BM25 text search over Episode nodes."""
        rows = await self._query(
            "search_episodes_bm25",
            {"query": query, "k": limit},
        )
        results: list[tuple[str, float]] = []
        for row in rows:
            eid = row.get("episode_id") or row.get("id", "")
            if eid:
                results.append((str(eid), self._extract_score(row)))
        return results, rows

    async def _bm25_search_cues(
        self,
        query: str,
        limit: int,
    ) -> tuple[list[tuple[str, float]], list[dict]]:
        """BM25 text search over EpisodeCue nodes."""
        rows = await self._query(
            "search_cues_bm25",
            {"query": query, "k": limit},
        )
        results: list[tuple[str, float]] = []
        for row in rows:
            eid = row.get("episode_id") or row.get("id", "")
            if eid:
                results.append((str(eid), self._extract_score(row)))
        return results, rows

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    async def index_entity(self, entity: Entity) -> None:
        """Embed and index an entity into the EntityVec vector store."""
        if not self._embeddings_enabled or not entity.name:
            return

        text = entity.name
        if entity.summary:
            text = f"{entity.name}: {entity.summary}"

        try:
            embeddings = await self._embed_texts([text])
            if not embeddings:
                self._embed_stats["entities_failed"] += 1
                logger.warning(
                    "Failed to embed entity %s (%s): embedding returned empty",
                    entity.id,
                    entity.name[:50],
                )
                return

            await self._query(
                "add_entity_vector",
                {
                    "entity_id": entity.id,
                    "group_id": entity.group_id,
                    "content_type": "entity",
                    "embed_provider": self._embed_provider,
                    "embed_model": self._embed_model,
                    "vec": embeddings[0],
                },
            )
            self._embed_stats["entities_indexed"] += 1
        except Exception as e:
            self._embed_stats["entities_failed"] += 1
            logger.warning("Failed to index entity %s: %s", entity.id, e)

    async def index_episode(self, episode: Episode) -> None:
        """Embed and index an episode into the EpisodeVec vector store.

        When the episode has image attachments and the provider supports
        multimodal embedding (``embed_multimodal``), the first image is
        embedded together with the text for a richer representation.

        Uses retry + fallback logic via ``_embed_texts()``. Tracks embedding
        statistics in ``_embed_stats`` for diagnostics.
        """
        if not self._embeddings_enabled or not episode.content:
            if not self._embeddings_enabled:
                logger.debug(
                    "index_episode(%s): skipped — embeddings disabled",
                    episode.id,
                )
            return

        try:
            embedding: list[float] | None = None

            # Try multimodal embedding for episodes with image attachments
            if episode.attachments and hasattr(self._provider, "embed_multimodal"):
                image_data = get_first_image_attachment(episode.attachments)
                if image_data is not None:
                    image_bytes, image_mime = image_data
                    try:
                        embedding = await self._provider.embed_multimodal(
                            text=episode.content,
                            image_bytes=image_bytes,
                            image_mime=image_mime,
                        )
                        if embedding and self._storage_dim > 0:
                            embedding = truncate_vectors([embedding], self._storage_dim)[0]
                    except Exception as mm_err:
                        logger.warning(
                            "Multimodal embedding failed for episode %s, "
                            "falling back to text-only: %s",
                            episode.id,
                            mm_err,
                        )

            # Fall back to text-only embedding (with retry + fallback)
            if not embedding:
                embeddings = await self._embed_texts([episode.content])
                if not embeddings:
                    self._embed_stats["episodes_failed"] += 1
                    logger.error(
                        "EMBEDDING FAILED for episode %s "
                        "(content_len=%d, group=%s): "
                        "all providers exhausted after retries. "
                        "Episode will be INVISIBLE to vector search. "
                        "[stats: indexed=%d, failed=%d]",
                        episode.id,
                        len(episode.content),
                        episode.group_id,
                        self._embed_stats["episodes_indexed"],
                        self._embed_stats["episodes_failed"],
                    )
                    return
                embedding = embeddings[0]

            await self._query(
                "add_episode_vector",
                {
                    "episode_id": episode.id,
                    "group_id": episode.group_id,
                    "content_type": "episode",
                    "vec": embedding,
                },
            )
            self._embed_stats["episodes_indexed"] += 1

            if self._embed_stats["episodes_indexed"] % 100 == 0:
                logger.info(
                    "Embedding progress: %d episodes indexed, %d failed, "
                    "%d chunks indexed, %d retries, %d fallback uses",
                    self._embed_stats["episodes_indexed"],
                    self._embed_stats["episodes_failed"],
                    self._embed_stats["chunks_indexed"],
                    self._embed_stats["retries"],
                    self._embed_stats["fallback_used"],
                )

            # Chunk long episodes for granular search.
            # Strategy priority:
            #   1. Round-level chunking (conversational content with speaker
            #      labels) — matches Naive RAG granularity (~5-6 chunks/session)
            #   2. Topic segmentation (non-conversational long text)
            #   3. Size-based chunking (last resort)
            if (
                self._helix_client is not None
                and len(episode.content) > self.CHUNK_MIN_LENGTH
            ):
                chunks: list[str] = []

                # 1. Try round-level chunking first (conversational content)
                chunks = self._chunk_by_rounds(episode.content)

                # 2. Fall back to topic segmentation for non-conversational text
                if not chunks and self._topic_segmentation:
                    try:
                        chunks = await self._segment_by_topic(
                            episode.content,
                            similarity_threshold=self._topic_threshold,
                        )
                    except Exception:
                        chunks = []

                # 3. Last resort: size-based chunking
                if not chunks:
                    chunks = self._chunk_text(episode.content)
                if len(chunks) > 1:  # Only chunk if content actually splits
                    # Check if native transport (server-side Embed() won't work)
                    is_native = getattr(
                        self._helix_config, "transport", "http"
                    ) == "native"
                    for i, chunk_text in enumerate(chunks):
                        used_server_embed = False
                        if not is_native:
                            try:
                                await self._query(
                                    "create_episode_chunk_embed",
                                    {
                                        "episode_id": episode.id,
                                        "group_id": episode.group_id,
                                        "chunk_text": chunk_text,
                                        "chunk_index": i,
                                        "content_type": "episode_chunk",
                                    },
                                )
                                used_server_embed = True
                            except Exception:
                                pass
                        if not used_server_embed:
                            chunk_vecs = await self._embed_texts([chunk_text])
                            if chunk_vecs:
                                await self._query(
                                    "create_episode_chunk_vec",
                                    {
                                        "episode_id": episode.id,
                                        "group_id": episode.group_id,
                                        "chunk_text": chunk_text,
                                        "chunk_index": i,
                                        "content_type": "episode_chunk",
                                        "vec": chunk_vecs[0],
                                    },
                                )
                                self._embed_stats["chunks_indexed"] += 1
                            else:
                                self._embed_stats["chunks_failed"] += 1
        except Exception as e:
            self._embed_stats["episodes_failed"] += 1
            logger.error(
                "Failed to index episode %s: %s "
                "[stats: indexed=%d, failed=%d]",
                episode.id,
                e,
                self._embed_stats["episodes_indexed"],
                self._embed_stats["episodes_failed"],
            )

    async def index_episode_cue(self, cue: EpisodeCue) -> None:
        """Embed and index episode cue text into the CueVec vector store.

        If cue_text is empty/None, this is a no-op (Helix does not support
        targeted vector deletion by metadata, so stale vectors will be
        overwritten on the next index with content).
        """
        if not self._embeddings_enabled or not cue.cue_text:
            return

        try:
            embeddings = await self._embed_texts([cue.cue_text])
            if not embeddings:
                return

            await self._query(
                "add_cue_vector",
                {
                    "episode_id": cue.episode_id,
                    "group_id": cue.group_id,
                    "content_type": "episode_cue",
                    "vec": embeddings[0],
                },
            )
        except Exception as e:
            logger.warning("Failed to index cue %s: %s", cue.episode_id, e)

    async def batch_index_entities(self, entities: list[Entity]) -> int:
        """Batch-embed and index multiple entities. Returns count indexed."""
        if not self._embeddings_enabled or not entities:
            return 0

        texts: list[str] = []
        valid: list[Entity] = []
        for e in entities:
            if e.name:
                t = e.name
                if e.summary:
                    t = f"{e.name}: {e.summary}"
                texts.append(t)
                valid.append(e)

        if not texts:
            return 0

        embeddings = await self._embed_texts(texts)
        if not embeddings or len(embeddings) != len(valid):
            return 0

        indexed = 0
        for entity, vec in zip(valid, embeddings):
            try:
                await self._query(
                    "add_entity_vector",
                    {
                        "entity_id": entity.id,
                        "group_id": entity.group_id,
                        "content_type": "entity",
                        "embed_provider": self._embed_provider,
                        "embed_model": self._embed_model,
                        "vec": vec,
                    },
                )
                indexed += 1
            except Exception as e:
                logger.warning("Failed to index entity %s: %s", entity.id, e)

        return indexed

    # ------------------------------------------------------------------
    # Search (hybrid vector + BM25 with RRF fusion)
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Hybrid entity search: vector + BM25 fused with RRF.

        1. Embeds the query and runs vector search + BM25 concurrently
        2. Fuses results with Reciprocal Rank Fusion
        3. Post-filters by group_id and entity_types
        4. Returns (entity_id, score) pairs normalized to 0-1

        When *group_id* is provided, the vector search path uses a filtered
        HelixQL query that pushes group filtering into the database, so no
        Python-side overfetch is needed for vectors.  BM25 results still
        require post-hoc group filtering with overfetch.
        """
        bm25_fetch_limit = limit * _OVERFETCH_FACTOR

        if not self._embeddings_enabled:
            # BM25 only
            fts_results, fts_rows = await self._bm25_search_entities(query, bm25_fetch_limit)
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            fts_results = self._filter_by_entity_type(fts_results, fts_rows, entity_types)
            return self._normalize_scores(fts_results[:limit])

        # Run embedding + BM25 concurrently
        query_vec, (fts_results, fts_rows) = await asyncio.gather(
            self._embed_text(query),
            self._bm25_search_entities(query, bm25_fetch_limit),
        )
        self._last_query_vec = query_vec

        if not query_vec:
            # Vector embedding failed, fall back to BM25 only
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            fts_results = self._filter_by_entity_type(fts_results, fts_rows, entity_types)
            return self._normalize_scores(fts_results[:limit])

        # Vector search — use filtered query when group_id is available
        # (no overfetch needed when filtering is pushed into HelixDB)
        vec_fetch_limit = limit if group_id else limit * _OVERFETCH_FACTOR
        vec_results, vec_rows, vec_filtered = await self._vector_search_entities(
            query_vec, vec_fetch_limit, group_id=group_id
        )

        # Post-filter BM25 by group (always needed)
        fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
        fts_results = self._filter_by_entity_type(fts_results, fts_rows, entity_types)

        # Post-filter vector results only when the filtered query was NOT used
        if not vec_filtered:
            vec_results = self._filter_by_group(vec_results, vec_rows, group_id)
        vec_results = self._filter_by_entity_type(vec_results, vec_rows, entity_types)

        if not vec_results:
            return self._normalize_scores(fts_results[:limit])

        # RRF fusion
        fused = _rrf_fusion(
            fts_results,
            vec_results,
            self._fts_weight,
            self._vec_weight,
        )
        return fused[:limit]

    async def search_episodes(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Hybrid episode search: vector + BM25 fused with RRF."""
        bm25_fetch_limit = limit * _OVERFETCH_FACTOR

        if not self._embeddings_enabled:
            fts_results, fts_rows = await self._bm25_search_episodes(query, bm25_fetch_limit)
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            return self._normalize_scores(fts_results[:limit])

        query_vec, (fts_results, fts_rows) = await asyncio.gather(
            self._embed_text(query),
            self._bm25_search_episodes(query, bm25_fetch_limit),
        )
        self._last_query_vec = query_vec

        if not query_vec:
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            return self._normalize_scores(fts_results[:limit])

        vec_fetch_limit = limit if group_id else limit * _OVERFETCH_FACTOR
        vec_results, vec_rows, vec_filtered = await self._vector_search_episodes(
            query_vec, vec_fetch_limit, group_id=group_id
        )

        fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
        if not vec_filtered:
            vec_results = self._filter_by_group(vec_results, vec_rows, group_id)

        if not vec_results:
            return self._normalize_scores(fts_results[:limit])

        fused = _rrf_fusion(
            fts_results,
            vec_results,
            self._fts_weight,
            self._vec_weight,
        )
        return fused[:limit]

    async def search_episode_cues(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Hybrid episode cue search: vector + BM25 fused with RRF."""
        bm25_fetch_limit = limit * _OVERFETCH_FACTOR

        if not self._embeddings_enabled:
            fts_results, fts_rows = await self._bm25_search_cues(query, bm25_fetch_limit)
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            return self._normalize_scores(fts_results[:limit])

        query_vec, (fts_results, fts_rows) = await asyncio.gather(
            self._embed_text(query),
            self._bm25_search_cues(query, bm25_fetch_limit),
        )
        self._last_query_vec = query_vec

        if not query_vec:
            fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
            return self._normalize_scores(fts_results[:limit])

        vec_fetch_limit = limit if group_id else limit * _OVERFETCH_FACTOR
        vec_results, vec_rows, vec_filtered = await self._vector_search_cues(
            query_vec, vec_fetch_limit, group_id=group_id
        )

        fts_results = self._filter_by_group(fts_results, fts_rows, group_id)
        if not vec_filtered:
            vec_results = self._filter_by_group(vec_results, vec_rows, group_id)

        if not vec_results:
            return self._normalize_scores(fts_results[:limit])

        fused = _rrf_fusion(
            fts_results,
            vec_results,
            self._fts_weight,
            self._vec_weight,
        )
        return fused[:limit]

    # ------------------------------------------------------------------
    # Chunk search
    # ------------------------------------------------------------------

    async def search_episode_chunks(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search episode chunks for granular retrieval.

        Returns chunk dicts with episode_id, chunk_text, chunk_index, and score.
        Deduplicates by episode_id, keeping the best-scoring chunk per episode.
        """
        if not self._embeddings_enabled:
            return []

        try:
            # Try server-side Embed() search first
            if self._helix_client is not None and group_id:
                results = await self._query(
                    "search_episode_chunks_embed_filtered",
                    {"query": query, "k": limit * 2, "gid": group_id},
                )
            elif self._helix_client is not None:
                results = await self._query(
                    "search_episode_chunks_embed",
                    {"query": query, "k": limit * 2},
                )
            else:
                # Fall back to client-side embedding
                query_vec = await self._embed_query(query)
                if not query_vec:
                    return []
                if group_id:
                    results = await self._query(
                        "search_episode_chunks_filtered",
                        {"vec": query_vec, "k": limit * 2, "gid": group_id},
                    )
                else:
                    results = await self._query(
                        "search_episode_chunks_vec",
                        {"vec": query_vec, "k": limit * 2},
                    )

            # Deduplicate by episode_id, keep best chunk per episode
            seen: dict[str, dict] = {}
            for r in results:
                ep_id = r.get("episode_id", "")
                if ep_id not in seen:
                    seen[ep_id] = r

            return list(seen.values())[:limit]
        except Exception as e:
            logger.warning("search_episode_chunks failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]:
        """Compute cosine similarity between query and stored entity embeddings.

        Embeds the query, retrieves entity vectors from Helix via a broad
        vector search, then computes cosine similarity in Python for the
        requested entity IDs.
        """
        if not self._embeddings_enabled or not entity_ids:
            return {}

        query_vec = await self._embed_text(query)
        if not query_vec:
            return {}

        # Retrieve stored entity embeddings
        stored = await self.get_entity_embeddings(entity_ids, group_id=group_id)
        if not stored:
            return {}

        results: dict[str, float] = {}
        for eid in entity_ids:
            vec = stored.get(eid)
            if vec:
                sim = _cosine_similarity(query_vec, vec)
                results[eid] = max(0.0, sim)

        return results

    # ------------------------------------------------------------------
    # Embedding retrieval
    # ------------------------------------------------------------------

    async def get_entity_embeddings(
        self,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Retrieve stored entity embedding vectors.

        Helix does not support fetching vectors by metadata filter directly.
        Strategy: perform a broad vector search using a zero vector to pull
        back a large number of entries, then filter by entity_id. This is
        a best-effort approach; for exact retrieval a Helix get-by-ID
        endpoint would be needed.

        If very few entity_ids are requested, we can also try searching with
        each entity's text to recover its vector indirectly. However, for
        simplicity we use the zero-vector approach with a large k.
        """
        if not self._embeddings_enabled or not entity_ids:
            return {}

        target_ids = set(entity_ids)
        results: dict[str, list[float]] = {}

        # Use a zero vector to do a broad sweep of the index
        dim = self._storage_dim if self._storage_dim > 0 else self._provider.dimension()
        if dim <= 0:
            return {}

        # Fetch a large batch from the vector index
        zero_vec = [0.0] * dim
        rows = await self._query(
            "search_entity_vectors",
            {"vec": zero_vec, "k": max(len(entity_ids) * 5, 200)},
        )

        for row in rows:
            eid = str(row.get("entity_id", ""))
            if eid in target_ids:
                vec = row.get("vec") or row.get("vector") or row.get("embedding")
                if vec and isinstance(vec, list):
                    results[eid] = [float(v) for v in vec]
                    if len(results) == len(target_ids):
                        break

        # Filter by group_id if specified
        if group_id:
            gid_lookup: dict[str, str] = {}
            for row in rows:
                eid = str(row.get("entity_id", ""))
                gid_lookup[eid] = str(row.get("group_id", ""))
            results = {
                eid: vec
                for eid, vec in results.items()
                if gid_lookup.get(eid) == group_id
            }

        return results

    async def get_graph_embeddings(
        self,
        entity_ids: list[str],
        method: str = "node2vec",
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """Retrieve graph structural embeddings from the GraphEmbedVec index.

        Similar to get_entity_embeddings, uses a broad vector search with a
        zero vector and filters by entity_id and method in Python.
        """
        if not entity_ids:
            return {}

        target_ids = set(entity_ids)
        results: dict[str, list[float]] = {}

        # We need to know the graph embedding dimension. Try a reasonable default.
        # Graph embeddings may have different dimensions than text embeddings.
        # Use a broad search with a moderate k.
        # First try with a small dimension zero vector; if it fails, return empty.
        try:
            # Start with a moderate-dimension zero vector for graph embeddings.
            # Node2Vec default is 64, TransE is 64, GNN is 64.
            graph_dim = 64
            zero_vec = [0.0] * graph_dim

            rows = await self._query(
                "search_graph_embed_vectors",
                {"vec": zero_vec, "k": max(len(entity_ids) * 5, 200)},
            )

            for row in rows:
                eid = str(row.get("entity_id", ""))
                row_method = str(row.get("method", ""))
                row_gid = str(row.get("group_id", ""))

                if eid not in target_ids:
                    continue
                if row_method != method:
                    continue
                if group_id and row_gid != group_id:
                    continue

                vec = row.get("vec") or row.get("vector") or row.get("embedding")
                if vec and isinstance(vec, list):
                    results[eid] = [float(v) for v in vec]
                    if len(results) == len(target_ids):
                        break
        except Exception as e:
            logger.warning("get_graph_embeddings failed: %s", e)

        return results

    # ------------------------------------------------------------------
    # Deletion (best-effort)
    # ------------------------------------------------------------------

    async def remove(self, entity_id: str) -> None:
        """Remove an entity from the search index.

        Helix does not expose a delete-by-metadata endpoint for vectors.
        This is a no-op. Stale vectors will be overwritten on re-index
        and will be filtered out by group/type checks during search.
        """
        logger.debug(
            "HelixSearchIndex.remove(%s): no-op (Helix lacks vector deletion by metadata)",
            entity_id,
        )

    async def delete_group(self, group_id: str) -> None:
        """Remove all search index entries for a group.

        Helix does not expose a bulk-delete-by-metadata endpoint for vectors.
        This is a no-op. Stale vectors will be overwritten on re-index
        and will be filtered out by group checks during search.
        """
        logger.debug(
            "HelixSearchIndex.delete_group(%s): no-op (Helix lacks vector deletion by metadata)",
            group_id,
        )

    # ------------------------------------------------------------------
    # Score normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_scores(
        results: list[tuple[str, float]],
    ) -> list[tuple[str, float]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        max_score = max(s for _, s in results)
        if max_score <= 0:
            return results
        return [(eid, score / max_score) for eid, score in results]
