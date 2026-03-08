"""FTS5-based search index for lite mode."""

from __future__ import annotations

import logging

import aiosqlite

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.episode_cue import EpisodeCue

logger = logging.getLogger(__name__)

# Stop words to strip from FTS5 queries — these match almost everything
# and cause massive intermediate result sets at scale.
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "its",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "when",
        "where",
        "why",
        "not",
        "no",
        "nor",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "any",
        "only",
        "own",
        "same",
        "just",
        "also",
        "tell",
        "know",
        "think",
        "see",
        "get",
        "make",
        "go",
        "come",
    }
)


class FTS5SearchIndex:
    """Full-text search using SQLite FTS5 with BM25 ranking."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self, db: aiosqlite.Connection | None = None) -> None:
        """Initialize, optionally sharing a db connection."""
        if db:
            self._db = db
        elif not self._db:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("FTS5SearchIndex not initialized.")
        return self._db

    async def index_entity(self, entity: Entity) -> None:
        """Index is maintained via triggers — this is a no-op for FTS5."""
        pass

    async def index_episode(self, episode: Episode) -> None:
        """Index is maintained via triggers — this is a no-op for FTS5."""
        pass

    async def index_episode_cue(self, cue: EpisodeCue) -> None:
        """Index is maintained via triggers — this is a no-op for FTS5."""
        pass

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        group_id: str | None = None,
        limit: int = 20,
    ) -> list[tuple[str, float]]:
        """Search entities using FTS5 BM25 scoring.

        Returns (entity_id, relevance_score) pairs normalized to 0.0-1.0.
        """
        fts_query = self._prepare_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT e.id, rank
            FROM entities_fts fts
            JOIN entities e ON e.rowid = fts.rowid
            WHERE entities_fts MATCH :query
              AND e.deleted_at IS NULL
        """
        params: dict = {"query": fts_query}

        if entity_types:
            placeholders = ",".join(f":type_{i}" for i in range(len(entity_types)))
            sql += f" AND e.entity_type IN ({placeholders})"
            for i, t in enumerate(entity_types):
                params[f"type_{i}"] = t

        if group_id:
            sql += " AND e.group_id = :group_id"
            params["group_id"] = group_id

        sql += " ORDER BY rank LIMIT :limit"
        params["limit"] = limit

        try:
            cursor = await self.db.execute(sql, params)
            rows = await cursor.fetchall()
        except Exception as e:
            logger.warning("FTS5 search failed for query %r: %s", query, e)
            return []

        if not rows:
            return []

        # Normalize BM25 scores to 0.0-1.0 (BM25 returns negative scores)
        max_score = abs(rows[0]["rank"])
        return [(row["id"], abs(row["rank"]) / max_score if max_score > 0 else 0.0) for row in rows]

    async def search_episodes(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search episodes using FTS5 BM25 scoring.

        Returns (episode_id, relevance_score) pairs normalized to 0.0-1.0.
        """
        fts_query = self._prepare_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT ep.id, rank
            FROM episodes_fts fts
            JOIN episodes ep ON ep.rowid = fts.rowid
            WHERE episodes_fts MATCH :query
              AND (ep.projection_state IS NULL OR ep.projection_state != 'merged')
        """
        params: dict = {"query": fts_query}

        if group_id:
            sql += " AND ep.group_id = :group_id"
            params["group_id"] = group_id

        sql += " ORDER BY rank LIMIT :limit"
        params["limit"] = limit

        try:
            cursor = await self.db.execute(sql, params)
            rows = await cursor.fetchall()
        except Exception as e:
            logger.warning("FTS5 episode search failed for query %r: %s", query, e)
            return []

        if not rows:
            return []

        # Normalize BM25 scores to 0.0-1.0 (BM25 returns negative scores)
        max_score = abs(rows[0]["rank"])
        return [(row["id"], abs(row["rank"]) / max_score if max_score > 0 else 0.0) for row in rows]

    async def search_episode_cues(
        self,
        query: str,
        group_id: str | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Search cue text using FTS5 BM25 scoring."""
        fts_query = self._prepare_query(query)
        if not fts_query:
            return []

        sql = """
            SELECT ec.episode_id, rank
            FROM episode_cues_fts fts
            JOIN episode_cues ec ON ec.rowid = fts.rowid
            WHERE episode_cues_fts MATCH :query
        """
        params: dict = {"query": fts_query}
        if group_id:
            sql += " AND ec.group_id = :group_id"
            params["group_id"] = group_id
        sql += " ORDER BY rank LIMIT :limit"
        params["limit"] = limit

        try:
            cursor = await self.db.execute(sql, params)
            rows = await cursor.fetchall()
        except Exception as e:
            logger.warning("FTS5 cue search failed for query %r: %s", query, e)
            return []

        if not rows:
            return []

        max_score = abs(rows[0]["rank"])
        return [
            (row["episode_id"], abs(row["rank"]) / max_score if max_score > 0 else 0.0)
            for row in rows
        ]

    async def close(self) -> None:
        """No-op — connection is shared with the graph store."""
        pass

    async def remove(self, entity_id: str) -> None:
        """Removal is handled by entity deletion triggers."""
        pass

    async def compute_similarity(
        self,
        query: str,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, float]:
        """FTS5 has no embeddings — always returns empty dict."""
        return {}

    async def get_entity_embeddings(
        self,
        entity_ids: list[str],
        group_id: str | None = None,
    ) -> dict[str, list[float]]:
        """FTS5 has no embeddings — always returns empty dict."""
        return {}

    @staticmethod
    def _prepare_query(query: str) -> str:
        """Convert natural language query to FTS5 query syntax.

        Strips common stop words to avoid matching nearly every row,
        which causes severe slowdowns at scale (5k+ entities).
        Falls back to the full token set if all tokens are stop words.
        """
        tokens = query.strip().split()
        if not tokens:
            return ""
        # Strip punctuation from each token for stop-word matching
        content = [t for t in tokens if t and t.strip("?!.,;:'\"").lower() not in _STOP_WORDS]
        # Fall back to original tokens if all were stop words
        if not content:
            content = [t for t in tokens if t]
        return " OR ".join(f'"{t}"*' for t in content)
