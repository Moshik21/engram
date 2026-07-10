"""Persistent, content-addressed extraction cache for deterministic evals.

Wraps any extractor exposing ``async extract(text, **kwargs) -> ExtractionResult``
and memoizes successful results in a SQLite table keyed by
``sha256(text) + extractor_model + prompt_version``. On a cache HIT no API call
is made and the byte-identical stored result is rehydrated, which is what makes
a re-run of the depth-tier eval reproducible: the only stochastic stage (LLM
extraction) is frozen after the first successful run.

Design notes:
  * The cache is keyed on the extractor MODEL and a PROMPT VERSION so caches are
    never mixed across providers (Haiku vs narrow produce different entities).
  * Only successful (OK / EMPTY) verdicts are cached. PARSE_ERROR / API_ERROR are
    intentionally NOT cached so a transient failure does not poison the corpus;
    the eval harness treats those as build-failing instead (hard-fail policy).
  * This is a wrapper, not a modification of the core extractor. ``GraphManager``
    receives the wrapped instance and uses it through the normal ``extract`` path.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram.extraction.extractor import ExtractionResult, ExtractionStatus

logger = logging.getLogger(__name__)

# Bump when the extraction prompt or output schema changes so stale verdicts are
# not reused against a different prompt.
PROMPT_VERSION = "v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extraction_verdicts (
    content_hash TEXT NOT NULL,
    extractor_model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    result_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (content_hash, extractor_model, prompt_version)
)
"""


def content_hash(text: str) -> str:
    """Stable SHA-256 of the extraction input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExtractionCache:
    """SQLite-backed memoizing wrapper around an extractor.

    Parameters
    ----------
    inner:
        The wrapped extractor (``EntityExtractor``, ``NarrowExtractorAdapter``,
        ...). Must expose ``async extract(text, **kwargs) -> ExtractionResult``.
    db_path:
        Path to the SQLite cache file. Created if missing.
    extractor_model:
        Identifier recorded in the cache key. Defaults to the inner extractor's
        ``_model`` attribute, else the class name. Keeps caches provider-specific.
    prompt_version:
        Schema/prompt version recorded in the cache key.
    """

    def __init__(
        self,
        inner: Any,
        db_path: str | Path,
        *,
        extractor_model: str | None = None,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self._inner = inner
        self._db_path = str(db_path)
        self._model = extractor_model or getattr(inner, "_model", type(inner).__name__)
        self._prompt_version = prompt_version
        # Hit/miss meters prove determinism in the eval report.
        self.hits = 0
        self.misses = 0
        self.stores = 0

        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    @property
    def extractor_model(self) -> str:
        return self._model

    @property
    def inner(self) -> Any:
        return self._inner

    def _key(self, text: str) -> tuple[str, str, str]:
        return (content_hash(text), self._model, self._prompt_version)

    def _lookup(self, text: str) -> ExtractionResult | None:
        ch, model, ver = self._key(text)
        row = self._conn.execute(
            "SELECT result_json, status FROM extraction_verdicts "
            "WHERE content_hash = ? AND extractor_model = ? AND prompt_version = ?",
            (ch, model, ver),
        ).fetchone()
        if row is None:
            return None
        return _deserialize(row[0], row[1])

    def _store(self, text: str, result: ExtractionResult) -> None:
        ch, model, ver = self._key(text)
        self._conn.execute(
            "INSERT OR REPLACE INTO extraction_verdicts "
            "(content_hash, extractor_model, prompt_version, result_json, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                ch,
                model,
                ver,
                _serialize(result),
                result.status.value,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        self.stores += 1

    def _inner_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Forward only the kwargs the wrapped extractor actually accepts.

        The projector introspects THIS wrapper's ``extract`` signature (which
        declares ``**kwargs``) and therefore passes ``episode_id``/``group_id``.
        The inner extractor may accept only ``(text)`` (e.g. EntityExtractor),
        so blind forwarding raises TypeError. Mirror the inner signature.
        """
        try:
            params = inspect.signature(self._inner.extract).parameters
        except (TypeError, ValueError):
            return {}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    async def extract(self, text: str, **kwargs: Any) -> ExtractionResult:
        """Return the cached verdict on a hit, else delegate and cache success.

        Errors (PARSE_ERROR / API_ERROR / TRUNCATED) are passed through WITHOUT
        caching so a transient failure cannot freeze a broken corpus; the harness
        is expected to hard-fail on such statuses.
        """
        cached = self._lookup(text)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        result = await self._inner.extract(text, **self._inner_kwargs(kwargs))

        if not getattr(result, "is_error", False):
            self._store(text, result)
        return result

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def stats(self) -> dict[str, Any]:
        return {
            "extractor_model": self._model,
            "prompt_version": self._prompt_version,
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "hit_rate": self.hit_rate,
        }

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - close is best-effort
            pass


def _serialize(result: ExtractionResult) -> str:
    """Serialize an ExtractionResult to a stable JSON string.

    ``sort_keys`` makes the stored bytes deterministic for a given result so the
    corpus is byte-identical across runs.
    """
    payload = {
        "entities": result.entities,
        "relationships": result.relationships,
        "status": result.status.value,
        "error": result.error,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _deserialize(result_json: str, status: str) -> ExtractionResult:
    data = json.loads(result_json)
    try:
        status_enum = ExtractionStatus(data.get("status", status))
    except ValueError:
        status_enum = ExtractionStatus(status)
    return ExtractionResult(
        entities=data.get("entities", []),
        relationships=data.get("relationships", []),
        status=status_enum,
        error=data.get("error"),
    )
