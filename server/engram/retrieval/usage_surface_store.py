"""Durable surfaced-payload registry for the surfaced -> used citation scan.

Ticket #7. The only producer of ``EpisodeCue.usage_used_count`` is
``record_observed_usage_events`` (ingestion/capture_surface.py), and it requires
a cue to be SURFACED and then ECHOED BACK. Until now the surfaced half lived
only in a process-local ring buffer (``SurfacedUsageBuffer``), so the signal
could fire only when both halves landed inside a single shell process lifetime.
An agent session spans shell restarts, so for the MCP/axi consumer the loop was
not merely unfired — it was architecturally incapable of firing.

This module is the durable half: a small SQLite sidecar, alongside the packet
cache and cue-index outbox, holding exactly what the ring holds — surfaced
entities, surfaced cues, the echo-mask token lists, and the used-event dedup
marks — scoped by group and bounded by the same cap plus a TTL window.

**Persisting the echo mask is not optional.** A durable surfaced-cue record
without its mask would make a post-restart VERBATIM PARROT of the payload we
just handed the agent read as novel reuse — i.e. the durability fix would
manufacture phantom uses, which is the one outcome the RF flip gate must never
see. Every row the in-process ring uses as a mask source is persisted with it.

Everything here is best-effort: a sidecar failure degrades to today's
process-local behaviour and never raises into recall or capture.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USAGE_SURFACE_FILENAME = "surfaced-usage.sqlite3"

_SCHEMA_VERSION = 1
_SQLITE_TIMEOUT_SECONDS = 0.25


@dataclass
class DurableUsageState:
    """One group's durable surfaced state, oldest first."""

    entities: list[tuple[str, str, float, list[str]]] = field(default_factory=list)
    cues: list[tuple[str, float, list[list[str]]]] = field(default_factory=list)
    texts: list[tuple[str, float, list[str]]] = field(default_factory=list)
    dedup: list[tuple[str, float]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.entities or self.cues or self.texts or self.dedup)


def resolve_usage_surface_path(cfg: Any) -> Path | None:
    """Return the sidecar path for ``cfg``, or None when none is resolvable.

    There is no dedicated knob for this sidecar (``config.py`` is owned by
    another lane; see the report). The directory is taken from whichever recall
    sidecar the runtime entrypoint already resolved, so the durable registry
    always lands next to the brain it belongs to. When neither is set — unit
    tests constructing a bare ``ActivationConfig`` — the store stays unbound and
    behaviour is byte-identical to the process-local ring.

    **Refuses rather than guesses.** The first version stringified whatever it
    was handed, so a `MagicMock` cfg in a test produced a *relative* directory
    (`MagicMock/mock._cfg.recall_packet_cache_path/`) and the store happily
    created it in the repo root. Anything that is not a non-empty ``str``
    resolving to an absolute path is not a brain directory, and an unbound
    registry (today's behaviour) beats a sidecar in an arbitrary location.
    """
    for attr in ("recall_packet_cache_path", "cue_index_outbox_path"):
        configured = getattr(cfg, attr, None)
        if not isinstance(configured, str) or not configured.strip():
            continue
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            logger.debug(
                "Ignoring non-absolute %s for the surfaced-usage sidecar: %r", attr, configured
            )
            continue
        return candidate.with_name(USAGE_SURFACE_FILENAME)
    return None


class SurfacedUsageStore:
    """SQLite sidecar for the surfaced-usage ring buffer.

    Rows are keyed by (group, id) and upserted, so two processes writing the
    same brain merge rather than clobber. Reads and writes are both bounded by
    a TTL floor supplied by the caller — the store never decides eligibility.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.failed = False
        self._initialize()

    # --- setup -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=_SQLITE_TIMEOUT_SECONDS)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _initialize(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS surfaced_entity (
                        group_id TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        ts REAL NOT NULL,
                        tokens_json TEXT NOT NULL,
                        PRIMARY KEY (group_id, entity_id)
                    );
                    CREATE TABLE IF NOT EXISTS surfaced_cue (
                        group_id TEXT NOT NULL,
                        episode_id TEXT NOT NULL,
                        ts REAL NOT NULL,
                        phrases_json TEXT NOT NULL,
                        PRIMARY KEY (group_id, episode_id)
                    );
                    CREATE TABLE IF NOT EXISTS surfaced_text (
                        group_id TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        ts REAL NOT NULL,
                        tokens_json TEXT NOT NULL,
                        PRIMARY KEY (group_id, digest)
                    );
                    CREATE TABLE IF NOT EXISTS used_dedup (
                        group_id TEXT NOT NULL,
                        dedup_key TEXT NOT NULL,
                        ts REAL NOT NULL,
                        PRIMARY KEY (group_id, dedup_key)
                    );
                    """
                )
                conn.execute(f"PRAGMA user_version={_SCHEMA_VERSION}")
        except Exception:
            self.failed = True
            logger.debug("Surfaced-usage sidecar init failed at %s", self.path, exc_info=True)

    # --- io --------------------------------------------------------------

    def load(self, group_id: str, *, min_ts: float, cap: int) -> DurableUsageState:
        """Return the group's rows at or after ``min_ts``, newest ``cap`` each."""
        state = DurableUsageState()
        if self.failed:
            return state
        try:
            with self._connect() as conn:
                for row in conn.execute(
                    "SELECT entity_id, name, ts, tokens_json FROM surfaced_entity "
                    "WHERE group_id = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                    (group_id, min_ts, cap),
                ):
                    state.entities.append((row[0], row[1], float(row[2]), json.loads(row[3])))
                for row in conn.execute(
                    "SELECT episode_id, ts, phrases_json FROM surfaced_cue "
                    "WHERE group_id = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                    (group_id, min_ts, cap),
                ):
                    state.cues.append((row[0], float(row[1]), json.loads(row[2])))
                for row in conn.execute(
                    "SELECT digest, ts, tokens_json FROM surfaced_text "
                    "WHERE group_id = ? AND ts >= ? ORDER BY ts DESC LIMIT ?",
                    (group_id, min_ts, cap),
                ):
                    state.texts.append((row[0], float(row[1]), json.loads(row[2])))
                for row in conn.execute(
                    "SELECT dedup_key, ts FROM used_dedup WHERE group_id = ? AND ts >= ?",
                    (group_id, min_ts),
                ):
                    state.dedup.append((row[0], float(row[1])))
        except Exception:
            logger.debug("Surfaced-usage sidecar read failed", exc_info=True)
            return DurableUsageState()
        state.entities.reverse()
        state.cues.reverse()
        state.texts.reverse()
        return state

    def save(
        self,
        group_id: str,
        *,
        entities: list[tuple[str, str, float, list[str]]],
        cues: list[tuple[str, float, list[list[str]]]],
        texts: list[tuple[str, float, list[str]]],
        dedup: list[tuple[str, float]],
        min_ts: float,
    ) -> bool:
        """Upsert the supplied rows and prune everything older than ``min_ts``."""
        if self.failed:
            return False
        try:
            with self._connect() as conn:
                if entities:
                    conn.executemany(
                        "INSERT INTO surfaced_entity (group_id, entity_id, name, ts, tokens_json) "
                        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(group_id, entity_id) DO UPDATE SET "
                        "name=excluded.name, ts=excluded.ts, tokens_json=excluded.tokens_json",
                        [
                            (group_id, eid, name, ts, json.dumps(tokens))
                            for eid, name, ts, tokens in entities
                        ],
                    )
                if cues:
                    conn.executemany(
                        "INSERT INTO surfaced_cue (group_id, episode_id, ts, phrases_json) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(group_id, episode_id) DO UPDATE SET "
                        "ts=excluded.ts, phrases_json=excluded.phrases_json",
                        [
                            (group_id, episode_id, ts, json.dumps(phrases))
                            for episode_id, ts, phrases in cues
                        ],
                    )
                if texts:
                    conn.executemany(
                        "INSERT INTO surfaced_text (group_id, digest, ts, tokens_json) "
                        "VALUES (?, ?, ?, ?) ON CONFLICT(group_id, digest) DO UPDATE SET "
                        "ts=excluded.ts, tokens_json=excluded.tokens_json",
                        [
                            (group_id, digest, ts, json.dumps(tokens))
                            for digest, ts, tokens in texts
                        ],
                    )
                if dedup:
                    conn.executemany(
                        "INSERT INTO used_dedup (group_id, dedup_key, ts) VALUES (?, ?, ?) "
                        "ON CONFLICT(group_id, dedup_key) DO UPDATE SET ts=excluded.ts",
                        [(group_id, key, ts) for key, ts in dedup],
                    )
                for table in (
                    "surfaced_entity",
                    "surfaced_cue",
                    "surfaced_text",
                    "used_dedup",
                ):
                    conn.execute(f"DELETE FROM {table} WHERE ts < ?", (min_ts,))
            return True
        except Exception:
            logger.debug("Surfaced-usage sidecar write failed", exc_info=True)
            return False

    def clear(self, group_id: str | None = None) -> None:
        if self.failed:
            return
        try:
            with self._connect() as conn:
                for table in (
                    "surfaced_entity",
                    "surfaced_cue",
                    "surfaced_text",
                    "used_dedup",
                ):
                    if group_id is None:
                        conn.execute(f"DELETE FROM {table}")
                    else:
                        conn.execute(f"DELETE FROM {table} WHERE group_id = ?", (group_id,))
        except Exception:
            logger.debug("Surfaced-usage sidecar clear failed", exc_info=True)
