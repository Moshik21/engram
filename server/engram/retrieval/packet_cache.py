"""Memory packet cache for bounded recall surfaces."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryPacketCacheEntry:
    """Cached serialized packets plus source ids used for invalidation."""

    cache_key: str
    group_id: str
    scope: str
    topic_hint: str | None
    project_path: str | None
    packets: list[dict[str, Any]]
    source_entity_ids: set[str] = field(default_factory=set)
    source_episode_ids: set[str] = field(default_factory=set)
    source_relationship_ids: set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    invalidated_at: float | None = None
    build_duration_ms: float = 0.0
    hit_count: int = 0
    last_hit_at: float | None = None

    def is_fresh(self, now: float) -> bool:
        if self.invalidated_at is not None:
            return False
        return self.expires_at is None or self.expires_at > now

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "group_id": self.group_id,
            "scope": self.scope,
            "topic_hint": self.topic_hint,
            "project_path": self.project_path,
            "packet_count": len(self.packets),
            "source_entity_ids": sorted(self.source_entity_ids),
            "source_episode_ids": sorted(self.source_episode_ids),
            "source_relationship_ids": sorted(self.source_relationship_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "invalidated_at": self.invalidated_at,
            "build_duration_ms": self.build_duration_ms,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
        }


@dataclass(frozen=True)
class MemoryPacketCacheHit:
    """Fresh cached packets returned to a recall surface."""

    packets: list[dict[str, Any]]
    entry: MemoryPacketCacheEntry


class MemoryPacketCache:
    """Small bounded packet cache with optional SQLite persistence."""

    def __init__(
        self,
        *,
        max_entries: int = 128,
        default_ttl_seconds: float = 300.0,
        persistence_path: str | Path | None = None,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._default_ttl_seconds = max(0.0, float(default_ttl_seconds))
        self._entries: OrderedDict[str, MemoryPacketCacheEntry] = OrderedDict()
        self._persistence_path = (
            Path(persistence_path).expanduser() if persistence_path else None
        )
        self._persistence_failed = False
        if self._persistence_path is not None:
            self._initialize_persistence()
            self._load_persistent_entries()

    def build_key(
        self,
        *,
        group_id: str,
        scope: str,
        topic_hint: str | None = None,
        project_path: str | None = None,
    ) -> str:
        topic_digest = _digest(topic_hint or "")
        project_digest = _digest(project_path or "")
        return f"{group_id}:{scope}:{topic_digest}:{project_digest}"

    def get(
        self,
        *,
        group_id: str,
        scope: str,
        topic_hint: str | None = None,
        project_path: str | None = None,
        now: float | None = None,
    ) -> MemoryPacketCacheHit | None:
        timestamp = time.time() if now is None else now
        key = self.build_key(
            group_id=group_id,
            scope=scope,
            topic_hint=topic_hint,
            project_path=project_path,
        )
        entry = self._entries.get(key)
        if entry is None:
            return None
        if not entry.is_fresh(timestamp):
            self._entries.pop(key, None)
            self._delete_persistent_keys([key])
            return None
        entry.hit_count += 1
        entry.last_hit_at = timestamp
        self._persist_entry(entry)
        self._entries.move_to_end(key)
        return MemoryPacketCacheHit(packets=[dict(packet) for packet in entry.packets], entry=entry)

    def put(
        self,
        *,
        group_id: str,
        scope: str,
        packets: Sequence[Mapping[str, Any]],
        topic_hint: str | None = None,
        project_path: str | None = None,
        ttl_seconds: float | None = None,
        build_duration_ms: float = 0.0,
        now: float | None = None,
    ) -> MemoryPacketCacheEntry:
        timestamp = time.time() if now is None else now
        ttl = self._default_ttl_seconds if ttl_seconds is None else max(0.0, ttl_seconds)
        key = self.build_key(
            group_id=group_id,
            scope=scope,
            topic_hint=topic_hint,
            project_path=project_path,
        )
        packet_list = [dict(packet) for packet in packets]
        entry = MemoryPacketCacheEntry(
            cache_key=key,
            group_id=group_id,
            scope=scope,
            topic_hint=topic_hint,
            project_path=project_path,
            packets=packet_list,
            source_entity_ids=_collect_ids(packet_list, "entity_ids", "entityIds"),
            source_episode_ids=_collect_ids(packet_list, "episode_ids", "episodeIds"),
            source_relationship_ids=_collect_ids(
                packet_list,
                "relationship_ids",
                "relationshipIds",
            ),
            created_at=timestamp,
            updated_at=timestamp,
            expires_at=None if ttl <= 0 else timestamp + ttl,
            build_duration_ms=build_duration_ms,
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._persist_entry(entry)
        self._evict_oldest()
        return entry

    def invalidate(
        self,
        *,
        group_id: str | None = None,
        entity_ids: Sequence[str] | None = None,
        episode_ids: Sequence[str] | None = None,
        relationship_ids: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
        now: float | None = None,
    ) -> int:
        timestamp = time.time() if now is None else now
        entity_set = set(entity_ids or [])
        episode_set = set(episode_ids or [])
        relationship_set = set(relationship_ids or [])
        scope_set = set(scopes or [])
        no_filters = not entity_set and not episode_set and not relationship_set and not scope_set
        invalidated = 0
        for entry in self._entries.values():
            if group_id is not None and entry.group_id != group_id:
                continue
            if not no_filters:
                if scope_set and entry.scope not in scope_set:
                    continue
                if entity_set and not (entry.source_entity_ids & entity_set):
                    continue
                if episode_set and not (entry.source_episode_ids & episode_set):
                    continue
                if relationship_set and not (entry.source_relationship_ids & relationship_set):
                    continue
            if entry.invalidated_at is None:
                entry.invalidated_at = timestamp
                invalidated += 1
                self._persist_entry(entry)
        return invalidated

    def clear(self, *, group_id: str | None = None) -> int:
        if group_id is None:
            count = len(self._entries)
            self._clear_persistent_entries(group_id=None)
            self._entries.clear()
            return count
        keys = [key for key, entry in self._entries.items() if entry.group_id == group_id]
        for key in keys:
            self._entries.pop(key, None)
        self._clear_persistent_entries(group_id=group_id)
        return len(keys)

    def summary(self, *, group_id: str | None = None, now: float | None = None) -> dict[str, Any]:
        timestamp = time.time() if now is None else now
        entries = [
            entry
            for entry in self._entries.values()
            if group_id is None or entry.group_id == group_id
        ]
        fresh = [entry for entry in entries if entry.is_fresh(timestamp)]
        return {
            "entry_count": len(entries),
            "fresh_count": len(fresh),
            "invalidated_count": sum(1 for entry in entries if entry.invalidated_at is not None),
            "expired_count": sum(
                1
                for entry in entries
                if entry.invalidated_at is None
                and entry.expires_at is not None
                and entry.expires_at <= timestamp
            ),
            "hit_count": sum(entry.hit_count for entry in entries),
            "scopes": _scope_counts(fresh),
            "persistent": self._persistence_path is not None and not self._persistence_failed,
            "path": str(self._persistence_path) if self._persistence_path else None,
        }

    def _evict_oldest(self) -> None:
        while len(self._entries) > self._max_entries:
            key, _entry = self._entries.popitem(last=False)
            self._delete_persistent_keys([key])

    def _initialize_persistence(self) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._persistence_path) as db:
                db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_packet_cache (
                        cache_key TEXT PRIMARY KEY,
                        group_id TEXT NOT NULL,
                        scope TEXT NOT NULL,
                        topic_hint TEXT,
                        project_path TEXT,
                        packets_json TEXT NOT NULL,
                        source_entity_ids_json TEXT NOT NULL,
                        source_episode_ids_json TEXT NOT NULL,
                        source_relationship_ids_json TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        expires_at REAL,
                        invalidated_at REAL,
                        build_duration_ms REAL NOT NULL DEFAULT 0,
                        hit_count INTEGER NOT NULL DEFAULT 0,
                        last_hit_at REAL
                    )
                    """
                )
                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_packet_cache_group_scope
                        ON memory_packet_cache(group_id, scope)
                    """
                )
                db.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memory_packet_cache_updated
                        ON memory_packet_cache(updated_at)
                    """
                )
                db.commit()
        except sqlite3.Error:
            self._persistence_failed = True
            LOGGER.warning("Memory packet cache persistence unavailable", exc_info=True)

    def _load_persistent_entries(self) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        now = time.time()
        expired_keys: list[str] = []
        try:
            with sqlite3.connect(self._persistence_path) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    """
                    SELECT * FROM memory_packet_cache
                    ORDER BY updated_at ASC
                    """
                ).fetchall()
        except sqlite3.Error:
            self._persistence_failed = True
            LOGGER.warning("Could not load persistent memory packet cache", exc_info=True)
            return

        for row in rows:
            entry = _entry_from_row(row)
            if entry is None:
                expired_keys.append(str(row["cache_key"]))
                continue
            if entry.expires_at is not None and entry.expires_at <= now:
                expired_keys.append(entry.cache_key)
                continue
            self._entries[entry.cache_key] = entry
            self._entries.move_to_end(entry.cache_key)
        self._delete_persistent_keys(expired_keys)
        self._evict_oldest()

    def _persist_entry(self, entry: MemoryPacketCacheEntry) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        try:
            with sqlite3.connect(self._persistence_path) as db:
                db.execute(
                    """
                    INSERT INTO memory_packet_cache (
                        cache_key,
                        group_id,
                        scope,
                        topic_hint,
                        project_path,
                        packets_json,
                        source_entity_ids_json,
                        source_episode_ids_json,
                        source_relationship_ids_json,
                        created_at,
                        updated_at,
                        expires_at,
                        invalidated_at,
                        build_duration_ms,
                        hit_count,
                        last_hit_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        group_id=excluded.group_id,
                        scope=excluded.scope,
                        topic_hint=excluded.topic_hint,
                        project_path=excluded.project_path,
                        packets_json=excluded.packets_json,
                        source_entity_ids_json=excluded.source_entity_ids_json,
                        source_episode_ids_json=excluded.source_episode_ids_json,
                        source_relationship_ids_json=excluded.source_relationship_ids_json,
                        created_at=excluded.created_at,
                        updated_at=excluded.updated_at,
                        expires_at=excluded.expires_at,
                        invalidated_at=excluded.invalidated_at,
                        build_duration_ms=excluded.build_duration_ms,
                        hit_count=excluded.hit_count,
                        last_hit_at=excluded.last_hit_at
                    """,
                    _entry_row_values(entry),
                )
                db.commit()
        except sqlite3.Error:
            self._persistence_failed = True
            LOGGER.warning("Could not persist memory packet cache entry", exc_info=True)

    def _delete_persistent_keys(self, keys: Sequence[str]) -> None:
        if not keys or self._persistence_path is None or self._persistence_failed:
            return
        try:
            with sqlite3.connect(self._persistence_path) as db:
                db.executemany(
                    "DELETE FROM memory_packet_cache WHERE cache_key = ?",
                    [(key,) for key in keys],
                )
                db.commit()
        except sqlite3.Error:
            self._persistence_failed = True
            LOGGER.warning("Could not delete memory packet cache entries", exc_info=True)

    def _clear_persistent_entries(self, *, group_id: str | None) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        try:
            with sqlite3.connect(self._persistence_path) as db:
                if group_id is None:
                    db.execute("DELETE FROM memory_packet_cache")
                else:
                    db.execute(
                        "DELETE FROM memory_packet_cache WHERE group_id = ?",
                        (group_id,),
                    )
                db.commit()
        except sqlite3.Error:
            self._persistence_failed = True
            LOGGER.warning("Could not clear memory packet cache entries", exc_info=True)


def _collect_ids(
    packets: Sequence[Mapping[str, Any]],
    snake_key: str,
    camel_key: str,
) -> set[str]:
    ids: set[str] = set()
    for packet in packets:
        raw = packet.get(snake_key)
        if raw is None:
            raw = packet.get(camel_key)
        if isinstance(raw, list):
            ids.update(str(value) for value in raw if value)
    return ids


def _scope_counts(entries: Sequence[MemoryPacketCacheEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.scope] = counts.get(entry.scope, 0) + 1
    return counts


def _digest(value: str) -> str:
    if not value:
        return "none"
    normalized = " ".join(value.split()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def packet_cache_json_size(packets: Sequence[Mapping[str, Any]]) -> int:
    """Return a deterministic byte-size estimate for serialized packet payloads."""
    return len(json.dumps(list(packets), sort_keys=True, separators=(",", ":")))


def _entry_row_values(entry: MemoryPacketCacheEntry) -> tuple[Any, ...]:
    return (
        entry.cache_key,
        entry.group_id,
        entry.scope,
        entry.topic_hint,
        entry.project_path,
        _json_dumps(entry.packets),
        _json_dumps(sorted(entry.source_entity_ids)),
        _json_dumps(sorted(entry.source_episode_ids)),
        _json_dumps(sorted(entry.source_relationship_ids)),
        entry.created_at,
        entry.updated_at,
        entry.expires_at,
        entry.invalidated_at,
        entry.build_duration_ms,
        entry.hit_count,
        entry.last_hit_at,
    )


def _entry_from_row(row: sqlite3.Row) -> MemoryPacketCacheEntry | None:
    packets = _json_list(row["packets_json"])
    packet_dicts = [dict(packet) for packet in packets if isinstance(packet, Mapping)]
    if not packet_dicts:
        return None
    return MemoryPacketCacheEntry(
        cache_key=str(row["cache_key"]),
        group_id=str(row["group_id"]),
        scope=str(row["scope"]),
        topic_hint=row["topic_hint"],
        project_path=row["project_path"],
        packets=packet_dicts,
        source_entity_ids=_string_set_from_json(row["source_entity_ids_json"]),
        source_episode_ids=_string_set_from_json(row["source_episode_ids_json"]),
        source_relationship_ids=_string_set_from_json(row["source_relationship_ids_json"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        expires_at=_float_or_none(row["expires_at"]),
        invalidated_at=_float_or_none(row["invalidated_at"]),
        build_duration_ms=float(row["build_duration_ms"] or 0.0),
        hit_count=int(row["hit_count"] or 0),
        last_hit_at=_float_or_none(row["last_hit_at"]),
    )


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    return value if isinstance(value, list) else []


def _string_set_from_json(raw: Any) -> set[str]:
    return {str(value) for value in _json_list(raw) if value}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
