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
_PERSISTENT_READ_UNAVAILABLE = object()
_STARTUP_RESIDENT_SCOPES = frozenset({"identity_core", "project_home"})


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
    persisted: bool = False
    last_persistent_sync_at: float | None = None

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
        persistence_timeout_seconds: float = 0.05,
        persistent_sync_interval_seconds: float = 300.0,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._default_ttl_seconds = max(0.0, float(default_ttl_seconds))
        self._entries: OrderedDict[str, MemoryPacketCacheEntry] = OrderedDict()
        self._persistence_path = (
            Path(persistence_path).expanduser() if persistence_path else None
        )
        self._persistence_timeout_seconds = max(
            0.001,
            float(persistence_timeout_seconds),
        )
        self._persistent_sync_interval_seconds = max(
            0.0,
            float(persistent_sync_interval_seconds),
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
        sync_persistent: bool = True,
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
        if (
            sync_persistent
            and self._persistence_path is not None
            and not self._persistence_failed
            and not (entry is not None and not entry.persisted)
            and self._should_sync_persistent_entry(entry, timestamp)
        ):
            persistent_entry = self._load_persistent_entry(key, timestamp)
            if persistent_entry is _PERSISTENT_READ_UNAVAILABLE:
                if entry is None:
                    return None
            elif persistent_entry is None:
                self._entries.pop(key, None)
                return None
            else:
                entry = persistent_entry
        if entry is None:
            return None
        if not entry.is_fresh(timestamp):
            self._entries.pop(key, None)
            self._delete_persistent_keys([key])
            return None
        entry.hit_count += 1
        entry.last_hit_at = timestamp
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
        persist: bool = True,
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
            persisted=bool(
                persist
                and self._persistence_path is not None
                and not self._persistence_failed
            ),
            last_persistent_sync_at=timestamp if persist else None,
        )
        self._entries[key] = entry
        self._entries.move_to_end(key)
        if persist and not self._persist_entry(entry):
            entry.persisted = False
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
        preserve_project_file_packets: bool = False,
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
            if preserve_project_file_packets and _entry_has_only_project_file_packets(entry):
                continue
            if entry.invalidated_at is None:
                entry.invalidated_at = timestamp
                invalidated += 1
                if entry.persisted and not self._persist_entry(entry):
                    entry.persisted = False
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
        self._sync_persistent_entries(now=timestamp)
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

    def recent_packets(
        self,
        *,
        group_id: str | None = None,
        scopes: Sequence[str] | None = None,
        limit_packets: int = 5,
        sync_persistent: bool = True,
        now: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return fresh packets from recent cache entries for degraded fallbacks."""
        timestamp = time.time() if now is None else now
        if sync_persistent:
            self._sync_persistent_entries(now=timestamp)
        scope_set = set(scopes or [])
        packets: list[dict[str, Any]] = []
        seen_packets: set[str] = set()
        entries = sorted(
            self._entries.values(),
            key=lambda entry: entry.updated_at,
            reverse=True,
        )
        for entry in entries:
            if len(packets) >= limit_packets:
                break
            if group_id is not None and entry.group_id != group_id:
                continue
            if scope_set and entry.scope not in scope_set:
                continue
            if not entry.is_fresh(timestamp):
                continue
            for packet in entry.packets:
                if len(packets) >= limit_packets:
                    break
                packet_payload = {**dict(packet), "_cache_scope": entry.scope}
                fingerprint = _packet_fingerprint(packet_payload)
                if fingerprint in seen_packets:
                    continue
                seen_packets.add(fingerprint)
                packets.append(packet_payload)
        return packets

    def _evict_oldest(self) -> None:
        while len(self._entries) > self._max_entries:
            key = self._oldest_evictable_key()
            _entry = self._entries.pop(key)
            self._delete_persistent_keys([key])

    def _oldest_evictable_key(self) -> str:
        for key, entry in self._entries.items():
            if entry.scope not in _STARTUP_RESIDENT_SCOPES:
                return key
        return next(iter(self._entries))

    def _initialize_persistence(self) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        try:
            self._persistence_path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as db:
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
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "initialize memory packet cache persistence"):
                return
            self._persistence_failed = True
            LOGGER.warning("Memory packet cache persistence unavailable", exc_info=True)

    def _load_persistent_entries(self) -> bool:
        if self._persistence_path is None or self._persistence_failed:
            return False
        now = time.time()
        expired_keys: list[str] = []
        try:
            with self._connect() as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    """
                    SELECT * FROM memory_packet_cache
                    ORDER BY updated_at ASC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "load persistent memory packet cache entries"):
                return False
            self._persistence_failed = True
            LOGGER.warning("Could not load persistent memory packet cache", exc_info=True)
            return False

        for row in rows:
            entry = _entry_from_row(row)
            if entry is None:
                expired_keys.append(str(row["cache_key"]))
                continue
            if entry.expires_at is not None and entry.expires_at <= now:
                expired_keys.append(entry.cache_key)
                continue
            entry.last_persistent_sync_at = now
            existing = self._entries.get(entry.cache_key)
            if existing is not None and not existing.persisted:
                continue
            if existing is not None:
                entry.hit_count = max(entry.hit_count, existing.hit_count)
                entry.last_hit_at = _max_optional_float(
                    entry.last_hit_at,
                    existing.last_hit_at,
                )
            self._entries[entry.cache_key] = entry
            self._entries.move_to_end(entry.cache_key)
        self._delete_persistent_keys(expired_keys)
        self._evict_oldest()
        return True

    def _sync_persistent_entries(self, *, now: float | None = None) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        before_keys = set(self._entries)
        if not self._load_persistent_entries():
            return
        persistent_keys = self._persistent_keys()
        if persistent_keys is None:
            return
        for key in before_keys.difference(persistent_keys):
            entry = self._entries.get(key)
            if entry is not None and not entry.persisted:
                continue
            self._entries.pop(key, None)
        timestamp = time.time() if now is None else now
        expired_keys = [
            key
            for key, entry in self._entries.items()
            if entry.invalidated_at is None
            and entry.expires_at is not None
            and entry.expires_at <= timestamp
        ]
        for key in expired_keys:
            self._entries.pop(key, None)
        self._delete_persistent_keys(expired_keys)

    def _load_persistent_entry(
        self,
        key: str,
        now: float,
    ) -> MemoryPacketCacheEntry | None | object:
        if self._persistence_path is None or self._persistence_failed:
            return None
        try:
            with self._connect() as db:
                db.row_factory = sqlite3.Row
                row = db.execute(
                    "SELECT * FROM memory_packet_cache WHERE cache_key = ?",
                    (key,),
                ).fetchone()
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "load memory packet cache entry"):
                return _PERSISTENT_READ_UNAVAILABLE
            self._persistence_failed = True
            LOGGER.warning("Could not load memory packet cache entry", exc_info=True)
            return None
        if row is None:
            return None
        entry = _entry_from_row(row)
        if entry is None or not entry.is_fresh(now):
            self._delete_persistent_keys([key])
            return None
        entry.last_persistent_sync_at = now
        self._entries[key] = entry
        self._entries.move_to_end(key)
        self._evict_oldest()
        return entry

    def _should_sync_persistent_entry(
        self,
        entry: MemoryPacketCacheEntry | None,
        now: float,
    ) -> bool:
        if entry is None:
            return True
        if not entry.persisted:
            return False
        last_sync = entry.last_persistent_sync_at
        if last_sync is None:
            return True
        return (now - last_sync) >= self._persistent_sync_interval_seconds

    def _persistent_keys(self) -> set[str] | None:
        if self._persistence_path is None or self._persistence_failed:
            return None
        try:
            with self._connect() as db:
                rows = db.execute("SELECT cache_key FROM memory_packet_cache").fetchall()
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "list memory packet cache keys"):
                return None
            self._persistence_failed = True
            LOGGER.warning("Could not list memory packet cache keys", exc_info=True)
            return None
        return {str(row[0]) for row in rows}

    def _persist_entry(self, entry: MemoryPacketCacheEntry) -> bool:
        if self._persistence_path is None or self._persistence_failed:
            return False
        try:
            with self._connect() as db:
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
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "persist memory packet cache entry"):
                return False
            self._persistence_failed = True
            LOGGER.warning("Could not persist memory packet cache entry", exc_info=True)
            return False
        return True

    def _delete_persistent_keys(self, keys: Sequence[str]) -> None:
        if not keys or self._persistence_path is None or self._persistence_failed:
            return
        try:
            with self._connect() as db:
                db.executemany(
                    "DELETE FROM memory_packet_cache WHERE cache_key = ?",
                    [(key,) for key in keys],
                )
                db.commit()
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "delete memory packet cache entries"):
                return
            self._persistence_failed = True
            LOGGER.warning("Could not delete memory packet cache entries", exc_info=True)

    def _clear_persistent_entries(self, *, group_id: str | None) -> None:
        if self._persistence_path is None or self._persistence_failed:
            return
        try:
            with self._connect() as db:
                if group_id is None:
                    db.execute("DELETE FROM memory_packet_cache")
                else:
                    db.execute(
                        "DELETE FROM memory_packet_cache WHERE group_id = ?",
                        (group_id,),
                    )
                db.commit()
        except sqlite3.Error as exc:
            if self._sqlite_busy(exc, "clear memory packet cache entries"):
                return
            self._persistence_failed = True
            LOGGER.warning("Could not clear memory packet cache entries", exc_info=True)

    def _connect(self) -> sqlite3.Connection:
        if self._persistence_path is None:
            raise sqlite3.OperationalError("memory packet cache persistence path missing")
        return sqlite3.connect(
            self._persistence_path,
            timeout=self._persistence_timeout_seconds,
        )

    def _sqlite_busy(self, exc: sqlite3.Error, operation: str) -> bool:
        if not isinstance(exc, sqlite3.OperationalError):
            return False
        message = str(exc).lower()
        if "database is locked" not in message and "database table is locked" not in message:
            return False
        LOGGER.debug("Skipped %s because packet cache SQLite sidecar is locked", operation)
        return True


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


def _entry_has_only_project_file_packets(entry: MemoryPacketCacheEntry) -> bool:
    if not entry.packets:
        return False
    for packet in entry.packets:
        trust = packet.get("trust")
        if not isinstance(trust, Mapping) or trust.get("source") != "project_file":
            return False
    return True


def _packet_fingerprint(packet: Mapping[str, Any]) -> str:
    provenance = packet.get("provenance") or packet.get("sources") or []
    if isinstance(provenance, list | tuple):
        provenance_text = "|".join(str(item) for item in provenance)
    else:
        provenance_text = str(provenance or "")
    if provenance_text:
        return f"provenance:{provenance_text}"
    title = str(packet.get("title") or "")
    summary = str(packet.get("summary") or "")
    packet_type = str(packet.get("packet_type") or packet.get("packetType") or "")
    return f"{packet_type}:{title}:{summary}"


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
        persisted=True,
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


def _max_optional_float(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
