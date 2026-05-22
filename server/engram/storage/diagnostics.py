"""Storage visibility helpers for operator and dashboard surfaces."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from engram.config import EngramConfig

LOGGER = logging.getLogger(__name__)
STARTUP_TIMEOUT_ENV = "ENGRAM_STORAGE_STARTUP_TIMEOUT_SECONDS"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 5.0


def default_helix_native_data_dir() -> Path:
    """Return the default path used by the bundled PyO3 Helix runtime."""
    return Path.home() / ".helix" / "engram-native"


def resolve_helix_native_data_dir(config: EngramConfig) -> Path:
    """Return the configured or default native Helix data directory."""
    if config.helix.data_dir:
        return Path(config.helix.data_dir).expanduser()
    return default_helix_native_data_dir()


def resolve_engram_home() -> Path:
    """Return the local Engram home used by installer-managed files."""
    return Path(os.environ.get("ENGRAM_HOME", "~/.engram")).expanduser()


@dataclass
class StorageDiagnostics:
    """Capture storage baselines and produce live storage reports."""

    config: EngramConfig
    mode: str
    graph_store: Any
    group_id: str
    started_at: float
    startup_counts: dict[str, int]
    startup_bytes: int
    startup_paths: list[dict[str, Any]]
    count_cache: dict[str, _CachedCounts] = field(default_factory=dict)
    path_cache: _CachedPaths | None = None
    _count_lock: Lock = field(default_factory=Lock, repr=False)

    @classmethod
    async def create(
        cls,
        *,
        config: EngramConfig,
        mode: str,
        graph_store: Any,
        group_id: str,
        startup_timeout_seconds: float | None = None,
    ) -> StorageDiagnostics:
        """Create diagnostics with a startup baseline."""
        started_at = time.time()
        timeout_seconds = (
            startup_timeout_seconds
            if startup_timeout_seconds is not None
            else _startup_timeout_seconds()
        )
        counts = await _startup_baseline(
            _read_counts(graph_store, group_id),
            label="storage count baseline",
            timeout_seconds=timeout_seconds,
            fallback=_empty_counts,
        )
        paths = await _startup_baseline(
            asyncio.to_thread(collect_storage_paths, config, mode),
            label="storage disk baseline",
            timeout_seconds=timeout_seconds,
            fallback=list,
        )
        captured_at = time.time()
        startup_counts = _clone_counts(counts)
        startup_paths = _clone_paths(paths)
        return cls(
            config=config,
            mode=mode,
            graph_store=graph_store,
            group_id=group_id,
            started_at=started_at,
            startup_counts=startup_counts,
            startup_bytes=sum(item["bytes"] for item in startup_paths),
            startup_paths=startup_paths,
            count_cache={
                group_id: _CachedCounts(
                    counts=startup_counts,
                    captured_at=captured_at,
                    status="startup",
                )
            },
            path_cache=_CachedPaths(
                paths=startup_paths,
                captured_at=captured_at,
                status="startup",
            ),
        )

    def record_counts_delta(
        self,
        group_id: str | None = None,
        *,
        episodes: int = 0,
        entities: int = 0,
        relationships: int = 0,
        cues: int = 0,
    ) -> None:
        """Update cached graph counts after a successful write-path mutation."""
        resolved_group = group_id or self.group_id
        deltas = {
            "episodes": episodes,
            "entities": entities,
            "relationships": relationships,
            "cues": cues,
        }
        if not any(deltas.values()):
            return
        with self._count_lock:
            cached = self.count_cache.get(resolved_group)
            base_counts = cached.counts if cached is not None else _empty_counts()
            updated = {
                key: max(0, int(base_counts.get(key, 0)) + int(delta))
                for key, delta in deltas.items()
            }
            captured_at = time.time()
            self.count_cache[resolved_group] = _CachedCounts(
                counts=updated,
                captured_at=captured_at,
                status="write_through",
            )

    async def snapshot(
        self,
        *,
        group_id: str | None = None,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-serializable storage report.

        The default path is startup-safe: it returns cached counts and paths without
        touching the loaded graph store. Operator surfaces can request ``live`` for
        a bounded refresh that updates the cache on success.
        """
        resolved_group = group_id or self.group_id
        timeout = _snapshot_timeout_seconds(timeout_seconds)
        started = time.perf_counter()
        count_started = time.perf_counter()
        count_snapshot = await self._counts_snapshot(
            group_id=resolved_group,
            live=live,
            timeout_seconds=timeout,
        )
        count_ms = _elapsed_ms(count_started)
        path_started = time.perf_counter()
        path_snapshot = await self._paths_snapshot(
            live=live,
            timeout_seconds=timeout,
        )
        path_ms = _elapsed_ms(path_started)
        counts = _clone_counts(count_snapshot.counts)
        baseline_counts = (
            _clone_counts(self.startup_counts)
            if resolved_group == self.group_id
            else _clone_counts(counts)
        )
        paths = _clone_paths(path_snapshot.paths)
        total_bytes = sum(item["bytes"] for item in paths)
        started_at = datetime.fromtimestamp(self.started_at, tz=UTC).isoformat()
        now = time.time()

        return {
            "mode": self.mode,
            "configuredMode": self.config.mode,
            "backend": _backend_label(self.config, self.mode),
            "groupId": resolved_group,
            "startedAt": started_at,
            "uptimeSeconds": max(0, int(time.time() - self.started_at)),
            "counts": counts,
            "startupCounts": baseline_counts,
            "growthSinceStartup": {
                "bytes": total_bytes - self.startup_bytes,
                "episodes": counts["episodes"] - baseline_counts["episodes"],
                "entities": counts["entities"] - baseline_counts["entities"],
                "relationships": counts["relationships"] - baseline_counts["relationships"],
                "cues": counts["cues"] - baseline_counts["cues"],
            },
            "disk": {
                "totalBytes": total_bytes,
                "humanSize": human_bytes(total_bytes),
                "startupBytes": self.startup_bytes,
                "startupHumanSize": human_bytes(self.startup_bytes),
            },
            "paths": paths,
            "diagnostics": {
                "live": live,
                "countsStatus": count_snapshot.status,
                "countsAgeSeconds": _age_seconds(count_snapshot.captured_at, now=now),
                "pathsStatus": path_snapshot.status,
                "pathsAgeSeconds": _age_seconds(path_snapshot.captured_at, now=now),
                "stageTimingsMs": {
                    "storage_counts": count_ms,
                    "storage_paths": path_ms,
                    "storage_snapshot": _elapsed_ms(started),
                },
            },
        }

    async def _counts_snapshot(
        self,
        *,
        group_id: str,
        live: bool,
        timeout_seconds: float,
    ) -> _CachedCounts:
        cached = self.count_cache.get(group_id)
        if not live:
            if cached is not None:
                return _cached_counts(
                    cached.counts,
                    captured_at=cached.captured_at,
                    status="cached" if cached.status == "startup" else cached.status,
                )
            return _cached_counts(_empty_counts(), captured_at=self.started_at, status="missing")

        try:
            counts = await _bounded(
                _read_counts(self.graph_store, group_id),
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            LOGGER.warning(
                "storage count snapshot timed out after %.1f seconds; using cached counts",
                timeout_seconds,
            )
            if cached is not None:
                return _cached_counts(
                    cached.counts,
                    captured_at=cached.captured_at,
                    status="cached_timeout",
                )
            return _cached_counts(_empty_counts(), captured_at=time.time(), status="timeout")

        snapshot = _cached_counts(counts, captured_at=time.time(), status="live")
        self.count_cache[group_id] = snapshot
        return snapshot

    async def _paths_snapshot(
        self,
        *,
        live: bool,
        timeout_seconds: float,
    ) -> _CachedPaths:
        cached = self.path_cache
        if not live:
            if cached is not None:
                return _cached_paths(
                    cached.paths,
                    captured_at=cached.captured_at,
                    status="cached",
                )
            return _cached_paths([], captured_at=self.started_at, status="missing")

        try:
            paths = await _bounded(
                asyncio.to_thread(collect_storage_paths, self.config, self.mode),
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError:
            LOGGER.warning(
                "storage disk snapshot timed out after %.1f seconds; using cached paths",
                timeout_seconds,
            )
            if cached is not None:
                return _cached_paths(
                    cached.paths,
                    captured_at=cached.captured_at,
                    status="cached_timeout",
                )
            return _cached_paths([], captured_at=time.time(), status="timeout")

        snapshot = _cached_paths(paths, captured_at=time.time(), status="live")
        self.path_cache = snapshot
        return snapshot


@dataclass(frozen=True)
class _CachedCounts:
    counts: dict[str, int]
    captured_at: float
    status: str


@dataclass(frozen=True)
class _CachedPaths:
    paths: list[dict[str, Any]]
    captured_at: float
    status: str


def collect_storage_paths(config: EngramConfig, mode: str) -> list[dict[str, Any]]:
    """Collect local storage paths and current disk sizes."""
    paths: list[tuple[str, Path]] = []
    sqlite_path = Path(config.sqlite.path).expanduser()
    engram_home = resolve_engram_home()

    if mode == "helix" and config.helix.transport in {"native", "auto"}:
        paths.append(("Helix native data", resolve_helix_native_data_dir(config)))
        paths.extend(_sqlite_companion_paths(sqlite_path, label_prefix="SQLite companion"))
    elif mode == "lite":
        paths.extend(_sqlite_companion_paths(sqlite_path, label_prefix="SQLite database"))
    elif mode == "full":
        paths.extend(_sqlite_companion_paths(sqlite_path, label_prefix="Local companion"))
    else:
        paths.extend(_sqlite_companion_paths(sqlite_path, label_prefix="SQLite database"))

    paths.extend(
        [
            ("Capture queue", engram_home / "capture-queue.jsonl"),
            ("Server log", engram_home / "logs" / "engram.log"),
        ]
    )
    if config.activation.recall_packet_cache_persistence_enabled:
        paths.extend(
            _sqlite_companion_paths(
                config.get_packet_cache_path(mode),
                label_prefix="Packet cache",
            )
        )
    if config.activation.cue_index_outbox_enabled:
        paths.extend(
            _sqlite_companion_paths(
                config.get_cue_index_outbox_path(mode),
                label_prefix="Cue index outbox",
            )
        )

    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for label, path in paths:
        path = path.expanduser()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(_path_info(label, path))
    return result


def human_bytes(num_bytes: int) -> str:
    """Format bytes for operator display."""
    value = float(max(num_bytes, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _sqlite_companion_paths(sqlite_path: Path, *, label_prefix: str) -> list[tuple[str, Path]]:
    return [
        (label_prefix, sqlite_path),
        (f"{label_prefix} WAL", Path(f"{sqlite_path}-wal")),
        (f"{label_prefix} shared memory", Path(f"{sqlite_path}-shm")),
    ]


def _backend_label(config: EngramConfig, mode: str) -> str:
    if mode == "helix":
        if config.helix.transport == "native":
            return "helix_native"
        return f"helix_{config.helix.transport}"
    if mode == "lite":
        return "sqlite"
    if mode == "full":
        return "falkordb_redis"
    return mode


async def _read_counts(graph_store: Any, group_id: str) -> dict[str, int]:
    stats = await graph_store.get_stats(group_id)
    cue_metrics = stats.get("cue_metrics") or {}
    return {
        "episodes": _int_stat(stats, "episodes", "episode_count", "total_episodes"),
        "entities": _int_stat(stats, "entities", "entity_count", "total_entities"),
        "relationships": _int_stat(
            stats,
            "relationships",
            "relationship_count",
            "total_relationships",
        ),
        "cues": _int_stat(cue_metrics, "cue_count", "cues"),
    }


def _empty_counts() -> dict[str, int]:
    return {
        "episodes": 0,
        "entities": 0,
        "relationships": 0,
        "cues": 0,
    }


def _startup_timeout_seconds() -> float:
    raw = os.environ.get(STARTUP_TIMEOUT_ENV, str(DEFAULT_STARTUP_TIMEOUT_SECONDS))
    try:
        return float(raw)
    except ValueError:
        LOGGER.warning(
            "Invalid %s=%r; using %.1f seconds",
            STARTUP_TIMEOUT_ENV,
            raw,
            DEFAULT_STARTUP_TIMEOUT_SECONDS,
        )
        return DEFAULT_STARTUP_TIMEOUT_SECONDS


def _snapshot_timeout_seconds(timeout_seconds: float | None) -> float:
    if timeout_seconds is None:
        return _startup_timeout_seconds()
    return max(0.0, float(timeout_seconds))


async def _startup_baseline(
    operation: Any,
    *,
    label: str,
    timeout_seconds: float,
    fallback: Any,
) -> Any:
    if timeout_seconds <= 0:
        return await operation
    try:
        return await asyncio.wait_for(operation, timeout=timeout_seconds)
    except TimeoutError:
        LOGGER.warning(
            "%s timed out after %.1f seconds; startup baseline will begin empty",
            label,
            timeout_seconds,
        )
        return fallback()


async def _bounded(operation: Any, *, timeout_seconds: float) -> Any:
    if timeout_seconds <= 0:
        return await operation
    return await asyncio.wait_for(operation, timeout=timeout_seconds)


def _cached_counts(
    counts: dict[str, int],
    *,
    captured_at: float,
    status: str,
) -> _CachedCounts:
    return _CachedCounts(
        counts=_clone_counts(counts),
        captured_at=captured_at,
        status=status,
    )


def _cached_paths(
    paths: list[dict[str, Any]],
    *,
    captured_at: float,
    status: str,
) -> _CachedPaths:
    return _CachedPaths(
        paths=_clone_paths(paths),
        captured_at=captured_at,
        status=status,
    )


def _clone_counts(counts: dict[str, int]) -> dict[str, int]:
    return {
        "episodes": _int_stat(counts, "episodes"),
        "entities": _int_stat(counts, "entities"),
        "relationships": _int_stat(counts, "relationships"),
        "cues": _int_stat(counts, "cues"),
    }


def _clone_paths(paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in paths if isinstance(item, dict)]


def _age_seconds(captured_at: float, *, now: float) -> int:
    return max(0, int(now - captured_at))


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 4)


def _int_stat(payload: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _path_info(label: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    kind = "missing"
    if path.is_dir():
        kind = "directory"
    elif path.is_file():
        kind = "file"
    size = _path_size(path) if exists else 0
    return {
        "label": label,
        "path": str(path),
        "exists": exists,
        "kind": kind,
        "bytes": size,
        "humanSize": human_bytes(size),
    }


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0

    total = 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if not Path(root, name).is_symlink()]
        for filename in files:
            candidate = Path(root, filename)
            try:
                if not candidate.is_symlink():
                    total += candidate.stat().st_size
            except OSError:
                continue
    return total
