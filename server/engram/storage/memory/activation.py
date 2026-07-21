"""In-memory activation store for lite mode."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from engram.config import ActivationConfig
from engram.models.activation import DEFAULT_USAGE_TIER_WEIGHTS, ActivationState

logger = logging.getLogger(__name__)

SNAPSHOT_VERSION = 2

# Tiers whose loss is not acceptable (rare, explicit user signal): these are
# write-through journaled at record time so a crash before snapshot save, or a
# non-owning process (MCP stdio while the shell is alive, brain), cannot lose
# them. See RF goal M1.3.
_JOURNALED_TIERS = frozenset({"confirmed", "corrected"})

USAGE_JOURNAL_FILENAME = "activation-usage-journal.jsonl"


def activation_snapshot_path() -> Path:
    """Where the shell persists the ACT-R activation snapshot at shutdown.

    The shell (engram serve) owns writes to this file; other runtimes
    (brain, one-shot CLI) load it read-only so a stale save can never
    clobber a newer shell save.
    """
    home = Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()
    return home / "activation-snapshot.json"


def activation_usage_journal_path() -> Path:
    """Append-only JSONL journal for confirmed/corrected usage events.

    Lives beside the snapshot. Any process may append; ONLY the process that
    is allowed to write the snapshot (per the mcp/server.py ownership rules)
    may truncate it, and only after folding its contents into the snapshot.
    """
    return activation_snapshot_path().parent / USAGE_JOURNAL_FILENAME


class MemoryActivationStore:
    """Dict-backed activation state. Acceptable for personal-scale graphs."""

    def __init__(
        self,
        cfg: ActivationConfig | None = None,
        journal_path: Path | None = None,
    ) -> None:
        self._states: dict[str, ActivationState] = {}
        self._group_map: dict[str, str] = {}
        self._lock = asyncio.Lock()
        self._cfg = cfg or ActivationConfig()
        # Resolved lazily so tests that point ENGRAM_HOME/HOME at a scratch
        # dir after construction still journal into the scratch dir.
        self._journal_path_override = journal_path

    @property
    def journal_path(self) -> Path:
        """Journal target for record-time appends (write-through)."""
        return self._journal_path_override or activation_usage_journal_path()

    def _journal_path_for(self, snapshot_path: Path) -> Path:
        """Journal beside a given snapshot file — the fold/replay target.

        The journal lives in the same dir as the snapshot, so save/load scope
        their fold/replay to THAT dir. In production both resolve to
        ~/.engram; a save to a nonstandard dir (backup, test tmp) then never
        folds or truncates the live journal.
        """
        return self._journal_path_override or snapshot_path.parent / USAGE_JOURNAL_FILENAME

    async def get_activation(self, entity_id: str) -> ActivationState | None:
        return self._states.get(entity_id)

    async def set_activation(self, entity_id: str, state: ActivationState) -> None:
        self._states[entity_id] = state

    async def batch_get(self, entity_ids: list[str]) -> dict[str, ActivationState]:
        return {eid: self._states[eid] for eid in entity_ids if eid in self._states}

    async def batch_set(self, states: dict[str, ActivationState]) -> None:
        self._states.update(states)

    def _tier_weight(self, tier: str) -> float:
        weights = self._cfg.usage_tier_weights or DEFAULT_USAGE_TIER_WEIGHTS
        try:
            return float(weights[tier])
        except KeyError:
            raise ValueError(f"Unknown usage tier: {tier!r} (known: {sorted(weights)})") from None

    async def record_access(
        self,
        entity_id: str,
        timestamp: float,
        group_id: str | None = None,
        tier: str = "surfaced",
    ) -> None:
        """Record an access event for an entity, creating state if needed.

        Every tier appends to ``access_history`` (hygiene: prune/mature/B).
        Nonzero-weight tiers additionally append ``(ts, weight)`` to
        ``usage_events`` (the ranking-side store — inert until M2), and
        confirmed/corrected events are write-through journaled (M1.3).
        """
        from engram.activation.engine import record_access as _record_access

        state = self._states.get(entity_id)
        if state is None:
            state = ActivationState(node_id=entity_id)
            self._states[entity_id] = state
        _record_access(state, timestamp, self._cfg)
        weight = self._tier_weight(tier)
        if weight > 0.0:
            state.record_usage_event(timestamp, weight)
            if tier in _JOURNALED_TIERS:
                self._journal_append(entity_id, timestamp, weight, group_id)
        if group_id:
            self._group_map[entity_id] = group_id

    async def clear_activation(self, entity_id: str) -> None:
        """Remove all activation state for an entity."""
        self._states.pop(entity_id, None)
        self._group_map.pop(entity_id, None)

    async def get_top_activated(
        self,
        group_id: str | None = None,
        limit: int = 20,
        now: float | None = None,
    ) -> list[tuple[str, ActivationState]]:
        import time

        from engram.activation.engine import compute_activation

        now = now if now is not None else time.time()
        scored = []
        for eid, state in self._states.items():
            if group_id and self._group_map.get(eid) != group_id:
                continue
            act = compute_activation(
                state.access_history,
                now,
                self._cfg,
                state.consolidated_strength,
            )
            scored.append((eid, state, act))
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(eid, state) for eid, state, _ in scored[:limit]]

    # ── Confirmed-event journal (M1.3) ──────────────────────────────

    def _journal_append(
        self,
        entity_id: str,
        ts: float,
        weight: float,
        group_id: str | None,
    ) -> None:
        """Write-through one confirmed/corrected event as a single JSONL line."""
        line = json.dumps(
            {"ts": ts, "weight": weight, "entity_id": entity_id, "group_id": group_id}
        )
        path = self.journal_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # One O_APPEND write per event: lines stay whole even with
            # concurrent appenders (shell + MCP stdio + brain).
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            # silent-ok: journaling is durability insurance; the event is
            # already live in RAM and the failure is logged with a traceback.
            logger.warning("Usage journal append failed: %s", path, exc_info=True)

    def _replay_journal_lines(self, lines: list[str]) -> int:
        """Idempotently apply journal lines to in-RAM state.

        Exact (ts, weight) membership in ``usage_events`` is the dedup guard:
        an event already folded (own write, snapshot, or prior replay) is
        skipped entirely, so replay never double-counts usage OR hygiene.
        """
        applied = 0
        for raw in lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
                ts = float(rec["ts"])
                weight = float(rec["weight"])
                entity_id = str(rec["entity_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                # silent-ok: skip a single malformed journal line; the rest replay.
                continue
            group_id = rec.get("group_id")
            state = self._states.get(entity_id)
            if state is None:
                state = ActivationState(node_id=entity_id)
                self._states[entity_id] = state
            if state.has_usage_event(ts, weight):
                continue
            state.record_usage_event(ts, weight)
            # Mirror the hygiene append the originating process performed.
            state.access_history.append(ts)
            state.access_count += 1
            if ts > state.last_accessed:
                state.last_accessed = ts
            if len(state.access_history) > self._cfg.max_history_size:
                state.access_history = state.access_history[-self._cfg.max_history_size :]
            if group_id and entity_id not in self._group_map:
                self._group_map[entity_id] = str(group_id)
            applied += 1
        return applied

    def _fold_journal(self, journal: Path) -> int:
        """(a) Replay the entire current journal into RAM pre-snapshot.

        Returns the byte offset consumed — only up to the last complete line,
        so a line mid-append at read time is left for the fresh segment.
        """
        try:
            raw = journal.read_bytes()
        except OSError:
            # silent-ok: no journal yet — nothing to fold.
            return 0
        cut = raw.rfind(b"\n") + 1
        if cut:
            self._replay_journal_lines(raw[:cut].decode("utf-8", errors="replace").splitlines())
        return cut

    def _compact_journal(self, journal: Path, folded_bytes: int) -> None:
        """(c) Truncate via rename-and-recreate on the captured old inode.

        Lines appended after the fold read (concurrent appenders during the
        snapshot write) are NOT in the snapshot: re-scan them from the renamed
        old inode into a fresh segment at the journal path (and into RAM).
        Appenders open per-line, so post-rename appends land directly in the
        fresh segment.
        """
        old = journal.with_name(journal.name + ".compacting")
        try:
            os.rename(journal, old)
        except OSError:
            # silent-ok: no journal file to compact.
            return
        try:
            tail = old.read_bytes()[folded_bytes:]
            if tail:
                text = tail.decode("utf-8", errors="replace")
                self._replay_journal_lines(text.splitlines())
                with open(journal, "a", encoding="utf-8") as fh:
                    fh.write(text if text.endswith("\n") else text + "\n")
            os.unlink(old)
        except OSError:
            # silent-ok: leave the .compacting segment behind rather than lose
            # events; it is inert (never replayed) and logged for diagnosis.
            logger.warning("Usage journal compaction failed: %s", journal, exc_info=True)

    # ── Snapshot persistence ────────────────────────────────────────

    def save_to_file(self, path: Path) -> int:
        """Persist activation states (incl. access_history) across restarts.

        ACT-R activation is computed from access_history, which lives only in
        this dict — without persistence every shell restart (12+/day under
        the 2h brain cadence) silently wiped all recency/frequency signal.

        Fold-then-compact (M1.3): only the snapshot-owning process calls this
        (mcp/server.py / main.py ownership rules). It (a) replays the whole
        current journal into RAM so the snapshot is a superset of every
        journaled event, (b) writes the v2 snapshot, (c) compacts the journal,
        preserving lines appended during the write. Non-owners append to the
        journal but never call save_to_file, so they never truncate.
        """
        journal = self._journal_path_for(path)
        folded_bytes = self._fold_journal(journal)
        states = {}
        for eid, state in list(self._states.items())[:50000]:
            entry = asdict(state)
            entry["group_id"] = self._group_map.get(eid)
            states[eid] = entry
        payload = {"version": SNAPSHOT_VERSION, "saved_at": time.time(), "states": states}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            # silent-ok: best-effort shutdown snapshot; failure is logged with a
            # traceback and returning 0 avoids aborting the rest of shutdown
            # cleanup. The journal is NOT compacted on a failed save.
            logger.warning("Activation snapshot write failed: %s", path, exc_info=True)
            return 0
        self._compact_journal(journal, folded_bytes)
        return len(states)

    def load_from_file(self, path: Path, max_age_days: float = 14.0) -> int:
        """Restore a prior snapshot; stale snapshots are ignored.

        The confirmed-event journal replays AFTER the snapshot (even when the
        snapshot is absent or stale): explicit user signal is exempt from the
        14-day age-out. Replay is append-only — non-owners never truncate.
        """
        loaded = self._load_snapshot(path, max_age_days)
        journal = self._journal_path_for(path)
        try:
            raw = journal.read_bytes()
        except OSError:
            # silent-ok: no journal to replay.
            return loaded
        replayed = self._replay_journal_lines(raw.decode("utf-8", errors="replace").splitlines())
        if replayed:
            logger.info("Replayed %d usage-journal events: %s", replayed, journal)
        return loaded

    def _load_snapshot(self, path: Path, max_age_days: float) -> int:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # silent-ok: no readable snapshot to restore; start empty (next save rewrites it).
            return 0
        saved_at = payload.get("saved_at")
        if not isinstance(saved_at, (int, float)):
            return 0
        if (time.time() - float(saved_at)) > max_age_days * 86400.0:
            return 0
        loaded = 0
        for eid, entry in (payload.get("states") or {}).items():
            if eid in self._states:
                continue  # live state wins over the snapshot
            group_id = entry.pop("group_id", None)
            try:
                # v1 snapshots have no usage_events (.get defaults to empty);
                # caches are recomputed from the events rather than trusted.
                usage_events = [
                    (float(ts), float(weight)) for ts, weight in entry.get("usage_events") or []
                ]
                state = ActivationState(
                    node_id=entry.get("node_id") or eid,
                    access_history=[float(t) for t in entry.get("access_history") or []],
                    last_accessed=float(entry.get("last_accessed") or 0.0),
                    access_count=int(entry.get("access_count") or 0),
                    consolidated_strength=float(entry.get("consolidated_strength") or 0.0),
                    last_compacted=float(entry.get("last_compacted") or 0.0),
                    ts_alpha=float(entry.get("ts_alpha") or 1.0),
                    ts_beta=float(entry.get("ts_beta") or 1.0),
                    usage_events=usage_events,
                    usage_weight_sum=sum(weight for _, weight in usage_events),
                    usage_last_ts=max((ts for ts, _ in usage_events), default=0.0),
                )
            except (TypeError, ValueError):
                # silent-ok: skip a single malformed snapshot entry; others still restore.
                continue
            self._states[eid] = state
            if group_id:
                self._group_map[eid] = group_id
            loaded += 1
        return loaded

    async def snapshot_to_graph(self, graph_store) -> None:
        """Persist current activation state to graph entity rows."""
        for eid, state in self._states.items():
            group_id = self._group_map.get(eid, "default")
            last_accessed = (
                datetime.fromtimestamp(state.last_accessed, tz=timezone.utc).replace(tzinfo=None)
                if state.last_accessed
                else None
            )
            await graph_store.update_entity(
                eid,
                {
                    "access_count": state.access_count,
                    "last_accessed": last_accessed,
                },
                group_id=group_id,
            )
