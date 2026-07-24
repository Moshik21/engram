"""Backfill deterministic episode cues for episodes captured before the cue layer.

Cue creation is capture-time ONLY (``CaptureService._store_episode_cue``).
Episodes stored before the cue layer shipped have no cue row and nothing ever
creates one, so they are permanently invisible to every cue-side consumer
(cue search, the usage/feedback loop, lifecycle counts). This module is the
missing drain.

Design constraints, each one paid for by a past bug in this codebase:

* SAME DERIVATION. It calls :func:`engram.extraction.cues.build_episode_cue` --
  the exact function capture calls, including its skip rules (empty content,
  ``discourse_class == "system"``). There is no second implementation to drift.
* IDEMPOTENT. Episodes whose cue already exists are skipped by probe, and the
  write itself is an upsert keyed on ``episode_id``.
* BOUNDED + RESUMABLE. Listing is one bounded episode listing -- never
  ``find_cues_by_group``, which takes no server-side limit and measured 20s+ as
  a loop-blocking native call on an 8.7k-cue brain. The walk is oldest-first
  from a persisted ``(created_ts, episode_id)`` cursor that advances only over a
  contiguous prefix of FINAL outcomes, so a failed write is retried next window
  instead of being stranded.
* ZERO LLM CALLS and zero embedding calls. Cue vectors are the existing
  ``backfill_missing_cue_vectors`` drain's job; this only writes the cue row.

What it deliberately does NOT do: touch ``episode.projection_state``. The
derived cue state is SCHEDULED for anything with priority >= 0.55, and blindly
syncing that onto an episode that has already been PROJECTED would demote it
and re-queue thousands of episodes for extraction. Instead, when the episode has
already advanced past the cue stage, the cue row inherits the episode's state so
the two never disagree.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.cues import build_episode_cue
from engram.models.episode import Episode, EpisodeProjectionState

# Shared with the cue-vector drain so both cursors key episodes identically.
from engram.storage.index_completeness import _created_ts

logger = logging.getLogger(__name__)

# Safety cap when listing episodes for the sweep (mirrors the cue-vector drain).
DEFAULT_EPISODE_SCAN_LIMIT = 50_000
# Default write budget per run. The shell must not be held for a 9000-episode
# sweep; the operator runs windows and the cursor resumes.
DEFAULT_BACKFILL_LIMIT = 500

# Episodes past the cue stage: the cue row must inherit their state rather than
# claim a fresh routing decision that the projection side already superseded.
_ADVANCED_PROJECTION_STATES = frozenset(
    {
        EpisodeProjectionState.PROJECTING.value,
        EpisodeProjectionState.PROJECTED.value,
        EpisodeProjectionState.MERGED.value,
        EpisodeProjectionState.FAILED.value,
        EpisodeProjectionState.DEAD_LETTER.value,
    }
)


@dataclass
class CueBackfillResult:
    """Outcome of one bounded backfill window."""

    group_id: str
    scanned: int = 0
    probed: int = 0
    already_cued: int = 0
    skipped_empty_content: int = 0
    skipped_by_policy: int = 0
    state_inherited: int = 0
    would_write: int = 0
    written: int = 0
    failed: int = 0
    dry_run: bool = True
    complete: bool = False
    cursor_next: tuple[float, str] | None = None
    written_ids: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "scanned": self.scanned,
            "probed": self.probed,
            "already_cued": self.already_cued,
            "skipped_empty_content": self.skipped_empty_content,
            "skipped_by_policy": self.skipped_by_policy,
            "state_inherited": self.state_inherited,
            "would_write": self.would_write,
            "written": self.written,
            "failed": self.failed,
            "dry_run": self.dry_run,
            "complete": self.complete,
            "cursor_next": list(self.cursor_next) if self.cursor_next else None,
            "written_ids": list(self.written_ids),
            "duration_ms": round(self.duration_ms, 2),
        }


def _projection_state_value(episode: Episode) -> str | None:
    state = getattr(episode, "projection_state", None)
    if state is None:
        return None
    return state.value if hasattr(state, "value") else str(state)


def build_backfill_cue(episode: Episode, cfg: ActivationConfig):
    """Derive the cue capture would have written, then reconcile projection state.

    Delegates ALL derivation to :func:`build_episode_cue` (the capture-time
    function) and only pins ``projection_state`` when the episode has already
    moved past the cue stage. Returns ``(cue, state_inherited)``; ``cue`` is
    ``None`` exactly when capture would also have written no cue.
    """
    cue = build_episode_cue(episode, cfg)
    if cue is None:
        return None, False
    episode_state = _projection_state_value(episode)
    if episode_state in _ADVANCED_PROJECTION_STATES:
        return cue.model_copy(update={"projection_state": episode_state}), True
    return cue, False


async def backfill_missing_episode_cues(
    graph_store: Any,
    cfg: ActivationConfig,
    group_id: str,
    *,
    limit: int = DEFAULT_BACKFILL_LIMIT,
    apply: bool = False,
    cursor: tuple[float, str] | None = None,
    scan_limit: int = DEFAULT_EPISODE_SCAN_LIMIT,
) -> CueBackfillResult:
    """Write the cue rows capture never got the chance to write.

    ``limit`` bounds the number of cue PROBES this window issues, which also
    caps writes -- a probe is the per-episode cost (~74ms on native) and every
    write is preceded by exactly one probe.

    ``apply=False`` (the default) changes nothing: it probes, derives, and
    reports what it would write.
    """
    started = time.perf_counter()
    result = CueBackfillResult(group_id=group_id, dry_run=not apply)

    get_episodes = getattr(graph_store, "get_episodes", None)
    get_cue = getattr(graph_store, "get_episode_cue", None)
    upsert_cue = getattr(graph_store, "upsert_episode_cue", None)
    if not callable(get_episodes) or not callable(get_cue) or not callable(upsert_cue):
        logger.warning("cue backfill: graph store lacks episode/cue APIs — nothing to do")
        result.duration_ms = (time.perf_counter() - started) * 1000.0
        return result

    episodes = await get_episodes(group_id=group_id, limit=max(1, int(scan_limit))) or []
    candidates = sorted(
        ((_created_ts(getattr(ep, "created_at", None)), str(ep.id)), ep)
        for ep in episodes
        if str(getattr(ep, "id", "") or "")
    )
    if cursor is not None:
        candidates = [(key, ep) for key, ep in candidates if key > cursor]

    max_probes = max(0, int(limit))
    prefix_intact = True
    cursor_candidate: tuple[float, str] | None = None
    exhausted = True

    # NOTE: no soft-delete filter. Episodes have no ``deleted_at`` field (only
    # entities do), so a `getattr(ep, "deleted_at", None)` guard here would be
    # permanently dead code.
    for key, episode in candidates:
        episode_id = key[1]
        if result.probed >= max_probes:
            exhausted = False
            break
        result.scanned += 1
        result.probed += 1
        try:
            existing = await get_cue(episode_id, group_id)
        except Exception:
            result.failed += 1
            prefix_intact = False
            logger.warning(
                "cue backfill: get_episode_cue failed for %s — window stopped early",
                episode_id,
                exc_info=True,
            )
            exhausted = False
            break
        if existing is not None and str(getattr(existing, "cue_text", "") or "").strip():
            result.already_cued += 1
            if prefix_intact:
                cursor_candidate = key
            continue

        cue, state_inherited = build_backfill_cue(episode, cfg)
        if cue is None:
            # build_episode_cue is authoritative about the skip; the split below
            # is a descriptive label for the operator, not a second decision.
            if not str(getattr(episode, "content", "") or "").strip():
                result.skipped_empty_content += 1
            else:
                result.skipped_by_policy += 1
            if prefix_intact:
                cursor_candidate = key
            continue

        if state_inherited:
            result.state_inherited += 1
        if not apply:
            result.would_write += 1
            if prefix_intact:
                cursor_candidate = key
            continue
        try:
            await upsert_cue(cue)
        except Exception:
            result.failed += 1
            prefix_intact = False
            logger.warning("cue backfill: upsert_episode_cue failed for %s", episode_id)
            continue
        result.written += 1
        result.written_ids.append(episode_id)
        if prefix_intact:
            cursor_candidate = key

    # A dry run must never advance the persisted cursor: it did no work, and
    # skipping those episodes on the next apply would strand them silently.
    result.cursor_next = cursor_candidate if apply else None
    result.complete = exhausted and result.failed == 0
    result.duration_ms = (time.perf_counter() - started) * 1000.0
    return result
