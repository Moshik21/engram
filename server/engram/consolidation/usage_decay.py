"""Surfaced-never-used decay (AGENT_EXPERIENCE M4.1, D4 = demotion-first).

P5: non-use is FORGETTING evidence, never ranking evidence. This mop-time
pass finds chronic surfaced-never-used items — surfaced >= N_min, used == 0,
older than the age floor — and writes an offline demotion marker:

- episodes/cues: ``usage_decay_demoted_at`` inside the episode's
  encoding_context JSON blob (the salience_class pattern; no schema change).
  Surfaced evidence is the cue record's surfaced_count + hit_count; used
  evidence is the citation-scan ``usage_used_count`` plus legacy used_count.
- entities: ``usage_decay_demoted_at`` inside the entity attributes JSON.
  Surfaced evidence is the activation store's access_history count; used
  evidence is the tier-weighted usage_events sum (n_eff).

The RANKER MAY NOT READ the marker (P5) — pinned by tests. Sanctioned
consumers only:

1. Prune feed (D4 measurement window): ``prune_feed_eligible`` reports when
   a marker has aged past ``usage_decay_prune_after_days``; episodic tier
   only, identity_core and durable-class exempt. The mop report counts these
   as ``prune_feed_ready``; actual prune-phase consumption lands only after
   the measurement window per D4.
2. ``demote_surfaced_never_used_results`` — recap-penalty-style presenter
   demotion behind ``usage_decay_presenter_demotion_enabled`` (default
   False). EVAL-GATED: flipping the flag to True requires an
   agent-experience battery pass + a continuity PASS (B1/B3/B5). Off, it
   returns the results object unchanged (byte identity).

Bounded per window (``usage_decay_max_per_window`` demotions), cursor-driven
(persisted by hygiene_ops beside the vector-backfill cursors), wall-clock
bounded via ``deadline_ts``, dry-run aware.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from engram.extraction.promotion import is_durable_recall_entity_type

logger = logging.getLogger(__name__)

# Marker keys. The demoted_at key must appear ONLY in this module (static
# boundary test); everything else goes through the helpers below.
USAGE_DECAY_MARKER = "usage_decay_demoted_at"
USAGE_DECAY_SURFACED = "usage_decay_surfaced"

# Bounded episode listing per window (the drain pattern: sort + cursor makes
# progression durable even when one window cannot cover the corpus).
_EPISODE_SCAN_LIMIT = 2000


def _ts(value: Any) -> float:
    """Best-effort epoch seconds from a datetime/ISO string/number."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        try:
            return value.timestamp()
        except (OverflowError, OSError, ValueError):  # silent-ok: cursor key fallback, epoch 0
            return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:  # silent-ok: unparseable marker/created_at reads as epoch 0
        return 0.0


def is_chronic_non_use(
    surfaced: float,
    used: float,
    age_days: float,
    *,
    min_surfaced: int,
    min_age_days: float,
) -> bool:
    """Chronic non-use: surfaced >= N_min AND used == 0 AND age > floor."""
    return surfaced >= min_surfaced and used <= 0.0 and age_days > min_age_days


def decode_usage_decay_marker(encoding_context: str | None) -> str:
    """Read a persisted demotion marker out of the encoding_context blob."""
    if not encoding_context:
        return ""
    try:
        data = json.loads(encoding_context)
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    value = data.get(USAGE_DECAY_MARKER, "")
    return value if isinstance(value, str) else ""


def encode_usage_decay_marker(
    encoding_context: str | None,
    demoted_at: str,
    surfaced: int,
) -> str | None:
    """Embed the demotion marker into the encoding_context JSON blob.

    Returns None when the existing encoding_context is not a JSON object
    (e.g. reflect cluster keys) — the episode is unmarkable and is skipped.
    """
    if encoding_context:
        try:
            data = json.loads(encoding_context)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
    else:
        data = {}
    data[USAGE_DECAY_MARKER] = demoted_at
    data[USAGE_DECAY_SURFACED] = int(surfaced)
    return json.dumps(data)


def prune_feed_eligible(
    demoted_at: str,
    *,
    now: float,
    cfg: Any,
    memory_tier: str = "episodic",
    identity_core: bool = False,
    entity_type: str | None = None,
) -> bool:
    """D4 prune feed: marker aged past the measurement window makes an item
    eligible for prune-phase consumption. Episodic tier only; identity_core
    and durable-class entities are never eligible."""
    if not demoted_at or identity_core:
        return False
    if (memory_tier or "episodic") != "episodic":
        return False
    if entity_type and is_durable_recall_entity_type(entity_type):
        return False
    marker_ts = _ts(demoted_at)
    if marker_ts <= 0.0:
        return False
    window_days = float(getattr(cfg, "usage_decay_prune_after_days", 30.0) or 30.0)
    return (now - marker_ts) / 86400.0 >= window_days


def _result_carries_marker(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    entity = result.get("entity")
    if isinstance(entity, dict):
        attrs = entity.get("attributes")
        if isinstance(attrs, dict) and attrs.get(USAGE_DECAY_MARKER):
            return True
    episode = result.get("episode")
    if isinstance(episode, dict) and decode_usage_decay_marker(episode.get("encoding_context")):
        return True
    return False


def demote_surfaced_never_used_results(results: list[dict], cfg: Any) -> list[dict]:
    """Recap-penalty-style presenter demotion of marker-carrying results.

    EVAL-GATED, default OFF (``usage_decay_presenter_demotion_enabled``):
    flipping it to True requires an agent-experience battery pass AND a
    continuity PASS first. Off, the input list object is returned unchanged
    (byte identity on every read path). On, marker-carrying results keep
    their relative order but sort after every unmarked result. The scorer
    never reads the marker at any flag state (P5)."""
    if not getattr(cfg, "usage_decay_presenter_demotion_enabled", False):
        return results
    demoted = [r for r in results if _result_carries_marker(r)]
    if not demoted:
        return results
    kept = [r for r in results if not _result_carries_marker(r)]
    return kept + demoted


@dataclass
class UsageDecayResult:
    scanned_episodes: int = 0
    scanned_entities: int = 0
    demoted_episodes: int = 0
    demoted_entities: int = 0
    exempt: int = 0
    already_marked: int = 0
    prune_feed_ready: int = 0
    errors: int = 0
    dry_run: bool = False
    budget: int = 0
    cursors_next: dict[str, Any] = field(default_factory=dict)

    @property
    def demoted_total(self) -> int:
        return self.demoted_episodes + self.demoted_entities

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": {
                "episodes": self.scanned_episodes,
                "entities": self.scanned_entities,
            },
            "demoted": {
                "episodes": self.demoted_episodes,
                "entities": self.demoted_entities,
            },
            "exempt": self.exempt,
            "already_marked": self.already_marked,
            "prune_feed_ready": self.prune_feed_ready,
            "errors": self.errors,
            "dry_run": self.dry_run,
            "budget": self.budget,
        }


def _episode_cursor(value: Any) -> tuple[float, str] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (float(value[0]), str(value[1]))
        except (TypeError, ValueError):
            return None
    return None


async def run_usage_decay(
    graph_store: Any,
    activation_store: Any,
    group_id: str,
    *,
    cfg: Any,
    dry_run: bool = False,
    cursors: dict[str, Any] | None = None,
    deadline_ts: float | None = None,
    now: float | None = None,
) -> UsageDecayResult:
    """Run one bounded usage-decay window. Returns counts + next cursors."""
    now = now if now is not None else time.time()
    min_surfaced = int(getattr(cfg, "usage_decay_min_surfaced", 12) or 12)
    min_age_days = float(getattr(cfg, "usage_decay_min_age_days", 14.0) or 14.0)
    budget = int(getattr(cfg, "usage_decay_max_per_window", 100) or 100)
    result = UsageDecayResult(dry_run=bool(dry_run), budget=budget)
    cursors = dict(cursors or {})
    demoted_at_iso = datetime.fromtimestamp(now, tz=UTC).isoformat()

    def _out_of_budget() -> bool:
        if result.demoted_total >= budget:
            return True
        return deadline_ts is not None and time.monotonic() >= deadline_ts

    # ── Episodes/cues ────────────────────────────────────────────────
    # The bulk cue listing is documented loop-blocking on native (20s+);
    # budgeted work lists episodes and probes cues by id instead.
    get_episodes = getattr(graph_store, "get_episodes", None)
    get_cue = getattr(graph_store, "get_episode_cue", None)
    update_episode = getattr(graph_store, "update_episode", None)
    if callable(get_episodes) and callable(get_cue) and callable(update_episode):
        episodes = await get_episodes(group_id=group_id, limit=_EPISODE_SCAN_LIMIT) or []
        keyed = sorted(
            ((_ts(getattr(ep, "created_at", None)), str(ep.id)), ep)
            for ep in episodes
            if str(getattr(ep, "id", "") or "") and getattr(ep, "deleted_at", None) is None
        )
        ep_cursor = _episode_cursor(cursors.get("episodes"))
        remaining = [pair for pair in keyed if pair[0] > ep_cursor] if ep_cursor else keyed
        if ep_cursor and not remaining:
            remaining = keyed  # full sweep done — wrap around
        for key, ep in remaining:
            if _out_of_budget():
                break
            result.cursors_next["episodes"] = list(key)
            result.scanned_episodes += 1
            if str(getattr(ep, "memory_tier", "episodic") or "episodic") != "episodic":
                result.exempt += 1
                continue
            encoding_context = getattr(ep, "encoding_context", None)
            marker = decode_usage_decay_marker(encoding_context)
            if marker:
                result.already_marked += 1
                if prune_feed_eligible(marker, now=now, cfg=cfg):
                    result.prune_feed_ready += 1
                continue
            cue = await get_cue(str(ep.id), group_id)
            if cue is None:
                continue
            surfaced = int(getattr(cue, "surfaced_count", 0) or 0) + int(
                getattr(cue, "hit_count", 0) or 0
            )
            used = float(getattr(cue, "usage_used_count", 0.0) or 0.0) + float(
                getattr(cue, "used_count", 0) or 0
            )
            age_days = (now - key[0]) / 86400.0 if key[0] > 0 else 0.0
            if not is_chronic_non_use(
                surfaced, used, age_days, min_surfaced=min_surfaced, min_age_days=min_age_days
            ):
                continue
            if dry_run:
                result.demoted_episodes += 1
                continue
            encoded = encode_usage_decay_marker(encoding_context, demoted_at_iso, surfaced)
            if encoded is None:
                result.exempt += 1  # opaque encoding_context — unmarkable
                continue
            try:
                await update_episode(str(ep.id), {"encoding_context": encoded}, group_id)
                result.demoted_episodes += 1
            except Exception:
                result.errors += 1
                logger.warning("usage-decay episode demote failed for %s", ep.id, exc_info=True)

    # ── Entities ─────────────────────────────────────────────────────
    # Ratio source is the activation store: access_history (surfaced events,
    # all tiers) vs usage_events (used tiers, n_eff). Guarded private access
    # — mirrors snapshot_to_graph's ownership of the same state.
    states = getattr(activation_store, "_states", None)
    group_map = getattr(activation_store, "_group_map", None)
    get_entity = getattr(graph_store, "get_entity", None)
    update_entity = getattr(graph_store, "update_entity", None)
    if isinstance(states, dict) and callable(get_entity) and callable(update_entity):
        ids = sorted(
            eid
            for eid in states
            if not isinstance(group_map, dict) or group_map.get(eid) == group_id
        )
        ent_cursor = cursors.get("entities")
        ent_cursor = str(ent_cursor) if isinstance(ent_cursor, str) and ent_cursor else None
        remaining_ids = [eid for eid in ids if eid > ent_cursor] if ent_cursor else ids
        if ent_cursor and not remaining_ids:
            remaining_ids = ids  # full sweep done — wrap around
        for eid in remaining_ids:
            if _out_of_budget():
                break
            result.cursors_next["entities"] = eid
            state = states.get(eid)
            if state is None:
                continue
            result.scanned_entities += 1
            surfaced = int(getattr(state, "access_count", 0) or 0)
            used = float(getattr(state, "n_eff", 0.0) or 0.0)
            if surfaced < min_surfaced or used > 0.0:
                continue
            try:
                entity = await get_entity(eid, group_id)
            except Exception:
                result.errors += 1
                logger.warning("usage-decay entity fetch failed for %s", eid, exc_info=True)
                continue
            if entity is None:
                continue
            if (
                bool(getattr(entity, "identity_core", False))
                or str(getattr(entity, "mat_tier", "episodic") or "episodic") != "episodic"
                or is_durable_recall_entity_type(getattr(entity, "entity_type", None))
            ):
                result.exempt += 1
                continue
            attrs_raw = getattr(entity, "attributes", None)
            attrs = dict(attrs_raw) if isinstance(attrs_raw, dict) else {}
            marker = attrs.get(USAGE_DECAY_MARKER)
            if isinstance(marker, str) and marker:
                result.already_marked += 1
                if prune_feed_eligible(
                    marker,
                    now=now,
                    cfg=cfg,
                    memory_tier=str(getattr(entity, "mat_tier", "episodic") or "episodic"),
                    identity_core=bool(getattr(entity, "identity_core", False)),
                    entity_type=getattr(entity, "entity_type", None),
                ):
                    result.prune_feed_ready += 1
                continue
            age_days = (now - _ts(getattr(entity, "created_at", None))) / 86400.0
            if not is_chronic_non_use(
                surfaced, used, age_days, min_surfaced=min_surfaced, min_age_days=min_age_days
            ):
                continue
            if dry_run:
                result.demoted_entities += 1
                continue
            attrs[USAGE_DECAY_MARKER] = demoted_at_iso
            attrs[USAGE_DECAY_SURFACED] = surfaced
            try:
                # JSON string: the common denominator every backend's
                # update_entity accepts for the attributes column.
                await update_entity(eid, {"attributes": json.dumps(attrs)}, group_id)
                result.demoted_entities += 1
            except Exception:
                result.errors += 1
                logger.warning("usage-decay entity demote failed for %s", eid, exc_info=True)

    if result.demoted_total:
        logger.info(
            "Usage decay: demoted episodes=%d entities=%d exempt=%d "
            "already_marked=%d prune_feed_ready=%d dry_run=%s",
            result.demoted_episodes,
            result.demoted_entities,
            result.exempt,
            result.already_marked,
            result.prune_feed_ready,
            dry_run,
        )
    return result
