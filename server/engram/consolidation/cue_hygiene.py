"""Budgeted cue hygiene: demote/drop never-used latent cues.

Local-only, no external LLM. Clears cue_text on eligible cues so they stop
polluting hybrid/cue search while leaving the episode record intact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from engram.consolidation.hygiene_debt import (
    CueHygieneCandidate,
    select_cue_hygiene_candidates,
)

logger = logging.getLogger(__name__)


def _age_days(raw: Any, *, now: float) -> float | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, (int, float)):
        ts = float(raw)
        # Heuristic: ms vs seconds
        if ts > 1e12:
            ts = ts / 1000.0
        return max(0.0, (now - ts) / 86400.0)
    text = str(raw).strip()
    if not text:
        return None
    try:
        from datetime import datetime

        # ISO with optional Z
        cleaned = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return max(0.0, (now - dt.timestamp()) / 86400.0)
    except Exception:
        return None


@dataclass
class CueHygieneResult:
    scanned: int = 0
    eligible: int = 0
    demoted: int = 0
    errors: int = 0
    dry_run: bool = False
    demoted_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": self.scanned,
            "eligible": self.eligible,
            "demoted": self.demoted,
            "errors": self.errors,
            "dry_run": self.dry_run,
            "demoted_ids": list(self.demoted_ids or []),
        }


async def run_cue_hygiene(
    graph_store: Any,
    group_id: str,
    *,
    max_per_cycle: int = 200,
    min_age_days: float = 14.0,
    dry_run: bool = False,
) -> CueHygieneResult:
    """Demote never-hit cues older than *min_age_days* (clear cue_text)."""
    result = CueHygieneResult(dry_run=dry_run, demoted_ids=[])
    fetch = getattr(graph_store, "_fetch_episode_cues_bulk", None)
    if not callable(fetch):
        # Fallback: no bulk API — nothing to do
        return result

    try:
        rows = await fetch(group_id)
    except Exception:
        logger.debug("cue hygiene bulk fetch failed", exc_info=True)
        return result

    if not rows:
        return result

    now = time.time()
    candidates_in: list[CueHygieneCandidate] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        result.scanned += 1
        created = row.get("created_at") or row.get("updated_at") or ""
        age_days = _age_days(created, now=now)
        if age_days is None:
            # No parseable timestamp — only touch zero-hit cues as old enough
            age_days = min_age_days + 1.0 if int(row.get("hit_count") or 0) == 0 else 0.0

        candidates_in.append(
            CueHygieneCandidate(
                episode_id=str(row.get("episode_id") or ""),
                hit_count=int(row.get("hit_count") or 0),
                surfaced_count=int(row.get("surfaced_count") or 0),
                cue_text=str(row.get("cue_text") or ""),
                age_days=age_days,
            )
        )

    eligible = select_cue_hygiene_candidates(
        candidates_in,
        max_age_days=min_age_days,
        max_hit_count=0,
        max_surfaced_count=0,
        limit=max_per_cycle,
    )
    result.eligible = len(eligible)

    update = getattr(graph_store, "update_episode_cue", None)
    upsert = getattr(graph_store, "upsert_episode_cue", None)

    for cue in eligible:
        if dry_run:
            result.demoted += 1
            result.demoted_ids.append(cue.episode_id)
            continue
        try:
            if callable(update):
                await update(
                    cue.episode_id,
                    {"cue_text": ""},
                    group_id,
                )
            elif callable(upsert):
                from engram.models.episode_cue import EpisodeCue

                await upsert(
                    EpisodeCue(
                        episode_id=cue.episode_id,
                        group_id=group_id,
                        cue_text="",
                        hit_count=cue.hit_count,
                        surfaced_count=cue.surfaced_count,
                    )
                )
            else:
                result.errors += 1
                continue
            result.demoted += 1
            result.demoted_ids.append(cue.episode_id)
        except Exception:
            result.errors += 1
            logger.debug("cue hygiene demote failed for %s", cue.episode_id, exc_info=True)

    if result.demoted:
        logger.info(
            "Cue hygiene: demoted=%d eligible=%d scanned=%d dry_run=%s",
            result.demoted,
            result.eligible,
            result.scanned,
            dry_run,
        )
    return result
