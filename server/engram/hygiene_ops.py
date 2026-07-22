"""Shared hygiene mop operations (CLI + cold brain).

Not a second consolidator — the same local drains as warm evidence hygiene,
plus the bounded adjudication/replay passes that give the deferred and
cue_only queues an actual consumer under mop-only scheduling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# RF M4.2: per-window budget for syncing activation counters onto graph rows.
_ACTIVATION_SNAPSHOT_SYNC_MAX = 5000

# Per-window budgets for the durable vector-debt drain. Episode vectors are
# written only at projection and cue capture-indexing is best-effort, so
# shell+quiet installs regrow vector debt forever without this drain (deep
# recall autopsy 2026-07: 1/8876 episode vectors brain-wide). index_episode
# measured ~0.21 eps/s on real dogfood episodes (full-content embed + chunk
# embeds), so 50 episodes ≈ 4 min of the 2h window — capacity ~600/day
# against ~50/day organic inflow. Cues index at ~5/s; 400 ≈ 80s.
_VECTOR_BACKFILL_EPISODES_MAX = 50
_VECTOR_BACKFILL_CUES_MAX = 400
# Wall-clock bounds per drain (first live mop ground past the runner deadline
# on compounding 20s native listing timeouts; see the timeout handler below).
_VECTOR_BACKFILL_EPISODES_SECONDS_MAX = 600.0
_VECTOR_BACKFILL_CUES_SECONDS_MAX = 240.0


def _hygiene_state_path() -> Path:
    home = Path(os.environ.get("ENGRAM_HOME", Path.home() / ".engram")).expanduser()
    return home / "hygiene-state.json"


def _read_hygiene_state() -> dict[str, Any]:
    path = _hygiene_state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_hygiene_state(state: dict[str, Any]) -> None:
    path = _hygiene_state_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state) + "\n", encoding="utf-8")
    except OSError:
        logger.debug("hygiene state write failed", exc_info=True)


def cue_scan_due(now: float | None = None, interval_hours: float | None = None) -> bool:
    """Whether a full cue-hygiene scan is due (watermarked, default daily).

    Cues become eligible by AGE (min_age_days), so scanning all of them every
    2h window is pure waste — one full scan per day tracks eligibility fine.
    """
    if interval_hours is None:
        raw = os.environ.get("ENGRAM_CUE_SCAN_INTERVAL_HOURS", "24")
        try:
            interval_hours = float(raw)
        except ValueError:
            interval_hours = 24.0
    if interval_hours <= 0:
        return True
    last = _read_hygiene_state().get("last_cue_scan_at")
    if not isinstance(last, (int, float)):
        return True
    now = now if now is not None else time.time()
    return (now - float(last)) >= interval_hours * 3600.0


def mark_cue_scan_done(now: float | None = None) -> None:
    state = _read_hygiene_state()
    state["last_cue_scan_at"] = now if now is not None else time.time()
    _write_hygiene_state(state)


async def execute_hygiene_mop(
    *,
    graph_store: Any,
    activation_store: Any,
    search_index: Any,
    activation_cfg: Any,
    group_id: str,
    budget: int = 200,
    dry_run: bool = False,
    loop_adj: Any | None = None,
    graph_manager: Any | None = None,
    extractor: Any | None = None,
    skip_when_no_work: bool = False,
) -> dict[str, Any]:
    """Run bounded debt drains (+ adjudication/replay when a manager/extractor
    is provided). Returns mop stats + debt_before/after."""
    from engram.consolidation.cue_hygiene import run_cue_hygiene
    from engram.consolidation.evidence_drain import (
        load_deferred_evidence,
        reject_evidence_rows,
        reject_junk_evidence,
        scaled_drain_budget,
        select_redundant_entity_evidence,
        select_stale_low_value_evidence,
    )
    from engram.consolidation.hygiene_debt import (
        collect_hygiene_debt_from_store,
        debt_pressure_contribution,
        debt_should_trigger_mop,
    )
    from engram.loop_adjustment import mop_knob_budgets

    debt = await collect_hygiene_debt_from_store(graph_store, group_id)
    debt_pressure = debt_pressure_contribution(debt)
    should_mop = debt_should_trigger_mop(
        debt,
        pressure_threshold=float(activation_cfg.consolidation_pressure_threshold),
    )
    report: dict[str, Any] = {
        "group_id": group_id,
        "debt": debt.to_dict(),
        "pressure": {
            # Event-bus pressure lives in the shell's accumulator; a fresh
            # process cannot see it, so report debt honestly instead of a
            # freshly-constructed ~0 that pretends to be a live reading.
            "event_bus": None,
            "hygiene_debt": round(debt_pressure, 2),
            "total": round(debt_pressure, 2),
            "threshold": activation_cfg.consolidation_pressure_threshold,
            "should_trigger_mop": should_mop,
        },
    }

    if skip_when_no_work and not should_mop:
        report["mop"] = {"skipped": True, "reason": "no actionable hygiene work"}
        report["debt_after"] = debt.to_dict()
        return report

    cli_floor = max(1, int(budget))
    knob_budgets = mop_knob_budgets(cli_floor, loop_adj)
    evidence_budget = knob_budgets["evidence_drain"]
    already_budget = knob_budgets["already_exists"]
    stale_budget = knob_budgets["stale_reject"]
    cue_budget = knob_budgets["cue_hygiene"]
    mop: dict[str, Any] = {
        "dry_run": bool(dry_run),
        "budget": cli_floor,
        "budgets": dict(knob_budgets),
    }

    deferred = await load_deferred_evidence(graph_store, group_id)
    drain_budget = scaled_drain_budget(
        len(deferred),
        base_budget=evidence_budget,
        max_budget=max(
            evidence_budget,
            int(getattr(activation_cfg, "consolidation_evidence_drain_max_budget", 5000) or 5000),
        ),
    )
    mop["evidence_drain"] = await reject_junk_evidence(
        graph_store,
        group_id=group_id,
        rows=deferred,
        dry_run=bool(dry_run),
        batch_size=min(200, drain_budget),
        prioritize_junk=True,
        max_reject=drain_budget,
    )
    if not dry_run and mop["evidence_drain"].get("rejected"):
        deferred = await load_deferred_evidence(graph_store, group_id)

    existing: set[str] = set()
    find_candidates = getattr(graph_store, "find_entity_candidates", None)
    if callable(find_candidates):
        from engram.consolidation.evidence_drain import _entity_name

        names = {_entity_name(r) for r in deferred if str(r.get("fact_class") or "") == "entity"}
        names.discard("")
        for name in list(names)[:500]:
            try:
                cands = await find_candidates(name, group_id=group_id, limit=5)
            except TypeError:
                try:
                    cands = await find_candidates(name, group_id)
                except Exception:
                    continue
            except Exception:
                continue
            for cand in cands or []:
                cand_name = getattr(cand, "name", None) or (
                    cand.get("name") if isinstance(cand, dict) else None
                )
                if cand_name and str(cand_name).casefold() == name.casefold():
                    existing.add(name)
                    break
    redundant = select_redundant_entity_evidence(deferred, existing, limit=already_budget)
    mop["already_exists"] = await reject_evidence_rows(
        graph_store,
        group_id=group_id,
        rows=redundant,
        reason_prefix="drain_already_exists",
        dry_run=bool(dry_run),
        reason_for_row=lambda _r: "entity_name_present",
    )
    if not dry_run and mop["already_exists"].get("rejected"):
        deferred = await load_deferred_evidence(graph_store, group_id)

    # Recovery: large deferred queues are mostly fresh narrow-extractor sludge
    # (deferred_cycles stays 0 until adjudication runs). Age floor drops so mop
    # can keep draining without waiting 21 days / 5 cycles.
    stale_days = float(activation_cfg.consolidation_evidence_stale_reject_days)
    min_cycles = int(activation_cfg.evidence_forced_commit_cycles)
    if len(deferred) >= 200:
        stale_days = min(stale_days, 3.0)
        min_cycles = 0
        mop["recovery_stale"] = {"max_age_days": stale_days, "min_deferred_cycles": min_cycles}
    stale = select_stale_low_value_evidence(
        deferred,
        max_age_days=stale_days,
        min_deferred_cycles=min_cycles,
        limit=stale_budget,
    )
    mop["stale"] = await reject_evidence_rows(
        graph_store,
        group_id=group_id,
        rows=stale,
        reason_prefix="drain_stale",
        dry_run=bool(dry_run),
        reason_for_row=lambda _r: "stale_uncorroborated",
    )

    if cue_scan_due():
        mop["cue_hygiene"] = (
            await run_cue_hygiene(
                graph_store,
                group_id,
                max_per_cycle=cue_budget,
                min_age_days=float(activation_cfg.consolidation_cue_hygiene_min_age_days),
                dry_run=bool(dry_run),
            )
        ).to_dict()
        if not dry_run:
            mark_cue_scan_done()
    else:
        mop["cue_hygiene"] = {"skipped": True, "reason": "scan watermark not due"}

    # Metabolize: the only legitimate exit for deferred/pending evidence and
    # open adjudication requests is adjudication (commit-or-reject), and the
    # only consumer for cue_only episodes is replay. Under mop-only
    # scheduling these never ran anywhere — the queues froze (54 deferred /
    # 3039 cue_only observed) while every window paid shell downtime.
    if graph_manager is not None:
        from engram.consolidation.phases.edge_adjudication import EdgeAdjudicationPhase
        from engram.consolidation.phases.evidence_adjudication import (
            EvidenceAdjudicationPhase,
        )
        from engram.models.consolidation import CycleContext

        for key, phase in (
            ("evidence_adjudication", EvidenceAdjudicationPhase(graph_manager)),
            ("edge_adjudication", EdgeAdjudicationPhase(graph_manager)),
        ):
            try:
                result, records = await phase.execute(
                    group_id=group_id,
                    graph_store=graph_store,
                    activation_store=activation_store,
                    search_index=search_index,
                    cfg=activation_cfg,
                    cycle_id="mop_cli",
                    dry_run=bool(dry_run),
                    context=CycleContext(),
                )
                mop[key] = {
                    "status": result.status,
                    "items_processed": result.items_processed,
                    "items_affected": result.items_affected,
                    "records": len(records),
                }
            except Exception:
                logger.exception("mop %s pass failed", key)
                mop[key] = {"status": "error"}

    if extractor is not None and getattr(activation_cfg, "consolidation_replay_enabled", False):
        from engram.consolidation.phases.replay import EpisodeReplayPhase
        from engram.models.consolidation import CycleContext

        try:
            # Backlog sweep: the default 24h replay window targets recent
            # triage-skips, but the mop is the ONLY projection consumer under
            # quiet/shell scheduling — widen to the config maximum (30 days)
            # so the dormant cue_only/queued backlog actually drains.
            replay_cfg = activation_cfg.model_copy(
                update={"consolidation_replay_window_hours": 720.0}
            )
            result, records = await EpisodeReplayPhase(extractor=extractor).execute(
                group_id=group_id,
                graph_store=graph_store,
                activation_store=activation_store,
                search_index=search_index,
                cfg=replay_cfg,
                cycle_id="mop_cli",
                dry_run=bool(dry_run),
                context=CycleContext(),
            )
            mop["replay"] = {
                "status": result.status,
                "items_processed": result.items_processed,
                "items_affected": result.items_affected,
                "records": len(records),
            }
        except Exception:
            logger.exception("mop replay pass failed")
            mop["replay"] = {"status": "error"}

    # Bounded dedup: consumer installs never run the merge phase, so exact
    # duplicates (the documented graph fragmentation) are otherwise permanent.
    # Deterministic short-circuits only — no embeddings, no ANN, no LLM.
    if getattr(activation_cfg, "hygiene_mop_merge_enabled", False):
        from engram.consolidation.phases.merge import run_exact_merge_slice

        try:
            mop["merge_slice"] = await run_exact_merge_slice(
                graph_store,
                group_id,
                budget=int(getattr(activation_cfg, "hygiene_mop_merge_budget", 25) or 25),
                activation_store=activation_store,
                search_index=search_index,
                dry_run=bool(dry_run),
                max_history_size=int(activation_cfg.max_history_size),
            )
        except Exception:
            logger.exception("mop merge slice failed")
            mop["merge_slice"] = {"status": "error"}
    else:
        mop["merge_slice"] = {"skipped": True, "reason": "hygiene_mop_merge_enabled=False"}

    from engram.consolidation.phases.prune import PrunePhase
    from engram.models.consolidation import CycleContext

    phase = PrunePhase()
    prune_budget = max(cli_floor, evidence_budget)
    cfg = activation_cfg.model_copy(
        update={
            "consolidation_prune_max_per_cycle": prune_budget,
            "consolidation_prune_low_value_max_per_cycle": min(50, prune_budget),
        }
    )
    result, records = await phase.execute(
        group_id=group_id,
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        cfg=cfg,
        cycle_id="mop_cli",
        dry_run=bool(dry_run),
        context=CycleContext(),
    )
    reason_counts: dict[str, int] = {}
    for r in records:
        reason = str(getattr(r, "reason", "unknown") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    mop["prune"] = {
        "items_processed": result.items_processed,
        "items_affected": result.items_affected,
        "records": len(records),
        "reasons": reason_counts,
    }

    # RF M4.2: sync activation counters onto graph entity rows inside the mop
    # window (shell paused ⇒ single-writer), budgeted per window. This is the
    # ONLY writer of the graph-row access_count/last_accessed columns, so they
    # are approximate — stale up to one mop window, and missing the shell's
    # crash-window accesses (the brain loads the last-clean-exit snapshot
    # read-only). Ranking never reads them; see
    # MemoryActivationStore.snapshot_to_graph for the full contract.
    snapshot_sync = getattr(activation_store, "snapshot_to_graph", None)
    if callable(snapshot_sync) and not dry_run:
        try:
            synced = await snapshot_sync(graph_store, limit=_ACTIVATION_SNAPSHOT_SYNC_MAX)
            mop["snapshot_sync"] = {
                "entities": int(synced or 0),
                "budget": _ACTIVATION_SNAPSHOT_SYNC_MAX,
            }
        except Exception:
            logger.exception("mop activation snapshot sync failed")
            mop["snapshot_sync"] = {"status": "error"}
    else:
        mop["snapshot_sync"] = {
            "skipped": True,
            "reason": "dry_run" if dry_run else "store lacks snapshot_to_graph",
        }

    # Durable vector-debt drain: backfill missing episode + cue vectors under
    # a per-window budget. Provider breakage is loud but never fails the mop
    # (the M2.6 disaster was a broken provider staying invisible).
    from engram.storage.index_completeness import (
        EmbeddingProviderUnavailableError,
        backfill_missing_cue_vectors,
        backfill_missing_episode_vectors,
    )

    embeddings_enabled = getattr(search_index, "_embeddings_enabled", None)
    if dry_run:
        mop["vector_backfill"] = {"skipped": True, "reason": "dry_run"}
    elif embeddings_enabled is not True:
        mop["vector_backfill"] = {
            "skipped": True,
            "reason": (
                "search index has no embedding support"
                if embeddings_enabled is None
                else "embeddings disabled"
            ),
        }
    else:
        # Durable progression: ANN census presence is inexact on helix-native
        # (a single-probe sweep surfaces only a small reachable subset), so the
        # drain keeps a per-group (created_ts, id) cursor in the hygiene state
        # file — without it every window would re-embed the same first budget
        # of items and grow duplicate vectors.
        state = _read_hygiene_state()
        cursors_all = state.get("vector_backfill_cursors")
        cursors_all = dict(cursors_all) if isinstance(cursors_all, dict) else {}
        group_cursors = cursors_all.get(group_id)
        group_cursors = dict(group_cursors) if isinstance(group_cursors, dict) else {}

        def _cursor(value: Any) -> tuple[float, str] | None:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                try:
                    return (float(value[0]), str(value[1]))
                except (TypeError, ValueError):
                    return None
            return None

        def _persist_cursor(kind: str, cursor_next: Any) -> None:
            # Persisted per-drain, immediately: losing an advanced cursor
            # re-embeds the same window next mop and duplicate-inserts vectors
            # on helix-native (AddV appends, no upsert).
            if cursor_next is None:
                return
            group_cursors[kind] = list(cursor_next)
            cursors_all[group_id] = group_cursors
            state["vector_backfill_cursors"] = cursors_all
            _write_hygiene_state(state)

        ep_backfill = cue_backfill = None
        try:
            # Per-drain wall-clock bounds: the first live mop with these
            # drains ground past the runner's 1800s deadline (20s native
            # listing timeouts compounding on the saturated pool). wait_for
            # bounds the grind case; it cannot preempt a loop-blocking sync
            # native call — that failure mode is tracked as a follow-up.
            ep_backfill = await asyncio.wait_for(
                backfill_missing_episode_vectors(
                    graph_store,
                    search_index,
                    group_id,
                    max_episodes=_VECTOR_BACKFILL_EPISODES_MAX,
                    cursor=_cursor(group_cursors.get("episodes")),
                ),
                timeout=_VECTOR_BACKFILL_EPISODES_SECONDS_MAX,
            )
            _persist_cursor("episodes", ep_backfill.cursor_next)
            cue_backfill = await asyncio.wait_for(
                backfill_missing_cue_vectors(
                    graph_store,
                    search_index,
                    group_id,
                    max_cues=_VECTOR_BACKFILL_CUES_MAX,
                    cursor=_cursor(group_cursors.get("cues")),
                ),
                timeout=_VECTOR_BACKFILL_CUES_SECONDS_MAX,
            )
            _persist_cursor("cues", cue_backfill.cursor_next)
            mop["vector_backfill"] = {
                "episodes": ep_backfill.indexed,
                "cues": cue_backfill.indexed,
                "failed": ep_backfill.failed + cue_backfill.failed,
                "missing_before": {
                    "episodes": ep_backfill.missing_before,
                    "cues": cue_backfill.missing_before,
                },
                "budgets": {
                    "episodes": _VECTOR_BACKFILL_EPISODES_MAX,
                    "cues": _VECTOR_BACKFILL_CUES_MAX,
                },
            }
        except EmbeddingProviderUnavailableError:
            logger.warning(
                "mop vector backfill: embedding provider unavailable — "
                "vector debt NOT drained this window",
                exc_info=True,
            )
            mop["vector_backfill"] = {
                "status": "provider_unavailable",
                "episodes": ep_backfill.indexed if ep_backfill else 0,
            }
        except TimeoutError:
            logger.warning(
                "mop vector backfill exceeded its wall-clock bound "
                "(episodes=%ss cues=%ss) — window closed early, cursor kept",
                _VECTOR_BACKFILL_EPISODES_SECONDS_MAX,
                _VECTOR_BACKFILL_CUES_SECONDS_MAX,
            )
            mop["vector_backfill"] = {
                "status": "timeout",
                "episodes": ep_backfill.indexed if ep_backfill else 0,
            }
        except Exception:
            logger.exception("mop vector backfill failed")
            mop["vector_backfill"] = {
                "status": "error",
                "episodes": ep_backfill.indexed if ep_backfill else 0,
            }

    after = await collect_hygiene_debt_from_store(graph_store, group_id)
    try:
        remaining_deferred = await load_deferred_evidence(graph_store, group_id)
        after.deferred_evidence = len(remaining_deferred)
    except Exception:
        logger.debug("post-mop deferred recount failed", exc_info=True)
    report["mop"] = mop
    report["debt_after"] = after.to_dict()
    return report
