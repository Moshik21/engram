"""Shared hygiene mop operations (CLI + cold brain).

Not a second consolidator — the same local drains as warm evidence hygiene,
plus the bounded adjudication/replay passes that give the deferred and
cue_only queues an actual consumer under mop-only scheduling.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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

    after = await collect_hygiene_debt_from_store(graph_store, group_id)
    try:
        remaining_deferred = await load_deferred_evidence(graph_store, group_id)
        after.deferred_evidence = len(remaining_deferred)
    except Exception:
        logger.debug("post-mop deferred recount failed", exc_info=True)
    report["mop"] = mop
    report["debt_after"] = after.to_dict()
    return report
