"""Shared hygiene mop operations (CLI + cold brain).

Not a second consolidator — same local drains as warm evidence hygiene.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


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
) -> dict[str, Any]:
    """Run bounded debt drains. Returns mop stats + debt_before/after."""
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
    from engram.consolidation.pressure import ConsolidationPressure
    from engram.loop_adjustment import mop_knob_budgets

    debt = await collect_hygiene_debt_from_store(graph_store, group_id)
    debt_pressure = debt_pressure_contribution(debt)
    event_pressure = ConsolidationPressure().compute(activation_cfg)
    total_pressure = event_pressure + debt_pressure
    should_mop = debt_should_trigger_mop(
        debt,
        pressure_threshold=float(activation_cfg.consolidation_pressure_threshold),
    )
    report: dict[str, Any] = {
        "group_id": group_id,
        "debt": debt.to_dict(),
        "pressure": {
            "event_bus": round(event_pressure, 2),
            "hygiene_debt": round(debt_pressure, 2),
            "total": round(total_pressure, 2),
            "threshold": activation_cfg.consolidation_pressure_threshold,
            "should_trigger_mop": should_mop,
        },
    }

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

    mop["cue_hygiene"] = (
        await run_cue_hygiene(
            graph_store,
            group_id,
            max_per_cycle=cue_budget,
            min_age_days=float(activation_cfg.consolidation_cue_hygiene_min_age_days),
            dry_run=bool(dry_run),
        )
    ).to_dict()

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
