"""Operator hygiene protocol: debt scoreboard + bounded mop drains.

Not a second consolidator. Autonomic sleep stays in-process; this CLI lets a
harness/operator inspect debt and run the same local drains consolidation uses.
Public MCP surface is unchanged (no mop tools on the golden loop).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def configure_hygiene_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "action",
        choices=["report", "mop"],
        help="report: print debt scoreboard; mop: bounded local drains",
    )
    parser.add_argument("--group-id", default=None)
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
    )
    parser.add_argument("--helix-data-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mop actions without writing",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=200,
        help="Max items per drain type (default 200)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
    )


async def run_hygiene_command(args: argparse.Namespace) -> int:
    from engram.config import EngramConfig
    from engram.consolidation.hygiene_debt import (
        collect_hygiene_debt_from_store,
        debt_pressure_contribution,
        debt_should_trigger_mop,
    )
    from engram.consolidation.pressure import ConsolidationPressure
    from engram.storage.bootstrap import (
        close_if_supported,
        create_local_runtime_stores,
        initialize_search_index_for_graph,
    )
    from engram.storage.resolver import resolve_mode

    requested = args.mode or "auto"
    config = EngramConfig(mode=requested)
    if args.helix_data_dir is not None:
        config.helix.data_dir = str(args.helix_data_dir.expanduser())
        config.helix.transport = "native"
        if args.mode is None:
            config.mode = "helix"

    group_id = args.group_id or config.default_group_id
    # Loop Steward overlay: mop budgets follow active adjustment without mutating boot cfg.
    loop_adj = None
    try:
        from engram.loop_adjustment import (
            effective_activation_config,
            load_active_adjustment,
        )

        loop_adj = load_active_adjustment(group_id)
        activation_cfg = effective_activation_config(config.activation, loop_adj)
    except Exception:
        activation_cfg = config.activation
        loop_adj = None

    graph_store = None
    activation_store = None
    search_index = None
    try:
        mode = await resolve_mode(config.mode)
        graph_store, activation_store, search_index = create_local_runtime_stores(mode, config)
        await graph_store.initialize()
        await initialize_search_index_for_graph(search_index, graph_store=graph_store, mode=mode)

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

        if args.action == "report":
            _emit(report, args.format)
            return 0

        from engram.hygiene_ops import execute_hygiene_mop

        report = await execute_hygiene_mop(
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            activation_cfg=activation_cfg,
            group_id=group_id,
            budget=max(1, int(args.budget)),
            dry_run=bool(args.dry_run),
            loop_adj=loop_adj,
        )
        _emit(report, args.format)
        return 0
    except Exception as exc:
        logger.exception("hygiene command failed")
        print(f"hygiene command failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if search_index is not None:
            await close_if_supported(search_index)
        if activation_store is not None:
            await close_if_supported(activation_store)
        if graph_store is not None:
            await close_if_supported(graph_store)


def _emit(payload: dict[str, Any], fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload, indent=2, default=str))
        return
    debt = payload.get("debt") or {}
    pressure = payload.get("pressure") or {}
    print("Hygiene debt scoreboard")
    print(
        f"  deferred_evidence={debt.get('deferred_evidence')} "
        f"pending_evidence={debt.get('pending_evidence')} "
        f"cue_only={debt.get('cue_only_episodes')} "
        f"cues={debt.get('cue_count')}"
    )
    print(
        f"  near_miss={debt.get('near_miss_count')} "
        f"open_adjudication={debt.get('open_adjudication')} "
        f"orphans={debt.get('orphan_candidates')} "
        f"open_work={debt.get('open_work')}"
    )
    print(
        f"  pressure total={pressure.get('total')} "
        f"(debt={pressure.get('hygiene_debt')} event={pressure.get('event_bus')}) "
        f"threshold={pressure.get('threshold')} "
        f"should_mop={pressure.get('should_trigger_mop')}"
    )
    mop = payload.get("mop")
    if mop:
        print("Mop results")
        print(f"  dry_run={mop.get('dry_run')} budget={mop.get('budget')}")
        ed = mop.get("evidence_drain") or {}
        print(
            f"  evidence_drain rejected={ed.get('rejected')} kept={ed.get('kept')} "
            f"errors={ed.get('errors')}"
        )
        ae = mop.get("already_exists") or {}
        if ae:
            print(f"  already_exists rejected={ae.get('rejected')} errors={ae.get('errors')}")
        st = mop.get("stale") or {}
        if st:
            print(f"  stale rejected={st.get('rejected')} errors={st.get('errors')}")
        ch = mop.get("cue_hygiene") or {}
        print(
            f"  cue_hygiene demoted={ch.get('demoted')} eligible={ch.get('eligible')} "
            f"scanned={ch.get('scanned')}"
        )
        pr = mop.get("prune") or {}
        print(f"  prune affected={pr.get('items_affected')} reasons={pr.get('reasons')}")
        after = payload.get("debt_after") or {}
        if after:
            print(
                f"  after deferred={after.get('deferred_evidence')} "
                f"cue_only={after.get('cue_only_episodes')} "
                f"open_work={after.get('open_work')}"
            )
