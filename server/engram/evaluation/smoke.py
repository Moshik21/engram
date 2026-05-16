"""Deterministic local evaluation smoke for the full brain loop."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.store import SQLiteConsolidationStore
from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    format_brain_loop_report_markdown,
    has_recall_runtime_metrics,
)
from engram.evaluation.store import (
    SQLiteEvaluationStore,
    StoredRecallEvalSample,
    StoredRecallRuntimeMetricsSnapshot,
    StoredSessionContinuitySample,
)
from engram.extraction.factory import create_extractor
from engram.graph_manager import GraphManager
from engram.retrieval.need import analyze_memory_need
from engram.storage.bootstrap import initialize_search_index_for_graph, initialize_store_for_graph
from engram.storage.factory import create_stores
from engram.storage.helix.consolidation import HelixConsolidationStore
from engram.storage.resolver import EngineMode

DEFAULT_GROUP_ID = "default"

SMOKE_EPISODES = [
    (
        "Konner is building Engram with FastAPI and SQLite. "
        "Engram uses FastAPI for the REST API and SQLite for lite-mode memory."
    ),
    (
        "The Engram brain loop captures episodes, creates cue summaries, "
        "projects durable entities, recalls relevant packets, and consolidates memories."
    ),
    (
        "Konner prefers concise engineering updates and uses Engram to remember "
        "architecture decisions about the one brain per person runtime."
    ),
]
LOAD_SMOKE_TOPICS = [
    "activation-aware recall under repeated use",
    "cue projection totals for the brain loop",
    "native PyO3 Helix persistence without Docker",
    "one brain per person group ownership",
    "evaluation report coherence under load",
    "consolidation visibility after projection",
]
LOAD_SMOKE_QUERY_TERMS = [
    "activation-aware recall under repeated use",
    "cue projection totals for the brain loop",
    "native PyO3 Helix persistence without Docker",
    "one brain per person group ownership",
    "evaluation report coherence under load",
    "consolidation visibility after projection",
]

REQUIRED_ABSENT_GAPS = {
    "projection yield cannot be measured until episodes are projected",
    "consolidation effects need at least one recent cycle",
    "consolidation calibration needs saved calibration snapshots",
}


async def run_projected_consolidated_smoke(
    sqlite_path: Path,
    *,
    group_id: str = DEFAULT_GROUP_ID,
    mode: EngineMode = EngineMode.LITE,
    helix_data_dir: Path | None = None,
    load_count: int = 0,
    recall_rounds: int = 0,
    min_duration_seconds: float = 0.0,
    pause_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the Project + Consolidate smoke and return the final report."""
    if mode not in (EngineMode.LITE, EngineMode.HELIX):
        raise SystemExit("Projected/consolidated smoke supports lite or helix native mode")
    load_count = max(0, load_count)
    recall_rounds = max(0, recall_rounds)
    min_duration_seconds = max(0.0, float(min_duration_seconds))
    pause_seconds = max(0.0, float(pause_seconds))

    helix_config: dict[str, Any] = {}
    if mode == EngineMode.HELIX:
        helix_config = {
            "transport": "native",
            "data_dir": str(helix_data_dir) if helix_data_dir else "",
        }

    config = EngramConfig(
        mode=mode.value,
        sqlite={"path": str(sqlite_path)},
        helix=helix_config,
        embedding={"provider": "noop"},
        activation={
            "extraction_provider": "narrow",
            "cue_layer_enabled": True,
            "triage_enabled": True,
            "triage_extract_ratio": 1.0,
            "triage_min_score": 0.0,
            "triage_llm_judge_enabled": False,
            "triage_llm_escalation_enabled": False,
            "worker_enabled": False,
            "consolidation_enabled": False,
            "consolidation_dry_run": False,
            "consolidation_calibration_min_examples": 1,
            "consolidation_merge_use_embeddings": False,
            "graph_embedding_node2vec_enabled": False,
            "graph_embedding_transe_enabled": False,
            "graph_embedding_gnn_enabled": False,
        },
        _env_file=None,
    )
    _apply_smoke_activation_overrides(config)

    graph_store, activation_store, search_index = create_stores(mode, config)
    await graph_store.initialize()
    await initialize_search_index_for_graph(
        search_index,
        graph_store=graph_store,
        mode=mode,
    )

    if mode == EngineMode.HELIX:
        consolidation_store = HelixConsolidationStore(
            config.helix,
            client=getattr(graph_store, "_helix_client", None),
            owns_client=False,
        )
    else:
        consolidation_store = SQLiteConsolidationStore(str(sqlite_path))
    evaluation_store = SQLiteEvaluationStore(str(sqlite_path))
    await initialize_store_for_graph(
        consolidation_store,
        graph_store=graph_store,
        mode=mode,
    )
    await initialize_store_for_graph(
        evaluation_store,
        graph_store=graph_store,
        mode=mode,
    )

    try:
        manager = GraphManager(
            graph_store,
            activation_store,
            search_index,
            create_extractor(config),
            cfg=config.activation,
            runtime_mode=mode.value,
        )
        engine = ConsolidationEngine(
            graph_store,
            activation_store,
            search_index,
            cfg=config.activation,
            consolidation_store=consolidation_store,
            graph_manager=manager,
        )

        episode_ids = []
        for content in SMOKE_EPISODES:
            episode_ids.append(
                await manager.store_episode(content, group_id, "projected-consolidated-smoke")
            )
        for content in _load_smoke_episodes(load_count):
            episode_ids.append(
                await manager.store_episode(
                    content,
                    group_id,
                    "projected-consolidated-load-smoke",
                )
            )
        cue_feedback_checks = await _run_smoke_cue_feedback_check(
            manager,
            group_id=group_id,
            episode_id=episode_ids[0],
        )

        expected_projected = len(SMOKE_EPISODES) + load_count
        cycles = []
        cycle = None
        for cycle_index in range(_max_smoke_cycles(expected_projected)):
            cycle = await engine.run_cycle(
                group_id=group_id,
                trigger=(
                    "projected_consolidated_smoke"
                    if cycle_index == 0
                    else "projected_consolidated_smoke_load_continue"
                ),
                dry_run=False,
                phase_names={"triage"},
            )
            cycles.append(cycle)
            cycle_stats = await graph_store.get_stats(group_id)
            if _projected_count(cycle_stats) >= expected_projected:
                break
        if cycle is None:
            raise SystemExit("Projected/consolidated smoke did not run a consolidation cycle")

        gate_recall_checks = await _run_smoke_recall_gate_check(
            manager,
            group_id=group_id,
            load_count=load_count,
        )
        recall_checks = 0
        if recall_rounds > 0:
            recall_checks = await _run_smoke_recall_checks(
                manager,
                group_id=group_id,
                load_count=load_count,
                rounds=recall_rounds,
            )
        duration_checks, duration_elapsed = await _run_sustained_recall_checks(
            manager,
            group_id=group_id,
            load_count=load_count,
            min_duration_seconds=min_duration_seconds,
            pause_seconds=pause_seconds,
        )

        await evaluation_store.save_recall_sample(
            StoredRecallEvalSample(
                group_id=group_id,
                recall_triggered=True,
                recall_helped=True,
                recall_needed=True,
                packets_surfaced=3,
                packets_used=2,
                false_recalls=0,
                source="projected_consolidated_smoke",
                query="What is Engram's brain loop?",
                notes="Smoke label for projected/consolidated coverage.",
            )
        )
        await evaluation_store.save_session_sample(
            StoredSessionContinuitySample(
                group_id=group_id,
                baseline_score=0.2,
                memory_score=0.8,
                open_loop_expected=True,
                open_loop_recovered=True,
                temporal_expected=True,
                temporal_correct=True,
                source="projected_consolidated_smoke",
                scenario="Continue the Engram architecture thread after projection.",
                notes="Smoke label for continuity coverage.",
            )
        )

        graph_state = await manager.get_graph_state(
            group_id=group_id,
            top_n=10,
            include_edges=False,
        )
        stats = graph_state.get("stats") or {}
        recall_metrics = stats.get("recall_metrics") or {}
        if has_recall_runtime_metrics(recall_metrics):
            await evaluation_store.save_recall_metrics_snapshot(
                StoredRecallRuntimeMetricsSnapshot(
                    group_id=group_id,
                    metrics=dict(recall_metrics),
                    source="projected_consolidated_smoke",
                )
            )
        recent_cycles = await consolidation_store.get_recent_cycles(group_id, limit=10)
        calibration_snapshots = []
        for recent_cycle in recent_cycles:
            calibration_snapshots.extend(
                await consolidation_store.get_calibration_snapshots(
                    recent_cycle.id,
                    group_id,
                )
            )
        recall_samples = await evaluation_store.get_recall_samples(group_id, limit=100)
        session_samples = await evaluation_store.get_session_samples(group_id, limit=100)

        report = build_brain_loop_report(
            stats,
            group_id=group_id,
            recent_cycles=recent_cycles,
            calibration_snapshots=calibration_snapshots,
            recall_samples=recall_samples,
            session_samples=session_samples,
        )
        report["smoke"] = {
            "mode": mode.value,
            "sqlite_path": str(sqlite_path),
            "helix_data_dir": str(helix_data_dir) if helix_data_dir else None,
            "cycle_id": cycle.id,
            "cycle_status": cycle.status,
            "phase_count": len(cycle.phase_results),
            "cycle_count": len(cycles),
            "cycle_ids": [item.id for item in cycles],
            "load_count": load_count,
            "cue_feedback_checks": cue_feedback_checks,
            "gate_recall_checks": gate_recall_checks,
            "recall_rounds": recall_rounds,
            "recall_checks": recall_checks,
            "min_duration_seconds": min_duration_seconds,
            "duration_recall_checks": duration_checks,
            "duration_elapsed_seconds": duration_elapsed,
            "pause_seconds": pause_seconds,
        }
        assert_smoke_report(report, expected_projected=expected_projected)
        return report
    finally:
        await _close_if_supported(evaluation_store)
        await _close_if_supported(consolidation_store)
        await _close_if_supported(search_index)
        await _close_if_supported(graph_store)


async def run_projected_consolidated_smoke_for_args(
    *,
    sqlite_path: Path | None,
    replace: bool,
    group_id: str = DEFAULT_GROUP_ID,
    mode: EngineMode = EngineMode.LITE,
    helix_data_dir: Path | None = None,
    load_count: int = 0,
    recall_rounds: int = 0,
    min_duration_seconds: float = 0.0,
    pause_seconds: float = 0.0,
) -> dict[str, Any]:
    """Run the smoke using supplied paths or disposable temp storage."""
    if sqlite_path is not None:
        db_path = sqlite_path.expanduser()
        if db_path.exists():
            if not replace:
                raise SystemExit(f"{db_path} already exists; pass --replace to reuse this path")
            db_path.unlink()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        native_dir = None
        if mode == EngineMode.HELIX:
            native_dir = _prepare_helix_data_dir(
                helix_data_dir,
                replace,
                fallback=db_path.parent / f"{db_path.stem}.helix",
            )
        return await run_projected_consolidated_smoke(
            db_path,
            group_id=group_id,
            mode=mode,
            helix_data_dir=native_dir,
            load_count=load_count,
            recall_rounds=recall_rounds,
            min_duration_seconds=min_duration_seconds,
            pause_seconds=pause_seconds,
        )

    with tempfile.TemporaryDirectory(prefix="engram-project-consolidate-") as tmp:
        native_dir = None
        if mode == EngineMode.HELIX:
            native_dir = _prepare_helix_data_dir(
                helix_data_dir,
                replace,
                fallback=Path(tmp) / "helix-native",
            )
        return await run_projected_consolidated_smoke(
            Path(tmp) / "smoke.db",
            group_id=group_id,
            mode=mode,
            helix_data_dir=native_dir,
            load_count=load_count,
            recall_rounds=recall_rounds,
            min_duration_seconds=min_duration_seconds,
            pause_seconds=pause_seconds,
        )


def assert_smoke_report(report: dict[str, Any], *, expected_projected: int = 1) -> None:
    """Raise SystemExit if the smoke report still has blocking gaps."""
    gaps = set(report.get("coverage_gaps") or [])
    blocking_gaps = sorted(gaps & REQUIRED_ABSENT_GAPS)
    if blocking_gaps:
        raise SystemExit(f"Projected/consolidated smoke still has coverage gaps: {blocking_gaps}")

    project = report.get("project") or {}
    consolidate = report.get("consolidate") or {}
    calibration = consolidate.get("calibration") or {}
    recall = report.get("recall") or {}

    projected_count = int(project.get("projected_count") or 0)
    if projected_count < expected_projected:
        raise SystemExit(
            "Projected/consolidated smoke projected "
            f"{projected_count} episodes; expected at least {expected_projected}"
        )
    if int((project.get("yield") or {}).get("linked_entity_count") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not link any entities")
    cue = report.get("cue") or {}
    if int(cue.get("surfaced_count") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not record cue feedback")
    if int(consolidate.get("cycle_count") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not persist a consolidation cycle")
    if calibration.get("status") != "measured":
        raise SystemExit("Projected/consolidated smoke did not persist calibration snapshots")
    if (recall.get("evaluation") or {}).get("status") != "measured":
        raise SystemExit("Projected/consolidated smoke did not persist recall labels")
    if (recall.get("continuity") or {}).get("status") != "measured":
        raise SystemExit("Projected/consolidated smoke did not persist continuity labels")
    if int(recall.get("total_analyses") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not record recall gate analysis")
    if int(recall.get("trigger_count") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not record a recall gate trigger")
    latency = recall.get("latency") or {}
    analyzer_latency = latency.get("analyzer_ms") or {}
    if float(analyzer_latency.get("p95_ms") or 0.0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not record recall analyzer latency")
    control = recall.get("control") or {}
    if int(control.get("surfaced_count") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not record surfaced recall feedback")
    smoke = report.get("smoke") or {}
    if int(smoke.get("gate_recall_checks") or 0) <= 0:
        raise SystemExit("Projected/consolidated smoke did not run a recall gate check")


def format_smoke_report(report: dict[str, Any]) -> str:
    """Render a smoke report with the smoke-specific cycle footer."""
    smoke = report.get("smoke") or {}
    return (
        format_brain_loop_report_markdown(report)
        + "\n"
        + "Smoke: "
        + f"mode={smoke.get('mode', 'lite')} "
        + f"cycle={smoke.get('cycle_id')} "
        + f"status={smoke.get('cycle_status')} "
        + f"load={smoke.get('load_count', 0)} "
        + f"cue_checks={smoke.get('cue_feedback_checks', 0)} "
        + f"gate_checks={smoke.get('gate_recall_checks', 0)} "
        + f"recall_checks={smoke.get('recall_checks', 0)} "
        + f"duration_checks={smoke.get('duration_recall_checks', 0)} "
        + f"db={smoke.get('sqlite_path')}\n"
    )


def _load_smoke_episodes(load_count: int) -> list[str]:
    """Build deterministic load-smoke episodes with stable queryable terms."""
    episodes: list[str] = []
    for index in range(load_count):
        topic = LOAD_SMOKE_TOPICS[index % len(LOAD_SMOKE_TOPICS)]
        episodes.append(
            f"Engram load smoke memory {index + 1:03d} records {topic} "
            f"for group-scoped brain runtime verification."
        )
    return episodes


def _load_smoke_queries(load_count: int) -> list[str]:
    """Return stable recall queries for the load smoke corpus."""
    if load_count <= 0:
        return ["Engram brain loop", "one brain per person runtime"]
    query_count = min(load_count, len(LOAD_SMOKE_QUERY_TERMS))
    return LOAD_SMOKE_QUERY_TERMS[:query_count]


def _max_smoke_cycles(expected_projected: int) -> int:
    """Allow enough triage-only cycles for larger load-smoke batches."""
    return max(1, (expected_projected + 99) // 100 + 1)


def _projected_count(stats: dict[str, Any]) -> int:
    """Read projected episode count from graph stats across store shapes."""
    projection_metrics = stats.get("projection_metrics")
    if not isinstance(projection_metrics, dict):
        return 0
    state_counts = projection_metrics.get("state_counts")
    if isinstance(state_counts, dict):
        return int(state_counts.get("projected") or 0)
    return int(projection_metrics.get("projected_count") or 0)


async def _run_smoke_cue_feedback_check(
    manager: GraphManager,
    *,
    group_id: str,
    episode_id: str,
) -> int:
    """Exercise cue feedback before projection so cue usefulness is measured."""
    query = "Engram brain loop creates cue summaries"
    episode = await manager._graph.get_episode_by_id(episode_id, group_id)
    if episode is None:
        raise SystemExit("Projected/consolidated smoke could not reload a cue episode")
    await manager._record_cue_hit(
        episode,
        0.72,
        query,
        interaction_type="surfaced",
    )
    return 1


async def _run_smoke_recall_checks(
    manager: GraphManager,
    *,
    group_id: str,
    load_count: int,
    rounds: int,
) -> int:
    """Run deterministic recall checks after projection."""
    checks = 0
    for _round in range(rounds):
        for query in _load_smoke_queries(load_count):
            await _run_recall_gate_query(
                manager,
                group_id=group_id,
                query=query,
                failure_prefix="Projected/consolidated smoke recall",
            )
            checks += 1
    return checks


async def _run_smoke_recall_gate_check(
    manager: GraphManager,
    *,
    group_id: str,
    load_count: int,
) -> int:
    """Exercise recall need analysis plus recall feedback once for the report."""
    query = "What did we decide about the Engram brain loop?"
    await _run_recall_gate_query(
        manager,
        group_id=group_id,
        query=query,
        failure_prefix="Projected/consolidated smoke gate recall",
    )
    return 1


async def _run_recall_gate_query(
    manager: GraphManager,
    *,
    group_id: str,
    query: str,
    failure_prefix: str,
) -> None:
    thresholds = manager.get_recall_need_thresholds(group_id)
    cfg = getattr(manager, "_cfg", None)
    memory_need = await analyze_memory_need(
        query,
        mode="evaluation_smoke",
        group_id=group_id,
        cfg=cfg,
        thresholds=thresholds,
    )
    manager.record_memory_need_analysis(group_id, memory_need)
    results = await manager.recall(
        query,
        group_id=group_id,
        limit=5,
        interaction_type="surfaced",
        interaction_source="evaluation_smoke",
        memory_need=memory_need,
    )
    if not results:
        raise SystemExit(f"{failure_prefix} returned no results: {query}")


async def _run_sustained_recall_checks(
    manager: GraphManager,
    *,
    group_id: str,
    load_count: int,
    min_duration_seconds: float,
    pause_seconds: float,
) -> tuple[int, float]:
    """Keep recall active for an operator-specified duration."""
    if min_duration_seconds <= 0:
        return 0, 0.0

    queries = _load_smoke_queries(load_count)
    checks = 0
    start = time.monotonic()
    while True:
        for query in queries:
            results = await manager.recall(query, group_id=group_id, limit=5)
            if not results:
                raise SystemExit(
                    f"Sustained smoke recall returned no results: {query}"
                )
            checks += 1
        elapsed = time.monotonic() - start
        if elapsed >= min_duration_seconds:
            return checks, round(elapsed, 3)
        if pause_seconds > 0:
            await asyncio.sleep(pause_seconds)


def _prepare_helix_data_dir(
    helix_data_dir: Path | None,
    replace: bool,
    *,
    fallback: Path | None = None,
) -> Path | None:
    """Prepare an optional native Helix data directory for smoke runs."""
    data_dir = helix_data_dir.expanduser() if helix_data_dir is not None else fallback
    if data_dir is None:
        return None
    if data_dir.exists():
        if not replace:
            raise SystemExit(f"{data_dir} already exists; pass --replace to reuse this path")
        if data_dir.is_dir():
            shutil.rmtree(data_dir)
        else:
            data_dir.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _apply_smoke_activation_overrides(config: EngramConfig) -> None:
    """Re-apply smoke settings after profile presets mutate activation config."""
    cfg = config.activation
    cfg.extraction_provider = "narrow"
    cfg.cue_layer_enabled = True
    cfg.triage_enabled = True
    cfg.triage_extract_ratio = 1.0
    cfg.triage_min_score = 0.0
    cfg.triage_llm_judge_enabled = False
    cfg.triage_llm_escalation_enabled = False
    cfg.worker_enabled = False
    cfg.consolidation_enabled = False
    cfg.consolidation_dry_run = False
    cfg.consolidation_calibration_min_examples = 1
    cfg.consolidation_merge_use_embeddings = False
    cfg.graph_embedding_node2vec_enabled = False
    cfg.graph_embedding_transe_enabled = False
    cfg.graph_embedding_gnn_enabled = False


async def _close_if_supported(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result
