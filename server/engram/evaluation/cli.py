"""Command helpers for local brain-loop evaluation reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram.config import EngramConfig
from engram.evaluation.benchmark_evidence import (
    benchmark_evidence_failure_message,
    load_benchmark_evidence,
)
from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    evaluation_signal_failure_message,
    format_brain_loop_report_markdown,
    is_brain_loop_report_payload,
    looks_like_partial_brain_loop_report,
    merge_recall_runtime_metrics,
    missing_brain_loop_report_sections,
)
from engram.evaluation.human_label_evidence import (
    build_human_label_evidence_template,
    human_label_evidence_failure_message,
    load_human_label_evidence,
    render_human_label_evidence_template_markdown,
)
from engram.evaluation.store import SQLiteEvaluationStore
from engram.storage.bootstrap import (
    create_consolidation_store_for_graph,
    create_evaluation_store_for_graph,
)
from engram.storage.resolver import EngineMode, resolve_mode
from engram.storage.sqlite.graph import SQLiteGraphStore


def configure_evaluate_parser(parser: argparse.ArgumentParser) -> None:
    """Attach brain-loop evaluation report options to a parser."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Run a deterministic Capture -> Cue -> Project -> Recall -> "
            "Consolidate smoke instead of reading an existing DB/report. "
            "Use --mode helix for the preferred native PyO3 full-backend path; "
            "bare --smoke remains the lite fallback."
        ),
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        help=(
            "Read stats/cycles/samples or a saved brain-loop report from JSON "
            "instead of the local SQLite DB."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["lite", "full", "helix", "auto"],
        default=None,
        help=(
            "Engine mode to inspect for live reports. Defaults to auto unless "
            "--sqlite-path is supplied."
        ),
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        help=(
            "SQLite DB path for lite reporting and saved evaluation samples. "
            "In Helix smoke mode this stores local evaluation labels. Defaults to config."
        ),
    )
    parser.add_argument(
        "--helix-data-dir",
        type=Path,
        help=(
            "Native Helix data directory for --mode helix reports or smoke runs. "
            "Smoke runs use a disposable directory unless this is supplied."
        ),
    )
    parser.add_argument(
        "--group-id",
        help="Group/brain ID. Defaults to config or the JSON payload group.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Recent consolidation cycles to include for live SQLite reports.",
    )
    parser.add_argument(
        "--recall-samples",
        type=Path,
        help="JSON file containing labeled recall_samples.",
    )
    parser.add_argument(
        "--session-samples",
        type=Path,
        help="JSON file containing session continuity samples.",
    )
    parser.add_argument(
        "--no-saved-samples",
        action="store_true",
        help="Do not read persisted evaluation samples from the SQLite DB.",
    )
    parser.add_argument(
        "--require-evaluation-signals",
        action="store_true",
        help=(
            "Exit non-zero unless all required evaluation_signals are present, "
            "measured, backed by evidence, and have a metric."
        ),
    )
    parser.add_argument(
        "--min-evaluation-signal-evidence",
        type=int,
        default=1,
        help=(
            "Minimum evidence_count required for each measured evaluation signal "
            "when --require-evaluation-signals is set (default: 1). Raise this for "
            "benchmark or release gates so one smoke label cannot satisfy the gate."
        ),
    )
    parser.add_argument(
        "--benchmark-artifact",
        type=Path,
        help=(
            "Showcase benchmark results.json artifact to attach to the report "
            "and optionally gate with --require-benchmark-evidence."
        ),
    )
    parser.add_argument(
        "--require-benchmark-evidence",
        action="store_true",
        help=(
            "Exit non-zero unless --benchmark-artifact contains measured benchmark "
            "evidence for the configured baseline."
        ),
    )
    parser.add_argument(
        "--benchmark-baseline",
        default="engram_full",
        help="Benchmark baseline to gate in --benchmark-artifact (default: engram_full).",
    )
    parser.add_argument(
        "--min-benchmark-scenarios",
        type=int,
        default=1,
        help="Minimum available scenarios required when gating benchmark evidence.",
    )
    parser.add_argument(
        "--min-benchmark-pass-rate",
        type=float,
        default=0.0,
        help="Minimum scenario pass rate required when gating benchmark evidence.",
    )
    parser.add_argument(
        "--evidence-bundle",
        type=Path,
        help=(
            "Write a JSON evidence bundle containing the report, attached benchmark "
            "evidence, source paths, and gate thresholds after gates pass."
        ),
    )
    parser.add_argument(
        "--human-label-artifact",
        type=Path,
        help=(
            "JSON artifact containing real human-labeled harness recall/session "
            "samples to attach to the report and optionally gate with "
            "--require-human-label-evidence."
        ),
    )
    parser.add_argument(
        "--human-label-template",
        action="store_true",
        help=(
            "Print a JSON/Markdown template for collecting real human-labeled "
            "harness evidence instead of building a report."
        ),
    )
    parser.add_argument(
        "--require-human-label-evidence",
        action="store_true",
        help=(
            "Exit non-zero unless --human-label-artifact contains real "
            "human-labeled harness evidence, not deterministic smoke/benchmark data."
        ),
    )
    parser.add_argument(
        "--min-human-recall-samples",
        type=int,
        default=1,
        help="Minimum human-labeled recall samples required when gating evidence.",
    )
    parser.add_argument(
        "--min-human-session-samples",
        type=int,
        default=1,
        help=(
            "Minimum human-labeled session-continuity samples required when "
            "gating evidence."
        ),
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="When used with --smoke and --sqlite-path, replace an existing smoke DB.",
    )
    parser.add_argument(
        "--smoke-load-count",
        type=int,
        default=0,
        help="Extra deterministic episodes to add during --smoke load verification.",
    )
    parser.add_argument(
        "--smoke-recall-rounds",
        type=int,
        default=0,
        help="Recall rounds to run against the projected smoke corpus during --smoke.",
    )
    parser.add_argument(
        "--smoke-min-duration-seconds",
        type=float,
        default=0.0,
        help=(
            "Minimum sustained recall-stress duration for --smoke after projection. "
            "Use with --mode helix for hour-scale native PyO3 operator soaks."
        ),
    )
    parser.add_argument(
        "--smoke-pause-seconds",
        type=float,
        default=0.0,
        help="Optional pause between sustained --smoke recall loops.",
    )
    parser.add_argument(
        "--saved-sample-limit",
        type=int,
        default=500,
        help="Maximum persisted recall/session samples to read per kind.",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )


async def build_report_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build a brain-loop report from parsed CLI arguments."""
    if args.smoke:
        from engram.evaluation.smoke import run_projected_consolidated_smoke_for_args

        mode = await _resolve_smoke_mode(args.mode)
        config = EngramConfig()
        return await run_projected_consolidated_smoke_for_args(
            sqlite_path=args.sqlite_path,
            replace=args.replace,
            group_id=args.group_id or config.default_group_id,
            mode=mode,
            helix_data_dir=args.helix_data_dir,
            load_count=max(0, args.smoke_load_count),
            recall_rounds=max(0, args.smoke_recall_rounds),
            min_duration_seconds=max(0.0, args.smoke_min_duration_seconds),
            pause_seconds=max(0.0, args.smoke_pause_seconds),
        )

    source_payload: dict[str, Any] = {}
    saved_recall_samples: list[Any] = []
    saved_session_samples: list[Any] = []

    if args.from_json:
        source_payload = _load_json(args.from_json)
        if is_brain_loop_report_payload(source_payload):
            report = dict(source_payload)
            if args.group_id:
                report["group_id"] = args.group_id
            return report
        if looks_like_partial_brain_loop_report(source_payload):
            raise SystemExit(
                "--from-json looks like a brain-loop report but is missing "
                "required report sections: "
                f"{missing_brain_loop_report_sections(source_payload)}"
            )
        stats, recent_cycles, calibration_snapshots, group_id = _extract_json_inputs(
            args,
            source_payload,
        )
    else:
        (
            stats,
            recent_cycles,
            calibration_snapshots,
            saved_recall_samples,
            saved_session_samples,
            group_id,
        ) = await _load_live_report(args)

    recall_samples = _load_optional_samples(
        args.recall_samples,
        source_payload,
        "recall_samples",
        "recallSamples",
    )
    if not recall_samples and not args.recall_samples:
        recall_samples = saved_recall_samples
    session_samples = _load_optional_samples(
        args.session_samples,
        source_payload,
        "session_samples",
        "sessionSamples",
    )
    if not session_samples and not args.session_samples:
        session_samples = saved_session_samples

    return build_brain_loop_report(
        stats,
        group_id=group_id,
        recent_cycles=recent_cycles,
        calibration_snapshots=calibration_snapshots,
        recall_samples=recall_samples,
        session_samples=session_samples,
    )


async def run_evaluate_command(args: argparse.Namespace) -> None:
    """Print a brain-loop report for parsed CLI arguments."""
    if getattr(args, "human_label_template", False):
        template = build_human_label_evidence_template()
        if args.format == "json":
            print(json.dumps(template, indent=2, sort_keys=True))
            return
        print(render_human_label_evidence_template_markdown(template), end="")
        return

    report = await build_report_from_args(args)
    report = _attach_benchmark_evidence_from_args(report, args)
    report = _attach_human_label_evidence_from_args(report, args)
    if getattr(args, "require_evaluation_signals", False):
        failure_message = evaluation_signal_failure_message(
            report,
            prefix="Brain-loop evaluation has unmeasured evaluation signals",
            min_evidence_count=max(
                1,
                getattr(args, "min_evaluation_signal_evidence", 1),
            ),
        )
        if failure_message:
            raise SystemExit(failure_message)
    if getattr(args, "require_benchmark_evidence", False):
        failure_message = benchmark_evidence_failure_message(
            report.get("benchmark_evidence"),
            prefix="Benchmark evidence failed gates",
        )
        if failure_message:
            raise SystemExit(failure_message)
    if getattr(args, "require_human_label_evidence", False):
        failure_message = human_label_evidence_failure_message(
            report.get("human_label_evidence"),
            prefix="Human label evidence failed gates",
        )
        if failure_message:
            raise SystemExit(failure_message)
    _write_evidence_bundle_from_args(report, args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if args.smoke:
        from engram.evaluation.smoke import format_smoke_report

        print(format_smoke_report(report), end="")
        return
    print(format_brain_loop_report_markdown(report))


def _attach_benchmark_evidence_from_args(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact_path = getattr(args, "benchmark_artifact", None)
    require_evidence = getattr(args, "require_benchmark_evidence", False)
    if artifact_path is None:
        if require_evidence:
            report = dict(report)
            report["benchmark_evidence"] = None
        return report
    try:
        evidence = load_benchmark_evidence(
            artifact_path,
            baseline=getattr(args, "benchmark_baseline", "engram_full"),
            min_scenarios=max(1, getattr(args, "min_benchmark_scenarios", 1)),
            min_pass_rate=max(0.0, getattr(args, "min_benchmark_pass_rate", 0.0)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid benchmark artifact {artifact_path}: {exc}") from exc
    report = dict(report)
    report["benchmark_evidence"] = evidence
    return report


def _attach_human_label_evidence_from_args(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact_path = getattr(args, "human_label_artifact", None)
    require_evidence = getattr(args, "require_human_label_evidence", False)
    if artifact_path is None:
        if require_evidence:
            report = dict(report)
            report["human_label_evidence"] = None
        return report
    try:
        evidence = load_human_label_evidence(
            artifact_path,
            min_recall_samples=max(0, getattr(args, "min_human_recall_samples", 1)),
            min_session_samples=max(0, getattr(args, "min_human_session_samples", 1)),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid human label artifact {artifact_path}: {exc}") from exc
    report = dict(report)
    report["human_label_evidence"] = evidence
    return report


def _write_evidence_bundle_from_args(report: dict[str, Any], args: argparse.Namespace) -> None:
    bundle_path = getattr(args, "evidence_bundle", None)
    if bundle_path is None:
        return
    payload = build_evidence_bundle(report, args)
    bundle_path = bundle_path.expanduser()
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_evidence_bundle(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Build the JSON artifact operators can archive for completion evidence."""
    return {
        "kind": "engram_brain_loop_evidence_bundle",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed",
        "group_id": report.get("group_id"),
        "sources": {
            "report_json": _optional_path_str(getattr(args, "from_json", None)),
            "benchmark_artifact": _optional_path_str(getattr(args, "benchmark_artifact", None)),
            "human_label_artifact": _optional_path_str(
                getattr(args, "human_label_artifact", None)
            ),
            "recall_samples": _optional_path_str(getattr(args, "recall_samples", None)),
            "session_samples": _optional_path_str(getattr(args, "session_samples", None)),
            "sqlite_path": _optional_path_str(getattr(args, "sqlite_path", None)),
            "helix_data_dir": _optional_path_str(getattr(args, "helix_data_dir", None)),
        },
        "gates": {
            "require_evaluation_signals": bool(
                getattr(args, "require_evaluation_signals", False)
            ),
            "min_evaluation_signal_evidence": max(
                1,
                getattr(args, "min_evaluation_signal_evidence", 1),
            ),
            "require_benchmark_evidence": bool(
                getattr(args, "require_benchmark_evidence", False)
            ),
            "benchmark_baseline": getattr(args, "benchmark_baseline", "engram_full"),
            "min_benchmark_scenarios": max(
                1,
                getattr(args, "min_benchmark_scenarios", 1),
            ),
            "min_benchmark_pass_rate": max(
                0.0,
                getattr(args, "min_benchmark_pass_rate", 0.0),
            ),
            "require_human_label_evidence": bool(
                getattr(args, "require_human_label_evidence", False)
            ),
            "min_human_recall_samples": max(
                0,
                getattr(args, "min_human_recall_samples", 1),
            ),
            "min_human_session_samples": max(
                0,
                getattr(args, "min_human_session_samples", 1),
            ),
        },
        "report": report,
    }


def _optional_path_str(path: Any) -> str | None:
    return str(path) if path is not None else None


async def _load_live_report(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[Any], list[Any], list[Any], list[Any], str]:
    requested_mode = args.mode or ("lite" if args.sqlite_path is not None else "auto")
    config = EngramConfig(mode=requested_mode)
    if args.sqlite_path:
        config.sqlite.path = str(args.sqlite_path)
    if args.helix_data_dir:
        config.helix.transport = "native"
        config.helix.data_dir = str(args.helix_data_dir.expanduser())

    group_id = args.group_id or config.default_group_id
    mode = await resolve_mode(config.mode)
    graph_store = _create_graph_store(mode, config)
    consolidation_store: Any | None = None
    evaluation_store: SQLiteEvaluationStore | None = None

    await graph_store.initialize()
    try:
        consolidation_store = await _create_consolidation_store(mode, config, graph_store)
        stats = await graph_store.get_stats(group_id)
        recent_cycles = await consolidation_store.get_recent_cycles(
            group_id,
            limit=max(1, args.cycles),
        )
        calibration_snapshots: list[Any] = []
        for cycle in recent_cycles:
            calibration_snapshots.extend(
                await consolidation_store.get_calibration_snapshots(cycle.id, group_id)
        )
        recall_samples: list[Any] = []
        session_samples: list[Any] = []
        if not args.no_saved_samples:
            evaluation_store = await create_evaluation_store_for_graph(
                config,
                graph_store=graph_store,
                mode=mode,
            )
            sample_limit = max(1, args.saved_sample_limit)
            stats = merge_recall_runtime_metrics(
                stats,
                await evaluation_store.get_latest_recall_metrics_snapshot(group_id),
            )
            recall_samples = await evaluation_store.get_recall_samples(group_id, sample_limit)
            session_samples = await evaluation_store.get_session_samples(group_id, sample_limit)
        return (
            stats,
            recent_cycles,
            calibration_snapshots,
            recall_samples,
            session_samples,
            group_id,
        )
    finally:
        await _maybe_close(evaluation_store)
        await _maybe_close(consolidation_store)
        await _maybe_close(graph_store)


def _create_graph_store(mode: EngineMode, config: EngramConfig) -> Any:
    if mode == EngineMode.LITE:
        return SQLiteGraphStore(str(config.get_sqlite_path()))

    from engram.storage.factory import create_stores

    graph_store, _activation_store, _search_index = create_stores(mode, config)
    return graph_store


async def _create_consolidation_store(
    mode: EngineMode,
    config: EngramConfig,
    graph_store: Any,
) -> Any:
    return await create_consolidation_store_for_graph(
        config,
        graph_store=graph_store,
        mode=mode,
    )


async def _resolve_smoke_mode(requested_mode: str | None) -> EngineMode:
    """Resolve the backend used by the deterministic smoke command."""
    if requested_mode is None or requested_mode == "lite":
        return EngineMode.LITE
    if requested_mode == "helix":
        return EngineMode.HELIX
    if requested_mode == "auto":
        mode = await resolve_mode("auto")
        if mode in (EngineMode.LITE, EngineMode.HELIX):
            return mode
        raise SystemExit("Projected/consolidated smoke supports lite or helix native mode")
    raise SystemExit("Projected/consolidated smoke supports lite or helix native mode")


async def _maybe_close(resource: Any) -> None:
    if resource is None:
        return
    close = getattr(resource, "close", None)
    if close is None:
        return
    result = close()
    if hasattr(result, "__await__"):
        await result


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _list_payload(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        samples = value.get("samples")
        if isinstance(samples, list):
            return samples
    return []


def _extract_list(payload: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return _list_payload(value)
    return []


def _extract_json_inputs(
    args: argparse.Namespace,
    payload: Any | None = None,
) -> tuple[dict[str, Any], list[Any], list[Any], str]:
    if payload is None:
        payload = _load_json(args.from_json)
    if not isinstance(payload, dict):
        raise SystemExit("--from-json must point to a JSON object")

    stats = payload.get("stats") or payload.get("graph_state") or payload.get("graphState")
    if not isinstance(stats, dict):
        stats = payload

    recent_cycles = _extract_list(payload, "recent_cycles", "recentCycles", "cycles")
    if not recent_cycles:
        consolidate = payload.get("consolidate")
        if isinstance(consolidate, dict):
            latest = consolidate.get("latest_cycle") or consolidate.get("latestCycle")
            if latest:
                recent_cycles = [latest]
    calibration_snapshots = _extract_list(
        payload,
        "calibration_snapshots",
        "calibrationSnapshots",
    )

    group_id = (
        args.group_id
        or payload.get("group_id")
        or payload.get("groupId")
        or stats.get("group_id")
        or stats.get("groupId")
        or EngramConfig().default_group_id
    )
    return stats, recent_cycles, calibration_snapshots, str(group_id)


def _load_optional_samples(path: Path | None, payload: dict[str, Any], *keys: str) -> list[Any]:
    if path:
        return _list_payload(_load_json(path))
    return _extract_list(payload, *keys)
