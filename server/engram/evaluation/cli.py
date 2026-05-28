"""Command helpers for local brain-loop evaluation reports."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engram import __version__
from engram.axi.client import AxiRestClient, AxiRestError
from engram.config import EngramConfig
from engram.evaluation.adoption_evidence import (
    adoption_client_set_failure_message,
    adoption_evidence_failure_message,
    build_adoption_client_set_evidence,
    link_adoption_to_human_label_evidence,
    load_adoption_evidence,
)
from engram.evaluation.benchmark_evidence import (
    benchmark_evidence_failure_message,
    load_benchmark_evidence,
)
from engram.evaluation.brain_loop_report import (
    build_brain_loop_report,
    evaluation_signal_failure_message,
    format_brain_loop_report_markdown,
    format_memory_value_markdown,
    is_brain_loop_report_payload,
    looks_like_partial_brain_loop_report,
    memory_value_failure_message,
    merge_memory_operation_metrics,
    merge_recall_runtime_metrics,
    missing_brain_loop_report_sections,
    with_release_evidence_summary,
)
from engram.evaluation.human_label_evidence import (
    DEFAULT_HUMAN_RECALL_SAMPLE_GATE,
    DEFAULT_HUMAN_SESSION_SAMPLE_GATE,
    DEFAULT_RELEASE_HUMAN_RECALL_SAMPLE_GATE,
    DEFAULT_RELEASE_HUMAN_SESSION_SAMPLE_GATE,
    build_human_label_evidence_template,
    human_label_evidence_failure_message,
    load_human_label_evidence,
    render_human_label_evidence_template_markdown,
)
from engram.evaluation.store import SQLiteEvaluationStore
from engram.storage.bootstrap import (
    close_if_supported,
    create_consolidation_store_for_graph,
    create_evaluation_store_for_graph,
    create_local_runtime_stores,
)
from engram.storage.resolver import EngineMode, resolve_mode

EVALUATE_LIVE_STATS_TIMEOUT_SECONDS = 2.0


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
        "--server-url",
        help=(
            "Read the brain-loop report from a running Engram REST API instead "
            "of opening a local graph runtime. Use this for large native stores "
            "where the service already has warm cache-backed stats."
        ),
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for --server-url report reads.",
    )
    parser.add_argument(
        "--live-cost",
        action="store_true",
        help="Ask --server-url reports to include live memory operation cost counters.",
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
        "--require-release-evidence",
        action="store_true",
        help=(
            "Composite release gate: require measured evaluation signals, "
            "real human-label evidence, and a passed live-client adoption report."
        ),
    )
    parser.add_argument(
        "--memory-value",
        action="store_true",
        help="Print only the memory value/cost section of the brain-loop report.",
    )
    parser.add_argument(
        "--require-memory-value",
        action="store_true",
        help=(
            "Exit non-zero unless the report has measured memory value cost and "
            "benefit evidence."
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
        "--human-label-template-out",
        type=Path,
        help=(
            "When used with --human-label-template, write the JSON template to "
            "this path for operators to fill with real labels."
        ),
    )
    parser.add_argument(
        "--adoption-report",
        type=Path,
        help=(
            "JSON report from `engram adoption --format json` to prefill "
            "--human-label-template metadata, attach to the brain-loop evidence "
            "bundle, and optionally gate with --require-adoption-evidence."
        ),
    )
    parser.add_argument(
        "--additional-adoption-report",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional `engram adoption --format json` report to attach for "
            "multi-client release evidence. Repeat for Cursor/Windsurf/second-client "
            "diversity gates."
        ),
    )
    parser.add_argument(
        "--require-adoption-evidence",
        action="store_true",
        help=(
            "Exit non-zero unless --adoption-report is a passed live-client "
            "Engram adoption validation report. When human-label evidence is "
            "also attached, client/session metadata must point at the same run."
        ),
    )
    parser.add_argument(
        "--require-adoption-client",
        help=(
            "Expected live MCP client label for --adoption-report. The report "
            "must have been validated with `engram adoption --require-client` "
            "for this client and the observed live client must match."
        ),
    )
    parser.add_argument(
        "--require-adoption-clients",
        nargs="+",
        default=[],
        help=(
            "Expected live MCP client labels across --adoption-report plus any "
            "--additional-adoption-report values. Each client must have a measured "
            "report validated with matching `engram adoption --require-client`."
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
        default=None,
        help=(
            "Minimum human-labeled recall samples required when gating evidence. "
            f"Defaults to {DEFAULT_RELEASE_HUMAN_RECALL_SAMPLE_GATE} with "
            f"--require-release-evidence, otherwise {DEFAULT_HUMAN_RECALL_SAMPLE_GATE}."
        ),
    )
    parser.add_argument(
        "--min-human-session-samples",
        type=int,
        default=None,
        help=(
            "Minimum human-labeled session-continuity samples required when "
            "gating evidence. Defaults to "
            f"{DEFAULT_RELEASE_HUMAN_SESSION_SAMPLE_GATE} with "
            f"--require-release-evidence, otherwise {DEFAULT_HUMAN_SESSION_SAMPLE_GATE}."
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
    if args.server_url:
        if args.smoke:
            raise SystemExit("--server-url cannot be used with --smoke")
        if args.from_json:
            raise SystemExit("--server-url cannot be used with --from-json")
        if args.group_id:
            raise SystemExit("--server-url uses the running server tenant; omit --group-id")
        return _load_server_report(args)

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


def _load_server_report(args: argparse.Namespace) -> dict[str, Any]:
    client = AxiRestClient(
        server_url=args.server_url,
        timeout_seconds=max(0.1, float(args.server_timeout)),
    )
    try:
        report = client.evaluation_report(
            live_cost=bool(getattr(args, "live_cost", False)),
            cycle_limit=max(1, int(args.cycles)),
            sample_limit=max(1, int(args.saved_sample_limit)),
        )
    except AxiRestError as exc:
        raise SystemExit(f"Failed to load server evaluation report: {exc}") from exc
    if is_brain_loop_report_payload(report):
        return dict(report)
    if looks_like_partial_brain_loop_report(report):
        raise SystemExit(
            "--server-url returned a partial brain-loop report missing "
            "required report sections: "
            f"{missing_brain_loop_report_sections(report)}"
        )
    raise SystemExit("--server-url did not return a brain-loop report")


async def run_evaluate_command(args: argparse.Namespace) -> None:
    """Print a brain-loop report for parsed CLI arguments."""
    if getattr(args, "human_label_template", False):
        if getattr(args, "additional_adoption_report", None) and not getattr(
            args,
            "adoption_report",
            None,
        ):
            raise SystemExit(
                "--additional-adoption-report requires --adoption-report "
                "when generating a human-label template"
            )
        adoption_evidence = None
        if getattr(args, "adoption_report", None):
            try:
                adoption_evidence = load_adoption_evidence(
                    args.adoption_report,
                    required_client=getattr(args, "require_adoption_client", None),
                )
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                raise SystemExit(
                    f"Invalid adoption report {args.adoption_report}: {exc}"
                ) from exc
            failure_message = adoption_evidence_failure_message(
                adoption_evidence,
                prefix="Adoption evidence failed gates",
            )
            if failure_message:
                raise SystemExit(failure_message)
        additional_adoption_evidences = [
            _load_adoption_report_evidence(path)
            for path in getattr(args, "additional_adoption_report", None) or []
        ]
        required_adoption_clients = _required_adoption_clients(args)
        if additional_adoption_evidences or required_adoption_clients:
            client_set_evidence = build_adoption_client_set_evidence(
                [
                    evidence
                    for evidence in [adoption_evidence, *additional_adoption_evidences]
                    if evidence is not None
                ],
                required_clients=required_adoption_clients,
            )
            failure_message = adoption_client_set_failure_message(
                client_set_evidence,
                prefix="Adoption client evidence failed gates",
            )
            if failure_message:
                raise SystemExit(failure_message)
        template = build_human_label_evidence_template(
            adoption_evidence=adoption_evidence,
            adoption_report_path=getattr(args, "adoption_report", None),
            required_adoption_client=getattr(args, "require_adoption_client", None),
            additional_adoption_evidences=additional_adoption_evidences,
            required_adoption_clients=required_adoption_clients,
        )
        _write_human_label_template_output(
            template,
            getattr(args, "human_label_template_out", None),
        )
        if args.format == "json":
            print(json.dumps(template, indent=2, sort_keys=True))
            return
        print(render_human_label_evidence_template_markdown(template), end="")
        return
    if getattr(args, "human_label_template_out", None) is not None:
        raise SystemExit("--human-label-template-out requires --human-label-template")

    report = await build_report_from_args(args)
    report = _attach_benchmark_evidence_from_args(report, args)
    report = _attach_human_label_evidence_from_args(report, args)
    report = _attach_adoption_evidence_from_args(report, args)
    report = link_adoption_to_human_label_evidence(report)
    report = with_release_evidence_summary(report)
    require_release = bool(getattr(args, "require_release_evidence", False))
    if require_release or getattr(args, "require_evaluation_signals", False):
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
    if getattr(args, "require_memory_value", False):
        failure_message = memory_value_failure_message(
            report,
            prefix="Memory value evidence failed gates",
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
    if require_release or getattr(args, "require_human_label_evidence", False):
        failure_message = human_label_evidence_failure_message(
            report.get("human_label_evidence"),
            prefix="Human label evidence failed gates",
        )
        if failure_message:
            raise SystemExit(failure_message)
    if (
        require_release
        or getattr(args, "require_adoption_evidence", False)
        or bool(getattr(args, "require_adoption_client", None))
    ):
        failure_message = adoption_evidence_failure_message(
            report.get("adoption_evidence"),
            prefix="Adoption evidence failed gates",
        )
        if failure_message:
            raise SystemExit(failure_message)
    if _should_gate_adoption_client_set(args, require_release) and report.get(
        "adoption_client_evidence"
    ):
        failure_message = adoption_client_set_failure_message(
            report.get("adoption_client_evidence"),
            prefix="Adoption client evidence failed gates",
        )
        if failure_message:
            raise SystemExit(failure_message)
    _write_evidence_bundle_from_args(report, args)
    if getattr(args, "memory_value", False):
        if args.format == "json":
            print(
                json.dumps(
                    {
                        "group_id": report.get("group_id"),
                        "generated_at": report.get("generated_at"),
                        "memory_value": report.get("memory_value"),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        print(format_memory_value_markdown(report), end="")
        return
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


def _write_human_label_template_output(
    template: dict[str, Any],
    output_path: Path | None,
) -> None:
    if output_path is None:
        return
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
            min_recall_samples=_effective_min_human_recall_samples(args),
            min_session_samples=_effective_min_human_session_samples(args),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid human label artifact {artifact_path}: {exc}") from exc
    report = dict(report)
    report["human_label_evidence"] = evidence
    return report


def _attach_adoption_evidence_from_args(
    report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    artifact_path = getattr(args, "adoption_report", None)
    additional_paths = list(getattr(args, "additional_adoption_report", None) or [])
    required_clients = _required_adoption_clients(args)
    require_evidence = getattr(args, "require_adoption_evidence", False) or bool(
        getattr(args, "require_adoption_client", None)
    )
    if artifact_path is None:
        if require_evidence:
            report = dict(report)
            report["adoption_evidence"] = None
        if additional_paths or required_clients:
            report = _attach_adoption_client_set(report, [], args)
        return report
    evidence = _load_adoption_report_evidence(
        artifact_path,
        required_client=getattr(args, "require_adoption_client", None),
    )
    report = dict(report)
    report["adoption_evidence"] = evidence
    if additional_paths or required_clients:
        report = _attach_adoption_client_set(report, [evidence], args)
    return report


def _attach_adoption_client_set(
    report: dict[str, Any],
    primary_evidences: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    evidences = list(primary_evidences)
    for path in getattr(args, "additional_adoption_report", None) or []:
        evidences.append(_load_adoption_report_evidence(path))
    report = dict(report)
    if len(evidences) > len(primary_evidences):
        report["additional_adoption_evidence"] = evidences[len(primary_evidences) :]
    report["adoption_client_evidence"] = build_adoption_client_set_evidence(
        evidences,
        required_clients=_required_adoption_clients(args),
    )
    return report


def _load_adoption_report_evidence(
    artifact_path: Path,
    *,
    required_client: str | None = None,
) -> dict[str, Any]:
    try:
        return load_adoption_evidence(
            artifact_path,
            required_client=required_client,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid adoption report {artifact_path}: {exc}") from exc


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
    sources = {
        "report_json": _optional_path_str(getattr(args, "from_json", None)),
        "benchmark_artifact": _optional_path_str(getattr(args, "benchmark_artifact", None)),
        "human_label_artifact": _optional_path_str(
            getattr(args, "human_label_artifact", None)
        ),
        "adoption_report": _optional_path_str(getattr(args, "adoption_report", None)),
        "additional_adoption_reports": _optional_paths(
            getattr(args, "additional_adoption_report", None)
        ),
        "recall_samples": _optional_path_str(getattr(args, "recall_samples", None)),
        "session_samples": _optional_path_str(getattr(args, "session_samples", None)),
        "server_url": getattr(args, "server_url", None),
        "sqlite_path": _optional_path_str(getattr(args, "sqlite_path", None)),
        "helix_data_dir": _optional_path_str(getattr(args, "helix_data_dir", None)),
    }
    return {
        "kind": "engram_brain_loop_evidence_bundle",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": _evidence_bundle_status(args),
        "gate_profile": _evidence_bundle_gate_profile(args),
        "release_ready": _evidence_bundle_release_ready(report, args),
        "group_id": report.get("group_id"),
        "provenance": {
            "engram_version": __version__,
            "git": _git_metadata(),
        },
        "sources": sources,
        "source_sha256": _source_sha256_map(sources),
        "gates": {
            "require_evaluation_signals": bool(
                getattr(args, "require_evaluation_signals", False)
            ),
            "require_release_evidence": bool(
                getattr(args, "require_release_evidence", False)
            ),
            "require_memory_value": bool(getattr(args, "require_memory_value", False)),
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
            "require_adoption_evidence": bool(
                getattr(args, "require_adoption_evidence", False)
            ),
            "require_adoption_client": getattr(args, "require_adoption_client", None),
            "require_adoption_clients": _required_adoption_clients(args),
            "min_human_recall_samples": max(
                0,
                _effective_min_human_recall_samples(args),
            ),
            "min_human_session_samples": max(
                0,
                _effective_min_human_session_samples(args),
            ),
        },
        "report": report,
    }


def _optional_path_str(path: Any) -> str | None:
    return str(path) if path is not None else None


def _optional_paths(paths: Any) -> list[str]:
    return [str(path) for path in paths or []]


def _source_sha256_map(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        name: _file_sha256(path)
        for name, path in sources.items()
        if name not in {"sqlite_path", "helix_data_dir", "server_url"}
    }


def _evidence_bundle_status(args: argparse.Namespace) -> str:
    return "passed" if _evidence_bundle_has_requested_gate(args) else "recorded"


def _evidence_bundle_gate_profile(args: argparse.Namespace) -> str:
    if getattr(args, "require_release_evidence", False):
        return "release"
    if getattr(args, "require_memory_value", False):
        return "memory_value"
    if (
        getattr(args, "require_benchmark_evidence", False)
        or getattr(args, "require_human_label_evidence", False)
        or getattr(args, "require_adoption_evidence", False)
        or getattr(args, "require_adoption_client", None)
        or getattr(args, "require_adoption_clients", None)
    ):
        return "evidence"
    if getattr(args, "require_evaluation_signals", False):
        return "evaluation"
    return "record_only"


def _evidence_bundle_release_ready(
    report: Mapping[str, Any],
    args: argparse.Namespace,
) -> bool:
    if not getattr(args, "require_release_evidence", False):
        return False
    release = report.get("release_evidence")
    return isinstance(release, Mapping) and release.get("status") == "measured"


def _evidence_bundle_has_requested_gate(args: argparse.Namespace) -> bool:
    return any(
        (
            getattr(args, "require_evaluation_signals", False),
            getattr(args, "require_release_evidence", False),
            getattr(args, "require_memory_value", False),
            getattr(args, "require_benchmark_evidence", False),
            getattr(args, "require_human_label_evidence", False),
            getattr(args, "require_adoption_evidence", False),
            bool(getattr(args, "require_adoption_client", None)),
            bool(getattr(args, "require_adoption_clients", None)),
        )
    )


def _file_sha256(path: Any) -> Any:
    if path is None:
        return None
    if isinstance(path, list):
        return [_file_sha256(item) for item in path]
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()


def _git_metadata() -> dict[str, Any]:
    """Return best-effort repository metadata for release evidence bundles."""
    root = _run_git("rev-parse", "--show-toplevel", cwd=Path.cwd())
    if root is None:
        return {"available": False}

    repo = Path(root)
    commit = _run_git("rev-parse", "HEAD", cwd=repo)
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo)
    status_short = _run_git("status", "--short", cwd=repo) or ""
    status_lines = [line for line in status_short.splitlines() if line.strip()]
    return {
        "available": True,
        "root": str(repo),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "status_short": status_lines,
    }


def _run_git(*args: str, cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _required_adoption_clients(args: argparse.Namespace) -> list[str]:
    clients = list(getattr(args, "require_adoption_clients", None) or [])
    singular = getattr(args, "require_adoption_client", None)
    if singular:
        clients.insert(0, singular)
    return _dedupe_strings(clients)


def _should_gate_adoption_client_set(
    args: argparse.Namespace,
    require_release: bool,
) -> bool:
    return (
        require_release
        or getattr(args, "require_adoption_evidence", False)
        or bool(getattr(args, "require_adoption_client", None))
        or bool(getattr(args, "require_adoption_clients", None))
    )


def _dedupe_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if not label:
            continue
        normalized = " ".join(label.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(label)
    return result


def _effective_min_human_recall_samples(args: argparse.Namespace) -> int:
    value = getattr(args, "min_human_recall_samples", None)
    if value is None:
        value = (
            DEFAULT_RELEASE_HUMAN_RECALL_SAMPLE_GATE
            if getattr(args, "require_release_evidence", False)
            else DEFAULT_HUMAN_RECALL_SAMPLE_GATE
        )
    return max(0, int(value))


def _effective_min_human_session_samples(args: argparse.Namespace) -> int:
    value = getattr(args, "min_human_session_samples", None)
    if value is None:
        value = (
            DEFAULT_RELEASE_HUMAN_SESSION_SAMPLE_GATE
            if getattr(args, "require_release_evidence", False)
            else DEFAULT_HUMAN_SESSION_SAMPLE_GATE
        )
    return max(0, int(value))


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
    graph_store, activation_store, search_index = create_local_runtime_stores(
        mode,
        config,
    )
    consolidation_store: Any | None = None
    evaluation_store: SQLiteEvaluationStore | None = None

    await graph_store.initialize()
    try:
        consolidation_store = await _create_consolidation_store(mode, config, graph_store)
        stats = await _load_live_stats_bounded(
            graph_store,
            group_id,
            timeout_seconds=EVALUATE_LIVE_STATS_TIMEOUT_SECONDS,
        )
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
            stats = merge_memory_operation_metrics(
                stats,
                await evaluation_store.get_latest_memory_operation_metrics_snapshot(group_id),
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
        await close_if_supported(evaluation_store)
        await close_if_supported(consolidation_store)
        await close_if_supported(search_index)
        await close_if_supported(activation_store)
        await close_if_supported(graph_store)


async def _load_live_stats_bounded(
    graph_store: Any,
    group_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    try:
        return await asyncio.wait_for(
            graph_store.get_stats(group_id),
            timeout=max(0.01, timeout_seconds),
        )
    except TimeoutError:
        return {
            "evaluation_degradations": [
                {
                    "stage": "graph_stats",
                    "status": "degraded",
                    "skip_reason": "graph_stats_timeout",
                    "timeout_ms": round(max(0.01, timeout_seconds) * 1000),
                }
            ]
        }


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
