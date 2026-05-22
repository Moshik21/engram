"""Local dogfood transcript replay and mode-comparison reporting."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engram.config import ActivationConfig
from engram.evaluation.human_label_evidence import (
    HUMAN_LABEL_EVIDENCE_KIND,
    build_human_label_evidence,
    human_label_evidence_failure_message,
    load_human_label_evidence,
)
from engram.evaluation.store import (
    SQLiteEvaluationStore,
    StoredMemoryOperationMetricsSnapshot,
    StoredRecallEvalSample,
    StoredSessionContinuitySample,
)
from engram.retrieval.budgets import recall_budget_for_profile
from engram.retrieval.need import analyze_memory_need

DOGFOOD_REPORT_KIND = "engram.dogfood_replay.v1"
DEFAULT_DOGFOOD_MODES = ("off", "startup", "cached", "gated_lite", "gated_medium", "deep")
SUPPORTED_DOGFOOD_MODES = frozenset(DEFAULT_DOGFOOD_MODES)
_DOGFOOD_PLACEHOLDER_TOKEN_RE = re.compile(r"<[^>\n]{1,120}>")


@dataclass(frozen=True)
class DogfoodTurn:
    """One parsed transcript turn with redaction-safe identity."""

    role: str
    content: str
    timestamp: str | None = None
    source: str = "transcript"
    metadata: dict[str, Any] | None = None

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


def configure_dogfood_parser(parser: argparse.ArgumentParser) -> None:
    """Configure `engram dogfood` commands."""
    parser.description = "Replay real local transcripts through Engram mode decisions"
    subparsers = parser.add_subparsers(dest="dogfood_command", required=True)
    replay = subparsers.add_parser("replay", help="Build a private dogfood replay report")
    replay.add_argument("--transcript", type=Path, required=True)
    replay.add_argument(
        "--trace",
        type=Path,
        action="append",
        default=[],
        help="Optional AXI hook trace JSONL to merge into replay cost evidence.",
    )
    replay.add_argument("--project", dest="project_path", default=None)
    replay.add_argument(
        "--modes",
        default=",".join(DEFAULT_DOGFOOD_MODES),
        help="Comma-separated modes: off,startup,cached,gated_lite,gated_medium,deep",
    )
    replay.add_argument("--group-id", default="default")
    replay.add_argument("--out", type=Path, default=None)
    replay.add_argument(
        "--label-template",
        action="store_true",
        help="Include an opt-in human-label template in the replay report.",
    )
    replay.add_argument(
        "--label-template-out",
        type=Path,
        default=None,
        help="Write only the opt-in human-label template to this JSON path.",
    )
    replay.add_argument(
        "--label-template-include-content",
        action="store_true",
        help=(
            "Explicitly include user-turn content in the label template for local "
            "human review. The default template remains redacted."
        ),
    )
    replay.add_argument("--include-content", action="store_true")
    replay.add_argument("--format", choices=["json", "markdown"], default="markdown")

    prepare = subparsers.add_parser(
        "prepare",
        help="Create a replay report plus fillable dogfood label-review bundle",
    )
    prepare.add_argument("--transcript", type=Path, required=True)
    prepare.add_argument(
        "--trace",
        type=Path,
        action="append",
        default=[],
        help="Optional AXI hook trace JSONL to merge into replay cost evidence.",
    )
    prepare.add_argument("--project", dest="project_path", default=None)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument(
        "--modes",
        default=",".join(DEFAULT_DOGFOOD_MODES),
        help="Comma-separated modes: off,startup,cached,gated_lite,gated_medium,deep",
    )
    prepare.add_argument("--group-id", default="default")
    prepare.add_argument("--min-recall-samples", type=int, default=1)
    prepare.add_argument("--min-session-samples", type=int, default=1)
    prepare.add_argument(
        "--label-template-include-content",
        action="store_true",
        help=(
            "Explicitly include user-turn content in the label template for local "
            "human review. The replay report remains redacted."
        ),
    )
    prepare.add_argument("--format", choices=["json", "markdown"], default="markdown")

    import_labels = subparsers.add_parser(
        "import-labels",
        help="Import opted-in dogfood review labels into the evaluation store",
    )
    import_labels.add_argument("--labels", type=Path, required=True)
    import_labels.add_argument("--sqlite-path", type=Path, default=None)
    import_labels.add_argument("--group-id", default=None)
    import_labels.add_argument(
        "--include-all-modes",
        action="store_true",
        help="Import every replayed mode for labeled turns instead of only labeled modes.",
    )
    import_labels.add_argument("--dry-run", action="store_true")
    import_labels.add_argument("--format", choices=["json", "markdown"], default="markdown")

    review = subparsers.add_parser(
        "review",
        help="Summarize dogfood label review readiness without importing",
    )
    review.add_argument("--labels", type=Path, required=True)
    review.add_argument("--group-id", default=None)
    review.add_argument("--min-recall-samples", type=int, default=1)
    review.add_argument("--min-session-samples", type=int, default=1)
    review.add_argument(
        "--need-type",
        default=None,
        help=(
            "Only suggest turn-label commands for this review need type "
            "(for example open_loop, fact_lookup, project_state)."
        ),
    )
    review.add_argument(
        "--command-limit",
        type=int,
        default=10,
        help="Maximum suggested turn-label commands to include in review output.",
    )
    review.add_argument(
        "--include-all-modes",
        action="store_true",
        help="Count every replayed mode for labeled turns instead of only labeled modes.",
    )
    review.add_argument(
        "--include-content",
        action="store_true",
        help=(
            "Explicitly inline local transcript content for the suggested review "
            "turns. Default review output stays redacted."
        ),
    )
    review.add_argument(
        "--context",
        type=int,
        default=1,
        help=(
            "Neighboring parsed transcript turns to include when --include-content "
            "is used."
        ),
    )
    review.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless reviewed labels are import-ready.",
    )
    review.add_argument("--format", choices=["json", "markdown"], default="markdown")

    inspect_turn = subparsers.add_parser(
        "inspect-turn",
        help="Inspect one local transcript turn behind a dogfood label template",
    )
    inspect_turn.add_argument("--labels", type=Path, required=True)
    target = inspect_turn.add_mutually_exclusive_group(required=True)
    target.add_argument("--turn", type=int, default=None)
    target.add_argument("--content-hash", default=None)
    target.add_argument(
        "--next",
        action="store_true",
        help="Inspect the next unreviewed/invalid turn in the dogfood review queue.",
    )
    inspect_turn.add_argument(
        "--need-type",
        default=None,
        help="When used with --next, inspect the next queued turn for this need type.",
    )
    inspect_turn.add_argument(
        "--context",
        type=int,
        default=1,
        help="Number of neighboring parsed transcript turns to include.",
    )
    inspect_turn.add_argument(
        "--include-content",
        action="store_true",
        help="Explicitly print local transcript content for human review.",
    )
    inspect_turn.add_argument("--format", choices=["json", "markdown"], default="markdown")

    scan = subparsers.add_parser(
        "scan",
        help="Find local transcript files with labelable dogfood turns",
    )
    scan.add_argument("--root", type=Path, required=True)
    scan.add_argument("--project", dest="project_path", default=None)
    scan.add_argument("--limit", type=int, default=20)
    scan.add_argument("--max-files", type=int, default=500)
    scan.add_argument("--min-turns", type=int, default=1)
    scan.add_argument(
        "--project-only",
        action="store_true",
        help="Only include transcripts whose recorded cwd is inside --project.",
    )
    scan.add_argument(
        "--include-trace-only",
        action="store_true",
        help="Include trace-only files in the top-level candidate list.",
    )
    scan.add_argument("--format", choices=["json", "markdown"], default="markdown")

    label_turn = subparsers.add_parser(
        "label-turn",
        help="Update one reviewed recall turn in a dogfood label template",
    )
    label_turn.add_argument("--labels", type=Path, required=True)
    label_turn.add_argument("--turn", type=int, required=True)
    label_turn.add_argument(
        "--memory-needed",
        choices=["yes", "no", "true", "false"],
        required=True,
    )
    label_turn.add_argument(
        "--best-mode",
        default=None,
        help="Best replay mode for this turn, or 'none'.",
    )
    label_turn.add_argument("--helpful-mode", action="append", default=[])
    label_turn.add_argument("--false-recall-mode", action="append", default=[])
    label_turn.add_argument("--stale-mode", action="append", default=[])
    label_turn.add_argument("--corrected-mode", action="append", default=[])
    label_turn.add_argument("--notes", default=None)
    label_turn.add_argument("--out", type=Path, default=None)
    label_turn.add_argument("--format", choices=["json", "markdown"], default="markdown")

    label_session = subparsers.add_parser(
        "label-session",
        help="Append one reviewed session-continuity label to a dogfood template",
    )
    label_session.add_argument("--labels", type=Path, required=True)
    label_session.add_argument("--scenario", required=True)
    label_session.add_argument("--baseline-score", type=float, required=True)
    label_session.add_argument("--memory-score", type=float, required=True)
    label_session.add_argument("--open-loop-expected", action="store_true")
    label_session.add_argument("--open-loop-recovered", action="store_true")
    label_session.add_argument("--temporal-expected", action="store_true")
    label_session.add_argument("--temporal-correct", action="store_true")
    label_session.add_argument("--notes", default=None)
    label_session.add_argument("--out", type=Path, default=None)
    label_session.add_argument("--format", choices=["json", "markdown"], default="markdown")

    export_evidence = subparsers.add_parser(
        "export-evidence",
        help="Convert reviewed dogfood labels into a human-label evidence artifact",
    )
    export_evidence.add_argument("--labels", type=Path, required=True)
    export_evidence.add_argument("--out", type=Path, required=True)
    export_evidence.add_argument("--source", required=True, help="Real harness/source name")
    export_evidence.add_argument("--client", required=True, help="Real client label")
    export_evidence.add_argument("--captured-at", required=True, help="ISO-8601 capture time")
    export_evidence.add_argument("--labeler", required=True, help="Human reviewer")
    export_evidence.add_argument("--session-id", default=None)
    export_evidence.add_argument("--group-id", default=None)
    export_evidence.add_argument(
        "--include-all-modes",
        action="store_true",
        help="Export every replayed mode for labeled turns instead of only labeled modes.",
    )
    export_evidence.add_argument("--format", choices=["json", "markdown"], default="markdown")

    closeout = subparsers.add_parser(
        "closeout",
        help="Print the native dogfood evidence closeout checklist",
    )
    closeout.add_argument("--labels", type=Path, required=True)
    closeout.add_argument("--human-label-artifact", type=Path, default=None)
    closeout.add_argument("--sqlite-path", type=Path, default=None)
    closeout.add_argument("--group-id", default=None)
    closeout.add_argument(
        "--mode",
        choices=["helix", "lite", "auto"],
        default="helix",
        help="Evaluation mode for the final memory-value command.",
    )
    closeout.add_argument("--helix-data-dir", type=Path, default=None)
    closeout.add_argument("--min-recall-samples", type=int, default=1)
    closeout.add_argument("--min-session-samples", type=int, default=1)
    closeout.add_argument(
        "--include-all-modes",
        action="store_true",
        help="Count every replayed mode for labeled turns instead of only labeled modes.",
    )
    closeout.add_argument(
        "--require-ready",
        action="store_true",
        help="Exit non-zero unless reviewed labels and human-label evidence are ready.",
    )
    closeout.add_argument("--format", choices=["json", "markdown"], default="markdown")

    finalize = subparsers.add_parser(
        "finalize",
        help="Review, import, export, and run the native dogfood memory-value gate",
    )
    finalize.add_argument("--labels", type=Path, required=True)
    finalize.add_argument(
        "--replay-report",
        type=Path,
        default=None,
        help=(
            "Optional dogfood replay report whose measured AXI trace evidence "
            "is persisted as memory operation cost metrics before evaluation."
        ),
    )
    finalize.add_argument(
        "--human-label-artifact",
        type=Path,
        default=Path("human-labels.json"),
        help="Output path for the exported human-label evidence artifact.",
    )
    finalize.add_argument("--sqlite-path", type=Path, default=None)
    finalize.add_argument("--source", required=True, help="Real harness/source name")
    finalize.add_argument("--client", required=True, help="Real client label")
    finalize.add_argument("--captured-at", required=True, help="ISO-8601 capture time")
    finalize.add_argument("--labeler", required=True, help="Human reviewer")
    finalize.add_argument("--session-id", default=None)
    finalize.add_argument("--group-id", default=None)
    finalize.add_argument(
        "--mode",
        choices=["helix", "lite", "auto"],
        default="helix",
        help="Evaluation mode for the final memory-value command.",
    )
    finalize.add_argument("--helix-data-dir", type=Path, default=None)
    finalize.add_argument("--min-recall-samples", type=int, default=1)
    finalize.add_argument("--min-session-samples", type=int, default=1)
    finalize.add_argument(
        "--include-all-modes",
        action="store_true",
        help="Import/export every replayed mode for labeled turns.",
    )
    finalize.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Stop after import/export/closeout without running `engram evaluate`.",
    )
    finalize.add_argument(
        "--evaluate-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for the final memory-value evaluation subprocess.",
    )
    finalize.add_argument("--format", choices=["json", "markdown"], default="markdown")


async def run_dogfood_command(args: argparse.Namespace) -> int:
    """Run a dogfood command and print/write the report."""
    if args.dogfood_command == "prepare":
        report = await prepare_dogfood_review_bundle(
            transcript_path=args.transcript,
            trace_paths=list(args.trace or []),
            project_path=args.project_path,
            output_dir=args.out_dir,
            group_id=args.group_id,
            modes=parse_modes(args.modes),
            include_template_content=bool(args.label_template_include_content),
            min_recall_samples=max(0, int(args.min_recall_samples)),
            min_session_samples=max(0, int(args.min_session_samples)),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_prepare_markdown(report), end="")
        return 0
    if args.dogfood_command == "finalize":
        report = await finalize_dogfood_labels(
            labels_path=args.labels,
            replay_report=args.replay_report,
            human_label_artifact=args.human_label_artifact,
            sqlite_path=args.sqlite_path,
            source=args.source,
            client=args.client,
            captured_at=args.captured_at,
            labeler=args.labeler,
            session_id=args.session_id,
            group_id=args.group_id,
            mode=args.mode,
            helix_data_dir=args.helix_data_dir,
            min_recall_samples=max(0, int(args.min_recall_samples)),
            min_session_samples=max(0, int(args.min_session_samples)),
            include_all_modes=bool(args.include_all_modes),
            skip_evaluate=bool(args.skip_evaluate),
            evaluate_timeout_seconds=max(0.1, float(args.evaluate_timeout_seconds)),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_finalize_markdown(report), end="")
        return 0 if report.get("ready") is True else 1
    if args.dogfood_command == "review":
        report = build_dogfood_review_report(
            labels_path=args.labels,
            group_id=args.group_id,
            min_recall_samples=max(0, int(args.min_recall_samples)),
            min_session_samples=max(0, int(args.min_session_samples)),
            include_all_modes=bool(args.include_all_modes),
            need_type=getattr(args, "need_type", None),
            command_limit=max(0, int(getattr(args, "command_limit", 10))),
            include_content=bool(getattr(args, "include_content", False)),
            context=max(0, int(getattr(args, "context", 1))),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_review_markdown(report), end="")
        if bool(getattr(args, "require_ready", False)) and report.get("ready") is not True:
            return 1
        return 0
    if args.dogfood_command == "inspect-turn":
        report = build_dogfood_turn_inspection_report(
            labels_path=args.labels,
            turn_index=args.turn,
            content_hash=args.content_hash,
            next_unreviewed=bool(getattr(args, "next", False)),
            need_type=getattr(args, "need_type", None),
            context=max(0, int(args.context)),
            include_content=bool(args.include_content),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_turn_inspection_markdown(report), end="")
        return 0 if report.get("status") == "ready" else 1
    if args.dogfood_command == "scan":
        report = build_dogfood_candidate_report(
            root_path=args.root,
            project_path=args.project_path,
            limit=max(1, int(args.limit)),
            max_files=max(1, int(args.max_files)),
            min_turns=max(1, int(args.min_turns)),
            include_trace_only=bool(args.include_trace_only),
            project_only=bool(args.project_only),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_candidate_markdown(report), end="")
        return 0
    if args.dogfood_command == "label-turn":
        report = update_dogfood_turn_label(
            labels_path=args.labels,
            turn_index=int(args.turn),
            memory_needed=_parse_label_bool(args.memory_needed),
            best_mode=args.best_mode,
            helpful_modes=list(args.helpful_mode or []),
            false_recall_modes=list(args.false_recall_mode or []),
            stale_modes=list(args.stale_mode or []),
            corrected_modes=list(args.corrected_mode or []),
            notes=args.notes,
            output_path=args.out,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_label_edit_markdown(report), end="")
        return 0
    if args.dogfood_command == "label-session":
        report = add_dogfood_session_label(
            labels_path=args.labels,
            scenario=args.scenario,
            baseline_score=float(args.baseline_score),
            memory_score=float(args.memory_score),
            open_loop_expected=bool(args.open_loop_expected),
            open_loop_recovered=bool(args.open_loop_recovered),
            temporal_expected=bool(args.temporal_expected),
            temporal_correct=bool(args.temporal_correct),
            notes=args.notes,
            output_path=args.out,
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_label_edit_markdown(report), end="")
        return 0
    if args.dogfood_command == "closeout":
        report = build_dogfood_closeout_report(
            labels_path=args.labels,
            human_label_artifact=args.human_label_artifact,
            sqlite_path=args.sqlite_path,
            group_id=args.group_id,
            mode=args.mode,
            helix_data_dir=args.helix_data_dir,
            min_recall_samples=max(0, int(args.min_recall_samples)),
            min_session_samples=max(0, int(args.min_session_samples)),
            include_all_modes=bool(args.include_all_modes),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_closeout_markdown(report), end="")
        if bool(getattr(args, "require_ready", False)) and report.get("ready") is not True:
            return 1
        return 0
    if args.dogfood_command == "export-evidence":
        report = export_dogfood_human_label_evidence(
            labels_path=args.labels,
            output_path=args.out,
            source=args.source,
            client=args.client,
            captured_at=args.captured_at,
            labeler=args.labeler,
            session_id=args.session_id,
            group_id=args.group_id,
            include_all_modes=bool(args.include_all_modes),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_export_markdown(report), end="")
        return 0 if report.get("status") == "exported" else 1
    if args.dogfood_command == "import-labels":
        report = await import_dogfood_label_artifact(
            labels_path=args.labels,
            sqlite_path=args.sqlite_path,
            group_id=args.group_id,
            include_all_modes=bool(args.include_all_modes),
            dry_run=bool(args.dry_run),
        )
        if args.format == "json":
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_dogfood_import_markdown(report), end="")
        return 0 if report.get("status") in {"imported", "dry_run"} else 1
    if args.dogfood_command != "replay":
        raise SystemExit(f"Unsupported dogfood command: {args.dogfood_command}")
    report = await build_dogfood_replay_report(
        transcript_path=args.transcript,
        trace_paths=list(getattr(args, "trace", None) or []),
        project_path=args.project_path,
        group_id=args.group_id,
        modes=parse_modes(args.modes),
        include_content=bool(args.include_content),
    )
    label_template = None
    if args.label_template or args.label_template_out:
        content_lookup = (
            _dogfood_turn_content_lookup(args.transcript)
            if bool(getattr(args, "label_template_include_content", False))
            else None
        )
        label_template = build_dogfood_label_template(
            report,
            include_content=bool(getattr(args, "label_template_include_content", False)),
            content_lookup=content_lookup,
        )
        report["label_template"] = label_template
        report["labels"]["template_available"] = True
    if args.label_template_out and label_template is not None:
        args.label_template_out.parent.mkdir(parents=True, exist_ok=True)
        args.label_template_out.write_text(
            json.dumps(label_template, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_dogfood_replay_markdown(report), end="")
    return 0


async def build_dogfood_replay_report(
    *,
    transcript_path: Path,
    trace_paths: list[Path] | None = None,
    project_path: str | None,
    group_id: str,
    modes: list[str],
    include_content: bool = False,
) -> dict[str, Any]:
    raw_text = transcript_path.read_text(encoding="utf-8")
    turns = parse_transcript_text(raw_text, source=str(transcript_path))
    resolved_trace_paths = list(trace_paths or [])
    trace_turns = _dogfood_trace_turns_from_paths(resolved_trace_paths)
    turns_with_traces = [*turns, *trace_turns]
    user_turns = [turn for turn in turns if turn.role == "user"]
    trace_evidence = build_dogfood_trace_evidence(turns_with_traces)
    cfg = ActivationConfig()
    replay_turns = []
    mode_summaries = {mode: _empty_mode_summary(mode) for mode in modes}

    for index, turn in enumerate(user_turns):
        recent_turns = [prior.content for prior in user_turns[max(0, index - 3) : index]]
        started = time.perf_counter()
        need = await analyze_memory_need(
            turn.content,
            recent_turns=recent_turns,
            mode="dogfood_replay",
            group_id=group_id,
            cfg=cfg,
        )
        analyzer_duration_ms = round((time.perf_counter() - started) * 1000, 4)
        decisions = [
            _mode_decision(
                mode,
                need_should_recall=need.should_recall,
                skip_reason=_skip_reason(need.reasons),
                cfg=cfg,
                analyzer_duration_ms=analyzer_duration_ms,
            )
            for mode in modes
        ]
        for decision in decisions:
            summary = mode_summaries[decision["mode"]]
            summary["turn_count"] += 1
            if decision["decision"] == "triggered":
                summary["triggered_count"] += 1
            else:
                summary["skipped_count"] += 1
            summary["estimated_latency_ms"] += decision["estimated_latency_ms"]

        replay_turns.append(
            {
                "index": index,
                "role": turn.role,
                "content_hash": turn.content_hash,
                **({"content": turn.content} if include_content else {}),
                "timestamp": turn.timestamp,
                "need": {
                    "need_type": need.need_type,
                    "should_recall": need.should_recall,
                    "confidence": round(need.confidence, 4),
                    "reasons": list(need.reasons),
                    "query_hint": need.query_hint if include_content else None,
                    "query_hint_redacted": bool(need.query_hint and not include_content),
                    "analyzer_latency_ms": round(need.analyzer_latency_ms, 4),
                },
                "decisions": decisions,
            }
        )

    for summary in mode_summaries.values():
        turn_count = max(1, summary["turn_count"])
        summary["trigger_rate"] = round(summary["triggered_count"] / turn_count, 4)
        summary["avg_estimated_latency_ms"] = round(
            summary.pop("estimated_latency_ms") / turn_count,
            4,
        )

    status = (
        "measured"
        if replay_turns or trace_evidence.get("status") == "measured"
        else "empty"
    )
    return {
        "kind": DOGFOOD_REPORT_KIND,
        "status": status,
        "source": {
            "path": str(transcript_path),
            "trace_paths": [str(path) for path in resolved_trace_paths],
            "transcript_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            "turn_count": len(turns_with_traces),
            "user_turn_count": len(user_turns),
            "content_redacted": not include_content,
        },
        "project_path": project_path,
        "group_id": group_id,
        "modes": modes,
        "mode_summaries": mode_summaries,
        "trace_evidence": trace_evidence,
        "turns": replay_turns,
        "labels": {
            "status": "template_only",
            "opt_in_required": True,
            "template_available": False,
            "note": "Replay decisions are not saved as evaluation labels automatically.",
        },
    }


async def prepare_dogfood_review_bundle(
    *,
    transcript_path: Path,
    trace_paths: list[Path] | None = None,
    project_path: str | None,
    output_dir: Path,
    group_id: str,
    modes: list[str],
    include_template_content: bool = False,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
) -> dict[str, Any]:
    """Create the replay, label template, and review status files for dogfood."""
    output_dir.mkdir(parents=True, exist_ok=True)
    replay_path = output_dir / "dogfood-replay.json"
    labels_path = output_dir / "dogfood-labels.json"
    review_path = output_dir / "dogfood-review.json"
    review_markdown_path = output_dir / "dogfood-review.md"

    replay_report = await build_dogfood_replay_report(
        transcript_path=transcript_path,
        trace_paths=list(trace_paths or []),
        project_path=project_path,
        group_id=group_id,
        modes=modes,
        include_content=False,
    )
    content_lookup = (
        _dogfood_turn_content_lookup(transcript_path) if include_template_content else None
    )
    label_report = replay_report
    if include_template_content:
        label_report = await build_dogfood_replay_report(
            transcript_path=transcript_path,
            trace_paths=list(trace_paths or []),
            project_path=project_path,
            group_id=group_id,
            modes=modes,
            include_content=True,
        )
    label_template = build_dogfood_label_template(
        label_report,
        include_content=include_template_content,
        content_lookup=content_lookup,
    )

    replay_path.write_text(
        json.dumps(replay_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        json.dumps(label_template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_report = build_dogfood_review_report(
        labels_path=labels_path,
        group_id=group_id,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    review_path.write_text(
        json.dumps(review_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    review_markdown_path.write_text(
        render_dogfood_review_markdown(review_report),
        encoding="utf-8",
    )
    label_turn_count = len(label_template.get("turns") or [])
    trace_status = (replay_report.get("trace_evidence") or {}).get("status")
    prepare_status = (
        "trace_only"
        if label_turn_count == 0 and trace_status == "measured"
        else "prepared"
    )
    next_commands = {
        "review": _shell_join(
            [
                "engram",
                "dogfood",
                "review",
                "--labels",
                str(labels_path),
                "--require-ready",
            ]
        ),
        "finalize": None,
    }
    if label_turn_count > 0:
        next_commands["finalize"] = _shell_join(
            [
                "engram",
                "dogfood",
                "finalize",
                "--labels",
                str(labels_path),
                "--replay-report",
                str(replay_path),
                "--human-label-artifact",
                str(output_dir / "human-labels.json"),
                "--source",
                "native_dogfood_harness",
                "--client",
                "<client>",
                "--captured-at",
                "<ISO-8601>",
                "--labeler",
                "<human-reviewer>",
            ]
        )
    notes = [
        "Replay and review files are local dogfood artifacts, not benchmark evidence.",
    ]
    if label_turn_count == 0:
        notes.append(
            "No labelable user turns were found; this bundle can validate trace/cost evidence only."
        )
    else:
        notes.append("Fill dogfood-labels.json from human review before running finalize.")
    return {
        "operation": "dogfood_prepare",
        "status": prepare_status,
        "output_dir": str(output_dir),
        "transcript": str(transcript_path),
        "project_path": project_path,
        "group_id": group_id,
        "modes": modes,
        "paths": {
            "replay_report": str(replay_path),
            "labels": str(labels_path),
            "review_report": str(review_path),
            "review_markdown": str(review_markdown_path),
        },
        "replay": {
            "status": replay_report.get("status"),
            "user_turn_count": (replay_report.get("source") or {}).get("user_turn_count", 0),
            "content_redacted": (replay_report.get("source") or {}).get(
                "content_redacted",
                True,
            ),
            "trace_status": (replay_report.get("trace_evidence") or {}).get("status"),
            "trace_count": (replay_report.get("trace_evidence") or {}).get(
                "trace_count",
                0,
            ),
        },
        "labels": {
            "content_redacted": (label_template.get("source") or {}).get(
                "content_redacted",
                True,
            ),
            "turn_count": label_turn_count,
        },
        "review": review_report,
        "next_commands": next_commands,
        "notes": notes,
    }


def build_dogfood_candidate_report(
    *,
    root_path: Path,
    project_path: str | None = None,
    limit: int = 20,
    max_files: int = 500,
    min_turns: int = 1,
    include_trace_only: bool = False,
    project_only: bool = False,
) -> dict[str, Any]:
    """Find redaction-safe local transcript candidates for dogfood review."""
    files = _dogfood_candidate_files(root_path, max_files=max_files)
    candidates: list[dict[str, Any]] = []
    trace_only: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    project_mismatches: list[dict[str, Any]] = []
    for path in files:
        item = _dogfood_candidate_from_path(path, project_path=project_path)
        if project_only and item.get("project_match") is not True:
            project_mismatches.append(item)
            continue
        status = item.get("status")
        if status == "candidate":
            candidates.append(item)
        elif status == "trace_only":
            trace_only.append(item)
            if include_trace_only:
                candidates.append(item)
        else:
            skipped.append(item)

    candidates.sort(
        key=lambda item: (
            1 if item.get("project_match") is True else 0,
            int(item.get("labelable_turn_count") or 0),
            float(item.get("modified_at_epoch") or 0.0),
        ),
        reverse=True,
    )
    selected = [
        item
        for item in candidates
        if item.get("status") == "trace_only"
        or int(item.get("labelable_turn_count") or 0) >= min_turns
    ][:limit]
    return {
        "operation": "dogfood_scan",
        "status": "measured",
        "root": str(root_path),
        "content_redacted": True,
        "files_scanned": len(files),
        "candidate_count": sum(1 for item in candidates if item.get("status") == "candidate"),
        "trace_only_count": len(trace_only),
        "skipped_count": len(skipped),
        "project_mismatch_count": len(project_mismatches),
        "limit": limit,
        "max_files": max_files,
        "min_turns": min_turns,
        "include_trace_only": include_trace_only,
        "project_only": project_only,
        "candidates": selected,
        "notes": [
            "Candidate scan is redacted; it reports counts and hashes, not transcript text.",
            "Run prepare on a candidate, then fill labels from the original local transcript.",
        ],
    }


def _dogfood_candidate_files(root_path: Path, *, max_files: int) -> list[Path]:
    if root_path.is_file():
        return [root_path]
    files = [path for path in root_path.rglob("*.jsonl") if path.is_file()]
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return files[:max_files]


def _dogfood_candidate_from_path(
    path: Path,
    *,
    project_path: str | None,
) -> dict[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
        turns = parse_transcript_text(raw_text, source=str(path))
        error = None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raw_text = ""
        turns = []
        error = str(exc)
    metadata = _dogfood_transcript_metadata(raw_text)
    session_cwd = _optional_str(metadata.get("session_cwd"))
    session_id = _optional_str(metadata.get("session_id"))
    project_match = _dogfood_project_match(
        session_cwd=session_cwd,
        project_path=project_path,
    )
    user_turns = [turn for turn in turns if turn.role == "user"]
    assistant_turns = [turn for turn in turns if turn.role == "assistant"]
    tool_turns = [turn for turn in turns if turn.role == "tool"]
    trace_evidence = build_dogfood_trace_evidence(turns)
    status = "candidate" if user_turns else "empty"
    if not user_turns and trace_evidence.get("status") == "measured":
        status = "trace_only"
    if error:
        status = "error"
    modified_at = path.stat().st_mtime if path.exists() else 0.0
    prepare_command = None
    if status == "candidate":
        prepare_command = _shell_join(
            [
                "engram",
                "dogfood",
                "prepare",
                "--transcript",
                str(path),
                "--project",
                project_path or "$PWD",
                "--out-dir",
                f"dogfood-review/{path.stem}",
            ]
        )
    return {
        "status": status,
        "path": str(path),
        "session_id": session_id,
        "session_cwd": session_cwd,
        "project_match": project_match,
        "modified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified_at)),
        "modified_at_epoch": modified_at,
        "transcript_hash": hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
        "labelable_turn_count": len(user_turns),
        "assistant_turn_count": len(assistant_turns),
        "tool_turn_count": len(tool_turns),
        "content_hashes": [turn.content_hash for turn in user_turns[:5]],
        "trace_status": trace_evidence.get("status"),
        "trace_count": trace_evidence.get("trace_count", 0),
        "prepare_command": prepare_command,
        "error": error,
    }


def _dogfood_transcript_metadata(raw_text: str) -> dict[str, Any]:
    session_id = None
    session_cwd = None
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("{"):
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        if data.get("type") == "session_meta":
            session_id = session_id or _optional_str(payload.get("id"))
            session_cwd = session_cwd or _optional_str(payload.get("cwd"))
        elif data.get("type") == "turn_context":
            session_cwd = session_cwd or _optional_str(payload.get("cwd"))
        if session_id and session_cwd:
            break
    return {"session_id": session_id, "session_cwd": session_cwd}


def _dogfood_project_match(
    *,
    session_cwd: str | None,
    project_path: str | None,
) -> bool | None:
    if not session_cwd or not project_path or project_path == "$PWD":
        return None
    try:
        session = Path(session_cwd).expanduser().resolve()
        project = Path(project_path).expanduser().resolve()
    except (OSError, RuntimeError):
        return None
    return session == project or project in session.parents


def build_dogfood_trace_evidence(turns: list[DogfoodTurn]) -> dict[str, Any]:
    """Summarize redaction-safe AXI trace metadata in a replay transcript."""
    trace_turns = [
        turn for turn in turns if turn.role == "tool" and isinstance(turn.metadata, dict)
    ]
    records = [
        turn.metadata
        for turn in trace_turns
        if _optional_str(turn.metadata.get("operation")) is not None
    ]
    if not records:
        return {
            "status": "missing",
            "trace_count": 0,
            "operation_counts": {},
            "status_counts": {},
            "origin_counts": {},
            "client_counts": {},
            "duration_ms": {
                "count": 0,
                "avg": 0.0,
                "p95": 0.0,
                "max": 0.0,
            },
            "timeout_count": 0,
            "degraded_count": 0,
            "cache_hit_count": 0,
            "session_start_count": 0,
            "followup_count": 0,
        }

    durations = [_trace_duration_ms(record) for record in records]
    durations = [duration for duration in durations if duration is not None]
    origins = [_optional_str(record.get("origin")) for record in records]
    statuses = [_optional_str(record.get("status")) for record in records]

    return {
        "status": "measured",
        "trace_count": len(records),
        "operation_counts": _count_values(
            _optional_str(record.get("operation")) for record in records
        ),
        "status_counts": _count_values(statuses),
        "origin_counts": _count_values(origins),
        "client_counts": _count_values(
            _optional_str(record.get("client")) for record in records
        ),
        "duration_ms": _duration_summary(durations),
        "timeout_count": sum(1 for record in records if _trace_timed_out(record)),
        "degraded_count": sum(
            1 for status in statuses if status in {"degraded", "timeout", "error", "failed"}
        ),
        "cache_hit_count": sum(1 for record in records if _trace_cache_hit(record)),
        "session_start_count": sum(1 for origin in origins if origin == "session-start-hook"),
        "followup_count": sum(1 for origin in origins if origin == "agent-followup"),
    }


def _dogfood_trace_turns_from_paths(trace_paths: list[Path]) -> list[DogfoodTurn]:
    turns: list[DogfoodTurn] = []
    for trace_path in trace_paths:
        raw_text = trace_path.read_text(encoding="utf-8")
        turns.extend(parse_transcript_text(raw_text, source=str(trace_path)))
    return turns


def build_dogfood_label_template(
    report: dict[str, Any],
    *,
    include_content: bool = False,
    content_lookup: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a redaction-safe template for opted-in human review labels."""
    source = report.get("source") or {}
    lookup = content_lookup or {}
    return {
        "kind": "engram.dogfood_label_template.v1",
        "status": "template",
        "opt_in_required": True,
        "source": {
            "path": source.get("path"),
            "transcript_hash": source.get("transcript_hash"),
            "content_redacted": not include_content,
            "user_turn_count": source.get("user_turn_count", 0),
        },
        "project_path": report.get("project_path"),
        "group_id": report.get("group_id"),
        "modes": list(report.get("modes") or []),
        "instructions": _dogfood_label_template_instructions(include_content),
        "turns": [
            {
                "index": turn.get("index"),
                "content_hash": turn.get("content_hash"),
                **_dogfood_template_content(
                    turn,
                    include_content=include_content,
                    content_lookup=lookup,
                ),
                "need_type": (turn.get("need") or {}).get("need_type"),
                "should_recall": (turn.get("need") or {}).get("should_recall"),
                "query_hint": (
                    (turn.get("need") or {}).get("query_hint") if include_content else None
                ),
                "query_hint_redacted": _dogfood_template_query_hint_redacted(
                    turn,
                    include_content=include_content,
                ),
                "decisions": [
                    {
                        "mode": decision.get("mode"),
                        "decision": decision.get("decision"),
                        "mode_executed": decision.get("mode_executed"),
                        "reason": decision.get("reason"),
                    }
                    for decision in turn.get("decisions") or []
                ],
                "labels": {
                    "memory_was_needed": None,
                    "best_mode": None,
                    "helpful_modes": [],
                    "false_recall_modes": [],
                    "stale_modes": [],
                    "corrected_modes": [],
                    "notes": "",
                },
            }
            for turn in report.get("turns") or []
        ],
    }


def _dogfood_template_query_hint_redacted(
    turn: dict[str, Any],
    *,
    include_content: bool,
) -> bool:
    need = turn.get("need") if isinstance(turn.get("need"), dict) else {}
    query_hint = _optional_str(need.get("query_hint"))
    source_redacted = bool(need.get("query_hint_redacted"))
    if include_content:
        return bool(source_redacted and not query_hint)
    return bool(query_hint or source_redacted)


def _dogfood_turn_content_lookup(transcript_path: Path) -> dict[str, str]:
    raw_text = transcript_path.read_text(encoding="utf-8")
    return {
        turn.content_hash: turn.content
        for turn in parse_transcript_text(raw_text, source=str(transcript_path))
        if turn.role == "user"
    }


def _dogfood_label_template_instructions(include_content: bool) -> list[str]:
    if include_content:
        return [
            "Content is included because the operator explicitly opted in.",
            "Keep this template local unless the transcript content is safe to share.",
            "Fill labels only from human review of the original local transcript.",
            "Import labels into evaluation evidence only through an explicit command.",
        ]
    return [
        "Do not paste private transcript content unless explicitly opting in.",
        "Fill labels only from human review of the original local transcript.",
        "Import labels into evaluation evidence only through an explicit command.",
    ]


def _dogfood_template_content(
    turn: dict[str, Any],
    *,
    include_content: bool,
    content_lookup: dict[str, str],
) -> dict[str, str]:
    if not include_content:
        return {}
    content_hash = _optional_str(turn.get("content_hash"))
    content = _optional_str(turn.get("content"))
    if not content and content_hash:
        content = _optional_str(content_lookup.get(content_hash))
    return {"content": content} if content else {}


def build_dogfood_label_import_samples(
    artifact: dict[str, Any],
    *,
    group_id: str | None = None,
    include_all_modes: bool = False,
) -> tuple[list[StoredRecallEvalSample], list[StoredSessionContinuitySample], dict[str, Any]]:
    """Convert a reviewed dogfood label artifact into evaluation-store samples."""
    template = _dogfood_label_template(artifact)
    source = template.get("source") or {}
    content_redacted = bool(source.get("content_redacted", True))
    transcript_hash = str(source.get("transcript_hash") or "unknown")
    resolved_group_id = group_id or str(template.get("group_id") or "default")
    recall_samples: list[StoredRecallEvalSample] = []
    skipped_turns = 0

    for turn in template.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        labels = turn.get("labels") if isinstance(turn.get("labels"), dict) else {}
        memory_needed = labels.get("memory_was_needed")
        if not isinstance(memory_needed, bool):
            skipped_turns += 1
            continue
        if _dogfood_has_placeholder_text(labels.get("notes")):
            skipped_turns += 1
            continue
        decisions = [d for d in turn.get("decisions") or [] if isinstance(d, dict)]
        selected_modes = _selected_label_modes(labels)
        if include_all_modes:
            selected_modes = {str(decision.get("mode")) for decision in decisions}
        if not selected_modes:
            skipped_turns += 1
            continue
        notes = _dogfood_label_notes(
            transcript_hash=transcript_hash,
            turn=turn,
            label_notes=labels.get("notes"),
        )
        helpful_modes = _string_set(labels.get("helpful_modes"))
        false_modes = _string_set(labels.get("false_recall_modes"))
        stale_modes = _string_set(labels.get("stale_modes"))
        corrected_modes = _string_set(labels.get("corrected_modes"))
        best_mode = _optional_str(labels.get("best_mode"))
        for decision in decisions:
            mode = _optional_str(decision.get("mode"))
            if not mode or mode not in selected_modes:
                continue
            triggered = bool(decision.get("decision") == "triggered" and mode != "off")
            helped = bool(mode == best_mode or mode in helpful_modes)
            false_recall = bool(
                mode in false_modes
                or (include_all_modes and triggered and not memory_needed and not helped)
            )
            recall_samples.append(
                StoredRecallEvalSample(
                    group_id=resolved_group_id,
                    recall_triggered=triggered,
                    recall_helped=helped,
                    recall_needed=memory_needed,
                    packets_surfaced=1 if triggered else 0,
                    packets_used=1 if triggered and helped else 0,
                    false_recalls=1 if false_recall else 0,
                    stale_packets=1 if mode in stale_modes else 0,
                    corrected_packets=1 if mode in corrected_modes else 0,
                    source=f"dogfood_review:{mode}",
                    query=None if content_redacted else _optional_str(turn.get("query_hint")),
                    notes=f"{notes}; mode={mode}",
                    id=_stable_sample_id(
                        "ers",
                        resolved_group_id,
                        transcript_hash,
                        turn.get("index"),
                        turn.get("content_hash"),
                        mode,
                    ),
                )
            )

    session_samples = _dogfood_session_samples(
        template,
        group_id=resolved_group_id,
        transcript_hash=transcript_hash,
    )
    summary = {
        "group_id": resolved_group_id,
        "transcript_hash": transcript_hash,
        "recall_sample_count": len(recall_samples),
        "session_sample_count": len(session_samples),
        "skipped_turn_count": skipped_turns,
        "include_all_modes": include_all_modes,
    }
    return recall_samples, session_samples, summary


async def import_dogfood_label_artifact(
    *,
    labels_path: Path,
    sqlite_path: Path | None = None,
    group_id: str | None = None,
    include_all_modes: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Import opted-in dogfood review labels into the local evaluation store."""
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    recall_samples, session_samples, summary = build_dogfood_label_import_samples(
        artifact,
        group_id=group_id,
        include_all_modes=include_all_modes,
    )
    review = _dogfood_label_review_details(
        _dogfood_label_template(artifact),
        include_all_modes=include_all_modes,
    )
    invalid_failures = _dogfood_invalid_label_failures(review)
    if invalid_failures:
        return {
            "operation": "dogfood_import_labels",
            "status": "invalid_labels",
            "ready": False,
            "labels_path": str(labels_path),
            "review": review,
            "failures": invalid_failures,
            "notes": [
                "No labels were imported.",
                "Replace placeholder review text before importing dogfood evidence.",
            ],
            **summary,
        }
    if dry_run:
        return {
            "operation": "dogfood_import_labels",
            "status": "dry_run",
            "labels_path": str(labels_path),
            **summary,
        }

    from engram.config import EngramConfig

    db_path = sqlite_path or EngramConfig().get_sqlite_path()
    store = SQLiteEvaluationStore(str(db_path))
    await store.initialize()
    try:
        for sample in recall_samples:
            await store.save_recall_sample(sample)
        for sample in session_samples:
            await store.save_session_sample(sample)
    finally:
        await store.close()
    return {
        "operation": "dogfood_import_labels",
        "status": "imported",
        "labels_path": str(labels_path),
        "sqlite_path": str(db_path),
        **summary,
    }


async def import_dogfood_replay_cost_metrics(
    *,
    replay_report: Path | None,
    sqlite_path: Path | None = None,
    group_id: str = "default",
) -> dict[str, Any]:
    """Persist measured replay trace evidence as memory operation cost metrics."""
    if replay_report is None:
        return {
            "operation": "dogfood_import_replay_cost",
            "status": "missing",
            "replay_report": None,
            "note": "No replay report was supplied; existing cost metrics must already be present.",
        }
    report = json.loads(replay_report.read_text(encoding="utf-8"))
    metrics = dogfood_memory_operation_metrics_from_replay_report(report)
    if not metrics:
        return {
            "operation": "dogfood_import_replay_cost",
            "status": "needs_trace_evidence",
            "replay_report": str(replay_report),
            "group_id": group_id,
            "trace_status": (report.get("trace_evidence") or {}).get("status"),
        }

    from engram.config import EngramConfig

    db_path = sqlite_path or EngramConfig().get_sqlite_path()
    store = SQLiteEvaluationStore(str(db_path))
    await store.initialize()
    try:
        await store.save_memory_operation_metrics_snapshot(
            StoredMemoryOperationMetricsSnapshot(
                group_id=group_id,
                metrics=metrics,
                source="dogfood_replay_trace",
                id=_stable_sample_id(
                    "emo",
                    group_id,
                    (report.get("source") or {}).get("transcript_hash"),
                    str(replay_report),
                ),
            )
        )
    finally:
        await store.close()
    return {
        "operation": "dogfood_import_replay_cost",
        "status": "imported",
        "replay_report": str(replay_report),
        "sqlite_path": str(db_path),
        "group_id": group_id,
        "operation_count": metrics.get("operation_count", 0),
        "duration_ms": metrics.get("duration_ms", {}),
        "cache_hit_count": metrics.get("cache_hit_count", 0),
        "timeout_count": metrics.get("timeout_count", 0),
        "degraded_count": metrics.get("degraded_count", 0),
    }


def dogfood_memory_operation_metrics_from_replay_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    """Convert measured dogfood trace evidence into memory operation metrics."""
    trace = report.get("trace_evidence") if isinstance(report, dict) else None
    if not isinstance(trace, dict) or trace.get("status") != "measured":
        return {}
    operation_count = _int(trace.get("trace_count"))
    duration = trace.get("duration_ms") if isinstance(trace.get("duration_ms"), dict) else {}
    if operation_count <= 0:
        return {}
    cache_hit_count = _int(trace.get("cache_hit_count"))
    cache_observed = min(operation_count, cache_hit_count)
    cache_miss_count = max(0, operation_count - cache_observed)
    timeout_count = _int(trace.get("timeout_count"))
    degraded_count = _int(trace.get("degraded_count"))
    status_counts = (
        trace.get("status_counts")
        if isinstance(trace.get("status_counts"), dict)
        else {}
    )
    operation_counts = (
        trace.get("operation_counts") if isinstance(trace.get("operation_counts"), dict) else {}
    )
    client_counts = (
        trace.get("client_counts")
        if isinstance(trace.get("client_counts"), dict)
        else {}
    )
    origin_counts = (
        trace.get("origin_counts")
        if isinstance(trace.get("origin_counts"), dict)
        else {}
    )
    source_counts = dict(
        client_counts or origin_counts or {"dogfood_replay_trace": operation_count}
    )
    return {
        "operation_count": operation_count,
        "duration_ms": {
            "avg": _float(duration.get("avg")),
            "p95": _float(duration.get("p95")),
        },
        "budget_ms": {"avg": 0.0, "p95": 0.0},
        "avg_budget_tokens": 0,
        "completed_count": _completed_trace_count(status_counts, operation_count),
        "skipped_count": _int(status_counts.get("skipped")),
        "timeout_count": timeout_count,
        "degraded_count": degraded_count,
        "error_count": _int(status_counts.get("error")) + _int(status_counts.get("failed")),
        "budget_miss_count": timeout_count,
        "cache_hit_count": cache_hit_count,
        "cache_miss_count": cache_miss_count,
        "status_counts": {str(key): _int(value) for key, value in status_counts.items()},
        "skip_reason_counts": {},
        "operation_counts": {
            str(key): _int(value) for key, value in operation_counts.items()
        },
        "source_counts": {str(key): _int(value) for key, value in source_counts.items()},
        "result_count": 0,
        "packet_count": 0,
    }


def _completed_trace_count(status_counts: dict[str, Any], operation_count: int) -> int:
    explicit = (
        _int(status_counts.get("ok"))
        + _int(status_counts.get("healthy"))
        + _int(status_counts.get("completed"))
    )
    if explicit:
        return explicit
    failures = (
        _int(status_counts.get("error"))
        + _int(status_counts.get("failed"))
        + _int(status_counts.get("timeout"))
    )
    return max(0, operation_count - failures)


def build_dogfood_review_report(
    *,
    labels_path: Path,
    group_id: str | None = None,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
    include_all_modes: bool = False,
    need_type: str | None = None,
    command_limit: int = 10,
    include_content: bool = False,
    context: int = 1,
) -> dict[str, Any]:
    """Summarize label review progress before import/export closeout."""
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    template = _dogfood_label_template(artifact)
    _recall_samples, _session_samples, summary = build_dogfood_label_import_samples(
        artifact,
        group_id=group_id,
        include_all_modes=include_all_modes,
    )
    review = _dogfood_label_review_details(
        template,
        include_all_modes=include_all_modes,
    )
    turn_count = int(review.get("turn_count") or 0)
    recall_count = int(summary.get("recall_sample_count") or 0)
    session_count = int(summary.get("session_sample_count") or 0)
    failures = _dogfood_reviewed_label_failures(
        recall_count=recall_count,
        session_count=session_count,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    if turn_count == 0 and failures:
        failures.insert(0, "no_labelable_turns")
    if review["invalid_label_count"]:
        failures.append(f"invalid_label_turns({review['invalid_label_count']})")
    failures.extend(_dogfood_invalid_session_failures(review))
    ready = not failures
    status = "ready_for_import" if ready else "needs_labels"
    if turn_count == 0 and not ready:
        status = "trace_only"
    if review["invalid_label_count"] or review.get("invalid_session_sample_count"):
        status = "invalid_labels"
    label_commands = _dogfood_review_label_commands(
        labels_path=labels_path,
        review=review,
        session_count=session_count,
        min_session_samples=min_session_samples,
        need_type=need_type,
        command_limit=command_limit,
    )
    inspection_previews = _dogfood_review_inspection_previews(
        labels_path=labels_path,
        turn_commands=label_commands.get("turns") or [],
        include_content=include_content,
        context=context,
    )
    importable_evidence = recall_count > 0 or session_count > 0
    next_commands = {
        "import_labels": None,
        "export_evidence": None,
    }
    if ready and importable_evidence:
        next_commands = {
            "import_labels": _dogfood_import_command(
                labels_path=labels_path,
                sqlite_path=None,
                group_id=group_id,
                include_all_modes=include_all_modes,
            ),
            "export_evidence": _dogfood_export_command(
                labels_path=labels_path,
                output_path=Path("human-labels.json"),
                group_id=group_id,
                include_all_modes=include_all_modes,
            ),
        }
    notes = [
        "This command only inspects reviewed labels; it does not import or export evidence.",
    ]
    if turn_count == 0:
        notes.append(
            "No labelable user turns were found; this artifact can support "
            "trace/cost evidence only."
        )
    else:
        notes.append(
            "Fill labels from the original local transcript before import/export closeout."
        )
    return {
        "operation": "dogfood_review",
        "status": status,
        "ready": ready,
        "labels_path": str(labels_path),
        "group_id": summary.get("group_id"),
        "transcript_hash": summary.get("transcript_hash"),
        "review": review,
        "reviewed_labels": {
            "recall_sample_count": recall_count,
            "session_sample_count": session_count,
            "skipped_turn_count": summary.get("skipped_turn_count", 0),
            "min_recall_samples": min_recall_samples,
            "min_session_samples": min_session_samples,
            "include_all_modes": include_all_modes,
        },
        "failures": failures,
        "next_commands": next_commands,
        "label_commands": label_commands,
        "inspection_previews": inspection_previews,
        "inspection": {
            "include_content": include_content,
            "context": max(0, int(context)),
            "preview_count": len(inspection_previews),
            "content_redacted": not include_content,
        },
        "notes": notes,
    }


def _dogfood_review_inspection_previews(
    *,
    labels_path: Path,
    turn_commands: list[dict[str, Any]],
    include_content: bool,
    context: int,
) -> list[dict[str, Any]]:
    """Build opt-in local transcript previews for a bounded review batch."""
    if not include_content:
        return []
    previews: list[dict[str, Any]] = []
    for command in turn_commands:
        turn_index = _int(command.get("turn"))
        if turn_index is None:
            continue
        previews.append(
            build_dogfood_turn_inspection_report(
                labels_path=labels_path,
                turn_index=turn_index,
                context=max(0, int(context)),
                include_content=True,
            )
        )
    return previews


def build_dogfood_turn_inspection_report(
    *,
    labels_path: Path,
    turn_index: int | None = None,
    content_hash: str | None = None,
    next_unreviewed: bool = False,
    need_type: str | None = None,
    context: int = 1,
    include_content: bool = False,
) -> dict[str, Any]:
    """Inspect the local source turn behind a redacted dogfood label template."""
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    template = _dogfood_label_template(artifact)
    source = template.get("source") if isinstance(template.get("source"), dict) else {}
    source_path = _optional_str(source.get("path"))
    label_turns = [turn for turn in template.get("turns") or [] if isinstance(turn, dict)]
    if next_unreviewed:
        label_turn = _find_next_dogfood_review_label_turn(
            template,
            need_type=need_type,
        )
    else:
        label_turn = _find_dogfood_label_turn(
            label_turns,
            turn_index=turn_index,
            content_hash=content_hash,
        )
    if label_turn is None:
        failure = "missing_next_review_turn" if next_unreviewed else "missing_label_turn"
        return {
            "operation": "dogfood_inspect_turn",
            "status": "missing_turn",
            "ready": False,
            "labels_path": str(labels_path),
            "source_path": source_path,
            "turn": turn_index,
            "content_hash": content_hash,
            "need_type_filter": _optional_str(need_type),
            "include_content": include_content,
            "failures": [failure],
        }
    if source_path is None:
        return {
            "operation": "dogfood_inspect_turn",
            "status": "missing_source",
            "ready": False,
            "labels_path": str(labels_path),
            "source_path": None,
            "target": _dogfood_inspection_target(label_turn),
            "include_content": include_content,
            "failures": ["missing_transcript_source_path"],
        }
    transcript_path = Path(source_path)
    if not transcript_path.exists():
        return {
            "operation": "dogfood_inspect_turn",
            "status": "missing_source",
            "ready": False,
            "labels_path": str(labels_path),
            "source_path": source_path,
            "target": _dogfood_inspection_target(label_turn),
            "include_content": include_content,
            "failures": [f"missing_transcript_source:{source_path}"],
        }

    turns = parse_transcript_text(
        transcript_path.read_text(encoding="utf-8"),
        source=source_path,
    )
    user_turns = [turn for turn in turns if turn.role == "user"]
    target_hash = _optional_str(label_turn.get("content_hash"))
    user_position = _dogfood_user_turn_position(
        user_turns,
        turn_index=_int(label_turn.get("index")),
        content_hash=target_hash,
    )
    if user_position is None:
        return {
            "operation": "dogfood_inspect_turn",
            "status": "missing_source_turn",
            "ready": False,
            "labels_path": str(labels_path),
            "source_path": source_path,
            "target": _dogfood_inspection_target(label_turn),
            "include_content": include_content,
            "failures": ["missing_source_turn"],
        }
    target_turn = user_turns[user_position]
    transcript_position = turns.index(target_turn)
    start = max(0, transcript_position - max(0, context))
    end = min(len(turns), transcript_position + max(0, context) + 1)
    review_item = _dogfood_review_queue_item_for_label_turn(template, label_turn)
    label_command = (
        _dogfood_turn_label_command(labels_path=labels_path, item=review_item)
        if review_item is not None
        else None
    )
    return {
        "operation": "dogfood_inspect_turn",
        "status": "ready",
        "ready": True,
        "labels_path": str(labels_path),
        "source_path": source_path,
        "transcript_hash": source.get("transcript_hash"),
        "need_type_filter": _optional_str(need_type),
        "include_content": include_content,
        "content_redacted": not include_content,
        "context": max(0, context),
        "target": {
            **_dogfood_inspection_target(label_turn),
            "user_turn_position": user_position,
            "transcript_turn_position": transcript_position,
            "source_content_hash": target_turn.content_hash,
        },
        "context_turns": [
            _dogfood_inspection_context_turn(
                turn,
                selected=(index == transcript_position),
                include_content=include_content,
            )
            for index, turn in enumerate(turns[start:end], start=start)
        ],
        "label_command": label_command,
        "notes": [
            (
                "Content is omitted by default. Rerun with --include-content for "
                "local human review."
            )
            if not include_content
            else "Content is included because --include-content was provided.",
            "This command is read-only and does not import or export evidence.",
        ],
    }


def _find_dogfood_label_turn(
    label_turns: list[dict[str, Any]],
    *,
    turn_index: int | None,
    content_hash: str | None,
) -> dict[str, Any] | None:
    normalized_hash = _optional_str(content_hash)
    for turn in label_turns:
        if turn_index is not None and _int(turn.get("index")) == turn_index:
            return turn
        if normalized_hash and _optional_str(turn.get("content_hash")) == normalized_hash:
            return turn
    return None


def _find_next_dogfood_review_label_turn(
    template: dict[str, Any],
    *,
    need_type: str | None,
) -> dict[str, Any] | None:
    review = _dogfood_label_review_details(template, include_all_modes=False)
    for item in review.get("review_queue") or []:
        if not isinstance(item, dict):
            continue
        if _optional_str(need_type) and _optional_str(item.get("need_type")) != _optional_str(
            need_type
        ):
            continue
        return _find_dogfood_label_turn(
            [turn for turn in template.get("turns") or [] if isinstance(turn, dict)],
            turn_index=_int(item.get("index")),
            content_hash=_optional_str(item.get("content_hash")),
        )
    return None


def _dogfood_review_queue_item_for_label_turn(
    template: dict[str, Any],
    label_turn: dict[str, Any],
) -> dict[str, Any] | None:
    review = _dogfood_label_review_details(template, include_all_modes=False)
    target_index = _int(label_turn.get("index"))
    target_hash = _optional_str(label_turn.get("content_hash"))
    for item in review.get("review_queue") or []:
        if not isinstance(item, dict):
            continue
        if _int(item.get("index")) == target_index:
            return item
        if target_hash and _optional_str(item.get("content_hash")) == target_hash:
            return item
    return None


def _dogfood_user_turn_position(
    user_turns: list[DogfoodTurn],
    *,
    turn_index: int,
    content_hash: str | None,
) -> int | None:
    if 0 <= turn_index < len(user_turns):
        candidate = user_turns[turn_index]
        if not content_hash or candidate.content_hash == content_hash:
            return turn_index
    if content_hash:
        for index, turn in enumerate(user_turns):
            if turn.content_hash == content_hash:
                return index
    return None


def _dogfood_inspection_target(turn: dict[str, Any]) -> dict[str, Any]:
    labels = turn.get("labels") if isinstance(turn.get("labels"), dict) else {}
    return {
        "turn": turn.get("index"),
        "content_hash": turn.get("content_hash"),
        "need_type": turn.get("need_type"),
        "should_recall": turn.get("should_recall"),
        "query_hint": turn.get("query_hint"),
        "query_hint_redacted": bool(turn.get("query_hint_redacted")),
        "available_modes": [
            decision.get("mode")
            for decision in turn.get("decisions") or []
            if isinstance(decision, dict) and decision.get("mode")
        ],
        "labels": {
            "memory_was_needed": labels.get("memory_was_needed"),
            "best_mode": labels.get("best_mode"),
            "helpful_modes": list(labels.get("helpful_modes") or []),
            "false_recall_modes": list(labels.get("false_recall_modes") or []),
            "notes": labels.get("notes"),
        },
    }


def _dogfood_inspection_context_turn(
    turn: DogfoodTurn,
    *,
    selected: bool,
    include_content: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": turn.role,
        "content_hash": turn.content_hash,
        "timestamp": turn.timestamp,
        "selected": selected,
    }
    if include_content:
        payload["content"] = turn.content
    return payload


def update_dogfood_turn_label(
    *,
    labels_path: Path,
    turn_index: int,
    memory_needed: bool,
    best_mode: str | None = None,
    helpful_modes: list[str] | None = None,
    false_recall_modes: list[str] | None = None,
    stale_modes: list[str] | None = None,
    corrected_modes: list[str] | None = None,
    notes: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Mutate one turn label in a dogfood label template."""
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    template = _dogfood_label_template(artifact)
    turns = [turn for turn in template.get("turns") or [] if isinstance(turn, dict)]
    selected_turn = next((turn for turn in turns if turn.get("index") == turn_index), None)
    if selected_turn is None:
        raise ValueError(f"dogfood label turn not found: {turn_index}")

    decisions = selected_turn.get("decisions") or []
    decision_modes = {
        mode
        for mode in (_optional_str(decision.get("mode")) for decision in decisions)
        if mode
    }
    existing_labels = selected_turn.get("labels") or {}
    label_payload = {
        "memory_was_needed": memory_needed,
        "best_mode": _normalize_label_mode(best_mode, decision_modes, allow_none=True),
        "helpful_modes": _normalize_label_modes(helpful_modes or [], decision_modes),
        "false_recall_modes": _normalize_label_modes(false_recall_modes or [], decision_modes),
        "stale_modes": _normalize_label_modes(stale_modes or [], decision_modes),
        "corrected_modes": _normalize_label_modes(corrected_modes or [], decision_modes),
        "notes": notes if notes is not None else existing_labels.get("notes", ""),
    }
    selected_turn["labels"] = label_payload
    output = output_path or labels_path
    _write_dogfood_label_artifact(artifact, template, output)
    review = build_dogfood_review_report(labels_path=output)
    return {
        "operation": "dogfood_label_turn",
        "status": "updated",
        "labels_path": str(output),
        "turn": turn_index,
        "content_hash": selected_turn.get("content_hash"),
        "labels": label_payload,
        "review": review,
    }


def add_dogfood_session_label(
    *,
    labels_path: Path,
    scenario: str,
    baseline_score: float,
    memory_score: float,
    open_loop_expected: bool = False,
    open_loop_recovered: bool = False,
    temporal_expected: bool = False,
    temporal_correct: bool = False,
    notes: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Append one session-continuity review label to a dogfood template."""
    artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    template = _dogfood_label_template(artifact)
    samples = template.setdefault("session_samples", [])
    if not isinstance(samples, list):
        raise ValueError("dogfood label artifact session_samples must be a list")
    sample = {
        "scenario": scenario,
        "baselineScore": _clamp_score(baseline_score),
        "memoryScore": _clamp_score(memory_score),
        "openLoopExpected": bool(open_loop_expected),
        "openLoopRecovered": bool(open_loop_recovered),
        "temporalExpected": bool(temporal_expected),
        "temporalCorrect": bool(temporal_correct),
        "notes": notes or "",
    }
    samples.append(sample)
    output = output_path or labels_path
    _write_dogfood_label_artifact(artifact, template, output)
    review = build_dogfood_review_report(labels_path=output)
    return {
        "operation": "dogfood_label_session",
        "status": "updated",
        "labels_path": str(output),
        "session_sample_count": len(samples),
        "sample": sample,
        "review": review,
    }


def build_dogfood_human_label_evidence_artifact(
    artifact: dict[str, Any],
    *,
    source: str,
    client: str,
    captured_at: str,
    labeler: str,
    session_id: str | None = None,
    group_id: str | None = None,
    include_all_modes: bool = False,
    labels_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert reviewed dogfood labels into standard human-label evidence."""
    recall_samples, session_samples, summary = build_dogfood_label_import_samples(
        artifact,
        group_id=group_id,
        include_all_modes=include_all_modes,
    )
    transcript_hash = str(summary.get("transcript_hash") or "unknown")
    artifact_payload = {
        "kind": HUMAN_LABEL_EVIDENCE_KIND,
        "humanLabeled": True,
        "source": source,
        "client": client,
        "capturedAt": captured_at,
        "sessionId": session_id or f"dogfood:{transcript_hash[:12]}",
        "labeler": labeler,
        "recallSamples": [
            _recall_sample_to_human_label(sample, source=source)
            for sample in recall_samples
        ],
        "sessionSamples": [
            _session_sample_to_human_label(sample, source=source)
            for sample in session_samples
        ],
        "dogfood": {
            "transcriptHash": transcript_hash,
            "groupId": summary.get("group_id"),
            "includeAllModes": include_all_modes,
            "labelsPath": str(labels_path) if labels_path is not None else None,
            "skippedTurnCount": summary.get("skipped_turn_count", 0),
        },
    }
    return artifact_payload, summary


def export_dogfood_human_label_evidence(
    *,
    labels_path: Path,
    output_path: Path,
    source: str,
    client: str,
    captured_at: str,
    labeler: str,
    session_id: str | None = None,
    group_id: str | None = None,
    include_all_modes: bool = False,
) -> dict[str, Any]:
    """Write standard human-label evidence from reviewed dogfood labels."""
    label_artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    review = _dogfood_label_review_details(
        _dogfood_label_template(label_artifact),
        include_all_modes=include_all_modes,
    )
    invalid_failures = _dogfood_invalid_label_failures(review)
    if invalid_failures:
        _recall_samples, _session_samples, summary = build_dogfood_label_import_samples(
            label_artifact,
            group_id=group_id,
            include_all_modes=include_all_modes,
        )
        return {
            "operation": "dogfood_export_evidence",
            "status": "failed",
            "ready": False,
            "labels_path": str(labels_path),
            "human_label_artifact": str(output_path),
            "source": source,
            "client": client,
            "captured_at": captured_at,
            "labeler": labeler,
            "recall_sample_count": 0,
            "session_sample_count": 0,
            "review": review,
            "failures": invalid_failures,
            "notes": [
                "No evidence artifact was written.",
                "Replace placeholder review text before exporting dogfood evidence.",
            ],
            **summary,
        }
    evidence_artifact, summary = build_dogfood_human_label_evidence_artifact(
        label_artifact,
        source=source,
        client=client,
        captured_at=captured_at,
        labeler=labeler,
        session_id=session_id,
        group_id=group_id,
        include_all_modes=include_all_modes,
        labels_path=labels_path,
    )
    evidence = build_human_label_evidence(evidence_artifact)
    failures = _dogfood_human_label_failures(evidence)
    if failures:
        return {
            "operation": "dogfood_export_evidence",
            "status": "failed",
            "ready": False,
            "labels_path": str(labels_path),
            "human_label_artifact": str(output_path),
            "source": source,
            "client": client,
            "captured_at": captured_at,
            "labeler": labeler,
            "recall_sample_count": evidence.get("recall_sample_count", 0),
            "session_sample_count": evidence.get("session_sample_count", 0),
            "human_label_evidence": evidence,
            "failures": failures,
            "notes": [
                "No evidence artifact was written.",
                "Fix source/client/capturedAt/labeler metadata and rerun export.",
            ],
            **summary,
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "operation": "dogfood_export_evidence",
        "status": "exported",
        "ready": True,
        "labels_path": str(labels_path),
        "human_label_artifact": str(output_path),
        "source": source,
        "client": client,
        "captured_at": captured_at,
        "labeler": labeler,
        "recall_sample_count": len(evidence_artifact["recallSamples"]),
        "session_sample_count": len(evidence_artifact["sessionSamples"]),
        "human_label_evidence": evidence,
        "failures": [],
        "validation_command": _dogfood_evidence_validation_command(output_path),
        **summary,
    }


def preflight_dogfood_human_label_evidence(
    *,
    labels_path: Path,
    source: str,
    client: str,
    captured_at: str,
    labeler: str,
    session_id: str | None = None,
    group_id: str | None = None,
    include_all_modes: bool = False,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
) -> dict[str, Any]:
    """Validate dogfood export metadata before mutating local evaluation state."""
    label_artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    review = _dogfood_label_review_details(
        _dogfood_label_template(label_artifact),
        include_all_modes=include_all_modes,
    )
    invalid_failures = _dogfood_invalid_label_failures(review)
    if invalid_failures:
        _recall_samples, _session_samples, summary = build_dogfood_label_import_samples(
            label_artifact,
            group_id=group_id,
            include_all_modes=include_all_modes,
        )
        return {
            "operation": "dogfood_human_label_preflight",
            "status": "failed",
            "ready": False,
            "labels_path": str(labels_path),
            "source": source,
            "client": client,
            "captured_at": captured_at,
            "labeler": labeler,
            "recall_sample_count": 0,
            "session_sample_count": 0,
            "review": review,
            "failures": invalid_failures,
            **summary,
        }
    artifact, summary = build_dogfood_human_label_evidence_artifact(
        label_artifact,
        source=source,
        client=client,
        captured_at=captured_at,
        labeler=labeler,
        session_id=session_id,
        group_id=group_id,
        include_all_modes=include_all_modes,
        labels_path=labels_path,
    )
    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    failures = _dogfood_human_label_failures(evidence)
    return {
        "operation": "dogfood_human_label_preflight",
        "status": "measured" if not failures else "failed",
        "ready": not failures,
        "labels_path": str(labels_path),
        "source": source,
        "client": client,
        "captured_at": captured_at,
        "labeler": labeler,
        "recall_sample_count": evidence.get("recall_sample_count", 0),
        "session_sample_count": evidence.get("session_sample_count", 0),
        "human_label_evidence": evidence,
        "failures": failures,
        **summary,
    }


async def finalize_dogfood_labels(
    *,
    labels_path: Path,
    replay_report: Path | None = None,
    human_label_artifact: Path,
    sqlite_path: Path | None = None,
    source: str,
    client: str,
    captured_at: str,
    labeler: str,
    session_id: str | None = None,
    group_id: str | None = None,
    mode: str = "helix",
    helix_data_dir: Path | None = None,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
    include_all_modes: bool = False,
    skip_evaluate: bool = False,
    evaluate_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Run the reviewed dogfood label closeout path as one hard gate."""
    review = build_dogfood_review_report(
        labels_path=labels_path,
        group_id=group_id,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
        include_all_modes=include_all_modes,
    )
    if review.get("ready") is not True:
        return {
            "operation": "dogfood_finalize",
            "status": "needs_labels",
            "ready": False,
            "phase": "review",
            "labels_path": str(labels_path),
            "human_label_artifact": str(human_label_artifact),
            "review": review,
            "failures": list(review.get("failures") or []),
            "notes": [
                "No labels were imported and no evidence artifact was exported.",
                "Fix the review queue, then rerun finalize.",
            ],
        }

    evidence_preflight = preflight_dogfood_human_label_evidence(
        labels_path=labels_path,
        source=source,
        client=client,
        captured_at=captured_at,
        labeler=labeler,
        session_id=session_id,
        group_id=group_id,
        include_all_modes=include_all_modes,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    if evidence_preflight.get("ready") is not True:
        return {
            "operation": "dogfood_finalize",
            "status": "needs_evidence",
            "ready": False,
            "phase": "evidence_preflight",
            "labels_path": str(labels_path),
            "human_label_artifact": str(human_label_artifact),
            "review": review,
            "evidence_preflight": evidence_preflight,
            "failures": list(evidence_preflight.get("failures") or []),
            "notes": [
                "No labels were imported and no evidence artifact was exported.",
                "Fix dogfood source/client/capturedAt/labeler metadata, then rerun finalize.",
            ],
        }

    import_report = await import_dogfood_label_artifact(
        labels_path=labels_path,
        sqlite_path=sqlite_path,
        group_id=group_id,
        include_all_modes=include_all_modes,
        dry_run=False,
    )
    cost_report = await import_dogfood_replay_cost_metrics(
        replay_report=replay_report,
        sqlite_path=sqlite_path,
        group_id=str(import_report.get("group_id") or group_id or "default"),
    )
    export_report = export_dogfood_human_label_evidence(
        labels_path=labels_path,
        output_path=human_label_artifact,
        source=source,
        client=client,
        captured_at=captured_at,
        labeler=labeler,
        session_id=session_id,
        group_id=group_id,
        include_all_modes=include_all_modes,
    )
    closeout = build_dogfood_closeout_report(
        labels_path=labels_path,
        human_label_artifact=human_label_artifact,
        sqlite_path=sqlite_path,
        group_id=group_id,
        mode=mode,
        helix_data_dir=helix_data_dir,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
        include_all_modes=include_all_modes,
    )
    if closeout.get("ready") is not True:
        return {
            "operation": "dogfood_finalize",
            "status": "needs_evidence",
            "ready": False,
            "phase": "closeout",
            "labels_path": str(labels_path),
            "human_label_artifact": str(human_label_artifact),
            "review": review,
            "import": import_report,
            "cost": cost_report,
            "export": export_report,
            "closeout": closeout,
            "failures": list(closeout.get("failures") or []),
        }

    if skip_evaluate:
        evaluation = {
            "status": "skipped",
            "reason": "skip_evaluate",
            "command": closeout.get("commands", {}).get("native_memory_value"),
            "ready": False,
            "notes": [
                (
                    "Run the native memory-value command before treating dogfood "
                    "finalization as complete."
                )
            ],
        }
    else:
        evaluation = await _run_dogfood_memory_value_evaluation(
            sqlite_path=sqlite_path,
            human_label_artifact=human_label_artifact,
            group_id=group_id,
            mode=mode,
            helix_data_dir=helix_data_dir,
            min_recall_samples=min_recall_samples,
            min_session_samples=min_session_samples,
            timeout_seconds=evaluate_timeout_seconds,
        )
    evaluation_status = str(evaluation.get("status") or "missing")
    evaluation_ready = evaluation_status == "measured"
    if evaluation_ready:
        status = "finalized"
        failures: list[str] = []
    elif evaluation_status == "skipped":
        status = "needs_evaluation"
        failures = ["native_memory_value_evaluation_skipped"]
    else:
        status = "evaluation_failed"
        failures = [f"evaluation:{evaluation_status}"]
    return {
        "operation": "dogfood_finalize",
        "status": status,
        "ready": evaluation_ready,
        "phase": "complete" if evaluation_ready else "evaluation",
        "labels_path": str(labels_path),
        "human_label_artifact": str(human_label_artifact),
        "review": review,
        "import": import_report,
        "cost": cost_report,
        "export": export_report,
        "closeout": closeout,
        "evaluation": evaluation,
        "manual_evaluation_command": evaluation.get("command"),
        "failures": failures,
    }


def build_dogfood_closeout_report(
    *,
    labels_path: Path,
    human_label_artifact: Path | None = None,
    sqlite_path: Path | None = None,
    group_id: str | None = None,
    mode: str = "helix",
    helix_data_dir: Path | None = None,
    min_recall_samples: int = 1,
    min_session_samples: int = 1,
    include_all_modes: bool = False,
) -> dict[str, Any]:
    """Build the operator closeout checklist for reviewed dogfood labels."""
    label_artifact = json.loads(labels_path.read_text(encoding="utf-8"))
    _recall_samples, _session_samples, summary = build_dogfood_label_import_samples(
        label_artifact,
        group_id=group_id,
        include_all_modes=include_all_modes,
    )
    recall_count = int(summary.get("recall_sample_count") or 0)
    session_count = int(summary.get("session_sample_count") or 0)
    failures = _dogfood_reviewed_label_failures(
        recall_count=recall_count,
        session_count=session_count,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    reviewed_labels_ready = not failures

    human_label_evidence = None
    human_label_failures = ["missing_human_label_artifact"]
    if human_label_artifact is not None:
        if not human_label_artifact.exists():
            human_label_evidence = {
                "status": "missing",
                "artifact_path": str(human_label_artifact),
                "failures": human_label_failures,
            }
        else:
            try:
                human_label_evidence = load_human_label_evidence(
                    human_label_artifact,
                    min_recall_samples=min_recall_samples,
                    min_session_samples=min_session_samples,
                )
                human_label_failures = _dogfood_human_label_failures(human_label_evidence)
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                human_label_failures = [f"invalid_human_label_artifact:{exc}"]
                human_label_evidence = {
                    "status": "failed",
                    "failures": human_label_failures,
                }
    failures.extend(human_label_failures)
    ready = not failures

    status = "ready_for_native_memory_value" if ready else "needs_evidence"
    evidence_path = human_label_artifact or Path("human-labels.json")
    import_command = None
    export_command = None
    native_memory_value_command = None
    if reviewed_labels_ready:
        import_command = _dogfood_import_command(
            labels_path=labels_path,
            sqlite_path=sqlite_path,
            group_id=group_id,
            include_all_modes=include_all_modes,
        )
        export_command = _dogfood_export_command(
            labels_path=labels_path,
            output_path=evidence_path,
            group_id=group_id,
            include_all_modes=include_all_modes,
        )
        if not human_label_failures:
            native_memory_value_command = _dogfood_native_memory_value_command(
                sqlite_path=sqlite_path,
                human_label_artifact=evidence_path,
                group_id=group_id,
                mode=mode,
                helix_data_dir=helix_data_dir,
                min_recall_samples=min_recall_samples,
                min_session_samples=min_session_samples,
            )
    return {
        "operation": "dogfood_closeout",
        "status": status,
        "ready": ready,
        "failures": failures,
        "labels_path": str(labels_path),
        "group_id": summary.get("group_id"),
        "transcript_hash": summary.get("transcript_hash"),
        "reviewed_labels": {
            "status": "measured" if reviewed_labels_ready else "needs_labels",
            "recall_sample_count": recall_count,
            "session_sample_count": session_count,
            "skipped_turn_count": summary.get("skipped_turn_count", 0),
            "min_recall_samples": min_recall_samples,
            "min_session_samples": min_session_samples,
            "include_all_modes": include_all_modes,
        },
        "human_label_evidence": human_label_evidence
        or {
            "status": "missing",
            "artifact_path": str(evidence_path),
            "failures": human_label_failures,
        },
        "commands": {
            "import_labels": import_command,
            "export_evidence": export_command,
            "native_memory_value": native_memory_value_command,
        },
        "notes": [
            (
                "Closeout commands are withheld until reviewed labels meet the "
                "configured recall/session sample minimums."
            ),
            (
                "Import labels before running the memory-value gate; the human-label "
                "artifact alone does not populate local benefit samples."
            ),
            (
                "Use real reviewed labels only. Replay decisions and synthetic benchmark "
                "artifacts do not satisfy this closeout."
            ),
        ],
    }


def _dogfood_reviewed_label_failures(
    *,
    recall_count: int,
    session_count: int,
    min_recall_samples: int,
    min_session_samples: int,
) -> list[str]:
    failures: list[str] = []
    if recall_count < min_recall_samples:
        failures.append(f"reviewed_recall_samples({recall_count}<{min_recall_samples})")
    if session_count < min_session_samples:
        failures.append(f"reviewed_session_samples({session_count}<{min_session_samples})")
    return failures


def _dogfood_invalid_label_failures(review: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    invalid_turn_count = int(review.get("invalid_label_count") or 0)
    if invalid_turn_count:
        failures.append(f"invalid_label_turns({invalid_turn_count})")
    failures.extend(_dogfood_invalid_session_failures(review))
    return failures


def _dogfood_invalid_session_failures(review: dict[str, Any]) -> list[str]:
    invalid_session_count = int(review.get("invalid_session_sample_count") or 0)
    if invalid_session_count:
        return [f"invalid_session_samples({invalid_session_count})"]
    return []


def _dogfood_human_label_failures(evidence: dict[str, Any]) -> list[str]:
    failure_message = human_label_evidence_failure_message(
        evidence,
        prefix="Human label evidence failed gates",
    )
    if failure_message is None:
        return []
    failures = evidence.get("failures")
    if isinstance(failures, list) and failures:
        return [f"human_label_evidence:{failure}" for failure in failures]
    return [f"human_label_evidence:{evidence.get('status', 'failed')}"]


def render_dogfood_closeout_markdown(report: dict[str, Any]) -> str:
    """Render the dogfood native closeout checklist."""
    reviewed = report.get("reviewed_labels") or {}
    human = report.get("human_label_evidence") or {}
    commands = report.get("commands") or {}
    command_lines = [
        str(command)
        for command in (
            commands.get("import_labels"),
            commands.get("export_evidence"),
            commands.get("native_memory_value"),
        )
        if command
    ]
    lines = [
        "# Engram Dogfood Closeout",
        "",
        f"- Status: {report.get('status')}",
        f"- Group: {report.get('group_id')}",
        f"- Transcript hash: {report.get('transcript_hash')}",
        f"- Reviewed labels: {reviewed.get('status')} "
        f"({reviewed.get('recall_sample_count', 0)} recall, "
        f"{reviewed.get('session_sample_count', 0)} session)",
        f"- Human-label evidence: {human.get('status')}",
        f"- Ready: {report.get('ready')}",
        "",
    ]
    if command_lines:
        lines.extend(["## Commands", "", "```bash"])
        lines.extend(command_lines)
        lines.extend(["```", ""])
    else:
        lines.extend(
            [
                "## Commands",
                "",
                (
                    "- No closeout command is suggested until reviewed labels meet "
                    "the configured sample minimums."
                ),
                "",
            ]
        )
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    lines.extend(["## Notes", ""])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def render_dogfood_export_markdown(report: dict[str, Any]) -> str:
    """Render a compact dogfood evidence export summary."""
    lines = [
        "# Engram Dogfood Evidence Export",
        "",
        f"- Status: {report.get('status')}",
        f"- Source: {report.get('source')}",
        f"- Client: {report.get('client')}",
        f"- Transcript hash: {report.get('transcript_hash')}",
        f"- Human-label artifact: {report.get('human_label_artifact')}",
        f"- Recall samples: {report.get('recall_sample_count', 0)}",
        f"- Session samples: {report.get('session_sample_count', 0)}",
        "",
    ]
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    if report.get("validation_command"):
        lines.extend(
            [
                "## Validate",
                "",
                "```bash",
                str(report.get("validation_command") or ""),
                "```",
                "",
            ]
        )
    notes = [str(note) for note in report.get("notes") or []]
    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def render_dogfood_import_markdown(report: dict[str, Any]) -> str:
    """Render a compact dogfood label import summary."""
    lines = [
        "# Engram Dogfood Label Import",
        "",
        f"- Status: {report.get('status')}",
        f"- Group: {report.get('group_id')}",
        f"- Transcript hash: {report.get('transcript_hash')}",
        f"- Recall samples: {report.get('recall_sample_count', 0)}",
        f"- Session samples: {report.get('session_sample_count', 0)}",
        f"- Skipped turns: {report.get('skipped_turn_count', 0)}",
        "",
    ]
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    return "\n".join(lines)


def render_dogfood_review_markdown(report: dict[str, Any]) -> str:
    """Render a compact dogfood label review status."""
    review = report.get("review") or {}
    labels = report.get("reviewed_labels") or {}
    label_commands = report.get("label_commands") or {}
    summary = review.get("review_queue_summary") or {}
    lines = [
        "# Engram Dogfood Label Review",
        "",
        f"- Status: {report.get('status')}",
        f"- Ready: {report.get('ready')}",
        f"- Group: {report.get('group_id')}",
        f"- Transcript hash: {report.get('transcript_hash')}",
        (
            "- Reviewed turns: "
            f"{review.get('reviewed_turn_count', 0)}/"
            f"{review.get('turn_count', 0)}"
        ),
        (
            "- Importable turns: "
            f"{review.get('importable_turn_count', 0)}/"
            f"{review.get('turn_count', 0)}"
        ),
        f"- Recall samples: {labels.get('recall_sample_count', 0)}",
        f"- Session samples: {labels.get('session_sample_count', 0)}",
        f"- Skipped turns: {labels.get('skipped_turn_count', 0)}",
        "",
    ]
    if summary.get("total"):
        lines.extend(
            [
                "## Review Summary",
                "",
                f"- Queue: {summary.get('total')} turn(s)",
                f"- By reason: {_format_counts(summary.get('by_reason'))}",
                f"- By need type: {_format_counts(summary.get('by_need_type'))}",
                (
                    "- Redacted query hints: "
                    f"{summary.get('redacted_query_hint_count', 0)}"
                ),
                "",
            ]
        )
    if label_commands.get("need_type_filter") or label_commands.get(
        "omitted_turn_command_count"
    ):
        focus = label_commands.get("need_type_filter") or "all"
        turn_command_count = label_commands.get(
            "turn_command_count",
            len(label_commands.get("turns") or []),
        )
        lines.extend(
            [
                f"- Command focus: {focus}",
                (
                    "- Suggested turn commands: "
                    f"{turn_command_count}/"
                    f"{label_commands.get('matching_turn_count', 0)}"
                ),
                (
                    "- Omitted matching commands: "
                    f"{label_commands.get('omitted_turn_command_count', 0)}"
                ),
                "",
            ]
        )
    inspection = report.get("inspection") or {}
    if inspection.get("include_content"):
        lines.extend(
            [
                (
                    "- Review packet: "
                    f"{inspection.get('preview_count', 0)} turn(s) with "
                    f"context={inspection.get('context', 0)}"
                ),
                "- Transcript content is included because `--include-content` was provided.",
                "",
            ]
        )
    next_turn_command = label_commands.get("next_turn") or {}
    if next_turn_command:
        lines.extend(["## Recommended Next Turn", "", "```bash"])
        lines.append(str(next_turn_command.get("inspect") or ""))
        lines.append("")
        lines.append("# If memory was needed:")
        lines.append(str(next_turn_command.get("memory_needed") or ""))
        lines.append("")
        lines.append("# If memory was not needed:")
        lines.append(str(next_turn_command.get("memory_not_needed") or ""))
        if next_turn_command.get("notes_hint"):
            lines.append("")
            lines.append(f"# {next_turn_command.get('notes_hint')}")
        lines.extend(["```", ""])
    queue = review.get("review_queue") or []
    display_queue = _dogfood_review_display_queue(
        queue,
        need_type=_optional_str(label_commands.get("need_type_filter")),
    )
    if display_queue:
        if label_commands.get("need_type_filter"):
            lines.extend(
                [
                    f"## Focused Review Queue ({label_commands.get('need_type_filter')})",
                    "",
                    f"- Showing {len(display_queue)} of {len(queue)} queued turn(s)",
                    "",
                ]
            )
        else:
            lines.extend(["## Review Queue", ""])
        for item in display_queue[:20]:
            reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
            modes = ", ".join(str(mode) for mode in item.get("available_modes") or [])
            mode_suffix = f" | modes: {modes}" if modes else ""
            lines.append(
                "- "
                f"turn {item.get('index')} "
                f"({item.get('content_hash') or 'no_hash'}): {reasons}"
                f"{mode_suffix}"
            )
        if len(display_queue) > 20:
            lines.append(f"- ... {len(display_queue) - 20} more")
        lines.append("")
    inspection_previews = report.get("inspection_previews") or []
    if inspection_previews:
        lines.extend(["## Review Packet", ""])
        for preview in inspection_previews[:10]:
            target = preview.get("target") or {}
            lines.append(
                f"### Turn {target.get('turn')} ({target.get('content_hash') or 'no_hash'})"
            )
            lines.append("")
            lines.append(f"- Status: {preview.get('status')}")
            lines.append(f"- Need type: {target.get('need_type')}")
            lines.append(f"- Content redacted: {preview.get('content_redacted')}")
            failures = [str(failure) for failure in preview.get("failures") or []]
            if failures:
                lines.append(f"- Failures: {', '.join(failures)}")
            context_turns = preview.get("context_turns") or []
            if context_turns:
                lines.append("")
                lines.append("#### Context")
                lines.append("")
            for item in context_turns:
                marker = " <- target" if item.get("selected") else ""
                lines.append(
                    f"##### {item.get('role')} {item.get('content_hash')}{marker}"
                )
                lines.append("")
                if "content" in item:
                    lines.append("```text")
                    lines.append(str(item.get("content") or ""))
                    lines.append("```")
                else:
                    lines.append("- Content omitted.")
                lines.append("")
            label_command = preview.get("label_command") or {}
            if label_command:
                lines.extend(["#### Label Commands", "", "```bash"])
                lines.append("# If memory was needed:")
                lines.append(str(label_command.get("memory_needed") or ""))
                lines.append("")
                lines.append("# If memory was not needed:")
                lines.append(str(label_command.get("memory_not_needed") or ""))
                if label_command.get("notes_hint"):
                    lines.append("")
                    lines.append(f"# {label_command.get('notes_hint')}")
                lines.extend(["```", ""])
        if len(inspection_previews) > 10:
            lines.append(f"- ... {len(inspection_previews) - 10} more preview(s)")
            lines.append("")
    invalid_session_samples = review.get("invalid_session_samples") or []
    if invalid_session_samples:
        lines.extend(["## Invalid Session Samples", ""])
        for item in invalid_session_samples[:20]:
            reasons = ", ".join(str(reason) for reason in item.get("reasons") or [])
            lines.append(
                "- "
                f"session sample {item.get('index')}: {reasons}"
            )
        if len(invalid_session_samples) > 20:
            lines.append(f"- ... {len(invalid_session_samples) - 20} more")
        lines.append("")
    turn_commands = label_commands.get("turns") or []
    if turn_commands:
        lines.extend(["## Suggested Turn Label Commands", ""])
        for item in turn_commands[:10]:
            lines.append(f"### Turn {item.get('turn')}")
            lines.append("")
            lines.append("```bash")
            lines.append(str(item.get("inspect") or ""))
            lines.append("")
            lines.append("# If memory was needed:")
            lines.append(str(item.get("memory_needed") or ""))
            lines.append("")
            lines.append("# If memory was not needed:")
            lines.append(str(item.get("memory_not_needed") or ""))
            if item.get("notes_hint"):
                lines.append("")
                lines.append(f"# {item.get('notes_hint')}")
            lines.append("```")
            lines.append("")
        if len(turn_commands) > 10:
            lines.append(f"- ... {len(turn_commands) - 10} more turn commands")
            lines.append("")
    session_command = label_commands.get("session")
    if session_command:
        lines.extend(["## Suggested Session Label Command", "", "```bash"])
        lines.append(str(session_command))
        if label_commands.get("session_notes_hint"):
            lines.append("")
            lines.append(f"# {label_commands.get('session_notes_hint')}")
        lines.extend(["```", ""])
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    commands = report.get("next_commands") or {}
    next_commands = [
        str(command)
        for command in (
            commands.get("import_labels"),
            commands.get("export_evidence"),
        )
        if command
    ]
    if next_commands:
        lines.extend(["## Next", "", "```bash"])
        lines.extend(next_commands)
        lines.extend(["```", ""])
    elif report.get("status") == "trace_only":
        lines.extend(
            [
                "## Next",
                "",
                "- No import/export command is suggested because this artifact has no "
                "labelable turns.",
                "",
            ]
        )
    return "\n".join(lines)


def _dogfood_review_display_queue(
    queue: list[dict[str, Any]],
    *,
    need_type: str | None,
) -> list[dict[str, Any]]:
    if not need_type:
        return queue
    return [
        item
        for item in queue
        if isinstance(item, dict) and _optional_str(item.get("need_type")) == need_type
    ]


def render_dogfood_turn_inspection_markdown(report: dict[str, Any]) -> str:
    """Render one local dogfood turn for human review."""
    target = report.get("target") or {}
    lines = [
        "# Engram Dogfood Turn Inspection",
        "",
        f"- Status: {report.get('status')}",
        f"- Labels: {report.get('labels_path')}",
        f"- Source: {report.get('source_path')}",
        f"- Turn: {target.get('turn')}",
        f"- Content hash: {target.get('content_hash')}",
        f"- Need type: {target.get('need_type')}",
        f"- Query hint redacted: {target.get('query_hint_redacted')}",
        f"- Content redacted: {report.get('content_redacted')}",
        "",
    ]
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    context_turns = report.get("context_turns") or []
    if context_turns:
        lines.extend(["## Context", ""])
        for item in context_turns:
            marker = " <- target" if item.get("selected") else ""
            lines.append(
                f"### {item.get('role')} {item.get('content_hash')}{marker}"
            )
            lines.append("")
            if "content" in item:
                lines.append("```text")
                lines.append(str(item.get("content") or ""))
                lines.append("```")
            else:
                lines.append("- Content omitted; rerun with `--include-content`.")
            lines.append("")
    label_command = report.get("label_command") or {}
    if label_command:
        lines.extend(["## Suggested Label Commands", "", "```bash"])
        lines.append("# If memory was needed:")
        lines.append(str(label_command.get("memory_needed") or ""))
        lines.append("")
        lines.append("# If memory was not needed:")
        lines.append(str(label_command.get("memory_not_needed") or ""))
        if label_command.get("notes_hint"):
            lines.append("")
            lines.append(f"# {label_command.get('notes_hint')}")
        lines.extend(["```", ""])
    notes = [str(note) for note in report.get("notes") or []]
    if notes:
        lines.extend(["## Notes", ""])
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    return "\n".join(lines)


def render_dogfood_label_edit_markdown(report: dict[str, Any]) -> str:
    """Render a compact label edit result."""
    review = report.get("review") or {}
    lines = [
        "# Engram Dogfood Label Edit",
        "",
        f"- Operation: {report.get('operation')}",
        f"- Status: {report.get('status')}",
        f"- Labels: {report.get('labels_path')}",
        f"- Review status: {review.get('status')}",
        f"- Ready: {review.get('ready')}",
        "",
    ]
    failures = [str(failure) for failure in review.get("failures") or []]
    if failures:
        lines.extend(["## Remaining", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    return "\n".join(lines)


def render_dogfood_finalize_markdown(report: dict[str, Any]) -> str:
    """Render the one-command dogfood finalization report."""
    evaluation = report.get("evaluation") or {}
    import_report = report.get("import") or {}
    cost_report = report.get("cost") or {}
    export_report = report.get("export") or {}
    closeout = report.get("closeout") or {}
    lines = [
        "# Engram Dogfood Finalize",
        "",
        f"- Status: {report.get('status')}",
        f"- Ready: {report.get('ready')}",
        f"- Phase: {report.get('phase')}",
        f"- Labels: {report.get('labels_path')}",
        f"- Human-label artifact: {report.get('human_label_artifact')}",
        f"- Import: {import_report.get('status', 'not_run')}",
        f"- Replay cost: {cost_report.get('status', 'not_run')}",
        f"- Export: {export_report.get('status', 'not_run')}",
        f"- Closeout: {closeout.get('status', 'not_run')}",
        f"- Evaluation: {evaluation.get('status', 'not_run')}",
        "",
    ]
    if report.get("status") == "needs_evaluation":
        lines.extend(
            [
                "## Manual Evaluation Required",
                "",
                (
                    "The dogfood labels were imported and exported, but finalization "
                    "is not complete until the native memory-value gate is measured."
                ),
                "",
            ]
        )
    if evaluation.get("command"):
        lines.extend(["## Evaluation Command", "", "```bash"])
        lines.append(str(evaluation.get("command")))
        lines.extend(["```", ""])
    failures = [str(failure) for failure in report.get("failures") or []]
    if failures:
        lines.extend(["## Failures", ""])
        for failure in failures:
            lines.append(f"- {failure}")
        lines.append("")
    review = report.get("review") or {}
    if report.get("phase") == "review":
        lines.append(render_dogfood_review_markdown(review).rstrip())
        lines.append("")
    return "\n".join(lines)


def _recall_sample_to_human_label(
    sample: StoredRecallEvalSample,
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "query": sample.query or _redacted_query_from_notes(sample.notes),
        "recallTriggered": sample.recall_triggered,
        "recallHelped": sample.recall_helped,
        "recallNeeded": sample.recall_needed,
        "packetsSurfaced": sample.packets_surfaced,
        "packetsUsed": sample.packets_used,
        "falseRecalls": sample.false_recalls,
        "stalePackets": sample.stale_packets,
        "correctedPackets": sample.corrected_packets,
        "notes": sample.notes or "dogfood reviewed recall label",
    }


def _session_sample_to_human_label(
    sample: StoredSessionContinuitySample,
    *,
    source: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "scenario": sample.scenario or _redacted_query_from_notes(sample.notes),
        "baselineScore": sample.baseline_score,
        "memoryScore": sample.memory_score,
        "openLoopExpected": sample.open_loop_expected,
        "openLoopRecovered": sample.open_loop_recovered,
        "temporalExpected": sample.temporal_expected,
        "temporalCorrect": sample.temporal_correct,
        "notes": sample.notes or "dogfood reviewed continuity label",
    }


def _redacted_query_from_notes(notes: str | None) -> str:
    if not notes:
        return "dogfood reviewed turn"
    turn_match = re.search(r"(?:^|; )turn=([^;]+)", notes)
    hash_match = re.search(r"(?:^|; )content_hash=([^;]+)", notes)
    parts = ["dogfood reviewed turn"]
    if turn_match:
        parts.append(str(turn_match.group(1)))
    if hash_match:
        parts.append(f"hash {hash_match.group(1)}")
    return " ".join(parts)


def _dogfood_evidence_validation_command(output_path: Path) -> str:
    path = shlex.quote(str(output_path))
    return (
        "engram evaluate "
        f"--human-label-artifact {path} "
        "--require-human-label-evidence "
        "--format json"
    )


async def _run_dogfood_memory_value_evaluation(
    *,
    sqlite_path: Path | None,
    human_label_artifact: Path,
    group_id: str | None,
    mode: str,
    helix_data_dir: Path | None,
    min_recall_samples: int,
    min_session_samples: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    invocation = _dogfood_memory_value_evaluation_invocation(
        sqlite_path=sqlite_path,
        human_label_artifact=human_label_artifact,
        group_id=group_id,
        mode=mode,
        helix_data_dir=helix_data_dir,
        min_recall_samples=min_recall_samples,
        min_session_samples=min_session_samples,
    )
    env = os.environ.copy()
    env.update(invocation["env"])
    try:
        process = await asyncio.create_subprocess_exec(
            *invocation["argv"],
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        try:
            process.kill()  # type: ignore[possibly-undefined]
        except ProcessLookupError:
            pass
        return {
            "status": "timeout",
            "exit_code": None,
            "command": invocation["display"],
            "timeout_seconds": timeout_seconds,
        }
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    parsed = _json_object_from_text(stdout)
    status = "measured" if process.returncode == 0 else "failed"
    result = {
        "status": status,
        "exit_code": process.returncode,
        "command": invocation["display"],
        "timeout_seconds": timeout_seconds,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }
    if parsed is not None:
        result["report"] = parsed
        if isinstance(parsed.get("memory_value"), dict):
            result["memory_value"] = parsed["memory_value"]
    return result


def _dogfood_memory_value_evaluation_invocation(
    *,
    sqlite_path: Path | None,
    human_label_artifact: Path,
    group_id: str | None,
    mode: str,
    helix_data_dir: Path | None,
    min_recall_samples: int,
    min_session_samples: int,
) -> dict[str, Any]:
    argv: list[str] = [
        sys.executable,
        "-m",
        "engram",
        "evaluate",
        "--mode",
        mode,
        "--memory-value",
        "--require-memory-value",
        "--human-label-artifact",
        str(human_label_artifact),
        "--require-human-label-evidence",
        "--min-human-recall-samples",
        str(min_recall_samples),
        "--min-human-session-samples",
        str(min_session_samples),
        "--format",
        "json",
    ]
    if sqlite_path is not None:
        argv.extend(["--sqlite-path", str(sqlite_path)])
    if group_id:
        argv.extend(["--group-id", group_id])
    if mode == "helix" and helix_data_dir is not None:
        argv.extend(["--helix-data-dir", str(helix_data_dir)])
    env: dict[str, str] = {}
    display_prefix = ""
    if mode == "helix":
        env = {"ENGRAM_MODE": "helix", "ENGRAM_HELIX__TRANSPORT": "native"}
        display_prefix = "ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native "
    display_argv = ["engram", *argv[3:]]
    return {
        "argv": argv,
        "env": env,
        "display": f"{display_prefix}{_shell_join(display_argv)}",
    }


def _json_object_from_text(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _dogfood_import_command(
    *,
    labels_path: Path,
    sqlite_path: Path | None,
    group_id: str | None,
    include_all_modes: bool,
) -> str:
    parts = [
        "engram",
        "dogfood",
        "import-labels",
        "--labels",
        str(labels_path),
    ]
    if sqlite_path is not None:
        parts.extend(["--sqlite-path", str(sqlite_path)])
    if group_id:
        parts.extend(["--group-id", group_id])
    if include_all_modes:
        parts.append("--include-all-modes")
    return _shell_join(parts)


def _dogfood_export_command(
    *,
    labels_path: Path,
    output_path: Path,
    group_id: str | None,
    include_all_modes: bool,
) -> str:
    parts = [
        "engram",
        "dogfood",
        "export-evidence",
        "--labels",
        str(labels_path),
        "--out",
        str(output_path),
        "--source",
        "native_dogfood_harness",
        "--client",
        "Codex",
        "--captured-at",
        "<ISO-8601>",
        "--labeler",
        "<human-reviewer>",
    ]
    if group_id:
        parts.extend(["--group-id", group_id])
    if include_all_modes:
        parts.append("--include-all-modes")
    return _shell_join(parts)


def _dogfood_review_label_commands(
    *,
    labels_path: Path,
    review: dict[str, Any],
    session_count: int,
    min_session_samples: int,
    need_type: str | None,
    command_limit: int,
) -> dict[str, Any]:
    turn_commands: list[dict[str, Any]] = []
    command_limit = max(0, int(command_limit))
    filtered_items = [
        item
        for item in review.get("review_queue") or []
        if isinstance(item, dict)
        and item.get("index") is not None
        and (
            not _optional_str(need_type)
            or _optional_str(item.get("need_type")) == _optional_str(need_type)
        )
    ]
    for item in filtered_items[:command_limit]:
        turn_command = _dogfood_turn_label_command(labels_path=labels_path, item=item)
        if turn_command is not None:
            turn_commands.append(turn_command)

    session_missing = max(0, min_session_samples - session_count)
    turn_count = int(review.get("turn_count") or 0)
    session_command = None
    if session_missing > 0 and turn_count > 0:
        session_command = _shell_join(
            [
                "engram",
                "dogfood",
                "label-session",
                "--labels",
                str(labels_path),
                "--scenario",
                "<reviewed continuity scenario>",
                "--baseline-score",
                "<0.0-1.0>",
                "--memory-score",
                "<0.0-1.0>",
            ]
        )

    return {
        "turns": turn_commands,
        "next_turn": turn_commands[0] if turn_commands else None,
        "need_type_filter": _optional_str(need_type),
        "command_limit": command_limit,
        "turn_command_count": len(turn_commands),
        "matching_turn_count": len(filtered_items),
        "omitted_turn_command_count": max(0, len(filtered_items) - len(turn_commands)),
        "session": session_command,
        "session_notes_hint": (
            "Add --notes with real session-level review notes if useful."
            if session_command
            else None
        ),
        "session_samples_missing": session_missing,
    }


def _dogfood_turn_label_command(
    *,
    labels_path: Path,
    item: dict[str, Any],
) -> dict[str, Any] | None:
    if item.get("index") is None:
        return None
    turn_index = str(item.get("index"))
    available_modes = [
        str(mode)
        for mode in item.get("available_modes") or []
        if _optional_str(mode) in SUPPORTED_DOGFOOD_MODES
    ]
    best_placeholder = _dogfood_best_mode_placeholder(available_modes)
    helpful_placeholder = _dogfood_helpful_mode_placeholder(available_modes)
    memory_not_needed = [
        "engram",
        "dogfood",
        "label-turn",
        "--labels",
        str(labels_path),
        "--turn",
        turn_index,
        "--memory-needed",
        "no",
    ]
    if "off" in available_modes:
        memory_not_needed.extend(["--best-mode", "off"])
    elif false_mode := _dogfood_false_mode(available_modes):
        memory_not_needed.extend(["--false-recall-mode", false_mode])
    else:
        memory_not_needed.extend(["--best-mode", best_placeholder])
    return {
        "turn": item.get("index"),
        "content_hash": item.get("content_hash"),
        "reasons": list(item.get("reasons") or []),
        "available_modes": available_modes,
        "inspect": _shell_join(
            [
                "engram",
                "dogfood",
                "inspect-turn",
                "--labels",
                str(labels_path),
                "--turn",
                turn_index,
                "--context",
                "1",
                "--include-content",
            ]
        ),
        "memory_needed": _shell_join(
            [
                "engram",
                "dogfood",
                "label-turn",
                "--labels",
                str(labels_path),
                "--turn",
                turn_index,
                "--memory-needed",
                "yes",
                "--best-mode",
                best_placeholder,
                "--helpful-mode",
                helpful_placeholder,
            ]
        ),
        "memory_not_needed": _shell_join(memory_not_needed),
        "notes_hint": "Add --notes with a real observed reason if useful.",
    }


def _dogfood_best_mode_placeholder(available_modes: list[str]) -> str:
    for mode in ("cached", "gated_lite", "gated_medium", "deep", "startup", "off"):
        if mode in available_modes:
            return mode
    return "<best-mode>"


def _dogfood_helpful_mode_placeholder(available_modes: list[str]) -> str:
    for mode in ("cached", "gated_lite", "gated_medium", "deep", "startup"):
        if mode in available_modes:
            return mode
    return _dogfood_best_mode_placeholder(available_modes)


def _dogfood_false_mode(available_modes: list[str]) -> str | None:
    for mode in ("deep", "gated_medium", "gated_lite", "cached", "startup"):
        if mode in available_modes:
            return mode
    return None


def _dogfood_review_queue_summary(review_queue: list[dict[str, Any]]) -> dict[str, Any]:
    by_reason: dict[str, int] = {}
    by_need_type: dict[str, int] = {}
    redacted_query_hint_count = 0

    for item in review_queue:
        need_type = _optional_str(item.get("need_type")) or "unknown"
        by_need_type[need_type] = by_need_type.get(need_type, 0) + 1
        if item.get("query_hint_redacted"):
            redacted_query_hint_count += 1
        reasons = [str(reason) for reason in item.get("reasons") or []]
        if not reasons:
            by_reason["unknown"] = by_reason.get("unknown", 0) + 1
            continue
        for reason in reasons:
            by_reason[reason] = by_reason.get(reason, 0) + 1

    return {
        "total": len(review_queue),
        "by_reason": _sorted_count_mapping(by_reason),
        "by_need_type": _sorted_count_mapping(by_need_type),
        "redacted_query_hint_count": redacted_query_hint_count,
        "next_review_turn": _dogfood_next_review_turn(review_queue),
    }


def _dogfood_next_review_turn(review_queue: list[dict[str, Any]]) -> dict[str, Any] | None:
    for item in review_queue:
        if item.get("index") is None:
            continue
        return {
            "index": item.get("index"),
            "content_hash": item.get("content_hash"),
            "need_type": item.get("need_type"),
            "reasons": list(item.get("reasons") or []),
            "query_hint_redacted": bool(item.get("query_hint_redacted")),
            "available_modes": list(item.get("available_modes") or []),
        }
    return None


def _sorted_count_mapping(counts: dict[str, int]) -> dict[str, int]:
    return {
        key: value
        for key, value in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    }


def _dogfood_native_memory_value_command(
    *,
    sqlite_path: Path | None,
    human_label_artifact: Path,
    group_id: str | None,
    mode: str,
    helix_data_dir: Path | None,
    min_recall_samples: int,
    min_session_samples: int,
) -> str:
    parts: list[str] = ["engram", "evaluate", "--mode", mode]
    if sqlite_path is not None:
        parts.extend(["--sqlite-path", str(sqlite_path)])
    parts.extend(["--memory-value", "--require-memory-value"])
    parts.extend(
        [
            "--human-label-artifact",
            str(human_label_artifact),
            "--require-human-label-evidence",
            "--min-human-recall-samples",
            str(min_recall_samples),
            "--min-human-session-samples",
            str(min_session_samples),
            "--format",
            "json",
        ]
    )
    if group_id:
        parts.extend(["--group-id", group_id])
    if mode == "helix" and helix_data_dir is not None:
        parts.extend(["--helix-data-dir", str(helix_data_dir)])
    command = _shell_join(parts)
    if mode == "helix":
        return f"ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native {command}"
    return command


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _stable_sample_id(prefix: str, *parts: Any) -> str:
    normalized = "\x1f".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _dogfood_label_template(artifact: dict[str, Any]) -> dict[str, Any]:
    if artifact.get("kind") == "engram.dogfood_label_template.v1":
        return artifact
    label_template = artifact.get("label_template")
    if isinstance(label_template, dict):
        return label_template
    raise ValueError(
        "dogfood labels must be an engram.dogfood_label_template.v1 artifact "
        "or a replay report containing label_template"
    )


def _write_dogfood_label_artifact(
    artifact: dict[str, Any],
    template: dict[str, Any],
    output_path: Path,
) -> None:
    if artifact.get("kind") == "engram.dogfood_label_template.v1":
        payload = template
    else:
        payload = dict(artifact)
        payload["label_template"] = template
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_label_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"yes", "true"}:
        return True
    if normalized in {"no", "false"}:
        return False
    raise ValueError(f"Unsupported boolean label: {value}")


def _normalize_label_mode(
    value: str | None,
    decision_modes: set[str],
    *,
    allow_none: bool = False,
) -> str | None:
    normalized = _optional_str(value)
    if normalized in {None, "none", "null"}:
        if allow_none:
            return None
        raise ValueError("mode is required")
    if normalized not in SUPPORTED_DOGFOOD_MODES:
        raise ValueError(f"Unsupported dogfood mode: {normalized}")
    if normalized not in decision_modes:
        raise ValueError(f"Mode was not replayed for this turn: {normalized}")
    return normalized


def _normalize_label_modes(values: list[str], decision_modes: set[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        mode = _normalize_label_mode(value, decision_modes)
        if mode is not None and mode not in normalized:
            normalized.append(mode)
    return normalized


def _clamp_score(value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, parsed))


def _dogfood_label_review_details(
    template: dict[str, Any],
    *,
    include_all_modes: bool,
) -> dict[str, Any]:
    raw_turns = [turn for turn in template.get("turns") or [] if isinstance(turn, dict)]
    source = template.get("source") if isinstance(template.get("source"), dict) else {}
    content_redacted = bool(source.get("content_redacted", True))
    review_queue: list[dict[str, Any]] = []
    invalid_turns: list[dict[str, Any]] = []
    invalid_session_samples: list[dict[str, Any]] = []
    selected_mode_counts: dict[str, int] = {}
    reviewed_turn_count = 0
    importable_turn_count = 0

    for turn in raw_turns:
        labels = turn.get("labels") if isinstance(turn.get("labels"), dict) else {}
        decisions = [d for d in turn.get("decisions") or [] if isinstance(d, dict)]
        decision_modes = {_optional_str(decision.get("mode")) for decision in decisions}
        decision_modes.discard(None)
        memory_needed = labels.get("memory_was_needed")
        reasons: list[str] = []
        if not isinstance(memory_needed, bool):
            reasons.append("memory_was_needed_missing")
            selected_modes: set[str] = set()
        else:
            reviewed_turn_count += 1
            selected_modes = (
                set(str(mode) for mode in decision_modes)
                if include_all_modes
                else _selected_label_modes(labels)
            )
            if not selected_modes:
                reasons.append("no_labeled_modes")

        unsupported_modes = sorted(
            mode for mode in selected_modes if mode not in SUPPORTED_DOGFOOD_MODES
        )
        unreplayed_modes = sorted(mode for mode in selected_modes if mode not in decision_modes)
        if unsupported_modes:
            reasons.append(f"unsupported_modes:{','.join(unsupported_modes)}")
        if unreplayed_modes:
            reasons.append(f"unreplayed_modes:{','.join(unreplayed_modes)}")
        if _dogfood_has_placeholder_text(labels.get("notes")):
            reasons.append("placeholder_label_notes")

        if reasons:
            queue_item = {
                "index": turn.get("index"),
                "content_hash": turn.get("content_hash"),
                "need_type": turn.get("need_type"),
                "query_hint": None if content_redacted else turn.get("query_hint"),
                "query_hint_redacted": bool(
                    content_redacted
                    and (
                        _optional_str(turn.get("query_hint"))
                        or bool(turn.get("query_hint_redacted"))
                    )
                ),
                "available_modes": sorted(str(mode) for mode in decision_modes if mode),
                "reasons": reasons,
            }
            review_queue.append(queue_item)
            if (
                unsupported_modes
                or unreplayed_modes
                or any(
                    str(reason).startswith("placeholder_")
                    for reason in reasons
                )
            ):
                invalid_turns.append(
                    {
                        **queue_item,
                        "unsupported_modes": unsupported_modes,
                        "unreplayed_modes": unreplayed_modes,
                    }
                )
            continue

        importable_turn_count += 1
        for mode in sorted(selected_modes):
            selected_mode_counts[mode] = selected_mode_counts.get(mode, 0) + 1

    for index, raw_sample in enumerate(_dogfood_raw_session_samples(template)):
        reasons = _dogfood_session_sample_reasons(raw_sample)
        if reasons:
            invalid_session_samples.append(
                {
                    "index": index,
                    "scenario": _optional_str(raw_sample.get("scenario")),
                    "reasons": reasons,
                }
            )

    return {
        "turn_count": len(raw_turns),
        "reviewed_turn_count": reviewed_turn_count,
        "importable_turn_count": importable_turn_count,
        "unreviewed_turn_count": max(0, len(raw_turns) - reviewed_turn_count),
        "review_queue_count": len(review_queue),
        "invalid_label_count": len(invalid_turns),
        "invalid_session_sample_count": len(invalid_session_samples),
        "selected_mode_counts": selected_mode_counts,
        "review_queue_summary": _dogfood_review_queue_summary(review_queue),
        "review_queue": review_queue,
        "invalid_turns": invalid_turns,
        "invalid_session_samples": invalid_session_samples,
    }


def _selected_label_modes(labels: dict[str, Any]) -> set[str]:
    modes = _string_set(labels.get("helpful_modes"))
    modes.update(_string_set(labels.get("false_recall_modes")))
    best_mode = _optional_str(labels.get("best_mode"))
    if best_mode:
        modes.add(best_mode)
    return modes


def _dogfood_label_notes(
    *,
    transcript_hash: str,
    turn: dict[str, Any],
    label_notes: Any,
) -> str:
    parts = [
        f"dogfood_transcript={transcript_hash}",
        f"turn={turn.get('index')}",
        f"content_hash={turn.get('content_hash')}",
    ]
    notes = _optional_str(label_notes)
    if notes:
        parts.append(f"review_notes={notes}")
    return "; ".join(parts)


def _dogfood_session_samples(
    template: dict[str, Any],
    *,
    group_id: str,
    transcript_hash: str,
) -> list[StoredSessionContinuitySample]:
    samples: list[StoredSessionContinuitySample] = []
    for index, raw in enumerate(_dogfood_raw_session_samples(template)):
        if _dogfood_session_sample_reasons(raw):
            continue
        samples.append(
            StoredSessionContinuitySample(
                group_id=group_id,
                baseline_score=_float(raw.get("baseline_score", raw.get("baselineScore"))),
                memory_score=_float(raw.get("memory_score", raw.get("memoryScore"))),
                open_loop_expected=bool(
                    raw.get("open_loop_expected", raw.get("openLoopExpected", False))
                ),
                open_loop_recovered=bool(
                    raw.get("open_loop_recovered", raw.get("openLoopRecovered", False))
                ),
                temporal_expected=bool(
                    raw.get("temporal_expected", raw.get("temporalExpected", False))
                ),
                temporal_correct=bool(
                    raw.get("temporal_correct", raw.get("temporalCorrect", False))
                ),
                source="dogfood_review",
                scenario=_optional_str(raw.get("scenario")),
                notes=(
                    f"dogfood_transcript={transcript_hash}; "
                    f"{_optional_str(raw.get('notes')) or 'session_review'}"
                ),
                id=_stable_sample_id("esc", group_id, transcript_hash, index),
            )
        )
    return samples


def _dogfood_raw_session_samples(template: dict[str, Any]) -> list[dict[str, Any]]:
    raw_samples = template.get("session_samples") or template.get("sessionSamples") or []
    return [raw for raw in raw_samples if isinstance(raw, dict)]


def _dogfood_session_sample_reasons(sample: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _dogfood_has_placeholder_text(sample.get("scenario")):
        reasons.append("placeholder_session_scenario")
    if _dogfood_has_placeholder_text(sample.get("notes")):
        reasons.append("placeholder_session_notes")
    return reasons


def _dogfood_has_placeholder_text(value: Any) -> bool:
    text = _optional_str(value)
    return bool(text and _DOGFOOD_PLACEHOLDER_TOKEN_RE.search(text))


def _string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if item}


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [mode for mode in modes if mode not in SUPPORTED_DOGFOOD_MODES]
    if unknown:
        raise SystemExit(f"Unsupported dogfood modes: {', '.join(unknown)}")
    return modes or list(DEFAULT_DOGFOOD_MODES)


def parse_transcript_text(raw_text: str, *, source: str = "transcript") -> list[DogfoodTurn]:
    stripped = raw_text.lstrip()
    if stripped.startswith("{"):
        return _parse_jsonl_turns(raw_text, source=source)
    jsonl_turns = _parse_jsonl_turns(raw_text, source=source, tolerate_errors=True)
    if jsonl_turns:
        return jsonl_turns
    return _parse_markdown_turns(raw_text, source=source)


def render_dogfood_replay_markdown(report: dict[str, Any]) -> str:
    source = report.get("source") or {}
    trace_evidence = report.get("trace_evidence") or {}
    lines = [
        "# Engram Dogfood Replay",
        "",
        f"- Status: {report.get('status')}",
        f"- Transcript hash: {source.get('transcript_hash')}",
        f"- User turns: {source.get('user_turn_count', 0)}",
        f"- Content redacted: {source.get('content_redacted')}",
        "",
        "## Mode Summary",
        "",
    ]
    for mode, summary in (report.get("mode_summaries") or {}).items():
        lines.append(
            "- "
            f"{mode}: triggered {summary.get('triggered_count', 0)}/"
            f"{summary.get('turn_count', 0)}, "
            f"avg estimated latency {summary.get('avg_estimated_latency_ms', 0)}ms"
        )
    if trace_evidence.get("status") == "measured":
        duration = trace_evidence.get("duration_ms") or {}
        lines.extend(
            [
                "",
                "## Trace Evidence",
                "",
                f"- AXI trace records: {trace_evidence.get('trace_count', 0)}",
                f"- Status counts: {_format_counts(trace_evidence.get('status_counts'))}",
                f"- Operation counts: {_format_counts(trace_evidence.get('operation_counts'))}",
                f"- Origin counts: {_format_counts(trace_evidence.get('origin_counts'))}",
                (
                    "- Duration: "
                    f"avg {duration.get('avg', 0)}ms, "
                    f"p95 {duration.get('p95', 0)}ms, "
                    f"max {duration.get('max', 0)}ms"
                ),
                f"- Cache hits: {trace_evidence.get('cache_hit_count', 0)}",
                f"- Degraded/timeouts: {trace_evidence.get('degraded_count', 0)}/"
                f"{trace_evidence.get('timeout_count', 0)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Labels",
            "",
            "- Replay does not save labels automatically; export this report for review.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_dogfood_prepare_markdown(report: dict[str, Any]) -> str:
    """Render the dogfood prepare bundle summary."""
    paths = report.get("paths") or {}
    replay = report.get("replay") or {}
    labels = report.get("labels") or {}
    commands = report.get("next_commands") or {}
    lines = [
        "# Engram Dogfood Prepare",
        "",
        f"- Status: {report.get('status')}",
        f"- Output: {report.get('output_dir')}",
        f"- User turns: {replay.get('user_turn_count', 0)}",
        f"- Replay content redacted: {replay.get('content_redacted')}",
        f"- Label content redacted: {labels.get('content_redacted')}",
        f"- Trace evidence: {replay.get('trace_status')} ({replay.get('trace_count', 0)})",
        "",
        "## Files",
        "",
        f"- Replay report: {paths.get('replay_report')}",
        f"- Labels: {paths.get('labels')}",
        f"- Review report: {paths.get('review_report')}",
        f"- Review markdown: {paths.get('review_markdown')}",
        "",
        "## Next",
        "",
        "```bash",
    ]
    for command in (commands.get("review"), commands.get("finalize")):
        if command:
            lines.append(str(command))
    lines.extend(["```", ""])
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def render_dogfood_candidate_markdown(report: dict[str, Any]) -> str:
    """Render redacted dogfood transcript candidates."""
    lines = [
        "# Engram Dogfood Candidates",
        "",
        f"- Status: {report.get('status')}",
        f"- Root: {report.get('root')}",
        f"- Files scanned: {report.get('files_scanned', 0)}",
        f"- Candidates: {report.get('candidate_count', 0)}",
        f"- Trace-only: {report.get('trace_only_count', 0)}",
        f"- Skipped: {report.get('skipped_count', 0)}",
        f"- Project mismatches: {report.get('project_mismatch_count', 0)}",
        "",
    ]
    candidates = report.get("candidates") or []
    if candidates:
        lines.extend(["## Top Candidates", ""])
        for index, item in enumerate(candidates, start=1):
            lines.extend(
                [
                    f"### {index}. {item.get('status')}",
                    "",
                    f"- Path: {item.get('path')}",
                    f"- Session cwd: {item.get('session_cwd') or 'unknown'}",
                    f"- Project match: {item.get('project_match')}",
                    f"- Labelable turns: {item.get('labelable_turn_count', 0)}",
                    f"- Assistant turns: {item.get('assistant_turn_count', 0)}",
                    (
                        "- Trace: "
                        f"{item.get('trace_status')} ({item.get('trace_count', 0)})"
                    ),
                    f"- Transcript hash: {item.get('transcript_hash')}",
                ]
            )
            command = item.get("prepare_command")
            if command:
                lines.extend(["", "```bash", str(command), "```"])
            lines.append("")
    else:
        lines.extend(
            [
                "## Top Candidates",
                "",
                "- No transcript files with labelable user turns were found.",
                "",
            ]
        )
    for note in report.get("notes") or []:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def _parse_jsonl_turns(
    raw_text: str,
    *,
    source: str,
    tolerate_errors: bool = False,
) -> list[DogfoodTurn]:
    turns: list[DogfoodTurn] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            if tolerate_errors:
                return []
            raise
        if not isinstance(data, dict):
            continue
        turn = _turn_from_json(data, source=source)
        if turn is not None:
            turns.append(turn)
    return turns


def _turn_from_json(data: dict[str, Any], *, source: str) -> DogfoodTurn | None:
    if data.get("type") == "response_item" and isinstance(data.get("payload"), dict):
        return _turn_from_json(data["payload"], source=source)

    if isinstance(data.get("message"), dict):
        nested = dict(data["message"])
        nested.setdefault("timestamp", data.get("timestamp"))
        return _turn_from_json(nested, source=source)

    role = str(data.get("role") or data.get("type") or "").lower()
    content = _content_text(data.get("content") or data.get("text"))
    if not content and data.get("operation"):
        role = "tool"
        content = f"{data.get('operation')} {data.get('status') or ''}".strip()
    if role not in {"user", "assistant", "system", "tool"} or not isinstance(content, str):
        return None
    if _is_dogfood_bootstrap_message(role=role, content=content):
        return None
    return DogfoodTurn(
        role=role,
        content=content,
        timestamp=data.get("timestamp") if isinstance(data.get("timestamp"), str) else None,
        source=source,
        metadata={key: value for key, value in data.items() if key not in {"content", "text"}},
    )


def _is_dogfood_bootstrap_message(*, role: str, content: str) -> bool:
    if role != "user":
        return False
    stripped = content.lstrip()
    return (
        stripped.startswith("# AGENTS.md instructions for ")
        or stripped.startswith("<environment_context>")
        or stripped.startswith("<goal_context>")
        or _is_dogfood_smoke_prompt(stripped)
    )


def _is_dogfood_smoke_prompt(content: str) -> bool:
    normalized = " ".join(content.strip().lower().rstrip(".").split())
    return normalized in {
        "reply exactly ok",
        "reply exactly: ok",
        "respond exactly ok",
        "respond exactly: ok",
    }


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_content_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "input_text", "output_text", "message"):
            text = _content_text(value.get(key))
            if text:
                return text
    return ""


def _parse_markdown_turns(raw_text: str, *, source: str) -> list[DogfoodTurn]:
    turns: list[DogfoodTurn] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_role, current_lines
        content = "\n".join(current_lines).strip()
        if current_role and content:
            turns.append(DogfoodTurn(role=current_role, content=content, source=source))
        current_role = None
        current_lines = []

    for line in raw_text.splitlines():
        match = re.match(r"^\s*(?:#{1,4}\s*)?(user|assistant|system|tool)\s*:\s*(.*)$", line, re.I)
        if not match:
            current_lines.append(line)
            continue
        flush()
        current_role = match.group(1).lower()
        remainder = match.group(2).strip()
        current_lines = [remainder] if remainder else []
    flush()
    return turns


def _mode_decision(
    mode: str,
    *,
    need_should_recall: bool,
    skip_reason: str,
    cfg: ActivationConfig,
    analyzer_duration_ms: float,
) -> dict[str, Any]:
    if mode == "off":
        return _decision(mode, "skipped", "none", "disabled", 0.0)
    if mode == "startup":
        return _decision(mode, "triggered", "cached", "startup_packet", 0.0)
    if mode == "cached":
        decision = "triggered" if need_should_recall else "skipped"
        reason = "cache_candidate" if need_should_recall else skip_reason
        return _decision(mode, decision, "cached" if need_should_recall else "none", reason, 1.0)
    if not need_should_recall:
        return _decision(mode, "skipped", "none", skip_reason, analyzer_duration_ms)

    profile = "auto_lite" if mode in {"gated_lite", "gated_medium"} else "auto_deep"
    budget = recall_budget_for_profile(
        cfg,
        profile,
        surface="dogfood",
        mode=mode.replace("gated_", ""),
    )
    return _decision(
        mode,
        "triggered",
        mode.replace("gated_", ""),
        "memory_need",
        min(float(budget.budget_ms), analyzer_duration_ms + float(budget.max_search_ms)),
    )


def _decision(
    mode: str,
    decision: str,
    mode_executed: str,
    reason: str,
    estimated_latency_ms: float,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "decision": decision,
        "mode_executed": mode_executed,
        "reason": reason,
        "estimated_latency_ms": round(estimated_latency_ms, 4),
    }


def _skip_reason(reasons: list[str]) -> str:
    if "acknowledgement" in reasons:
        return "skipped_ack"
    if "empty_turn" in reasons:
        return "skipped_low_signal"
    if reasons:
        return "skipped_low_signal"
    return "skipped_low_signal"


def _empty_mode_summary(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "turn_count": 0,
        "triggered_count": 0,
        "skipped_count": 0,
        "estimated_latency_ms": 0.0,
    }


def _trace_duration_ms(record: dict[str, Any]) -> float | None:
    value = record.get("durationMs", record.get("duration_ms"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_timeout_seconds(record: dict[str, Any]) -> float | None:
    value = record.get("timeoutSeconds", record.get("timeout_seconds"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trace_timed_out(record: dict[str, Any]) -> bool:
    status = _optional_str(record.get("status"))
    if status == "timeout":
        return True
    duration = _trace_duration_ms(record)
    timeout = _trace_timeout_seconds(record)
    return bool(duration is not None and timeout is not None and duration >= timeout * 1000)


def _trace_cache_hit(record: dict[str, Any]) -> bool:
    value = record.get("cacheHit", record.get("cache_hit"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "hit"}
    return False


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value).strip() if value is not None else ""
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _duration_summary(durations: list[float]) -> dict[str, Any]:
    if not durations:
        return {
            "count": 0,
            "avg": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(durations)
    p95_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return {
        "count": len(ordered),
        "avg": round(sum(ordered) / len(ordered), 4),
        "p95": round(ordered[p95_index], 4),
        "max": round(ordered[-1], 4),
    }


def _format_counts(counts: Any) -> str:
    if not isinstance(counts, dict) or not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay Engram dogfood transcripts")
    configure_dogfood_parser(parser)
    return asyncio.run(run_dogfood_command(parser.parse_args(argv)))
