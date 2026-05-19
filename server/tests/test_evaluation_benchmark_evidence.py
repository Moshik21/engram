from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from engram.benchmark.showcase.runner import run_showcase_benchmark
from engram.evaluation.benchmark_evidence import (
    benchmark_evidence_failure_message,
    build_benchmark_evidence,
    load_benchmark_evidence,
)
from engram.evaluation.cli import configure_evaluate_parser, run_evaluate_command


def test_benchmark_evidence_accepts_showcase_artifact() -> None:
    evidence = build_benchmark_evidence(
        _showcase_artifact(pass_rate=0.75, scenario_count=4),
        artifact_path=Path("results.json"),
        min_scenarios=4,
        min_pass_rate=0.7,
    )

    assert evidence["status"] == "measured"
    assert evidence["artifact_path"] == "results.json"
    assert evidence["benchmark"] == "showcase"
    assert evidence["baseline"] == "engram_full"
    assert evidence["scenario_count"] == 4
    assert evidence["scenario_pass_rate"] == 0.75
    assert evidence["fairness"] == {
        "strict": True,
        "transcript_hash_count": 4,
        "baseline_contract_present": True,
    }
    assert benchmark_evidence_failure_message(evidence, prefix="Benchmark") is None


def test_benchmark_evidence_reports_gate_failures() -> None:
    evidence = build_benchmark_evidence(
        _showcase_artifact(pass_rate=0.25, scenario_count=2),
        min_scenarios=3,
        min_pass_rate=0.8,
    )

    assert evidence["status"] == "failed"
    assert evidence["failures"] == [
        "insufficient_benchmark_scenarios(2<3)",
        "benchmark_pass_rate_below_threshold(0.250<0.800)",
    ]
    assert benchmark_evidence_failure_message(evidence, prefix="Benchmark") == (
        "Benchmark: ['insufficient_benchmark_scenarios(2<3)', "
        "'benchmark_pass_rate_below_threshold(0.250<0.800)']"
    )


@pytest.mark.asyncio
async def test_benchmark_evidence_accepts_real_showcase_results_artifact(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "showcase"
    await run_showcase_benchmark(
        mode="full",
        seeds=[7],
        output_dir=output_dir,
        scenario_ids=["temporal_override"],
        baseline_names=["engram_full"],
        include_ablations=False,
    )

    evidence = load_benchmark_evidence(
        output_dir / "results.json",
        min_scenarios=1,
        min_pass_rate=1.0,
    )

    assert evidence["status"] == "measured"
    assert evidence["benchmark"] == "showcase"
    assert evidence["scenario_count"] == 1
    assert evidence["scenario_pass_rate"] == 1.0
    assert evidence["fairness"]["baseline_contract_present"] is True
    assert evidence["fairness"]["transcript_hash_count"] == 1


@pytest.mark.asyncio
async def test_evaluate_cli_attaches_and_gates_benchmark_artifact(
    capsys,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "measured-report-source.json"
    benchmark_path = tmp_path / "showcase-results.json"
    report_path.write_text(json.dumps(_measured_evaluation_payload()), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps(_showcase_artifact(pass_rate=1.0, scenario_count=3)),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--benchmark-artifact",
            str(benchmark_path),
            "--require-benchmark-evidence",
            "--min-benchmark-scenarios",
            "3",
            "--min-benchmark-pass-rate",
            "0.9",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    assert report["benchmark_evidence"]["status"] == "measured"
    assert report["benchmark_evidence"]["scenario_count"] == 3
    assert report["benchmark_evidence"]["min_pass_rate"] == 0.9


@pytest.mark.asyncio
async def test_evaluate_cli_writes_evidence_bundle_after_gates_pass(
    capsys,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "measured-report-source.json"
    benchmark_path = tmp_path / "showcase-results.json"
    bundle_path = tmp_path / "evidence" / "brain-loop-evidence.json"
    report_path.write_text(json.dumps(_measured_evaluation_payload()), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps(_showcase_artifact(pass_rate=1.0, scenario_count=3)),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--min-evaluation-signal-evidence",
            "1",
            "--benchmark-artifact",
            str(benchmark_path),
            "--require-benchmark-evidence",
            "--min-benchmark-scenarios",
            "3",
            "--min-benchmark-pass-rate",
            "0.9",
            "--evidence-bundle",
            str(bundle_path),
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["kind"] == "engram_brain_loop_evidence_bundle"
    assert bundle["status"] == "passed"
    assert bundle["group_id"] == "operator_brain"
    assert bundle["sources"]["report_json"] == str(report_path)
    assert bundle["sources"]["benchmark_artifact"] == str(benchmark_path)
    assert bundle["sources"]["human_label_artifact"] is None
    assert bundle["sources"]["adoption_report"] is None
    assert bundle["sources"]["additional_adoption_reports"] == []
    assert bundle["source_sha256"]["report_json"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert bundle["source_sha256"]["benchmark_artifact"] == hashlib.sha256(
        benchmark_path.read_bytes()
    ).hexdigest()
    assert bundle["source_sha256"]["human_label_artifact"] is None
    assert bundle["source_sha256"]["adoption_report"] is None
    assert bundle["source_sha256"]["additional_adoption_reports"] == []
    assert bundle["gates"] == {
        "require_evaluation_signals": True,
        "require_release_evidence": False,
        "min_evaluation_signal_evidence": 1,
        "require_benchmark_evidence": True,
        "benchmark_baseline": "engram_full",
        "min_benchmark_scenarios": 3,
        "min_benchmark_pass_rate": 0.9,
        "require_human_label_evidence": False,
        "require_adoption_evidence": False,
        "require_adoption_client": None,
        "require_adoption_clients": [],
        "min_human_recall_samples": 1,
        "min_human_session_samples": 1,
    }
    assert bundle["report"]["benchmark_evidence"] == report["benchmark_evidence"]


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_missing_benchmark_artifact_when_required(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "measured-report-source.json"
    report_path.write_text(json.dumps(_measured_evaluation_payload()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-benchmark-evidence",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == "Benchmark evidence failed gates: ['missing_benchmark_evidence']"


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_benchmark_artifact_below_threshold(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "measured-report-source.json"
    benchmark_path = tmp_path / "showcase-results.json"
    report_path.write_text(json.dumps(_measured_evaluation_payload()), encoding="utf-8")
    benchmark_path.write_text(
        json.dumps(_showcase_artifact(pass_rate=0.5, scenario_count=2)),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--benchmark-artifact",
            str(benchmark_path),
            "--require-benchmark-evidence",
            "--min-benchmark-scenarios",
            "3",
            "--min-benchmark-pass-rate",
            "0.75",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    message = str(exc_info.value)
    assert "Benchmark evidence failed gates" in message
    assert "insufficient_benchmark_scenarios(2<3)" in message
    assert "benchmark_pass_rate_below_threshold(0.500<0.750)" in message


def _showcase_artifact(*, pass_rate: float, scenario_count: int) -> dict:
    passed_count = round(pass_rate * scenario_count)
    return {
        "track": "showcase",
        "mode": "quick",
        "generated_at": "2026-05-18T22:30:00+00:00",
        "seeds": [7],
        "fairness_contract": {
            "strict_fairness": True,
            "baseline_contracts": {"engram_full": {"answer_prompt_id": "showcase_answer_v2"}},
            "transcript_hashes": {
                f"scenario_{index}": f"hash_{index}"
                for index in range(scenario_count)
            },
        },
        "baseline_summaries": [
            {
                "baseline_name": "engram_full",
                "available": True,
                "scenario_pass_rate": pass_rate,
                "false_recall_rate": 0.0,
            }
        ],
        "scenario_results": [
            {
                "scenario_id": f"scenario_{index}",
                "baseline_name": "engram_full",
                "available": True,
                "passed": index < passed_count,
            }
            for index in range(scenario_count)
        ],
    }


def _measured_evaluation_payload() -> dict:
    return {
        "group_id": "operator_brain",
        "stats": {
            "episodes": 2,
            "entities": 3,
            "relationships": 1,
            "active_entities": 1,
            "cue_metrics": {
                "cue_count": 2,
                "cue_coverage": 1.0,
                "cue_surfaced_count": 2,
                "cue_selected_count": 1,
                "cue_used_count": 1,
            },
            "projection_metrics": {
                "state_counts": {"projected": 2},
                "yield": {
                    "linked_entity_count": 3,
                    "relationship_count": 1,
                    "avg_linked_entities_per_projected_episode": 1.5,
                    "avg_relationships_per_projected_episode": 0.5,
                },
            },
            "recall_metrics": {
                "total_analyses": 1,
                "trigger_count": 1,
                "surfaced_count": 1,
                "used_count": 1,
                "analyzer_latency_ms": {"avg": 4.0, "p95": 7.0},
            },
        },
        "recent_cycles": [
            {
                "id": "cyc_cli",
                "status": "completed",
                "phases": [
                    {
                        "phase": "triage",
                        "status": "success",
                        "itemsProcessed": 2,
                        "itemsAffected": 1,
                    }
                ],
            }
        ],
        "calibration_snapshots": [
            {
                "cycle_id": "cyc_cli",
                "phase": "triage",
                "totalTraces": 4,
                "labeledExamples": 2,
                "accuracy": 0.75,
                "expectedCalibrationError": 0.05,
            }
        ],
        "recall_samples": [
            {
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 1,
                "packetsUsed": 1,
                "falseRecalls": 0,
            }
        ],
        "session_samples": [
            {
                "baselineScore": 0.2,
                "memoryScore": 0.8,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": True,
            }
        ],
    }
