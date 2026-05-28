from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from engram.evaluation import cli as evaluation_cli
from engram.evaluation.brain_loop_report import EVALUATION_SIGNAL_ORDER
from engram.evaluation.cli import configure_evaluate_parser, run_evaluate_command
from engram.evaluation.human_label_evidence import (
    build_human_label_evidence,
    build_human_label_evidence_template,
    human_label_evidence_failure_message,
    load_human_label_evidence,
    render_human_label_evidence_template_markdown,
)


def test_human_label_evidence_accepts_real_harness_artifact() -> None:
    evidence = build_human_label_evidence(
        _human_label_artifact(recall_count=2, session_count=1),
        artifact_path=Path("human-labels.json"),
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "measured"
    assert evidence["artifact_path"] == "human-labels.json"
    assert evidence["source"] == "staging_harness"
    assert evidence["client"] == "Cursor"
    assert evidence["captured_at"] == "2026-05-18T23:00:00Z"
    assert evidence["labeler"] == "operator"
    assert evidence["human_labeled"] is True
    assert evidence["recall_sample_count"] == 2
    assert evidence["session_sample_count"] == 1
    assert evidence["sample_sources"] == ["staging_harness"]
    assert human_label_evidence_failure_message(evidence, prefix="Human") is None


def test_load_human_label_evidence_records_artifact_hash(tmp_path: Path) -> None:
    artifact_path = tmp_path / "human-labels.json"
    artifact_path.write_text(
        json.dumps(_human_label_artifact(recall_count=2, session_count=1)),
        encoding="utf-8",
    )

    evidence = load_human_label_evidence(
        artifact_path,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "measured"
    assert evidence["artifact_path"] == str(artifact_path)
    assert evidence["artifact_sha256"] == hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()


def test_human_label_evidence_rejects_synthetic_or_unreviewed_artifact() -> None:
    artifact = _human_label_artifact(recall_count=1, session_count=0)
    artifact.update(
        {
            "humanLabeled": False,
            "source": "deterministic_smoke",
            "labeler": "",
        }
    )
    artifact["recallSamples"][0]["source"] = "deterministic_smoke"

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert evidence["failures"] == [
        "missing_human_labeled_flag",
        "synthetic_human_label_source(deterministic_smoke)",
        "synthetic_sample_sources(deterministic_smoke)",
        "missing_human_labeler",
        "insufficient_human_recall_samples(1<2)",
        "insufficient_human_session_samples(0<1)",
    ]
    assert "missing_human_labeled_flag" in (
        human_label_evidence_failure_message(evidence, prefix="Human") or ""
    )


def test_human_label_evidence_requires_sample_source_traceability() -> None:
    artifact = _human_label_artifact(recall_count=2, session_count=1)
    artifact["recallSamples"][0].pop("source")
    artifact["recallSamples"][1]["source"] = "other_staging_harness"

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert "missing_sample_sources(1)" in evidence["failures"]
    assert "sample_source_mismatch(other_staging_harness!=staging_harness)" in evidence[
        "failures"
    ]


def test_human_label_evidence_requires_reviewable_sample_text() -> None:
    artifact = _human_label_artifact(recall_count=2, session_count=1)
    artifact["recallSamples"][0].pop("query")
    artifact["recallSamples"][1]["notes"] = (
        "dogfood_transcript=abc; review_notes=<why memory helped>"
    )
    artifact["sessionSamples"][0].pop("scenario")

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert "missing_recall_sample_queries(1)" in evidence["failures"]
    assert "missing_recall_sample_notes(1)" in evidence["failures"]
    assert "missing_session_sample_scenarios(1)" in evidence["failures"]


def test_human_label_evidence_requires_sample_label_values() -> None:
    artifact = _human_label_artifact(recall_count=1, session_count=1)
    artifact["recallSamples"][0].pop("recallTriggered")
    artifact["recallSamples"][0].pop("packetsUsed")
    artifact["sessionSamples"][0].pop("temporalCorrect")

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=1,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert "missing_recall_sample_recall_triggered(1)" in evidence["failures"]
    assert "missing_recall_sample_packets_used(1)" in evidence["failures"]
    assert "missing_session_sample_temporal_correct(1)" in evidence["failures"]


def test_human_label_evidence_requires_parseable_capture_timestamp() -> None:
    artifact = _human_label_artifact(recall_count=2, session_count=1)
    artifact["capturedAt"] = "after review"

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert "invalid_harness_captured_at(after review)" in evidence["failures"]


def test_human_label_evidence_requires_expected_artifact_kind() -> None:
    artifact = _human_label_artifact(recall_count=2, session_count=1)
    artifact["kind"] = "generic_label_export"

    evidence = build_human_label_evidence(
        artifact,
        min_recall_samples=2,
        min_session_samples=1,
    )

    assert evidence["status"] == "failed"
    assert evidence["kind"] == "generic_label_export"
    expected_failure = (
        "invalid_human_label_kind(generic_label_export!=engram_human_label_evidence)"
    )
    assert expected_failure in evidence["failures"]


def test_human_label_template_is_not_valid_evidence_until_filled() -> None:
    template = build_human_label_evidence_template()

    evidence = build_human_label_evidence(
        template,
        min_recall_samples=2,
        min_session_samples=1,
    )
    markdown = render_human_label_evidence_template_markdown(template)

    assert evidence["status"] == "failed"
    assert "placeholder_human_label_source" in evidence["failures"]
    assert "placeholder_sample_sources(<same real harness source>)" in evidence["failures"]
    assert "placeholder_harness_client" in evidence["failures"]
    assert "placeholder_harness_captured_at" in evidence["failures"]
    assert "placeholder_human_labeler" in evidence["failures"]
    assert "## Validation" in markdown
    assert "--require-release-evidence" in markdown
    assert '"kind": "engram_human_label_evidence"' in markdown


@pytest.mark.asyncio
async def test_evaluate_cli_attaches_and_gates_human_label_artifact(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    human_path = tmp_path / "human-labels.json"
    bundle_path = tmp_path / "brain-loop-release-evidence.json"
    git_metadata = {
        "available": True,
        "root": "/repo",
        "commit": "abc123",
        "branch": "main",
        "dirty": False,
        "status_short": [],
    }
    monkeypatch.setattr("engram.evaluation.cli._git_metadata", lambda: git_metadata)
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    human_path.write_text(
        json.dumps(_human_label_artifact(recall_count=2, session_count=1)),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--human-label-artifact",
            str(human_path),
            "--require-human-label-evidence",
            "--min-human-recall-samples",
            "2",
            "--min-human-session-samples",
            "1",
            "--evidence-bundle",
            str(bundle_path),
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    expected_hash = hashlib.sha256(human_path.read_bytes()).hexdigest()
    assert report["human_label_evidence"]["status"] == "measured"
    assert report["human_label_evidence"]["client"] == "Cursor"
    assert report["human_label_evidence"]["artifact_sha256"] == expected_hash
    assert report["human_label_evidence"]["recall_sample_count"] == 2
    assert report["human_label_evidence"]["min_session_samples"] == 1
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["sources"]["human_label_artifact"] == str(human_path)
    assert bundle["status"] == "passed"
    assert bundle["gate_profile"] == "evidence"
    assert bundle["release_ready"] is False
    assert bundle["provenance"] == {
        "engram_version": "0.1.0",
        "git": git_metadata,
    }
    assert bundle["source_sha256"]["human_label_artifact"] == expected_hash
    assert bundle["source_sha256"]["report_json"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    assert bundle["gates"]["require_human_label_evidence"] is True
    assert bundle["gates"]["min_human_recall_samples"] == 2
    assert bundle["gates"]["min_human_session_samples"] == 1
    assert bundle["report"]["human_label_evidence"] == report["human_label_evidence"]


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_missing_human_label_artifact_when_required(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-human-label-evidence",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Human label evidence failed gates: ['missing_human_label_evidence']"
    )


@pytest.mark.asyncio
async def test_evaluate_cli_outputs_human_label_template(capsys) -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(["--human-label-template", "--format", "markdown"])

    await run_evaluate_command(args)

    output = capsys.readouterr().out
    assert "# Engram Human Label Evidence Template" in output
    assert "--human-label-artifact human-labels.json" in output
    assert '"recallSamples"' in output


@pytest.mark.asyncio
async def test_evaluate_cli_outputs_and_gates_memory_value(
    capsys,
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    report = _measured_report()
    report["memory_value"] = _measured_memory_value()
    report_path.write_text(json.dumps(report), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--memory-value",
            "--require-memory-value",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    output = json.loads(capsys.readouterr().out)
    assert output["memory_value"]["status"] == "measured"
    assert output["memory_value"]["cost"]["operation_count"] == 3


@pytest.mark.asyncio
async def test_evaluate_cli_loads_server_report_for_memory_value(
    capsys,
    monkeypatch,
) -> None:
    report = _measured_report()
    report["memory_value"] = _measured_memory_value()
    calls: list[dict] = []

    class FakeClient:
        def __init__(self, *, server_url: str, timeout_seconds: float) -> None:
            calls.append(
                {
                    "server_url": server_url,
                    "timeout_seconds": timeout_seconds,
                }
            )

        def evaluation_report(
            self,
            *,
            live_cost: bool,
            cycle_limit: int,
            sample_limit: int,
        ) -> dict:
            calls.append(
                {
                    "live_cost": live_cost,
                    "cycle_limit": cycle_limit,
                    "sample_limit": sample_limit,
                }
            )
            return dict(report)

    monkeypatch.setattr(evaluation_cli, "AxiRestClient", FakeClient)
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--server-url",
            "http://127.0.0.1:8100",
            "--server-timeout",
            "3.5",
            "--live-cost",
            "--cycles",
            "7",
            "--saved-sample-limit",
            "42",
            "--memory-value",
            "--require-memory-value",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    output = json.loads(capsys.readouterr().out)
    assert output["memory_value"]["status"] == "measured"
    assert calls == [
        {
            "server_url": "http://127.0.0.1:8100",
            "timeout_seconds": 3.5,
        },
        {"live_cost": True, "cycle_limit": 7, "sample_limit": 42},
    ]


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_server_report_with_group_override() -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--server-url",
            "http://127.0.0.1:8100",
            "--group-id",
            "other_brain",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == "--server-url uses the running server tenant; omit --group-id"


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_missing_memory_value_when_required(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-memory-value",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "Memory value evidence failed gates: ['memory_value:missing']"
    )


@pytest.mark.asyncio
async def test_evaluate_live_stats_timeout_returns_degraded_fallback() -> None:
    class SlowGraphStore:
        async def get_stats(self, group_id: str) -> dict:
            await asyncio.sleep(1)
            return {"episodes": 10, "group_id": group_id}

    stats = await evaluation_cli._load_live_stats_bounded(
        SlowGraphStore(),
        "default",
        timeout_seconds=0.01,
    )

    assert stats["evaluation_degradations"] == [
        {
            "stage": "graph_stats",
            "status": "degraded",
            "skip_reason": "graph_stats_timeout",
            "timeout_ms": 10,
        }
    ]


@pytest.mark.asyncio
async def test_evaluate_cli_writes_human_label_template_artifact(
    capsys,
    tmp_path: Path,
) -> None:
    template_path = tmp_path / "human-label-template.json"
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--human-label-template",
            "--human-label-template-out",
            str(template_path),
            "--format",
            "markdown",
        ]
    )

    await run_evaluate_command(args)

    output = capsys.readouterr().out
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert "# Engram Human Label Evidence Template" in output
    assert template["kind"] == "engram_human_label_evidence"
    assert template["humanLabeled"] is True
    assert template["recallSamples"]
    assert template["sessionSamples"]


@pytest.mark.asyncio
async def test_evaluate_cli_rejects_template_out_without_template(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "brain-loop-report.json"
    template_path = tmp_path / "human-label-template.json"
    report_path.write_text(json.dumps(_measured_report()), encoding="utf-8")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--human-label-template-out",
            str(template_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    assert str(exc_info.value) == (
        "--human-label-template-out requires --human-label-template"
    )
    assert not template_path.exists()


def _human_label_artifact(*, recall_count: int, session_count: int) -> dict:
    return {
        "kind": "engram_human_label_evidence",
        "humanLabeled": True,
        "source": "staging_harness",
        "client": "Cursor",
        "capturedAt": "2026-05-18T23:00:00Z",
        "labeler": "operator",
        "recallSamples": [
            {
                "source": "staging_harness",
                "query": f"operator recall probe {_index}",
                "recallTriggered": True,
                "recallHelped": True,
                "recallNeeded": True,
                "packetsSurfaced": 2,
                "packetsUsed": 1,
                "falseRecalls": 0,
                "notes": f"recall helped on probe {_index}",
            }
            for _index in range(recall_count)
        ],
        "sessionSamples": [
            {
                "source": "staging_harness",
                "scenario": f"operator continuity task {_index}",
                "baselineScore": 0.2,
                "memoryScore": 0.8,
                "openLoopExpected": True,
                "openLoopRecovered": True,
                "temporalExpected": True,
                "temporalCorrect": True,
                "notes": f"memory preserved task context {_index}",
            }
            for _index in range(session_count)
        ],
    }


def _measured_report() -> dict:
    return {
        "group_id": "operator_brain",
        "generated_at": "2026-05-18T23:01:00Z",
        "totals": {},
        "capture": {},
        "cue": {},
        "project": {},
        "recall": {},
        "consolidate": {},
        "evaluation_signals": {
            signal: {
                "status": "measured",
                "evidence_count": 2,
                "metric": 1.0,
                "gap": None,
            }
            for signal in EVALUATION_SIGNAL_ORDER
        },
        "coverage_gaps": [],
    }


def _measured_memory_value() -> dict:
    return {
        "status": "measured",
        "cost": {
            "status": "measured",
            "operation_count": 3,
            "avg_added_latency_ms": 12.0,
            "p95_added_latency_ms": 18.0,
            "timeout_rate": 0.0,
            "budget_miss_rate": 0.0,
            "cache_hit_rate": 0.6667,
        },
        "benefit": {
            "status": "measured",
            "recall_sample_count": 2,
            "session_sample_count": 1,
            "memory_need_precision": 1.0,
            "memory_need_recall": 1.0,
            "useful_packet_rate": 0.75,
            "false_recall_rate": 0.0,
            "session_continuity_lift": 0.4,
            "open_loop_recovery_rate": 1.0,
            "temporal_correctness": 1.0,
            "stale_packet_rate": 0.0,
            "corrected_packet_rate": 0.0,
        },
    }
