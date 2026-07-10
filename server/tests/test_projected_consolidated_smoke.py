from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from engram.config import EngramConfig
from engram.evaluation.cli import (
    build_report_from_args,
    configure_evaluate_parser,
    run_evaluate_command,
)
from engram.evaluation.smoke import (
    _apply_smoke_activation_overrides,
    assert_smoke_report,
    format_smoke_report,
    run_projected_consolidated_smoke,
)
from engram.evaluation.store import SQLiteEvaluationStore, StoredRecallRuntimeMetricsSnapshot
from engram.storage.resolver import EngineMode

SERVER_ROOT = Path(__file__).resolve().parents[1]
MEMORY_VALUE_COST_GAP = "memory value needs memory operation cost samples"


def _coverage_gaps_without_memory_cost(report: dict) -> list[str]:
    return [gap for gap in report["coverage_gaps"] if gap != MEMORY_VALUE_COST_GAP]


class _FakeClosable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self.closed = closed

    async def close(self) -> None:
        self.closed.append(self.name)


def test_projected_consolidated_smoke_closes_runtime_store_triple() -> None:
    source = inspect.getsource(run_projected_consolidated_smoke)

    assert "close_if_supported(search_index)" in source
    assert "close_if_supported(activation_store)" in source
    assert "close_if_supported(graph_store)" in source
    assert "_close_if_supported" not in source


def test_projected_consolidated_smoke_uses_synchronous_capture() -> None:
    config = EngramConfig(
        activation={
            "capture_store_timeout_ms": 1000,
            "capture_cue_store_timeout_ms": 250,
        },
        _env_file=None,
    )

    _apply_smoke_activation_overrides(config)

    assert config.activation.capture_store_timeout_ms == 0
    assert config.activation.capture_cue_store_timeout_ms == 0


@pytest.mark.asyncio
async def test_projected_consolidated_smoke_produces_full_report(tmp_path) -> None:
    report = await run_projected_consolidated_smoke(tmp_path / "smoke.db")

    assert report["coverage_gaps"] == []
    assert report["capture"]["episode_count"] == 3
    assert report["cue"]["cue_count"] == 3
    assert report["cue"]["surfaced_count"] >= 1
    assert report["project"]["projected_count"] == 3
    assert report["project"]["yield"]["linked_entity_count"] > 0
    assert report["recall"]["evaluation"]["status"] == "measured"
    assert report["recall"]["continuity"]["status"] == "measured"
    assert report["recall"]["total_analyses"] >= 1
    assert report["recall"]["trigger_count"] >= 1
    assert report["recall"]["latency"]["analyzer_ms"]["p95_ms"] > 0
    assert report["recall"]["control"]["surfaced_count"] > 0
    assert report["consolidate"]["cycle_count"] == 1
    assert report["consolidate"]["latest_status"] == "completed"
    assert report["consolidate"]["calibration"]["status"] == "measured"
    assert report["smoke"]["cycle_status"] == "completed"
    assert report["smoke"]["cycle_count"] == 1
    assert report["smoke"]["cue_feedback_checks"] == 1
    assert report["smoke"]["gate_recall_checks"] == 1
    assert {signal["status"] for signal in report["evaluation_signals"].values()} == {"measured"}
    assert "cue_checks=1" in format_smoke_report(report)
    assert "gate_checks=1" in format_smoke_report(report)


@pytest.mark.asyncio
async def test_evaluate_cli_smoke_flag_produces_full_report(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--smoke",
            "--sqlite-path",
            str(tmp_path / "cli-smoke.db"),
            "--group-id",
            "operator_brain",
            "--format",
            "json",
        ]
    )

    report = await build_report_from_args(args)

    assert report["group_id"] == "operator_brain"
    assert report["coverage_gaps"] == []
    assert report["project"]["projected_count"] == 3
    assert report["cue"]["surfaced_count"] >= 1
    assert report["consolidate"]["calibration"]["status"] == "measured"
    assert report["smoke"]["cycle_status"] == "completed"
    assert report["smoke"]["cue_feedback_checks"] == 1


@pytest.mark.asyncio
async def test_evaluate_cli_smoke_uses_configured_default_group(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("ENGRAM_DEFAULT_GROUP_ID", "operator_brain")
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--smoke",
            "--sqlite-path",
            str(tmp_path / "cli-smoke-default-group.db"),
            "--format",
            "json",
        ]
    )
    calls: list[dict] = []

    async def fake_smoke_for_args(**kwargs):
        calls.append(kwargs)
        return {
            "group_id": kwargs["group_id"],
            "coverage_gaps": [],
            "project": {"projected_count": 1},
            "consolidate": {"calibration": {"status": "measured"}},
            "smoke": {"mode": kwargs["mode"].value, "cycle_status": "completed"},
        }

    monkeypatch.setattr(
        "engram.evaluation.smoke.run_projected_consolidated_smoke_for_args",
        fake_smoke_for_args,
    )

    report = await build_report_from_args(args)

    assert report["group_id"] == "operator_brain"
    assert calls[0]["group_id"] == "operator_brain"


def test_smoke_verifier_rejects_unmeasured_evaluation_signals() -> None:
    report = {
        "coverage_gaps": [],
        "project": {
            "projected_count": 1,
            "yield": {"linked_entity_count": 1},
        },
        "cue": {"surfaced_count": 1},
        "consolidate": {
            "cycle_count": 1,
            "calibration": {"status": "measured"},
        },
        "recall": {
            "evaluation": {"status": "measured"},
            "continuity": {"status": "measured"},
            "total_analyses": 1,
            "trigger_count": 1,
            "latency": {"analyzer_ms": {"p95_ms": 1.0}},
            "control": {"surfaced_count": 1},
        },
        "evaluation_signals": {
            "cue_usefulness": {
                "status": "measured",
                "evidence_count": 1,
                "metric": 1.0,
                "gap": None,
            },
            "projection_yield": {
                "status": "measured",
                "evidence_count": 1,
                "metric": 1.0,
                "gap": None,
            },
            "recall_quality": {
                "status": "measured",
                "evidence_count": 1,
                "metric": 1.0,
                "gap": None,
            },
            "false_recall": {
                "status": "needs_surfaced_packets",
                "evidence_count": 0,
                "metric": 0.0,
                "gap": "false recall needs labeled surfaced packet counts",
            },
            "triage_calibration": {
                "status": "measured",
                "evidence_count": 1,
                "metric": 0.0,
                "gap": None,
            },
            "consolidation_effect": {
                "status": "measured",
                "evidence_count": 1,
                "metric": 1.0,
                "gap": None,
            },
        },
        "smoke": {"gate_recall_checks": 1},
    }

    with pytest.raises(SystemExit, match="unmeasured evaluation signals"):
        assert_smoke_report(report)


@pytest.mark.asyncio
async def test_evaluate_cli_smoke_load_options_extend_report(tmp_path) -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--smoke",
            "--sqlite-path",
            str(tmp_path / "cli-load-smoke.db"),
            "--group-id",
            "operator_brain",
            "--smoke-load-count",
            "4",
            "--smoke-recall-rounds",
            "2",
            "--smoke-min-duration-seconds",
            "0.001",
            "--format",
            "json",
        ]
    )

    report = await build_report_from_args(args)

    assert report["group_id"] == "operator_brain"
    assert report["coverage_gaps"] == []
    assert report["capture"]["episode_count"] == 7
    assert report["cue"]["cue_count"] == 7
    assert report["cue"]["surfaced_count"] >= 1
    assert report["project"]["projected_count"] == 7
    assert report["smoke"]["load_count"] == 4
    assert report["smoke"]["cue_feedback_checks"] == 1
    assert report["smoke"]["gate_recall_checks"] == 1
    assert report["smoke"]["recall_rounds"] == 2
    assert report["smoke"]["recall_checks"] == 8
    assert report["recall"]["total_analyses"] >= 9
    assert report["recall"]["trigger_count"] >= 1
    assert report["recall"]["control"]["surfaced_count"] > 0
    assert report["smoke"]["min_duration_seconds"] == 0.001
    assert report["smoke"]["duration_recall_checks"] > 0
    assert report["smoke"]["duration_elapsed_seconds"] >= 0.001
    assert report["smoke"]["cycle_count"] == 1


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_projected_consolidated_smoke_supports_native_helix(tmp_path) -> None:
    labels_path = tmp_path / "native-smoke-labels.db"
    helix_data_dir = tmp_path / "native-helix-data"
    report = await run_projected_consolidated_smoke(
        labels_path,
        group_id="native_brain",
        mode=EngineMode.HELIX,
        helix_data_dir=helix_data_dir,
    )

    assert report["group_id"] == "native_brain"
    assert report["coverage_gaps"] == []
    assert report["capture"]["episode_count"] == 3
    assert report["cue"]["cue_count"] == 3
    assert report["cue"]["surfaced_count"] >= 1
    assert report["project"]["projected_count"] == 3
    assert report["consolidate"]["cycle_count"] == 1
    assert report["consolidate"]["calibration"]["status"] == "measured"
    assert report["recall"]["total_analyses"] >= 1
    assert report["recall"]["trigger_count"] >= 1
    assert report["recall"]["control"]["surfaced_count"] > 0
    assert report["smoke"]["cue_feedback_checks"] == 1
    assert report["smoke"]["mode"] == "helix"
    assert report["smoke"]["helix_data_dir"] is not None

    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--mode",
            "helix",
            "--sqlite-path",
            str(labels_path),
            "--helix-data-dir",
            str(helix_data_dir),
            "--group-id",
            "native_brain",
            "--format",
            "json",
        ]
    )

    live_report = await build_report_from_args(args)

    assert _coverage_gaps_without_memory_cost(live_report) == []
    assert live_report["capture"]["episode_count"] == 3
    assert live_report["project"]["yield"]["linked_entity_count"] > 0
    assert live_report["consolidate"]["cycle_count"] == 1
    assert live_report["recall"]["evaluation"]["status"] == "measured"
    assert live_report["recall"]["total_analyses"] >= 1
    assert live_report["recall"]["latency"]["analyzer_ms"]["p95_ms"] > 0
    assert live_report["recall"]["control"]["surfaced_count"] > 0


@pytest.mark.asyncio
async def test_evaluate_cli_live_report_resolves_configured_mode(monkeypatch) -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--mode",
            "helix",
            "--group-id",
            "native_brain",
            "--no-saved-samples",
            "--format",
            "json",
        ]
    )
    requested_modes: list[str] = []
    closed: list[str] = []

    class FakeGraphStore(_FakeClosable):
        async def initialize(self) -> None:
            pass

        async def get_stats(self, group_id: str | None = None) -> dict:
            assert group_id == "native_brain"
            return {
                "episodes": 1,
                "entities": 2,
                "relationships": 1,
                "cue_metrics": {
                    "cue_count": 1,
                    "cue_coverage": 1.0,
                    "cue_surfaced_count": 1,
                },
                "projection_metrics": {"state_counts": {"projected": 1}},
            }

    class FakeConsolidationStore(_FakeClosable):
        async def get_recent_cycles(self, group_id: str, limit: int = 10) -> list:
            assert group_id == "native_brain"
            assert limit == 10
            return []

        async def get_calibration_snapshots(self, cycle_id: str, group_id: str) -> list:
            return []

    async def fake_resolve_mode(mode: str) -> EngineMode:
        requested_modes.append(mode)
        return EngineMode.HELIX

    monkeypatch.setattr("engram.evaluation.cli.resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(
        "engram.evaluation.cli.create_local_runtime_stores",
        lambda mode, config: (
            FakeGraphStore("graph", closed),
            _FakeClosable("activation", closed),
            _FakeClosable("search", closed),
        ),
    )

    async def fake_consolidation_store(mode, config, graph_store):
        assert mode == EngineMode.HELIX
        return FakeConsolidationStore("consolidation", closed)

    monkeypatch.setattr(
        "engram.evaluation.cli._create_consolidation_store",
        fake_consolidation_store,
    )

    report = await build_report_from_args(args)

    assert requested_modes == ["helix"]
    assert report["group_id"] == "native_brain"
    assert report["totals"]["episodes"] == 1
    assert report["project"]["projected_count"] == 1
    assert closed == ["consolidation", "search", "activation", "graph"]


@pytest.mark.asyncio
async def test_evaluate_cli_live_report_uses_saved_recall_runtime_snapshot(
    monkeypatch,
    tmp_path,
) -> None:
    labels_path = tmp_path / "labels.db"
    store = SQLiteEvaluationStore(str(labels_path))
    await store.initialize()
    try:
        await store.save_recall_metrics_snapshot(
            StoredRecallRuntimeMetricsSnapshot(
                group_id="native_brain",
                metrics={
                    "total_analyses": 2,
                    "trigger_count": 1,
                    "analyzer_latency_ms": {"avg": 7.0, "p95": 14.0},
                    "surfaced_count": 3,
                },
                source="test",
            )
        )
    finally:
        await store.close()

    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--mode",
            "helix",
            "--sqlite-path",
            str(labels_path),
            "--group-id",
            "native_brain",
            "--format",
            "json",
        ]
    )

    class FakeGraphStore:
        async def initialize(self) -> None:
            pass

        async def close(self) -> None:
            pass

        async def get_stats(self, group_id: str | None = None) -> dict:
            assert group_id == "native_brain"
            return {
                "episodes": 1,
                "entities": 2,
                "relationships": 1,
                "cue_metrics": {
                    "cue_count": 1,
                    "cue_coverage": 1.0,
                    "cue_surfaced_count": 1,
                },
                "projection_metrics": {"state_counts": {"projected": 1}},
                "recall_metrics": {"total_analyses": 0},
            }

    class FakeConsolidationStore:
        async def close(self) -> None:
            pass

        async def get_recent_cycles(self, group_id: str, limit: int = 10) -> list:
            return []

        async def get_calibration_snapshots(self, cycle_id: str, group_id: str) -> list:
            return []

    async def fake_resolve_mode(mode: str) -> EngineMode:
        assert mode == "helix"
        return EngineMode.HELIX

    monkeypatch.setattr("engram.evaluation.cli.resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(
        "engram.evaluation.cli.create_local_runtime_stores",
        lambda mode, config: (FakeGraphStore(), object(), object()),
    )

    async def fake_consolidation_store(mode, config, graph_store):
        return FakeConsolidationStore()

    monkeypatch.setattr(
        "engram.evaluation.cli._create_consolidation_store",
        fake_consolidation_store,
    )

    report = await build_report_from_args(args)

    assert report["recall"]["total_analyses"] == 2
    assert report["recall"]["trigger_count"] == 1
    assert report["recall"]["latency"]["analyzer_ms"]["p95_ms"] == 14.0
    assert "recall gate needs runtime analyses" not in report["coverage_gaps"]


@pytest.mark.asyncio
async def test_evaluate_cli_smoke_flag_passes_helix_mode(monkeypatch, tmp_path) -> None:
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--smoke",
            "--mode",
            "helix",
            "--sqlite-path",
            str(tmp_path / "labels.db"),
            "--helix-data-dir",
            str(tmp_path / "native-data"),
            "--group-id",
            "native_brain",
            "--smoke-min-duration-seconds",
            "3600",
            "--smoke-pause-seconds",
            "1.5",
            "--format",
            "json",
        ]
    )
    calls: list[dict] = []

    async def fake_smoke_for_args(**kwargs):
        calls.append(kwargs)
        return {
            "group_id": kwargs["group_id"],
            "coverage_gaps": [],
            "project": {"projected_count": 1},
            "consolidate": {"calibration": {"status": "measured"}},
            "smoke": {"mode": kwargs["mode"].value, "cycle_status": "completed"},
        }

    monkeypatch.setattr(
        "engram.evaluation.smoke.run_projected_consolidated_smoke_for_args",
        fake_smoke_for_args,
    )

    report = await build_report_from_args(args)

    assert report["group_id"] == "native_brain"
    assert calls == [
        {
            "sqlite_path": tmp_path / "labels.db",
            "replace": False,
            "group_id": "native_brain",
            "mode": EngineMode.HELIX,
            "helix_data_dir": tmp_path / "native-data",
            "load_count": 0,
            "recall_rounds": 0,
            "min_duration_seconds": 3600.0,
            "pause_seconds": 1.5,
        }
    ]


@pytest.mark.asyncio
async def test_evaluate_cli_from_json_uses_configured_default_group(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("ENGRAM_DEFAULT_GROUP_ID", "operator_brain")
    payload_path = tmp_path / "report-source.json"
    payload_path.write_text(json.dumps({"stats": {"episodes": 0}}))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(payload_path),
            "--format",
            "json",
        ]
    )

    report = await build_report_from_args(args)

    assert report["group_id"] == "operator_brain"


@pytest.mark.asyncio
async def test_evaluate_cli_require_evaluation_signals_accepts_measured_from_json(
    capsys,
    tmp_path,
) -> None:
    payload_path = tmp_path / "measured-report-source.json"
    payload_path.write_text(json.dumps(_measured_evaluation_payload()))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(payload_path),
            "--require-evaluation-signals",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    assert _coverage_gaps_without_memory_cost(report) == []
    assert {signal["status"] for signal in report["evaluation_signals"].values()} == {"measured"}
    assert report["evaluation_signals"]["cue_usefulness"]["evidence_count"] == 2
    assert report["evaluation_signals"]["false_recall"]["metric"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_cli_require_evaluation_signals_accepts_saved_report_artifact(
    capsys,
    tmp_path,
) -> None:
    raw_payload_path = tmp_path / "measured-report-source.json"
    raw_payload_path.write_text(json.dumps(_measured_evaluation_payload()))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    raw_args = parser.parse_args(
        [
            "--from-json",
            str(raw_payload_path),
            "--format",
            "json",
        ]
    )
    report_artifact = await build_report_from_args(raw_args)
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(report_artifact))
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--format",
            "json",
        ]
    )

    await run_evaluate_command(args)

    report = json.loads(capsys.readouterr().out)
    assert report["group_id"] == "operator_brain"
    assert report["totals"]["episodes"] == 2
    assert report["evaluation_signals"]["projection_yield"]["metric"] == 1.5


@pytest.mark.asyncio
async def test_evaluate_command_accepts_saved_report_artifact(tmp_path) -> None:
    raw_payload_path = tmp_path / "measured-report-source.json"
    raw_payload_path.write_text(json.dumps(_measured_evaluation_payload()))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    raw_args = parser.parse_args(
        [
            "--from-json",
            str(raw_payload_path),
            "--format",
            "json",
        ]
    )
    report_artifact = await build_report_from_args(raw_args)
    report_path = tmp_path / "brain-loop-report.json"
    report_path.write_text(json.dumps(report_artifact))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram",
            "evaluate",
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--format",
            "json",
        ],
        cwd=SERVER_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    report = json.loads(result.stdout)
    assert result.stderr == ""
    assert report["group_id"] == "operator_brain"
    assert report["evaluation_signals"]["triage_calibration"]["status"] == "measured"


def test_evaluate_command_rejects_unmeasured_report_artifact(tmp_path) -> None:
    report_path = tmp_path / "unmeasured-report.json"
    report_path.write_text(
        json.dumps(
            {
                "group_id": "operator_brain",
                "totals": {"episodes": 0},
                "capture": {"status": "empty"},
                "cue": {"status": "attention"},
                "project": {"status": "ready"},
                "recall": {"status": "ready"},
                "consolidate": {"status": "needs_cycles"},
                "evaluation_signals": {
                    "cue_usefulness": {
                        "status": "needs_data",
                        "evidence_count": 0,
                        "metric": None,
                        "gap": "cue usefulness cannot be measured until episodes have cues",
                    }
                },
                "coverage_gaps": ["capture has no stored episodes yet"],
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram",
            "evaluate",
            "--from-json",
            str(report_path),
            "--require-evaluation-signals",
            "--format",
            "json",
        ],
        cwd=SERVER_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Brain-loop evaluation has unmeasured evaluation signals" in result.stderr
    assert "cue_usefulness:needs_data" in result.stderr
    assert "false_recall:missing" in result.stderr


def test_evaluate_command_rejects_partial_report_artifact(tmp_path) -> None:
    report_path = tmp_path / "partial-report.json"
    report_path.write_text(
        json.dumps(
            {
                "group_id": "operator_brain",
                "loop": ["capture", "cue", "project", "recall", "consolidate"],
                "totals": {"episodes": 1},
                "capture": {"status": "ready"},
                "evaluation_signals": {},
            }
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "engram",
            "evaluate",
            "--from-json",
            str(report_path),
            "--format",
            "json",
        ],
        cwd=SERVER_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "--from-json looks like a brain-loop report" in result.stderr
    assert "required report sections" in result.stderr
    assert "consolidate" in result.stderr


@pytest.mark.asyncio
async def test_evaluate_cli_from_json_rejects_partial_report_artifact(tmp_path) -> None:
    report_path = tmp_path / "partial-report.json"
    report_path.write_text(
        json.dumps(
            {
                "group_id": "operator_brain",
                "loop": ["capture", "cue", "project", "recall", "consolidate"],
                "totals": {"episodes": 1},
                "capture": {"status": "ready"},
                "evaluation_signals": {},
            }
        )
    )
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(report_path),
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await build_report_from_args(args)

    message = str(exc_info.value)
    assert "--from-json looks like a brain-loop report" in message
    assert "cue" in message
    assert "consolidate" in message


@pytest.mark.asyncio
async def test_evaluate_cli_require_evaluation_signals_rejects_gaps(tmp_path) -> None:
    payload_path = tmp_path / "empty-report-source.json"
    payload_path.write_text(json.dumps({"stats": {"episodes": 0}}))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(payload_path),
            "--require-evaluation-signals",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    message = str(exc_info.value)
    assert "Brain-loop evaluation has unmeasured evaluation signals" in message
    assert "cue_usefulness:needs_data" in message
    assert "triage_calibration:needs_snapshots" in message


@pytest.mark.asyncio
async def test_evaluate_cli_require_evaluation_signals_rejects_low_evidence_count(
    tmp_path,
) -> None:
    payload_path = tmp_path / "measured-report-source.json"
    payload_path.write_text(json.dumps(_measured_evaluation_payload()))
    parser = argparse.ArgumentParser()
    configure_evaluate_parser(parser)
    args = parser.parse_args(
        [
            "--from-json",
            str(payload_path),
            "--require-evaluation-signals",
            "--min-evaluation-signal-evidence",
            "2",
            "--format",
            "json",
        ]
    )

    with pytest.raises(SystemExit) as exc_info:
        await run_evaluate_command(args)

    message = str(exc_info.value)
    assert "Brain-loop evaluation has unmeasured evaluation signals" in message
    assert "recall_quality:insufficient_evidence(1<2)" in message
    assert "false_recall:insufficient_evidence(1<2)" in message
    assert "consolidation_effect:insufficient_evidence(1<2)" in message


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
