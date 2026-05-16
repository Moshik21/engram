from __future__ import annotations

import argparse
import importlib.util
import json

import pytest

from engram.evaluation.cli import build_report_from_args, configure_evaluate_parser
from engram.evaluation.smoke import (
    assert_smoke_report,
    format_smoke_report,
    run_projected_consolidated_smoke,
)
from engram.evaluation.store import SQLiteEvaluationStore, StoredRecallRuntimeMetricsSnapshot
from engram.storage.resolver import EngineMode


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
    assert {
        signal["status"]
        for signal in report["evaluation_signals"].values()
    } == {"measured"}
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
    assert report["recall"]["control"]["surfaced_count"] >= 9
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

    assert live_report["coverage_gaps"] == []
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
            }

    class FakeConsolidationStore:
        async def close(self) -> None:
            pass

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
        "engram.evaluation.cli._create_graph_store",
        lambda mode, config: FakeGraphStore(),
    )

    async def fake_consolidation_store(mode, config, graph_store):
        assert mode == EngineMode.HELIX
        return FakeConsolidationStore()

    monkeypatch.setattr(
        "engram.evaluation.cli._create_consolidation_store",
        fake_consolidation_store,
    )

    report = await build_report_from_args(args)

    assert requested_modes == ["helix"]
    assert report["group_id"] == "native_brain"
    assert report["totals"]["episodes"] == 1
    assert report["project"]["projected_count"] == 1


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
        "engram.evaluation.cli._create_graph_store",
        lambda mode, config: FakeGraphStore(),
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
