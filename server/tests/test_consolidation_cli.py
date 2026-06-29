from __future__ import annotations

import json
from argparse import Namespace

import pytest

from engram.consolidation import cli
from engram.models.consolidation import ConsolidationCycle, PhaseResult
from engram.storage.resolver import EngineMode


class _FakeClosable:
    def __init__(self, name: str, closed: list[str]) -> None:
        self.name = name
        self._closed = closed

    async def close(self) -> None:
        self._closed.append(self.name)


def test_print_cycle_result_reports_completed_cycle(capsys):
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="completed",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="success",
                items_processed=3,
                items_affected=2,
            )
        ],
    )
    cycle.total_duration_ms = 12.5

    cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "completed"
    assert payload["error"] is None
    assert payload["phases"][0]["error"] is None
    assert "[DRY RUN] Consolidation complete: 3 items processed, 2 affected" in captured.out
    assert captured.err == ""


def test_print_cycle_result_warns_for_completed_cycle_with_phase_error(capsys):
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="completed",
        phase_results=[
            PhaseResult(
                phase="graph_embed",
                status="error",
                items_processed=1,
                items_affected=0,
                error="optional vector index unavailable",
            )
        ],
    )
    cycle.total_duration_ms = 8.0

    cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "completed"
    assert payload["phases"][0]["error"] == "optional vector index unavailable"
    assert (
        "[DRY RUN] Consolidation completed with warnings: "
        "1 items processed, 0 affected"
    ) in captured.out
    assert (
        "Consolidation warning: graph_embed: optional vector index unavailable"
        in captured.err
    )


def test_print_cycle_result_exits_nonzero_for_failed_cycle(capsys):
    cycle = ConsolidationCycle(
        group_id="test",
        dry_run=True,
        status="failed",
        error="Phase 'triage' requires graph_store methods: missing_method",
        phase_results=[
            PhaseResult(
                phase="triage",
                status="error",
                error="missing_method",
            )
        ],
    )
    cycle.total_duration_ms = 4.2

    with pytest.raises(SystemExit) as exc_info:
        cli._print_cycle_result(cycle, profile="observe", graph_stats={"episodes": 3})

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("\n\n[DRY RUN]")[0])
    assert payload["status"] == "failed"
    assert payload["error"] == "Phase 'triage' requires graph_store methods: missing_method"
    assert payload["phases"][0]["error"] == "missing_method"
    assert "[DRY RUN] Consolidation failed: 0 items processed, 0 affected" in captured.out
    assert "Consolidation complete" not in captured.out
    assert (
        "Consolidation failed: Phase 'triage' requires graph_store methods: missing_method"
        in captured.err
    )


@pytest.mark.asyncio
async def test_consolidation_cli_closes_runtime_stores_after_phase_validation_error(
    monkeypatch,
) -> None:
    closed: list[str] = []
    created_modes: list[EngineMode] = []
    expected_extractor = object()

    class FakeGraphStore(_FakeClosable):
        async def initialize(self) -> None:
            pass

        async def get_stats(self, group_id: str | None = None) -> dict:
            assert group_id == "native_brain"
            return {"episodes": 1}

    class FakeGraphManager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeEngine:
        def __init__(
            self,
            graph_store,
            activation_store,
            search_index,
            *,
            cfg,
            consolidation_store,
            extractor,
            graph_manager=None,
        ) -> None:
            assert graph_store.name == "graph"
            assert activation_store.name == "activation"
            assert search_index.name == "search"
            assert consolidation_store.name == "consolidation"
            assert extractor is expected_extractor
            assert graph_manager is not None

        async def run_cycle(self, **kwargs):
            assert kwargs["group_id"] == "native_brain"
            assert kwargs["phase_names"] == {"missing_phase"}
            raise ValueError("unknown phase: missing_phase")

    async def fake_resolve_mode(mode: str) -> EngineMode:
        return EngineMode.HELIX

    def fake_create_stores(mode: EngineMode, config):
        created_modes.append(mode)
        return (
            FakeGraphStore("graph", closed),
            _FakeClosable("activation", closed),
            _FakeClosable("search", closed),
        )

    async def fake_initialize_search_index_for_graph(search_index, *, graph_store, mode):
        assert search_index.name == "search"
        assert graph_store.name == "graph"
        assert mode == EngineMode.HELIX

    async def fake_create_consolidation_store_for_graph(
        config,
        *,
        graph_store,
        mode,
        sqlite_path=None,
    ):
        assert graph_store.name == "graph"
        assert mode == EngineMode.HELIX
        assert sqlite_path is None
        return _FakeClosable("consolidation", closed)

    monkeypatch.setattr(cli, "resolve_mode", fake_resolve_mode)
    monkeypatch.setattr(cli, "create_stores", fake_create_stores)
    monkeypatch.setattr(
        cli,
        "initialize_search_index_for_graph",
        fake_initialize_search_index_for_graph,
    )
    monkeypatch.setattr(
        cli,
        "create_consolidation_store_for_graph",
        fake_create_consolidation_store_for_graph,
    )
    monkeypatch.setattr(cli, "create_extractor", lambda config: expected_extractor)
    monkeypatch.setattr(cli, "GraphManager", FakeGraphManager)
    monkeypatch.setattr(cli, "ConsolidationEngine", FakeEngine)

    args = Namespace(
        profile="observe",
        group_id="native_brain",
        dry_run=True,
        scan_edges=None,
        scan_entities=None,
        phases=["missing_phase"],
    )

    with pytest.raises(SystemExit) as exc_info:
        await cli.run(args)

    assert exc_info.value.code == 2
    assert created_modes == [EngineMode.HELIX]
    assert closed == ["consolidation", "search", "activation", "graph"]
