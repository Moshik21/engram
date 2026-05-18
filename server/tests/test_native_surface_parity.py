from __future__ import annotations

import asyncio
import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient

from engram.config import EngramConfig
from engram.evaluation.smoke import (
    _apply_smoke_activation_overrides,
    run_projected_consolidated_smoke,
)
from engram.events.bus import get_event_bus
from engram.main import _app_state, _shutdown, _startup, create_app
from engram.mcp.server import SessionState
from engram.models import Entity, Relationship
from engram.models.adjudication import AdjudicationRequest
from engram.models.atlas import AtlasRegion, AtlasSnapshot
from engram.notifications.models import MemoryNotification
from engram.storage.resolver import EngineMode

NATIVE_GROUP_ID = "native_brain"
NATIVE_LOAD_CONTENTS = [
    "Native load memory alpha records that Engram checks PyO3 Helix recall under repeated use.",
    (
        "Native load memory beta records that lifecycle summaries should keep "
        "capture cue project recall consolidate totals aligned."
    ),
    (
        "Native load memory gamma records that evaluation reports must preserve "
        "projection yield and recall labels."
    ),
    (
        "Native load memory delta records that FastAPI ASGI checks avoid local "
        "socket binding requirements."
    ),
    (
        "Native load memory epsilon records that group native_brain owns this "
        "one brain per person dataset."
    ),
]
NATIVE_REPLAY_CONTENT = (
    "Native replay queue memory zeta records that offline capture replay stays "
    "inside the active PyO3 native brain."
)
NATIVE_LOAD_RECALL_QUERIES = [
    "Engram brain loop",
    "Native load memory alpha",
    "PyO3 Helix recall repeated use",
    "projection yield and recall labels",
    "one brain per person native_brain",
    "offline capture replay active PyO3 native brain",
]
NATIVE_SOAK_BATCHES = [
    [
        "Native soak memory atlas records multi batch PyO3 writes for activation recall.",
        "Native soak memory beacon records recall stability after repeated remember calls.",
        "Native soak memory compass records lifecycle totals during native batch load.",
        "Native soak memory delta records cue projection counts under native load.",
    ],
    [
        "Native soak memory ember records project stage durability across Helix reopen.",
        "Native soak memory forge records FastAPI ASGI load without binding sockets.",
        "Native soak memory grove records native_brain group ownership during soak.",
        "Native soak memory harbor records recall query coverage after every batch.",
    ],
    [
        "Native soak memory ion records evaluation report coherence after larger load.",
        "Native soak memory jasper records consolidation cycle visibility under soak.",
        "Native soak memory keystone records one brain per person semantics at scale.",
        "Native soak memory lantern records reopened native Helix persistence.",
    ],
]
NATIVE_SOAK_RECALL_QUERIES = [
    "multi batch PyO3 writes",
    "recall stability after repeated remember calls",
    "lifecycle totals during native batch load",
    "project stage durability across Helix reopen",
    "native_brain group ownership during soak",
    "evaluation report coherence after larger load",
    "one brain per person semantics at scale",
    "reopened native Helix persistence",
]


def _native_surface_config(labels_path: Path, helix_data_dir: Path) -> EngramConfig:
    """Build a quiet native config that reopens an existing PyO3 Helix brain."""
    config = EngramConfig(
        mode="helix",
        default_group_id=NATIVE_GROUP_ID,
        sqlite={"path": str(labels_path)},
        helix={"transport": "native", "data_dir": str(helix_data_dir)},
        embedding={"provider": "noop"},
        auth={"default_group_id": NATIVE_GROUP_ID},
        _env_file=None,
    )
    _apply_smoke_activation_overrides(config)
    config.activation.consolidation_pressure_enabled = False
    config.activation.notification_surfacing_enabled = False
    config.activation.notification_temporal_enabled = False
    config.activation.auto_recall_enabled = False
    config.activation.auto_recall_on_tool_call = False
    config.activation.recall_packets_enabled = False
    config.activation.reranker_enabled = False
    return config


async def _assert_native_rest_surfaces(
    client: httpx.AsyncClient,
    *,
    min_episodes: int = 3,
    min_cues: int = 3,
    min_projected: int = 3,
) -> None:
    lifecycle_resp = await client.get("/api/lifecycle/summary")
    assert lifecycle_resp.status_code == 200
    lifecycle = lifecycle_resp.json()
    assert lifecycle["groupId"] == NATIVE_GROUP_ID
    assert lifecycle["loop"] == [
        "capture",
        "cue",
        "project",
        "recall",
        "consolidate",
    ]
    assert lifecycle["totals"]["episodes"] >= min_episodes
    assert lifecycle["totals"]["cues"] >= min_cues
    assert lifecycle["totals"]["projected"] >= min_projected
    assert lifecycle["consolidate"]["cycleCount"] == 1

    report_resp = await client.get("/api/evaluation/brain-loop/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["group_id"] == NATIVE_GROUP_ID
    assert report["coverage_gaps"] == []
    assert report["project"]["projected_count"] >= min_projected
    assert report["project"]["yield"]["linked_entity_count"] > 0
    assert report["recall"]["evaluation"]["status"] == "measured"
    assert report["recall"]["continuity"]["status"] == "measured"
    assert report["consolidate"]["cycle_count"] == 1

    recall_resp = await client.get(
        "/api/knowledge/recall",
        params={"q": "Engram brain loop", "limit": 5},
    )
    assert recall_resp.status_code == 200
    recall = recall_resp.json()
    assert recall["query"] == "Engram brain loop"
    assert recall["items"]


async def _assert_native_rest_health_surface(client: httpx.AsyncClient) -> None:
    health_resp = await client.get("/health")
    assert health_resp.status_code == 200
    health = health_resp.json()
    assert health["status"] == "healthy"
    assert health["mode"] == "helix"
    assert health["services"]["graph_store"] == "healthy"


async def _assert_native_rest_admin_benchmark_surface(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subject_id = "ent_native_admin_benchmark_subject"
    target_id = "ent_native_admin_benchmark_target"
    relationship_id = "rel_native_admin_benchmark"
    subject_name = "NativeAdminBenchmarkSubject"
    target_name = "NativeAdminBenchmarkTarget"

    class TinyNativeCorpusGenerator:
        def __init__(self, seed: int = 42) -> None:
            self.seed = seed

        def generate(self):
            return type(
                "TinyNativeCorpus",
                (),
                {
                    "entities": [
                        Entity(
                            id=subject_id,
                            name=subject_name,
                            entity_type="BenchmarkEntity",
                            summary="Native admin benchmark subject.",
                            group_id="stale_payload_group",
                        ),
                        Entity(
                            id=target_id,
                            name=target_name,
                            entity_type="BenchmarkEntity",
                            summary="Native admin benchmark target.",
                            group_id="stale_payload_group",
                        ),
                    ],
                    "relationships": [
                        Relationship(
                            id=relationship_id,
                            source_id=subject_id,
                            target_id=target_id,
                            predicate="VALIDATES",
                            weight=1.0,
                            confidence=0.99,
                            group_id="stale_payload_group",
                        )
                    ],
                    "access_events": [(subject_id, time.time()), (target_id, time.time())],
                    "ground_truth": [object()],
                },
            )()

        async def load(
            self,
            corpus,
            graph_store,
            activation_store,
            search_index,
            *,
            structure_aware: bool = False,
        ) -> float:
            assert self.seed == 7
            assert structure_aware is True
            assert {entity.group_id for entity in corpus.entities} == {NATIVE_GROUP_ID}
            assert {rel.group_id for rel in corpus.relationships} == {NATIVE_GROUP_ID}
            for entity in corpus.entities:
                await graph_store.create_entity(entity)
                await activation_store.record_access(
                    entity.id,
                    time.time(),
                    group_id=NATIVE_GROUP_ID,
                )
            for relationship in corpus.relationships:
                await graph_store.create_relationship(relationship)
            return 0.42

    monkeypatch.setattr(
        "engram.benchmark.corpus.CorpusGenerator",
        TinyNativeCorpusGenerator,
    )

    load_resp = await client.post(
        "/api/admin/load-benchmark",
        params={"seed": 7, "structure_aware": "true"},
    )
    assert load_resp.status_code == 200
    payload = load_resp.json()
    assert payload["loaded"] is True
    assert payload["seed"] == 7
    assert payload["group_id"] == NATIVE_GROUP_ID
    assert payload["entities"] == 2
    assert payload["relationships"] == 1
    assert payload["access_events"] == 2
    assert payload["queries"] == 1
    assert payload["elapsed_seconds"] == 0.42
    assert payload["structure_aware"] is True

    graph_store = _app_state["graph_store"]
    subject = await graph_store.get_entity(subject_id, NATIVE_GROUP_ID)
    assert subject is not None
    assert subject.group_id == NATIVE_GROUP_ID
    assert await graph_store.get_entity(subject_id, "stale_payload_group") is None
    relationships = await graph_store.get_relationships(
        subject_id,
        direction="outgoing",
        predicate="VALIDATES",
        active_only=True,
        group_id=NATIVE_GROUP_ID,
    )
    assert {relationship.target_id for relationship in relationships} == {target_id}

    search_resp = await client.get(
        "/api/entities/search",
        params={"q": subject_name, "limit": 5},
    )
    assert search_resp.status_code == 200
    assert any(item["id"] == subject_id for item in search_resp.json()["items"])

    facts_resp = await client.get(
        "/api/knowledge/facts",
        params={"subject": subject_name, "predicate": "VALIDATES", "limit": 5},
    )
    assert facts_resp.status_code == 200
    assert any(
        fact["subject"] == subject_name
        and fact["predicate"] == "VALIDATES"
        and fact["object"] == target_name
        for fact in facts_resp.json()["items"]
    )


async def _record_native_rest_evaluation_labels(client: httpx.AsyncClient) -> dict:
    recall_resp = await client.post(
        "/api/evaluation/recall-samples",
        json={
            "recallTriggered": True,
            "recallHelped": True,
            "recallNeeded": True,
            "packetsSurfaced": 4,
            "packetsUsed": 3,
            "falseRecalls": 0,
            "source": "native-rest-smoke",
            "query": "native REST evaluation label",
            "notes": "stored through native parity test",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert recall_resp.status_code == 201
    recall_label = recall_resp.json()
    assert recall_label["status"] == "stored"
    assert recall_label["groupId"] == NATIVE_GROUP_ID
    assert recall_label["sample"]["source"] == "native-rest-smoke"
    assert recall_label["sample"]["recallNeeded"] is True

    session_resp = await client.post(
        "/api/evaluation/session-samples",
        json={
            "baselineScore": 0.3,
            "memoryScore": 0.9,
            "openLoopExpected": True,
            "openLoopRecovered": True,
            "temporalExpected": True,
            "temporalCorrect": True,
            "source": "native-rest-smoke",
            "scenario": "native REST continuity label",
            "notes": "stored through native parity test",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert session_resp.status_code == 201
    session_label = session_resp.json()
    assert session_label["status"] == "stored"
    assert session_label["groupId"] == NATIVE_GROUP_ID
    assert session_label["sample"]["source"] == "native-rest-smoke"

    report_resp = await client.get("/api/evaluation/brain-loop/report")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert report["group_id"] == NATIVE_GROUP_ID
    assert report["recall"]["evaluation"]["sample_count"] >= 2
    assert report["recall"]["continuity"]["sample_count"] >= 2
    assert report["coverage_gaps"] == []
    return report


async def _assert_native_project_artifact_surfaces(
    client: httpx.AsyncClient,
    project_dir: Path,
) -> None:
    project_dir.mkdir()
    (project_dir / "README.md").write_text(
        "# Native Artifact Project\n"
        "Launch path: OpenClaw native artifact search stays inside native_brain.\n",
    )

    bootstrap_resp = await client.post(
        "/api/knowledge/bootstrap",
        json={
            "project_path": str(project_dir),
            "session_id": "native-bootstrap-session",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert bootstrap_resp.status_code == 200
    bootstrap = bootstrap_resp.json()
    assert bootstrap["status"] == "bootstrapped"
    assert "README.md" in bootstrap["files_observed"]

    search_resp = await client.get(
        "/api/knowledge/artifacts/search",
        params={"q": "OpenClaw native artifact", "project_path": str(project_dir)},
    )
    assert search_resp.status_code == 200
    search = search_resp.json()
    assert search["projectPath"] == str(project_dir)
    assert search["items"]
    assert search["items"][0]["path"] == "README.md"
    assert search["items"][0]["artifactClass"] == "readme"


async def _assert_native_mcp_project_artifact_surface(mcp_server, project_dir: Path) -> None:
    payload = json.loads(
        await mcp_server.search_artifacts(
            query="OpenClaw native artifact",
            project_path=str(project_dir),
            limit=5,
        )
    )
    assert payload["query"] == "OpenClaw native artifact"
    assert payload["project_path"] == str(project_dir)
    assert payload["total"] >= 1
    assert payload["items"][0]["path"] == "README.md"
    assert payload["items"][0]["artifactClass"] == "readme"


async def _assert_native_mcp_project_bootstrap_surface(mcp_server, project_dir: Path) -> None:
    payload = json.loads(await mcp_server.bootstrap_project(project_path=str(project_dir)))
    assert payload["status"] == "already_bootstrapped"
    assert payload["project_entity_id"]


def _assert_native_runtime_state(payload: dict, project_dir: Path) -> None:
    assert payload["runtime"]["mode"] == "helix"
    assert payload["artifactBootstrap"]["projectPath"] == str(project_dir)
    assert payload["artifactBootstrap"]["artifactCount"] >= 1
    assert payload["artifactBootstrap"]["freshArtifactCount"] >= 1
    assert "artifactBootstrapEnabled" in payload["features"]
    assert "artifactRecallEnabled" in payload["features"]


async def _assert_native_rest_runtime_surface(
    client: httpx.AsyncClient,
    project_dir: Path,
) -> None:
    runtime_resp = await client.get(
        "/api/knowledge/runtime",
        params={"project_path": str(project_dir)},
    )
    assert runtime_resp.status_code == 200
    _assert_native_runtime_state(runtime_resp.json(), project_dir)


async def _assert_native_rest_consolidation_surfaces(client: httpx.AsyncClient) -> None:
    status_resp = await client.get("/api/consolidation/status")
    assert status_resp.status_code == 200
    status = status_resp.json()
    assert status["is_running"] is False
    assert status["scheduler_active"] is False
    latest = status["latest_cycle"]
    assert latest["status"] == "completed"
    assert latest["dry_run"] is False
    assert latest["phases"]
    cycle_id = latest["id"]

    history_resp = await client.get("/api/consolidation/history", params={"limit": 5})
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert any(cycle["id"] == cycle_id for cycle in history["cycles"])

    detail_resp = await client.get(f"/api/consolidation/cycle/{cycle_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == cycle_id
    assert detail["status"] == "completed"
    assert detail["dry_run"] is False
    assert detail["phases"]
    assert isinstance(detail["merges"], list)
    assert isinstance(detail["inferred_edges"], list)
    assert isinstance(detail["replays"], list)
    assert isinstance(detail["calibration_snapshots"], list)


async def _assert_native_rest_consolidation_trigger_surface(client: httpx.AsyncClient) -> None:
    trigger_resp = await client.post(
        "/api/consolidation/trigger",
        params={"dry_run": True},
    )
    assert trigger_resp.status_code == 200
    payload = trigger_resp.json()
    assert payload == {
        "status": "triggered",
        "group_id": NATIVE_GROUP_ID,
        "dry_run": True,
    }

    latest: dict | None = None
    for _ in range(20):
        history_resp = await client.get("/api/consolidation/history", params={"limit": 1})
        assert history_resp.status_code == 200
        cycles = history_resp.json()["cycles"]
        if cycles and cycles[0]["dry_run"] is True:
            latest = cycles[0]
            if latest["status"] == "completed":
                break
        await asyncio.sleep(0.05)

    assert latest is not None
    assert latest["trigger"] == "manual"
    assert latest["status"] == "completed"
    assert latest["phases"]


async def _assert_native_rest_notification_surfaces(client: httpx.AsyncClient) -> None:
    notification_store = _app_state["notification_store"]
    created_at = time.time()
    native_notification = MemoryNotification(
        group_id=NATIVE_GROUP_ID,
        notification_type="schema_discovery",
        priority="high",
        title="Native notification parity",
        body="Surface this PyO3 native notification only for native_brain.",
        entity_ids=["ent_native_notification"],
        metadata={"source": "native_surface_parity"},
        created_at=created_at,
        source_cycle_id="cycle_native_notification",
    )
    wrong_group_notification = MemoryNotification(
        group_id="wrong_native_brain",
        notification_type="schema_discovery",
        priority="high",
        title="Wrong notification",
        body="This notification must not appear in native_brain.",
        entity_ids=[],
        metadata={},
        created_at=created_at + 1,
    )
    notification_store.add(wrong_group_notification)
    notification_store.add(native_notification)

    pending_resp = await client.get("/api/knowledge/notifications", params={"limit": 10})
    assert pending_resp.status_code == 200
    pending = pending_resp.json()["notifications"]
    assert any(
        item["id"] == native_notification.id
        and item["group_id"] == NATIVE_GROUP_ID
        and item["notification_type"] == "schema_discovery"
        and item["priority"] == "high"
        and item["source_cycle_id"] == "cycle_native_notification"
        for item in pending
    )
    assert all(item["id"] != wrong_group_notification.id for item in pending)

    since_resp = await client.get(
        "/api/knowledge/notifications",
        params={"since": created_at - 0.001},
    )
    assert since_resp.status_code == 200
    since_items = since_resp.json()["notifications"]
    assert any(item["id"] == native_notification.id for item in since_items)
    assert all(item["id"] != wrong_group_notification.id for item in since_items)

    dismiss_resp = await client.post(
        "/api/knowledge/notifications/dismiss",
        json={
            "ids": [
                native_notification.id,
                wrong_group_notification.id,
                "missing_native_notification",
            ]
        },
    )
    assert dismiss_resp.status_code == 200
    assert dismiss_resp.json() == {"dismissed": 1}
    assert wrong_group_notification.dismissed_at is None

    pending_after = await client.get("/api/knowledge/notifications")
    assert pending_after.status_code == 200
    assert all(
        item["id"] != native_notification.id
        for item in pending_after.json()["notifications"]
    )

    since_after = await client.get(
        "/api/knowledge/notifications",
        params={"since": created_at - 0.001},
    )
    assert since_after.status_code == 200
    dismissed_items = since_after.json()["notifications"]
    dismissed = next(item for item in dismissed_items if item["id"] == native_notification.id)
    assert dismissed["dismissed_at"] is not None


async def _assert_native_mcp_runtime_surface(mcp_server, project_dir: Path) -> None:
    payload = json.loads(await mcp_server.get_runtime_state(project_path=str(project_dir)))
    _assert_native_runtime_state(payload, project_dir)


async def _assert_native_mcp_memory_authority_surface(mcp_server, project_dir: Path) -> None:
    payload = json.loads(
        await mcp_server.claim_authority(
            project_path=str(project_dir),
            user_message="I am building Engram as cross-context AI memory.",
            file_memory_present=True,
        )
    )
    assert payload["authority"]["source_of_truth"] == "portable_cross_context_memory"
    assert "cross-project user facts" in payload["authority"]["engram_owns"]
    assert payload["runtime"]["artifactBootstrap"]["projectPath"] == str(project_dir)
    assert payload["onboarding"]["state"] in {"fresh_runtime", "ready"}
    assert payload["agent_protocol"]["file_memory_present"] is True
    assert payload["agent_protocol"]["capture"]["tool"] == "remember"
    assert payload["lifecycle"] == ["Capture", "Cue", "Project", "Recall", "Consolidate"]


async def _assert_native_mcp_consolidation_control_surface(mcp_server) -> None:
    status = json.loads(await mcp_server.get_consolidation_status())
    assert status["is_running"] is False
    assert "trigger_consolidation" in status["message"]
    assert status["latest_cycle"]["status"] == "completed"
    assert status["latest_cycle"]["phases"]

    triggered = json.loads(await mcp_server.trigger_consolidation(dry_run=True))
    assert triggered["status"] == "completed"
    assert triggered["dry_run"] is True
    assert triggered["graph_stats"]["episodes"] > 0
    assert triggered["graph_stats"]["entities"] > 0
    assert triggered["phases"]
    assert triggered["summary"]["total_processed"] >= 0
    assert triggered["summary"]["total_affected"] >= 0
    assert "Dry run cycle" in triggered["summary"]["description"]

    status_after_trigger = json.loads(await mcp_server.get_consolidation_status())
    assert status_after_trigger["latest_cycle"]["id"] == triggered["cycle_id"]
    assert status_after_trigger["latest_cycle"]["dry_run"] is True


async def _assert_native_mcp_graph_stats_resource(mcp_server) -> None:
    stats = json.loads(await mcp_server.graph_stats_resource())
    assert stats["episodes"] > 0
    assert stats["entities"] > 0
    assert stats["relationships"] >= 0
    assert stats["entity_type_distribution"]["TestMemory"] >= 2

    cue_metrics = stats["cue_metrics"]
    assert cue_metrics["cue_count"] >= 3
    assert cue_metrics["cue_coverage"] > 0
    assert cue_metrics["projected_cue_count"] >= 3

    projection_metrics = stats["projection_metrics"]
    assert projection_metrics["state_counts"]["projected"] >= 3
    assert projection_metrics["attempted_episode_count"] >= 3
    assert projection_metrics["yield"]["linked_entity_count"] > 0


async def _assert_native_mcp_evaluation_write_surface(mcp_server) -> None:
    recall_label = json.loads(
        await mcp_server.record_recall_evaluation(
            recall_triggered=True,
            recall_helped=True,
            recall_needed=True,
            packets_surfaced=5,
            packets_used=4,
            false_recalls=0,
            source="native-mcp-smoke",
            query="native MCP evaluation label",
            notes="stored through native parity test",
        )
    )
    assert recall_label["status"] == "stored"
    assert recall_label["operation"] == "record_recall_evaluation"
    assert recall_label["group_id"] == NATIVE_GROUP_ID
    assert recall_label["sample"]["source"] == "native-mcp-smoke"
    assert recall_label["sample"]["recall_needed"] is True
    assert recall_label["sample"]["packets_used"] == 4

    session_label = json.loads(
        await mcp_server.record_session_continuity_evaluation(
            baseline_score=0.4,
            memory_score=0.95,
            open_loop_expected=True,
            open_loop_recovered=True,
            temporal_expected=True,
            temporal_correct=True,
            source="native-mcp-smoke",
            scenario="native MCP continuity label",
            notes="stored through native parity test",
        )
    )
    assert session_label["status"] == "stored"
    assert session_label["operation"] == "record_session_continuity_evaluation"
    assert session_label["group_id"] == NATIVE_GROUP_ID
    assert session_label["sample"]["source"] == "native-mcp-smoke"
    assert session_label["sample"]["memory_score"] == 0.95


def _assert_native_route_payload(payload: dict) -> None:
    assert payload["questionFrame"]["mode"] == "reconcile"
    assert payload["answerContract"]["operator"] == "reconcile"
    assert payload["evidencePlan"]["requiredNextSources"] == ["artifacts"]
    assert "facts" in payload["evidencePlan"]["discouragedSources"]


async def _assert_native_rest_route_surface(
    client: httpx.AsyncClient,
    project_dir: Path,
) -> None:
    route_resp = await client.post(
        "/api/knowledge/route",
        json={
            "question": "what did we decide about launching Engram publicly?",
            "project_path": str(project_dir),
            "history": [{"role": "user", "content": "Keep this project-grounded."}],
        },
    )
    assert route_resp.status_code == 200
    _assert_native_route_payload(route_resp.json())


async def _assert_native_mcp_route_surface(mcp_server, project_dir: Path) -> None:
    payload = json.loads(
        await mcp_server.route_question(
            question="what did we decide about launching Engram publicly?",
            project_path=str(project_dir),
            history=["Keep this project-grounded."],
        )
    )
    _assert_native_route_payload(payload)


async def _assert_native_mcp_route_auto_observe_surface(
    mcp_server,
    project_dir: Path,
) -> None:
    cfg = mcp_server._activation_cfg
    assert cfg is not None
    original_auto_recall_on_tool_call = cfg.auto_recall_on_tool_call
    cfg.auto_recall_on_tool_call = True
    question = (
        "When reconciling native route questions, capture this long PyO3 "
        "tool call as an observed episode for the active brain."
    )
    try:
        payload = json.loads(
            await mcp_server.route_question(
                question=question,
                project_path=str(project_dir),
                history=["This route call should also auto-observe."],
            )
        )
    finally:
        cfg.auto_recall_on_tool_call = original_auto_recall_on_tool_call
    assert payload["questionFrame"]["mode"] in {"remember", "inspect", "reconcile"}
    assert payload["answerContract"]["operator"]
    assert isinstance(payload["evidencePlan"]["requiredNextSources"], list)

    episodes, _cursor = await _app_state["graph_store"].get_episodes_paginated(
        group_id=NATIVE_GROUP_ID,
        source="tool_piggyback",
        limit=20,
    )
    matches = [episode for episode in episodes if episode.content == question]
    assert matches
    episode = matches[0]
    assert episode.group_id == NATIVE_GROUP_ID
    assert episode.source == "tool_piggyback"
    cue = await _app_state["graph_store"].get_episode_cue(
        episode.id,
        NATIVE_GROUP_ID,
    )
    assert cue is not None


async def _assert_native_mcp_auto_recall_surface(mcp_server) -> None:
    anchor = Entity(
        id="ent_native_auto_recall_anchor",
        name="Native Auto Recall Anchor",
        entity_type="Concept",
        summary="Anchor entity for native MCP auto-recall middleware parity.",
        group_id=NATIVE_GROUP_ID,
    )
    await _app_state["graph_store"].create_entity(anchor)

    cfg = mcp_server._activation_cfg
    assert cfg is not None
    original_auto_recall_enabled = cfg.auto_recall_enabled
    original_auto_recall_on_tool_call = cfg.auto_recall_on_tool_call
    original_auto_recall_session_prime = cfg.auto_recall_session_prime
    original_auto_recall_level = cfg.auto_recall_level
    cfg.auto_recall_enabled = True
    cfg.auto_recall_on_tool_call = True
    cfg.auto_recall_session_prime = False
    cfg.auto_recall_level = "lite"

    try:
        payload = json.loads(
            await mcp_server.search_entities(
                name=(
                    "Native Auto Recall Anchor keeps MCP piggyback recall "
                    "grounded in the PyO3 brain"
                ),
                limit=3,
            )
        )
    finally:
        cfg.auto_recall_enabled = original_auto_recall_enabled
        cfg.auto_recall_on_tool_call = original_auto_recall_on_tool_call
        cfg.auto_recall_session_prime = original_auto_recall_session_prime
        cfg.auto_recall_level = original_auto_recall_level

    recalled = payload.get("recalled_context")
    assert recalled is not None
    assert recalled["source"] == "recall_lite"
    assert recalled["entities"]
    recalled_names = {item["name"] for item in recalled["entities"]}
    assert "Native Auto Recall Anchor" in recalled_names


async def _assert_native_rest_intention_surfaces(client: httpx.AsyncClient) -> None:
    trigger_text = "Native REST intention trigger"
    action_text = "Surface native REST intention action."
    create_resp = await client.post(
        "/api/knowledge/intentions",
        json={
            "trigger_text": trigger_text,
            "action_text": action_text,
            "trigger_type": "activation",
            "priority": "high",
            "context": "native REST intention context",
            "see_also": ["native parity"],
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["status"] == "created"
    assert created["triggerText"] == trigger_text
    assert created["actionText"] == action_text
    assert created["triggerType"] == "activation"
    assert created["refreshTrigger"] == "manual"
    intention_id = created["intentionId"]

    list_resp = await client.get("/api/knowledge/intentions")
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert any(
        item["id"] == intention_id
        and item["triggerText"] == trigger_text
        and item["actionText"] == action_text
        and item["enabled"] is True
        and item["priority"] == "high"
        for item in listed["intentions"]
    )

    refresh_resp = await client.post(
        "/api/knowledge/intentions",
        json={
            "trigger_text": "Native REST pinned context topic",
            "action_text": "Keep the native REST pinned context fresh.",
            "trigger_type": "refresh_context",
            "refresh_trigger": "after_consolidation",
            "context": "native REST pinned context background",
        },
    )
    assert refresh_resp.status_code == 200
    refresh_created = refresh_resp.json()
    assert refresh_created["triggerType"] == "refresh_context"
    assert refresh_created["refreshTrigger"] == "after_consolidation"
    refresh_id = refresh_created["intentionId"]
    refresh_list_resp = await client.get("/api/knowledge/intentions")
    assert refresh_list_resp.status_code == 200
    assert any(
        item["id"] == refresh_id
        and item["triggerType"] == "refresh_context"
        and item["refreshTrigger"] == "after_consolidation"
        and item["hasPinnedResult"] is False
        for item in refresh_list_resp.json()["intentions"]
    )

    dismiss_resp = await client.delete(f"/api/knowledge/intentions/{intention_id}")
    assert dismiss_resp.status_code == 200
    dismissed = dismiss_resp.json()
    assert dismissed == {
        "status": "dismissed",
        "intentionId": intention_id,
        "hard": False,
    }

    list_all_resp = await client.get(
        "/api/knowledge/intentions",
        params={"enabled_only": False},
    )
    assert list_all_resp.status_code == 200
    listed_all = list_all_resp.json()
    assert any(
        item["id"] == intention_id
        and item["triggerText"] == trigger_text
        and item["enabled"] is False
        for item in listed_all["intentions"]
    )

    hard_resp = await client.post(
        "/api/knowledge/intentions",
        json={
            "trigger_text": "Native REST hard-delete intention trigger",
            "action_text": "Remove this native REST intention completely.",
        },
    )
    assert hard_resp.status_code == 200
    hard_id = hard_resp.json()["intentionId"]
    hard_delete_resp = await client.delete(
        f"/api/knowledge/intentions/{hard_id}",
        params={"hard": True},
    )
    assert hard_delete_resp.status_code == 200
    assert hard_delete_resp.json() == {
        "status": "dismissed",
        "intentionId": hard_id,
        "hard": True,
    }
    after_hard_resp = await client.get(
        "/api/knowledge/intentions",
        params={"enabled_only": False},
    )
    assert after_hard_resp.status_code == 200
    assert all(item["id"] != hard_id for item in after_hard_resp.json()["intentions"])


async def _assert_native_mcp_intention_surfaces(mcp_server) -> None:
    trigger_text = "Native MCP intention trigger"
    action_text = "Surface native MCP intention action."
    created = json.loads(
        await mcp_server.intend(
            trigger_text=trigger_text,
            action_text=action_text,
            trigger_type="activation",
            priority="critical",
            context="native MCP intention context",
            see_also=["native parity"],
        )
    )
    assert created["status"] == "created"
    assert created["trigger_type"] == "activation"
    assert created["refresh_trigger"] == "manual"
    intention_id = created["intention_id"]

    listed = json.loads(await mcp_server.list_intentions())
    assert any(
        item["id"] == intention_id
        and item["trigger_text"] == trigger_text
        and item["action_text"] == action_text
        and item["enabled"] is True
        and item["priority"] == "critical"
        for item in listed["intentions"]
    )

    refresh_created = json.loads(
        await mcp_server.intend(
            trigger_text="Native MCP pinned context topic",
            action_text="Keep the native MCP pinned context fresh.",
            trigger_type="refresh_context",
            refresh_trigger="after_consolidation",
            context="native MCP pinned context background",
        )
    )
    assert refresh_created["trigger_type"] == "refresh_context"
    assert refresh_created["refresh_trigger"] == "after_consolidation"
    refresh_id = refresh_created["intention_id"]
    refresh_listed = json.loads(await mcp_server.list_intentions())
    assert any(
        item["id"] == refresh_id
        and item["trigger_type"] == "refresh_context"
        and item["refresh_trigger"] == "after_consolidation"
        and item.get("has_pinned_result") is None
        for item in refresh_listed["intentions"]
    )

    dismissed = json.loads(await mcp_server.dismiss_intention(intention_id=intention_id))
    assert dismissed["status"] == "dismissed"
    assert dismissed["intention_id"] == intention_id
    assert dismissed["hard"] is False

    listed_all = json.loads(await mcp_server.list_intentions(enabled_only=False))
    assert any(
        item["id"] == intention_id
        and item["trigger_text"] == trigger_text
        and item["enabled"] is False
        for item in listed_all["intentions"]
    )

    hard_created = json.loads(
        await mcp_server.intend(
            trigger_text="Native MCP hard-delete intention trigger",
            action_text="Remove this native MCP intention completely.",
        )
    )
    hard_id = hard_created["intention_id"]
    hard_dismissed = json.loads(
        await mcp_server.dismiss_intention(intention_id=hard_id, hard=True)
    )
    assert hard_dismissed["status"] == "dismissed"
    assert hard_dismissed["intention_id"] == hard_id
    assert hard_dismissed["hard"] is True
    listed_after_hard = json.loads(await mcp_server.list_intentions(enabled_only=False))
    assert all(item["id"] != hard_id for item in listed_after_hard["intentions"])


async def _create_native_adjudication_request(suffix: str) -> str:
    manager = _app_state["graph_manager"]
    graph_store = _app_state["graph_store"]
    episode_id = await manager.store_episode(
        content=(
            f"Native adjudication memory {suffix} records that ambiguous "
            "evidence can be resolved through PyO3 Helix."
        ),
        group_id=NATIVE_GROUP_ID,
        source=f"native-adjudication-{suffix}",
    )
    request = AdjudicationRequest(
        episode_id=episode_id,
        group_id=NATIVE_GROUP_ID,
        ambiguity_tags=["relationship_direction"],
        selected_text=f"Native adjudication selected text {suffix}",
        request_reason=f"native_adjudication_parity:{suffix}",
    )
    await graph_store.store_adjudication_requests(
        [request.to_dict()],
        group_id=NATIVE_GROUP_ID,
    )
    stored = await graph_store.get_adjudication_request(
        request.request_id,
        NATIVE_GROUP_ID,
    )
    assert stored is not None
    assert stored["status"] == "pending"
    assert stored["group_id"] == NATIVE_GROUP_ID
    return request.request_id


async def _assert_native_rest_adjudication_surface(client: httpx.AsyncClient) -> None:
    request_id = await _create_native_adjudication_request("rest")
    adjudicate_resp = await client.post(
        "/api/knowledge/adjudicate",
        json={
            "request_id": request_id,
            "model_tier": "sonnet",
            "rationale": "native REST adjudication parity",
        },
    )
    assert adjudicate_resp.status_code == 200
    payload = adjudicate_resp.json()
    assert payload == {
        "status": "rejected",
        "requestId": request_id,
        "committedIds": {},
        "supersededEvidenceIds": [],
        "replacementEvidenceIds": [],
    }

    resolved = await _app_state["graph_store"].get_adjudication_request(
        request_id,
        NATIVE_GROUP_ID,
    )
    assert resolved is not None
    assert resolved["status"] == "rejected"
    assert resolved["resolution_source"] == "client_adjudication"
    assert resolved["resolution_payload"]["model_tier"] == "sonnet"
    assert resolved["resolution_payload"]["rationale"] == "native REST adjudication parity"
    assert resolved["attempt_count"] == 1
    assert resolved["resolved_at"]


async def _assert_native_mcp_adjudication_surface(mcp_server) -> None:
    request_id = await _create_native_adjudication_request("mcp")
    payload = json.loads(
        await mcp_server.adjudicate_evidence(
            request_id=request_id,
            model_tier="opus",
            rationale="native MCP adjudication parity",
        )
    )
    assert payload == {
        "status": "rejected",
        "request_id": request_id,
        "committed_ids": {},
        "superseded_evidence_ids": [],
        "replacement_evidence_ids": [],
    }

    resolved = await _app_state["graph_store"].get_adjudication_request(
        request_id,
        NATIVE_GROUP_ID,
    )
    assert resolved is not None
    assert resolved["status"] == "rejected"
    assert resolved["resolution_source"] == "client_adjudication"
    assert resolved["resolution_payload"]["model_tier"] == "opus"
    assert resolved["resolution_payload"]["rationale"] == "native MCP adjudication parity"
    assert resolved["attempt_count"] == 1
    assert resolved["resolved_at"]


async def _assert_native_rest_auto_observe_surface(client: httpx.AsyncClient) -> None:
    from engram.api.knowledge import _DEDUP_CACHE

    _DEDUP_CACHE.clear()
    content = "Native REST auto-observe captures PyO3 hook content for cues."
    body = {
        "content": content,
        "source": "auto:native-test",
        "project": "Engram",
        "role": "user",
        "session_id": "native-auto-session",
        "conversation_date": "2026-05-13T14:00:00",
    }
    auto_resp = await client.post("/api/knowledge/auto-observe", json=body)
    assert auto_resp.status_code == 200
    payload = auto_resp.json()
    assert payload["status"] == "observed"
    assert payload["operation"] == "observe"
    assert payload["lifecycle"]["stage"] == "cue"
    assert payload["lifecycle"]["captureStatus"] == "stored"
    assert payload["lifecycle"]["projectionMode"] == "background"
    assert payload["lifecycle"]["projectionStatus"] == "queued"

    episode = await _app_state["graph_store"].get_episode_by_id(
        payload["episodeId"],
        NATIVE_GROUP_ID,
    )
    assert episode is not None
    assert episode.group_id == NATIVE_GROUP_ID
    assert episode.content == content
    assert episode.source == "auto:native-test"
    assert episode.session_id == "native-auto-session"
    assert episode.conversation_date is not None
    cue = await _app_state["graph_store"].get_episode_cue(
        payload["episodeId"],
        NATIVE_GROUP_ID,
    )
    assert cue is not None

    duplicate_resp = await client.post("/api/knowledge/auto-observe", json=body)
    assert duplicate_resp.status_code == 200
    duplicate = duplicate_resp.json()
    assert duplicate["status"] == "dedup_skipped"
    assert duplicate["operation"] == "observe"
    assert duplicate["lifecycle"] == {
        "stage": "capture",
        "captureStatus": "skipped",
        "projectionMode": None,
        "projectionStatus": None,
    }


async def _assert_native_rest_observe_surface(client: httpx.AsyncClient) -> None:
    content = "Native REST observe surface queues raw PyO3 capture for cues."
    observe_resp = await client.post(
        "/api/knowledge/observe",
        json={
            "content": content,
            "source": "native-rest-observe",
            "conversation_date": "2026-05-13T13:00:00",
        },
    )
    assert observe_resp.status_code == 200
    payload = observe_resp.json()
    assert payload["status"] == "observed"
    assert payload["operation"] == "observe"
    assert payload["lifecycle"]["stage"] == "cue"
    assert payload["lifecycle"]["captureStatus"] == "stored"
    assert payload["lifecycle"]["projectionMode"] == "background"
    assert payload["lifecycle"]["projectionStatus"] == "queued"

    episode = await _app_state["graph_store"].get_episode_by_id(
        payload["episodeId"],
        NATIVE_GROUP_ID,
    )
    assert episode is not None
    assert episode.group_id == NATIVE_GROUP_ID
    assert episode.content == content
    assert episode.source == "native-rest-observe"
    assert episode.conversation_date is not None
    cue = await _app_state["graph_store"].get_episode_cue(
        payload["episodeId"],
        NATIVE_GROUP_ID,
    )
    assert cue is not None


async def _assert_native_attachment_episode(
    *,
    episode_id: str,
    content: str,
    source: str,
    mime_type: str,
    data_url: str,
    description: str,
) -> None:
    episode = await _app_state["graph_store"].get_episode_by_id(
        episode_id,
        NATIVE_GROUP_ID,
    )
    assert episode is not None
    assert episode.group_id == NATIVE_GROUP_ID
    assert episode.content == content
    assert episode.source == source
    assert len(episode.attachments) == 1
    attachment = episode.attachments[0]
    assert attachment.mime_type == mime_type
    assert attachment.data_url == data_url
    assert attachment.description == description

    cue = await _app_state["graph_store"].get_episode_cue(
        episode_id,
        NATIVE_GROUP_ID,
    )
    assert cue is not None


async def _assert_native_rest_attachment_surfaces(client: httpx.AsyncClient) -> None:
    image_data = "data:image/png;base64,bmF0aXZlLWltYWdl"
    image_description = "Native REST image attachment"
    image_resp = await client.post(
        "/api/knowledge/observe-image",
        json={
            "image_data": image_data,
            "mime_type": "image/png",
            "description": image_description,
            "source": "native-rest-image",
        },
    )
    assert image_resp.status_code == 200
    image_payload = image_resp.json()
    assert image_payload["status"] == "stored"
    assert image_payload["operation"] == "observe"
    assert image_payload["episodeId"] == image_payload["episode_id"]
    assert image_payload["lifecycle"]["stage"] == "cue"
    assert image_payload["lifecycle"]["attachmentKind"] == "image"
    await _assert_native_attachment_episode(
        episode_id=image_payload["episodeId"],
        content=image_description,
        source="native-rest-image",
        mime_type="image/png",
        data_url=image_data,
        description=image_description,
    )

    file_data = "data:text/plain;base64,bmF0aXZlLWZpbGU="
    file_description = "Native REST file attachment"
    file_resp = await client.post(
        "/api/knowledge/observe-file",
        json={
            "file_data": file_data,
            "mime_type": "text/plain",
            "description": file_description,
            "source": "native-rest-file",
        },
    )
    assert file_resp.status_code == 200
    file_payload = file_resp.json()
    assert file_payload["status"] == "stored"
    assert file_payload["operation"] == "observe"
    assert file_payload["episodeId"] == file_payload["episode_id"]
    assert file_payload["lifecycle"]["stage"] == "cue"
    assert file_payload["lifecycle"]["attachmentKind"] == "file"
    await _assert_native_attachment_episode(
        episode_id=file_payload["episodeId"],
        content=file_description,
        source="native-rest-file",
        mime_type="text/plain",
        data_url=file_data,
        description=file_description,
    )


async def _assert_native_conversation_surfaces(client: httpx.AsyncClient) -> None:
    create_resp = await client.post(
        "/api/conversations/",
        json={
            "title": "Native brain thread",
            "session_date": "2026-05-13",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert create_resp.status_code == 200
    conversation_id = create_resp.json()["id"]

    append_resp = await client.post(
        f"/api/conversations/{conversation_id}/messages",
        json={
            "messages": [
                {"role": "user", "content": "Remember the native conversation surface."},
                {"role": "assistant", "content": "Stored inside native_brain."},
            ],
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert append_resp.status_code == 200
    message_ids = append_resp.json()["ids"]
    assert len(message_ids) == 2

    list_resp = await client.get("/api/conversations/", params={"limit": 10})
    assert list_resp.status_code == 200
    conversations = list_resp.json()["conversations"]
    assert any(item["id"] == conversation_id for item in conversations)

    messages_resp = await client.get(f"/api/conversations/{conversation_id}/messages")
    assert messages_resp.status_code == 200
    messages = messages_resp.json()["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Remember the native conversation surface."
    assert messages[1]["content"] == "Stored inside native_brain."

    update_resp = await client.patch(
        f"/api/conversations/{conversation_id}",
        json={"title": "Native brain thread renamed"},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["status"] == "updated"

    renamed_resp = await client.get("/api/conversations/", params={"limit": 10})
    assert renamed_resp.status_code == 200
    renamed_items = renamed_resp.json()["conversations"]
    assert any(
        item["id"] == conversation_id and item["title"] == "Native brain thread renamed"
        for item in renamed_items
    )

    delete_resp = await client.delete(f"/api/conversations/{conversation_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json()["status"] == "deleted"

    deleted_messages_resp = await client.get(f"/api/conversations/{conversation_id}/messages")
    assert deleted_messages_resp.status_code == 404

    deleted_list_resp = await client.get("/api/conversations/", params={"limit": 10})
    assert deleted_list_resp.status_code == 200
    assert all(
        item["id"] != conversation_id
        for item in deleted_list_resp.json()["conversations"]
    )

    raw_messages = await _app_state["conversation_store"]._query(
        "find_messages_by_conversation", {"conv_id": conversation_id}
    )
    assert raw_messages == []
    assert all(message_id for message_id in message_ids)


def _parse_sse_events(text: str) -> list[dict]:
    events: list[dict] = []
    for line in text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        events.append(json.loads(line.removeprefix("data: ")))
    return events


def _make_chat_response(text: str) -> MagicMock:
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


async def _assert_native_chat_surface(client: httpx.AsyncClient) -> None:
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(
        return_value=_make_chat_response("Native chat turn persisted in Helix.")
    )

    with patch("engram.retrieval.chat_runtime.anthropic.AsyncAnthropic", return_value=mock_client):
        chat_resp = await client.post(
            "/api/knowledge/chat",
            json={
                "message": "Persist this native chat turn.",
                "session_date": "2026-05-13",
                "groupId": "wrong_native_brain",
                "group_id": "wrong_native_brain",
            },
        )

    assert chat_resp.status_code == 200
    events = _parse_sse_events(chat_resp.text)
    finish_events = [event for event in events if event.get("type") == "finish"]
    assert finish_events
    assert finish_events[-1]["finishReason"] == "stop"
    conversation_id = finish_events[-1]["conversationId"]

    text_deltas = [event["delta"] for event in events if event.get("type") == "text-delta"]
    assert "".join(text_deltas) == "Native chat turn persisted in Helix."

    messages: list[dict] = []
    for _attempt in range(20):
        messages_resp = await client.get(f"/api/conversations/{conversation_id}/messages")
        assert messages_resp.status_code == 200
        messages = messages_resp.json()["messages"]
        if len(messages) >= 2:
            break
        await asyncio.sleep(0.01)

    assert [message["role"] for message in messages[:2]] == ["user", "assistant"]
    assert messages[0]["content"] == "Persist this native chat turn."
    assert messages[1]["content"] == "Native chat turn persisted in Helix."


async def _assert_native_rest_context_surface(client: httpx.AsyncClient) -> None:
    context_resp = await client.get(
        "/api/knowledge/context",
        params={"topic_hint": "Engram", "max_tokens": 500, "format": "structured"},
    )
    assert context_resp.status_code == 200
    payload = context_resp.json()
    assert payload["format"] == "structured"
    assert payload["entityCount"] > 0
    assert payload["tokenEstimate"] > 0
    assert "Engram" in payload["context"]


async def _assert_native_mcp_context_surface(mcp_server) -> None:
    payload = json.loads(
        await mcp_server.get_context(
            max_tokens=500,
            topic_hint="Engram",
            format="structured",
        )
    )
    assert payload["format"] == "structured"
    assert payload["entity_count"] > 0
    assert payload["token_estimate"] > 0
    assert "Engram" in payload["context"]


async def _assert_native_mcp_notification_surface(mcp_server) -> None:
    cfg = mcp_server._activation_cfg
    assert cfg is not None
    original_enabled = cfg.notification_surfacing_enabled
    cfg.notification_surfacing_enabled = True
    notification_store = _app_state["notification_store"]
    native_notification = MemoryNotification(
        group_id=NATIVE_GROUP_ID,
        notification_type="schema_discovery",
        priority="high",
        title="Native MCP notification parity",
        body="MCP should surface this PyO3 notification for native_brain.",
        entity_ids=["ent_native_mcp_notification"],
        metadata={"source": "native_mcp_surface_parity"},
        created_at=time.time(),
    )
    wrong_group_notification = MemoryNotification(
        group_id="wrong_native_brain",
        notification_type="schema_discovery",
        priority="high",
        title="Wrong MCP notification",
        body="MCP must not surface this notification for native_brain.",
        entity_ids=[],
        metadata={},
        created_at=time.time(),
    )
    notification_store.add(wrong_group_notification)
    notification_store.add(native_notification)
    try:
        payload = json.loads(
            await mcp_server.get_context(
                max_tokens=500,
                topic_hint="native MCP notification parity",
                format="structured",
            )
        )
    finally:
        cfg.notification_surfacing_enabled = original_enabled

    notifications = payload["memory_notifications"]
    assert any(
        item["title"] == "Native MCP notification parity"
        and item["type"] == "schema_discovery"
        and item["priority"] == "high"
        for item in notifications
    )
    assert all(item["title"] != "Wrong MCP notification" for item in notifications)
    assert native_notification.surfaced_count == 1
    assert wrong_group_notification.surfaced_count == 0


async def _assert_native_mcp_write_surfaces(mcp_server) -> None:
    remember_content = (
        "Native MCP remember surface records that PyO3 writes stay in native_brain."
    )
    remember_payload = json.loads(
        await mcp_server.remember(
            content=remember_content,
            source="native-mcp-remember",
        )
    )
    assert remember_payload["status"] == "stored"
    assert remember_payload["operation"] == "remember"
    assert remember_payload["lifecycle"]["stage"] == "project"
    assert remember_payload["lifecycle"]["capture_status"] == "stored"
    assert remember_payload["lifecycle"]["projection_mode"] == "synchronous"
    assert remember_payload["lifecycle"]["projection_status"] == "attempted"

    remember_episode = await _app_state["graph_store"].get_episode_by_id(
        remember_payload["episode_id"],
        NATIVE_GROUP_ID,
    )
    assert remember_episode is not None
    assert remember_episode.group_id == NATIVE_GROUP_ID
    assert remember_episode.content == remember_content
    assert remember_episode.source == "native-mcp-remember"

    observe_content = "Native MCP observe surface queues raw PyO3 capture for cues."
    observe_payload = json.loads(
        await mcp_server.observe(
            content=observe_content,
            source="native-mcp-observe",
        )
    )
    assert observe_payload["status"] == "stored"
    assert observe_payload["operation"] == "observe"
    assert observe_payload["lifecycle"]["stage"] == "cue"
    assert observe_payload["lifecycle"]["capture_status"] == "stored"
    assert observe_payload["lifecycle"]["projection_mode"] == "background"
    assert observe_payload["lifecycle"]["projection_status"] == "queued"

    observe_episode = await _app_state["graph_store"].get_episode_by_id(
        observe_payload["episode_id"],
        NATIVE_GROUP_ID,
    )
    assert observe_episode is not None
    assert observe_episode.group_id == NATIVE_GROUP_ID
    assert observe_episode.content == observe_content
    assert observe_episode.source == "native-mcp-observe"
    observe_cue = await _app_state["graph_store"].get_episode_cue(
        observe_payload["episode_id"],
        NATIVE_GROUP_ID,
    )
    assert observe_cue is not None

    image_data = "data:image/png;base64,bmF0aXZlLW1jcC1pbWFnZQ=="
    image_description = "Native MCP image attachment"
    image_payload = json.loads(
        await mcp_server.observe_image(
            image_data=image_data,
            mime_type="image/png",
            description=image_description,
            source="native-mcp-image",
        )
    )
    assert image_payload["status"] == "stored"
    assert image_payload["operation"] == "observe"
    assert image_payload["lifecycle"]["stage"] == "cue"
    assert image_payload["lifecycle"]["attachment_kind"] == "image"
    await _assert_native_attachment_episode(
        episode_id=image_payload["episode_id"],
        content=image_description,
        source="native-mcp-image",
        mime_type="image/png",
        data_url=image_data,
        description=image_description,
    )

    file_data = "data:text/plain;base64,bmF0aXZlLW1jcC1maWxl"
    file_description = "Native MCP file attachment"
    file_payload = json.loads(
        await mcp_server.observe_file(
            file_data=file_data,
            mime_type="text/plain",
            description=file_description,
            source="native-mcp-file",
        )
    )
    assert file_payload["status"] == "stored"
    assert file_payload["operation"] == "observe"
    assert file_payload["lifecycle"]["stage"] == "cue"
    assert file_payload["lifecycle"]["attachment_kind"] == "file"
    await _assert_native_attachment_episode(
        episode_id=file_payload["episode_id"],
        content=file_description,
        source="native-mcp-file",
        mime_type="text/plain",
        data_url=file_data,
        description=file_description,
    )


async def _create_native_test_entity(entity_id: str, name: str) -> None:
    graph_store = _app_state["graph_store"]
    activation_store = _app_state["activation_store"]
    await graph_store.create_entity(
        Entity(
            id=entity_id,
            name=name,
            entity_type="TestMemory",
            summary=f"{name} validates native mutation surface parity.",
            group_id=NATIVE_GROUP_ID,
        )
    )
    await activation_store.record_access(entity_id, time.time(), group_id=NATIVE_GROUP_ID)


async def _create_native_test_fact(suffix: str) -> tuple[str, str, str, str]:
    graph_store = _app_state["graph_store"]
    activation_store = _app_state["activation_store"]
    subject_id = f"ent_native_fact_subject_{suffix}"
    object_id = f"ent_native_fact_object_{suffix}"
    relationship_id = f"rel_native_fact_{suffix}"
    subject_name = f"NativeFactSubject{suffix.title()}"
    object_name = f"NativeFactObject{suffix.title()}"

    await graph_store.create_entity(
        Entity(
            id=subject_id,
            name=subject_name,
            entity_type="TestMemory",
            summary=f"{subject_name} validates native fact lookup parity.",
            group_id=NATIVE_GROUP_ID,
        )
    )
    await graph_store.create_entity(
        Entity(
            id=object_id,
            name=object_name,
            entity_type="TestMemory",
            summary=f"{object_name} anchors native relationship lookup parity.",
            group_id=NATIVE_GROUP_ID,
        )
    )
    await graph_store.create_relationship(
        Relationship(
            id=relationship_id,
            source_id=subject_id,
            target_id=object_id,
            predicate="USES",
            weight=1.0,
            confidence=0.95,
            group_id=NATIVE_GROUP_ID,
        )
    )
    await activation_store.record_access(subject_id, time.time(), group_id=NATIVE_GROUP_ID)
    return subject_id, object_id, subject_name, object_name


def _assert_native_fact_result(
    facts: list[dict],
    *,
    subject_name: str,
    object_name: str,
) -> None:
    assert any(
        fact["subject"] == subject_name
        and fact["predicate"] == "USES"
        and fact["object"] == object_name
        for fact in facts
    )


async def _assert_native_rest_entity_fact_lookup_surface(client: httpx.AsyncClient) -> None:
    subject_id, object_id, subject_name, object_name = await _create_native_test_fact("rest")

    entity_resp = await client.get("/api/entities/search", params={"q": subject_name})
    assert entity_resp.status_code == 200
    entity_payload = entity_resp.json()
    assert any(
        item["id"] == subject_id
        and item["name"] == subject_name
        and item["entityType"] == "TestMemory"
        for item in entity_payload["items"]
    )

    facts_resp = await client.get(
        "/api/knowledge/facts",
        params={"subject": subject_name, "predicate": "USES", "limit": 5},
    )
    assert facts_resp.status_code == 200
    _assert_native_fact_result(
        facts_resp.json()["items"],
        subject_name=subject_name,
        object_name=object_name,
    )

    detail_resp = await client.get(f"/api/entities/{subject_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["id"] == subject_id
    assert detail["name"] == subject_name
    assert detail["entityType"] == "TestMemory"
    assert detail["accessCount"] >= 1
    assert any(
        fact["predicate"] == "USES"
        and fact["direction"] == "outgoing"
        and fact["other"]["id"] == object_id
        and fact["other"]["name"] == object_name
        for fact in detail["facts"]
    )

    neighbors_resp = await client.get(f"/api/entities/{subject_id}/neighbors")
    assert neighbors_resp.status_code == 200
    neighbors = neighbors_resp.json()
    assert neighbors["centerId"] == subject_id
    assert {node["id"] for node in neighbors["nodes"]} >= {subject_id, object_id}
    assert any(
        edge["source"] == subject_id
        and edge["target"] == object_id
        and edge["predicate"] == "USES"
        for edge in neighbors["edges"]
    )


async def _assert_native_mcp_entity_fact_lookup_surface(mcp_server) -> None:
    subject_id, object_id, subject_name, object_name = await _create_native_test_fact("mcp")

    entity_payload = json.loads(await mcp_server.search_entities(name=subject_name, limit=5))
    assert any(
        item["id"] == subject_id
        and item["name"] == subject_name
        and item["entity_type"] == "TestMemory"
        for item in entity_payload["entities"]
    )

    fact_payload = json.loads(
        await mcp_server.search_facts(
            query=subject_name,
            subject=subject_name,
            predicate="USES",
            limit=5,
        )
    )
    _assert_native_fact_result(
        fact_payload["facts"],
        subject_name=subject_name,
        object_name=object_name,
    )

    profile_payload = json.loads(await mcp_server.entity_profile_resource(subject_id))
    assert profile_payload["id"] == subject_id
    assert profile_payload["name"] == subject_name
    assert profile_payload["entity_type"] == "TestMemory"
    assert profile_payload["activation"]["access_count"] >= 1
    assert any(
        fact["predicate"] == "USES" and fact["object"] == object_name
        for fact in profile_payload["facts"]
    )

    neighbors_payload = json.loads(await mcp_server.entity_neighbors_resource(subject_id))
    assert any(
        item["entity"]["id"] == object_id
        and item["relationship"]["source_id"] == subject_id
        and item["relationship"]["target_id"] == object_id
        and item["relationship"]["predicate"] == "USES"
        for item in neighbors_payload
    )

    graph_state = json.loads(
        await mcp_server.get_graph_state(
            top_n=10,
            include_edges=True,
            entity_types=["TestMemory"],
        )
    )
    assert graph_state["group_id"] == NATIVE_GROUP_ID
    assert graph_state["stats"]["entity_type_distribution"]["TestMemory"] >= 2
    assert any(item["id"] == subject_id for item in graph_state["top_activated"])
    assert any(
        edge["source"] == subject_name
        and edge["target"] == object_name
        and edge["predicate"] == "USES"
        for edge in graph_state["edges"]
    )


async def _assert_native_rest_dashboard_read_surfaces(client: httpx.AsyncClient) -> None:
    subject_id, object_id, subject_name, _object_name = await _create_native_test_fact(
        "dashboard"
    )

    stats_resp = await client.get("/api/stats", params={"days": 7})
    assert stats_resp.status_code == 200
    stats_payload = stats_resp.json()
    assert stats_payload["groupId"] == NATIVE_GROUP_ID
    assert stats_payload["stats"]["episodes"] > 0
    assert stats_payload["stats"]["entity_type_distribution"]["TestMemory"] >= 2
    assert any(item["id"] == subject_id for item in stats_payload["topActivated"])
    assert any(item["edgeCount"] > 0 for item in stats_payload["topConnected"])
    assert any(day["episodes"] > 0 for day in stats_payload["growthTimeline"])

    activation_resp = await client.get("/api/activation/snapshot", params={"limit": 20})
    assert activation_resp.status_code == 200
    activation_payload = activation_resp.json()
    assert any(
        item["entityId"] == subject_id
        and item["name"] == subject_name
        and item["entityType"] == "TestMemory"
        for item in activation_payload["topActivated"]
    )

    curve_resp = await client.get(
        f"/api/activation/{subject_id}/curve",
        params={"hours": 1, "points": 4},
    )
    assert curve_resp.status_code == 200
    curve_payload = curve_resp.json()
    assert curve_payload["entityId"] == subject_id
    assert curve_payload["entityName"] == subject_name
    assert curve_payload["points"] == 4
    assert len(curve_payload["curve"]) == 4
    assert curve_payload["accessEvents"]

    neighborhood_resp = await client.get(
        "/api/graph/neighborhood",
        params={"center": subject_id, "depth": 1, "max_nodes": 10},
    )
    assert neighborhood_resp.status_code == 200
    neighborhood = neighborhood_resp.json()
    assert neighborhood["centerId"] == subject_id
    assert neighborhood["representation"]["scope"] == "neighborhood"
    assert {node["id"] for node in neighborhood["nodes"]} >= {subject_id, object_id}
    assert any(
        edge["source"] == subject_id
        and edge["target"] == object_id
        and edge["predicate"] == "USES"
        for edge in neighborhood["edges"]
    )

    temporal_resp = await client.get(
        "/api/graph/at",
        params={
            "center": subject_id,
            "at": "2099-01-01T00:00:00",
            "depth": 1,
            "max_nodes": 10,
        },
    )
    assert temporal_resp.status_code == 200
    temporal = temporal_resp.json()
    assert temporal["centerId"] == subject_id
    assert temporal["representation"]["scope"] == "temporal"
    assert {node["id"] for node in temporal["nodes"]} >= {subject_id, object_id}
    assert any(
        edge["source"] == subject_id
        and edge["target"] == object_id
        and edge["predicate"] == "USES"
        for edge in temporal["edges"]
    )

    episodes_resp = await client.get(
        "/api/episodes",
        params={"source": "native-rest-observe", "limit": 5},
    )
    assert episodes_resp.status_code == 200
    episodes_payload = episodes_resp.json()
    assert episodes_payload["total"] >= 1
    assert any(
        item["source"] == "native-rest-observe"
        and item["content"] == "Native REST observe surface queues raw PyO3 capture for cues."
        and item["cue"] is not None
        for item in episodes_payload["items"]
    )

    queued_resp = await client.get(
        "/api/episodes",
        params={"source": "native-rest-observe", "status": "queued", "limit": 5},
    )
    assert queued_resp.status_code == 200
    queued_payload = queued_resp.json()
    assert queued_payload["total"] >= 1
    assert all(item["status"] == "queued" for item in queued_payload["items"])
    observed_items = [
        item
        for item in queued_payload["items"]
        if item["source"] == "native-rest-observe"
        and item["content"] == "Native REST observe surface queues raw PyO3 capture for cues."
    ]
    assert observed_items
    observed_item = observed_items[0]
    assert observed_item["cue"] is not None
    assert observed_item["projectionState"] == observed_item["cue"]["projectionState"]
    assert observed_item["projectionState"] in {"cued", "scheduled", "queued"}

    first_page_resp = await client.get("/api/episodes", params={"limit": 1})
    assert first_page_resp.status_code == 200
    first_page = first_page_resp.json()
    assert first_page["total"] == 1
    assert first_page["items"]
    assert first_page["nextCursor"] is not None
    first_episode_id = first_page["items"][0]["episodeId"]

    second_page_resp = await client.get(
        "/api/episodes",
        params={"limit": 1, "cursor": first_page["nextCursor"]},
    )
    assert second_page_resp.status_code == 200
    second_page = second_page_resp.json()
    assert second_page["total"] == 1
    assert second_page["items"][0]["episodeId"] != first_episode_id


async def _assert_native_rest_atlas_surfaces(client: httpx.AsyncClient) -> None:
    atlas_resp = await client.get("/api/graph/atlas", params={"refresh": True})
    assert atlas_resp.status_code == 200
    atlas = atlas_resp.json()
    assert atlas["representation"]["scope"] == "atlas"
    assert atlas["representation"]["layout"] == "precomputed"
    assert atlas["representation"]["snapshotId"]
    assert atlas["stats"]["totalEntities"] > 0
    assert atlas["stats"]["totalRegions"] > 0
    assert atlas["regions"]

    snapshot_id = atlas["representation"]["snapshotId"]
    region_id = atlas["regions"][0]["id"]

    history_resp = await client.get("/api/graph/atlas/history", params={"limit": 10})
    assert history_resp.status_code == 200
    history = history_resp.json()["items"]
    assert any(item["id"] == snapshot_id for item in history)

    snapshot_resp = await client.get(
        "/api/graph/atlas",
        params={"snapshot_id": snapshot_id},
    )
    assert snapshot_resp.status_code == 200
    snapshot = snapshot_resp.json()
    assert snapshot["representation"]["snapshotId"] == snapshot_id
    assert snapshot["stats"]["totalEntities"] == atlas["stats"]["totalEntities"]

    region_resp = await client.get(
        f"/api/graph/regions/{region_id}",
        params={"snapshot_id": snapshot_id},
    )
    assert region_resp.status_code == 200
    region = region_resp.json()
    assert region["representation"]["scope"] == "region"
    assert region["representation"]["layout"] == "precomputed"
    assert region["representation"]["snapshotId"] == snapshot_id
    assert region["region"]["id"] == region_id
    assert region["memberIds"]
    assert region["nodes"]
    assert isinstance(region["edges"], list)
    assert isinstance(region["topEntities"], list)


async def _assert_native_atlas_store_upsert_surface() -> None:
    store = _app_state["atlas_store"]
    snapshot_id = "atlas_native_upsert"
    first = AtlasSnapshot(
        id=snapshot_id,
        group_id=NATIVE_GROUP_ID,
        represented_entity_count=1,
        represented_edge_count=0,
        displayed_node_count=1,
        displayed_edge_count=0,
        total_entities=1,
        total_relationships=0,
        total_regions=1,
        hottest_region_id="region_old",
        fastest_growing_region_id="region_old",
        regions=[
            AtlasRegion(
                id="region_old",
                label="Old native atlas region",
                subtitle=None,
                kind="topic",
                member_count=1,
                represented_edge_count=0,
                activation_score=0.2,
                growth_7d=1,
                growth_30d=1,
                dominant_entity_types={"TestMemory": 1},
                hub_entity_ids=["ent_native_atlas_old"],
                center_entity_id="ent_native_atlas_old",
            )
        ],
        region_members={"region_old": ["ent_native_atlas_old"]},
    )
    second = AtlasSnapshot(
        id=snapshot_id,
        group_id=NATIVE_GROUP_ID,
        represented_entity_count=1,
        represented_edge_count=0,
        displayed_node_count=1,
        displayed_edge_count=0,
        total_entities=1,
        total_relationships=0,
        total_regions=1,
        hottest_region_id="region_new",
        fastest_growing_region_id="region_new",
        regions=[
            AtlasRegion(
                id="region_new",
                label="New native atlas region",
                subtitle="upsert replacement",
                kind="topic",
                member_count=1,
                represented_edge_count=0,
                activation_score=0.4,
                growth_7d=1,
                growth_30d=1,
                dominant_entity_types={"TestMemory": 1},
                hub_entity_ids=["ent_native_atlas_new"],
                center_entity_id="ent_native_atlas_new",
            )
        ],
        region_members={"region_new": ["ent_native_atlas_new"]},
    )

    await store.save_snapshot(first)
    loaded_first = await store.get_snapshot(snapshot_id, NATIVE_GROUP_ID)
    assert loaded_first is not None
    assert [region.id for region in loaded_first.regions] == ["region_old"]
    assert loaded_first.region_members == {"region_old": ["ent_native_atlas_old"]}

    await store.save_snapshot(second)
    loaded_second = await store.get_snapshot(snapshot_id, NATIVE_GROUP_ID)
    assert loaded_second is not None
    assert [region.id for region in loaded_second.regions] == ["region_new"]
    assert loaded_second.regions[0].label == "New native atlas region"
    assert loaded_second.region_members == {"region_new": ["ent_native_atlas_new"]}
    assert (
        await store.get_region_members(snapshot_id, "region_old", NATIVE_GROUP_ID)
        == []
    )


async def _assert_native_rest_entity_mutation_surface(client: httpx.AsyncClient) -> None:
    entity_id = "ent_native_entity_mutation_rest"
    original_name = "NativeEntityMutationRest"
    updated_name = "NativeEntityMutationRestUpdated"
    updated_summary = "Updated native entity mutation summary."
    await _create_native_test_entity(entity_id, original_name)

    patch_resp = await client.patch(
        f"/api/entities/{entity_id}",
        json={"name": updated_name, "summary": updated_summary},
    )
    assert patch_resp.status_code == 200
    patched = patch_resp.json()
    assert patched["id"] == entity_id
    assert patched["name"] == updated_name
    assert patched["summary"] == updated_summary
    assert patched["entityType"] == "TestMemory"

    stored = await _app_state["graph_store"].get_entity(entity_id, NATIVE_GROUP_ID)
    assert stored is not None
    assert stored.name == updated_name
    assert stored.summary == updated_summary

    detail_resp = await client.get(f"/api/entities/{entity_id}")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["name"] == updated_name

    delete_resp = await client.delete(f"/api/entities/{entity_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {
        "status": "deleted",
        "id": entity_id,
        "name": updated_name,
    }
    assert await _app_state["graph_store"].get_entity(entity_id, NATIVE_GROUP_ID) is None
    assert await _app_state["activation_store"].get_activation(entity_id) is None

    detail_after = await client.get(f"/api/entities/{entity_id}")
    assert detail_after.status_code == 404
    search_after = await client.get("/api/entities/search", params={"q": updated_name})
    assert search_after.status_code == 200
    assert all(item["id"] != entity_id for item in search_after.json()["items"])


async def _assert_native_rest_forget_surface(client: httpx.AsyncClient) -> None:
    entity_id = "ent_native_forget_rest"
    entity_name = "NativeForgetRest"
    await _create_native_test_entity(entity_id, entity_name)

    search_resp = await client.get("/api/entities/search", params={"q": entity_name})
    assert search_resp.status_code == 200
    assert any(item["id"] == entity_id for item in search_resp.json()["items"])

    forget_resp = await client.post(
        "/api/knowledge/forget",
        json={
            "entity_name": entity_name,
            "reason": "native REST forget parity",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert forget_resp.status_code == 200
    forget_payload = forget_resp.json()
    assert forget_payload["status"] == "forgotten"
    assert forget_payload["target_type"] == "entity"
    assert forget_payload["target"] == entity_name

    search_after = await client.get("/api/entities/search", params={"q": entity_name})
    assert search_after.status_code == 200
    assert all(item["id"] != entity_id for item in search_after.json()["items"])
    activation = await _app_state["activation_store"].get_activation(entity_id)
    assert activation is None


async def _assert_native_mcp_forget_surface(mcp_server) -> None:
    entity_id = "ent_native_forget_mcp"
    entity_name = "NativeForgetMcp"
    await _create_native_test_entity(entity_id, entity_name)

    result = json.loads(
        await mcp_server.forget(
            entity_name=entity_name,
            reason="native MCP forget parity",
        )
    )
    assert result["status"] == "forgotten"
    assert result["target_type"] == "entity"
    assert result["target"] == entity_name
    search_after = await _app_state["graph_manager"].search_entities(
        group_id=NATIVE_GROUP_ID,
        name=entity_name,
    )
    assert search_after == []
    activation = await _app_state["activation_store"].get_activation(entity_id)
    assert activation is None


async def _preference_edge_targets(predicate: str) -> set[str]:
    graph_store = _app_state["graph_store"]
    prefs = await graph_store.find_entities(
        name="UserPreference",
        entity_type="PreferenceProfile",
        group_id=NATIVE_GROUP_ID,
        limit=1,
    )
    assert prefs
    relationships = await graph_store.get_relationships(
        prefs[0].id,
        direction="outgoing",
        predicate=predicate,
        active_only=True,
        group_id=NATIVE_GROUP_ID,
    )
    return {relationship.target_id for relationship in relationships}


async def _assert_native_rest_feedback_surface(client: httpx.AsyncClient) -> None:
    entity_id = "ent_native_feedback_rest"
    await _create_native_test_entity(entity_id, "NativeFeedbackRest")

    feedback_resp = await client.post(
        "/api/knowledge/feedback",
        json={
            "entity_id": entity_id,
            "rating": 5,
            "comment": "native REST feedback parity",
            "groupId": "wrong_native_brain",
            "group_id": "wrong_native_brain",
        },
    )
    assert feedback_resp.status_code == 200
    payload = feedback_resp.json()
    assert payload["status"] == "recorded"
    assert payload["entity_id"] == entity_id
    assert payload["edge_type"] == "PREFERS"
    assert payload["edge_weight"] == 1.0
    assert entity_id in await _preference_edge_targets("PREFERS")


async def _assert_native_mcp_feedback_surface(mcp_server) -> None:
    entity_id = "ent_native_feedback_mcp"
    await _create_native_test_entity(entity_id, "NativeFeedbackMcp")

    payload = json.loads(
        await mcp_server.feedback(
            entity_id=entity_id,
            rating=1,
            comment="native MCP feedback parity",
        )
    )
    assert payload["status"] == "recorded"
    assert payload["entity_id"] == entity_id
    assert payload["edge_type"] == "AVOIDS"
    assert payload["edge_weight"] == 1.0
    assert entity_id in await _preference_edge_targets("AVOIDS")


async def _assert_native_mcp_identity_core_surface(mcp_server) -> None:
    entity_id = "ent_native_identity_core_mcp"
    entity_name = "NativeIdentityCoreMcp"
    await _create_native_test_entity(entity_id, entity_name)

    marked = json.loads(
        await mcp_server.mark_identity_core(
            entity_name=entity_name,
            identity_core=True,
        )
    )
    assert marked["status"] == "updated"
    assert marked["entity"] == entity_name
    assert marked["identity_core"] is True
    entity = await _app_state["graph_store"].get_entity(entity_id, NATIVE_GROUP_ID)
    assert entity is not None
    assert entity.identity_core is True

    unmarked = json.loads(
        await mcp_server.mark_identity_core(
            entity_name=entity_name,
            identity_core=False,
        )
    )
    assert unmarked["status"] == "updated"
    assert unmarked["entity"] == entity_name
    assert unmarked["identity_core"] is False
    entity = await _app_state["graph_store"].get_entity(entity_id, NATIVE_GROUP_ID)
    assert entity is not None
    assert entity.identity_core is False


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces(
    monkeypatch,
    tmp_path,
) -> None:
    labels_path = tmp_path / "native-surface-labels.db"
    helix_data_dir = tmp_path / "native-surface-data"

    smoke_report = await run_projected_consolidated_smoke(
        labels_path,
        group_id=NATIVE_GROUP_ID,
        mode=EngineMode.HELIX,
        helix_data_dir=helix_data_dir,
    )
    assert smoke_report["coverage_gaps"] == []
    assert smoke_report["project"]["yield"]["linked_entity_count"] > 0

    config = _native_surface_config(labels_path, helix_data_dir)
    app = create_app(config)
    started = False
    project_dir = tmp_path / "native-artifact-project"
    try:
        await _startup(app, config)
        started = True
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await _assert_native_rest_health_surface(client)
            await _assert_native_rest_surfaces(client)
            await _assert_native_rest_consolidation_surfaces(client)
            await _assert_native_rest_notification_surfaces(client)
            await _record_native_rest_evaluation_labels(client)
            await _assert_native_project_artifact_surfaces(
                client,
                project_dir,
            )
            await _assert_native_rest_runtime_surface(client, project_dir)
            await _assert_native_rest_route_surface(client, project_dir)
            await _assert_native_rest_intention_surfaces(client)
            await _assert_native_rest_adjudication_surface(client)
            await _assert_native_rest_auto_observe_surface(client)
            await _assert_native_rest_observe_surface(client)
            await _assert_native_rest_attachment_surfaces(client)
            await _assert_native_conversation_surfaces(client)
            await _assert_native_chat_surface(client)
            await _assert_native_rest_context_surface(client)
            await _assert_native_rest_entity_fact_lookup_surface(client)
            await _assert_native_rest_dashboard_read_surfaces(client)
            await _assert_native_rest_atlas_surfaces(client)
            await _assert_native_atlas_store_upsert_surface()
            await _assert_native_rest_entity_mutation_surface(client)
            await _assert_native_rest_forget_surface(client)
            await _assert_native_rest_feedback_surface(client)
            await _assert_native_rest_admin_benchmark_surface(client, monkeypatch)

        from engram.mcp import server as mcp_server

        monkeypatch.setattr(mcp_server, "_manager", _app_state["graph_manager"])
        monkeypatch.setattr(mcp_server, "_group_id", NATIVE_GROUP_ID)
        monkeypatch.setattr(mcp_server, "_session", SessionState(group_id=NATIVE_GROUP_ID))
        monkeypatch.setattr(mcp_server, "_activation_cfg", config.activation)
        monkeypatch.setattr(mcp_server, "_evaluation_store", _app_state["evaluation_store"])
        monkeypatch.setattr(mcp_server, "_consolidation_store", _app_state["consolidation_store"])

        await _assert_native_mcp_project_bootstrap_surface(mcp_server, project_dir)
        await _assert_native_mcp_project_artifact_surface(mcp_server, project_dir)
        await _assert_native_mcp_runtime_surface(mcp_server, project_dir)
        await _assert_native_mcp_memory_authority_surface(mcp_server, project_dir)
        await _assert_native_mcp_consolidation_control_surface(mcp_server)
        await _assert_native_mcp_entity_fact_lookup_surface(mcp_server)
        await _assert_native_mcp_graph_stats_resource(mcp_server)
        await _assert_native_mcp_evaluation_write_surface(mcp_server)
        await _assert_native_mcp_forget_surface(mcp_server)
        await _assert_native_mcp_feedback_surface(mcp_server)
        await _assert_native_mcp_identity_core_surface(mcp_server)
        await _assert_native_mcp_context_surface(mcp_server)
        await _assert_native_mcp_notification_surface(mcp_server)
        await _assert_native_mcp_auto_recall_surface(mcp_server)

        mcp_lifecycle = json.loads(await mcp_server.get_lifecycle_summary())
        assert mcp_lifecycle["groupId"] == NATIVE_GROUP_ID
        assert mcp_lifecycle["totals"]["episodes"] >= 4
        assert mcp_lifecycle["totals"]["projected"] == 3
        assert mcp_lifecycle["consolidate"]["cycleCount"] >= 2

        mcp_report = json.loads(await mcp_server.get_evaluation_report())
        assert mcp_report["group_id"] == NATIVE_GROUP_ID
        assert mcp_report["coverage_gaps"] == []
        assert mcp_report["project"]["yield"]["linked_entity_count"] > 0
        assert mcp_report["recall"]["evaluation"]["status"] == "measured"
        assert mcp_report["recall"]["evaluation"]["sample_count"] >= 3
        assert mcp_report["recall"]["continuity"]["sample_count"] >= 3
        assert {
            signal["status"]
            for signal in mcp_report["evaluation_signals"].values()
        } == {"measured"}

        mcp_recall = json.loads(await mcp_server.recall("Engram brain loop", limit=5))
        assert mcp_recall["total_candidates"] > 0
        assert mcp_recall["results"]

        await _assert_native_mcp_route_surface(mcp_server, project_dir)
        await _assert_native_mcp_route_auto_observe_surface(mcp_server, project_dir)
        await _assert_native_mcp_intention_surfaces(mcp_server)
        await _assert_native_mcp_adjudication_surface(mcp_server)
        await _assert_native_mcp_write_surfaces(mcp_server)

        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await _assert_native_rest_consolidation_trigger_surface(client)
    finally:
        if started:
            await _shutdown()
        _app_state.clear()


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_rest_surfaces_survive_repeated_runtime_reopen(
    tmp_path,
) -> None:
    labels_path = tmp_path / "native-reopen-labels.db"
    helix_data_dir = tmp_path / "native-reopen-data"

    smoke_report = await run_projected_consolidated_smoke(
        labels_path,
        group_id=NATIVE_GROUP_ID,
        mode=EngineMode.HELIX,
        helix_data_dir=helix_data_dir,
    )
    assert smoke_report["coverage_gaps"] == []

    for _attempt in range(3):
        config = _native_surface_config(labels_path, helix_data_dir)
        app = create_app(config)
        started = False
        try:
            await _startup(app, config)
            started = True
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                await _assert_native_rest_surfaces(client)
        finally:
            if started:
                await _shutdown()
            _app_state.clear()


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
def test_native_helix_dashboard_websocket_uses_native_group(tmp_path) -> None:
    labels_path = tmp_path / "native-ws-labels.db"
    helix_data_dir = tmp_path / "native-ws-data"
    config = _native_surface_config(labels_path, helix_data_dir)
    app = create_app(config)

    try:
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                ws.send_json({"type": "ping"})
                pong = ws.receive_json()
                assert pong["type"] == "pong"
                assert "timestamp" in pong

                bus = get_event_bus()
                seed_seq = bus.publish(
                    NATIVE_GROUP_ID,
                    "native.websocket.seed",
                    {"surface": "seed"},
                )
                seed_event = ws.receive_json()
                assert seed_event["type"] == "native.websocket.seed"
                assert seed_event["seq"] == seed_seq

                seq = bus.publish(
                    NATIVE_GROUP_ID,
                    "native.websocket",
                    {"surface": "dashboard"},
                )
                bus.publish(
                    "wrong_native_brain",
                    "native.websocket",
                    {"surface": "wrong"},
                )
                event = ws.receive_json()
                assert event["type"] == "native.websocket"
                assert event["group_id"] == NATIVE_GROUP_ID
                assert event["seq"] == seq
                assert event["surface"] == "dashboard"

                ws.send_json(
                    {
                        "type": "command",
                        "command": "resync",
                        "lastSeq": seed_seq,
                    }
                )
                resync = ws.receive_json()
                assert resync["type"] == "resync"
                assert resync["isFull"] is False
                assert any(
                    item["seq"] == seq and item["group_id"] == NATIVE_GROUP_ID
                    for item in resync["events"]
                )
                assert all(
                    item["group_id"] == NATIVE_GROUP_ID for item in resync["events"]
                )
    finally:
        _app_state.clear()


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_rest_surfaces_handle_bounded_remember_recall_load(
    monkeypatch,
    tmp_path,
) -> None:
    labels_path = tmp_path / "native-load-labels.db"
    helix_data_dir = tmp_path / "native-load-data"

    smoke_report = await run_projected_consolidated_smoke(
        labels_path,
        group_id=NATIVE_GROUP_ID,
        mode=EngineMode.HELIX,
        helix_data_dir=helix_data_dir,
    )
    assert smoke_report["coverage_gaps"] == []

    config = _native_surface_config(labels_path, helix_data_dir)
    app = create_app(config)
    started = False
    expected_count = 3 + len(NATIVE_LOAD_CONTENTS) + 1
    expected_projected_count = 3 + len(NATIVE_LOAD_CONTENTS)
    try:
        await _startup(app, config)
        started = True
        from engram.api import knowledge as knowledge_api

        monkeypatch.setattr(
            knowledge_api,
            "drain_queue",
            lambda: [
                {
                    "content": NATIVE_REPLAY_CONTENT,
                    "group_id": "wrong_native_brain",
                    "source": "offline:native-load-smoke",
                    "session_id": "native-replay-session",
                }
            ],
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            for content in NATIVE_LOAD_CONTENTS:
                response = await client.post(
                    "/api/knowledge/remember",
                    json={"content": content, "source": "native-load-smoke"},
                )
                assert response.status_code == 200
                payload = response.json()
                assert payload["status"] == "remembered"
                assert payload["operation"] == "remember"
                assert payload["lifecycle"]["stage"] == "project"

            replay_resp = await client.post("/api/knowledge/replay-queue")
            assert replay_resp.status_code == 200
            replay_payload = replay_resp.json()
            assert replay_payload == {
                "status": "replayed",
                "replayed": 1,
                "skipped": 0,
                "total": 1,
            }

            for _round in range(4):
                for query in NATIVE_LOAD_RECALL_QUERIES:
                    recall_resp = await client.get(
                        "/api/knowledge/recall",
                        params={"q": query, "limit": 5},
                    )
                    assert recall_resp.status_code == 200
                    recall_payload = recall_resp.json()
                    assert recall_payload["query"] == query
                    assert recall_payload["items"]

            await _assert_native_rest_surfaces(
                client,
                min_episodes=expected_count,
                min_cues=expected_count,
                min_projected=expected_projected_count,
            )
    finally:
        if started:
            await _shutdown()
        _app_state.clear()


@pytest.mark.requires_helix
@pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)
@pytest.mark.asyncio
async def test_native_helix_rest_surfaces_survive_multi_batch_load_and_reopen(
    tmp_path,
) -> None:
    labels_path = tmp_path / "native-soak-labels.db"
    helix_data_dir = tmp_path / "native-soak-data"

    smoke_report = await run_projected_consolidated_smoke(
        labels_path,
        group_id=NATIVE_GROUP_ID,
        mode=EngineMode.HELIX,
        helix_data_dir=helix_data_dir,
    )
    assert smoke_report["coverage_gaps"] == []

    total_soak_items = sum(len(batch) for batch in NATIVE_SOAK_BATCHES)
    expected_count = 3 + total_soak_items

    config = _native_surface_config(labels_path, helix_data_dir)
    app = create_app(config)
    started = False
    try:
        await _startup(app, config)
        started = True
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            remembered = 0
            for batch_index, batch in enumerate(NATIVE_SOAK_BATCHES, start=1):
                for content in batch:
                    response = await client.post(
                        "/api/knowledge/remember",
                        json={"content": content, "source": f"native-soak-batch-{batch_index}"},
                    )
                    assert response.status_code == 200
                    payload = response.json()
                    assert payload["status"] == "remembered"
                    assert payload["lifecycle"]["stage"] == "project"
                    remembered += 1

                lifecycle_resp = await client.get("/api/lifecycle/summary")
                assert lifecycle_resp.status_code == 200
                lifecycle = lifecycle_resp.json()
                assert lifecycle["groupId"] == NATIVE_GROUP_ID
                assert lifecycle["totals"]["episodes"] >= 3 + remembered
                assert lifecycle["totals"]["projected"] >= 3 + remembered

                for query in NATIVE_SOAK_RECALL_QUERIES[: batch_index * 2]:
                    recall_resp = await client.get(
                        "/api/knowledge/recall",
                        params={"q": query, "limit": 5},
                    )
                    assert recall_resp.status_code == 200
                    recall_payload = recall_resp.json()
                    assert recall_payload["query"] == query
                    assert recall_payload["items"]

            await _assert_native_rest_surfaces(
                client,
                min_episodes=expected_count,
                min_cues=expected_count,
                min_projected=expected_count,
            )
    finally:
        if started:
            await _shutdown()
        _app_state.clear()

    reopen_config = _native_surface_config(labels_path, helix_data_dir)
    reopen_app = create_app(reopen_config)
    reopened = False
    try:
        await _startup(reopen_app, reopen_config)
        reopened = True
        transport = httpx.ASGITransport(app=reopen_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await _assert_native_rest_surfaces(
                client,
                min_episodes=expected_count,
                min_cues=expected_count,
                min_projected=expected_count,
            )
            for query in NATIVE_SOAK_RECALL_QUERIES:
                recall_resp = await client.get(
                    "/api/knowledge/recall",
                    params={"q": query, "limit": 5},
                )
                assert recall_resp.status_code == 200
                assert recall_resp.json()["items"]
    finally:
        if reopened:
            await _shutdown()
        _app_state.clear()
