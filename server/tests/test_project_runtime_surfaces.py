from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.ingestion.project_bootstrap import (
    build_project_bootstrap_surface,
    project_bootstrap_http_status,
)
from engram.retrieval.memory_authority import (
    build_mcp_memory_authority_surface,
    validate_agent_protocol_calls,
)
from engram.retrieval.runtime_state import RuntimeStateService, build_runtime_state_surface


@pytest.mark.asyncio
async def test_project_bootstrap_surface_forwards_group_path_and_session() -> None:
    manager = MagicMock()
    manager.bootstrap_project = AsyncMock(
        return_value={
            "status": "bootstrapped",
            "project_entity_id": "ent_project",
            "files_observed": ["README.md"],
        }
    )

    result = await build_project_bootstrap_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        session_id="sess_native",
    )

    assert result["status"] == "bootstrapped"
    assert project_bootstrap_http_status(result) == 200
    manager.bootstrap_project.assert_awaited_once_with(
        project_path="/tmp/engram",
        group_id="native_brain",
        include_patterns=None,
        session_id="sess_native",
    )


def test_project_bootstrap_http_status_maps_skipped_to_bad_request() -> None:
    assert project_bootstrap_http_status({"status": "skipped", "reason": "invalid_path"}) == 400
    assert project_bootstrap_http_status({"status": "already_bootstrapped"}) == 200


@pytest.mark.asyncio
async def test_runtime_state_surface_forwards_group_and_project_path() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "projectName": "engram",
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {"projectPath": "/tmp/engram"},
        }
    )

    result = await build_runtime_state_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
    )

    assert result["runtime"]["mode"] == "helix"
    manager.get_runtime_state.assert_awaited_once_with(
        group_id="native_brain",
        project_path="/tmp/engram",
    )


@pytest.mark.asyncio
async def test_runtime_state_includes_empty_runtime_adoption_guidance() -> None:
    async def list_project_artifacts(**_: object) -> list[object]:
        return []

    service = RuntimeStateService(
        cfg=ActivationConfig(artifact_bootstrap_enabled=True),
        runtime_mode="helix",
        list_project_artifacts=list_project_artifacts,
        artifact_is_stale=lambda *_: False,
        get_recall_metrics=lambda _: {},
        get_memory_operation_metrics=lambda _: {},
        get_epistemic_metrics=lambda _: {},
        get_packet_cache_summary=lambda _: {"fresh_count": 1, "hit_count": 2},
    )

    result = await service.get_runtime_state(
        group_id="native_brain",
        project_path="/tmp/engram",
    )

    guidance = result["agentAdoption"]
    assert result["stats"]["packetCache"] == {"fresh_count": 1, "hit_count": 2}
    assert guidance["status"] == "fresh_runtime"
    assert guidance["doNotTreatEmptyAsFailure"] is True
    assert guidance["requiredNextTools"] == [
        "claim_authority",
        "bootstrap_project",
        "get_context",
    ]
    assert guidance["beforeAnswer"] == {
        "required": True,
        "tools": ["claim_authority", "bootstrap_project", "get_context"],
        "reason": (
            "Follow these Engram tools before the next substantive answer; "
            "runtime-state metrics are diagnostics, not a substitute for "
            "authority, bootstrap, and context loading."
        ),
    }
    assert "onboarding state" in guidance["emptyRuntimePolicy"]
    assert "portable cross-context memory authority" in guidance["fileMemoryPolicy"]
    assert guidance["claimAuthority"]["args"] == {
        "project_path": "/tmp/engram",
        "file_memory_present": "<true if local/file memory is visible>",
    }
    assert guidance["bootstrap"] == {
        "tool": "bootstrap_project",
        "required": True,
        "args": {"project_path": "/tmp/engram"},
        "reason": (
            "A fresh or empty artifact substrate is onboarding state, not proof "
            "that Engram has no useful memory."
        ),
    }


@pytest.mark.asyncio
async def test_memory_authority_surface_recommends_bootstrap_for_fresh_runtime() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 0,
                "staleArtifactCount": 0,
                "lastObservedAt": None,
            },
            "stats": {"recallMetrics": {}, "epistemicMetrics": {}},
        }
    )

    result = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message=(
            "I am actively building Engram as cross-context memory for every AI harness."
        ),
        file_memory_present=True,
    )

    assert result["authority"]["source_of_truth"] == "portable_cross_context_memory"
    assert "cross-project user facts" in result["authority"]["engram_owns"]
    assert "repo-specific coding conventions" in result["authority"]["project_local_files_own"]
    assert result["onboarding"]["state"] == "fresh_runtime"
    assert result["onboarding"]["should_bootstrap"] is True
    assert result["onboarding"]["recommended_actions"][0] == {
        "tool": "bootstrap_project",
        "args": {"project_path": "/tmp/engram"},
        "reason": "Seed project artifacts before judging recall usefulness.",
    }
    assert result["runtime"]["runtime"]["mode"] == "helix"
    assert result["lifecycle"] == ["Capture", "Cue", "Project", "Recall", "Consolidate"]
    assert result["agent_protocol"]["file_memory_present"] is True
    assert result["agent_protocol"]["file_memory_is_substitute"] is False
    assert result["agent_protocol"]["required_tools_before_answer"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert result["agent_protocol"]["capture"] == {
        "destination": "engram",
        "tool": "remember",
        "reason": "High-signal cross-context user fact or durable decision.",
    }
    assert result["agent_protocol"]["verification"]["command"] == (
        "engram adoption --authority claim-authority.json --calls mcp-calls.jsonl"
    )
    assert result["agent_protocol"]["verification"]["live_evidence_command"] == (
        "engram adoption --authority claim-authority.json "
        "--calls live-harness-transcript.json --require-live-evidence"
    )
    assert result["agent_protocol"]["verification"]["capture_required"] is True
    assert result["agent_protocol"]["verification"]["transcript_schema"][
        "required_fields"
    ] == ["phase", "tool"]
    assert result["agent_protocol"]["verification"]["transcript_schema"]["example"] == [
        {"phase": "before_answer", "tool": "bootstrap_project"},
        {"phase": "before_answer", "tool": "get_context"},
        {"phase": "before_answer", "tool": "recall"},
        {"phase": "capture", "tool": "remember"},
    ]


@pytest.mark.asyncio
async def test_memory_authority_surface_is_ready_after_bootstrap() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 3,
                "staleArtifactCount": 0,
                "lastObservedAt": "2026-05-18T10:00:00+00:00",
            },
            "stats": {
                "recallMetrics": {"total_queries": 4},
                "epistemicMetrics": {"routes": {"total": 2}},
            },
        }
    )

    result = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
    )

    assert result["onboarding"]["state"] == "ready"
    assert result["onboarding"]["should_bootstrap"] is False
    assert result["onboarding"]["recommended_actions"][0]["tool"] == "get_context"


@pytest.mark.asyncio
async def test_memory_authority_surface_bootstraps_missing_artifacts_even_with_metrics() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "lite"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 0,
                "staleArtifactCount": 0,
                "lastObservedAt": None,
            },
            "stats": {
                "recallMetrics": {"total_queries": 7},
                "epistemicMetrics": {"routes": {"total": 3}},
            },
        }
    )

    result = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="What did we decide about Engram adoption?",
        file_memory_present=True,
    )

    assert result["onboarding"]["state"] == "needs_project_bootstrap"
    assert result["onboarding"]["should_bootstrap"] is True
    assert result["agent_protocol"]["required_tools_before_answer"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]


@pytest.mark.asyncio
async def test_memory_authority_protocol_keeps_task_scratch_project_local() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "lite"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 2,
                "staleArtifactCount": 0,
                "lastObservedAt": "2026-05-18T10:00:00+00:00",
            },
            "stats": {"recallMetrics": {"total_queries": 1}, "epistemicMetrics": {}},
        }
    )

    result = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="Current task scratch: rename the local fixture in this branch.",
        file_memory_present=True,
    )

    assert result["agent_protocol"]["required_tools_before_answer"] == ["get_context"]
    assert result["agent_protocol"]["capture"] == {
        "destination": "project_local",
        "tool": None,
        "reason": "Treat as current-task scratch unless it becomes cross-context.",
    }
    assert result["agent_protocol"]["verification"]["capture_required"] is False
    assert result["agent_protocol"]["verification"]["transcript_schema"]["example"] == [
        {"phase": "before_answer", "tool": "get_context"},
    ]
    assert result["agent_protocol"]["verification"]["live_evidence_schema"]["example"][
        "calls"
    ] == [{"phase": "before_answer", "tool": "get_context"}]


@pytest.mark.asyncio
async def test_memory_authority_protocol_validation_accepts_followed_sequence() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 0,
                "staleArtifactCount": 0,
                "lastObservedAt": None,
            },
            "stats": {"recallMetrics": {}, "epistemicMetrics": {}},
        }
    )
    surface = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="I am building Engram as the cross-context AI memory brain.",
        file_memory_present=True,
    )

    validation = validate_agent_protocol_calls(
        surface["agent_protocol"],
        [
            {"phase": "before_answer", "tool": "bootstrap_project"},
            {"phase": "before_answer", "tool": "get_context"},
            {"phase": "before_answer", "tool": "recall"},
            {"phase": "capture", "tool": "remember"},
        ],
    )

    assert validation["status"] == "passed"
    assert validation["failures"] == []
    assert validation["required_tools_before_answer"]["missing"] == []
    assert validation["capture"]["missing"] is False
    assert validation["file_memory"]["substituted_for_engram"] is False


@pytest.mark.asyncio
async def test_memory_authority_protocol_validation_flags_file_memory_bypass() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 0,
                "staleArtifactCount": 0,
                "lastObservedAt": None,
            },
            "stats": {"recallMetrics": {}, "epistemicMetrics": {}},
        }
    )
    surface = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="I prefer Engram as the source of truth across AI harnesses.",
        file_memory_present=True,
    )

    validation = validate_agent_protocol_calls(
        surface["agent_protocol"],
        [
            {
                "phase": "before_answer",
                "tool": "read_file_memory",
                "source": "project_local_memory",
            }
        ],
    )

    assert validation["status"] == "failed"
    assert validation["required_tools_before_answer"]["missing"] == [
        "bootstrap_project",
        "get_context",
        "recall",
    ]
    assert validation["capture"]["missing"] is True
    assert validation["file_memory"]["substituted_for_engram"] is True
    assert "file_memory_used_as_substitute" in validation["failures"]


@pytest.mark.asyncio
async def test_memory_authority_protocol_validation_flags_out_of_order_calls() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 0,
                "staleArtifactCount": 0,
                "lastObservedAt": None,
            },
            "stats": {"recallMetrics": {}, "epistemicMetrics": {}},
        }
    )
    surface = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="What was the prior Engram decision?",
        file_memory_present=False,
    )

    validation = validate_agent_protocol_calls(
        surface["agent_protocol"],
        [
            {"phase": "before_answer", "tool": "recall"},
            {"phase": "before_answer", "tool": "get_context"},
            {"phase": "before_answer", "tool": "bootstrap_project"},
            {"phase": "capture", "tool": "observe"},
        ],
    )

    assert validation["status"] == "failed"
    assert validation["required_tools_before_answer"]["missing"] == []
    assert validation["required_tools_before_answer"]["in_order"] is False
    assert "required_before_answer_tools_out_of_order" in validation["failures"]


@pytest.mark.asyncio
async def test_memory_authority_protocol_validation_accepts_project_local_capture() -> None:
    manager = MagicMock()
    manager.get_runtime_state = AsyncMock(
        return_value={
            "runtime": {"mode": "lite"},
            "artifactBootstrap": {
                "enabled": True,
                "projectPath": "/tmp/engram",
                "artifactCount": 2,
                "staleArtifactCount": 0,
                "lastObservedAt": "2026-05-18T10:00:00+00:00",
            },
            "stats": {"recallMetrics": {"total_queries": 1}, "epistemicMetrics": {}},
        }
    )
    surface = await build_mcp_memory_authority_surface(
        manager,
        group_id="native_brain",
        project_path="/tmp/engram",
        user_message="Current task scratch: update this local fixture only.",
        file_memory_present=True,
    )

    validation = validate_agent_protocol_calls(
        surface["agent_protocol"],
        [{"phase": "before_answer", "tool": "get_context"}],
    )

    assert validation["status"] == "passed"
    assert validation["capture"]["destination"] == "project_local"
    assert validation["capture"]["unexpected_engram_capture_tools"] == []
