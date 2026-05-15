from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.ingestion.project_bootstrap import (
    build_project_bootstrap_surface,
    project_bootstrap_http_status,
)
from engram.retrieval.runtime_state import build_runtime_state_surface


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
