from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.models.epistemic import ArtifactHit, EvidenceClaim
from engram.retrieval.artifacts import (
    build_api_artifact_search_surface,
    build_mcp_artifact_search_surface,
    build_mcp_artifact_search_tool_surface,
)


@pytest.mark.asyncio
async def test_artifact_search_surfaces_share_hit_loading_with_surface_keys() -> None:
    manager = MagicMock()
    manager.search_artifacts = AsyncMock(
        return_value=[
            ArtifactHit(
                artifact_id="art_readme",
                path="README.md",
                artifact_class="readme",
                snippet="OpenClaw native artifact",
                score=0.87,
                supporting_claims=[
                    EvidenceClaim(
                        subject="Engram",
                        predicate="USES",
                        object="OpenClaw",
                        source_type="artifact",
                        authority_type="workspace",
                        externalization_state="externalized",
                        confidence=0.9,
                        provenance={"path": "README.md"},
                    )
                ],
            )
        ]
    )

    api = await build_api_artifact_search_surface(
        manager,
        group_id="native_brain",
        query="OpenClaw",
        project_path="/tmp/project",
        limit=5,
    )
    mcp = await build_mcp_artifact_search_surface(
        manager,
        group_id="native_brain",
        query="OpenClaw",
        project_path="/tmp/project",
        limit=5,
    )

    assert manager.search_artifacts.await_count == 2
    assert api["projectPath"] == "/tmp/project"
    assert mcp["project_path"] == "/tmp/project"
    assert api["items"][0]["artifactId"] == "art_readme"
    assert mcp["items"][0]["supportingClaims"][0]["object"] == "OpenClaw"


@pytest.mark.asyncio
async def test_mcp_artifact_search_tool_surface_runs_middleware() -> None:
    hit = ArtifactHit(
        artifact_id="art_readme",
        path="README.md",
        artifact_class="readme",
        snippet="OpenClaw native artifact",
        score=0.87,
    )
    manager = MagicMock()
    manager.search_artifacts = AsyncMock(return_value=[hit])
    recall_middleware = AsyncMock()

    payload = await build_mcp_artifact_search_tool_surface(
        manager,
        group_id="native_brain",
        query="OpenClaw",
        project_path="/tmp/project",
        limit=5,
        recall_middleware=recall_middleware,
    )

    assert payload["items"][0]["artifactId"] == "art_readme"
    recall_middleware.assert_awaited_once_with(
        "OpenClaw",
        payload,
        tool_name="search_artifacts",
    )
