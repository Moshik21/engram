"""Tests for project bootstrap: auto-create Project entity, observe files, PART_OF edges."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.entity import Entity

# ─── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a temporary project directory with key files."""
    (tmp_path / "README.md").write_text("# My Project\nThis is a test project.")
    (tmp_path / "package.json").write_text('{"name": "my-project", "version": "1.0.0"}')
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "my-project"')
    return tmp_path


@pytest.fixture()
def empty_project(tmp_path: Path) -> Path:
    """Create an empty project directory with no recognized files."""
    return tmp_path


@pytest.fixture()
def manager() -> GraphManager:
    """Create a GraphManager with mocked stores."""
    graph = AsyncMock()
    activation = AsyncMock()
    search = AsyncMock()
    extractor = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.structure_aware_embeddings = False

    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.update_entity = AsyncMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.get_neighbors = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    search.index_entity = AsyncMock()

    mgr = GraphManager(graph, activation, search, extractor, cfg=cfg)
    return mgr


# ─── bootstrap_project Tests ────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_creates_project_entity(manager: GraphManager, tmp_project: Path):
    """Project entity created with correct name, type, and attributes."""
    result = await manager.bootstrap_project(str(tmp_project))

    assert result["status"] == "bootstrapped"
    assert result["project_entity_id"].startswith("ent_")

    # Verify entity creation
    create_call = manager._graph.create_entity.call_args_list[0]
    entity = create_call[0][0]
    assert entity.name == tmp_project.name
    assert entity.entity_type == "Project"
    assert entity.attributes["project_path"] == str(tmp_project)


@pytest.mark.asyncio
async def test_bootstrap_observes_files(manager: GraphManager, tmp_project: Path):
    """Episodes stored for each found file with correct tags."""
    # Mock store_episode to track calls
    stored_episodes: list[dict] = []

    async def mock_store(content: str, group_id: str = "default",
                         source: str | None = None, session_id: str | None = None) -> str:
        stored_episodes.append({
            "content": content, "source": source, "session_id": session_id,
        })
        return f"ep_test_{len(stored_episodes)}"

    manager.store_episode = mock_store

    result = await manager.bootstrap_project(str(tmp_project), session_id="sess_123")

    assert result["status"] == "bootstrapped"
    assert set(result["files_observed"]) == {"README.md", "package.json", "pyproject.toml"}

    # Check episode tags
    for ep in stored_episodes:
        assert ep["source"] == "auto:bootstrap"
        assert ep["session_id"] == "sess_123"
        assert ep["content"].startswith(f"[project-bootstrap|{tmp_project.name}|")


@pytest.mark.asyncio
async def test_bootstrap_idempotent_when_fresh(
    manager: GraphManager, tmp_project: Path,
):
    """Recent bootstrap returns already_bootstrapped, no re-observation."""
    recent_ts = datetime.utcnow().isoformat()
    existing = Entity(
        id="ent_existing123",
        name=tmp_project.name,
        entity_type="Project",
        group_id="default",
        attributes={"last_bootstrapped": recent_ts},
    )
    manager._graph.find_entities = AsyncMock(return_value=[existing])

    result = await manager.bootstrap_project(str(tmp_project))

    assert result["status"] == "already_bootstrapped"
    assert result["project_entity_id"] == "ent_existing123"
    manager._graph.create_entity.assert_not_called()


@pytest.mark.asyncio
async def test_bootstrap_refreshes_when_stale(
    manager: GraphManager, tmp_project: Path,
):
    """Stale project (>24h) re-observes files and returns refreshed."""
    stale_ts = (datetime.utcnow() - timedelta(hours=25)).isoformat()
    existing = Entity(
        id="ent_existing123",
        name=tmp_project.name,
        entity_type="Project",
        group_id="default",
        attributes={
            "project_path": str(tmp_project),
            "last_bootstrapped": stale_ts,
        },
    )
    manager._graph.find_entities = AsyncMock(return_value=[existing])
    manager._graph.update_entity = AsyncMock()

    stored_episodes: list[str] = []

    async def mock_store(content: str, **kwargs) -> str:
        stored_episodes.append(content)
        return "ep_test"

    manager.store_episode = mock_store

    result = await manager.bootstrap_project(str(tmp_project))

    assert result["status"] == "refreshed"
    assert result["project_entity_id"] == "ent_existing123"
    assert len(result["files_observed"]) > 0
    assert len(stored_episodes) > 0
    # Entity was NOT re-created
    manager._graph.create_entity.assert_not_called()
    # Timestamp was updated
    manager._graph.update_entity.assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_empty_directory(manager: GraphManager, empty_project: Path):
    """Project entity created, no episodes (no files found)."""
    stored = []

    async def mock_store(**kwargs):
        stored.append(kwargs)
        return "ep_test"

    manager.store_episode = mock_store

    result = await manager.bootstrap_project(str(empty_project))

    assert result["status"] == "bootstrapped"
    assert result["files_observed"] == []
    assert len(stored) == 0
    manager._graph.create_entity.assert_called_once()


@pytest.mark.asyncio
async def test_bootstrap_truncation(manager: GraphManager, tmp_path: Path):
    """Large README truncated to 2000 chars."""
    big_readme = "x" * 5000
    (tmp_path / "README.md").write_text(big_readme)

    stored_episodes: list[str] = []

    async def mock_store(content: str, **kwargs) -> str:
        stored_episodes.append(content)
        return "ep_test"

    manager.store_episode = mock_store

    await manager.bootstrap_project(str(tmp_path))

    assert len(stored_episodes) == 1
    # Tag prefix + 2000 chars of content
    tag_prefix = f"[project-bootstrap|{tmp_path.name}|README.md]\n"
    assert len(stored_episodes[0]) == len(tag_prefix) + 2000


@pytest.mark.asyncio
async def test_bootstrap_skips_missing_files(manager: GraphManager, tmp_path: Path):
    """Only observes files that exist."""
    # Only create README.md, not package.json or others
    (tmp_path / "README.md").write_text("Hello")

    stored_episodes: list[str] = []

    async def mock_store(content: str, **kwargs) -> str:
        stored_episodes.append(content)
        return "ep_test"

    manager.store_episode = mock_store

    result = await manager.bootstrap_project(str(tmp_path))

    assert result["files_observed"] == ["README.md"]
    assert len(stored_episodes) == 1


@pytest.mark.asyncio
async def test_bootstrap_skips_invalid_paths(manager: GraphManager):
    """Skips home directory and root path."""
    result = await manager.bootstrap_project(str(Path.home()))
    assert result["status"] == "skipped"

    result = await manager.bootstrap_project("/")
    assert result["status"] == "skipped"


# ─── get_context auto-create Tests ──────────────────────────────────


@pytest.mark.asyncio
async def test_get_context_auto_creates_project(tmp_path: Path):
    """get_context(project_path=...) creates Project entity if missing."""
    graph = AsyncMock()
    activation = AsyncMock()
    search = AsyncMock()
    extractor = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    # No existing project
    graph.find_entities = AsyncMock(return_value=[])
    graph.create_entity = AsyncMock()
    graph.get_neighbors = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_top_activated = AsyncMock(return_value=[])

    mgr = GraphManager(graph, activation, search, extractor, cfg=cfg)

    # Mock recall to return empty
    mgr.recall = AsyncMock(return_value=[])

    await mgr.get_context(
        group_id="default",
        project_path=str(tmp_path),
    )

    # Should have created a Project entity
    assert graph.create_entity.called
    created = graph.create_entity.call_args[0][0]
    assert created.entity_type == "Project"
    assert created.name == tmp_path.name


@pytest.mark.asyncio
async def test_get_context_project_neighbors_in_layer2(tmp_path: Path):
    """PART_OF-connected entities appear in Layer 2."""
    graph = AsyncMock()
    activation = AsyncMock()
    search = AsyncMock()
    extractor = MagicMock()
    cfg = ActivationConfig()
    cfg.identity_core_enabled = False
    cfg.briefing_enabled = False

    project_entity = Entity(
        id="ent_proj1", name=tmp_path.name, entity_type="Project",
        group_id="default",
    )
    neighbor_entity = Entity(
        id="ent_neighbor1", name="FastAPI", entity_type="Technology",
        summary="Web framework", group_id="default",
    )
    mock_rel = MagicMock()
    mock_rel.predicate = "PART_OF"
    mock_rel.source_id = "ent_neighbor1"
    mock_rel.target_id = "ent_proj1"
    mock_rel.weight = 0.8

    graph.find_entities = AsyncMock(return_value=[project_entity])
    graph.get_neighbors = AsyncMock(return_value=[(neighbor_entity, mock_rel)])
    graph.get_relationships = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    activation.get_top_activated = AsyncMock(return_value=[])

    mgr = GraphManager(graph, activation, search, extractor, cfg=cfg)
    mgr.recall = AsyncMock(return_value=[])

    result = await mgr.get_context(
        group_id="default",
        project_path=str(tmp_path),
    )

    # Context should include the neighbor
    assert "FastAPI" in result["context"]


# ─── project_episode PART_OF edge Tests ─────────────────────────────


@pytest.mark.asyncio
async def test_project_episode_creates_part_of_edges():
    """Bootstrap-sourced episodes get PART_OF edges to Project."""
    from engram.models.episode import Episode, EpisodeStatus

    graph = AsyncMock()
    activation = AsyncMock()
    search = AsyncMock()
    extractor = MagicMock()
    cfg = ActivationConfig()
    cfg.structure_aware_embeddings = False
    cfg.surprise_detection_enabled = False
    cfg.prospective_memory_enabled = False

    project_entity = Entity(
        id="ent_proj1", name="myproject", entity_type="Project",
        group_id="default",
    )

    episode = Episode(
        id="ep_test1",
        content='[project-bootstrap|myproject|README.md]\n# My Project\nHello world',
        source="auto:bootstrap",
        status=EpisodeStatus.QUEUED,
        group_id="default",
    )

    graph.get_episode_by_id = AsyncMock(return_value=episode)
    graph.find_entities = AsyncMock(return_value=[project_entity])
    graph.create_entity = AsyncMock()
    graph.create_relationship = AsyncMock()
    graph.link_episode_entity = AsyncMock()
    graph.update_episode = AsyncMock()
    graph.get_relationships = AsyncMock(return_value=[])
    graph.find_entity_candidates = AsyncMock(return_value=[])
    activation.record_access = AsyncMock()
    activation.get_activation = AsyncMock(return_value=None)
    search.index_entity = AsyncMock()
    search.index_episode = AsyncMock()

    # Mock extractor to return entities
    mock_result = MagicMock()
    mock_result.entities = [
        {"name": "My Project", "entity_type": "CreativeWork", "summary": "A project"},
    ]
    mock_result.relationships = []
    extractor.extract = AsyncMock(return_value=mock_result)

    mgr = GraphManager(graph, activation, search, extractor, cfg=cfg)

    await mgr.project_episode("ep_test1", "default")

    # Check that PART_OF relationship was created
    create_rel_calls = graph.create_relationship.call_args_list
    part_of_rels = [
        c for c in create_rel_calls
        if c[0][0].predicate == "PART_OF"
    ]
    assert len(part_of_rels) == 1
    rel = part_of_rels[0][0][0]
    assert rel.target_id == "ent_proj1"
    assert rel.weight == 0.8


# ─── REST endpoint test ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_endpoint():
    """REST endpoint returns correct response."""

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from engram.api.knowledge import router

    app = FastAPI()
    app.include_router(router)

    mock_manager = AsyncMock()
    mock_manager.bootstrap_project = AsyncMock(return_value={
        "status": "bootstrapped",
        "project_entity_id": "ent_abc123",
        "files_observed": ["README.md", "package.json"],
    })

    with (
        patch("engram.api.knowledge.get_manager", return_value=mock_manager),
        patch("engram.api.knowledge.get_tenant") as mock_tenant,
    ):
        mock_tenant.return_value = MagicMock(group_id="default")
        client = TestClient(app)

        response = client.post(
            "/api/knowledge/bootstrap",
            json={"project_path": "/tmp/myproject", "session_id": "sess_1"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "bootstrapped"
    assert data["project_entity_id"] == "ent_abc123"
    assert "README.md" in data["files_observed"]
