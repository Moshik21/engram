"""Tests for Engram AutoCapture — auto-observe endpoint, worker fixes, batching."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio

from engram.api.knowledge import _DEDUP_CACHE, _dedup_check
from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.graph_manager import GraphManager
from engram.ingestion.capture_surface import build_api_auto_observe_surface
from engram.ingestion.worker_batching import PendingEpisode
from engram.models.episode import EpisodeProjectionState
from engram.storage.memory.activation import MemoryActivationStore
from engram.worker import EpisodeWorker
from tests.conftest import MockExtractor, _helix_available

# GROUP is now generated per-test via gid fixture

@pytest.fixture
def gid():
    return f"test_{uuid4().hex[:8]}"


_HOOK_NAMES = (
    "capture-prompt.sh",
    "capture-response.sh",
    "session-start.sh",
    "session-end.sh",
)


# ── Dedup Tests ─────────────────────────────────────────────────────


class TestDedupCache:
    """Tests for the in-memory TTL dedup cache."""

    def setup_method(self):
        _DEDUP_CACHE.clear()

    def test_first_check_passes(self):
        """First time seeing content returns False (not a dup)."""
        assert _dedup_check("Hello world, this is a test") is False

    def test_duplicate_within_ttl_blocked(self):
        """Same content within TTL returns True (duplicate)."""
        content = "Hello world, this is a test"
        assert _dedup_check(content) is False
        assert _dedup_check(content) is True

    def test_different_content_passes(self):
        """Different content is not considered a duplicate."""
        assert _dedup_check("Content A for testing") is False
        assert _dedup_check("Content B for testing") is False

    def test_expired_entry_passes(self):
        """Content is allowed again after TTL expires."""
        content = "Expiring content test"
        assert _dedup_check(content) is False
        # Manually expire the entry
        for key in list(_DEDUP_CACHE):
            _DEDUP_CACHE[key] = time.time() - 400
        assert _dedup_check(content) is False


# ── Auto-Observe Endpoint Tests ─────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_observe_surface_stores_capture_with_route_policy():
    manager = SimpleNamespace(store_episode=AsyncMock(return_value="ep_auto_1"))

    payload = await build_api_auto_observe_surface(
        manager,
        content="[user|Engram] capture this prompt for latent memory",
        group_id="tenant_brain",
        source="auto:prompt",
        session_id="sess-123",
        conversation_date="2026-05-15T12:00:00",
        auto_observe_enabled=True,
        dedup_check=lambda _content: False,
    )

    assert payload["status"] == "observed"
    assert payload["episodeId"] == "ep_auto_1"
    kwargs = manager.store_episode.await_args.kwargs
    assert kwargs["group_id"] == "tenant_brain"
    assert kwargs["session_id"] == "sess-123"
    assert kwargs["conversation_date"].isoformat() == "2026-05-15T12:00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "content", "is_duplicate", "reason", "status"),
    [
        (False, "[user|Engram] long enough", False, "disabled", "skipped"),
        (True, "short", False, "too_short", "skipped"),
        (True, "[user|Engram] duplicate prompt body", True, None, "dedup_skipped"),
    ],
)
async def test_auto_observe_surface_skips_before_store(
    enabled,
    content,
    is_duplicate,
    reason,
    status,
):
    manager = SimpleNamespace(store_episode=AsyncMock())

    payload = await build_api_auto_observe_surface(
        manager,
        content=content,
        group_id="tenant_brain",
        source="auto:prompt",
        auto_observe_enabled=enabled,
        dedup_check=lambda _content: is_duplicate,
    )

    assert payload["status"] == status
    if reason is not None:
        assert payload["reason"] == reason
    manager.store_episode.assert_not_awaited()


@pytest_asyncio.fixture
async def api_client(tmp_path):
    """Create an httpx.AsyncClient wired to the full FastAPI app."""
    from engram.config import EngramConfig
    from engram.main import _shutdown, _startup, create_app

    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "auto_observe_api.db")},
    )
    app = create_app(config)
    await _startup(app, config)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client

    await _shutdown()


@pytest.mark.asyncio
async def test_auto_observe_endpoint(api_client):
    """Auto-observe endpoint stores episode and returns observed status."""
    _DEDUP_CACHE.clear()

    resp = await api_client.post(
        "/api/knowledge/auto-observe",
        json={
            "content": "[user|TestProject] This is a user prompt about Python",
            "source": "auto:prompt",
            "project": "TestProject",
            "role": "user",
            "session_id": "sess-123",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "observed"
    assert "episodeId" in data


@pytest.mark.asyncio
async def test_auto_observe_endpoint_accepts_claude_prompt_hook_payload(api_client):
    """Raw Claude UserPromptSubmit hook payloads normalize instead of 422ing."""
    _DEDUP_CACHE.clear()

    resp = await api_client.post(
        "/api/knowledge/auto-observe",
        json={
            "prompt": "Please remember that Engram should be the cross-context brain.",
            "cwd": "/Users/konnermoshier/Engram",
            "session_id": "sess-hook-prompt",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "observed"
    assert "episodeId" in data


@pytest.mark.asyncio
async def test_auto_observe_endpoint_accepts_claude_stream_message_payload(api_client):
    """Claude stream-json assistant records normalize into response captures."""
    _DEDUP_CACHE.clear()

    resp = await api_client.post(
        "/api/knowledge/auto-observe",
        json={
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Engram should call claim_authority before relying on file memory.",
                    }
                ],
            },
            "cwd": "/Users/konnermoshier/Engram",
            "session_id": "sess-hook-message",
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "observed"
    assert "episodeId" in data


@pytest.mark.asyncio
async def test_auto_observe_endpoint_skips_invalid_json(api_client):
    """Malformed async hook traffic returns a capture skip, not a 422."""
    _DEDUP_CACHE.clear()

    resp = await api_client.post(
        "/api/knowledge/auto-observe",
        content="{",
        headers={"Content-Type": "application/json"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    assert resp.json()["reason"] == "invalid_json"


@pytest.mark.asyncio
async def test_auto_observe_dedup(api_client):
    """Same content within 5 minutes is skipped as dedup."""
    _DEDUP_CACHE.clear()

    body = {
        "content": "[user|TestProject] Duplicate content for dedup test",
        "source": "auto:prompt",
        "project": "TestProject",
        "role": "user",
    }
    resp1 = await api_client.post("/api/knowledge/auto-observe", json=body)
    assert resp1.json()["status"] == "observed"

    resp2 = await api_client.post("/api/knowledge/auto-observe", json=body)
    assert resp2.json()["status"] == "dedup_skipped"


@pytest.mark.asyncio
async def test_auto_observe_short_content_skipped(api_client):
    """Content shorter than 10 chars is skipped."""
    _DEDUP_CACHE.clear()

    resp = await api_client.post(
        "/api/knowledge/auto-observe",
        json={"content": "hi", "source": "auto:prompt"},
    )
    assert resp.json()["status"] == "skipped"


@pytest.mark.asyncio
async def test_auto_observe_can_be_disabled(tmp_path):
    """Auto-observe can be disabled for clean local smoke/demo runs."""
    from engram.config import EngramConfig
    from engram.main import _shutdown, _startup, create_app

    _DEDUP_CACHE.clear()
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "auto_observe_disabled.db")},
        server={"auto_observe_enabled": False},
    )
    app = create_app(config)
    await _startup(app, config)

    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/knowledge/auto-observe",
                json={
                    "content": "[user|TestProject] This content would normally be captured",
                    "source": "auto:prompt",
                    "project": "TestProject",
                    "role": "user",
                    "session_id": "sess-disabled",
                },
            )
            episodes = await client.get("/api/episodes")
    finally:
        await _shutdown()

    assert resp.status_code == 200
    assert resp.json()["status"] == "skipped"
    assert resp.json()["reason"] == "disabled"
    assert episodes.status_code == 200
    assert episodes.json()["items"] == []


# ── Worker Full Content Fetch Tests ─────────────────────────────────


@pytest_asyncio.fixture
async def worker_setup(tmp_path):
    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore

    """Set up worker with graph store for testing."""
    graph_store = HelixGraphStore(HelixDBConfig(host="localhost", port=6969))
    await graph_store.initialize()
    activation_store = MemoryActivationStore(cfg=ActivationConfig())
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    search_index = HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969),
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=0,
        embed_provider="noop",
        embed_model="noop",
    )
    await search_index.initialize()

    cfg = ActivationConfig()
    cfg.triage_enabled = True
    cfg.triage_min_score = 0.1
    cfg.worker_enabled = True

    extractor = MockExtractor()
    mgr = GraphManager(
        graph_store,
        activation_store,
        search_index,
        extractor,
        cfg=cfg,
    )

    worker = EpisodeWorker(mgr, cfg)
    event_bus = EventBus()
    yield mgr, worker, event_bus, graph_store, cfg
    await graph_store.close()


@pytest.mark.asyncio
@pytest.mark.requires_helix
@pytest.mark.skipif(not _helix_available(), reason="HelixDB not available")
async def test_worker_full_content_fetch(worker_setup, gid):
    """Worker fetches full episode content for auto: sources."""
    mgr, worker, event_bus, graph_store, cfg = worker_setup

    full_content = (
        "[user|Engram] This is a detailed user prompt about building "
        "a persistent memory system with knowledge graphs"
    )
    episode_id = await mgr.store_episode(
        content=full_content,
        group_id=gid,
        source="auto:prompt",
        session_id="sess-456",
    )

    # Simulate event with truncated content
    event = {
        "type": "episode.queued",
        "payload": {
            "episode": {
                "episodeId": episode_id,
                "content": "[user|Engram] This is a detailed",
                "source": "auto:prompt",
            }
        },
    }

    worker._queue = asyncio.Queue()
    await worker._queue.put(event)

    task = asyncio.create_task(worker._consume(gid))
    await asyncio.sleep(0.1)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Episode was buffered for batching (auto: episodes)
    # Verify worker didn't crash and handled the event


@pytest.mark.asyncio
@pytest.mark.requires_helix
@pytest.mark.skipif(not _helix_available(), reason="HelixDB not available")
async def test_turn_batching(worker_setup, gid):
    """Adjacent auto turns within window are merged into one episode."""
    mgr, worker, event_bus, graph_store, cfg = worker_setup

    ep1 = await mgr.store_episode(
        content="[user|Engram] What is spreading activation?",
        group_id=gid,
        source="auto:prompt",
    )
    ep2_content = "[assistant|Engram] Spreading activation is a method used in cognitive science"
    ep2 = await mgr.store_episode(
        content=ep2_content,
        group_id=gid,
        source="auto:response",
    )

    worker._batch_buffer = [
        PendingEpisode(
            ep1,
            "[user|Engram] What is spreading activation?",
            "auto:prompt",
        ),
        PendingEpisode(ep2, ep2_content, "auto:response"),
    ]

    await worker._flush_batch(gid)

    primary = await graph_store.get_episode_by_id(ep1, gid)
    assert primary is not None
    assert "What is spreading activation?" in primary.content
    assert "Spreading activation is a method" in primary.content

    secondary = await graph_store.get_episode_by_id(ep2, gid)
    assert secondary is not None


@pytest.mark.asyncio
@pytest.mark.requires_helix
@pytest.mark.skipif(not _helix_available(), reason="HelixDB not available")
async def test_turn_batching_rebuilds_primary_cue_and_retires_secondary(worker_setup, gid):
    """Batch merge rebuilds the surviving cue and suppresses merged-away cue recall."""
    mgr, worker, event_bus, graph_store, cfg = worker_setup
    cfg.cue_layer_enabled = True
    cfg.cue_recall_enabled = True
    cfg.cue_vector_index_enabled = False
    cfg.triage_enabled = False
    worker._process = AsyncMock()

    ep1_content = "[user|Engram] What is spreading activation?"
    ep1 = await mgr.store_episode(
        content=ep1_content,
        group_id=gid,
        source="auto:prompt",
    )
    ep2_content = "[assistant|Engram] Spreading activation is a method used in cognitive science"
    ep2 = await mgr.store_episode(
        content=ep2_content,
        group_id=gid,
        source="auto:response",
    )

    primary_cue_before = await graph_store.get_episode_cue(ep1, gid)
    assert primary_cue_before is not None
    assert "cognitive science" not in primary_cue_before.cue_text

    worker._batch_buffer = [
        PendingEpisode(ep1, ep1_content, "auto:prompt"),
        PendingEpisode(ep2, ep2_content, "auto:response"),
    ]

    await worker._flush_batch(gid)

    primary = await graph_store.get_episode_by_id(ep1, gid)
    primary_cue = await graph_store.get_episode_cue(ep1, gid)
    secondary = await graph_store.get_episode_by_id(ep2, gid)
    secondary_cue = await graph_store.get_episode_cue(ep2, gid)

    assert primary is not None
    assert primary_cue is not None
    assert primary.projection_state == primary_cue.projection_state
    assert primary.last_projection_reason == primary_cue.route_reason
    assert "cognitive science" in primary_cue.cue_text

    assert secondary is not None
    assert secondary.projection_state == EpisodeProjectionState.MERGED
    assert secondary.last_projection_reason == f"merged_into:{ep1}"
    assert secondary_cue is None

    results = await mgr._search.search_episode_cues("cognitive science", group_id=gid)
    assert [episode_id for episode_id, _ in results] == [ep1]

    raw_results = await mgr._search.search_episodes("cognitive science", group_id=gid)
    assert ep2 not in {episode_id for episode_id, _ in raw_results}

    recall_results = await mgr.recall("cognitive science", group_id=gid, limit=5)
    recalled_episode_ids = {
        result["episode"]["id"] for result in recall_results if "episode" in result
    }
    assert ep1 in recalled_episode_ids
    assert ep2 not in recalled_episode_ids
    worker._process.assert_awaited_once_with(ep1, gid)


# ── Setup/Hooks Tests ───────────────────────────────────────────────


class TestInstallHooks:
    """Tests for the hook installation function."""

    def test_install_hooks_generates_default_scripts(self, tmp_path):
        """install_hooks writes bundled hook scripts for a fresh install."""
        from engram.setup import install_hooks

        hooks_dir = tmp_path / "hooks"
        settings_path = tmp_path / "settings.json"

        result = install_hooks(
            hooks_dir=hooks_dir,
            settings_path=settings_path,
        )

        assert result["settings_updated"] is True
        assert len(result["scripts"]) == 4

        prompt_script = hooks_dir / "capture-prompt.sh"
        response_script = hooks_dir / "capture-response.sh"
        session_start_script = hooks_dir / "session-start.sh"
        session_end_script = hooks_dir / "session-end.sh"
        for script in (
            prompt_script,
            response_script,
            session_start_script,
            session_end_script,
        ):
            assert script.exists()
            assert script.stat().st_mode & 0o111

        assert "/api/knowledge/auto-observe" in prompt_script.read_text()
        assert "ENGRAM_ADOPTION_TRACE_FILE" in prompt_script.read_text()
        assert '"tool": tool' in prompt_script.read_text()
        assert "rest_hook_response" in response_script.read_text()
        assert "rest_hook_replay" in session_start_script.read_text()
        assert "/api/consolidation/trigger" in session_end_script.read_text()

    def test_install_hooks_creates_scripts(self, tmp_path):
        """install_hooks creates the settings file with hook config."""
        from engram.setup import install_hooks

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()

        for name in _HOOK_NAMES:
            script = hooks_dir / name
            script.write_text("#!/bin/bash\nexit 0\n")
            script.chmod(0o755)

        settings_path = tmp_path / "settings.json"
        result = install_hooks(
            hooks_dir=hooks_dir,
            settings_path=settings_path,
        )

        assert result["settings_updated"] is True
        assert len(result["scripts"]) == 4
        assert settings_path.exists()
        assert (hooks_dir / "capture-prompt.sh").read_text() == "#!/bin/bash\nexit 0\n"

        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        for event in ("UserPromptSubmit", "Stop", "SessionStart", "SessionEnd"):
            assert event in settings["hooks"]
            # All events use matcher + hooks wrapper format
            entry = settings["hooks"][event][0]
            assert "matcher" in entry
            assert "hooks" in entry
            assert entry["hooks"][0]["type"] == "command"

    def test_hooks_config_merge(self, tmp_path):
        """install_hooks merges into existing settings."""
        from engram.setup import install_hooks

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        for name in _HOOK_NAMES:
            script = hooks_dir / name
            script.write_text("#!/bin/bash\nexit 0\n")
            script.chmod(0o755)

        settings_path = tmp_path / "settings.json"

        existing = {
            "permissions": {"allow": ["Bash(ls:*)"]},
            "hooks": {
                "UserPromptSubmit": [
                    {
                        "matcher": "",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "/usr/local/bin/my-hook.sh",
                            }
                        ],
                    }
                ]
            },
        }
        settings_path.write_text(json.dumps(existing))

        install_hooks(
            hooks_dir=hooks_dir,
            settings_path=settings_path,
        )

        settings = json.loads(settings_path.read_text())

        assert "Bash(ls:*)" in settings["permissions"]["allow"]

        # Existing hook preserved + Engram hook added
        entries = settings["hooks"]["UserPromptSubmit"]
        all_cmds = []
        for entry in entries:
            for h in entry.get("hooks", []):
                all_cmds.append(h.get("command", ""))

        assert "/usr/local/bin/my-hook.sh" in all_cmds
        engram_cmds = [c for c in all_cmds if "capture-prompt" in c]
        assert len(engram_cmds) == 1

    def test_hooks_idempotent(self, tmp_path):
        """Running install_hooks twice doesn't duplicate entries."""
        from engram.setup import install_hooks

        hooks_dir = tmp_path / "hooks"
        hooks_dir.mkdir()
        for name in _HOOK_NAMES:
            script = hooks_dir / name
            script.write_text("#!/bin/bash\nexit 0\n")
            script.chmod(0o755)

        settings_path = tmp_path / "settings.json"

        install_hooks(
            hooks_dir=hooks_dir,
            settings_path=settings_path,
        )
        install_hooks(
            hooks_dir=hooks_dir,
            settings_path=settings_path,
        )

        settings = json.loads(settings_path.read_text())
        assert len(settings["hooks"]["UserPromptSubmit"]) == 1
        assert len(settings["hooks"]["Stop"]) == 1


# ── Observe MCP Tool Message Test ───────────────────────────────────


def test_observe_response_message():
    """Observe tool response should use the simplified message."""
    from engram.ingestion.capture_surface import build_mcp_observe_write_surface
    from engram.mcp.server import observe

    observe_src = inspect.getsource(observe)
    surface_src = inspect.getsource(build_mcp_observe_write_surface)
    assert "build_mcp_observe_write_surface" in observe_src
    assert "Stored for background processing" in surface_src
    assert "Use trigger_consolidation" not in observe_src
    assert "Use trigger_consolidation" not in surface_src
