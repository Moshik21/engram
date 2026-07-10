"""Regression tests for Cluster A5 silent-inert fixes (B6, B7, B10, B14, B15).

B6  — prospective-memory writes (fire/dismiss) must persist on lite/SQLite
       (update_entity needs json.dumps(attrs), not a raw dict).
B7  — template query reformulation must emit clean statements, not garbled
       whitespace-spliced fragments.
B10 — the MCP intend response must flag that firing is inert when
       prospective_memory_enabled is False.
B14 — Helix get_entity_embeddings must warn on partial recovery (logic-level
       assertions only; the Helix path itself is exercised by the orchestrator).
B15 — get_context(format="briefing") must not silently swap to "structured";
       it must flag degradation, and template_briefing must harvest the
       Cached Memory Packets section.
"""

from __future__ import annotations

import pytest

# ───────────────────────── B7: query reformulation ──────────────────────────


class TestReformulateQueryB7:
    def test_documented_patterns_are_clean(self):
        from engram.retrieval.graph_expansion import reformulate_query

        cases = {
            "What is my favorite language?": "my favorite language is",
            "What is my car?": "my car is",
            "What languages do I like?": "I like languages",
            "Do I like coffee?": "I like coffee",
            "Where do I work?": "I work at",
            "When did I start?": "I start on",
            "Who is my manager?": "my manager is",
            "How many cars do I have?": "I have cars",
        }
        for query, expected in cases.items():
            assert reformulate_query(query) == expected

    def test_no_whitespace_spliced_fragments(self):
        """The original bug spliced words: 'my f isavorite language?'."""
        from engram.retrieval.graph_expansion import reformulate_query

        result = reformulate_query("What is my favorite language?")
        assert result is not None
        # No garbled fragment, and every token is a normal word.
        assert "isavorite" not in result
        assert all(len(tok) <= 30 for tok in result.split())
        # Words are space-separated and intact.
        assert result.split() == ["my", "favorite", "language", "is"]

    def test_non_matching_query_returns_none(self):
        from engram.retrieval.graph_expansion import reformulate_query

        assert reformulate_query("Random unrelated sentence") is None

    def test_clean_reformulation_guard_rejects_splice(self):
        from engram.retrieval.graph_expansion import _is_clean_reformulation

        assert _is_clean_reformulation("my favorite language is") is True
        assert _is_clean_reformulation("my f isavoritelanguageisaveryverylongtoken") is False
        assert _is_clean_reformulation("") is False


# ─────────────────── B6: lite prospective write persistence ──────────────────


class _StubSearch:
    async def index_entity(self, entity):  # pragma: no cover - trivial
        return None


class _StubActivation:
    async def record_access(self, *args, **kwargs):  # pragma: no cover - trivial
        return None


def _noop_publish(group_id, event, payload):  # pragma: no cover - trivial
    return None


async def _make_lite_service(tmp_path):
    from engram.config import ActivationConfig
    from engram.retrieval.prospective import ProspectiveMemoryService
    from engram.storage.sqlite.graph import SQLiteGraphStore

    store = SQLiteGraphStore(str(tmp_path / "a5_prospective.db"))
    await store.initialize()
    cfg = ActivationConfig(
        prospective_memory_enabled=True,
        prospective_graph_embedded=True,
    )
    service = ProspectiveMemoryService(
        graph_store=store,
        activation_store=_StubActivation(),
        search_index=_StubSearch(),
        cfg=cfg,
        publish_event=_noop_publish,
    )
    return store, service


@pytest.mark.asyncio
async def test_b6_fire_increments_count_on_lite(tmp_path):
    """A matched intention's fire write must persist on SQLite (no dict-bind error)."""
    store, service = await _make_lite_service(tmp_path)
    try:
        gid = "a5_fire"
        intention_id = await service.create_intention(
            trigger_text="ship the release",
            action_text="run the checklist",
            trigger_type="activation",
            group_id=gid,
        )

        # This used to raise 'type dict is not supported' on SQLite.
        await service.update_intention_fire(intention_id, gid, episode_id="ep_1")

        entity = await store.get_entity(intention_id, gid)
        assert entity is not None
        assert entity.attributes["fire_count"] == 1
        assert entity.attributes["last_fired"] is not None

        # Second fire keeps counting.
        await service.update_intention_fire(intention_id, gid, episode_id="ep_2")
        entity = await store.get_entity(intention_id, gid)
        assert entity.attributes["fire_count"] == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_b6_soft_dismiss_disables_on_lite(tmp_path):
    """Soft dismiss must persist enabled=False on SQLite (no dict-bind error)."""
    store, service = await _make_lite_service(tmp_path)
    try:
        gid = "a5_dismiss"
        intention_id = await service.create_intention(
            trigger_text="follow up with Alex",
            action_text="send the email",
            trigger_type="activation",
            group_id=gid,
        )
        entity = await store.get_entity(intention_id, gid)
        assert entity.attributes["enabled"] is True

        await service.dismiss_intention(intention_id, gid, hard=False)

        entity = await store.get_entity(intention_id, gid)
        assert entity is not None
        assert entity.attributes["enabled"] is False
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_b6_update_meta_persists_on_lite(tmp_path):
    """update_intention_meta must persist arbitrary fields on SQLite."""
    store, service = await _make_lite_service(tmp_path)
    try:
        gid = "a5_meta"
        intention_id = await service.create_intention(
            trigger_text="refresh the dashboard",
            action_text="pin the latest context",
            trigger_type="activation",
            group_id=gid,
        )
        await service.update_intention_meta(
            intention_id, gid, {"last_refreshed": "2026-05-28T00:00:00+00:00"}
        )
        entity = await store.get_entity(intention_id, gid)
        assert entity.attributes["last_refreshed"] == "2026-05-28T00:00:00+00:00"
    finally:
        await store.close()


# ─────────────── B10: intend response flags inert firing ─────────────────────


class _FiringManager:
    """Minimal manager exposing the prospective service + threshold delegation."""

    def __init__(self, service):
        self._prospective_memory_service = service

    async def create_intention(self, **kwargs):
        return "int_stub_b10"

    def effective_intention_threshold(self, threshold=None):
        return self._prospective_memory_service.effective_activation_threshold(threshold)


async def _make_manager(tmp_path, *, firing_enabled: bool):
    from engram.config import ActivationConfig
    from engram.retrieval.prospective import ProspectiveMemoryService
    from engram.storage.sqlite.graph import SQLiteGraphStore

    store = SQLiteGraphStore(str(tmp_path / "a5_b10.db"))
    await store.initialize()
    cfg = ActivationConfig(
        prospective_memory_enabled=firing_enabled,
        prospective_graph_embedded=True,
    )
    service = ProspectiveMemoryService(
        graph_store=store,
        activation_store=_StubActivation(),
        search_index=_StubSearch(),
        cfg=cfg,
        publish_event=_noop_publish,
    )
    return store, _FiringManager(service)


@pytest.mark.asyncio
async def test_b10_intend_flags_inert_when_firing_disabled(tmp_path):
    from engram.retrieval.prospective import build_mcp_create_intention_surface

    store, manager = await _make_manager(tmp_path, firing_enabled=False)
    try:
        payload = await build_mcp_create_intention_surface(
            manager,
            group_id="default",
            trigger_text="ping me later",
            action_text="do the thing",
            trigger_type="activation",
            entity_names=None,
            threshold=None,
            priority="normal",
            context=None,
            see_also=None,
            refresh_trigger="manual",
        )
        # Status stays "created" for back-compat, but the inert state is surfaced.
        assert payload["status"] == "created"
        assert payload["firing_enabled"] is False
        assert "created_but_inert" in payload["status_note"]
        assert "prospective_memory_enabled" in payload["message"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_b10_intend_not_flagged_when_firing_enabled(tmp_path):
    from engram.retrieval.prospective import build_mcp_create_intention_surface

    store, manager = await _make_manager(tmp_path, firing_enabled=True)
    try:
        payload = await build_mcp_create_intention_surface(
            manager,
            group_id="default",
            trigger_text="ping me later",
            action_text="do the thing",
            trigger_type="activation",
            entity_names=None,
            threshold=None,
            priority="normal",
            context=None,
            see_also=None,
            refresh_trigger="manual",
        )
        assert payload["status"] == "created"
        assert "firing_enabled" not in payload
        assert "status_note" not in payload
    finally:
        await store.close()


def test_b10_firing_enabled_flag_reflects_config():
    from engram.config import ActivationConfig
    from engram.retrieval.prospective import ProspectiveMemoryService

    on = ProspectiveMemoryService(
        graph_store=None,
        activation_store=None,
        search_index=None,
        cfg=ActivationConfig(prospective_memory_enabled=True),
        publish_event=_noop_publish,
    )
    off = ProspectiveMemoryService(
        graph_store=None,
        activation_store=None,
        search_index=None,
        cfg=ActivationConfig(prospective_memory_enabled=False),
        publish_event=_noop_publish,
    )
    assert on.firing_enabled() is True
    assert off.firing_enabled() is False


# ─────────────── B15: briefing degradation + cached packet tier ──────────────


def test_b15_template_briefing_harvests_cached_packets():
    """template_briefing must include the Cached Memory Packets section."""
    from engram.config import ActivationConfig
    from engram.retrieval.context_builder import MemoryContextBuilder

    builder = MemoryContextBuilder.__new__(MemoryContextBuilder)
    builder._cfg = ActivationConfig()
    builder._briefing_cache = {}

    context = (
        "## Cached Memory Packets\n"
        "\n"
        "- [cache/fresh] Project Home: Engram — persistent memory layer\n"
        "\n"
        "## Identity\n"
        "\n"
        "- Konner (Person, identity core)\n"
    )
    briefing = builder.template_briefing(context, "default", None)
    assert "Cached memory:" in briefing
    assert "Project Home: Engram" in briefing
    assert "Known context:" in briefing


# B15 degradation behavior of get_context is exercised through the lite recall
# pipeline below; the pure briefing-degraded branch is asserted via reading the
# returned dict keys.


async def _empty_recall(*args, **kwargs):
    return []


async def _empty_list_intentions(*args, **kwargs):
    return []


async def _resolve_name(entity_id, group_id):
    return entity_id


async def _publish_access(*args, **kwargs):
    return None


@pytest.mark.asyncio
async def test_b15_briefing_degraded_flag_when_no_entities(tmp_path):
    """A briefing request with no activated entities must flag degradation,
    not silently swap to 'structured'."""
    from engram.config import ActivationConfig
    from engram.retrieval.context_builder import MemoryContextBuilder
    from engram.storage.memory.activation import MemoryActivationStore
    from engram.storage.sqlite.graph import SQLiteGraphStore

    graph = SQLiteGraphStore(str(tmp_path / "a5_b15_graph.db"))
    await graph.initialize()
    activation = MemoryActivationStore(cfg=ActivationConfig())
    cfg = ActivationConfig(briefing_enabled=True)
    builder = MemoryContextBuilder(
        graph_store=graph,
        activation_store=activation,
        cfg=cfg,
        recall=_empty_recall,
        list_intentions=_empty_list_intentions,
        resolve_entity_name=_resolve_name,
        publish_access_event=_publish_access,
        get_cached_packets=None,  # no cached packets -> nothing to brief on
    )
    try:
        result = await builder.get_context(group_id="empty_a5", format="briefing")
        assert result["entity_count"] == 0
        # Must not silently masquerade as a populated structured result.
        assert result.get("briefing_degraded") is True
        assert result["briefing_degraded_reason"] == "no_briefable_content"
    finally:
        await graph.close()


# ─────────────── B14: exact helix entity-vector recovery ───────────────


def test_b14_exact_recovery_warns_on_missing_vectors(monkeypatch, caplog):
    """get_entity_embeddings must use exact metadata lookup and warn on misses."""
    import logging

    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    index = HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969),
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=4,
        embed_provider="noop",
        embed_model="noop",
    )
    index._embeddings_enabled = True
    index._storage_dim = 4
    calls: list[tuple[str, dict]] = []

    async def _fake_query(name, params):
        calls.append((name, params))
        assert name == "find_entity_vectors_by_ids"
        assert set(params["entity_ids"]) == {"ent_a", "ent_b"}
        assert params["gid"] == "g"
        return [
            {"entity_id": "ent_a", "data": [0.1, 0.2, 0.3, 0.4], "group_id": "g"},
        ]

    monkeypatch.setattr(index, "_query", _fake_query)

    import asyncio

    with caplog.at_level(logging.WARNING, logger="engram.storage.helix.search"):
        result = asyncio.run(index.get_entity_embeddings(["ent_a", "ent_b"], group_id="g"))

    assert "ent_a" in result
    assert "ent_b" not in result
    assert len(calls) == 1
    assert any(
        "exact lookup recovered 1/2 entity vectors" in rec.getMessage() for rec in caplog.records
    ), "expected a WARNING about partial embedding recovery"


def test_b14_exact_full_recovery_no_warning(monkeypatch, caplog):
    import logging

    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    index = HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969),
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=4,
        embed_provider="noop",
        embed_model="noop",
    )
    index._embeddings_enabled = True
    index._storage_dim = 4
    calls: list[tuple[str, dict]] = []

    async def _fake_query(name, params):
        calls.append((name, params))
        assert name == "find_entity_vectors_by_ids"
        assert set(params["entity_ids"]) == {"ent_a", "ent_b"}
        return [
            {"entity_id": "ent_a", "vec": [0.1, 0.2, 0.3, 0.4], "group_id": "g"},
            {"entity_id": "ent_b", "vec": [0.5, 0.6, 0.7, 0.8], "group_id": "g"},
        ]

    monkeypatch.setattr(index, "_query", _fake_query)

    import asyncio

    with caplog.at_level(logging.WARNING, logger="engram.storage.helix.search"):
        result = asyncio.run(index.get_entity_embeddings(["ent_a", "ent_b"], group_id="g"))

    assert set(result) == {"ent_a", "ent_b"}
    assert len(calls) == 1
    assert not any("recovered" in rec.getMessage() for rec in caplog.records)


def test_b14_exact_recovery_uses_all_groups_endpoint(monkeypatch):
    from engram.config import EmbeddingConfig, HelixDBConfig
    from engram.embeddings.provider import NoopProvider
    from engram.storage.helix.search import HelixSearchIndex

    index = HelixSearchIndex(
        helix_config=HelixDBConfig(host="localhost", port=6969),
        provider=NoopProvider(),
        embed_config=EmbeddingConfig(),
        storage_dim=4,
        embed_provider="noop",
        embed_model="noop",
    )
    index._embeddings_enabled = True
    calls: list[tuple[str, dict]] = []

    async def _fake_query(name, params):
        calls.append((name, params))
        assert name == "find_entity_vectors_by_ids_all"
        assert set(params["entity_ids"]) == {"ent_a"}
        return [{"entity_id": "ent_a", "vec": [0.1, 0.2, 0.3, 0.4]}]

    monkeypatch.setattr(index, "_query", _fake_query)

    import asyncio

    result = asyncio.run(index.get_entity_embeddings(["ent_a"]))

    assert result == {"ent_a": [0.1, 0.2, 0.3, 0.4]}
    assert len(calls) == 1
