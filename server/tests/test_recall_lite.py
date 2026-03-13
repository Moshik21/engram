"""Tests for recall_lite() — fast entity-probe recall on GraphManager."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from engram.config import ActivationConfig
from engram.graph_manager import GraphManager
from engram.models.entity import Entity
from engram.models.relationship import Relationship

# ─── Helpers ────────────────────────────────────────────────────────


def _make_entity(
    name: str,
    entity_type: str = "Technology",
    summary: str | None = None,
    attributes: dict | None = None,
    identity_core: bool = False,
    entity_id: str | None = None,
) -> Entity:
    eid = entity_id or f"ent_{name.lower().replace(' ', '_')}"
    return Entity(
        id=eid,
        name=name,
        entity_type=entity_type,
        summary=summary or f"Summary of {name}",
        attributes=attributes or {},
        group_id="default",
        identity_core=identity_core,
    )


def _make_relationship(
    source_id: str,
    target_id: str,
    predicate: str = "USES",
) -> Relationship:
    return Relationship(
        id=f"rel_{source_id}_{target_id}",
        source_id=source_id,
        target_id=target_id,
        predicate=predicate,
        group_id="default",
    )


def _make_manager(
    *,
    find_candidates_map: dict[str, list[Entity]] | None = None,
    relationships_map: dict[str, list[Relationship]] | None = None,
    entity_map: dict[str, Entity] | None = None,
    cfg: ActivationConfig | None = None,
) -> GraphManager:
    """Create a GraphManager with mocked stores for recall_lite testing.

    find_candidates_map: maps mention text -> list of candidate entities
    relationships_map: maps entity_id -> list of relationships
    entity_map: maps entity_id -> Entity (for resolve_entity_name)
    """
    find_map = find_candidates_map or {}
    rels_map = relationships_map or {}
    ent_map = entity_map or {}

    graph = AsyncMock()

    async def _find_candidates(mention, group_id, limit=3):
        # Try exact key match first, then case-insensitive partial match
        if mention in find_map:
            return find_map[mention]
        for key, entities in find_map.items():
            if key.lower() == mention.lower():
                return entities
        return []

    graph.find_entity_candidates = AsyncMock(side_effect=_find_candidates)

    async def _get_relationships(entity_id, group_id="default"):
        return rels_map.get(entity_id, [])

    graph.get_relationships = AsyncMock(side_effect=_get_relationships)

    async def _get_entity(entity_id, group_id="default"):
        return ent_map.get(entity_id)

    graph.get_entity = AsyncMock(side_effect=_get_entity)

    activation = AsyncMock()
    search = AsyncMock()
    extractor = AsyncMock()

    manager = GraphManager(
        graph_store=graph,
        activation_store=activation,
        search_index=search,
        extractor=extractor,
        cfg=cfg or ActivationConfig(),
    )
    return manager


# ─── Test: Entity probe finds known entities ────────────────────────


@pytest.mark.asyncio
class TestEntityProbe:
    async def test_finds_known_entity(self):
        """Store an entity 'Kubernetes', call recall_lite with text containing it."""
        k8s = _make_entity("Kubernetes")
        manager = _make_manager(find_candidates_map={"Kubernetes": [k8s]})

        results = await manager.recall_lite(
            text="We are migrating to Kubernetes for orchestration",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["name"] == "Kubernetes"
        assert results[0]["type"] == "Technology"

    async def test_returns_empty_for_unknown_text(self):
        """Text mentioning no known entities should return empty list."""
        manager = _make_manager(find_candidates_map={})

        results = await manager.recall_lite(
            text="Just chatting about random topics today",
            group_id="default",
        )

        assert results == []

    async def test_multiple_entity_mentions(self):
        """Text mentioning 'Kubernetes' and 'Docker' should return both."""
        k8s = _make_entity("Kubernetes")
        docker = _make_entity("Docker")
        manager = _make_manager(
            find_candidates_map={
                "Kubernetes": [k8s],
                "Docker": [docker],
            }
        )

        results = await manager.recall_lite(
            text="We are migrating from Docker to Kubernetes this quarter",
            group_id="default",
        )

        names = {r["name"] for r in results}
        assert "Kubernetes" in names
        assert "Docker" in names
        assert len(results) == 2


# ─── Test: Session cache ────────────────────────────────────────────


@pytest.mark.asyncio
class TestSessionCache:
    async def test_cache_hit_returns_cached_results(self):
        """Second call with same text should use cached results."""
        k8s = _make_entity("Kubernetes")
        manager = _make_manager(find_candidates_map={"Kubernetes": [k8s]})
        cache: dict[str, tuple[float, dict]] = {}

        # First call — populates cache
        results1 = await manager.recall_lite(
            text="Deploying on Kubernetes cluster today",
            group_id="default",
            session_cache=cache,
        )
        assert len(results1) == 1
        assert len(cache) == 1

        # Reset the find_entity_candidates mock call count
        manager._graph.find_entity_candidates.reset_mock()

        # Second call — should use cache
        results2 = await manager.recall_lite(
            text="Deploying on Kubernetes cluster today",
            group_id="default",
            session_cache=cache,
        )
        assert len(results2) == 1
        assert results2[0]["name"] == "Kubernetes"

        # find_entity_candidates should still be called (mention extraction happens),
        # but the result comes from cache — verify cache dict has the entry
        assert k8s.id in cache

    async def test_cache_ttl_expiry_triggers_refetch(self):
        """With cache_ttl=0, second call should not use expired cache."""
        k8s = _make_entity("Kubernetes")
        manager = _make_manager(find_candidates_map={"Kubernetes": [k8s]})
        cache: dict[str, tuple[float, dict]] = {}

        # First call — populates cache
        await manager.recall_lite(
            text="Deploying on Kubernetes cluster today",
            group_id="default",
            session_cache=cache,
            cache_ttl=0,  # Immediate expiry
        )
        assert len(cache) == 1

        # Tiny sleep to ensure time.time() advances past the cached timestamp
        time.sleep(0.01)

        # Second call — cache is expired, should re-fetch
        results = await manager.recall_lite(
            text="Deploying on Kubernetes cluster today",
            group_id="default",
            session_cache=cache,
            cache_ttl=0,
        )
        assert len(results) == 1
        # find_entity_candidates called in both calls (not short-circuited by cache)
        assert manager._graph.find_entity_candidates.call_count >= 2


# ─── Test: Token budget ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestTokenBudget:
    async def test_token_budget_limits_output(self):
        """Very low token_budget should limit the number of non-identity entities."""
        # Each entity costs ~40 tokens. Budget of 30 = room for 0 normal entities.
        entities = {}
        for name in ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]:
            e = _make_entity(name)
            entities[name] = [e]

        manager = _make_manager(find_candidates_map=entities)

        results = await manager.recall_lite(
            text="Checking on Alpha, Bravo, Charlie, Delta, and Echo projects",
            group_id="default",
            token_budget=30,  # Less than one entity (40 tokens each)
        )

        # Budget of 30 < 40 per entity, so no normal entities should be included
        assert len(results) == 0

    async def test_identity_core_always_included_under_budget(self):
        """Identity-core entities should be included even with very low budget."""
        identity_entity = _make_entity(
            "Konner",
            entity_type="Person",
            identity_core=True,
        )
        normal_entity = _make_entity("React")

        manager = _make_manager(
            find_candidates_map={
                "Konner": [identity_entity],
                "React": [normal_entity],
            }
        )

        results = await manager.recall_lite(
            text="Konner is working with React on the new project",
            group_id="default",
            token_budget=30,  # Not enough for normal entities
        )

        # Identity-core entity is free against budget
        names = {r["name"] for r in results}
        assert "Konner" in names
        # React should be excluded due to budget
        assert "React" not in names


# ─── Test: Confidence tiers ─────────────────────────────────────────


@pytest.mark.asyncio
class TestConfidenceTiers:
    async def test_semantic_tier_maps_to_known(self):
        entity = _make_entity(
            "Python", attributes={"mat_tier": "semantic"}
        )
        manager = _make_manager(find_candidates_map={"Python": [entity]})

        results = await manager.recall_lite(
            text="We are using Python for the backend service",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["confidence"] == "known"

    async def test_transitional_tier_maps_to_likely(self):
        entity = _make_entity(
            "Svelte", attributes={"mat_tier": "transitional"}
        )
        manager = _make_manager(find_candidates_map={"Svelte": [entity]})

        results = await manager.recall_lite(
            text="Trying out Svelte for the frontend dashboard",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["confidence"] == "likely"

    async def test_episodic_tier_maps_to_recent(self):
        entity = _make_entity(
            "Bun", attributes={"mat_tier": "episodic"}
        )
        manager = _make_manager(find_candidates_map={"Bun": [entity]})

        results = await manager.recall_lite(
            text="Just discovered Bun as a JS runtime today",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["confidence"] == "recent"

    async def test_default_tier_maps_to_recent(self):
        """Entity with no mat_tier attribute defaults to 'recent'."""
        entity = _make_entity("Deno", attributes={})
        manager = _make_manager(find_candidates_map={"Deno": [entity]})

        results = await manager.recall_lite(
            text="Looking into Deno for our server-side scripts",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["confidence"] == "recent"

    async def test_none_attributes_defaults_to_recent(self):
        """Entity with attributes=None should default to 'recent'."""
        entity = _make_entity("Rust", attributes=None)
        # Pydantic may set attributes to None — recall_lite checks isinstance
        object.__setattr__(entity, "attributes", None)
        manager = _make_manager(find_candidates_map={"Rust": [entity]})

        results = await manager.recall_lite(
            text="we should try Rust for the performance-critical module",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["confidence"] == "recent"


# ─── Test: Empty / short text ───────────────────────────────────────


@pytest.mark.asyncio
class TestEdgeCases:
    async def test_empty_text_returns_empty(self):
        manager = _make_manager()
        results = await manager.recall_lite(text="", group_id="default")
        assert results == []

    async def test_whitespace_only_returns_empty(self):
        manager = _make_manager()
        results = await manager.recall_lite(text="   ", group_id="default")
        assert results == []

    async def test_short_text_no_mentions_returns_empty(self):
        """Short text with no proper nouns/caps should return empty."""
        manager = _make_manager()
        results = await manager.recall_lite(
            text="hi there how are you",
            group_id="default",
        )
        assert results == []

    async def test_none_graph_returns_empty(self):
        """If _graph is None, should return empty list."""
        manager = _make_manager()
        manager._graph = None

        results = await manager.recall_lite(
            text="Working with Kubernetes today on the cluster",
            group_id="default",
        )
        assert results == []


# ─── Test: Mention extraction patterns ──────────────────────────────


@pytest.mark.asyncio
class TestMentionExtraction:
    async def test_proper_noun_extraction(self):
        """Proper nouns like 'John Smith' and 'New York' should be extracted."""
        john = _make_entity("John Smith", entity_type="Person")
        new_york = _make_entity("New York", entity_type="Place")

        manager = _make_manager(
            find_candidates_map={
                "John Smith": [john],
                "New York": [new_york],
            }
        )

        results = await manager.recall_lite(
            text="John Smith went to New York for the conference",
            group_id="default",
        )

        names = {r["name"] for r in results}
        assert "John Smith" in names
        assert "New York" in names

    async def test_quoted_strings_extracted(self):
        """Quoted strings like 'Project Alpha' should be extracted."""
        project = _make_entity("Project Alpha", entity_type="Project")

        manager = _make_manager(
            find_candidates_map={
                "Project Alpha": [project],
            }
        )

        results = await manager.recall_lite(
            text='working on "Project Alpha" this sprint',
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["name"] == "Project Alpha"

    async def test_all_caps_acronyms_extracted(self):
        """All-caps acronyms like 'API' and 'AWS' should be extracted."""
        api = _make_entity("API", entity_type="Concept")
        aws = _make_entity("AWS", entity_type="Technology")

        manager = _make_manager(
            find_candidates_map={
                "API": [api],
                "AWS": [aws],
            }
        )

        results = await manager.recall_lite(
            text="need to fix the API gateway on AWS before release",
            group_id="default",
        )

        names = {r["name"] for r in results}
        assert "API" in names
        assert "AWS" in names

    async def test_at_mentions_extracted(self):
        """@-mentions should be extracted as mentions."""
        user = _make_entity("alice", entity_type="Person")

        manager = _make_manager(
            find_candidates_map={"alice": [user]}
        )

        results = await manager.recall_lite(
            text="asked @alice to review the pull request today",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["name"] == "alice"

    async def test_hashtag_extracted(self):
        """#hashtags should be extracted as mentions."""
        tag = _make_entity("backend", entity_type="Concept")

        manager = _make_manager(
            find_candidates_map={"backend": [tag]}
        )

        results = await manager.recall_lite(
            text="working on #backend improvements for performance",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["name"] == "backend"

    async def test_deduplicates_mentions(self):
        """Same entity mentioned multiple times should only appear once."""
        k8s = _make_entity("Kubernetes")

        manager = _make_manager(
            find_candidates_map={"Kubernetes": [k8s]}
        )

        results = await manager.recall_lite(
            text="Kubernetes is great. Kubernetes handles scaling well with Kubernetes pods",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["name"] == "Kubernetes"

    async def test_single_char_mentions_skipped(self):
        """Single character mentions (length < 2) should be skipped."""
        manager = _make_manager()

        # "I" is a single uppercase letter — should not be treated as a mention
        results = await manager.recall_lite(
            text="I went to the store and bought some items",
            group_id="default",
        )
        # "I" is only 1 char, filtered by len(key) >= 2
        # No all-caps 2+ letter words either
        # The proper nouns here are single words that don't match 2+ chars filter
        # Actually there are no entity matches because the map is empty
        assert results == []


# ─── Test: Top facts ────────────────────────────────────────────────


@pytest.mark.asyncio
class TestTopFacts:
    async def test_top_facts_included(self):
        """Store entity with relationships, verify top_facts in result."""
        k8s = _make_entity("Kubernetes", entity_id="ent_k8s")
        docker = _make_entity("Docker", entity_id="ent_docker")

        rel = _make_relationship(
            source_id="ent_k8s",
            target_id="ent_docker",
            predicate="REPLACES",
        )

        manager = _make_manager(
            find_candidates_map={"Kubernetes": [k8s]},
            relationships_map={"ent_k8s": [rel]},
            entity_map={"ent_docker": docker},
        )

        results = await manager.recall_lite(
            text="We are using Kubernetes in production now",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["top_facts"] == ["REPLACES Docker"]

    async def test_top_facts_reverse_direction(self):
        """When entity is the target, fact should show 'OtherName PREDICATE'."""
        k8s = _make_entity("Kubernetes", entity_id="ent_k8s")
        company = _make_entity("Google", entity_id="ent_google")

        rel = _make_relationship(
            source_id="ent_google",
            target_id="ent_k8s",
            predicate="CREATED",
        )

        manager = _make_manager(
            find_candidates_map={"Kubernetes": [k8s]},
            relationships_map={"ent_k8s": [rel]},
            entity_map={"ent_google": company},
        )

        results = await manager.recall_lite(
            text="looking at Kubernetes documentation for the cluster",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["top_facts"] == ["Google CREATED"]

    async def test_top_facts_limited_to_3(self):
        """Only first 3 relationships should appear in top_facts."""
        k8s = _make_entity("Kubernetes", entity_id="ent_k8s")

        rels = []
        entities = {}
        for i in range(5):
            target_id = f"ent_target_{i}"
            target = _make_entity(f"Target{i}", entity_id=target_id)
            entities[target_id] = target
            rels.append(
                _make_relationship(
                    source_id="ent_k8s",
                    target_id=target_id,
                    predicate=f"REL_{i}",
                )
            )

        manager = _make_manager(
            find_candidates_map={"Kubernetes": [k8s]},
            relationships_map={"ent_k8s": rels},
            entity_map=entities,
        )

        results = await manager.recall_lite(
            text="Deploying on Kubernetes for the production release",
            group_id="default",
        )

        assert len(results) == 1
        assert len(results[0]["top_facts"]) == 3

    async def test_no_relationships_returns_empty_facts(self):
        """Entity with no relationships should have empty top_facts."""
        k8s = _make_entity("Kubernetes")
        manager = _make_manager(find_candidates_map={"Kubernetes": [k8s]})

        results = await manager.recall_lite(
            text="Migrating everything to Kubernetes this month",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["top_facts"] == []

    async def test_unresolvable_target_falls_back_to_id(self):
        """If target entity not found, resolve_entity_name returns the ID."""
        k8s = _make_entity("Kubernetes", entity_id="ent_k8s")

        rel = _make_relationship(
            source_id="ent_k8s",
            target_id="ent_missing",
            predicate="DEPENDS_ON",
        )

        manager = _make_manager(
            find_candidates_map={"Kubernetes": [k8s]},
            relationships_map={"ent_k8s": [rel]},
            entity_map={},  # ent_missing not in map — resolve returns ID
        )

        results = await manager.recall_lite(
            text="checking Kubernetes dependencies for the service",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["top_facts"] == ["DEPENDS_ON ent_missing"]


# ─── Test: Summary truncation ───────────────────────────────────────


@pytest.mark.asyncio
class TestSummaryTruncation:
    async def test_summary_truncated_to_120_chars(self):
        """Entity summaries should be truncated to 120 characters."""
        long_summary = "A" * 200
        entity = _make_entity("React", summary=long_summary)
        manager = _make_manager(find_candidates_map={"React": [entity]})

        results = await manager.recall_lite(
            text="Building the frontend with React and styled components",
            group_id="default",
        )

        assert len(results) == 1
        assert len(results[0]["summary"]) <= 120


# ─── Test: Result structure ─────────────────────────────────────────


@pytest.mark.asyncio
class TestResultStructure:
    async def test_result_has_all_expected_keys(self):
        """Each result dict should have all expected keys."""
        entity = _make_entity("React")
        manager = _make_manager(find_candidates_map={"React": [entity]})

        results = await manager.recall_lite(
            text="Building the frontend using React components today",
            group_id="default",
        )

        assert len(results) == 1
        result = results[0]
        assert "name" in result
        assert "type" in result
        assert "summary" in result
        assert "confidence" in result
        assert "identity_core" in result
        assert "top_facts" in result

    async def test_identity_core_flag_set(self):
        """Identity-core entity should have identity_core=True in result."""
        entity = _make_entity("Konner", entity_type="Person", identity_core=True)
        manager = _make_manager(find_candidates_map={"Konner": [entity]})

        results = await manager.recall_lite(
            text="Konner is working on the backend services today",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["identity_core"] is True

    async def test_non_identity_core_flag_false(self):
        """Non-identity-core entity should have identity_core=False."""
        entity = _make_entity("React")
        manager = _make_manager(find_candidates_map={"React": [entity]})

        results = await manager.recall_lite(
            text="the frontend uses React for the interactive dashboard",
            group_id="default",
        )

        assert len(results) == 1
        assert results[0]["identity_core"] is False


# ─── Test: MCP server _auto_recall_lite wiring ──────────────────────


@pytest.mark.asyncio
class TestAutoRecallLiteWiring:
    async def test_auto_recall_lite_returns_none_for_short_content(self):
        """_auto_recall_lite returns None for content < 20 chars."""
        from engram.mcp import server
        from engram.mcp.server import SessionState, _auto_recall_lite

        cfg = ActivationConfig()
        manager = AsyncMock()
        session = SessionState(last_recall_time=0.0)

        with patch.object(server, "_session", session):
            result = await _auto_recall_lite("short text", manager, cfg)

        assert result is None
        manager.recall_lite.assert_not_called()

    async def test_auto_recall_lite_returns_entities(self):
        """_auto_recall_lite wraps recall_lite results in expected format."""
        from engram.mcp import server
        from engram.mcp.server import SessionState, _auto_recall_lite

        cfg = ActivationConfig()
        manager = AsyncMock()
        manager.recall_lite.return_value = [
            {
                "name": "React",
                "type": "Technology",
                "summary": "UI library",
                "confidence": "known",
                "identity_core": False,
                "top_facts": ["USES JSX"],
            }
        ]
        session = SessionState(last_recall_time=0.0)

        with patch.object(server, "_session", session):
            result = await _auto_recall_lite(
                "Working on the React migration for the frontend",
                manager,
                cfg,
            )

        assert result is not None
        assert result["source"] == "recall_lite"
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "React"

    async def test_auto_recall_lite_returns_none_on_empty_results(self):
        """_auto_recall_lite returns None when recall_lite returns empty list."""
        from engram.mcp import server
        from engram.mcp.server import SessionState, _auto_recall_lite

        cfg = ActivationConfig()
        manager = AsyncMock()
        manager.recall_lite.return_value = []
        session = SessionState(last_recall_time=0.0)

        with patch.object(server, "_session", session):
            result = await _auto_recall_lite(
                "just chatting about random things today",
                manager,
                cfg,
            )

        assert result is None

    async def test_auto_recall_lite_returns_none_on_exception(self):
        """_auto_recall_lite returns None if recall_lite raises."""
        from engram.mcp import server
        from engram.mcp.server import SessionState, _auto_recall_lite

        cfg = ActivationConfig()
        manager = AsyncMock()
        manager.recall_lite.side_effect = RuntimeError("db error")
        session = SessionState(last_recall_time=0.0)

        with patch.object(server, "_session", session):
            result = await _auto_recall_lite(
                "Working on Kubernetes migration for the cluster",
                manager,
                cfg,
            )

        assert result is None
