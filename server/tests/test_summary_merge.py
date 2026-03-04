"""Tests for meta-summary rejection in _merge_entity_attributes and _is_meta_summary."""

from __future__ import annotations

from engram.graph_manager import GraphManager
from engram.models.entity import Entity


def _make_entity(
    name: str = "TestEntity",
    entity_type: str = "Person",
    summary: str = "Original summary",
) -> Entity:
    return Entity(
        id="ent_test",
        name=name,
        entity_type=entity_type,
        summary=summary,
        group_id="default",
    )


class TestIsMetaSummary:
    """Tests for _is_meta_summary pattern detection."""

    def test_activation_score(self):
        assert GraphManager._is_meta_summary("Has activation score of 0.91")

    def test_access_count(self):
        assert GraphManager._is_meta_summary("Entity with access count 5")

    def test_knowledge_graph_node(self):
        assert GraphManager._is_meta_summary("A knowledge graph node in the store")

    def test_retrieval_mention(self):
        assert GraphManager._is_meta_summary("Found via retrieval with high relevance")

    def test_entity_resolution(self):
        assert GraphManager._is_meta_summary("Matched via entity resolution")

    def test_cold_session(self):
        assert GraphManager._is_meta_summary("Low activation in cold session")

    def test_system_id(self):
        assert GraphManager._is_meta_summary("Referenced as ent_abc123 in the store")

    def test_spreading_activation(self):
        assert GraphManager._is_meta_summary("Boosted by spreading activation")

    def test_mcp_tool(self):
        assert GraphManager._is_meta_summary("Accessed via MCP tool recall")

    def test_episode_worker(self):
        assert GraphManager._is_meta_summary("Processed by episode worker")

    def test_normal_summary_not_meta(self):
        assert not GraphManager._is_meta_summary("Data scientist at Acme Corp")

    def test_creative_work_not_meta(self):
        assert not GraphManager._is_meta_summary("A fantasy novel set in ancient times")

    def test_location_not_meta(self):
        assert not GraphManager._is_meta_summary("Located in San Francisco, CA")


class TestMergeEntityAttributes:
    """Tests for _merge_entity_attributes meta-summary rejection."""

    def test_rejects_meta_summary_for_person(self):
        """Meta-summary should be rejected for Person entities."""
        entity = _make_entity(name="Kallon", entity_type="Person")
        updates = GraphManager._merge_entity_attributes(
            entity, "Has activation score 0.91 in knowledge graph"
        )
        # Summary should NOT be updated
        assert "summary" not in updates

    def test_rejects_meta_summary_for_creative_work(self):
        """Meta-summary should be rejected for CreativeWork entities."""
        entity = _make_entity(
            name="The Wound Between Worlds", entity_type="CreativeWork"
        )
        updates = GraphManager._merge_entity_attributes(
            entity, "Used as test case for indirect retrieval"
        )
        assert "summary" not in updates

    def test_rejects_meta_summary_for_location(self):
        entity = _make_entity(name="Mesa", entity_type="Location")
        updates = GraphManager._merge_entity_attributes(
            entity, "Entity in the knowledge graph node with access count 3"
        )
        assert "summary" not in updates

    def test_rejects_meta_summary_for_event(self):
        entity = _make_entity(name="Team Meeting", entity_type="Event")
        updates = GraphManager._merge_entity_attributes(
            entity, "Consolidated during triage phase"
        )
        assert "summary" not in updates

    def test_rejects_meta_summary_for_organization(self):
        entity = _make_entity(name="Acme Corp", entity_type="Organization")
        updates = GraphManager._merge_entity_attributes(
            entity, "Found via retrieval pipeline with score 0.8"
        )
        assert "summary" not in updates

    def test_allows_meta_summary_for_technology(self):
        """Technology entities might legitimately discuss technical terms."""
        entity = _make_entity(
            name="Entity Resolution", entity_type="Technology"
        )
        updates = GraphManager._merge_entity_attributes(
            entity, "A technique for entity resolution in knowledge graphs"
        )
        # Should be allowed since Technology is not a protected type
        assert "summary" in updates

    def test_allows_meta_summary_for_concept(self):
        entity = _make_entity(name="Spreading Activation", entity_type="Concept")
        updates = GraphManager._merge_entity_attributes(
            entity, "Spreading activation is used in cognitive architectures"
        )
        assert "summary" in updates

    def test_allows_meta_summary_for_software(self):
        entity = _make_entity(name="Engram", entity_type="Software")
        updates = GraphManager._merge_entity_attributes(
            entity, "Uses retrieval pipeline for memory search"
        )
        assert "summary" in updates

    def test_allows_normal_summary_for_person(self):
        """Normal real-world summaries should always be accepted."""
        entity = _make_entity(name="Alice", entity_type="Person")
        updates = GraphManager._merge_entity_attributes(
            entity, "Works as a data scientist at Acme Corp"
        )
        assert "summary" in updates
        assert "data scientist" in updates["summary"]

    def test_allows_normal_summary_for_creative_work(self):
        entity = _make_entity(
            name="The Wound Between Worlds", entity_type="CreativeWork"
        )
        updates = GraphManager._merge_entity_attributes(
            entity, "A fantasy novel about interdimensional conflict"
        )
        assert "summary" in updates

    def test_pii_still_merged_despite_meta_summary_rejection(self):
        """PII flags should still be updated even if summary is rejected."""
        entity = _make_entity(name="Bob", entity_type="Person")
        updates = GraphManager._merge_entity_attributes(
            entity,
            "Has activation score 0.5 in the knowledge graph",
            new_pii=True,
            new_pii_categories=["name"],
        )
        # Summary rejected, but PII should still be flagged
        assert "summary" not in updates
        assert updates.get("pii_detected") == 1
