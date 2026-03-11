"""Tests for the discourse classifier — meta-commentary detection."""

from __future__ import annotations

from engram.extraction.discourse import classify_discourse


class TestSystemDiscourse:
    """Content that should be classified as 'system' (2+ pattern matches)."""

    def test_activation_score_with_entity_id(self):
        text = "Entity ent_abc123 has activation score 0.91"
        assert classify_discourse(text) == "system"

    def test_pipeline_terms_with_metrics(self):
        text = "The extraction pipeline gave activation score 0.5 for this entity"
        assert classify_discourse(text) == "system"

    def test_entity_resolution_and_graph_store(self):
        text = "Entity resolution matched against the graph store candidates"
        assert classify_discourse(text) == "system"

    def test_retrieval_pipeline_with_activation(self):
        text = "Retrieval pipeline returned activation score 0.3 for the entity"
        assert classify_discourse(text) == "system"

    def test_consolidation_and_triage(self):
        text = "Consolidation phase promoted ep_abc123 after triage scoring"
        assert classify_discourse(text) == "system"

    def test_mcp_tool_with_activation(self):
        text = "The MCP tool recall returned activation score 0.8 for Alice"
        assert classify_discourse(text) == "system"

    def test_meta_testing_with_system_ids(self):
        text = "Use rel_abc123 as test case for indirect retrieval"
        assert classify_discourse(text) == "system"

    def test_cold_session_with_access_count(self):
        text = "In a cold session the access count resets to zero"
        assert classify_discourse(text) == "system"

    def test_spreading_bonus_and_knowledge_graph(self):
        text = "The spreading bonus propagated through the knowledge graph node"
        assert classify_discourse(text) == "system"


class TestWorldDiscourse:
    """Content that should be classified as 'world' (0 pattern matches)."""

    def test_personal_fact(self):
        text = "Alice is a data scientist at Acme Corp"
        assert classify_discourse(text) == "world"

    def test_creative_work(self):
        text = "The Wound Between Worlds is a fantasy novel by Alex"
        assert classify_discourse(text) == "world"

    def test_location_info(self):
        text = "Alex lives in Mesa, Arizona and works remotely"
        assert classify_discourse(text) == "world"

    def test_project_discussion(self):
        text = "We decided to use React and TypeScript for the frontend"
        assert classify_discourse(text) == "world"

    def test_everyday_conversation(self):
        text = "I had a meeting with Bob about the Q3 roadmap"
        assert classify_discourse(text) == "world"

    def test_empty_string(self):
        assert classify_discourse("") == "world"

    def test_short_greeting(self):
        assert classify_discourse("hello") == "world"


class TestHybridDiscourse:
    """Content with exactly 1 system pattern match — mixed content."""

    def test_entity_name_with_activation(self):
        text = "Kallon has an activation score of 0.91 and lives in the forest"
        assert classify_discourse(text) == "hybrid"

    def test_single_system_id_in_context(self):
        text = "Alice created a project called ent_demo for the hackathon"
        assert classify_discourse(text) == "hybrid"

    def test_retrieval_mentioned_once(self):
        text = "Bob used the retrieval pipeline to find his old notes"
        assert classify_discourse(text) == "hybrid"


class TestEdgeCases:
    """Edge cases: legitimate tech discussions that could false-positive."""

    def test_user_discussing_knowledge_graphs_generally(self):
        """A user talking about knowledge graphs as a topic (not Engram internals)
        may trigger 'hybrid' — this is acceptable since it only blocks at 'system' level."""
        text = "I'm building a knowledge graph for my research project"
        result = classify_discourse(text)
        # Should be hybrid at most (1 match for "knowledge graph"), not system
        assert result in ("world", "hybrid")

    def test_user_discussing_embeddings_generally(self):
        text = "We're using embeddings to improve search quality in our app"
        result = classify_discourse(text)
        assert result in ("world", "hybrid")

    def test_pure_code_discussion(self):
        """Code snippets with variable names shouldn't false-positive unless
        they match system ID patterns."""
        text = "def process_data(entity_list, config): return sorted(entity_list)"
        assert classify_discourse(text) == "world"
