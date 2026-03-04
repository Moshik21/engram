"""Tests for query-type routing."""

from __future__ import annotations

import pytest

from engram.config import ActivationConfig
from engram.retrieval.router import QueryType, apply_route, classify_query


@pytest.mark.asyncio
class TestClassifyQuery:
    async def test_temporal_recent(self):
        """'recently' keyword triggers TEMPORAL."""
        qt = await classify_query("What have I been working on recently?")
        assert qt == QueryType.TEMPORAL

    async def test_temporal_last(self):
        """'last' keyword triggers TEMPORAL."""
        qt = await classify_query("What was discussed last week?")
        assert qt == QueryType.TEMPORAL

    async def test_temporal_today(self):
        """'today' keyword triggers TEMPORAL."""
        qt = await classify_query("What did I do today?")
        assert qt == QueryType.TEMPORAL

    async def test_temporal_lately(self):
        """'lately' keyword triggers TEMPORAL."""
        qt = await classify_query("What topics came up lately?")
        assert qt == QueryType.TEMPORAL

    async def test_associative_connection(self):
        """'connection' keyword triggers ASSOCIATIVE."""
        qt = await classify_query("What is the connection between Alice and Bob?")
        assert qt == QueryType.ASSOCIATIVE

    async def test_associative_link(self):
        """'link' keyword triggers ASSOCIATIVE."""
        qt = await classify_query("How are these projects linked?")
        assert qt == QueryType.ASSOCIATIVE

    async def test_associative_between(self):
        """'between' keyword triggers ASSOCIATIVE."""
        qt = await classify_query("What relation between Python and React?")
        assert qt == QueryType.ASSOCIATIVE

    async def test_associative_bridge(self):
        """'bridge' keyword triggers ASSOCIATIVE."""
        qt = await classify_query("Who is the bridge between the two teams?")
        assert qt == QueryType.ASSOCIATIVE

    async def test_direct_lookup_high_score(self):
        """High top-1 search score triggers DIRECT_LOOKUP."""
        qt = await classify_query(
            "Tell me about Alice",
            search_results=[("alice_id", 0.95), ("bob_id", 0.3)],
        )
        assert qt == QueryType.DIRECT_LOOKUP

    async def test_default_no_keywords(self):
        """Generic query with no keywords returns DEFAULT."""
        qt = await classify_query("Tell me about machine learning")
        assert qt == QueryType.DEFAULT

    async def test_default_low_search_score(self):
        """Low top-1 search score doesn't trigger DIRECT_LOOKUP."""
        qt = await classify_query(
            "Tell me about something",
            search_results=[("eid", 0.5)],
        )
        assert qt == QueryType.DEFAULT

    async def test_temporal_takes_priority_over_direct(self):
        """Temporal keywords take priority even with high search score."""
        qt = await classify_query(
            "What was recently discussed?",
            search_results=[("eid", 0.95)],
        )
        assert qt == QueryType.TEMPORAL

    async def test_frequency_most_important(self):
        """'most important' triggers FREQUENCY."""
        qt = await classify_query("What are my most important topics?")
        assert qt == QueryType.FREQUENCY

    async def test_frequency_focus(self):
        """'focus' triggers FREQUENCY."""
        qt = await classify_query("What do I focus on the most?")
        assert qt == QueryType.FREQUENCY

    async def test_frequency_frequently_accessed(self):
        """'frequently' triggers FREQUENCY."""
        qt = await classify_query("Show me my most frequently accessed items")
        assert qt == QueryType.FREQUENCY

    async def test_frequency_top_areas(self):
        """'top' triggers FREQUENCY."""
        qt = await classify_query("My top focus areas")
        assert qt == QueryType.FREQUENCY

    async def test_temporal_whats_new(self):
        """'what's new' triggers TEMPORAL."""
        qt = await classify_query("What's new in service mesh?")
        assert qt == QueryType.TEMPORAL

    async def test_temporal_takes_priority_over_frequency(self):
        """Temporal keywords take priority over frequency keywords."""
        qt = await classify_query("What have I focused on recently?")
        assert qt == QueryType.TEMPORAL

    async def test_creation_wrote(self):
        """'written' keyword triggers CREATION."""
        qt = await classify_query("books written by Konner")
        assert qt == QueryType.CREATION

    async def test_creation_authored(self):
        """'authored' keyword triggers CREATION."""
        qt = await classify_query("papers authored by Alice")
        assert qt == QueryType.CREATION

    async def test_creation_built(self):
        """'built' keyword triggers CREATION."""
        qt = await classify_query("apps built by team")
        assert qt == QueryType.CREATION

    async def test_creation_published_with_temporal(self):
        """Temporal takes priority over CREATION when both present."""
        qt = await classify_query("articles published last year")
        assert qt == QueryType.TEMPORAL

    async def test_creation_priority_after_frequency(self):
        """CREATION checked after FREQUENCY, before ASSOCIATIVE."""
        qt = await classify_query("things created between teams")
        # "created" matches CREATION, "between" matches ASSOCIATIVE
        # CREATION is checked first
        assert qt == QueryType.CREATION


class TestApplyRoute:
    def test_direct_lookup_weights(self):
        """DIRECT_LOOKUP overrides to 0.75/0.10/0.05/0.10."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.DIRECT_LOOKUP, cfg)
        assert routed.weight_semantic == 0.75
        assert routed.weight_activation == 0.10
        assert routed.weight_spreading == 0.05
        assert routed.weight_edge_proximity == 0.10

    def test_temporal_weights(self):
        """TEMPORAL overrides to 0.20/0.55/0.15/0.10."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.TEMPORAL, cfg)
        assert routed.weight_semantic == 0.20
        assert routed.weight_activation == 0.55
        assert routed.weight_spreading == 0.15
        assert routed.weight_edge_proximity == 0.10

    def test_associative_weights(self):
        """ASSOCIATIVE overrides to 0.55/0.10/0.20/0.15."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.ASSOCIATIVE, cfg)
        assert routed.weight_semantic == 0.55
        assert routed.weight_activation == 0.10
        assert routed.weight_spreading == 0.20
        assert routed.weight_edge_proximity == 0.15

    def test_default_weights(self):
        """DEFAULT overrides to 0.40/0.25/0.15/0.15."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.DEFAULT, cfg)
        assert routed.weight_semantic == 0.40
        assert routed.weight_activation == 0.25
        assert routed.weight_spreading == 0.15
        assert routed.weight_edge_proximity == 0.15

    def test_does_not_mutate_original(self):
        """apply_route returns a copy, does not modify original."""
        cfg = ActivationConfig(
            weight_semantic=0.50,
            weight_activation=0.50,
            weight_edge_proximity=0.00,
        )
        routed = apply_route(QueryType.TEMPORAL, cfg)
        assert cfg.weight_semantic == 0.50  # original unchanged
        assert routed.weight_semantic == 0.20  # copy changed

    def test_frequency_weights(self):
        """FREQUENCY overrides to 0.15/0.60/0.15/0.10."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.FREQUENCY, cfg)
        assert routed.weight_semantic == 0.15
        assert routed.weight_activation == 0.60
        assert routed.weight_spreading == 0.15
        assert routed.weight_edge_proximity == 0.10

    def test_creation_weights(self):
        """CREATION overrides to 0.30/0.10/0.25/0.30."""
        cfg = ActivationConfig()
        routed = apply_route(QueryType.CREATION, cfg)
        assert routed.weight_semantic == 0.30
        assert routed.weight_activation == 0.10
        assert routed.weight_spreading == 0.25
        assert routed.weight_edge_proximity == 0.30

    def test_preserves_other_fields(self):
        """Non-weight fields are preserved from original config."""
        cfg = ActivationConfig(
            decay_exponent=0.7,
            retrieval_top_k=100,
            exploration_weight=0.10,
        )
        routed = apply_route(QueryType.DEFAULT, cfg)
        assert routed.decay_exponent == 0.7
        assert routed.retrieval_top_k == 100
        assert routed.exploration_weight == 0.10
