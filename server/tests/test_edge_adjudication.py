"""Tests for v3 edge adjudication flows."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from engram.config import ActivationConfig
from engram.consolidation.phases.edge_adjudication import EdgeAdjudicationPhase
from engram.graph_manager import GraphManager
from tests.conftest import MockExtractor


@pytest_asyncio.fixture
async def edge_manager(graph_store, activation_store, search_index) -> GraphManager:
    cfg = ActivationConfig(
        evidence_extraction_enabled=True,
        evidence_store_deferred=True,
        edge_adjudication_enabled=True,
        edge_adjudication_client_enabled=True,
        edge_adjudication_server_enabled=False,
    )
    return GraphManager(
        graph_store=graph_store,
        activation_store=activation_store,
        search_index=search_index,
        extractor=MockExtractor(),
        cfg=cfg,
    )


class TestEdgeAdjudicationHotPath:
    @pytest.mark.asyncio
    async def test_clean_remember_creates_no_requests(self, edge_manager, graph_store):
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google.",
            source="test",
        )

        requests = await edge_manager.get_episode_adjudications(
            episode_id,
            group_id="default",
        )
        evidence = await graph_store.get_episode_evidence(episode_id, group_id="default")

        assert requests == []
        assert evidence
        assert {row["status"] for row in evidence} == {"committed"}

    @pytest.mark.asyncio
    async def test_ambiguous_remember_creates_request_and_defers_relationship(
        self,
        edge_manager,
        graph_store,
    ):
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )

        requests = await edge_manager.get_episode_adjudications(
            episode_id,
            group_id="default",
        )
        evidence = await graph_store.get_episode_evidence(episode_id, group_id="default")
        relationship_rows = [
            row for row in evidence if row["fact_class"] == "relationship"
        ]

        assert len(requests) == 1
        assert requests[0]["ambiguity_tags"] == ["negation_scope"]
        assert len(requests[0]["candidate_evidence"]) == 1
        assert relationship_rows[0]["status"] == "pending"
        assert relationship_rows[0]["commit_reason"] == "needs_adjudication"
        assert relationship_rows[0]["adjudication_request_id"] == requests[0]["request_id"]
        assert all(
            row["status"] == "committed"
            for row in evidence
            if row["fact_class"] == "entity"
        )

    @pytest.mark.asyncio
    async def test_client_resolution_supersedes_original_and_materializes_replacement(
        self,
        edge_manager,
        graph_store,
    ):
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )
        request = (
            await edge_manager.get_episode_adjudications(episode_id, group_id="default")
        )[0]

        outcome = await edge_manager.submit_adjudication_resolution(
            request["request_id"],
            entities=[
                {"name": "Alice", "entity_type": "Person"},
                {"name": "Google", "entity_type": "Organization"},
            ],
            relationships=[
                {
                    "subject": "Alice",
                    "predicate": "WORKS_AT",
                    "object": "Google",
                    "polarity": "negative",
                },
            ],
            rationale="The clause indicates the employment fact is being reversed.",
            group_id="default",
        )

        evidence = await graph_store.get_episode_evidence(episode_id, group_id="default")
        original_relationship = next(
            row
            for row in evidence
            if row["fact_class"] == "relationship"
            and row["source_type"] == "narrow_extractor"
        )
        replacement_relationship = next(
            row
            for row in evidence
            if row["fact_class"] == "relationship"
            and row["source_type"] == "client_adjudication"
        )
        stored_request = await graph_store.get_adjudication_request(
            request["request_id"],
            group_id="default",
        )

        assert outcome.status == "materialized"
        assert outcome.committed_ids
        assert original_relationship["status"] == "superseded"
        assert replacement_relationship["status"] == "committed"
        assert replacement_relationship["adjudication_request_id"] == request["request_id"]
        assert stored_request is not None
        assert stored_request["status"] == "materialized"
        assert stored_request["resolution_source"] == "client_adjudication"

    @pytest.mark.asyncio
    async def test_reject_only_resolution_marks_original_terminal(
        self,
        edge_manager,
        graph_store,
    ):
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )
        request = (
            await edge_manager.get_episode_adjudications(episode_id, group_id="default")
        )[0]
        original_row = next(
            row
            for row in await graph_store.get_episode_evidence(
                episode_id,
                group_id="default",
            )
            if row["fact_class"] == "relationship"
            and row["source_type"] == "narrow_extractor"
        )

        outcome = await edge_manager.submit_adjudication_resolution(
            request["request_id"],
            reject_evidence_ids=[original_row["evidence_id"]],
            group_id="default",
        )

        updated = await graph_store.get_episode_evidence(episode_id, group_id="default")
        rejected = next(row for row in updated if row["evidence_id"] == original_row["evidence_id"])
        stored_request = await graph_store.get_adjudication_request(
            request["request_id"],
            group_id="default",
        )

        assert outcome.status == "rejected"
        assert rejected["status"] == "rejected"
        assert stored_request is not None
        assert stored_request["status"] == "rejected"


class TestEdgeAdjudicationPhase:
    @pytest.mark.asyncio
    async def test_phase_uses_shared_resolution_path(
        self,
        edge_manager,
        graph_store,
        search_index,
        activation_store,
    ):
        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_store_deferred=True,
            edge_adjudication_enabled=True,
            edge_adjudication_server_enabled=True,
            edge_adjudication_server_min_age_minutes=0,
            edge_adjudication_server_max_per_cycle=5,
            edge_adjudication_server_daily_budget=5,
        )
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )
        request = (
            await edge_manager.get_episode_adjudications(episode_id, group_id="default")
        )[0]
        phase = EdgeAdjudicationPhase(graph_manager=edge_manager)
        phase._call_server_adjudicator = AsyncMock(
            return_value={
                "entities": [
                    {"name": "Alice", "entity_type": "Person"},
                    {"name": "Google", "entity_type": "Organization"},
                ],
                "relationships": [
                    {
                        "subject": "Alice",
                        "predicate": "WORKS_AT",
                        "object": "Google",
                        "polarity": "negative",
                    },
                ],
                "reject_evidence_ids": [],
                "rationale": "server adjudicated",
            },
        )

        result, records = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_edge",
        )

        stored_request = await graph_store.get_adjudication_request(
            request["request_id"],
            group_id="default",
        )
        assert result.status == "success"
        assert records == []
        assert stored_request is not None
        assert stored_request["status"] == "materialized"
        assert stored_request["resolution_source"] == "server_adjudication"

    @pytest.mark.asyncio
    async def test_phase_respects_cycle_and_daily_budget(
        self,
        edge_manager,
        graph_store,
        search_index,
        activation_store,
    ):
        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_store_deferred=True,
            edge_adjudication_enabled=True,
            edge_adjudication_server_enabled=True,
            edge_adjudication_server_min_age_minutes=0,
            edge_adjudication_server_max_per_cycle=1,
            edge_adjudication_server_daily_budget=1,
        )
        phase = EdgeAdjudicationPhase(graph_manager=edge_manager)
        phase._daily_budget.clear()
        phase._call_server_adjudicator = AsyncMock(
            side_effect=[
                {
                    "entities": [
                        {"name": "Alice", "entity_type": "Person"},
                        {"name": "Google", "entity_type": "Organization"},
                    ],
                    "relationships": [
                        {
                            "subject": "Alice",
                            "predicate": "WORKS_AT",
                            "object": "Google",
                        },
                    ],
                    "reject_evidence_ids": [],
                },
            ],
        )

        episode_one = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )
        episode_two = await edge_manager.ingest_episode(
            content="Bob works at Anthropic, but maybe not anymore.",
            source="test",
        )

        first_result, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_budget_1",
        )
        second_result, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_budget_2",
        )

        first_requests = await edge_manager.get_episode_adjudications(
            episode_one,
            group_id="default",
        )
        second_requests = await edge_manager.get_episode_adjudications(
            episode_two,
            group_id="default",
        )
        stored_one = await graph_store.get_episode_adjudications(
            episode_one,
            group_id="default",
        )
        stored_two = await graph_store.get_episode_adjudications(
            episode_two,
            group_id="default",
        )

        assert first_result.status == "success"
        assert second_result.status == "success"
        assert phase._call_server_adjudicator.await_count == 1
        assert stored_one[0]["status"] == "materialized"
        assert stored_two[0]["status"] == "pending"
        assert first_requests == []
        assert len(second_requests) == 1

    @pytest.mark.asyncio
    async def test_phase_expires_stale_request(
        self,
        edge_manager,
        graph_store,
        search_index,
        activation_store,
    ):
        cfg = ActivationConfig(
            evidence_extraction_enabled=True,
            evidence_store_deferred=True,
            edge_adjudication_enabled=True,
            edge_adjudication_server_enabled=False,
            edge_adjudication_request_ttl_hours=1,
        )
        episode_id = await edge_manager.ingest_episode(
            content="Alice works at Google, but maybe not anymore.",
            source="test",
        )
        request = (
            await edge_manager.get_episode_adjudications(episode_id, group_id="default")
        )[0]
        await graph_store.db.execute(
            "UPDATE episode_adjudications SET created_at = ? WHERE request_id = ?",
            ("2026-03-01T00:00:00", request["request_id"]),
        )
        await graph_store.db.commit()

        phase = EdgeAdjudicationPhase(graph_manager=edge_manager)
        result, _ = await phase.execute(
            group_id="default",
            graph_store=graph_store,
            activation_store=activation_store,
            search_index=search_index,
            cfg=cfg,
            cycle_id="cyc_expire",
        )

        stored_request = await graph_store.get_adjudication_request(
            request["request_id"],
            group_id="default",
        )
        evidence = await graph_store.get_episode_evidence(episode_id, group_id="default")
        request_rows = [
            row
            for row in evidence
            if row.get("adjudication_request_id") == request["request_id"]
        ]

        assert result.status == "success"
        assert stored_request is not None
        assert stored_request["status"] == "expired"
        assert request_rows
        assert {row["status"] for row in request_rows} == {"expired"}
