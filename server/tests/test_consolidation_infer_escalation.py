"""Tests for Sonnet escalation in the edge inference phase."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from engram.config import ActivationConfig
from engram.consolidation.phases.infer import EdgeInferencePhase
from engram.models.entity import Entity


def _entity(name, entity_type="Person", group_id="test"):
    import uuid
    return Entity(
        id=f"ent_{uuid.uuid4().hex[:8]}",
        name=name,
        entity_type=entity_type,
        group_id=group_id,
    )


def _make_llm_response(verdict: str, reason: str = "test"):
    content_block = MagicMock()
    content_block.text = json.dumps({"verdict": verdict, "reason": reason})
    response = MagicMock()
    response.content = [content_block]
    return response


def _mock_graph_store(pairs, entities):
    gs = AsyncMock()
    gs.get_co_occurring_entity_pairs.return_value = pairs
    gs.get_entity_episode_counts.return_value = {}
    gs.get_stats.return_value = {"total_episodes": 0}
    gs.get_relationships_by_predicate.return_value = []

    async def _get_entity(eid, gid):
        return entities.get(eid)

    gs.get_entity.side_effect = _get_entity
    gs.create_relationship.return_value = "rel_mock"
    gs.invalidate_relationship = AsyncMock()
    return gs


class TestInferEscalation:
    """Tests for Sonnet escalation of uncertain edge verdicts."""

    @pytest.mark.asyncio
    async def test_escalation_disabled_no_effect(self):
        """Escalation disabled leaves uncertain edges unchanged."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        # Haiku returns uncertain
        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=False,
        )
        phase = EdgeInferencePhase(llm_client=haiku_client)
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert len(records) == 1
        assert records[0].llm_verdict == "uncertain"
        assert records[0].escalation_verdict is None

    @pytest.mark.asyncio
    async def test_escalation_approves_uncertain(self):
        """Sonnet approves an uncertain edge."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        # Haiku returns uncertain, Sonnet returns approved
        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert len(records) == 1
        assert records[0].escalation_verdict == "approved"
        assert records[0].infer_type == "escalation_approved"
        sonnet_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalation_rejects_uncertain(self):
        """Sonnet rejects an uncertain edge and invalidates relationship."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.return_value = _make_llm_response("rejected")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert len(records) == 1
        assert records[0].escalation_verdict == "rejected"
        assert records[0].infer_type == "escalation_rejected"
        gs.invalidate_relationship.assert_called_once()

    @pytest.mark.asyncio
    async def test_escalation_dry_run(self):
        """Escalation dry run: LLM validation skips, no uncertain candidates."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        sonnet_client = MagicMock()

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=True,
        )

        # LLM validation sets dry_run_skipped, not uncertain
        assert records[0].llm_verdict == "dry_run_skipped"
        # No uncertain records → escalation has no work → verdict stays None
        assert records[0].escalation_verdict is None
        # Neither Haiku nor Sonnet called in dry run
        haiku_client.messages.create.assert_not_called()
        sonnet_client.messages.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_escalation_max_per_cycle(self):
        """At most N uncertain edges escalated."""
        entities = {}
        pairs = []
        for i in range(5):
            e1 = _entity(f"Entity{i}A")
            e2 = _entity(f"Entity{i}B")
            entities[e1.id] = e1
            entities[e2.id] = e2
            pairs.append((e1.id, e2.id, 5))

        gs = _mock_graph_store(pairs, entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
            consolidation_infer_escalation_max_per_cycle=2,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        escalated = [r for r in records if r.escalation_verdict is not None]
        assert len(escalated) == 2
        assert sonnet_client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_escalation_error_nonfatal(self):
        """Escalation API error caught, sets error verdict."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.side_effect = RuntimeError("Sonnet down")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        result, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert result.status == "success"
        assert records[0].escalation_verdict == "error"

    @pytest.mark.asyncio
    async def test_escalation_uses_correct_model(self):
        """Escalation uses the configured escalation model."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
            consolidation_infer_escalation_model="claude-sonnet-4-6-20250514",
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        call_kwargs = sonnet_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6-20250514"

    @pytest.mark.asyncio
    async def test_escalation_uses_cached_system_prompt(self):
        """Escalation uses cached system prompt."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("uncertain")

        sonnet_client = MagicMock()
        sonnet_client.messages.create.return_value = _make_llm_response("approved")

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, _ = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        call_kwargs = sonnet_client.messages.create.call_args[1]
        assert isinstance(call_kwargs["system"], list)
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_no_uncertain_no_escalation(self):
        """If all edges are approved, no escalation calls made."""
        e1 = _entity("Alice")
        e2 = _entity("Bob")
        entities = {e1.id: e1, e2.id: e2}
        gs = _mock_graph_store([(e1.id, e2.id, 5)], entities)

        haiku_client = MagicMock()
        haiku_client.messages.create.return_value = _make_llm_response("approved")

        sonnet_client = MagicMock()

        cfg = ActivationConfig(
            consolidation_infer_cooccurrence_min=3,
            consolidation_infer_confidence_floor=0.7,
            consolidation_infer_llm_enabled=True,
            consolidation_infer_llm_confidence_threshold=0.5,
            consolidation_infer_escalation_enabled=True,
        )
        phase = EdgeInferencePhase(
            llm_client=haiku_client, escalation_client=sonnet_client,
        )
        _, records = await phase.execute(
            group_id="test",
            graph_store=gs,
            activation_store=AsyncMock(),
            search_index=AsyncMock(),
            cfg=cfg,
            cycle_id="cyc_test",
            dry_run=False,
        )

        assert records[0].llm_verdict == "approved"
        assert records[0].escalation_verdict is None
        sonnet_client.messages.create.assert_not_called()
