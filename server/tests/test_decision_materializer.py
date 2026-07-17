"""Tests for DecisionMaterializer human-readable names, dedup, and vocabulary config."""

from __future__ import annotations

import uuid

import pytest

from engram.config import ActivationConfig
from engram.extraction.promotion import is_decision_statement_noise
from engram.ingestion.decision_materializer import DecisionMaterializer
from engram.models.entity import Entity
from engram.models.epistemic import EvidenceClaim
from engram.storage.sqlite.graph import SQLiteGraphStore

GROUP = "default"


def _claim(
    predicate: str = "decision_statement",
    object_: str = "we decided to launch Engram through OpenClaw",
    subject: str = "Engram",
) -> EvidenceClaim:
    return EvidenceClaim(
        subject=subject,
        predicate=predicate,
        object=object_,
        source_type="memory",
        authority_type="historical",
        externalization_state="discussed",
        claim_state="decided",
        confidence=0.9,
    )


@pytest.fixture
async def graph():
    store = SQLiteGraphStore(":memory:")
    await store.initialize()
    yield store
    await store.close()


@pytest.fixture
def materializer(graph):
    async def _noop_index(entity: Entity, group_id: str) -> None:
        return None

    return DecisionMaterializer(
        graph_store=graph,
        cfg=ActivationConfig(decision_graph_enabled=True),
        index_entity=_noop_index,
    )


@pytest.mark.asyncio
class TestDecisionNames:
    async def test_decision_statement_gets_human_name(self, materializer):
        decision = await materializer.upsert_decision_entity(_claim(), group_id=GROUP)
        assert ":decision_statement:" not in decision.name
        assert "decision_statement" not in decision.name
        assert not is_decision_statement_noise(decision.name)
        assert decision.name == "Engram: we decided to launch Engram through OpenClaw"

    async def test_profile_predicate_gets_readable_name(self, materializer):
        decision = await materializer.upsert_decision_entity(
            _claim(predicate="consolidation_profile", object_="standard"),
            group_id=GROUP,
        )
        assert decision.name == "Engram: consolidation profile = standard"

    async def test_config_predicate_strips_config_prefix(self, materializer):
        decision = await materializer.upsert_decision_entity(
            _claim(
                predicate="config:engram_activation__recall_profile",
                object_="wave2",
            ),
            group_id=GROUP,
        )
        assert ":" not in decision.name.replace("Engram: ", "")
        assert decision.name == "Engram: engram activation recall profile = wave2"

    async def test_long_statement_truncated_at_word_boundary(self, materializer):
        decision = await materializer.upsert_decision_entity(
            _claim(object_="we decided " + "the launch plan must stay local " * 8),
            group_id=GROUP,
        )
        assert len(decision.name) <= 97
        assert decision.name.endswith("…")
        assert not decision.name[:-1].endswith(" ")


@pytest.mark.asyncio
class TestDedup:
    async def test_rematerialization_does_not_duplicate(self, materializer, graph):
        first = await materializer.upsert_decision_entity(_claim(), group_id=GROUP)
        second = await materializer.upsert_decision_entity(_claim(), group_id=GROUP)
        assert second.id == first.id
        decisions = await graph.find_entities(entity_type="Decision", group_id=GROUP, limit=50)
        assert len(decisions) == 1

    async def test_rematerialization_dedups_against_legacy_noisy_row(self, materializer, graph):
        claim = _claim()
        legacy = Entity(
            id=f"dec_{uuid.uuid4().hex[:12]}",
            name=f"{claim.subject}:{claim.predicate}:{claim.object[:80]}",
            entity_type="Decision",
            summary=f"{claim.subject} -> {claim.predicate} -> {claim.object}",
            attributes={
                "subject": claim.subject,
                "canonical_predicate": claim.predicate,
                "decision_object": claim.object,
            },
            group_id=GROUP,
        )
        await graph.create_entity(legacy)

        result = await materializer.upsert_decision_entity(claim, group_id=GROUP)
        assert result.id == legacy.id
        decisions = await graph.find_entities(entity_type="Decision", group_id=GROUP, limit=50)
        assert len(decisions) == 1

    async def test_different_object_supersedes_old_decision(self, materializer, graph):
        old = await materializer.upsert_decision_entity(
            _claim(predicate="consolidation_profile", object_="conservative"),
            group_id=GROUP,
        )
        new = await materializer.upsert_decision_entity(
            _claim(predicate="consolidation_profile", object_="standard"),
            group_id=GROUP,
        )
        assert new.id != old.id
        relationships = await graph.get_relationships(old.id, direction="outgoing", group_id=GROUP)
        assert any(
            rel.target_id == new.id and rel.predicate == "SUPERSEDED_BY" for rel in relationships
        )


@pytest.mark.asyncio
class TestConversationFlow:
    async def test_conversation_decision_names_are_human(self, materializer, graph):
        await materializer.materialize_conversation_decisions(
            "we decided the plan is to launch Engram through OpenClaw",
            episode_id="ep_test_1",
            group_id=GROUP,
        )
        decisions = await graph.find_entities(entity_type="Decision", group_id=GROUP, limit=50)
        assert decisions
        for decision in decisions:
            assert ":decision_statement:" not in decision.name
            assert not is_decision_statement_noise(decision.name)
            assert decision.name.startswith("Engram: ")


class TestVocabularyConfig:
    def test_default_predicates_still_match(self):
        assert DecisionMaterializer.is_decision_claim(_claim())
        assert DecisionMaterializer.is_decision_claim(
            _claim(predicate="config:engram_activation__recall_profile")
        )
        assert not DecisionMaterializer.is_decision_claim(_claim(predicate="unrelated_pred"))

    def test_custom_vocabulary_overrides_predicates(self):
        vocab = {"predicates": ["ship_gate"], "predicate_prefixes": ["policy:"]}
        assert DecisionMaterializer.is_decision_claim(_claim(predicate="ship_gate"), vocab)
        assert DecisionMaterializer.is_decision_claim(_claim(predicate="policy:release"), vocab)
        assert not DecisionMaterializer.is_decision_claim(_claim(), vocab)

    def test_default_subject_inference(self):
        assert DecisionMaterializer.infer_decision_subject("the engram rollout") == "Engram"
        assert DecisionMaterializer.infer_decision_subject("our repo layout") == "Project"
        assert DecisionMaterializer.infer_decision_subject("unrelated chatter") is None

    def test_custom_subject_terms(self):
        vocab = {"subject_terms:Acme": ["acme", "widget"]}
        assert DecisionMaterializer.infer_decision_subject("the acme migration", vocab) == "Acme"
        assert DecisionMaterializer.infer_decision_subject("the engram rollout", vocab) is None

    def test_materializer_uses_cfg_vocabulary(self):
        async def _noop_index(entity: Entity, group_id: str) -> None:
            return None

        cfg = ActivationConfig(
            decision_graph_enabled=True,
            decision_vocabulary={
                "predicates": ["ship_gate"],
                "subject_terms:Acme": ["acme"],
            },
        )
        materializer = DecisionMaterializer(
            graph_store=None,  # vocabulary lookup only; store unused
            cfg=cfg,
            index_entity=_noop_index,
        )
        assert materializer.is_decision_claim(
            _claim(predicate="ship_gate"), materializer._vocabulary()
        )
        assert not materializer.is_decision_claim(_claim(), materializer._vocabulary())
        assert (
            materializer.infer_decision_subject("the acme migration", materializer._vocabulary())
            == "Acme"
        )
