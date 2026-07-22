"""M1.4 squatter guard: entity name cap + observation corroboration hold.

Reproduces the battery's squatter-entity class (a milestone observation
extracted into a sentence-long entity name that scored 0.99 on unrelated
ranking queries) and pins the salience_class storage round-trip.
"""

from __future__ import annotations

import json

import pytest

from engram.config import HelixDBConfig
from engram.extraction.commit_policy import AdaptiveCommitPolicy
from engram.extraction.evidence import EvidenceBundle, EvidenceCandidate
from engram.models.episode import Episode
from engram.storage.helix.graph import HelixGraphStore

# The live squatter entity was a full milestone sentence (battery defect #2).
_SQUATTER_NAME = "Flip decision usage ranking stays disabled until organic used-tier yield arrives"


def _entity_candidate(
    name: str = "Postgres",
    confidence: float = 0.9,
    signals: list[str] | None = None,
    summary: str = "",
    source_type: str = "narrow_extractor",
) -> EvidenceCandidate:
    payload = {"name": name, "entity_type": "Technology"}
    if summary:
        payload["summary"] = summary
    return EvidenceCandidate(
        episode_id="ep1",
        group_id="default",
        fact_class="entity",
        confidence=confidence,
        source_type=source_type,
        extractor_name="test",
        payload=payload,
        corroborating_signals=list(signals) if signals is not None else ["technical_token"],
    )


def _evaluate_one(candidate: EvidenceCandidate, episode_source: str | None = None):
    policy = AdaptiveCommitPolicy()
    bundle = EvidenceBundle(episode_id="ep1", candidates=[candidate])
    return policy.evaluate(bundle, entity_count=100, episode_source=episode_source)[0]


class TestEntityNameCap:
    def test_sentence_name_from_battery_is_capped(self, caplog):
        candidate = _entity_candidate(name=_SQUATTER_NAME)
        with caplog.at_level("WARNING", logger="engram.extraction.commit_policy"):
            decision = _evaluate_one(candidate, episode_source="test")
        assert decision.action == "commit"
        capped = candidate.payload["name"]
        assert len(capped.split()) == 6
        assert capped == "Flip decision usage ranking stays disabled"
        # Excess folds into the summary — nothing is silently dropped.
        assert _SQUATTER_NAME in candidate.payload["summary"]
        assert "name_capped" in candidate.corroborating_signals
        # Loud log, not a silent rewrite.
        assert any("Squatter guard" in rec.message for rec in caplog.records)

    def test_six_token_name_untouched(self):
        name = "New York City Transit Authority Board"
        candidate = _entity_candidate(name=name)
        _evaluate_one(candidate, episode_source="test")
        assert candidate.payload["name"] == name
        assert "name_capped" not in candidate.corroborating_signals
        assert "summary" not in candidate.payload

    def test_cap_prepends_to_existing_summary(self):
        candidate = _entity_candidate(name=_SQUATTER_NAME, summary="Existing summary.")
        _evaluate_one(candidate, episode_source="test")
        assert candidate.payload["summary"].startswith(_SQUATTER_NAME)
        assert candidate.payload["summary"].endswith("Existing summary.")

    def test_cap_applies_to_client_proposals(self):
        candidate = _entity_candidate(
            name=_SQUATTER_NAME,
            signals=["span_verified", "high_signal_type"],
            source_type="client_proposal",
        )
        decision = _evaluate_one(candidate, episode_source=None)
        assert decision.action == "commit"
        assert len(candidate.payload["name"].split()) == 6


class TestObservationCorroborationHold:
    @pytest.mark.parametrize(
        "source",
        ["mcp_observe", "api_auto_observe", "axi", "auto:prompt"],
    )
    def test_observation_sourced_entity_defers(self, source):
        candidate = _entity_candidate(confidence=0.9)
        decision = _evaluate_one(candidate, episode_source=source)
        assert decision.action == "defer"
        assert decision.reason == "observation_needs_corroboration"
        # The signal persists into the deferred evidence row so the
        # adjudication corroboration gate (count >= 2) recognizes the hold.
        assert "observation_sourced" in candidate.corroborating_signals

    @pytest.mark.parametrize("source", [None, "test", "mcp_remember", "claude-code"])
    def test_non_observation_sources_commit_unchanged(self, source):
        candidate = _entity_candidate(confidence=0.9)
        decision = _evaluate_one(candidate, episode_source=source)
        assert decision.action == "commit"
        assert "observation_sourced" not in (candidate.corroborating_signals or [])

    def test_identity_signals_exempt_from_hold(self):
        candidate = _entity_candidate(
            name="Konner",
            confidence=0.9,
            signals=["identity_pattern"],
        )
        decision = _evaluate_one(candidate, episode_source="mcp_observe")
        assert decision.action == "commit"

    def test_relationships_not_held(self):
        candidate = EvidenceCandidate(
            episode_id="ep1",
            group_id="default",
            fact_class="relationship",
            confidence=0.9,
            source_type="narrow_extractor",
            extractor_name="test",
            payload={"subject": "Konner", "object": "Engram", "predicate": "BUILDS"},
            corroborating_signals=[],
        )
        decision = _evaluate_one(candidate, episode_source="mcp_observe")
        assert decision.action == "commit"

    def test_observation_span_verified_proposal_also_held(self):
        # The squatter path: axi observe with an agent-proposed entity must
        # not commit on first sight without a second episode.
        candidate = _entity_candidate(
            confidence=0.95,
            signals=["span_verified", "high_signal_type"],
            source_type="client_proposal",
        )
        decision = _evaluate_one(candidate, episode_source="axi")
        assert decision.action == "defer"
        assert decision.reason == "observation_needs_corroboration"


class _FakeHelixClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def query(self, endpoint: str, payload: dict) -> list[dict]:
        self.calls.append((endpoint, payload))
        return [{"id": "helix-1"}]


class TestHelixSalienceSerialization:
    @pytest.mark.asyncio
    async def test_create_episode_embeds_salience_class(self):
        client = _FakeHelixClient()
        store = HelixGraphStore(HelixDBConfig(), client=client, owns_client=False)
        episode = Episode(
            id="ep_mach",
            content="[session-end|Engram] Session ended",
            source="auto:session",
            salience_class="machinery",
        )
        await store.create_episode(episode)
        endpoint, payload = client.calls[0]
        assert endpoint == "create_episode"
        assert json.loads(payload["encoding_context_json"]) == {"salience_class": "machinery"}

    @pytest.mark.asyncio
    async def test_create_episode_without_class_is_byte_identical(self):
        client = _FakeHelixClient()
        store = HelixGraphStore(HelixDBConfig(), client=client, owns_client=False)
        episode = Episode(id="ep_plain", content="Genuine content.")
        await store.create_episode(episode)
        _endpoint, payload = client.calls[0]
        assert payload["encoding_context_json"] == "{}"

    def test_dict_to_episode_decodes_salience_class(self):
        store = HelixGraphStore(HelixDBConfig(), client=_FakeHelixClient(), owns_client=False)
        episode = store._dict_to_episode(
            {
                "id": "helix-1",
                "episode_id": "ep_mach",
                "group_id": "default",
                "content": "[session-end|Engram] Session ended",
                "status": "queued",
                "projection_state": "queued",
                "encoding_context_json": '{"salience_class": "machinery"}',
            }
        )
        assert episode.salience_class == "machinery"
        episode = store._dict_to_episode(
            {
                "id": "helix-2",
                "episode_id": "ep_plain",
                "group_id": "default",
                "content": "Genuine content.",
                "status": "queued",
                "projection_state": "queued",
                "encoding_context_json": "{}",
            }
        )
        assert episode.salience_class == ""
