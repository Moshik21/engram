"""Harness-as-extractor hard/trust/scoreboard paths on shipped code."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from engram.config import ActivationConfig
from engram.extraction.client_proposals import proposals_to_evidence, span_is_verified
from engram.extraction.commit_policy import AdaptiveCommitPolicy
from engram.extraction.evidence import EvidenceBundle
from engram.extraction.harness_metrics import (
    get_harness_metrics,
    harness_scoreboard_payload,
    record_remember_call,
    reset_harness_metrics,
)
from engram.extraction.promotion import (
    identity_core_summary_conflict,
    is_allowed_client_predicate,
    normalize_client_predicate,
)


@pytest.fixture(autouse=True)
def _reset_metrics():
    reset_harness_metrics()
    yield
    reset_harness_metrics()


def test_span_is_verified_real_path():
    content = "LongMemEval is not Engram north star. Continuity is the metric."
    assert span_is_verified("LongMemEval is not Engram north star", content)
    assert not span_is_verified("this span is not in the content", content)


def test_span_unverified_defers_not_commits():
    content = "Alice prefers sparse promotion."
    candidates = proposals_to_evidence(
        [
            {
                "name": "Alice prefers sparse promotion",
                "entity_type": "Preference",
                "source_span": "NOT IN CONTENT AT ALL",
            }
        ],
        None,
        "ep1",
        "default",
        "sonnet",
        episode_content=content,
        verify_spans=True,
    )
    assert len(candidates) == 1
    assert "span_unverified" in candidates[0].corroborating_signals
    policy = AdaptiveCommitPolicy()
    decision = policy.evaluate(EvidenceBundle(candidates=candidates), entity_count=0)[0]
    assert decision.action == "defer"
    assert decision.reason == "span_unverified"


def test_span_verified_commits_high_signal():
    content = "LongMemEval is not Engram north star."
    candidates = proposals_to_evidence(
        [
            {
                "name": "LongMemEval is not Engram north star",
                "entity_type": "Decision",
                "source_span": "LongMemEval is not Engram north star",
            }
        ],
        None,
        "ep1",
        "default",
        "sonnet",
        episode_content=content,
        verify_spans=True,
    )
    policy = AdaptiveCommitPolicy()
    decision = policy.evaluate(EvidenceBundle(candidates=candidates), entity_count=0)[0]
    assert decision.action == "commit"
    assert decision.reason == "client_proposal_span_verified"


def test_predicate_allowlist_rejects_disallowed():
    assert is_allowed_client_predicate("DECIDED")
    assert is_allowed_client_predicate("prefers")  # normalized
    assert not is_allowed_client_predicate("HACKS_INTO")
    assert normalize_client_predicate("works-at") == "WORKS_AT"

    candidates = proposals_to_evidence(
        None,
        [
            {
                "subject": "Alice",
                "predicate": "HACKS_INTO",
                "object": "Mainframe",
                "source_span": "Alice HACKS_INTO Mainframe",
            }
        ],
        "ep1",
        "default",
        "sonnet",
        episode_content="Alice HACKS_INTO Mainframe",
        verify_spans=True,
    )
    assert "predicate_not_allowed" in candidates[0].corroborating_signals
    policy = AdaptiveCommitPolicy()
    decision = policy.evaluate(EvidenceBundle(candidates=candidates), entity_count=0)[0]
    assert decision.action == "reject"
    assert decision.reason == "predicate_not_allowed"


def test_identity_core_summary_conflict_helper():
    assert identity_core_summary_conflict(
        "User prefers markdown handoffs",
        "User prefers JSON APIs only",
    )
    assert not identity_core_summary_conflict(
        "User prefers markdown",
        "User prefers markdown handoffs until proven",
    )
    assert not identity_core_summary_conflict(
        "old",
        "new correction",
        entity_type="Correction",
    )


def test_build_evidence_bundle_skips_pipeline_and_external_when_proposals():
    from engram.graph_manager import GraphManager

    cfg = ActivationConfig(
        evidence_extraction_enabled=True,
        evidence_client_proposals_enabled=True,
    )
    manager = GraphManager(
        graph_store=MagicMock(),
        activation_store=MagicMock(),
        search_index=MagicMock(),
        extractor=MagicMock(),
        cfg=cfg,
    )
    manager._evidence_pipeline = MagicMock()
    manager._evidence_pipeline.extract.side_effect = AssertionError(
        "must not call narrow/LLM when proposals present"
    )
    # Simulate external LLM extractor attached
    manager._extractor.extract = MagicMock(
        side_effect=AssertionError("external extractor must not run"),
    )

    before = get_harness_metrics().external_extractor_skipped
    bundle = manager._build_evidence_bundle(
        text="Prefer sparse agent promotion.",
        episode_id="ep1",
        group_id="default",
        proposed_entities=[
            {
                "name": "Prefer sparse agent promotion",
                "entity_type": "Decision",
                "source_span": "Prefer sparse agent promotion",
            }
        ],
        model_tier="sonnet",
    )
    assert bundle.extractor_stats["extraction_path"] == "client_proposals"
    assert all(c.source_type == "client_proposal" for c in bundle.candidates)
    manager._evidence_pipeline.extract.assert_not_called()
    assert get_harness_metrics().external_extractor_skipped == before + 1


def test_proposals_force_evidence_path_even_when_extraction_disabled():
    from engram.graph_manager import GraphManager

    cfg = ActivationConfig(evidence_extraction_enabled=False)
    manager = GraphManager(
        graph_store=MagicMock(),
        activation_store=MagicMock(),
        search_index=MagicMock(),
        extractor=MagicMock(),
        cfg=cfg,
    )
    assert manager._should_use_evidence_pipeline(
        proposed_entities=[{"name": "X", "entity_type": "Decision"}],
    )
    # Without proposals, narrow pipeline is off → no evidence path for LLM.
    assert not manager._should_use_evidence_pipeline()


def test_promote_rate_and_scoreboard():
    record_remember_call(has_proposals=True)
    record_remember_call(has_proposals=True)
    record_remember_call(has_proposals=False)
    snap = get_harness_metrics()
    assert snap.promote_rate() == pytest.approx(2 / 3)
    payload = harness_scoreboard_payload()
    assert payload["status"] == "ok"
    assert "client_proposal_share" in payload["metrics"]
    assert payload["metrics"]["promote_rate"] == pytest.approx(2 / 3)


def test_harness_scoreboard_cli():
    from engram.harness_cli import run_harness_command

    record_remember_call(has_proposals=True)
    args = SimpleNamespace(harness_command="scoreboard", format="json", reset=False)
    assert run_harness_command(args) == 0
