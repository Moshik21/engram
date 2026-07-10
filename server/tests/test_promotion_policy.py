"""Tests for sparse agent promotion + durable recall ranking."""

from __future__ import annotations

from engram.extraction.client_proposals import (
    HIGH_SIGNAL_VERIFIED_CONFIDENCE,
    proposals_to_evidence,
)
from engram.extraction.commit_policy import AdaptiveCommitPolicy
from engram.extraction.evidence import EvidenceBundle
from engram.extraction.promotion import (
    DEFAULT_PROMOTE_CAP_PER_WINDOW,
    DEFAULT_SESSION_PROMOTE_CAP,
    filter_promotion_proposals,
    is_session_recap,
)
from engram.ingestion.capture_surface import apply_promotion_filter
from engram.retrieval.result_selection import prefer_durable_facts


def test_session_recap_rejected():
    assert is_session_recap("What we did today: fixed bugs and shipped a PR")
    filtered = filter_promotion_proposals(
        "Session recap: what we worked on today across five tickets",
        [{"name": "Bugfix", "entity_type": "Decision", "source_span": "Bugfix"}],
        None,
    )
    assert filtered.is_recap
    assert filtered.entities == []


def test_filter_autofills_source_span_from_content():
    content = "We decided LongMemEval is not the product north star."
    filtered = filter_promotion_proposals(
        content,
        [
            {
                "name": "LongMemEval is not the product north star",
                "entity_type": "Decision",
            }
        ],
        None,
    )
    assert not filtered.is_recap
    assert filtered.entities[0]["source_span"] == "LongMemEval is not the product north star"


def test_apply_promotion_filter_rejects_recap():
    entities, rels, meta = apply_promotion_filter(
        "In this session we discussed five tickets and merged two PRs",
        [{"name": "Ticket dump", "entity_type": "Decision"}],
        None,
    )
    assert meta["rejected_as_recap"] is True
    assert entities is None
    assert rels is None


def test_high_signal_client_proposal_commits():
    content = "LongMemEval is not the product north star."
    candidates = proposals_to_evidence(
        [
            {
                "name": "LongMemEval is not the product north star",
                "entity_type": "Decision",
                "source_span": "LongMemEval is not the product north star",
            }
        ],
        [
            {
                "subject": "Engram",
                "predicate": "DECIDED",
                "object": "LongMemEval is not the product north star",
                "source_span": "LongMemEval is not the product north star",
            }
        ],
        "ep_test",
        "default",
        "sonnet",
        episode_content=content,
        verify_spans=True,
    )
    assert all("span_verified" in c.corroborating_signals for c in candidates)
    decision = next(c for c in candidates if c.fact_class == "entity")
    assert decision.confidence >= HIGH_SIGNAL_VERIFIED_CONFIDENCE

    # Dense-graph thresholds must still commit verified high-signal proposals.
    policy = AdaptiveCommitPolicy()
    bundle = EvidenceBundle(episode_id="ep_test", group_id="default", candidates=candidates)
    decisions = policy.evaluate(bundle, entity_count=2000)
    assert all(d.action == "commit" for d in decisions)
    assert any(d.reason == "client_proposal_span_verified" for d in decisions)


def test_prefer_durable_facts_ranks_decisions_above_cues():
    results = [
        {
            "result_type": "cue_episode",
            "score": 0.95,
            "cue": {"episode_id": "ep1"},
            "source": "auto:prompt",
        },
        {
            "result_type": "entity",
            "score": 0.4,
            "entity": {
                "id": "e1",
                "name": "LongMemEval is not the product north star",
                "type": "Decision",
            },
        },
        {
            "result_type": "entity",
            "score": 0.7,
            "entity": {"id": "e2", "name": "React", "type": "Technology"},
        },
    ]
    ranked = prefer_durable_facts(results)
    assert ranked[0]["entity"]["type"] == "Decision"
    assert ranked[-1]["result_type"] == "cue_episode"


def test_session_promote_cap_constant():
    assert DEFAULT_SESSION_PROMOTE_CAP == 5
    assert DEFAULT_PROMOTE_CAP_PER_WINDOW == 5


def test_identity_core_blocks_bad_merges():
    from types import SimpleNamespace

    from engram.extraction.promotion import identity_core_blocks_merge

    core = SimpleNamespace(
        name="LongMemEval is not Engram north star",
        identity_core=True,
    )
    golden = SimpleNamespace(
        name="GOLDEN_PATH_DECISION LongMemEval not north star",
        identity_core=False,
    )
    scrap = SimpleNamespace(
        name="MachineShopScheduler:decision_statement:Making a Decision",
        identity_core=False,
    )
    same = SimpleNamespace(
        name="LongMemEval is not Engram north star",
        identity_core=True,
    )
    assert identity_core_blocks_merge(core, golden) is True
    assert identity_core_blocks_merge(core, scrap) is True
    assert identity_core_blocks_merge(core, same) is False
    assert (
        identity_core_blocks_merge(
            SimpleNamespace(name="A", identity_core=False),
            SimpleNamespace(name="B", identity_core=False),
        )
        is False
    )


def test_decision_statement_names_pass_validation():
    from engram.extraction.resolver import validate_entity_name

    long_decision = "GOLDEN_DECISION_1783643390: LongMemEval is not product north star"
    assert validate_entity_name(long_decision) is False  # narrow default
    assert (
        validate_entity_name(
            long_decision,
            entity_type="Decision",
            client_proposal=True,
        )
        is True
    )
    # Still reject giant recap dumps even for proposals.
    dump = " ".join(["word"] * 30)
    assert validate_entity_name(dump, entity_type="Decision", client_proposal=True) is False


class _FakeSession:
    def __init__(self) -> None:
        self.remember_count = 0
        self.promotion_window_id = None
        self.promotion_window_started_at = None
        self.last_remember_at = None
        self.promotion_window_reset_reason = "session_start"


def test_promotion_window_resets_on_compaction_id():
    from engram.extraction.promotion import (
        record_promotion_in_window,
        resolve_promotion_window,
    )

    session = _FakeSession()
    w1 = resolve_promotion_window(
        session,
        compaction_id="compact-1",
        now=1_000.0,
        read_external_window=False,
    )
    assert w1.remember_count == 0
    record_promotion_in_window(session, w1, now=1_001.0)
    record_promotion_in_window(session, w1, now=1_002.0)
    assert session.remember_count == 2

    # Same compaction id keeps the budget.
    w_same = resolve_promotion_window(
        session,
        compaction_id="compact-1",
        now=1_010.0,
        read_external_window=False,
    )
    assert w_same.remember_count == 2
    assert w_same.window_id == "compact-1"

    # New compaction id opens a fresh 0–5 window (multi-day / post-compress).
    w2 = resolve_promotion_window(
        session,
        compaction_id="compact-2",
        now=1_020.0,
        read_external_window=False,
    )
    assert w2.remember_count == 0
    assert w2.window_id == "compact-2"
    assert w2.reset_reason == "compaction_id"


def test_promotion_window_resets_on_idle_gap():
    from engram.extraction.promotion import (
        DEFAULT_PROMOTION_WINDOW_IDLE_SECONDS,
        record_promotion_in_window,
        resolve_promotion_window,
    )

    session = _FakeSession()
    w1 = resolve_promotion_window(session, now=1_000.0, read_external_window=False)
    record_promotion_in_window(session, w1, now=1_000.0)
    assert session.remember_count == 1

    still = resolve_promotion_window(session, now=1_000.0 + 60, read_external_window=False)
    assert still.remember_count == 1

    after_idle = resolve_promotion_window(
        session,
        now=1_000.0 + DEFAULT_PROMOTION_WINDOW_IDLE_SECONDS + 1,
        read_external_window=False,
    )
    assert after_idle.remember_count == 0
    assert after_idle.reset_reason == "idle_gap"


def test_promotion_window_resets_on_compaction_source():
    from engram.extraction.promotion import resolve_promotion_window

    session = _FakeSession()
    session.remember_count = 5
    session.promotion_window_id = "old"
    session.last_remember_at = 1_000.0
    w = resolve_promotion_window(
        session,
        source="claude:precompact",
        now=1_100.0,
        read_external_window=False,
    )
    assert w.remember_count == 0
    assert w.reset_reason == "compaction_source"


def test_promotion_window_reads_precompact_file(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from engram.extraction.promotion import resolve_promotion_window

    window_file = tmp_path / "promotion-window.json"
    window_file.write_text(
        json.dumps(
            {
                "compaction_id": "compact_from_file",
                "reset_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": "claude:precompact",
            }
        )
    )
    monkeypatch.setenv("ENGRAM_PROMOTION_WINDOW_FILE", str(window_file))

    session = _FakeSession()
    session.remember_count = 4
    session.promotion_window_id = "old_window"
    w = resolve_promotion_window(session, now=2_000.0)
    assert w.window_id == "compact_from_file"
    assert w.remember_count == 0
    assert w.reset_reason == "precompact_file"
