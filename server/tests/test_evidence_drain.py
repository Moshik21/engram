"""Tests for deferred evidence drain classification and CLI helpers."""

from __future__ import annotations

import pytest

from engram.consolidation.evidence_drain import (
    audit_deferred_evidence,
    classify_deferred_evidence,
    classify_extraction_candidate,
    reject_junk_evidence,
    scaled_drain_budget,
    select_redundant_entity_evidence,
    select_stale_low_value_evidence,
    should_force_commit_evidence,
)
from engram.extraction.evidence import EvidenceCandidate


def _row(
    *,
    evidence_id: str = "evi_test",
    fact_class: str = "entity",
    name: str = "Engram",
    confidence: float = 0.7,
    source_span: str = "Engram is a memory layer",
    extractor_name: str = "identity_entity",
) -> dict:
    return {
        "evidence_id": evidence_id,
        "fact_class": fact_class,
        "confidence": confidence,
        "extractor_name": extractor_name,
        "source_span": source_span,
        "payload": {"name": name, "entity_type": "Concept"},
        "status": "deferred",
    }


class TestEvidenceDrainClassification:
    def test_keep_legitimate_entity(self):
        result = classify_deferred_evidence(_row())
        assert result.disposition == "keep"

    def test_reject_unknown_name(self):
        result = classify_deferred_evidence(_row(name="?"))
        assert result.disposition == "reject_junk"
        assert result.reason == "unknown_name"

    def test_reject_path_like_name(self):
        result = classify_deferred_evidence(_row(name="docs/archive"))
        assert result.disposition == "reject_junk"
        assert result.reason == "path_like_name"

    def test_reject_bootstrap_span(self):
        result = classify_deferred_evidence(
            _row(source_span="[project-bootstrap|Engram|README.md]\nEngram docs"),
        )
        assert result.disposition == "reject_junk"
        assert result.reason == "bootstrap_span"

    def test_reject_low_confidence_identity(self):
        result = classify_deferred_evidence(_row(confidence=0.55))
        assert result.disposition == "reject_junk"
        assert result.reason == "low_confidence_identity"

    def test_reject_markdown_fragment_name(self):
        result = classify_deferred_evidence(_row(name="Engram \n   \n     Long"))
        assert result.disposition == "reject_junk"
        assert result.reason == "markdown_fragment_name"

    def test_reject_token_slash_pair_but_keep_tech(self):
        junk = classify_deferred_evidence(_row(name="before/after"))
        assert junk.disposition == "reject_junk"
        assert junk.reason == "token_slash_pair"
        ratio = classify_deferred_evidence(_row(name="5/6"))
        assert ratio.disposition == "reject_junk"
        ab = classify_deferred_evidence(_row(name="A/B"))
        assert ab.disposition == "reject_junk"
        hyphen_scrap = classify_deferred_evidence(_row(name="awarded/no-award"))
        assert hyphen_scrap.disposition == "reject_junk"
        keep = classify_deferred_evidence(_row(name="shadcn/ui"))
        assert keep.disposition == "keep"
        pkg = classify_deferred_evidence(_row(name="ai-sdk/openai"))
        assert pkg.disposition == "keep"
        pkg2 = classify_deferred_evidence(_row(name="react-pdf/renderer"))
        assert pkg2.disposition == "keep"

    def test_reject_broken_relationship_endpoints(self):
        row = {
            "evidence_id": "evi_rel",
            "fact_class": "relationship",
            "confidence": 0.75,
            "extractor_name": "relationship_pattern",
            "source_span": "trust and use Engram",
            "payload": {"subject": "and", "predicate": "USES", "object": "Engram"},
            "status": "deferred",
        }
        result = classify_deferred_evidence(row)
        assert result.disposition == "reject_junk"
        assert result.reason == "broken_relationship_endpoint"

    def test_reject_markup_noise_attribute(self):
        row = _row(
            fact_class="attribute",
            name="",
            source_span='shields.io/badge/tests-pytest alt="Pytest"',
            extractor_name="attribute",
        )
        row["payload"] = {}
        result = classify_deferred_evidence(row)
        assert result.disposition == "reject_junk"
        assert result.reason == "markup_noise_span"

    def test_select_junk_prioritizes_over_prefix_slice(self):
        from engram.consolidation.evidence_drain import select_junk_evidence_rows

        rows = [
            _row(evidence_id="keep1"),
            _row(evidence_id="keep2"),
            _row(evidence_id="junk1", name="before/after"),
            _row(evidence_id="junk2", name="README.md"),
        ]
        # Old mop path: rows[:2] would reject nothing
        selected = select_junk_evidence_rows(rows, limit=2)
        assert [r["evidence_id"] for r in selected] == ["junk1", "junk2"]

    def test_audit_summary_counts(self):
        rows = [
            _row(evidence_id="evi_keep"),
            _row(evidence_id="evi_junk", name="docs/mobe3"),
        ]
        summary = audit_deferred_evidence(rows)
        assert summary.total == 2
        assert summary.keep == 1
        assert summary.reject_junk == 1
        assert summary.by_reason["path_like_name"] == 1


class _FakeGraphStore:
    def __init__(self) -> None:
        self.updates: list[tuple[str, str, dict | None]] = []

    async def update_evidence_status(
        self,
        evidence_id: str,
        status: str,
        updates: dict | None = None,
        group_id: str = "default",
    ) -> None:
        self.updates.append((evidence_id, status, updates))


@pytest.mark.asyncio
async def test_reject_junk_evidence_dry_run():
    store = _FakeGraphStore()
    rows = [
        _row(evidence_id="evi_keep"),
        _row(evidence_id="evi_junk", name="README.md"),
    ]
    result = await reject_junk_evidence(
        store,
        group_id="default",
        rows=rows,
        dry_run=True,
    )
    assert result["rejected"] == 1
    assert result["kept"] == 1
    assert store.updates == []


@pytest.mark.asyncio
async def test_reject_junk_evidence_writes_rejected_status():
    store = _FakeGraphStore()
    rows = [_row(evidence_id="evi_junk", name="private/tmp")]
    result = await reject_junk_evidence(
        store,
        group_id="default",
        rows=rows,
        dry_run=False,
    )
    assert result["rejected"] == 1
    assert store.updates[0][0] == "evi_junk"
    assert store.updates[0][1] == "rejected"
    assert store.updates[0][2]["commit_reason"].startswith("drain_evidence:")


@pytest.mark.asyncio
async def test_reject_junk_prioritize_fills_budget_from_full_pool():
    store = _FakeGraphStore()
    rows = [_row(evidence_id=f"keep{i}") for i in range(20)]
    rows.append(_row(evidence_id="junk_deep", name="before/after"))
    result = await reject_junk_evidence(
        store,
        group_id="default",
        rows=rows,
        dry_run=False,
        prioritize_junk=True,
        max_reject=5,
    )
    assert result["rejected"] == 1
    assert result["pool_size"] == 21
    assert store.updates[0][0] == "junk_deep"


def test_scaled_drain_budget_grows_with_debt():
    assert scaled_drain_budget(100, base_budget=500) == 500
    assert scaled_drain_budget(8000, base_budget=500, max_budget=5000) == 2000
    assert scaled_drain_budget(100_000, base_budget=500, max_budget=5000) == 5000


def test_select_redundant_entity_evidence():
    rows = [
        _row(evidence_id="e1", name="API"),
        _row(evidence_id="e2", name="Novel Thing"),
        {
            **_row(evidence_id="e3", name="API"),
            "source_type": "client_proposal",
        },
    ]
    selected = select_redundant_entity_evidence(rows, {"api"}, limit=10)
    assert [r["evidence_id"] for r in selected] == ["e1"]


def test_select_stale_low_value_and_force_commit_policy():
    stale = _row(evidence_id="stale", name="API", confidence=0.7)
    stale["deferred_cycles"] = 5
    stale["payload"] = {"name": "API", "entity_type": "Concept"}
    selected = select_stale_low_value_evidence(
        [stale],
        min_deferred_cycles=5,
        max_age_days=21.0,
        limit=10,
    )
    assert len(selected) == 1
    assert not should_force_commit_evidence(stale)

    decision = _row(evidence_id="dec", name="Ship continuity")
    decision["payload"] = {"name": "Ship continuity", "entity_type": "Decision"}
    decision["source_type"] = "client_proposal"
    assert should_force_commit_evidence(decision)
    assert (
        select_stale_low_value_evidence([decision], min_deferred_cycles=0, max_age_days=0, limit=10)
        == []
    )


def test_hot_path_candidate_junk_classification():
    cand = EvidenceCandidate(
        evidence_id="evi_j",
        fact_class="entity",
        confidence=0.8,
        source_type="narrow_extractor",
        extractor_name="identity_entity",
        payload={"name": "before/after", "entity_type": "Concept"},
        source_span="before/after",
    )
    result = classify_extraction_candidate(cand)
    assert result.disposition == "reject_junk"
