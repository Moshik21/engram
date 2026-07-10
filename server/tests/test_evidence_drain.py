"""Tests for deferred evidence drain classification and CLI helpers."""

from __future__ import annotations

import pytest

from engram.consolidation.evidence_drain import (
    audit_deferred_evidence,
    classify_deferred_evidence,
    reject_junk_evidence,
)


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
