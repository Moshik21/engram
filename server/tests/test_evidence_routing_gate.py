"""Regression: a non-narrow extractor's output must not be silently discarded by v2 routing.

The v2 evidence pipeline (_build_evidence_bundle) only ever runs the narrow regex
extractor and never consumes any other extractor's output, so routing an extractor
that produces its own final entities there DISCARDS them and persists narrow-regex
fragments instead ('Voss wants me to' rather than 'Dr. Voss'). The since-deleted
Anthropic EntityExtractor hit exactly this. _should_use_evidence_pipeline must
therefore route a bare non-narrow extractor to the legacy committing path, while
still using v2 for narrow extraction and for client-proposal enrichment.
"""

from types import SimpleNamespace

from engram.extraction.extractor import ExtractionResult
from engram.extraction.narrow_adapter import NarrowExtractorAdapter
from engram.graph_manager import GraphManager


class _FinalOutputExtractor:
    """Any extractor that emits commit-quality entities itself (no canned _result)."""

    async def extract(self, text: str) -> ExtractionResult:
        return ExtractionResult(
            entities=[{"name": "Dr. Voss", "entity_type": "Person"}],
            relationships=[],
        )


def _gate(extractor, *, proposed_entities=None, proposed_relationships=None):
    """Call the routing gate against a minimal stub (it only reads attributes)."""
    stub = SimpleNamespace(
        _extractor=extractor,
        _cfg=SimpleNamespace(
            evidence_extraction_enabled=True,
            evidence_client_proposals_enabled=True,
        ),
        _evidence_pipeline=object(),
        _commit_policy=object(),
        _evidence_bridge=object(),
    )
    return GraphManager._should_use_evidence_pipeline(
        stub,
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
    )


def test_final_output_extractor_routes_to_legacy_not_evidence():
    # THE FIX: an extractor with its own final output must NOT use the evidence
    # pipeline, otherwise its entities are discarded and regex fragments persist.
    assert _gate(_FinalOutputExtractor()) is False


def test_final_output_extractor_with_client_proposals_still_uses_evidence():
    # Enrichment path preserved: client proposals route through v2 adjudication
    # even with a non-narrow extractor present.
    assert _gate(_FinalOutputExtractor(), proposed_entities=[{"name": "Acme"}]) is True


def test_narrow_extractor_uses_evidence_pipeline():
    # The narrow adapter legitimately feeds the evidence pipeline.
    assert _gate(NarrowExtractorAdapter()) is True
