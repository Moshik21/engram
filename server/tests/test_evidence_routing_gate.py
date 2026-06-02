"""Regression: LLM extractor output must not be silently discarded by v2 routing.

The Anthropic EntityExtractor produces final, commit-quality entities directly. The
v2 evidence pipeline (_build_evidence_bundle) only ever runs the narrow regex
extractor and never consumes an LLM extractor's output, so routing the LLM there
DISCARDS its clean entities and persists narrow-regex fragments instead
('Voss wants me to' rather than 'Dr. Voss'). _should_use_evidence_pipeline must
therefore route a bare LLM extractor to the legacy committing path, while still
using v2 for narrow extraction and for client-proposal enrichment.
"""

from types import SimpleNamespace

from engram.extraction.extractor import EntityExtractor
from engram.extraction.narrow_adapter import NarrowExtractorAdapter
from engram.graph_manager import GraphManager


def _gate(extractor, *, proposed_entities=None, proposed_relationships=None):
    """Call the routing gate against a minimal stub (it only reads attributes)."""
    stub = SimpleNamespace(
        _extractor=extractor,
        _cfg=SimpleNamespace(evidence_extraction_enabled=True),
        _evidence_pipeline=object(),
        _commit_policy=object(),
        _evidence_bridge=object(),
    )
    return GraphManager._should_use_evidence_pipeline(
        stub,
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
    )


def test_llm_extractor_routes_to_legacy_not_evidence():
    # THE FIX: a bare Anthropic LLM extractor must NOT use the evidence pipeline,
    # otherwise its clean entities are discarded and regex fragments persist.
    assert _gate(EntityExtractor()) is False


def test_llm_extractor_with_client_proposals_still_uses_evidence():
    # Enrichment path preserved: client proposals route through v2 adjudication
    # even with an LLM extractor present.
    assert _gate(EntityExtractor(), proposed_entities=[{"name": "Acme"}]) is True


def test_narrow_extractor_uses_evidence_pipeline():
    # The narrow adapter legitimately feeds the evidence pipeline.
    assert _gate(NarrowExtractorAdapter()) is True
