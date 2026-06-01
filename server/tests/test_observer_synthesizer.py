"""Unit tests for the deterministic observation synthesizer + cache freeze."""

from __future__ import annotations

import pytest

from engram.consolidation.observer.synthesizer import (
    LLMObservationSynthesizer,
    TemplateObservationSynthesizer,
    select_synthesizer,
)
from engram.extraction.extraction_cache import ExtractionCache
from engram.extraction.extractor import ExtractionResult, ExtractionStatus
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeStatus
from engram.models.relationship import Relationship


def _ep(eid: str, content: str) -> Episode:
    return Episode(id=eid, content=content, status=EpisodeStatus.COMPLETED, group_id="default")


def _entity(eid: str, name: str, role: str | None = None) -> Entity:
    attrs = {"role": role} if role else {}
    return Entity(id=eid, name=name, entity_type="Person", group_id="default", attributes=attrs)


def _rel(
    src: str,
    tgt: str,
    predicate: str = "WORKS_AT",
    polarity: str = "positive",
) -> Relationship:
    return Relationship(
        id=f"rel_{src}_{tgt}_{predicate}",
        source_id=src,
        target_id=tgt,
        predicate=predicate,
        polarity=polarity,
        group_id="default",
    )


# --- Template determinism ---


def test_template_is_deterministic_bytes():
    eps = [_ep("ep_1", "x"), _ep("ep_2", "y")]
    entities = [_entity("e2", "Bob", role="Manager"), _entity("e1", "Alice", role="Engineer")]
    rels = [_rel("e1", "e2", "REPORTS_TO")]
    synth = TemplateObservationSynthesizer()

    out1 = synth.synthesize(eps, entities, rels)
    out2 = synth.synthesize(list(reversed(eps)), list(reversed(entities)), list(rels))
    assert out1 == out2  # order-independent, byte-identical
    # FOCUSED: one observation per subject entity, each dense about that subject.
    joined = " ".join(out1)
    # Uses the canonical current-value convention.
    assert "Current role: Engineer." in joined
    assert "Current role: Manager." in joined
    # Relationship fact rendered, grouped under its subject (Alice).
    assert any("Alice reports to Bob." in o for o in out1)
    assert all("Alice" in o for o in out1 if "reports to" in o)  # rel stays with subject


def test_template_facts_stably_sorted():
    eps = [_ep("ep_1", "x")]
    entities = [_entity("e1", "Alice"), _entity("e2", "Bob"), _entity("e3", "Carol")]
    rels = [_rel("e3", "e1", "KNOWS"), _rel("e1", "e2", "KNOWS")]
    synth = TemplateObservationSynthesizer()
    out = synth.synthesize(eps, entities, rels)
    # Carol's fact text sorts after Alice's; assert both present in stable form.
    assert "Alice knows Bob." in out
    assert "Carol knows Alice." in out
    # Re-running yields identical bytes.
    assert out == synth.synthesize(eps, entities, rels)


def test_template_only_positive_polarity_facts():
    eps = [_ep("ep_1", "x")]
    entities = [_entity("e1", "Alice"), _entity("e2", "Acme")]
    rels = [_rel("e1", "e2", "WORKS_AT", polarity="negative")]
    synth = TemplateObservationSynthesizer()
    out = synth.synthesize(eps, entities, rels)
    assert "works at" not in " ".join(out)  # negated edge must not be surfaced


# --- No network on default path ---


def test_template_no_network(monkeypatch):
    # Forbid any anthropic client construction during the default template path.
    import engram.extraction.extractor as extractor_mod

    def _boom(*a, **k):  # pragma: no cover - should never be called
        raise AssertionError("LLM client constructed on the default template path")

    monkeypatch.setattr(extractor_mod, "EntityExtractor", _boom, raising=False)

    synth = select_synthesizer(llm_enabled=False)
    assert isinstance(synth, TemplateObservationSynthesizer)
    out = synth.synthesize([_ep("ep_1", "x")], [_entity("e1", "Alice", role="Engineer")], [])
    assert any("Engineer" in o for o in out)


def test_select_synthesizer_llm_stub():
    synth = select_synthesizer(llm_enabled=True, extractor=object(), llm_client=object())
    assert isinstance(synth, LLMObservationSynthesizer)
    # Stub falls back to deterministic template output (no live call).
    out = synth.synthesize([_ep("ep_1", "x")], [_entity("e1", "Alice", role="Engineer")], [])
    assert any("Engineer" in o for o in out)


# --- ExtractionCache freeze determinism (mechanism the LLM path will rely on) ---


class _StubExtractor:
    """Counts calls; returns a fixed verdict. Stands in for the LLM extractor."""

    _model = "stub-model"

    def __init__(self) -> None:
        self.calls = 0

    async def extract(self, text: str) -> ExtractionResult:
        self.calls += 1
        return ExtractionResult(
            entities=[{"name": "Alice", "type": "Person"}],
            relationships=[],
            status=ExtractionStatus.OK,
        )


@pytest.mark.asyncio
async def test_llm_path_frozen_by_cache(tmp_path):
    stub = _StubExtractor()
    cache = ExtractionCache(stub, tmp_path / "freeze.db")

    first = await cache.extract("synthesize this cluster")
    second = await cache.extract("synthesize this cluster")

    # Byte-identical verdict, second is a cache HIT, stub called exactly once.
    assert first.entities == second.entities
    assert cache.stats()["hits"] == 1
    assert cache.stats()["misses"] == 1
    assert stub.calls == 1
    cache.close()
