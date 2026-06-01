"""Observation synthesizers for the write-side reflect phase.

``TemplateObservationSynthesizer`` (default) renders a deterministic, fact-dense
observation from STRUCTURED graph state — never from free text it might
hallucinate over. For each salient entity it emits the canonical current value
using the same ``Current role: X.`` convention recall already prepends (see
``result_builder.entity_result``) so the observation is answer-shaped, plus
clear-predicate relationship facts (``Subject PREDICATE Object``). Entities and
facts are stably sorted before joining, so identical inputs always produce
byte-identical output.

``LLMObservationSynthesizer`` (behind ``observer_reflect_llm_enabled``) is a
GUARDED STUB for now: the master flag (``observer_reflect_enabled``) ships dark
and the eval arm runs the template path, so no live LLM call is wired. When a
measured win justifies it, this synthesizer will build a synthesis prompt over
the cluster text and call the extractor/llm_client through the ExtractionCache
for a frozen, byte-identical verdict.
"""

from __future__ import annotations

from typing import Protocol

from engram.models.entity import Entity
from engram.models.episode import Episode
from engram.models.relationship import Relationship


class ObservationSynthesizer(Protocol):
    """Selects how a cluster is turned into observation content."""

    @property
    def name(self) -> str: ...

    def synthesize(
        self,
        episodes: list[Episode],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str: ...


def _entity_current_value(entity: Entity) -> str | None:
    """Render an entity's canonical current value, if it carries one.

    Reuses the recall ``Current role: X.`` convention so the synthesized text is
    answer-shaped for current-value/synthesis judges. Only emits facts present in
    the entity's resolved state (the ``role`` attribute) — never invents.
    """
    attrs = entity.attributes if isinstance(entity.attributes, dict) else {}
    role = attrs.get("role")
    if role and str(role).strip():
        return f"{entity.name} — Current role: {str(role).strip()}."
    return None


def _relationship_facts(
    entities: list[Entity],
    relationships: list[Relationship],
) -> list[str]:
    """Render clear-predicate relationship facts as ``Subject PREDICATE Object``.

    Only positive-polarity edges between entities in the cluster are emitted, so
    the observation never resurfaces a negated or out-of-cluster claim.
    """
    name_by_id = {e.id: e.name for e in entities}
    facts: list[str] = []
    for rel in relationships:
        if rel.polarity != "positive":
            continue
        src = name_by_id.get(rel.source_id)
        tgt = name_by_id.get(rel.target_id)
        if not src or not tgt or not rel.predicate:
            continue
        facts.append(f"{src} {rel.predicate.replace('_', ' ').lower()} {tgt}.")
    return facts


class TemplateObservationSynthesizer:
    """Default, zero-cost, fully deterministic template synthesizer."""

    name = "template"

    def synthesize(
        self,
        episodes: list[Episode],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str:
        # Stable ordering by entity name, then by full fact string, guarantees
        # byte-identical output for identical inputs.
        sorted_entities = sorted(entities, key=lambda e: (e.name, e.id))

        value_facts: list[str] = []
        for entity in sorted_entities:
            value = _entity_current_value(entity)
            if value:
                value_facts.append(value)

        rel_facts = sorted(set(_relationship_facts(sorted_entities, relationships)))

        names = sorted({e.name for e in entities})
        if names:
            header = "Observation across related memories about " + ", ".join(names) + "."
        else:
            header = "Observation across related memories."

        parts = [header, *value_facts, *rel_facts]
        return " ".join(p for p in parts if p)


class LLMObservationSynthesizer:
    """Optional LLM synthesizer — GUARDED STUB (no live call wired yet).

    Constructed only when ``observer_reflect_llm_enabled`` is True. The phase
    itself only constructs a synthesizer when ``observer_reflect_enabled`` is
    True, so an OFF runtime never imports or builds an LLM client. Today this
    falls back to the deterministic template so the path is exercisable without
    a network dependency; the frozen-cache LLM call lands when a measured win
    justifies it.
    """

    name = "llm"

    def __init__(
        self,
        *,
        extractor: object | None = None,
        llm_client: object | None = None,
    ) -> None:
        self._extractor = extractor
        self._llm_client = llm_client
        self._fallback = TemplateObservationSynthesizer()

    def synthesize(
        self,
        episodes: list[Episode],
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> str:
        # STUB: no live LLM call. Deterministic template until the frozen-cache
        # synthesis prompt is wired (see module docstring).
        return self._fallback.synthesize(episodes, entities, relationships)


def select_synthesizer(
    *,
    llm_enabled: bool,
    extractor: object | None = None,
    llm_client: object | None = None,
) -> ObservationSynthesizer:
    """Pick the synthesizer at phase construction.

    Template unless ``llm_enabled`` is True, in which case the (stub) LLM
    synthesizer is built with the threaded extractor/llm_client. The OFF default
    never reaches here (the phase guards first), so no LLM client is constructed
    while shipping dark.
    """
    if llm_enabled:
        return LLMObservationSynthesizer(extractor=extractor, llm_client=llm_client)
    return TemplateObservationSynthesizer()
