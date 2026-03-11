"""Graph resonance probe for memory-need analysis."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from engram.models.entity import Entity

_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.+-]{1,}")
_PHRASE_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
_PERSON_TYPES = {"person", "human", "contact"}
_RELATIONAL_FAMILY_ALIASES = {
    "child": {
        "son",
        "daughter",
        "kid",
        "kids",
        "child",
        "children",
        "baby",
    },
    "parent": {"mom", "dad", "mother", "father", "parent", "parents"},
    "mentor": {"mentor", "teacher", "professor", "coach", "tutor"},
    "caregiver": {"doctor", "therapist", "dentist", "vet"},
}
_RELATIONAL_PREDICATES = {
    "child": {"PARENT_OF", "CHILD_OF"},
    "parent": {"PARENT_OF", "CHILD_OF"},
    "mentor": {"MENTORS"},
    "caregiver": {"TREATS"},
}


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


@dataclass
class ProbeResult:
    """Output from the graph resonance probe."""

    resonance_score: float = 0.0
    detected_entities: list[str] = field(default_factory=list)
    entity_scores: dict[str, float] = field(default_factory=dict)
    match_sources: dict[str, str] = field(default_factory=dict)
    anchored_entity_ids: list[str] = field(default_factory=list)


@dataclass
class _CacheEntry:
    entities: list[Entity] = field(default_factory=list)
    refreshed_at: float = 0.0


class GraphProbe:
    """Storage-agnostic graph resonance probe."""

    def __init__(
        self,
        graph_store=None,
        activation_store=None,
        *,
        max_term_candidates: int = 6,
        max_detected_entities: int = 5,
        max_relationships: int = 8,
    ) -> None:
        self._graph_store = graph_store
        self._activation_store = activation_store
        self._max_term_candidates = max_term_candidates
        self._max_detected_entities = max_detected_entities
        self._max_relationships = max_relationships
        self._group_versions: dict[str, tuple[int, int]] = {}
        self._token_cache: dict[str, dict[str, _CacheEntry]] = {}
        self._identity_cache: dict[str, list[str]] = {}

    async def probe(
        self,
        text: str,
        lowered: str,
        *,
        referents: list[str] | None = None,
        group_id: str = "default",
    ) -> ProbeResult:
        """Probe the graph for entity resonance.

        Detection order:
        1. token-to-entity index lookups
        2. relational noun resolution through stored relationships
        3. bounded `find_entity_candidates()` phrase fallback
        """
        if self._graph_store is None or self._activation_store is None:
            return ProbeResult()

        await self._refresh_group_cache(group_id)

        candidate_scores: dict[str, float] = {}
        entity_lookup: dict[str, Entity] = {}
        match_sources: dict[str, str] = {}
        anchored_entities: set[str] = set()
        probe_terms = _extract_probe_terms(text, referents or [])

        for term in probe_terms:
            for entity in await self._lookup_term(group_id, term):
                entity_lookup[entity.id] = entity
                candidate_scores[entity.id] = max(
                    candidate_scores.get(entity.id, 0.0),
                    _term_match_score(term, entity.name),
                )
                source = _match_source_for_term(term, entity.name)
                match_sources[entity.id] = _prefer_match_source(
                    match_sources.get(entity.id),
                    source,
                )
                if source in {"relational", "exact_phrase"}:
                    anchored_entities.add(entity.id)

        for entity_id, entity in await self._resolve_relational_referents(
            referents or [],
            group_id,
        ):
            entity_lookup[entity_id] = entity
            candidate_scores[entity_id] = max(candidate_scores.get(entity_id, 0.0), 0.52)
            match_sources[entity_id] = _prefer_match_source(
                match_sources.get(entity_id),
                "relational",
            )
            anchored_entities.add(entity_id)

        if not candidate_scores:
            for phrase in _extract_probe_phrases(text):
                for entity in await self._bounded_find_candidates(group_id, phrase):
                    entity_lookup[entity.id] = entity
                    candidate_scores[entity.id] = max(
                        candidate_scores.get(entity.id, 0.0),
                        _term_match_score(phrase, entity.name),
                    )
                    source = _match_source_for_term(phrase, entity.name)
                    match_sources[entity.id] = _prefer_match_source(
                        match_sources.get(entity.id),
                        source,
                    )
                    if source == "exact_phrase":
                        anchored_entities.add(entity.id)

        if not candidate_scores:
            return ProbeResult()

        ranked_ids = [
            entity_id
            for entity_id, _score in sorted(
                candidate_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self._max_detected_entities]
        ]
        activations = await self._activation_store.batch_get(ranked_ids)
        entity_scores: dict[str, float] = {}

        for entity_id in ranked_ids:
            entity_match = entity_lookup.get(entity_id)
            rels = []
            try:
                rels = await self._graph_store.get_relationships(
                    entity_id,
                    direction="both",
                    group_id=group_id,
                )
            except Exception:
                rels = []
            entity_scores[entity_id] = _compose_entity_resonance(
                base_match=candidate_scores.get(entity_id, 0.0),
                relation_count=len(rels),
                activation_state=activations.get(entity_id),
                identity_core=bool(getattr(entity_match, "identity_core", False)),
            )

        detected_entities = [
            entity_id
            for entity_id, _score in sorted(
                entity_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self._max_detected_entities]
        ]
        resonance_score = _aggregate_resonance(
            [entity_scores[entity_id] for entity_id in detected_entities],
        )
        return ProbeResult(
            resonance_score=resonance_score,
            detected_entities=detected_entities,
            entity_scores=entity_scores,
            match_sources=match_sources,
            anchored_entity_ids=sorted(anchored_entities),
        )

    async def _refresh_group_cache(self, group_id: str) -> None:
        try:
            stats = await self._graph_store.get_stats(group_id)
        except Exception:
            return
        version = (
            _coerce_int(stats.get("entity_count", stats.get("entities", 0)))
            if isinstance(stats, dict)
            else 0,
            _coerce_int(stats.get("relationship_count", stats.get("relationships", 0)))
            if isinstance(stats, dict)
            else 0,
        )
        if self._group_versions.get(group_id) == version:
            return
        self._group_versions[group_id] = version
        self._token_cache[group_id] = {}
        self._identity_cache.pop(group_id, None)

    async def _lookup_term(self, group_id: str, term: str) -> list[Entity]:
        cache = self._token_cache.setdefault(group_id, {})
        key = term.lower()
        cached = cache.get(key)
        if cached is not None:
            return cached.entities
        entities = await self._bounded_find_candidates(group_id, term)
        cache[key] = _CacheEntry(entities=entities, refreshed_at=time.time())
        return entities

    async def _bounded_find_candidates(self, group_id: str, term: str) -> list[Entity]:
        try:
            entities = await self._graph_store.find_entity_candidates(
                term,
                group_id,
                limit=self._max_term_candidates,
            )
        except Exception:
            entities = []
        return entities or []

    async def _resolve_relational_referents(
        self,
        referents: list[str],
        group_id: str,
    ) -> list[tuple[str, Entity]]:
        family_names = {
            family
            for referent in referents
            for family, aliases in _RELATIONAL_FAMILY_ALIASES.items()
            if referent.lower() in aliases
        }
        if not family_names or not hasattr(self._graph_store, "get_identity_core_entities"):
            return []

        if group_id not in self._identity_cache:
            try:
                identity_entities = await self._graph_store.get_identity_core_entities(group_id)
            except Exception:
                identity_entities = []
            self._identity_cache[group_id] = [entity.id for entity in identity_entities]

        matched: dict[str, Entity] = {}
        for anchor_id in self._identity_cache.get(group_id, []):
            try:
                rels = await self._graph_store.get_relationships(
                    anchor_id,
                    direction="both",
                    group_id=group_id,
                )
            except Exception:
                continue
            for rel in rels[: self._max_relationships]:
                family = _predicate_family(rel.predicate)
                if family is None or family not in family_names:
                    continue
                neighbor_id = rel.target_id if rel.source_id == anchor_id else rel.source_id
                try:
                    entity = await self._graph_store.get_entity(neighbor_id, group_id)
                except Exception:
                    entity = None
                if entity is None:
                    continue
                entity_type = (entity.entity_type or "").lower()
                if family in {"child", "parent"} and entity_type not in _PERSON_TYPES:
                    continue
                matched[entity.id] = entity
        return list(matched.items())


def _extract_probe_terms(text: str, referents: list[str]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for phrase in _extract_probe_phrases(text):
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            terms.append(phrase)
    for token in _TOKEN_PATTERN.findall(text):
        normalized = token.strip("?!.,;:'\"")
        key = normalized.lower()
        if len(key) < 3 or key in seen:
            continue
        seen.add(key)
        terms.append(normalized)
    for referent in referents:
        key = referent.lower()
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        terms.append(referent)
    return terms


def _extract_probe_phrases(text: str) -> list[str]:
    phrases = [match.group(0).strip() for match in _PHRASE_PATTERN.finditer(text)]
    return _dedupe(phrases)


def _term_match_score(term: str, entity_name: str) -> float:
    term_key = term.lower().strip()
    name_key = entity_name.lower().strip()
    if not term_key or not name_key:
        return 0.0
    if term_key == name_key:
        return 0.72
    if term_key in name_key or name_key in term_key:
        return 0.58
    term_tokens = set(term_key.split())
    name_tokens = set(_TOKEN_PATTERN.findall(name_key))
    if not term_tokens or not name_tokens:
        return 0.0
    overlap = len(term_tokens & name_tokens) / max(1, len(term_tokens))
    return min(0.48, 0.24 + (0.24 * overlap))


def _match_source_for_term(term: str, entity_name: str) -> str:
    term_key = term.lower().strip()
    name_key = entity_name.lower().strip()
    if " " in term_key and term_key == name_key:
        return "exact_phrase"
    if " " in term_key:
        return "phrase"
    if term_key == name_key:
        return "exact_token"
    return "token"


def _prefer_match_source(current: str | None, candidate: str) -> str:
    rank = {
        "relational": 5,
        "exact_phrase": 4,
        "phrase": 3,
        "exact_token": 2,
        "token": 1,
    }
    if current is None:
        return candidate
    return candidate if rank.get(candidate, 0) >= rank.get(current, 0) else current


def _compose_entity_resonance(
    *,
    base_match: float,
    relation_count: int,
    activation_state,
    identity_core: bool,
) -> float:
    relation_bonus = min(0.22, relation_count * 0.035)
    activation_bonus = 0.0
    if activation_state is not None:
        access_count = float(max(0, getattr(activation_state, "access_count", 0)))
        activation_bonus += min(0.18, access_count * 0.03)
        last_accessed = float(max(0.0, getattr(activation_state, "last_accessed", 0.0)))
        if last_accessed > 0:
            age_seconds = max(0.0, time.time() - last_accessed)
            activation_bonus += max(0.0, 0.10 * (1.0 - min(age_seconds / 604800.0, 1.0)))
    if identity_core:
        activation_bonus += 0.08
    return max(0.0, min(base_match + relation_bonus + activation_bonus, 1.0))


def _aggregate_resonance(scores: list[float]) -> float:
    if not scores:
        return 0.0
    total = 1.0
    for score in scores[:3]:
        total *= 1.0 - max(0.0, min(score, 1.0))
    return round(1.0 - total, 4)


def _predicate_family(predicate: str) -> str | None:
    upper = (predicate or "").upper()
    for family, predicates in _RELATIONAL_PREDICATES.items():
        if upper in predicates:
            return family
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
