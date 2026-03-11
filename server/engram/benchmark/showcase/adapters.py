"""Baseline adapters for the showcase benchmark."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from engram.benchmark.showcase.answering import synthesize_answer
from engram.benchmark.showcase.models import (
    AdapterCostStats,
    AnswerTask,
    BudgetProfile,
    EvidenceItem,
    ExtractionSpec,
    ScenarioProbe,
    ScenarioTurn,
    estimate_tokens,
)
from engram.config import ActivationConfig
from engram.embeddings.provider import EmbeddingProvider, FastEmbedProvider, VoyageProvider
from engram.extraction.extractor import EntityExtractor, ExtractionResult
from engram.graph_manager import GraphManager
from engram.storage.memory.activation import MemoryActivationStore
from engram.storage.protocols import SearchIndex
from engram.storage.sqlite.graph import SQLiteGraphStore
from engram.storage.sqlite.hybrid_search import HybridSearchIndex
from engram.storage.sqlite.search import FTS5SearchIndex
from engram.storage.sqlite.vectors import SQLiteVectorStore, cosine_similarity

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "before",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "while",
    "who",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOP_WORDS]


def _lexical_score(query: str, text: str) -> float:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(_tokenize(text))
    overlap = len(query_tokens & text_tokens)
    if overlap == 0:
        return 0.0
    phrase_bonus = 0.2 if query.lower() in text.lower() else 0.0
    return (overlap / len(query_tokens)) + phrase_bonus


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    index = (percentile / 100.0) * (len(sorted_vals) - 1)
    lo = int(index)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = index - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _trim_evidence(items: list[EvidenceItem], max_tokens: int) -> list[EvidenceItem]:
    if max_tokens <= 0:
        return []
    trimmed: list[EvidenceItem] = []
    used_tokens = 0
    for item in items:
        remaining = max_tokens - used_tokens
        if remaining <= 0:
            break
        if item.tokens <= remaining:
            trimmed.append(item)
            used_tokens += item.tokens
            continue
        text = item.text[: remaining * 4]
        if not text.strip():
            break
        trimmed_item = EvidenceItem(
            result_type=item.result_type,
            text=text,
            source_id=item.source_id,
            score=item.score,
            metadata=dict(item.metadata),
        )
        trimmed.append(trimmed_item)
        used_tokens += trimmed_item.tokens
        break
    return trimmed


def _format_showcase_relationship(
    source_name: str,
    predicate: str,
    target_name: str,
    *,
    polarity: str,
) -> str:
    if polarity == "negative":
        return f"{source_name} NEGATED({predicate}) {target_name}"
    if polarity == "uncertain":
        return f"{source_name} UNCERTAIN({predicate}) {target_name}"
    return f"{source_name} {predicate} {target_name}"


class DeterministicExtractor:
    """Scenario-driven extractor that returns fixed outputs for known content."""

    def __init__(
        self,
        extractions: dict[str, ExtractionSpec],
        stats: AdapterCostStats,
    ) -> None:
        self._extractions = extractions
        self._stats = stats

    async def extract(self, text: str) -> ExtractionResult:
        self._stats.extraction_calls += 1

        exact = self._extractions.get(text)
        if exact is not None:
            return ExtractionResult(
                entities=list(exact.entities),
                relationships=list(exact.relationships),
            )

        match_spec: ExtractionSpec | None = None
        match_length = -1
        for candidate_text, spec in self._extractions.items():
            if candidate_text in text or text in candidate_text:
                if len(candidate_text) > match_length:
                    match_spec = spec
                    match_length = len(candidate_text)
        if match_spec is None:
            return ExtractionResult(entities=[], relationships=[])
        return ExtractionResult(
            entities=list(match_spec.entities),
            relationships=list(match_spec.relationships),
        )


class CountingEmbeddingProvider(EmbeddingProvider):
    """Provider wrapper that tracks embedding activity on adapter stats."""

    def __init__(self, provider: EmbeddingProvider, stats: AdapterCostStats) -> None:
        self._provider = provider
        self._stats = stats

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if texts:
            self._stats.embedding_calls += len(texts)
        return await self._provider.embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        if text:
            self._stats.embedding_calls += 1
        return await self._provider.embed_query(text)

    def dimension(self) -> int:
        return self._provider.dimension()

    async def close(self) -> None:
        close = getattr(self._provider, "close", None)
        if callable(close):
            await close()


class BaselineAdapter(ABC):
    """Common adapter interface for all showcase baselines."""

    def __init__(
        self,
        name: str,
        family: str,
        *,
        is_ablation: bool = False,
    ) -> None:
        self.name = name
        self.family = family
        self.is_ablation = is_ablation
        self.stats = AdapterCostStats()
        self.available = True
        self.availability_reason: str | None = None

    async def initialize(self) -> None:
        """Optional async setup hook."""

    @abstractmethod
    async def apply_turn(self, turn: ScenarioTurn) -> None:
        """Apply one scenario turn to the adapter state."""

    @abstractmethod
    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        """Return surfaced evidence for a probe."""

    async def run_probe(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        """Backward-compatible alias used by older tests/callers."""
        return await self.retrieve_evidence(probe)

    async def answer_question(
        self,
        task: AnswerTask,
        evidence: list[EvidenceItem],
        *,
        answer_prompt: str,
        answer_model: str,
        answer_provider: str,
    ):
        del answer_prompt, answer_model, answer_provider
        return synthesize_answer(task, evidence)

    def budget_contract(self) -> dict[str, object]:
        return {
            "evidence_budget_source": "scenario_probe",
            "retrieval_limit_source": "scenario_probe",
            "vector_provider_family": "none",
            "answer_prompt_id": "showcase_answer_v2",
        }

    async def close(self) -> None:
        """Optional async teardown hook."""


@dataclass
class _StoredNote:
    note_id: str
    kind: str
    text: str
    order: int
    source: str
    session_id: str | None = None
    active: bool = True
    vector: list[float] | None = None

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class _CanonicalNotebook:
    """Deterministic notebook state used by stronger non-Engram baselines."""

    current_facts: dict[str, str] = field(default_factory=dict)
    corrections: dict[str, str] = field(default_factory=dict)
    open_loops: dict[str, str] = field(default_factory=dict)
    intentions: dict[str, str] = field(default_factory=dict)
    dependencies: dict[str, str] = field(default_factory=dict)
    session_notes: list[str] = field(default_factory=list)

    def ingest(self, note: _StoredNote) -> None:
        text = note.text.strip()
        self.session_notes.append(text)

        if note.kind == "intention":
            key = _stable_key(text)
            self.intentions[key] = text
            return

        open_loop = _extract_open_loop(text)
        if open_loop is not None:
            self.open_loops[_stable_key(open_loop)] = open_loop

        correction = _extract_correction(text)
        if correction is not None:
            key, current_text, correction_text = correction
            self.current_facts[key] = current_text
            self.corrections[key] = correction_text
            return

        structured_facts = _extract_structured_facts(text)
        if structured_facts:
            self.current_facts.update(structured_facts)

        fact = _extract_current_fact(text)
        if fact is not None:
            key, fact_text = fact
            self.current_facts[key] = fact_text

        dependency = _extract_dependency(text)
        if dependency is not None:
            key, dep_text = dependency
            self.dependencies[key] = dep_text

    def render_markdown_sections(self) -> list[_StoredNote]:
        notes: list[_StoredNote] = []
        sections = [
            ("current", "## Current Facts", list(self.current_facts.values())),
            ("correction", "## Corrections", list(self.corrections.values())),
            ("dependency", "## Dependencies", list(self.dependencies.values())),
            ("open_loop", "## Open Loops", list(self.open_loops.values())),
            ("intention", "## Intentions", list(self.intentions.values())),
        ]
        order = 0
        for kind, header, entries in sections:
            if not entries:
                continue
            body = "\n".join(f"- {entry}" for entry in entries[:8])
            notes.append(
                _StoredNote(
                    note_id=f"section_{kind}",
                    kind=kind,
                    text=f"{header}\n{body}",
                    order=order,
                    source="showcase:canonical",
                )
            )
            order += 1
        if self.session_notes:
            body = "\n".join(f"- {entry}" for entry in self.session_notes[-6:])
            notes.append(
                _StoredNote(
                    note_id="section_session",
                    kind="session",
                    text=f"## Session Notes\n{body}",
                    order=order,
                    source="showcase:canonical",
                )
            )
        return notes

    def render_compact_summary(self) -> str:
        lines = ["## Summary"]
        for entry in list(self.current_facts.values())[:4]:
            lines.append(f"- {entry}")
        for entry in list(self.open_loops.values())[:2]:
            lines.append(f"- open: {entry}")
        for entry in list(self.intentions.values())[:2]:
            lines.append(f"- intention: {entry}")
        for entry in list(self.dependencies.values())[:2]:
            lines.append(f"- depends: {entry}")
        return "\n".join(lines)


@dataclass
class _StructuredMemoryRecord:
    record_id: str
    memory_class: str
    key: str
    text: str
    order: int
    source: str
    session_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    entity_names: list[str] = field(default_factory=list)
    active: bool = True

    @property
    def search_text(self) -> str:
        extra = " ".join(self.aliases + self.entity_names)
        return f"{self.text} {extra}".strip()


def _memory_class_for_slot(slot: str) -> str:
    if slot in {"writing_style", "style"}:
        return "preference"
    return "fact"


def _parse_fact_value(text: str) -> str:
    if ": " in text:
        return text.split(": ", 1)[1].strip()
    parts = text.split()
    return parts[-1].strip() if parts else text.strip()


def _records_from_turn(turn: ScenarioTurn, order: int) -> list[_StructuredMemoryRecord]:
    records: list[_StructuredMemoryRecord] = []
    seen_keys: set[str] = set()

    def add_record(
        key: str,
        memory_class: str,
        text: str,
        *,
        aliases: list[str] | None = None,
        entity_names: list[str] | None = None,
    ) -> None:
        if key in seen_keys:
            return
        seen_keys.add(key)
        records.append(
            _StructuredMemoryRecord(
                record_id=f"{turn.id}:{len(records)}",
                memory_class=memory_class,
                key=key,
                text=text.strip(),
                order=order,
                source=turn.source,
                session_id=turn.session_id,
                aliases=list(aliases or []),
                entity_names=list(entity_names or []),
            )
        )

    if turn.action == "intend":
        action_text = turn.action_text or ""
        trigger_text = turn.trigger_text or turn.id
        entity_names = list(turn.entity_names)
        add_record(
            f"intention:{turn.id}",
            "intention",
            (
                f"Intention: {trigger_text}. Action: {action_text}. "
                f"Related entities: {', '.join(entity_names)}"
            ).strip(),
            entity_names=entity_names,
        )
        return records

    if turn.content is None:
        return records

    text = turn.content.strip()
    open_loop = _extract_open_loop(text)
    if open_loop is not None:
        add_record(
            f"open_loop:{_stable_key(open_loop)}",
            "open_loop",
            open_loop,
        )

    dependency = _extract_dependency(text)
    if dependency is not None:
        dep_key, dep_text = dependency
        subject = dep_key.split(":", 1)[0]
        add_record(
            dep_key,
            "dependency",
            dep_text,
            entity_names=[subject],
        )

    correction = _extract_correction(text)
    if correction is not None:
        key, current_text, correction_text = correction
        subject = key.split(":", 1)[0]
        add_record(
            key,
            "correction",
            current_text,
            aliases=[correction_text],
            entity_names=[subject],
        )

    structured_facts = _extract_structured_facts(text)
    if structured_facts:
        for key, fact_text in structured_facts.items():
            subject, slot = key.split(":", 1)
            add_record(
                key,
                _memory_class_for_slot(slot),
                fact_text,
                entity_names=[subject],
            )

    fact = _extract_current_fact(text)
    if fact is not None:
        key, fact_text = fact
        subject, slot = key.split(":", 1)
        add_record(
            key,
            _memory_class_for_slot(slot),
            fact_text,
            entity_names=[subject],
        )

    extraction = turn.extraction
    if extraction is None:
        if not records:
            add_record(
                f"session:{turn.id}",
                "session_note",
                text,
            )
        return records

    for entity in extraction.entities:
        name = str(entity.get("name") or "").strip()
        if not name:
            continue
        summary = str(entity.get("summary") or "").strip()
        if summary:
            add_record(
                f"summary:{name}",
                "fact",
                f"{name} summary: {summary}",
                entity_names=[name],
            )
        attributes = entity.get("attributes") or {}
        for attr, value in attributes.items():
            key = f"{name}:{attr}"
            add_record(
                key,
                _memory_class_for_slot(str(attr)),
                f"{name} {attr}: {value}",
                entity_names=[name],
            )

    for relationship in extraction.relationships:
        source = str(relationship.get("source") or "").strip()
        target = str(relationship.get("target") or "").strip()
        predicate = str(relationship.get("predicate") or "").strip()
        if not source or not target or not predicate:
            continue
        polarity = str(relationship.get("polarity") or "positive")
        record_key = f"relationship:{source}:{predicate}:{target}"
        aliases = [f"{source}:{predicate}", target]
        add_record(
            record_key,
            "correction" if polarity != "positive" else "relation",
            _format_showcase_relationship(
                source,
                predicate,
                target,
                polarity=polarity,
            ),
            aliases=aliases,
            entity_names=[source, target],
        )

    if not records:
        add_record(
            f"session:{turn.id}",
            "session_note",
            text,
        )
    return records


def _record_to_note(record: _StructuredMemoryRecord, *, prefix_class: bool = False) -> _StoredNote:
    text = record.text
    if prefix_class and record.memory_class not in {"open_loop", "dependency", "intention"}:
        text = f"[{record.memory_class}] {text}"
    return _StoredNote(
        note_id=record.record_id,
        kind=record.memory_class,
        text=text,
        order=record.order,
        source=record.source,
        session_id=record.session_id,
        active=record.active,
    )


def _structured_context_text(notes: list[_StoredNote]) -> str:
    lines = ["## Context"]
    for note in notes:
        lines.append(f"- {note.text}")
    return "\n".join(lines)


def _stable_key(text: str) -> str:
    return " ".join(_tokenize(text[:120]))


def _extract_open_loop(text: str) -> str | None:
    if text.lower().startswith("open loop:"):
        return text.split(":", 1)[1].strip()
    return None


def _extract_dependency(text: str) -> tuple[str, str] | None:
    match = re.search(
        r"(?P<subject>.+?) requires (?P<value>.+?) before shipping\.?$",
        text,
        re.IGNORECASE,
    )
    if not match:
        return None
    subject = match.group("subject").strip()
    value = match.group("value").strip()
    return f"{subject}:requires", f"{subject} REQUIRES {value}"


def _extract_structured_facts(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}

    codename = re.search(
        r"(?P<subject>.+?)'s codename is (?P<value>[A-Za-z0-9._-]+)\.?",
        text,
        re.IGNORECASE,
    )
    if codename:
        subject = codename.group("subject").strip()
        value = codename.group("value").strip()
        facts[f"{subject}:codename"] = f"{subject} codename: {value}"

    prefers = re.search(
        r"(?P<subject>.+?) prefers (?P<value>.+?)\.?$",
        text,
        re.IGNORECASE,
    )
    if prefers:
        subject = prefers.group("subject").strip()
        value = prefers.group("value").strip()
        facts[f"{subject}:writing_style"] = f"{subject} writing_style: {value}"

    style = re.search(
        r"(?P<subject>.+?) should stay (?P<value>.+?)\.?$",
        text,
        re.IGNORECASE,
    )
    if style:
        subject = style.group("subject").strip()
        value = style.group("value").strip()
        facts[f"{subject}:style"] = f"{subject} style: {value}"

    board = re.search(
        r"Board decision:\s*(?P<subject>.+?) should (?P<value>.+?)\.?$",
        text,
        re.IGNORECASE,
    )
    if board:
        subject = board.group("subject").strip()
        value = board.group("value").strip()
        facts[f"{subject}:decision"] = f"{subject} decision: {value}"

    status = re.search(
        r"(?:Correction:\s*)?(?P<subject>.+?) status is (?P<value>[A-Za-z0-9._-]+)\.?",
        text,
        re.IGNORECASE,
    )
    if status:
        subject = status.group("subject").replace("Correction:", "").strip()
        value = status.group("value").strip()
        facts[f"{subject}:status"] = f"{subject} status: {value}"

    payments = re.search(
        r"durable facts are that (?P<owner>[A-Za-z][A-Za-z -]+?) owns "
        r"the migration, the deadline is (?P<deadline>[A-Za-z0-9 ]+?), "
        r"and the rollback path is (?P<rollback>.+?)\.?$",
        text,
        re.IGNORECASE,
    )
    if payments:
        facts["Payments Migration:owner"] = (
            f"Payments Migration owner: {payments.group('owner').strip()}"
        )
        facts["Payments Migration:deadline"] = (
            f"Payments Migration deadline: {payments.group('deadline').strip()}"
        )
        facts["Payments Migration:rollback"] = (
            f"Payments Migration rollback: {payments.group('rollback').strip()}"
        )

    return facts


def _extract_correction(text: str) -> tuple[str, str, str] | None:
    base_url = re.search(
        r"(?:Correction:\s*)?(?P<subject>.+?) now uses "
        r"(?P<value>[A-Za-z0-9._-]+) as (?:its )?base URL\.?$",
        text,
        re.IGNORECASE,
    )
    if base_url:
        subject = base_url.group("subject").replace("Correction:", "").strip()
        value = base_url.group("value").strip()
        return (
            f"{subject}:base_url",
            f"{subject} base_url: {value}",
            text.strip(),
        )

    stopped_using = re.search(
        r"(?P<subject>.+?) stopped using (?P<old>[A-Za-z0-9._-]+) "
        r"and now uses (?P<new>[A-Za-z0-9._-]+)\.?",
        text,
        re.IGNORECASE,
    )
    if stopped_using:
        subject = stopped_using.group("subject").strip()
        new_value = stopped_using.group("new").strip()
        return (
            f"{subject}:uses",
            f"{subject} USES {new_value}",
            text.strip(),
        )

    status = re.search(
        r"Correction:\s*(?P<subject>.+?) status is (?P<value>[A-Za-z0-9._-]+)\.?",
        text,
        re.IGNORECASE,
    )
    if status:
        subject = status.group("subject").strip()
        value = status.group("value").strip()
        return (
            f"{subject}:status",
            f"{subject} status: {value}",
            text.strip(),
        )
    return None


def _extract_current_fact(text: str) -> tuple[str, str] | None:
    structured = _extract_structured_facts(text)
    if structured:
        first_key = next(iter(structured))
        return first_key, structured[first_key]

    base_url = re.search(
        r"(?P<subject>.+?) currently uses (?P<value>[A-Za-z0-9._-]+) as (?:its )?base URL\.?$",
        text,
        re.IGNORECASE,
    )
    if base_url:
        subject = base_url.group("subject").strip()
        value = base_url.group("value").strip()
        return f"{subject}:base_url", f"{subject} base_url: {value}"

    uses = re.search(
        r"(?P<subject>.+?) uses (?P<value>[A-Za-z0-9._-]+)(?: for| as|\.|$)",
        text,
        re.IGNORECASE,
    )
    if uses:
        subject = uses.group("subject").strip()
        value = uses.group("value").strip()
        return f"{subject}:uses", f"{subject} USES {value}"
    return None


def _query_prefers_current_state(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in (" now", " current", " currently", " latest"))


class _RawNoteAdapter(BaselineAdapter):
    """Shared logic for raw-memory comparison baselines."""

    def __init__(
        self,
        name: str,
        family: str,
        *,
        history_token_budget: int = 220,
        vector_provider_family: str = "none",
    ) -> None:
        super().__init__(name, family)
        self._history_token_budget = history_token_budget
        self._vector_provider_family = vector_provider_family
        self._notes: list[_StoredNote] = []
        self._refs: dict[str, str] = {}
        self._order = 0

    async def apply_turn(self, turn: ScenarioTurn) -> None:
        if turn.action in {"observe", "remember"}:
            if turn.content:
                self.stats.observed_turns += 1
                note = _StoredNote(
                    note_id=turn.id,
                    kind="episode",
                    text=turn.content,
                    order=self._order,
                    source=turn.source,
                    session_id=turn.session_id,
                )
                self._order += 1
                await self._store_note(note)
                self._refs[turn.id] = note.note_id
            return

        if turn.action == "project":
            return

        if turn.action == "intend":
            note = _StoredNote(
                note_id=turn.id,
                kind="intention",
                text=(
                    f"Intention: {turn.trigger_text or ''}. "
                    f"Action: {turn.action_text or ''}. "
                    f"Related entities: {', '.join(turn.entity_names)}"
                ).strip(),
                order=self._order,
                source="showcase:intention",
            )
            self._order += 1
            await self._store_note(note)
            self._refs[turn.id] = note.note_id
            return

        if turn.action == "dismiss_intention":
            target_id = self._refs.get(turn.ref or "")
            if target_id is None:
                raise ValueError(f"Unknown intention ref: {turn.ref}")
            for note in self._notes:
                if note.note_id == target_id:
                    note.active = False
                    break
            return

        raise ValueError(f"Unsupported turn action for {self.name}: {turn.action}")

    async def _store_note(self, note: _StoredNote) -> None:
        self._notes.append(note)

    def _active_notes(self) -> list[_StoredNote]:
        return [note for note in self._notes if note.active]

    def _visible_notes(self) -> list[_StoredNote]:
        return self._active_notes()

    def _score_note(self, note: _StoredNote, query: str) -> float:
        recency_bonus = min(0.15, 0.01 * max(0, note.order))
        return _lexical_score(query, note.text) + recency_bonus

    def _rank_notes(self, query: str, limit: int) -> list[_StoredNote]:
        visible = self._visible_notes()
        ranked = sorted(
            visible,
            key=lambda note: (self._score_note(note, query), note.order),
            reverse=True,
        )
        return ranked[:limit]

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        search_query = probe.query or probe.topic_hint or ""
        if probe.operation == "get_context":
            ranked = self._rank_notes(search_query, limit=4)
            lines = ["## Context"]
            for note in ranked:
                lines.append(f"- {note.text}")
            context_text = "\n".join(lines)
            return _trim_evidence(
                [EvidenceItem(result_type="context", text=context_text, source_id="context")],
                probe.max_tokens,
            )

        ranked = self._rank_notes(search_query, limit=max(3, probe.limit))
        evidence = [
            EvidenceItem(
                result_type="intention" if note.kind == "intention" else "episode",
                text=note.text,
                source_id=note.note_id,
                score=self._score_note(note, search_query),
            )
            for note in ranked
        ]
        return _trim_evidence(evidence, probe.max_tokens)

    def budget_contract(self) -> dict[str, object]:
        return {
            "evidence_budget_source": "scenario_probe",
            "retrieval_limit_source": "scenario_probe",
            "vector_provider_family": self._vector_provider_family,
            "history_token_budget": self._history_token_budget,
            "answer_prompt_id": "showcase_answer_v2",
        }


class ContextWindowAdapter(_RawNoteAdapter):
    """Recent-turn-only baseline constrained by a fixed working-memory budget."""

    def __init__(self) -> None:
        super().__init__("context_window", "alternative", history_token_budget=180)

    def _visible_notes(self) -> list[_StoredNote]:
        visible: list[_StoredNote] = []
        used = 0
        for note in reversed(self._active_notes()):
            if used + note.tokens > self._history_token_budget:
                break
            visible.append(note)
            used += note.tokens
        visible.reverse()
        return visible

    def _score_note(self, note: _StoredNote, query: str) -> float:
        return _lexical_score(query, note.text) + (0.03 * note.order)


class MarkdownMemoryAdapter(_RawNoteAdapter):
    """Flat timestamped note file baseline with lexical retrieval only."""

    def __init__(self) -> None:
        super().__init__("markdown_memory", "alternative", history_token_budget=260)

    async def _store_note(self, note: _StoredNote) -> None:
        markdown_text = f"- [{note.order:03d}] {note.text}"
        note.text = markdown_text
        self._notes.append(note)


class VectorRagAdapter(_RawNoteAdapter):
    """Chunked raw-turn retrieval using local embeddings plus lexical fallback."""

    def __init__(self) -> None:
        super().__init__(
            "vector_rag",
            "alternative",
            history_token_budget=260,
            vector_provider_family="local",
        )
        self._provider: FastEmbedProvider | None = None

    async def initialize(self) -> None:
        try:
            import asyncio

            self._provider = await asyncio.wait_for(
                asyncio.to_thread(FastEmbedProvider),
                timeout=5.0,
            )
        except Exception as exc:
            self.available = False
            self.availability_reason = f"local embeddings unavailable: {exc}"

    async def _store_note(self, note: _StoredNote) -> None:
        if self._provider is not None:
            self.stats.embedding_calls += 1
            embeddings = await self._provider.embed([note.text])
            if embeddings:
                note.vector = embeddings[0]
        self._notes.append(note)

    def _score_note(self, note: _StoredNote, query: str) -> float:
        lexical = _lexical_score(query, note.text)
        if self._provider is None or note.vector is None:
            return lexical
        return lexical

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        if not self.available:
            return []

        query = probe.query or probe.topic_hint or ""
        query_vector: list[float] | None = None
        if self._provider is not None:
            self.stats.embedding_calls += 1
            query_vector = await self._provider.embed_query(query)

        ranked: list[tuple[_StoredNote, float]] = []
        for note in self._visible_notes():
            lexical = _lexical_score(query, note.text)
            vector_score = 0.0
            if query_vector and note.vector:
                vector_score = max(0.0, cosine_similarity(query_vector, note.vector))
            score = (0.75 * vector_score) + (0.25 * lexical)
            ranked.append((note, score))

        ranked.sort(key=lambda item: (item[1], item[0].order), reverse=True)
        if probe.operation == "get_context":
            context_lines = ["## Context"]
            for note, _score in ranked[:4]:
                context_lines.append(f"- {note.text}")
            return _trim_evidence(
                [
                    EvidenceItem(
                        result_type="context",
                        text="\n".join(context_lines),
                        source_id="context",
                    )
                ],
                probe.max_tokens,
            )

        evidence = [
            EvidenceItem(
                result_type="intention" if note.kind == "intention" else "episode",
                text=note.text,
                source_id=note.note_id,
                score=score,
            )
            for note, score in ranked[: max(3, probe.limit)]
        ]
        return _trim_evidence(evidence, probe.max_tokens)


class ContextSummaryAdapter(_RawNoteAdapter):
    """Recent window plus deterministic rolling summary of older turns."""

    def __init__(self) -> None:
        super().__init__("context_summary", "alternative", history_token_budget=220)
        self._summary_every = 3
        self._recent_note_count = 3
        self._notebook = _CanonicalNotebook()
        self._summary_note: _StoredNote | None = None

    async def _store_note(self, note: _StoredNote) -> None:
        self._notes.append(note)
        self._notebook.ingest(note)
        if len(self._notes) % self._summary_every == 0:
            self._rebuild_summary()

    def _rebuild_summary(self) -> None:
        summary_text = self._notebook.render_compact_summary()
        self._summary_note = _StoredNote(
            note_id="rolling_summary",
            kind="summary",
            text=summary_text,
            order=max((note.order for note in self._notes), default=0) + 1,
            source="showcase:summary",
        )

    def _visible_notes(self) -> list[_StoredNote]:
        recent = self._active_notes()[-self._recent_note_count :]
        visible = list(recent)
        if self._summary_note is not None:
            visible.insert(0, self._summary_note)
        return visible

    def _score_note(self, note: _StoredNote, query: str) -> float:
        score = super()._score_note(note, query)
        if note.kind == "summary":
            score += 0.08
        if _query_prefers_current_state(query) and note.kind == "summary":
            score += 0.12
        return score


class MarkdownCanonicalAdapter(_RawNoteAdapter):
    """Structured Markdown notebook with deterministic latest-win semantics."""

    def __init__(self) -> None:
        super().__init__("markdown_canonical", "alternative", history_token_budget=260)
        self._notebook = _CanonicalNotebook()

    async def _store_note(self, note: _StoredNote) -> None:
        self._notebook.ingest(note)
        note.text = f"- [{note.order:03d}] {note.text}"
        self._notes.append(note)

    def _visible_notes(self) -> list[_StoredNote]:
        return self._notebook.render_markdown_sections() + self._active_notes()[-4:]

    def _score_note(self, note: _StoredNote, query: str) -> float:
        score = super()._score_note(note, query)
        if _query_prefers_current_state(query) and note.kind in {"current", "correction"}:
            score += 0.18
        if "open" in query.lower() and note.kind == "open_loop":
            score += 0.15
        if "remember" in query.lower() and note.kind == "intention":
            score += 0.12
        return score


class HybridRagTemporalAdapter(VectorRagAdapter):
    """Raw chunk retrieval with deterministic temporal/correction filtering."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "hybrid_rag_temporal"
        self._notebook = _CanonicalNotebook()

    async def _store_note(self, note: _StoredNote) -> None:
        self._notebook.ingest(note)
        await super()._store_note(note)

    def _apply_temporal_filter(
        self,
        query: str,
        ranked: list[tuple[_StoredNote, float]],
    ) -> list[tuple[_StoredNote, float]]:
        if not _query_prefers_current_state(query):
            return ranked

        structured = self._notebook.render_markdown_sections()
        if structured:
            ranked = [
                (note, score) for note, score in ranked if note.kind not in {"episode", "summary"}
            ]
            for index, note in enumerate(reversed(structured)):
                ranked.insert(0, (note, 2.0 - (index * 0.1)))
        return ranked

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        if not self.available:
            return []

        query = probe.query or probe.topic_hint or ""
        query_vector: list[float] | None = None
        if self._provider is not None:
            self.stats.embedding_calls += 1
            query_vector = await self._provider.embed_query(query)

        ranked: list[tuple[_StoredNote, float]] = []
        notes = self._visible_notes() + self._notebook.render_markdown_sections()
        seen_ids: set[str] = set()
        for note in notes:
            if note.note_id in seen_ids:
                continue
            seen_ids.add(note.note_id)
            lexical = _lexical_score(query, note.text)
            vector_score = 0.0
            if query_vector and note.vector:
                vector_score = max(0.0, cosine_similarity(query_vector, note.vector))
            section_bonus = 0.0
            if note.kind in {"current", "correction", "dependency"}:
                section_bonus = 0.12
            ranked.append((note, (0.65 * vector_score) + (0.25 * lexical) + section_bonus))

        ranked.sort(key=lambda item: (item[1], item[0].order), reverse=True)
        ranked = self._apply_temporal_filter(query, ranked)

        if probe.operation == "get_context":
            lines = ["## Context"]
            for note, _score in ranked[:4]:
                lines.append(f"- {note.text}")
            return _trim_evidence(
                [EvidenceItem(result_type="context", text="\n".join(lines), source_id="context")],
                probe.max_tokens,
            )

        evidence = [
            EvidenceItem(
                result_type="intention" if note.kind == "intention" else "episode",
                text=note.text,
                source_id=note.note_id,
                score=score,
            )
            for note, score in ranked[: max(3, probe.limit)]
        ]
        return _trim_evidence(evidence, probe.max_tokens)


class LangGraphStoreMemoryAdapter(_RawNoteAdapter):
    """Proxy baseline for LangGraph-style summary plus durable store memory."""

    def __init__(self) -> None:
        super().__init__("langgraph_store_memory", "external_proxy", history_token_budget=220)
        self._recent_note_count = 2
        self._records_by_key: dict[str, _StructuredMemoryRecord] = {}
        self._records_by_id: dict[str, _StructuredMemoryRecord] = {}
        self._summary_note: _StoredNote | None = None

    async def apply_turn(self, turn: ScenarioTurn) -> None:
        await super().apply_turn(turn)
        if turn.action in {"observe", "remember", "intend"}:
            if turn.action in {"observe", "remember"} and turn.content:
                self.stats.extraction_calls += 1
            records = _records_from_turn(turn, max(0, self._order - 1))
            if any(record.memory_class != "session_note" for record in records):
                self.stats.projected_turns += 1
            for record in records:
                self._upsert_record(record)
            self._rebuild_summary()
            return

        if turn.action == "dismiss_intention" and turn.ref:
            key = f"intention:{turn.ref}"
            existing_record = self._records_by_key.get(key)
            if existing_record is not None:
                existing_record.active = False
            self._rebuild_summary()

    def _upsert_record(self, record: _StructuredMemoryRecord) -> None:
        existing = self._records_by_key.get(record.key)
        if existing is not None and existing.order > record.order:
            return
        if existing is not None:
            self._records_by_id.pop(existing.record_id, None)
        self._records_by_key[record.key] = record
        self._records_by_id[record.record_id] = record

    def _active_records(self) -> list[_StructuredMemoryRecord]:
        return [
            record
            for record in self._records_by_key.values()
            if record.active and record.memory_class != "session_note"
        ]

    def _record_score(self, record: _StructuredMemoryRecord, query: str) -> float:
        score = _lexical_score(query, record.search_text)
        if record.memory_class in {"correction", "fact", "preference"}:
            score += 0.08
        if record.memory_class in {"open_loop", "intention"} and "remember" in query.lower():
            score += 0.06
        if _query_prefers_current_state(query) and record.memory_class in {
            "correction",
            "fact",
            "preference",
            "dependency",
        }:
            score += 0.2
        return score

    def _structured_notes(self) -> list[_StoredNote]:
        ranked = sorted(
            self._active_records(),
            key=lambda record: (record.order, record.key),
        )
        notes = [_record_to_note(record) for record in ranked]
        if self._summary_note is not None:
            return [self._summary_note] + notes
        return notes

    def _rebuild_summary(self) -> None:
        records = sorted(
            self._active_records(),
            key=lambda record: (record.order, record.key),
            reverse=True,
        )
        if not records:
            self._summary_note = None
            return
        lines = ["## Thread Summary"]
        for record in records[:5]:
            lines.append(f"- {record.text}")
        self._summary_note = _StoredNote(
            note_id="langgraph_summary",
            kind="summary",
            text="\n".join(lines),
            order=max((record.order for record in records), default=0) + 1,
            source="showcase:langgraph_summary",
        )

    def _visible_notes(self) -> list[_StoredNote]:
        recent = [note for note in self._active_notes() if note.kind == "episode"][
            -self._recent_note_count :
        ]
        return self._structured_notes() + recent

    def _score_note(self, note: _StoredNote, query: str) -> float:
        if note.kind == "summary":
            return _lexical_score(query, note.text) + 0.18
        record = self._records_by_id.get(note.note_id)
        if record is not None:
            return self._record_score(record, query)
        return super()._score_note(note, query)

    def _current_state_notes(self, query: str, limit: int) -> list[_StoredNote]:
        candidates = [
            _record_to_note(record)
            for record in sorted(
                self._active_records(),
                key=lambda record: (self._record_score(record, query), record.order),
                reverse=True,
            )
            if record.memory_class not in {"session_note", "relation"}
        ]
        return candidates[:limit]

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        query = probe.query or probe.topic_hint or ""
        if probe.operation == "get_context" or _query_prefers_current_state(query):
            current_notes = self._current_state_notes(query, limit=max(3, probe.limit))
            if current_notes:
                return _trim_evidence(
                    [
                        EvidenceItem(
                            result_type="context",
                            text=_structured_context_text(current_notes[:4]),
                            source_id=f"{self.name}:context",
                        )
                    ],
                    probe.max_tokens,
                )
        return await super().retrieve_evidence(probe)


class Mem0StyleMemoryAdapter(_RawNoteAdapter):
    """Proxy baseline for extracted memory objects with latest-win updates."""

    def __init__(self) -> None:
        super().__init__("mem0_style_memory", "external_proxy", history_token_budget=220)
        self._records_by_key: dict[str, _StructuredMemoryRecord] = {}
        self._records_by_id: dict[str, _StructuredMemoryRecord] = {}

    async def apply_turn(self, turn: ScenarioTurn) -> None:
        await super().apply_turn(turn)
        if turn.action in {"observe", "remember", "intend"}:
            if turn.action in {"observe", "remember"} and turn.content:
                self.stats.extraction_calls += 1
            records = _records_from_turn(turn, max(0, self._order - 1))
            if any(record.memory_class != "session_note" for record in records):
                self.stats.projected_turns += 1
            for record in records:
                self._upsert_record(record)
            return

        if turn.action == "dismiss_intention" and turn.ref:
            key = f"intention:{turn.ref}"
            existing_record = self._records_by_key.get(key)
            if existing_record is not None:
                existing_record.active = False

    def _upsert_record(self, record: _StructuredMemoryRecord) -> None:
        existing = self._records_by_key.get(record.key)
        if existing is not None and existing.order > record.order:
            return
        if existing is not None:
            self._records_by_id.pop(existing.record_id, None)
        self._records_by_key[record.key] = record
        self._records_by_id[record.record_id] = record

    def _active_records(self) -> list[_StructuredMemoryRecord]:
        return [record for record in self._records_by_key.values() if record.active]

    def _record_score(self, record: _StructuredMemoryRecord, query: str) -> float:
        score = _lexical_score(query, record.search_text)
        if record.memory_class in {"correction", "fact", "preference"}:
            score += 0.14
        if record.memory_class == "dependency":
            score += 0.1
        if "open" in query.lower() and record.memory_class == "open_loop":
            score += 0.22
        if "remember" in query.lower() and record.memory_class == "intention":
            score += 0.18
        if _query_prefers_current_state(query) and record.memory_class in {
            "correction",
            "fact",
            "preference",
        }:
            score += 0.26
        return score

    def _memory_notes(self) -> list[_StoredNote]:
        records = sorted(
            self._active_records(),
            key=lambda record: (record.order, record.key),
            reverse=True,
        )
        return [_record_to_note(record, prefix_class=True) for record in records]

    def _visible_notes(self) -> list[_StoredNote]:
        return self._memory_notes()

    def _score_note(self, note: _StoredNote, query: str) -> float:
        record = self._records_by_id.get(note.note_id)
        if record is not None:
            return self._record_score(record, query)
        return super()._score_note(note, query)

    def _current_state_notes(self, query: str, limit: int) -> list[_StoredNote]:
        records = [
            record
            for record in sorted(
                self._active_records(),
                key=lambda record: (self._record_score(record, query), record.order),
                reverse=True,
            )
            if record.memory_class not in {"session_note", "relation", "intention"}
        ]
        return [_record_to_note(record, prefix_class=True) for record in records[:limit]]

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        query = probe.query or probe.topic_hint or ""
        if probe.operation == "get_context" or _query_prefers_current_state(query):
            notes = self._current_state_notes(query, limit=max(3, probe.limit))
            if notes:
                return _trim_evidence(
                    [
                        EvidenceItem(
                            result_type="context",
                            text=_structured_context_text(notes[:4]),
                            source_id=f"{self.name}:context",
                        )
                    ],
                    probe.max_tokens,
                )

        ranked = sorted(
            self._active_records(),
            key=lambda record: (self._record_score(record, query), record.order),
            reverse=True,
        )
        evidence = [
            EvidenceItem(
                result_type="intention" if record.memory_class == "intention" else "episode",
                text=record.text,
                source_id=record.record_id,
                score=self._record_score(record, query),
            )
            for record in ranked[: max(3, probe.limit)]
        ]
        return _trim_evidence(evidence, probe.max_tokens)


@dataclass
class _GraphNodeState:
    name: str
    entity_type: str
    summary: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    attribute_order: dict[str, int] = field(default_factory=dict)
    order: int = 0
    vector: list[float] | None = None

    @property
    def text(self) -> str:
        lines = [f"{self.name} ({self.entity_type})"]
        if self.summary:
            lines.append(self.summary)
        for attr, value in sorted(self.attributes.items()):
            lines.append(f"{self.name} {attr}: {value}")
        return "\n".join(lines)


@dataclass
class _GraphEdgeState:
    source: str
    target: str
    predicate: str
    polarity: str
    order: int


class GraphitiTemporalGraphAdapter(VectorRagAdapter):
    """Proxy baseline for temporal graph memory with 2-hop expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "graphiti_temporal_graph"
        self.family = "external_proxy"
        self._nodes: dict[str, _GraphNodeState] = {}
        self._edges: list[_GraphEdgeState] = []
        self._intention_state: dict[str, dict[str, object]] = {}

    async def apply_turn(self, turn: ScenarioTurn) -> None:
        await super().apply_turn(turn)
        order = max(0, self._order - 1)

        if turn.action == "dismiss_intention" and turn.ref:
            state = self._intention_state.get(turn.ref)
            if state is not None:
                state["active"] = False
            return

        if turn.action == "intend":
            self.stats.projected_turns += 1
            self._intention_state[turn.id] = {
                "active": True,
                "action_text": turn.action_text or "",
                "trigger_text": turn.trigger_text or turn.id,
                "entity_names": list(turn.entity_names),
            }
            return

        if turn.action not in {"observe", "remember"} or not turn.content:
            return

        self.stats.extraction_calls += 1
        updated = False
        extraction = turn.extraction
        if extraction is not None:
            for entity in extraction.entities:
                updated = await self._upsert_entity(entity, order) or updated
            for relationship in extraction.relationships:
                source = str(relationship.get("source") or "").strip()
                target = str(relationship.get("target") or "").strip()
                predicate = str(relationship.get("predicate") or "").strip()
                polarity = str(relationship.get("polarity") or "positive")
                if source and target and predicate:
                    self._ensure_node(source, order=order)
                    self._ensure_node(target, order=order)
                    self._edges.append(
                        _GraphEdgeState(
                            source=source,
                            target=target,
                            predicate=predicate,
                            polarity=polarity,
                            order=order,
                        )
                    )
                    updated = True

        updated = await self._ingest_graph_heuristics(turn.content, order) or updated
        if updated:
            self.stats.projected_turns += 1

    def _ensure_node(
        self,
        name: str,
        *,
        entity_type: str = "Concept",
        summary: str = "",
        order: int = 0,
    ) -> _GraphNodeState:
        node = self._nodes.get(name)
        if node is None:
            node = _GraphNodeState(name=name, entity_type=entity_type, summary=summary, order=order)
            self._nodes[name] = node
            return node
        if summary and len(summary) > len(node.summary):
            node.summary = summary
        if order > node.order:
            node.order = order
        if entity_type and node.entity_type == "Concept":
            node.entity_type = entity_type
        return node

    async def _upsert_entity(self, entity: dict, order: int) -> bool:
        name = str(entity.get("name") or "").strip()
        if not name:
            return False
        node = self._ensure_node(
            name,
            entity_type=str(entity.get("entity_type") or "Concept"),
            summary=str(entity.get("summary") or ""),
            order=order,
        )
        updated = True
        for attr, value in (entity.get("attributes") or {}).items():
            attr_name = str(attr)
            value_text = str(value)
            last_order = node.attribute_order.get(attr_name, -1)
            if order >= last_order:
                node.attributes[attr_name] = value_text
                node.attribute_order[attr_name] = order
        if self._provider is not None:
            self.stats.embedding_calls += 1
            embeddings = await self._provider.embed([node.text])
            if embeddings:
                node.vector = embeddings[0]
        return updated

    async def _ingest_graph_heuristics(self, text: str, order: int) -> bool:
        updated = False
        fact = _extract_current_fact(text)
        correction = _extract_correction(text)
        dependency = _extract_dependency(text)
        if correction is not None:
            key, current_text, _correction_text = correction
            subject, slot = key.split(":", 1)
            node = self._ensure_node(subject, order=order)
            node.attributes[slot] = _parse_fact_value(current_text)
            node.attribute_order[slot] = order
            updated = True
        elif fact is not None:
            key, fact_text = fact
            subject, slot = key.split(":", 1)
            node = self._ensure_node(subject, order=order)
            node.attributes[slot] = _parse_fact_value(fact_text)
            node.attribute_order[slot] = order
            updated = True

        if dependency is not None:
            dep_key, dep_text = dependency
            subject = dep_key.split(":", 1)[0]
            required = dep_text.split(" REQUIRES ", 1)[1]
            self._ensure_node(subject, order=order)
            self._ensure_node(required, order=order)
            self._edges.append(
                _GraphEdgeState(
                    source=subject,
                    target=required,
                    predicate="REQUIRES",
                    polarity="positive",
                    order=order,
                )
            )
            updated = True
        return updated

    def _seed_nodes(self, query: str, query_vector: list[float] | None) -> list[tuple[str, float]]:
        ranked: list[tuple[str, float]] = []
        for name, node in self._nodes.items():
            lexical = max(
                _lexical_score(query, name),
                _lexical_score(query, node.text),
            )
            vector_score = 0.0
            if query_vector and node.vector:
                vector_score = max(0.0, cosine_similarity(query_vector, node.vector))
            score = (0.75 * lexical) + (0.25 * vector_score)
            if score > 0.0:
                ranked.append((name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:3]

    def _current_edges(self, source: str) -> list[_GraphEdgeState]:
        latest: dict[tuple[str, str], _GraphEdgeState] = {}
        for edge in self._edges:
            if edge.source != source:
                continue
            latest[(edge.predicate, edge.target)] = edge
        current = [edge for edge in latest.values() if edge.polarity == "positive"]
        current.sort(key=lambda edge: (edge.order, edge.predicate, edge.target), reverse=True)
        return current

    def _current_state_nodes(self, query: str) -> list[_GraphNodeState]:
        matches = [
            (node, max(_lexical_score(query, node.name), _lexical_score(query, node.text)))
            for node in self._nodes.values()
        ]
        ranked = [item for item in matches if item[1] > 0.0]
        ranked.sort(key=lambda item: (item[1], item[0].order), reverse=True)
        return [node for node, _score in ranked[:4]]

    async def _fallback_note_evidence(
        self,
        probe: ScenarioProbe,
        query: str,
        query_vector: list[float] | None,
    ) -> list[EvidenceItem]:
        ranked: list[tuple[_StoredNote, float]] = []
        for note in self._visible_notes():
            lexical = _lexical_score(query, note.text)
            vector_score = 0.0
            if query_vector and note.vector:
                vector_score = max(0.0, cosine_similarity(query_vector, note.vector))
            score = (0.65 * vector_score) + (0.35 * lexical)
            if score > 0.0:
                ranked.append((note, score))
        ranked.sort(key=lambda item: (item[1], item[0].order), reverse=True)
        evidence = [
            EvidenceItem(
                result_type="intention" if note.kind == "intention" else "episode",
                text=note.text,
                source_id=note.note_id,
                score=score,
            )
            for note, score in ranked[: max(3, probe.limit)]
        ]
        return _trim_evidence(evidence, probe.max_tokens)

    def _entity_evidence(
        self,
        node: _GraphNodeState,
        *,
        score: float,
        path_note: str | None = None,
    ) -> EvidenceItem:
        lines = [f"{node.name} ({node.entity_type})"]
        if node.summary:
            lines.append(node.summary)
        for attr, value in sorted(node.attributes.items()):
            lines.append(f"{node.name} {attr}: {value}")
        for edge in self._current_edges(node.name):
            lines.append(
                _format_showcase_relationship(
                    edge.source,
                    edge.predicate,
                    edge.target,
                    polarity=edge.polarity,
                )
            )
        if path_note:
            lines.append(path_note)
        return EvidenceItem(
            result_type="entity",
            text="\n".join(lines),
            source_id=node.name,
            score=score,
        )

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        if not self.available:
            return []

        query = probe.query or probe.topic_hint or ""
        query_vector: list[float] | None = None
        if self._provider is not None:
            self.stats.embedding_calls += 1
            query_vector = await self._provider.embed_query(query)

        if probe.operation == "get_context" or _query_prefers_current_state(query):
            current_nodes = self._current_state_nodes(query)
            if current_nodes:
                return _trim_evidence(
                    [
                        EvidenceItem(
                            result_type="context",
                            text=_structured_context_text(
                                [
                                    _StoredNote(
                                        note_id=node.name,
                                        kind="entity",
                                        text=self._entity_evidence(node, score=1.0).text,
                                        order=node.order,
                                        source="showcase:graphiti_context",
                                    )
                                    for node in current_nodes[:3]
                                ]
                            ),
                            source_id=f"{self.name}:context",
                        )
                    ],
                    probe.max_tokens,
                )

        seeds = self._seed_nodes(query, query_vector)
        if not seeds:
            return await self._fallback_note_evidence(probe, query, query_vector)

        scored_nodes: dict[str, tuple[float, str | None]] = {}
        for seed_name, seed_score in seeds:
            scored_nodes[seed_name] = max(
                scored_nodes.get(seed_name, (0.0, None)),
                (seed_score + 0.3, None),
            )
            frontier = [(seed_name, 0)]
            seen = {seed_name}
            while frontier:
                current_name, depth = frontier.pop(0)
                if depth >= 2:
                    continue
                for edge in self._current_edges(current_name):
                    if edge.target in seen:
                        continue
                    seen.add(edge.target)
                    path_bonus = 0.22 if depth == 0 else 0.14
                    path_note = f"Path: {seed_name} -> {current_name} -> {edge.target}"
                    existing = scored_nodes.get(edge.target, (0.0, None))
                    candidate_score = seed_score + path_bonus
                    if candidate_score > existing[0]:
                        scored_nodes[edge.target] = (candidate_score, path_note)
                    frontier.append((edge.target, depth + 1))

        evidence = [
            self._entity_evidence(self._nodes[name], score=score, path_note=path_note)
            for name, (score, path_note) in sorted(
                scored_nodes.items(),
                key=lambda item: item[1][0],
                reverse=True,
            )
            if name in self._nodes
        ]
        if evidence:
            return _trim_evidence(evidence[: max(3, probe.limit)], probe.max_tokens)
        return await self._fallback_note_evidence(probe, query, query_vector)


class EngramAdapter(BaselineAdapter):
    """Shared adapter implementation for Engram-backed baselines."""

    def __init__(
        self,
        name: str,
        cfg: ActivationConfig,
        extractions: dict[str, ExtractionSpec],
        *,
        family: str = "engram",
        is_ablation: bool = False,
        vector_provider: str = "none",
        budget_profile: BudgetProfile | None = None,
    ) -> None:
        super().__init__(name, family, is_ablation=is_ablation)
        self._cfg = cfg
        self._extractions = extractions
        self._vector_provider = vector_provider
        self._budget_profile = budget_profile or BudgetProfile()
        self._temp_dir: Path | None = None
        self._graph_store: SQLiteGraphStore | None = None
        self._search_index: SearchIndex | None = None
        self._manager: GraphManager | None = None
        self._refs: dict[str, str] = {}
        self._embedding_provider: CountingEmbeddingProvider | None = None

    async def initialize(self) -> None:
        self._temp_dir = Path(tempfile.mkdtemp(prefix=f"engram_showcase_{self.name}_"))
        db_path = str(self._temp_dir / "showcase.db")
        self._graph_store = SQLiteGraphStore(db_path)
        await self._graph_store.initialize()
        activation_store = MemoryActivationStore(cfg=self._cfg)
        self._search_index = await self._build_search_index(db_path)
        extractor = DeterministicExtractor(self._extractions, self.stats)
        self._manager = GraphManager(
            self._graph_store,
            activation_store,
            self._search_index,
            cast(EntityExtractor, extractor),
            cfg=self._cfg,
            runtime_mode="showcase",
        )

    async def _build_search_index(self, db_path: str) -> SearchIndex:
        graph_store = self._graph_store
        assert graph_store is not None
        if self._vector_provider == "none":
            fts_search = FTS5SearchIndex(db_path)
            await fts_search.initialize(db=graph_store._db)
            return cast(SearchIndex, fts_search)

        provider = await self._create_embedding_provider(self._vector_provider)
        if provider is None:
            # Keep initialization deterministic: unavailable vector-backed baselines
            # should be reported, not crash the whole showcase run.
            fts_search = FTS5SearchIndex(db_path)
            await fts_search.initialize(db=graph_store._db)
            return cast(SearchIndex, fts_search)
        wrapped = CountingEmbeddingProvider(provider, self.stats)
        self._embedding_provider = wrapped
        fts = FTS5SearchIndex(db_path)
        vectors = SQLiteVectorStore(db_path)
        hybrid_search = HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=wrapped,
            fts_weight=0.3,
            vec_weight=0.7,
            cfg=self._cfg,
        )
        await hybrid_search.initialize(db=graph_store._db)
        return cast(SearchIndex, hybrid_search)

    async def _create_embedding_provider(self, provider_name: str) -> EmbeddingProvider | None:
        import asyncio

        if provider_name == "local":
            return await asyncio.wait_for(
                asyncio.to_thread(FastEmbedProvider),
                timeout=5.0,
            )

        if provider_name == "voyage":
            api_key = os.environ.get("VOYAGE_API_KEY", "")
            if not api_key:
                self.available = False
                self.availability_reason = "VOYAGE_API_KEY not set"
                return None
            return VoyageProvider(api_key=api_key)

        if provider_name == "auto":
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(FastEmbedProvider),
                    timeout=5.0,
                )
            except Exception:
                api_key = os.environ.get("VOYAGE_API_KEY", "")
                if api_key:
                    return VoyageProvider(api_key=api_key)
                self.available = False
                self.availability_reason = "no local or voyage embeddings available"
                return None

        self.available = False
        self.availability_reason = f"unknown vector provider: {provider_name}"
        return None

    @property
    def manager(self) -> GraphManager:
        if self._manager is None:
            raise RuntimeError("Engram adapter is not initialized")
        return self._manager

    async def apply_turn(self, turn: ScenarioTurn) -> None:
        manager = self.manager
        if turn.action == "observe":
            if turn.content is None:
                raise ValueError("observe turn requires content")
            self.stats.observed_turns += 1
            self.stats.bump("store_episode")
            episode_id = await manager.store_episode(
                turn.content,
                source=turn.source,
                session_id=turn.session_id,
            )
            self._refs[turn.id] = episode_id
            return

        if turn.action == "remember":
            if turn.content is None:
                raise ValueError("remember turn requires content")
            self.stats.observed_turns += 1
            self.stats.projected_turns += 1
            self.stats.bump("ingest_episode")
            episode_id = await manager.ingest_episode(
                turn.content,
                source=turn.source,
                session_id=turn.session_id,
            )
            self._refs[turn.id] = episode_id
            return

        if turn.action == "project":
            project_episode_id = self._refs.get(turn.ref or "")
            if project_episode_id is None:
                raise ValueError(f"Unknown episode ref for project: {turn.ref}")
            self.stats.projected_turns += 1
            self.stats.bump("project_episode")
            await manager.project_episode(project_episode_id)
            self._refs[turn.id] = project_episode_id
            return

        if turn.action == "intend":
            self.stats.bump("create_intention")
            intention_id = await manager.create_intention(
                trigger_text=turn.trigger_text or turn.id,
                action_text=turn.action_text or "",
                trigger_type=turn.trigger_type,
                entity_names=turn.entity_names,
                threshold=turn.threshold,
                priority=turn.priority,
                context=turn.context,
                see_also=turn.see_also,
            )
            self._refs[turn.id] = intention_id
            return

        if turn.action == "dismiss_intention":
            dismiss_intention_id = self._refs.get(turn.ref or "")
            if dismiss_intention_id is None:
                raise ValueError(f"Unknown intention ref for dismiss: {turn.ref}")
            self.stats.bump("dismiss_intention")
            await manager.dismiss_intention(dismiss_intention_id, hard=turn.hard_delete)
            self._refs[turn.id] = dismiss_intention_id
            return

        raise ValueError(f"Unsupported turn action for {self.name}: {turn.action}")

    async def retrieve_evidence(self, probe: ScenarioProbe) -> list[EvidenceItem]:
        if probe.operation == "get_context":
            self.stats.bump("get_context")
            context = await self.manager.get_context(
                max_tokens=probe.max_tokens,
                topic_hint=probe.topic_hint or probe.query,
            )
            return [
                EvidenceItem(
                    result_type="context",
                    text=context["context"],
                    source_id="context",
                    tokens=int(context.get("token_estimate", estimate_tokens(context["context"]))),
                )
            ]

        self.stats.bump("recall")
        results = await self.manager.recall(
            query=probe.query or "",
            limit=max(probe.limit, 5),
            interaction_type="used",
            interaction_source="showcase_recall",
        )
        evidence = []
        for result in results:
            evidence.append(await self._result_to_evidence(result))
        return _trim_evidence(evidence, probe.max_tokens)

    def budget_contract(self) -> dict[str, object]:
        provider_family = self._vector_provider if self._vector_provider != "none" else "none"
        return {
            "evidence_budget_source": "scenario_probe",
            "retrieval_limit_source": "scenario_probe",
            "vector_provider_family": provider_family,
            "answer_prompt_id": "showcase_answer_v2",
            "integration_profile": self._cfg.integration_profile,
            "budget_profile": {
                "retrieval_limit": self._budget_profile.retrieval_limit,
                "evidence_max_tokens": self._budget_profile.evidence_max_tokens,
                "answer_budget_tokens": self._budget_profile.answer_budget_tokens,
            },
        }

    async def _result_to_evidence(self, result: dict) -> EvidenceItem:
        result_type = result.get("result_type", "entity")
        if result_type == "cue_episode":
            cue = result.get("cue", {})
            episode = result.get("episode", {})
            text = cue.get("cue_text") or ""
            supporting = cue.get("supporting_spans") or []
            if supporting:
                text = f"{text}\n" + "\n".join(str(span) for span in supporting)
            return EvidenceItem(
                result_type="cue_episode",
                text=text,
                source_id=episode.get("id"),
                score=float(result.get("score", 0.0)),
            )

        if result_type == "episode":
            episode = result.get("episode", {})
            return EvidenceItem(
                result_type="episode",
                text=episode.get("content", ""),
                source_id=episode.get("id"),
                score=float(result.get("score", 0.0)),
            )

        entity = result.get("entity", {})
        lines = [f"{entity.get('name', '')} ({entity.get('type', '')})"]
        summary = entity.get("summary")
        if summary:
            lines.append(summary)
        for rel in result.get("relationships", []):
            source_name = await self.manager.resolve_entity_name(rel["source_id"], "default")
            target_name = await self.manager.resolve_entity_name(rel["target_id"], "default")
            lines.append(
                _format_showcase_relationship(
                    source_name,
                    rel["predicate"],
                    target_name,
                    polarity=str(rel.get("polarity") or "positive"),
                )
            )
        if result.get("intention_meta"):
            meta = result["intention_meta"]
            lines.append(f"priority={meta.get('priority')} action={meta.get('action_text')}")
        return EvidenceItem(
            result_type="entity",
            text="\n".join(line for line in lines if line),
            source_id=entity.get("id"),
            score=float(result.get("score", 0.0)),
        )

    async def close(self) -> None:
        if self._embedding_provider is not None:
            await self._embedding_provider.close()
        if self._graph_store is not None:
            await self._graph_store.close()
        if self._temp_dir is not None and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)


def create_primary_adapter(
    baseline_name: str,
    extractions: dict[str, ExtractionSpec],
    *,
    engram_vector_provider: str = "none",
    budget_profile: BudgetProfile | None = None,
) -> BaselineAdapter:
    """Factory for primary baselines."""
    if baseline_name == "engram_full":
        return EngramAdapter(
            "engram_full",
            ActivationConfig(integration_profile="rework"),
            extractions,
            budget_profile=budget_profile,
        )
    if baseline_name == "engram_full_hybrid":
        provider = engram_vector_provider if engram_vector_provider != "none" else "auto"
        return EngramAdapter(
            "engram_full_hybrid",
            ActivationConfig(integration_profile="rework"),
            extractions,
            vector_provider=provider,
            budget_profile=budget_profile,
        )
    if baseline_name == "langgraph_store_memory":
        return LangGraphStoreMemoryAdapter()
    if baseline_name == "mem0_style_memory":
        return Mem0StyleMemoryAdapter()
    if baseline_name == "graphiti_temporal_graph":
        return GraphitiTemporalGraphAdapter()
    if baseline_name == "context_summary":
        return ContextSummaryAdapter()
    if baseline_name == "markdown_canonical":
        return MarkdownCanonicalAdapter()
    if baseline_name == "hybrid_rag_temporal":
        return HybridRagTemporalAdapter()
    if baseline_name == "context_window":
        return ContextWindowAdapter()
    if baseline_name == "markdown_memory":
        return MarkdownMemoryAdapter()
    if baseline_name == "vector_rag":
        return VectorRagAdapter()
    raise ValueError(f"Unknown primary baseline: {baseline_name}")


def create_ablation_adapter(
    baseline_name: str,
    extractions: dict[str, ExtractionSpec],
) -> BaselineAdapter:
    """Factory for Engram ablation baselines."""
    if baseline_name == "engram_no_cues":
        return EngramAdapter(
            "engram_no_cues",
            ActivationConfig(
                integration_profile="off",
                consolidation_profile="standard",
                recall_profile="all",
                cue_layer_enabled=False,
                cue_recall_enabled=False,
                cue_policy_learning_enabled=False,
                targeted_projection_enabled=False,
                projector_v2_enabled=False,
                projection_planner_enabled=False,
            ),
            extractions,
            family="ablation",
            is_ablation=True,
        )
    if baseline_name == "engram_search_only":
        return EngramAdapter(
            "engram_search_only",
            ActivationConfig(
                integration_profile="off",
                recall_profile="off",
                consolidation_profile="off",
                cue_layer_enabled=False,
                cue_recall_enabled=False,
                multi_pool_enabled=False,
                weight_semantic=1.0,
                weight_activation=0.0,
                weight_spreading=0.0,
                weight_edge_proximity=0.0,
                working_memory_enabled=False,
                recall_planner_enabled=False,
                conv_context_enabled=False,
                prospective_memory_enabled=False,
                prospective_graph_embedded=False,
                exploration_weight=0.0,
                rediscovery_weight=0.0,
                ts_enabled=False,
            ),
            extractions,
            family="ablation",
            is_ablation=True,
        )
    raise ValueError(f"Unknown ablation baseline: {baseline_name}")


def build_extraction_map(turns: list[ScenarioTurn]) -> dict[str, ExtractionSpec]:
    """Extract the deterministic extraction fixtures used by Engram adapters."""
    mapping: dict[str, ExtractionSpec] = {}
    for turn in turns:
        if turn.content and turn.extraction is not None:
            mapping[turn.content] = turn.extraction
    return mapping


def latency_percentiles(results: list[float]) -> tuple[float, float]:
    """Convenience helper for latency reporting."""
    return _percentile(results, 50.0), _percentile(results, 95.0)
