"""Baseline adapters for the showcase benchmark."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

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
from engram.extraction.extractor import ExtractionResult
from engram.graph_manager import GraphManager
from engram.storage.memory.activation import MemoryActivationStore
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
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in _STOP_WORDS
    ]


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
                (note, score)
                for note, score in ranked
                if note.kind not in {"episode", "summary"}
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
        self._search_index: object | None = None
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
            extractor,
            cfg=self._cfg,
            runtime_mode="showcase",
        )

    async def _build_search_index(self, db_path: str):
        if self._vector_provider == "none":
            search_index = FTS5SearchIndex(db_path)
            await search_index.initialize(db=self._graph_store._db)
            return search_index

        provider = await self._create_embedding_provider(self._vector_provider)
        if provider is None:
            # Keep initialization deterministic: unavailable vector-backed baselines
            # should be reported, not crash the whole showcase run.
            search_index = FTS5SearchIndex(db_path)
            await search_index.initialize(db=self._graph_store._db)
            return search_index
        wrapped = CountingEmbeddingProvider(provider, self.stats)
        self._embedding_provider = wrapped
        fts = FTS5SearchIndex(db_path)
        vectors = SQLiteVectorStore(db_path)
        search_index = HybridSearchIndex(
            fts=fts,
            vector_store=vectors,
            provider=wrapped,
            fts_weight=0.3,
            vec_weight=0.7,
            cfg=self._cfg,
        )
        await search_index.initialize(db=self._graph_store._db)
        return search_index

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
            episode_id = self._refs.get(turn.ref or "")
            if episode_id is None:
                raise ValueError(f"Unknown episode ref for project: {turn.ref}")
            self.stats.projected_turns += 1
            self.stats.bump("project_episode")
            await manager.project_episode(episode_id)
            self._refs[turn.id] = episode_id
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
            intention_id = self._refs.get(turn.ref or "")
            if intention_id is None:
                raise ValueError(f"Unknown intention ref for dismiss: {turn.ref}")
            self.stats.bump("dismiss_intention")
            await manager.dismiss_intention(intention_id, hard=turn.hard_delete)
            self._refs[turn.id] = intention_id
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
            lines.append(
                f"priority={meta.get('priority')} action={meta.get('action_text')}"
            )
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
