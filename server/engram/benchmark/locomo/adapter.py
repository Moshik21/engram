"""LoCoMo dataset adapter: loads JSON, converts to Engram episodes and probes.

Based on the LoCoMo benchmark dataset.
Reference: https://github.com/snap-stanford/LoCoMo
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from engram.models.episode import Episode, EpisodeStatus

LOCOMO_BENCHMARK_GROUP_ID = "locomo_benchmark"


# LoCoMo question categories (Maharana et al.): the numeric labels in the dataset.
LOCOMO_CATEGORIES = {
    "1": "multi_hop",
    "2": "temporal",
    "3": "open_domain",
    "4": "single_hop",
    "5": "adversarial",
}


@dataclass
class LoCoMoProbe:
    """A memory probe from the LoCoMo dataset."""

    probe_id: str
    question: str
    answer: str
    category: str = ""
    evidence: list[str] = field(default_factory=list)


@dataclass
class LoCoMoSession:
    """One dated session of a LoCoMo conversation (a list of speaker turns)."""

    session_id: str
    date: str
    turns: list[dict]  # {speaker, dia_id, text}


@dataclass
class LoCoMoConversation:
    """A single LoCoMo conversation with sessions (real schema) and probes.

    ``turns`` is kept for backward compatibility with turn-level helpers/tests;
    the functional pipeline ingests ``sessions`` (one episode per dated session).
    """

    conversation_id: str
    turns: list[dict]
    probes: list[LoCoMoProbe]
    sessions: list[LoCoMoSession] = field(default_factory=list)


def category_label(category: str) -> str:
    """Map a LoCoMo numeric category to its human label (passthrough if unknown)."""
    return LOCOMO_CATEGORIES.get(str(category).strip(), str(category))


def load_locomo_dataset(
    dataset_path: str | Path,
    max_conversations: int | None = None,
) -> list[LoCoMoConversation]:
    """Load the real LoCoMo schema (e.g. locomo10.json).

    Each entry has:
      - "qa": list of {question, answer, evidence, category}
      - "conversation": {speaker_a, speaker_b, session_<N>: [turns],
                         session_<N>_date_time: str}  for N = 1..K
    Each turn is {speaker, dia_id, text}.
    """
    path = Path(dataset_path)
    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("conversations") or list(raw.values())

    conversations: list[LoCoMoConversation] = []
    if not isinstance(raw, list):
        return conversations

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        if max_conversations and len(conversations) >= max_conversations:
            break

        conv_id = str(entry.get("sample_id", entry.get("conversation_id", f"conv_{i}")))
        conv = entry.get("conversation", {})

        # Sessions in numeric order: session_1, session_2, ...
        sessions: list[LoCoMoSession] = []
        flat_turns: list[dict] = []
        if isinstance(conv, dict):
            def _session_n(key: str) -> int | None:
                parts = key.split("_")
                if not key.startswith("session_") or key.endswith("date_time"):
                    return None
                return int(parts[1]) if parts[1].isdigit() else None

            session_nums = sorted(n for k in conv if (n := _session_n(k)) is not None)
            for n in session_nums:
                turns = conv.get(f"session_{n}")
                if not isinstance(turns, list):
                    continue
                sessions.append(
                    LoCoMoSession(
                        session_id=f"session_{n}",
                        date=str(conv.get(f"session_{n}_date_time", "")),
                        turns=turns,
                    )
                )
                flat_turns.extend(turns)

        # Probes from the qa block.
        raw_probes = entry.get("qa", entry.get("questions", []))
        if not isinstance(raw_probes, list):
            raw_probes = []
        probes: list[LoCoMoProbe] = []
        for j, p in enumerate(raw_probes):
            if not isinstance(p, dict):
                continue
            ev = p.get("evidence", [])
            if isinstance(ev, str):
                ev = [ev]
            probes.append(
                LoCoMoProbe(
                    probe_id=p.get("id", f"{conv_id}_q{j}"),
                    question=str(p.get("question", "")),
                    answer=str(p.get("answer", p.get("adversarial_answer", ""))),
                    category=str(p.get("category", "")),
                    evidence=[str(e) for e in ev] if isinstance(ev, list) else [],
                )
            )

        conversations.append(
            LoCoMoConversation(
                conversation_id=conv_id,
                turns=flat_turns,
                probes=probes,
                sessions=sessions,
            )
        )

    return conversations


def conversation_to_session_contents(
    conversation: LoCoMoConversation,
) -> list[tuple[str, str, str]]:
    """Yield (session_id, date, content) per session for episode ingestion.

    Content is the dated, speaker-prefixed transcript of the session — the same
    shape Engram ingests for other conversational benchmarks.
    """
    out: list[tuple[str, str, str]] = []
    for s in conversation.sessions:
        lines = []
        for t in s.turns:
            if not isinstance(t, dict):
                continue
            speaker = t.get("speaker", "")
            text = t.get("text") or t.get("content") or ""
            lines.append(f"{speaker}: {text}" if speaker else str(text))
        body = "\n".join(lines).strip()
        if not body:
            continue
        content = f"[Conversation from {s.date}]\n{body}" if s.date else body
        out.append((s.session_id, s.date, content))
    return out


def conversation_to_episodes(
    conversation: LoCoMoConversation,
    group_id: str = LOCOMO_BENCHMARK_GROUP_ID,
) -> list[Episode]:
    """Convert a LoCoMo conversation into Engram episodes (one per turn)."""
    episodes = []
    for i, turn in enumerate(conversation.turns):
        if not isinstance(turn, dict):
            turn = cast(dict[str, Any], {"content": str(turn)})

        # Extract text content from various LoCoMo formats
        text = turn.get("text") or turn.get("content") or turn.get("utterance") or str(turn)

        # Prefix with speaker if available
        speaker = turn.get("speaker", turn.get("role", ""))
        if speaker:
            text = f"{speaker}: {text}"

        ep = Episode(
            id=str(uuid.uuid4()),
            content=text,
            source=f"locomo:{conversation.conversation_id}",
            status=EpisodeStatus.COMPLETED,
            group_id=group_id,
        )
        episodes.append(ep)

    return episodes


def probes_to_queries(
    probes: list[LoCoMoProbe],
) -> list[tuple[str, str, str]]:
    """Convert probes to (question, answer, category) tuples."""
    return [(p.question, p.answer, p.category) for p in probes if p.question and p.answer]
