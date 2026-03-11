"""LoCoMo dataset adapter: loads JSON, converts to Engram episodes and probes.

Based on the LoCoMo benchmark dataset.
Reference: https://github.com/snap-stanford/LoCoMo
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from engram.models.episode import Episode, EpisodeStatus


@dataclass
class LoCoMoProbe:
    """A memory probe from the LoCoMo dataset."""

    probe_id: str
    question: str
    answer: str
    category: str = ""


@dataclass
class LoCoMoConversation:
    """A single LoCoMo conversation with turns and probes."""

    conversation_id: str
    turns: list[dict]  # raw turn dicts
    probes: list[LoCoMoProbe]


def load_locomo_dataset(
    dataset_path: str | Path,
    max_conversations: int | None = None,
) -> list[LoCoMoConversation]:
    """Load LoCoMo dataset from a JSON file.

    Expected format: list of conversations, each with:
      - "conversation_id": str
      - "conversation": list of turn dicts with "text" or "content"
      - "questions": list of probe dicts with "question", "answer", "category"

    Adapts to common LoCoMo JSON layouts.
    """
    path = Path(dataset_path)
    with open(path) as f:
        raw = json.load(f)

    # Handle both list and dict-wrapped formats
    if isinstance(raw, dict):
        if "data" in raw:
            raw = raw["data"]
        elif "conversations" in raw:
            raw = raw["conversations"]
        else:
            raw = list(raw.values())

    conversations: list[LoCoMoConversation] = []
    if not isinstance(raw, list):
        return conversations

    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        if max_conversations and len(conversations) >= max_conversations:
            break

        conv_id = entry.get("conversation_id", entry.get("id", f"conv_{i}"))
        conv_id = str(conv_id)

        # Extract turns
        turns = entry.get(
            "conversation",
            entry.get("turns", entry.get("dialogue", [])),
        )
        if not isinstance(turns, list):
            turns = []

        # Extract probes
        raw_probes = entry.get(
            "questions",
            entry.get("probes", entry.get("memory_probes", [])),
        )
        if not isinstance(raw_probes, list):
            raw_probes = []
        probes: list[LoCoMoProbe] = []
        for j, p in enumerate(raw_probes):
            if not isinstance(p, dict):
                continue
            probes.append(
                LoCoMoProbe(
                    probe_id=p.get("id", f"{conv_id}_q{j}"),
                    question=str(p.get("question", p.get("query", ""))),
                    answer=str(p.get("answer", p.get("ground_truth", ""))),
                    category=str(p.get("category", p.get("type", ""))),
                )
            )

        conversations.append(
            LoCoMoConversation(
                conversation_id=conv_id,
                turns=turns,
                probes=probes,
            )
        )

    return conversations


def conversation_to_episodes(
    conversation: LoCoMoConversation,
    group_id: str = "default",
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
