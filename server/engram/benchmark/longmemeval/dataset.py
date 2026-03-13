"""LongMemEval dataset loader: downloads from HuggingFace or loads local JSON."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# HuggingFace URLs for the cleaned dataset
_HF_BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
_VARIANT_FILES = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Canonical question types
QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
]


@dataclass
class LongMemEvalTurn:
    """A single dialogue turn within a session."""

    role: str
    content: str
    has_answer: bool = False


@dataclass
class LongMemEvalSession:
    """A single conversation session in the haystack."""

    session_id: str
    date: str
    turns: list[LongMemEvalTurn]

    @property
    def text(self) -> str:
        """Render session as a formatted dialogue transcript."""
        lines = []
        for turn in self.turns:
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        return "\n".join(lines)


@dataclass
class LongMemEvalInstance:
    """A single LongMemEval question with its haystack."""

    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[LongMemEvalSession]
    answer_session_ids: list[str]

    @property
    def is_abstention(self) -> bool:
        return self.question_id.endswith("_abs")

    @property
    def num_sessions(self) -> int:
        return len(self.sessions)

    @property
    def evidence_sessions(self) -> list[LongMemEvalSession]:
        ids = set(self.answer_session_ids)
        return [s for s in self.sessions if s.session_id in ids]


@dataclass
class LongMemEvalDataset:
    """A loaded LongMemEval dataset variant."""

    variant: str
    instances: list[LongMemEvalInstance]

    @property
    def question_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for inst in self.instances:
            counts[inst.question_type] = counts.get(inst.question_type, 0) + 1
        return counts

    @property
    def abstention_count(self) -> int:
        return sum(1 for inst in self.instances if inst.is_abstention)

    def filter_types(self, types: list[str]) -> LongMemEvalDataset:
        filtered = [i for i in self.instances if i.question_type in types]
        return LongMemEvalDataset(variant=self.variant, instances=filtered)

    def subset(self, n: int, *, seed: int = 42) -> LongMemEvalDataset:
        """Return a deterministic subset of n instances."""
        import random

        rng = random.Random(seed)
        sampled = rng.sample(self.instances, min(n, len(self.instances)))
        return LongMemEvalDataset(variant=self.variant, instances=sampled)

    def stratified_subset(self, n_per_type: int, *, seed: int = 42) -> LongMemEvalDataset:
        """Return a stratified subset with n_per_type instances per question type."""
        import random

        rng = random.Random(seed)
        by_type: dict[str, list[LongMemEvalInstance]] = {}
        for inst in self.instances:
            by_type.setdefault(inst.question_type, []).append(inst)

        sampled: list[LongMemEvalInstance] = []
        for qtype in sorted(by_type.keys()):
            pool = by_type[qtype]
            sampled.extend(rng.sample(pool, min(n_per_type, len(pool))))
        return LongMemEvalDataset(variant=self.variant, instances=sampled)


def _parse_turns(raw_turns: list[Any]) -> list[LongMemEvalTurn]:
    """Parse turn dicts from various formats."""
    turns = []
    for t in raw_turns:
        if isinstance(t, dict):
            role = t.get("role", t.get("speaker", "user"))
            content = t.get("content", t.get("text", t.get("utterance", "")))
            has_answer = bool(t.get("has_answer", False))
            turns.append(
                LongMemEvalTurn(
                    role=str(role),
                    content=str(content),
                    has_answer=has_answer,
                )
            )
        elif isinstance(t, str):
            turns.append(LongMemEvalTurn(role="user", content=t))
    return turns


def _parse_sessions(
    raw_sessions: Any,
    session_ids: list[str] | None = None,
    session_dates: list[str] | None = None,
) -> list[LongMemEvalSession]:
    """Parse sessions from various LongMemEval JSON formats."""
    sessions: list[LongMemEvalSession] = []

    if isinstance(raw_sessions, dict):
        # Format: {"session_id": [turns, ...]}
        for sid, turns_data in raw_sessions.items():
            turns = _parse_turns(turns_data if isinstance(turns_data, list) else [])
            sessions.append(LongMemEvalSession(session_id=str(sid), date="", turns=turns))

    elif isinstance(raw_sessions, list):
        for i, entry in enumerate(raw_sessions):
            has_conv_key = "conversation" in entry or "turns" in entry or "dialogue" in entry
            if isinstance(entry, dict) and has_conv_key:
                # Format: [{"session_id": ..., "conversation": [...]}]
                sid = str(entry.get("session_id", entry.get("id", f"ses_{i}")))
                date = str(entry.get("date", entry.get("timestamp", "")))
                raw_turns = entry.get("conversation", entry.get("turns", entry.get("dialogue", [])))
                turns = _parse_turns(raw_turns if isinstance(raw_turns, list) else [])
                sessions.append(LongMemEvalSession(session_id=sid, date=date, turns=turns))
            elif isinstance(entry, list):
                # Format: [[turns], [turns], ...] — indexed by position
                sid = session_ids[i] if session_ids and i < len(session_ids) else f"ses_{i}"
                date = session_dates[i] if session_dates and i < len(session_dates) else ""
                turns = _parse_turns(entry)
                sessions.append(
                    LongMemEvalSession(
                        session_id=str(sid),
                        date=str(date),
                        turns=turns,
                    )
                )

    # Backfill dates from session_dates if sessions have empty dates
    if session_dates and len(session_dates) == len(sessions):
        for session, date in zip(sessions, session_dates):
            if not session.date:
                session.date = str(date)

    return sessions


def load_dataset(
    path: str | Path,
    *,
    max_instances: int | None = None,
    variant: str = "auto",
) -> LongMemEvalDataset:
    """Load a LongMemEval dataset from a local JSON file.

    Args:
        path: Path to the JSON file.
        max_instances: Maximum number of instances to load.
        variant: Dataset variant name. Auto-detected from filename if "auto".
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Auto-detect variant from filename
    if variant == "auto":
        name = path.stem.lower()
        if "oracle" in name:
            variant = "oracle"
        elif "_m" in name or name.endswith("_m_cleaned"):
            variant = "m"
        else:
            variant = "s"

    with open(path) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("instances", list(raw.values())))
    if not isinstance(raw, list):
        raise ValueError(f"Expected list of instances, got {type(raw)}")

    instances: list[LongMemEvalInstance] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        if max_instances and len(instances) >= max_instances:
            break

        question_id = str(entry.get("question_id", entry.get("id", f"q_{len(instances)}")))
        question_type = str(entry.get("question_type", entry.get("type", "unknown")))
        question = str(entry.get("question", entry.get("query", "")))
        answer = str(entry.get("answer", entry.get("ground_truth", "")))
        question_date = str(entry.get("question_date", entry.get("date", "")))

        # Parse sessions
        raw_sessions = entry.get(
            "haystack_sessions",
            entry.get("sessions", entry.get("conversations", [])),
        )
        session_ids = entry.get("haystack_session_ids", entry.get("session_ids"))
        session_dates = entry.get("haystack_dates", entry.get("session_dates"))
        sessions = _parse_sessions(raw_sessions, session_ids, session_dates)

        answer_session_ids = [
            str(sid)
            for sid in entry.get(
                "answer_session_ids",
                entry.get("evidence_session_ids", []),
            )
        ]

        instances.append(
            LongMemEvalInstance(
                question_id=question_id,
                question_type=question_type,
                question=question,
                answer=answer,
                question_date=question_date,
                sessions=sessions,
                answer_session_ids=answer_session_ids,
            )
        )

    logger.info("Loaded %d instances from %s (variant=%s)", len(instances), path, variant)
    return LongMemEvalDataset(variant=variant, instances=instances)


async def download_dataset(
    variant: str = "s",
    output_dir: str | Path = "data/longmemeval",
) -> Path:
    """Download a LongMemEval dataset variant from HuggingFace.

    Returns the path to the downloaded file.
    """
    import httpx

    if variant not in _VARIANT_FILES:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(_VARIANT_FILES.keys())}")

    filename = _VARIANT_FILES[variant]
    url = f"{_HF_BASE}/{filename}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if output_path.exists():
        logger.info("Dataset already exists at %s", output_path)
        return output_path

    logger.info("Downloading %s from %s", filename, url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        output_path.write_bytes(response.content)

    logger.info("Downloaded %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path
