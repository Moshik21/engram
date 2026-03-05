"""Conversation context tracking for Wave 2 retrieval awareness."""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class SessionEntityEntry:
    """Tracks an entity mentioned during this session."""

    entity_id: str
    name: str
    entity_type: str
    mention_weight: float  # cumulative, incremented each mention
    first_seen: float
    last_seen: float


class ConversationContext:
    """Rolling conversation state: turns, fingerprint, and session entities."""

    def __init__(
        self,
        alpha: float = 0.85,
        max_turns: int = 20,
        topic_shift_threshold: float = 0.60,
    ) -> None:
        self._alpha = alpha
        self._max_turns = max_turns
        self._turns: deque[str] = deque(maxlen=max_turns)
        self._fingerprint: list[float] | None = None
        self._session_entities: dict[str, SessionEntityEntry] = {}
        self._turn_count: int = 0
        # Topic shift detection (Wave 3)
        self._prev_fingerprint: list[float] | None = None
        self._topic_shifted: bool = False
        self._topic_shift_threshold: float = topic_shift_threshold
        # State-dependent retrieval (Brain Architecture 1C)
        self._cognitive_state: object | None = None
        self._arousal_ema: float = 0.3
        self._session_start: float = time.time()
        self._recent_entity_types: list[str] = []
        self._max_entity_types: int = 100

    def add_turn(self, text: str, embedding: list[float] | None = None) -> None:
        """Append a turn and optionally update the fingerprint."""
        self._turns.append(text)
        self._turn_count += 1
        if embedding is not None:
            self.update_fingerprint(embedding)

    def update_fingerprint(self, embedding: list[float]) -> None:
        """EMA update: new = alpha * old + (1-alpha) * embedding, then L2-normalize."""
        if self._fingerprint is None:
            # First turn — set directly (then normalize)
            self._fingerprint = list(embedding)
        else:
            # Snapshot previous fingerprint before EMA update (Wave 3)
            self._prev_fingerprint = list(self._fingerprint)

            # Topic shift detection (Wave 3): compare raw incoming embedding
            # against previous fingerprint (before EMA blending dilutes the signal)
            sim = self._cosine_sim(self._prev_fingerprint, embedding)
            if sim < self._topic_shift_threshold:
                self._topic_shifted = True

            a = self._alpha
            self._fingerprint = [
                a * old + (1.0 - a) * new
                for old, new in zip(self._fingerprint, embedding)
            ]
        # L2 normalize
        norm = math.sqrt(sum(x * x for x in self._fingerprint))
        if norm > 0:
            self._fingerprint = [x / norm for x in self._fingerprint]

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def add_session_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        weight_increment: float = 1.0,
        now: float | None = None,
    ) -> None:
        """Track or increment a session entity."""
        ts = now if now is not None else time.time()
        if entity_id in self._session_entities:
            entry = self._session_entities[entity_id]
            entry.mention_weight += weight_increment
            entry.last_seen = ts
        else:
            self._session_entities[entity_id] = SessionEntityEntry(
                entity_id=entity_id,
                name=name,
                entity_type=entity_type,
                mention_weight=weight_increment,
                first_seen=ts,
                last_seen=ts,
            )

    def get_fingerprint(self) -> list[float] | None:
        """Return the current EMA fingerprint, or None if no embeddings yet."""
        return self._fingerprint

    def get_top_entities(self, n: int = 5) -> list[SessionEntityEntry]:
        """Return top-N session entities sorted by mention_weight desc."""
        entries = sorted(
            self._session_entities.values(),
            key=lambda e: e.mention_weight,
            reverse=True,
        )
        return entries[:n]

    def get_recent_turns(self, n: int = 3) -> list[str]:
        """Return the last N turns."""
        turns = list(self._turns)
        return turns[-n:] if len(turns) > n else turns

    def detect_topic_shift(self) -> bool:
        """Return whether a topic shift has been detected since last acknowledgement."""
        return self._topic_shifted

    def acknowledge_shift(self) -> None:
        """Clear the topic shift flag after it has been consumed."""
        self._topic_shifted = False

    def fingerprint_similarity(self, vec: list[float]) -> float:
        """Cosine similarity between fingerprint and vec. Returns 0.0 if no fingerprint."""
        fp = self._fingerprint
        if fp is None or not vec:
            return 0.0
        dot = sum(a * b for a, b in zip(fp, vec))
        norm_v = math.sqrt(sum(x * x for x in vec))
        # fingerprint is already L2-normalized
        if norm_v == 0:
            return 0.0
        return max(0.0, dot / norm_v)

    @property
    def arousal_level(self) -> float:
        """Current arousal EMA level."""
        return self._arousal_ema

    def update_arousal(
        self,
        new_composite: float,
        alpha: float = 0.3,
    ) -> None:
        """Update arousal EMA with a new composite value."""
        self._arousal_ema = alpha * new_composite + (1.0 - alpha) * self._arousal_ema

    def track_entity_type(self, entity_type: str) -> None:
        """Record an entity type encountered in this session."""
        self._recent_entity_types.append(entity_type)
        if len(self._recent_entity_types) > self._max_entity_types:
            self._recent_entity_types = self._recent_entity_types[-self._max_entity_types:]

    @property
    def recent_entity_types(self) -> list[str]:
        """Return recent entity types tracked this session."""
        return list(self._recent_entity_types)

    def update_cognitive_state(
        self,
        query: str,
        salience_composite: float = 0.0,
        alpha: float = 0.3,
    ) -> None:
        """Update cognitive state from latest query."""
        from engram.retrieval.state import (
            CognitiveState,
            get_time_bucket,
            infer_cognitive_mode,
        )

        mode = infer_cognitive_mode(query)
        # EMA for arousal
        self._arousal_ema = (
            alpha * salience_composite + (1 - alpha) * self._arousal_ema
        )
        self._cognitive_state = CognitiveState(
            arousal_level=self._arousal_ema,
            mode=mode,
            time_bucket=get_time_bucket(),
        )

    @property
    def cognitive_state(self) -> object | None:
        """Return current cognitive state or None."""
        return self._cognitive_state

    def clear(self) -> None:
        """Reset all state."""
        self._turns.clear()
        self._fingerprint = None
        self._prev_fingerprint = None
        self._topic_shifted = False
        self._session_entities.clear()
        self._turn_count = 0
        self._cognitive_state = None
        self._arousal_ema = 0.3
        self._session_start = time.time()
        self._recent_entity_types = []


class ConversationFingerprinter:
    """Static helper for ingesting turns with optional embedding."""

    @staticmethod
    async def ingest_turn(
        ctx: ConversationContext,
        text: str,
        embed_fn=None,
    ) -> None:
        """Add a turn to the context, embedding it if possible."""
        embedding = None
        if embed_fn and text:
            try:
                embedding = await embed_fn(text[:500])
            except Exception:
                pass
        ctx.add_turn(text, embedding)
