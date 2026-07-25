"""Recall telemetry, interaction, and usage-detection helpers."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from engram.config import ActivationConfig
from engram.events.bus import EventBus
from engram.extraction.policy import ProjectionPolicy
from engram.ingestion.projection_state import sync_projection_state
from engram.models.entity import Entity
from engram.models.episode import Episode, EpisodeProjectionState, EpisodeStatus
from engram.models.episode_cue import EpisodeCue
from engram.models.recall import MemoryInteractionEvent, MemoryNeed
from engram.retrieval.control import RecallNeedController
from engram.retrieval.usage_surface_store import (
    SurfacedUsageStore,
    resolve_usage_surface_path,
)
from engram.storage.protocols import ActivationStore, GraphStore
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)


class _LabileTracker(Protocol):
    def mark_labile(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        query: str,
    ) -> None: ...


def _coerce_int(value: object, default: int = 0) -> int:
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
            return default
    return default


def _projection_state_value(episode: Episode) -> str | None:
    state = episode.projection_state
    value = getattr(state, "value", None)
    if isinstance(value, str):
        return value
    return state if isinstance(state, str) else None


def publish_memory_need_analysis(
    event_bus: EventBus | None,
    group_id: str,
    need: MemoryNeed,
    *,
    source: str,
    mode: str,
    turn_text: str,
) -> None:
    """Publish a recall.need.analyzed event if an event bus is available."""
    if event_bus is None:
        return
    event_bus.publish(
        group_id,
        "recall.need.analyzed",
        need.to_payload(
            source=source,
            mode=mode,
            turn_preview=turn_text.strip()[:200],
        ),
    )


def publish_memory_interaction(
    event_bus: EventBus | None,
    interaction: MemoryInteractionEvent,
) -> None:
    """Publish a single recall.interaction event."""
    if event_bus is None:
        return
    event_bus.publish(
        interaction.group_id,
        "recall.interaction",
        interaction.to_payload(),
    )


async def publish_activation_access(
    *,
    event_bus: EventBus | None,
    activation_store: ActivationStore,
    cfg: ActivationConfig,
    entity_id: str,
    name: str,
    entity_type: str,
    group_id: str,
    accessed_via: str,
) -> None:
    """Publish an activation.access event with the entity's current activation."""
    if event_bus is None:
        return

    from engram.activation.engine import compute_activation

    now = time.time()
    state = await activation_store.get_activation(entity_id)
    activation = 0.0
    if state:
        activation = compute_activation(state.access_history, now, cfg)
    event_bus.publish(
        group_id,
        "activation.access",
        {
            "entityId": entity_id,
            "name": name,
            "entityType": entity_type,
            "activation": round(activation, 4),
            "accessedVia": accessed_via,
        },
    )


# --- M1.4: echo-guarded citation scan (surfaced -> used tier) -----------------
#
# Recall surfacing populates a bounded per-group ring buffer of surfaced
# payloads. The Capture fast path (store_episode via capture_surface) scans the
# next observed turn against it: a surfaced entity mentioned in NOVEL tokens
# (outside any echoed shingle span of the surfaced payload) records a
# `used`-tier access event. Everything is gated on
# `recall_usage_feedback_enabled` (default False) and `w_used` stays 0 in
# ranking until G7 passes — usage_events accumulate inertly in M1.

_USAGE_RING_CAP = 32
_USAGE_DEDUP_WINDOW_SECONDS = 30 * 60.0
# Ticket #7: how long a surfaced payload stays ELIGIBLE to be counted as used.
# Deliberately the same number as the dedup window, and it must not be larger:
# with the two equal, a cue surfaced at t0 ages out at t0+TTL while its dedup
# mark blocks it until t_fire+TTL >= t0+TTL, so a fired cue can never fire a
# second time without being re-surfaced. Before durability this bound was
# supplied for free by process death; the durable registry has to state it.
# Erring short is the conservative direction — it misses late uses rather than
# manufacturing phantom ones.
_USAGE_SURFACE_TTL_SECONDS = _USAGE_DEDUP_WINDOW_SECONDS
_USAGE_SNIPPET_MAX_CHARS = 400
_USAGE_SHINGLE_TOKENS = 5
_USAGE_MIN_SHINGLE_TOKENS = 3
_USAGE_CONTEXT_WINDOW_TOKENS = 12
_USAGE_CONTEXT_MIN_TOKEN_LEN = 4
_USAGE_PROMOTION_ENTITY_CAP = 32


_CUE_USAGE_MIN_PHRASE_TOKENS = 2
_CUE_USAGE_MAX_PHRASE_TOKENS = 5
_CUE_USAGE_MIN_PHRASE_CHARS = 10


@dataclass(frozen=True)
class SurfacedEntry:
    """One surfaced recall payload remembered for the citation scan."""

    entity_id: str
    name: str
    ts: float
    snippet_tokens: tuple[str, ...]


@dataclass(frozen=True)
class SurfacedCueEntry:
    """One surfaced cue-backed episode remembered for the citation scan (M5.1)."""

    episode_id: str
    ts: float
    phrase_token_lists: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SurfacedTextEntry:
    """One mask-only surfaced text payload (episode content, packet summary)."""

    digest: str
    ts: float
    tokens: tuple[str, ...]


def _text_digest(tokens: tuple[str, ...]) -> str:
    return hashlib.sha1(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


class SurfacedUsageBuffer:
    """Bounded per-group ring buffer of surfaced payloads + used-event dedup.

    Ticket #7: the ring is optionally backed by a durable sidecar
    (``SurfacedUsageStore``). Without it the surfaced half of the surfaced ->
    used loop dies with the process, so an agent whose session outlives a shell
    restart can never record a use. With it, ``hydrate`` restores the ring — and
    critically the ECHO MASK and the dedup marks with it — in the next process.
    """

    def __init__(
        self,
        *,
        cap: int = _USAGE_RING_CAP,
        dedup_window_seconds: float = _USAGE_DEDUP_WINDOW_SECONDS,
        surface_ttl_seconds: float = _USAGE_SURFACE_TTL_SECONDS,
        store: SurfacedUsageStore | None = None,
    ) -> None:
        self._cap = cap
        self._dedup_window = dedup_window_seconds
        self._surface_ttl = surface_ttl_seconds
        self._entries: dict[str, deque[SurfacedEntry]] = {}
        # M5.1: surfaced cue-backed episodes, scanned in the SAME pass with the
        # same echo mask; dedup shares _last_used_event under a "cue::" key.
        self._cue_entries: dict[str, deque[SurfacedCueEntry]] = {}
        self._last_used_event: dict[tuple[str, str], float] = {}
        # Mask-only ring: shingle sources for ALL surfaced text (episode
        # content, packet summaries) — not entity-bound. Without this, an
        # agent echoing a surfaced EPISODE verbatim reads as "novel" tokens
        # relative to the entity snippets and fires false used events.
        self._text_masks: dict[str, deque[SurfacedTextEntry]] = {}
        self._store = store
        self._hydrated: set[str] = set()
        # Rows written since the last persist, so a persist per surfaced payload
        # costs one row rather than a full rewrite of the ring.
        self._pending: dict[str, dict[str, set[str]]] = {}

    # --- durability ------------------------------------------------------

    def attach_store(self, store: SurfacedUsageStore | None) -> None:
        """Bind (or unbind) the durable sidecar; drops hydration state."""
        self._store = store
        self._hydrated.clear()

    @property
    def store(self) -> SurfacedUsageStore | None:
        return self._store

    def _mark_pending(self, group_id: str, kind: str, key: str) -> None:
        if self._store is None:
            return
        self._pending.setdefault(group_id, {}).setdefault(kind, set()).add(key)

    def hydrate(self, group_id: str, now: float | None = None) -> None:
        """Load this group's durable surfaced state once per process.

        Entries land to the LEFT of anything this process already noted, so the
        ring stays oldest-first, and never evict in-process entries.
        """
        store = self._store
        if store is None or group_id in self._hydrated:
            return
        self._hydrated.add(group_id)
        ts = now if now is not None else time.time()
        state = store.load(group_id, min_ts=ts - self._surface_ttl, cap=self._cap)
        if state.is_empty():
            return

        ring = self._entries.setdefault(group_id, deque(maxlen=self._cap))
        known = {entry.entity_id for entry in ring}
        for entity_id, name, ets, tokens in reversed(state.entities):
            if entity_id in known or len(ring) >= self._cap:
                continue
            ring.appendleft(
                SurfacedEntry(entity_id=entity_id, name=name, ts=ets, snippet_tokens=tuple(tokens))
            )

        cue_ring = self._cue_entries.setdefault(group_id, deque(maxlen=self._cap))
        known_cues = {entry.episode_id for entry in cue_ring}
        for episode_id, cts, phrases in reversed(state.cues):
            if episode_id in known_cues or len(cue_ring) >= self._cap:
                continue
            cue_ring.appendleft(
                SurfacedCueEntry(
                    episode_id=episode_id,
                    ts=cts,
                    phrase_token_lists=tuple(tuple(phrase) for phrase in phrases),
                )
            )

        mask_ring = self._text_masks.setdefault(group_id, deque(maxlen=self._cap))
        known_texts = {entry.digest for entry in mask_ring}
        for digest, tts, tokens in reversed(state.texts):
            if digest in known_texts or len(mask_ring) >= self._cap:
                continue
            mask_ring.appendleft(SurfacedTextEntry(digest=digest, ts=tts, tokens=tuple(tokens)))

        for dedup_key, dts in state.dedup:
            key = (group_id, dedup_key)
            if dts > self._last_used_event.get(key, 0.0):
                self._last_used_event[key] = dts

    def persist(self, group_id: str, now: float | None = None) -> bool:
        """Flush rows noted since the last persist. Best-effort, never raises."""
        store = self._store
        pending = self._pending.get(group_id)
        if store is None or not pending:
            return False
        ts = now if now is not None else time.time()
        entity_keys = pending.get("entities") or set()
        cue_keys = pending.get("cues") or set()
        text_keys = pending.get("texts") or set()
        dedup_keys = pending.get("dedup") or set()
        entities = [
            (entry.entity_id, entry.name, entry.ts, list(entry.snippet_tokens))
            for entry in self._entries.get(group_id) or ()
            if entry.entity_id in entity_keys
        ]
        cues = [
            (entry.episode_id, entry.ts, [list(phrase) for phrase in entry.phrase_token_lists])
            for entry in self._cue_entries.get(group_id) or ()
            if entry.episode_id in cue_keys
        ]
        texts = [
            (entry.digest, entry.ts, list(entry.tokens))
            for entry in self._text_masks.get(group_id) or ()
            if entry.digest in text_keys
        ]
        dedup = [
            (key, value)
            for (gid, key), value in self._last_used_event.items()
            if gid == group_id and key in dedup_keys
        ]
        ok = store.save(
            group_id,
            entities=entities,
            cues=cues,
            texts=texts,
            dedup=dedup,
            min_ts=ts - self._surface_ttl,
        )
        if ok:
            self._pending.pop(group_id, None)
        return ok

    # --- registration ----------------------------------------------------

    def note_surfaced(
        self,
        group_id: str,
        *,
        entity_id: str,
        name: str,
        snippet: str,
        ts: float,
    ) -> None:
        if not entity_id or not name:
            return
        self.hydrate(group_id, ts)
        tokens = tuple(_normalize_text((snippet or "")[:_USAGE_SNIPPET_MAX_CHARS]).split())
        ring = self._entries.setdefault(group_id, deque(maxlen=self._cap))
        ring.append(SurfacedEntry(entity_id=entity_id, name=name, ts=ts, snippet_tokens=tokens))
        self._mark_pending(group_id, "entities", entity_id)

    def note_surfaced_text(self, group_id: str, text: str, ts: float) -> None:
        """Register surfaced payload text as an echo-mask source (mask-only).

        No entity binding and no used-event eligibility — these tokens only
        widen the echoed-span mask so parroted result text never counts as
        reliance. ``ts`` is the durable age-out key: a mask row that outlived
        its payload would suppress real uses, and a missing one would fabricate
        them, so the mask ages on exactly the same clock as what it masks.
        """
        tokens = tuple(_normalize_text((text or "")[: _USAGE_SNIPPET_MAX_CHARS * 4]).split())
        if not tokens:
            return
        self.hydrate(group_id, ts)
        digest = _text_digest(tokens)
        ring = self._text_masks.setdefault(group_id, deque(maxlen=self._cap))
        ring.append(SurfacedTextEntry(digest=digest, ts=ts, tokens=tokens))
        self._mark_pending(group_id, "texts", digest)

    def note_surfaced_cue(
        self,
        group_id: str,
        *,
        episode_id: str,
        cue_text: str,
        supporting_spans: list[str] | None = None,
        ts: float,
    ) -> None:
        """Register a surfaced cue-backed episode for the citation scan (M5.1).

        The cue's text and spans become both match candidates (novel reuse of a
        cue phrase = reliance) and echo-mask sources (verbatim parroting never
        fires).
        """
        if not episode_id:
            return
        phrase_token_lists: list[tuple[str, ...]] = []
        for candidate in [cue_text, *(supporting_spans or [])]:
            tokens = tuple(_normalize_text((candidate or "")[:_USAGE_SNIPPET_MAX_CHARS]).split())
            if tokens:
                phrase_token_lists.append(tokens)
        if not phrase_token_lists:
            return
        self.hydrate(group_id, ts)
        ring = self._cue_entries.setdefault(group_id, deque(maxlen=self._cap))
        # Re-surfacing the same episode refreshes its entry instead of adding a
        # second one: the pipeline recorder and the explicit-recall response
        # feed both register, and duplicates would evict distinct cues from the
        # bounded ring.
        for existing in list(ring):
            if existing.episode_id == episode_id:
                ring.remove(existing)
        ring.append(
            SurfacedCueEntry(
                episode_id=episode_id,
                ts=ts,
                phrase_token_lists=tuple(phrase_token_lists),
            )
        )
        self._mark_pending(group_id, "cues", episode_id)

    def is_empty(self, group_id: str) -> bool:
        # The capture fast path short-circuits on this, so it is also the hook
        # that pulls the previous process's surfaced payloads back in.
        self.hydrate(group_id)
        return not self._entries.get(group_id) and not self._cue_entries.get(group_id)

    def scan_novel_mentions(
        self,
        group_id: str,
        content: str,
        now: float,
    ) -> list[SurfacedEntry]:
        """Return surfaced entries genuinely relied on by ``content``.

        Echo guard: token positions covered by shingles of the buffered
        surfaced payloads count as echoed; a mention wholly inside an echoed
        span never fires. Dedup: at most one used event per (entity, group)
        per rolling window.
        """
        self.hydrate(group_id, now)
        ring = self._entries.get(group_id)
        if not ring:
            return []
        content_tokens = _normalize_text(content).split()
        if not content_tokens:
            return []

        echoed = _echoed_token_mask(content_tokens, self._mask_sources(group_id))
        fired: list[SurfacedEntry] = []
        seen: set[str] = set()
        for entry in reversed(ring):
            if entry.entity_id in seen:
                continue
            seen.add(entry.entity_id)
            if not self._is_eligible(entry.ts, now):
                continue
            last = self._last_used_event.get((group_id, entry.entity_id))
            if last is not None and (now - last) < self._dedup_window:
                continue
            if _has_novel_mention(content_tokens, echoed, entry):
                self._last_used_event[(group_id, entry.entity_id)] = now
                self._mark_pending(group_id, "dedup", entry.entity_id)
                fired.append(entry)
        self._prune_dedup(now)
        return fired

    def scan_novel_cue_matches(
        self,
        group_id: str,
        content: str,
        now: float,
    ) -> list[SurfacedCueEntry]:
        """Return surfaced cues genuinely relied on by ``content`` (M5.1).

        Same pass semantics as scan_novel_mentions: the SAME echo mask (a
        verbatim parrot of any surfaced payload never fires) and the SAME
        dedup class/window, keyed ("cue::" + episode_id) in _last_used_event.
        A cue fires when a contiguous >=2-token, >=10-char phrase shared with
        its cue text/spans appears at a not-wholly-echoed content position.
        """
        self.hydrate(group_id, now)
        ring = self._cue_entries.get(group_id)
        if not ring:
            return []
        content_tokens = _normalize_text(content).split()
        if not content_tokens:
            return []

        echoed = _echoed_token_mask(content_tokens, self._mask_sources(group_id))
        fired: list[SurfacedCueEntry] = []
        seen: set[str] = set()
        for entry in reversed(ring):
            if entry.episode_id in seen:
                continue
            seen.add(entry.episode_id)
            if not self._is_eligible(entry.ts, now):
                continue
            marker = f"cue::{entry.episode_id}"
            dedup_key = (group_id, marker)
            last = self._last_used_event.get(dedup_key)
            if last is not None and (now - last) < self._dedup_window:
                continue
            if _has_novel_cue_match(content_tokens, echoed, entry):
                self._last_used_event[dedup_key] = now
                self._mark_pending(group_id, "dedup", marker)
                fired.append(entry)
        self._prune_dedup(now)
        return fired

    def _is_eligible(self, surfaced_ts: float, now: float) -> bool:
        """Whether a payload surfaced at ``surfaced_ts`` may still count as used.

        Process death used to supply this bound implicitly. A durable registry
        has to state it, and it must be a bound rather than "forever": a phrase
        collision hours after the surfacing is not reliance.
        """
        return (now - surfaced_ts) < self._surface_ttl

    def _mask_sources(self, group_id: str) -> list[tuple[str, ...]]:
        """Echo-mask shingle sources shared by the entity and cue scans."""
        sources = [entry.snippet_tokens for entry in self._entries.get(group_id) or ()]
        for cue_entry in self._cue_entries.get(group_id) or ():
            sources.extend(cue_entry.phrase_token_lists)
        sources.extend(entry.tokens for entry in self._text_masks.get(group_id) or ())
        return sources

    def _prune_dedup(self, now: float) -> None:
        stale = [
            key for key, ts in self._last_used_event.items() if (now - ts) >= self._dedup_window
        ]
        for key in stale:
            del self._last_used_event[key]

    def reset(self) -> None:
        self._entries.clear()
        self._cue_entries.clear()
        self._last_used_event.clear()
        self._text_masks.clear()
        self._hydrated.clear()
        self._pending.clear()
        self._store = None


_USAGE_BUFFER = SurfacedUsageBuffer()
_USAGE_SURFACE_PATH: Path | None = None


def bind_usage_surface_store(path: str | Path | None) -> SurfacedUsageStore | None:
    """Attach the durable surfaced registry to the process-wide buffer.

    Idempotent: rebinding the same path keeps the existing store. ``None``
    unbinds and restores the pre-ticket-#7 process-local behaviour, which is
    what unit tests and bare ``ActivationConfig`` callers get.
    """
    global _USAGE_SURFACE_PATH

    if path is None:
        if _USAGE_BUFFER.store is not None:
            _USAGE_BUFFER.attach_store(None)
        _USAGE_SURFACE_PATH = None
        return None
    resolved = Path(path).expanduser()
    if _USAGE_BUFFER.store is not None and _USAGE_SURFACE_PATH == resolved:
        return _USAGE_BUFFER.store
    store = SurfacedUsageStore(resolved)
    _USAGE_BUFFER.attach_store(store)
    _USAGE_SURFACE_PATH = resolved
    logger.info("Surfaced-usage registry bound to %s", resolved)
    return store


def bind_usage_surface_store_from_config(cfg) -> SurfacedUsageStore | None:
    """Bind the durable registry for a runtime config, if one is resolvable.

    Deliberately self-arming from whatever config reaches the hot path rather
    than from an entrypoint: a "remember to call bind() at startup" contract is
    exactly how a mechanism goes silently inert (§2.2), and this one has been
    inert since it was written. Returns None — leaving today's process-local
    behaviour — when the config names no sidecar directory, which is the case
    for a bare ``ActivationConfig`` in unit tests.
    """
    if not getattr(cfg, "recall_usage_feedback_enabled", False):
        return None
    path = resolve_usage_surface_path(cfg)
    if path is None:
        return None
    if _USAGE_BUFFER.store is not None and _USAGE_SURFACE_PATH == path:
        return _USAGE_BUFFER.store
    return bind_usage_surface_store(path)


def _ensure_usage_surface_store(cfg, buffer: SurfacedUsageBuffer) -> None:
    """Arm the process-wide durable registry; no-op for injected buffers."""
    if buffer is not _USAGE_BUFFER:
        return
    try:
        bind_usage_surface_store_from_config(cfg)
    except Exception:
        # silent-ok: durability is protective telemetry. A sidecar that cannot
        # be opened degrades to the process-local ring, never to a failed recall.
        logger.debug("Surfaced-usage registry bind failed", exc_info=True)


def usage_surface_store_path() -> Path | None:
    """Path of the bound durable registry, or None when process-local only."""
    return _USAGE_SURFACE_PATH if _USAGE_BUFFER.store is not None else None


def note_surfaced_texts_from_response(
    group_id: str,
    response: dict,
    cfg,
    *,
    now: float | None = None,
) -> None:
    """Feed every surfaced result/packet text into the echo mask (mask-only).

    Called by the recall surface at response time so the citation scan can
    never mistake a parroted result for reliance. Flag-gated and best-effort:
    masking must never fail a recall.
    """
    if not getattr(cfg, "recall_usage_feedback_enabled", False):
        return
    try:
        ts = now if now is not None else time.time()
        buffer = get_usage_buffer()
        _ensure_usage_surface_store(cfg, buffer)
        for result in response.get("results") or []:
            text = result.get("text") or result.get("content") or ""
            if text:
                buffer.note_surfaced_text(group_id, str(text), ts)
            # M5.1: surfaced cue-backed episodes join the citation scan. Their
            # cue text/spans double as echo-mask sources inside the cue entry.
            if result.get("result_type") == "cue_episode":
                episode_id = result.get("episode_id") or ""
                cue_text = result.get("cue_text") or ""
                spans = [str(span) for span in result.get("supporting_spans") or []]
                if episode_id and (cue_text or spans):
                    buffer.note_surfaced_cue(
                        group_id,
                        episode_id=str(episode_id),
                        cue_text=str(cue_text),
                        supporting_spans=spans,
                        ts=ts,
                    )
        for packet in response.get("packets") or []:
            text = packet.get("summary") or packet.get("text") or ""
            title = packet.get("title") or ""
            if text or title:
                buffer.note_surfaced_text(group_id, f"{title} {text}".strip(), ts)
        # Ticket #7: the surfaced half has to outlive this process, or the
        # agent's next turn lands in a shell that never saw the surfacing.
        buffer.persist(group_id, ts)
    except Exception:
        # silent-ok: echo masking is protective telemetry; a mask miss only
        # risks an extra used event, never a failed recall.
        logger.debug("Surfaced-text mask feed failed", exc_info=True)


def get_usage_buffer() -> SurfacedUsageBuffer:
    """Process-wide surfaced-usage ring buffer shared by recall + capture."""
    return _USAGE_BUFFER


def _echoed_token_mask(
    content_tokens: list[str],
    snippet_token_lists: list[tuple[str, ...]],
) -> list[bool]:
    """Mark content token positions covered by surfaced-payload shingles."""
    shingles_by_len: dict[int, set[tuple[str, ...]]] = {}
    for tokens in snippet_token_lists:
        if len(tokens) >= _USAGE_SHINGLE_TOKENS:
            bucket = shingles_by_len.setdefault(_USAGE_SHINGLE_TOKENS, set())
            for i in range(len(tokens) - _USAGE_SHINGLE_TOKENS + 1):
                bucket.add(tokens[i : i + _USAGE_SHINGLE_TOKENS])
        elif len(tokens) >= _USAGE_MIN_SHINGLE_TOKENS:
            shingles_by_len.setdefault(len(tokens), set()).add(tokens)

    mask = [False] * len(content_tokens)
    for size, shingles in shingles_by_len.items():
        for i in range(len(content_tokens) - size + 1):
            if tuple(content_tokens[i : i + size]) in shingles:
                for j in range(i, i + size):
                    mask[j] = True
    return mask


def _has_novel_mention(
    content_tokens: list[str],
    echoed: list[bool],
    entry: SurfacedEntry,
) -> bool:
    """True when the entity is mentioned in novel (non-echoed) tokens.

    Single-token names additionally require the novel context around the
    mention to share vocabulary with the surfaced payload — an incidental
    common-word mention with no topical tie to what was surfaced does not
    count as reliance.
    """
    name_tokens = tuple(_normalize_text(entry.name).split())
    if not name_tokens:
        return False
    if len(name_tokens) == 1 and len(name_tokens[0]) < 3:
        return False

    n = len(name_tokens)
    snippet_context = {
        token
        for token in entry.snippet_tokens
        if len(token) >= _USAGE_CONTEXT_MIN_TOKEN_LEN and token not in name_tokens
    }
    for i in range(len(content_tokens) - n + 1):
        if tuple(content_tokens[i : i + n]) != name_tokens:
            continue
        span = range(i, i + n)
        if all(echoed[j] for j in span):
            continue
        if n == 1:
            lo = max(0, i - _USAGE_CONTEXT_WINDOW_TOKENS)
            hi = min(len(content_tokens), i + 1 + _USAGE_CONTEXT_WINDOW_TOKENS)
            context = {content_tokens[j] for j in range(lo, hi) if j != i and not echoed[j]}
            if not (context & snippet_context):
                continue
        return True
    return False


def _has_novel_cue_match(
    content_tokens: list[str],
    echoed: list[bool],
    entry: SurfacedCueEntry,
) -> bool:
    """True when a shared cue phrase appears in novel (non-echoed) tokens.

    Positional mirror of _matches_cue_content: a contiguous n-gram (2..5
    tokens, >=10 chars joined) present in both the content and one of the
    cue's text/span token lists counts as reliance, unless every token of the
    matched span sits inside an echoed shingle (verbatim parroting).
    """
    candidate_ngrams: set[tuple[str, ...]] = set()
    for tokens in entry.phrase_token_lists:
        max_n = min(_CUE_USAGE_MAX_PHRASE_TOKENS, len(tokens))
        for size in range(_CUE_USAGE_MIN_PHRASE_TOKENS, max_n + 1):
            for i in range(len(tokens) - size + 1):
                gram = tokens[i : i + size]
                if len(" ".join(gram)) >= _CUE_USAGE_MIN_PHRASE_CHARS:
                    candidate_ngrams.add(gram)
    if not candidate_ngrams:
        return False

    for size in range(_CUE_USAGE_MIN_PHRASE_TOKENS, _CUE_USAGE_MAX_PHRASE_TOKENS + 1):
        for i in range(len(content_tokens) - size + 1):
            gram = tuple(content_tokens[i : i + size])
            if gram in candidate_ngrams and not all(echoed[j] for j in range(i, i + size)):
                return True
    return False


async def record_observed_usage_events(
    *,
    activation_store: ActivationStore,
    cfg: ActivationConfig,
    group_id: str,
    content: str,
    now: float | None = None,
    usage_buffer: SurfacedUsageBuffer | None = None,
    graph_store=None,
) -> list[str]:
    """Record used-tier usage for surfaced entities AND cues relied on in content.

    Requires ``recall_usage_feedback_enabled=True``; short-circuits (one dict
    lookup) when no recall has surfaced anything for the group. Entities append
    a used-tier access event; cue-backed episodes (M5.1, same pass, same echo
    mask + dedup class) bump the cue record's tier-weighted usage_used_count /
    usage_last_used_at when ``graph_store`` is provided. Returns the fired
    entity ids plus ``"cue::<episode_id>"`` markers for fired cues.
    """
    if not cfg.recall_usage_feedback_enabled:
        return []
    buffer = usage_buffer if usage_buffer is not None else _USAGE_BUFFER
    _ensure_usage_surface_store(cfg, buffer)
    if buffer.is_empty(group_id):
        return []
    ts = now if now is not None else time.time()
    fired = buffer.scan_novel_mentions(group_id, content, ts)
    # Dedup marks must outlive this process too, or a restart re-arms a cue
    # that already fired and the gate double-counts one reliance event.
    buffer.persist(group_id, ts)
    for entry in fired:
        await activation_store.record_access(
            entry.entity_id,
            ts,
            group_id=group_id,
            tier="used",
        )
    fired_ids = [entry.entity_id for entry in fired]
    if graph_store is None:
        return fired_ids

    fired_cues = buffer.scan_novel_cue_matches(group_id, content, ts)
    buffer.persist(group_id, ts)
    if not fired_cues:
        return fired_ids
    used_weight = float(cfg.usage_tier_weights.get("used", 0.0))
    for cue_entry in fired_cues:
        cue = await graph_store.get_episode_cue(cue_entry.episode_id, group_id)
        if cue is None:
            continue
        await graph_store.update_episode_cue(
            cue_entry.episode_id,
            {
                "usage_used_count": (cue.usage_used_count or 0.0) + used_weight,
                "usage_last_used_at": datetime.fromtimestamp(ts, tz=timezone.utc),
            },
            group_id=group_id,
        )
        fired_ids.append(f"cue::{cue_entry.episode_id}")
    if fired_ids:
        # Invalidate the episode-u skip cache: this group now has cue usage,
        # so the ranking-side tiebreaker must resume its cue reads.
        from engram.retrieval.pipeline import note_group_cue_usage_written

        note_group_cue_usage_written(group_id)
    return fired_ids


class RecallEntityAccessRecorder:
    """Record true Recall-stage entity access and reconsolidation side effects."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        activation_store: ActivationStore,
        event_bus: EventBus | None,
        labile_tracker: _LabileTracker | None,
        usage_buffer: SurfacedUsageBuffer | None = None,
    ) -> None:
        self._cfg = cfg
        self._activation = activation_store
        self._event_bus = event_bus
        self._labile_tracker = labile_tracker
        self._usage_buffer = usage_buffer if usage_buffer is not None else _USAGE_BUFFER

    async def publish_access_event(
        self,
        *,
        entity_id: str,
        name: str,
        entity_type: str,
        group_id: str,
        accessed_via: str,
    ) -> None:
        await publish_activation_access(
            event_bus=self._event_bus,
            activation_store=self._activation,
            cfg=self._cfg,
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            group_id=group_id,
            accessed_via=accessed_via,
        )

    async def record_entity_access(
        self,
        entity: Entity,
        *,
        group_id: str,
        query: str,
        source: str,
        timestamp: float | None = None,
        tier: str = "surfaced",
    ) -> None:
        now = timestamp if timestamp is not None else time.time()
        await self._activation.record_access(entity.id, now, group_id=group_id, tier=tier)
        if tier == "surfaced" and self._cfg.recall_usage_feedback_enabled:
            _ensure_usage_surface_store(self._cfg, self._usage_buffer)
            self._usage_buffer.note_surfaced(
                group_id,
                entity_id=entity.id,
                name=entity.name,
                snippet=f"{entity.name} {entity.summary or ''}",
                ts=now,
            )
            # Auto-recall / get_context never reach the response-time mask hook,
            # so the durable flush has to happen where the payload is noted.
            self._usage_buffer.persist(group_id, now)
        await self.publish_access_event(
            entity_id=entity.id,
            name=entity.name,
            entity_type=entity.entity_type,
            group_id=group_id,
            accessed_via=source,
        )

        if self._labile_tracker is not None:
            self._labile_tracker.mark_labile(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                query,
            )


class RecallCueFeedbackRecorder:
    """Record Recall-stage cue feedback and schedule hot cues for projection."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        graph_store: GraphStore,
        projection_policy: ProjectionPolicy,
        recall_need_controller: RecallNeedController,
        event_bus: EventBus | None,
        activation_store: ActivationStore | None = None,
        usage_buffer: SurfacedUsageBuffer | None = None,
    ) -> None:
        self._cfg = cfg
        self._graph = graph_store
        self._projection_policy = projection_policy
        self._recall_need_controller = recall_need_controller
        self._event_bus = event_bus
        self._activation = activation_store
        self._usage_buffer = usage_buffer if usage_buffer is not None else _USAGE_BUFFER

    def _register_surfaced_cue(self, episode: Episode, cue: EpisodeCue) -> None:
        """Register a delivered cue payload for the citation scan.

        The entity half of the buffer is filled by the shared materializer
        (RecallEntityAccessRecorder.record_entity_access), so entities are
        registered on every recall surface. The cue half used to be filled only
        by the two explicit-recall response builders, so cues surfaced through
        auto-recall / get_context were counted as surfaced but never became
        eligible for a used event. Registering here — the one place every
        surfaced cue passes through — restores the symmetry.
        """
        ts = time.time()
        _ensure_usage_surface_store(self._cfg, self._usage_buffer)
        self._usage_buffer.note_surfaced_cue(
            episode.group_id,
            episode_id=episode.id,
            cue_text=cue.cue_text or "",
            supporting_spans=[str(span) for span in cue.first_spans or []],
            ts=ts,
        )
        # The episode content travels with a cue result on some surfaces; mask
        # it so parroting the delivered payload never reads as reliance.
        if episode.content:
            self._usage_buffer.note_surfaced_text(episode.group_id, episode.content, ts)
        # Ticket #7: flush cue + mask together. Persisting the cue without its
        # mask would make a post-restart verbatim parrot read as novel reuse.
        self._usage_buffer.persist(episode.group_id, ts)

    async def record_cue_feedback(
        self,
        episode: Episode,
        score: float,
        query: str,
        *,
        interaction_type: str | None = None,
        near_miss: bool = False,
        count_hit: bool = True,
    ) -> None:
        cue = await self._graph.get_episode_cue(episode.id, episode.group_id)
        if cue is None:
            return

        if not near_miss and self._cfg.recall_usage_feedback_enabled:
            self._register_surfaced_cue(episode, cue)

        now_dt = utc_now()
        feedback_type = "near_miss" if near_miss else (interaction_type or "surfaced")
        feedback = self._projection_policy.apply_feedback(
            cue,
            interaction_type=feedback_type,
            score=score,
            count_hit=count_hit,
        )
        cue_updates: dict[str, object] = dict(feedback.updates)
        cue_updates["last_feedback_at"] = now_dt
        if not near_miss and "hit_count" in cue_updates:
            cue_updates["last_hit_at"] = now_dt

        current_projection_state = _projection_state_value(episode)
        event_payload = {
            "episodeId": episode.id,
            "projectionState": current_projection_state,
            "interactionType": feedback_type,
            "score": round(score, 4),
            "query": query[:200],
        }
        if "hit_count" in cue_updates:
            event_payload["hitCount"] = cue_updates["hit_count"]
        if "policy_score" in cue_updates:
            event_payload["policyScore"] = cue_updates["policy_score"]
        self._publish(
            episode.group_id,
            "cue.hit" if not near_miss else "cue.near_miss",
            event_payload,
        )

        if not near_miss and interaction_type in {"surfaced", "selected"}:
            self._recall_need_controller.record_interaction(
                episode.group_id,
                interaction_type,
                result_type="cue_episode",
                memory_id=f"cue:{episode.id}",
            )

        promotable_states = {
            EpisodeProjectionState.CUED.value,
            EpisodeProjectionState.CUE_ONLY.value,
            EpisodeProjectionState.QUEUED.value,
            EpisodeProjectionState.FAILED.value,
        }
        hit_count = _coerce_int(
            cue_updates.get("hit_count", cue.hit_count or 0),
            cue.hit_count or 0,
        )
        should_promote = (
            hit_count >= self._cfg.cue_recall_hit_threshold or feedback.should_promote
        ) and current_projection_state in promotable_states

        if should_promote:
            promotion_reason = (
                "cue_recall_hits"
                if hit_count >= self._cfg.cue_recall_hit_threshold
                else (feedback.promotion_reason or "cue_policy")
            )
            await sync_projection_state(
                self._graph,
                episode.id,
                group_id=episode.group_id,
                state=EpisodeProjectionState.SCHEDULED,
                reason=promotion_reason,
                episode_updates={"status": EpisodeStatus.QUEUED.value, "error": None},
                cue_reason=promotion_reason,
                cue_updates=cue_updates,
                cue_layer_enabled=True,
                log_prefix="Recall cue feedback",
            )
            self._publish(
                episode.group_id,
                "cue.promoted",
                {
                    "episodeId": episode.id,
                    "hitCount": hit_count,
                    "reason": promotion_reason,
                    "score": round(score, 4),
                    "policyScore": cue_updates.get("policy_score", cue.policy_score),
                },
            )
            self._publish(
                episode.group_id,
                "episode.projection_scheduled",
                {
                    "episodeId": episode.id,
                    "reason": promotion_reason,
                    "hitCount": hit_count,
                },
            )
            await self._record_promotion_usage(episode)
            return

        if self._cfg.cue_policy_learning_enabled and "policy_score" in cue_updates:
            self._publish(
                episode.group_id,
                "cue.policy_updated",
                {
                    "episodeId": episode.id,
                    "interactionType": feedback_type,
                    "policyScore": cue_updates["policy_score"],
                    "projectionState": current_projection_state,
                },
            )

        await self._graph.update_episode_cue(episode.id, cue_updates, group_id=episode.group_id)

    async def _record_promotion_usage(self, episode: Episode) -> None:
        """M1.2: cue-hit promotion is a confirmed-tier usage signal.

        Records a confirmed-tier access event for the entities actually linked
        to the promoted episode at promotion time (bounded). Gated on
        `recall_usage_feedback_enabled` so default-config recall output stays
        byte-identical (M1 inertness invariant).
        """
        if self._activation is None or not self._cfg.recall_usage_feedback_enabled:
            return
        try:
            entity_ids = await self._graph.get_episode_entities(
                episode.id,
                group_id=episode.group_id,
            )
        except Exception:
            # silent-ok: usage tagging must never fail or delay cue promotion.
            logger.debug("Cue promotion usage lookup failed", exc_info=True)
            return
        now = time.time()
        for entity_id in list(entity_ids or [])[:_USAGE_PROMOTION_ENTITY_CAP]:
            if not entity_id:
                continue
            try:
                await self._activation.record_access(
                    entity_id,
                    now,
                    group_id=episode.group_id,
                    tier="confirmed",
                )
            except Exception:
                # silent-ok: usage tagging must never fail or delay cue promotion.
                logger.debug("Cue promotion usage record failed", exc_info=True)

    def _publish(self, group_id: str, event_type: str, payload: dict | None = None) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(group_id, event_type, payload)


class RecallInteractionRecorder:
    """Publish Recall-stage interaction telemetry and update recall-need learning."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        event_bus: EventBus | None,
        recall_need_controller: RecallNeedController,
    ) -> None:
        self._cfg = cfg
        self._event_bus = event_bus
        self._recall_need_controller = recall_need_controller

    def record_entity_interaction(
        self,
        *,
        group_id: str,
        entity: Entity,
        interaction_type: str | None,
        source: str,
        query: str,
        score: float,
        recorded_access: bool,
    ) -> None:
        self.record_memory_interaction(
            group_id=group_id,
            memory_id=entity.id,
            entity_name=entity.name,
            entity_type=entity.entity_type,
            interaction_type=interaction_type,
            source=source,
            query=query,
            score=score,
            recorded_access=recorded_access,
        )

    def record_memory_interaction(
        self,
        *,
        group_id: str,
        memory_id: str,
        entity_name: str | None,
        entity_type: str | None,
        interaction_type: str | None,
        source: str,
        query: str,
        score: float | None,
        recorded_access: bool,
        result_type: str = "entity",
    ) -> None:
        if not interaction_type:
            return

        if self._cfg.recall_telemetry_enabled or self._cfg.recall_usage_feedback_enabled:
            publish_memory_interaction(
                self._event_bus,
                MemoryInteractionEvent(
                    group_id=group_id,
                    entity_id=memory_id,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    interaction_type=interaction_type,
                    source=source,
                    query=query,
                    score=score,
                    recorded_access=recorded_access,
                ),
            )

        self._recall_need_controller.record_interaction(
            group_id,
            interaction_type,
            result_type=result_type,
            memory_id=memory_id,
            # Ticket #37: the controller classifies this into a write surface so
            # a 0 count can be told apart from a surface with no emitter.
            source=source,
        )


class RecallMemoryInteractionApplier:
    """Apply explicit post-response memory feedback to recalled entities and cues."""

    _VALID_TYPES = {
        "surfaced",
        "selected",
        "used",
        "confirmed",
        "dismissed",
        "corrected",
    }

    # M1.2: explicit feedback records tier-tagged access events.
    _ACCESS_TIER_BY_INTERACTION = {
        "used": "used",
        "confirmed": "confirmed",
        "corrected": "corrected",
    }

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cue_feedback_recorder: RecallCueFeedbackRecorder,
        entity_access_recorder: RecallEntityAccessRecorder,
        interaction_recorder: RecallInteractionRecorder,
        recall_need_controller: RecallNeedController,
    ) -> None:
        self._cfg = cfg
        self._graph = graph_store
        self._activation = activation_store
        self._cue_feedback_recorder = cue_feedback_recorder
        self._entity_access_recorder = entity_access_recorder
        self._interaction_recorder = interaction_recorder
        self._recall_need_controller = recall_need_controller

    async def apply(
        self,
        memory_ids: list[str],
        *,
        interaction_type: str,
        group_id: str = "default",
        query: str = "",
        source: str = "recall_feedback",
        result_lookup: dict[str, dict] | None = None,
    ) -> None:
        if interaction_type not in self._VALID_TYPES:
            raise ValueError(f"Unknown interaction_type: {interaction_type}")

        access_tier = self._ACCESS_TIER_BY_INTERACTION.get(interaction_type)

        seen_ids: set[str] = set()
        now = time.time()
        for memory_id in memory_ids:
            if not memory_id or memory_id in seen_ids:
                continue
            seen_ids.add(memory_id)

            metadata = result_lookup.get(memory_id, {}) if result_lookup else {}
            result_type = metadata.get("result_type")
            if result_type is None and isinstance(memory_id, str) and memory_id.startswith("cue:"):
                result_type = "cue_episode"
            if result_type == "cue_episode":
                await self._apply_cue_interaction(
                    memory_id,
                    metadata,
                    group_id=group_id,
                    query=query,
                    source=source,
                    interaction_type=interaction_type,
                )
                continue

            await self._apply_entity_interaction(
                memory_id,
                metadata,
                group_id=group_id,
                query=query,
                source=source,
                interaction_type=interaction_type,
                access_tier=access_tier,
                timestamp=now,
            )

    async def _apply_cue_interaction(
        self,
        memory_id: str,
        metadata: dict,
        *,
        group_id: str,
        query: str,
        source: str,
        interaction_type: str,
    ) -> None:
        episode_id = metadata.get("episode_id")
        if not episode_id and isinstance(memory_id, str) and memory_id.startswith("cue:"):
            episode_id = memory_id.split(":", 1)[1]
        if not episode_id:
            return
        episode = await self._graph.get_episode_by_id(episode_id, group_id)
        if episode is None:
            return

        cue_score = metadata.get("score")
        await self._cue_feedback_recorder.record_cue_feedback(
            episode,
            float(cue_score) if cue_score is not None else 0.0,
            query,
            interaction_type=interaction_type,
            count_hit=bool(metadata.get("count_hit", False)),
        )
        self._recall_need_controller.record_interaction(
            group_id,
            interaction_type,
            result_type="cue_episode",
            memory_id=f"cue:{episode_id}",
            source=source,
        )

    async def _apply_entity_interaction(
        self,
        memory_id: str,
        metadata: dict,
        *,
        group_id: str,
        query: str,
        source: str,
        interaction_type: str,
        access_tier: str | None,
        timestamp: float,
    ) -> None:
        entity_name = metadata.get("entity_name")
        entity_type = metadata.get("entity_type")
        score = metadata.get("score")

        entity = await self._graph.get_entity(memory_id, group_id)
        if entity is not None:
            entity_name = entity.name
            entity_type = entity.entity_type

        recorded_access = False
        if access_tier is not None and entity is not None:
            await self._entity_access_recorder.record_entity_access(
                entity,
                group_id=group_id,
                query=query,
                source=source,
                timestamp=timestamp,
                tier=access_tier,
            )
            recorded_access = True

        # Thompson Sampling posterior updates were deleted (M5.3/F4 KILL):
        # confirmed/corrected interactions record tier-tagged access events
        # only (the M1.2 usage tiers above).

        self._interaction_recorder.record_memory_interaction(
            group_id=group_id,
            memory_id=memory_id,
            entity_name=entity_name,
            entity_type=entity_type,
            interaction_type=interaction_type,
            source=source,
            query=query,
            score=score,
            recorded_access=recorded_access,
            result_type="entity",
        )


def extract_recall_targets(recall_results: list[dict]) -> list[dict]:
    """Extract deduplicated feedback targets from raw recall results."""
    targets: list[dict] = []
    seen_ids: set[str] = set()

    for result in recall_results:
        result_type = result.get("result_type")
        cue = result.get("cue")
        if result_type == "cue_episode" or (result_type is None and isinstance(cue, dict)):
            if not isinstance(cue, dict):
                continue
            episode = result.get("episode", {})
            episode_id = cue.get("episode_id") or episode.get("id")
            if not episode_id:
                continue
            lookup_id = f"cue:{episode_id}"
            if lookup_id in seen_ids:
                continue
            seen_ids.add(lookup_id)
            targets.append(
                {
                    "lookup_id": lookup_id,
                    "result_type": "cue_episode",
                    "episode_id": episode_id,
                    "cue_text": cue.get("cue_text"),
                    "supporting_spans": cue.get("supporting_spans", []),
                    "score": result.get("score"),
                    # Post-response upgrades should not double-count the initial cue hit.
                    "count_hit": False,
                }
            )
            continue

        entity = result.get("entity")
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("id")
        if not entity_id or entity_id in seen_ids:
            continue
        seen_ids.add(entity_id)
        targets.append(
            {
                "lookup_id": entity_id,
                "result_type": "entity",
                "entity_id": entity_id,
                "entity_name": entity.get("name"),
                "entity_type": entity.get("type"),
                "score": result.get("score"),
            }
        )

    return targets


def extract_recall_entities(recall_results: list[dict]) -> list[dict]:
    """Extract deduplicated entity metadata from raw recall results."""
    return [
        target
        for target in extract_recall_targets(recall_results)
        if target.get("result_type") == "entity"
    ]


def partition_recall_targets_by_usage(
    response_text: str,
    recall_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition recalled entities/cues into used vs dismissed via response mention."""
    targets = extract_recall_targets(recall_results)
    if not targets:
        return [], []

    normalized_response = _normalize_text(response_text)
    if not normalized_response:
        return [], targets

    used: list[dict] = []
    dismissed: list[dict] = []
    haystack = f" {normalized_response} "

    for target in targets:
        if target.get("result_type") == "cue_episode":
            if _matches_cue_content(
                haystack,
                target.get("cue_text"),
                target.get("supporting_spans", []),
            ):
                used.append(target)
            else:
                dismissed.append(target)
            continue

        name = target.get("entity_name")
        if _matches_entity_name(haystack, name):
            used.append(target)
        else:
            dismissed.append(target)

    return used, dismissed


def partition_recall_entities_by_usage(
    response_text: str,
    recall_results: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition recalled entities into used vs dismissed via response mention."""
    used, dismissed = partition_recall_targets_by_usage(response_text, recall_results)
    return (
        [target for target in used if target.get("result_type") == "entity"],
        [target for target in dismissed if target.get("result_type") == "entity"],
    )


def _matches_entity_name(normalized_response: str, entity_name: str | None) -> bool:
    """Heuristic full-name match against normalized response text."""
    normalized_name = _normalize_text(entity_name or "")
    if not normalized_name:
        return False

    parts = normalized_name.split()
    if len(parts) == 1 and len(parts[0]) < 3:
        return False

    pattern = rf"(?<![a-z0-9]){re.escape(normalized_name)}(?![a-z0-9])"
    return re.search(pattern, normalized_response) is not None


def _matches_cue_content(
    normalized_response: str,
    cue_text: str | None,
    supporting_spans: list[str] | None,
) -> bool:
    """Heuristic span match for cue-backed recall results."""
    candidates = list(supporting_spans or [])
    if cue_text:
        candidates.append(cue_text)

    for candidate in candidates:
        if _matches_text_fragment(normalized_response, candidate):
            return True
    return False


def _matches_text_fragment(normalized_response: str, text: str | None) -> bool:
    """Match a meaningful fragment of cue text against the normalized response."""
    normalized_text = _normalize_text(text or "")
    if not normalized_text:
        return False
    if len(normalized_text) >= 12 and _contains_phrase(normalized_response, normalized_text):
        return True

    label_tokens = {"mentions", "spans", "quotes", "time"}
    tokens = [
        token for token in normalized_text.split() if len(token) >= 4 and token not in label_tokens
    ]
    for size in range(min(4, len(tokens)), 1, -1):
        for idx in range(len(tokens) - size + 1):
            phrase = " ".join(tokens[idx : idx + size])
            if len(phrase) < 10:
                continue
            if _contains_phrase(normalized_response, phrase):
                return True
    return False


def _contains_phrase(normalized_response: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, normalized_response) is not None


def _normalize_text(text: str) -> str:
    """Normalize text for cheap mention matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
