"""Write-side attention: deterministic machinery-class detection at capture.

M1.1/M1.2 (AGENT_EXPERIENCE_GOAL, P1): what enters *semantic space* is a
decision, not a default. Machinery-class episodes — protocol frames, tool-use
id dumps, task-notification shapes, exit-code dumps, path spam, harness
boilerplate — stay stored and BM25/grep-reachable, but earn no capture-time
vectors and are skipped by the mop vector drain.

Zero LLM. Pure regex + token statistics, biased hard toward zero false
positives on substantive content: when unsure, classify substantive.
"""

from __future__ import annotations

import json
import re
from typing import Any

SALIENCE_MACHINERY = "machinery"
SALIENCE_SUBSTANTIVE = "substantive"

# Sources that are pure session-lifecycle traffic (never substantive content).
_MACHINERY_SOURCES = frozenset({"auto:session", "claude:precompact"})

# Hook captures carry a "[role|project]" or "[role|project|session]" frame header.
_FRAME_HEADER_RE = re.compile(r"^\[(?P<role>[A-Za-z][A-Za-z-]*)\|[^\]\n]{0,200}\][ \t]*")
_MACHINERY_HEADER_ROLES = frozenset({"session-start", "session-end", "compaction"})

_SESSION_LIFECYCLE_RE = re.compile(r"^(new session started|session ended)\s*$", re.IGNORECASE)
_COMPACTION_RE = re.compile(r"^context compaction\b", re.IGNORECASE)
_HARNESS_BOILERPLATE_RE = re.compile(
    r"^caveat: the messages below were generated",
    re.IGNORECASE,
)
_EXIT_CODE_LINE_RE = re.compile(r"^\(?exit code[: ]\s*-?\d+\)?\s*$", re.IGNORECASE)

# Protocol/XML frames the harness emits around commands and task results.
_MACHINERY_XML_TAG_RE = re.compile(
    r"^<(task-notification|task-id|tool-use-id|output-file|system-reminder|"
    r"command-name|command-message|command-args|command-contents|"
    r"local-command-stdout|local-command-stderr|local-command-caveat|"
    r"command-output|tool_use_error)\b"
)

_TOOL_USE_ID_RE = re.compile(r"^toolu_[A-Za-z0-9]{8,}$")
# Absolute path with at least two segments ("/goal" alone is a slash command).
_PATH_TOKEN_RE = re.compile(r"^[\"'`(]?(/|~/)[\w@%+=:,.~-]+(/[\w@%+=:,.~-]*)+[\"'`),.:;]?$")
# Opaque machine identifier: lowercase alnum with at least one digit (task ids,
# hex ids). The digit requirement keeps ordinary words out.
_OPAQUE_ID_TOKEN_RE = re.compile(r"^[a-z0-9_-]*\d[a-z0-9_-]*$")
# A natural-language word: 3+ letters, optional trailing punctuation.
_WORD_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'’-]{2,}[.,;:!?)\]}]*$")


def classify_salience(content: str, source: str | None = None) -> str:
    """Classify episode content as machinery or substantive. Deterministic, no LLM."""
    if source in _MACHINERY_SOURCES:
        return SALIENCE_MACHINERY

    text = (content or "").strip()
    if not text:
        # Nothing to embed; the vector path skips empties anyway.
        return SALIENCE_MACHINERY

    header = _FRAME_HEADER_RE.match(text)
    body = text[header.end() :].strip() if header else text
    if header and header.group("role").lower() in _MACHINERY_HEADER_ROLES:
        return SALIENCE_MACHINERY

    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not lines:
        return SALIENCE_MACHINERY
    first = lines[0]

    if (
        _SESSION_LIFECYCLE_RE.match(first)
        or _COMPACTION_RE.match(first)
        or _HARNESS_BOILERPLATE_RE.match(first)
        or _EXIT_CODE_LINE_RE.match(first)
    ):
        return SALIENCE_MACHINERY

    # Protocol frames: a machinery XML tag opening any of the first lines.
    for line in lines[:3]:
        if _MACHINERY_XML_TAG_RE.match(line):
            return SALIENCE_MACHINERY

    # Stripped task-notification shape: the body leads with a bare opaque
    # task id on its own line (e.g. "bm6s67jbk" / "wkonv1206"). Pure-hex
    # tokens are excluded — a genuine prompt can lead with a commit hash
    # ("b8c8d07\nis this the right commit to tag?") and must stay substantive.
    if (
        len(lines) > 1
        and len(first.split()) == 1
        and 6 <= len(first) <= 20
        and _OPAQUE_ID_TOKEN_RE.match(first)
        and not re.fullmatch(r"[0-9a-f]{6,40}", first)
    ):
        return SALIENCE_MACHINERY

    tokens = body.split()
    words = sum(1 for tok in tokens if _WORD_TOKEN_RE.match(tok))
    paths = sum(1 for tok in tokens if _PATH_TOKEN_RE.match(tok))
    ids = sum(
        1
        for tok in tokens
        if _TOOL_USE_ID_RE.match(tok) or (len(tok) >= 6 and _OPAQUE_ID_TOKEN_RE.match(tok))
    )

    # tool_use id dump: a toolu_* token with almost no natural language around it.
    if words < 5 and any(_TOOL_USE_ID_RE.match(tok) for tok in tokens):
        return SALIENCE_MACHINERY

    # Path/id spam: the body is dominated by paths and opaque identifiers.
    if len(tokens) >= 3 and words < 4 and (paths + ids) / len(tokens) >= 0.7:
        return SALIENCE_MACHINERY

    return SALIENCE_SUBSTANTIVE


def is_machinery(content: str, source: str | None = None) -> bool:
    """Convenience wrapper: True when content classifies as machinery."""
    return classify_salience(content, source) == SALIENCE_MACHINERY


def vector_index_exempt(episode: Any, cfg: Any | None = None) -> bool:
    """True when this episode must be excluded from semantic vector indexing.

    The single predicate for both the capture-time gate and the mop vector
    drain (``hygiene_ops`` wires it as a one-line filter). Storage, BM25, and
    grep reachability are never affected by this predicate.

    Pass ``cfg`` to honor the ``salience_gated_embedding_enabled`` kill
    switch; with ``cfg=None`` the gate is assumed enabled.
    """
    if cfg is not None and not getattr(cfg, "salience_gated_embedding_enabled", True):
        return False
    salience_class = getattr(episode, "salience_class", "") or ""
    if not salience_class:
        salience_class = decode_salience_class(getattr(episode, "encoding_context", None))
    if not salience_class:
        salience_class = classify_salience(
            getattr(episode, "content", "") or "",
            getattr(episode, "source", None),
        )
    return salience_class == SALIENCE_MACHINERY


def decode_salience_class(encoding_context: str | None) -> str:
    """Read a persisted salience class out of the encoding_context JSON blob."""
    if not encoding_context:
        return ""
    try:
        data = json.loads(encoding_context)
    except (TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    value = data.get("salience_class", "")
    return value if isinstance(value, str) else ""


def encode_salience_class(encoding_context: str | None, salience_class: str) -> str | None:
    """Embed a salience class into the encoding_context JSON blob for storage.

    Returns the input unchanged when ``salience_class`` is empty (byte-identity
    for the kill switch) or when the existing encoding_context is not a JSON
    object (e.g. reflect cluster keys) — the classifier re-derives on read in
    that case.
    """
    if not salience_class:
        return encoding_context
    if encoding_context:
        try:
            data = json.loads(encoding_context)
        except (TypeError, ValueError):
            return encoding_context
        if not isinstance(data, dict):
            return encoding_context
    else:
        data = {}
    data["salience_class"] = salience_class
    return json.dumps(data)


# --- M1.4 squatter guard: observation-class sources -------------------------

# Episode sources whose entities need >=2-episode corroboration before full
# commit (the squatter-entity class came from one-shot milestone observations).
_OBSERVATION_SOURCES = frozenset({"mcp_observe", "mcp_observe_attachment", "api_auto_observe"})
_OBSERVATION_SOURCE_PREFIXES = ("auto:", "axi")


def is_observation_source(source: str | None) -> bool:
    """True when episodes from this source are observation-class captures."""
    if not source:
        return False
    return source in _OBSERVATION_SOURCES or source.startswith(_OBSERVATION_SOURCE_PREFIXES)
