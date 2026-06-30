"""Route-facing helpers for Capture-stage observe and remember writes."""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from json import JSONDecodeError
from typing import Any

from engram.config import ActivationConfig
from engram.extraction.client_proposals import events_to_proposals, proposals_to_evidence
from engram.ingestion.adjudication_surface import load_client_enabled_episode_adjudication_requests
from engram.ingestion.presenter import (
    memory_write_contract,
    present_api_memory_write,
    present_api_observe_skip,
    present_mcp_memory_write,
)
from engram.models.episode import Attachment
from engram.retrieval.memory_operations import (
    MemoryOperationSample,
    measured_memory_operation,
    record_manager_memory_operation,
)
from engram.utils.dates import utc_now

logger = logging.getLogger(__name__)
_MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS = 0.075
_MCP_WRITE_LIVE_TURN_TIMEOUT_SECONDS = 0.01
_AGENT_WRITE_CAPTURE_STORE_TIMEOUT_MS = 100
_SESSION_RECENT_PACKET_SCOPE = "session_recent"
_SESSION_RECENT_PACKET_LIMIT = 5


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _text_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return _text_value(value.get("content")) or _text_value(value.get("text"))
    if isinstance(value, list | tuple):
        parts = [part for item in value if (part := _text_value(item))]
        return "\n".join(parts) if parts else None
    return None


def _project_from_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    name = cwd.rstrip("/").rsplit("/", 1)[-1]
    return name or None


def _source_for_role(role: str | None) -> str:
    if role == "assistant":
        return "auto:response"
    if role == "user":
        return "auto:prompt"
    return "auto:hook"


def _tag_hook_content(role: str, project: str, content: str) -> str:
    return f"[{role}|{project}] {content}"


_HARNESS_XML_TAG_STRIP = re.compile(r"<[^>]{1,80}>", re.MULTILINE)


def normalize_harness_observe_content(content: str) -> str:
    """Strip harness/XML wrappers so narrow extraction sees plain user text."""
    text = content.strip()
    user_query_match = re.search(
        r"<user_query>\s*(.*?)\s*</user_query>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if user_query_match:
        text = user_query_match.group(1).strip()
    return _HARNESS_XML_TAG_STRIP.sub(" ", text).strip()


def normalize_auto_observe_payload(raw: Mapping[str, Any]) -> dict[str, str | None]:
    """Normalize installed-hook and raw Claude-hook payloads for auto-observe."""
    project = (
        _string_value(raw.get("project"))
        or _project_from_cwd(_string_value(raw.get("cwd")))
        or "unknown"
    )
    raw_role = _string_value(raw.get("role"))
    session_id = _string_value(raw.get("session_id"))
    conversation_date = _string_value(raw.get("conversation_date"))

    content = _text_value(raw.get("content"))
    if content is not None:
        role = raw_role or "user"
        return {
            "content": content,
            "source": _string_value(raw.get("source")) or _source_for_role(role),
            "project": project,
            "role": role,
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    prompt = _text_value(raw.get("prompt"))
    if prompt is not None:
        return {
            "content": _tag_hook_content("user", project, prompt),
            "source": _string_value(raw.get("source")) or "auto:prompt",
            "project": project,
            "role": "user",
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    assistant_response = _text_value(raw.get("last_assistant_message"))
    if assistant_response is not None:
        return {
            "content": _tag_hook_content("assistant", project, assistant_response),
            "source": _string_value(raw.get("source")) or "auto:response",
            "project": project,
            "role": "assistant",
            "session_id": session_id,
            "conversation_date": conversation_date,
        }

    message = raw.get("message")
    if isinstance(message, Mapping):
        role = (
            _string_value(message.get("role"))
            or raw_role
            or _string_value(raw.get("type"))
            or "system"
        )
        message_content = _text_value(message.get("content"))
        if message_content is not None:
            return {
                "content": _tag_hook_content(role, project, message_content),
                "source": _string_value(raw.get("source")) or _source_for_role(role),
                "project": project,
                "role": role,
                "session_id": session_id,
                "conversation_date": conversation_date,
            }

    return {
        "content": "",
        "source": _string_value(raw.get("source")) or "auto:hook",
        "project": project,
        "role": raw_role or "system",
        "session_id": session_id,
        "conversation_date": conversation_date,
    }


def build_api_auto_observe_skip_surface(reason: str) -> dict[str, Any]:
    """Return a non-throwing auto-observe skip response for hook compatibility."""
    return present_api_observe_skip("skipped", reason=reason)


def parse_conversation_date(value: str | None) -> datetime | None:
    """Parse optional ISO conversation dates without failing public writes."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


async def _record_write_operation(
    manager: Any,
    group_id: str,
    finish_operation: Callable[..., MemoryOperationSample],
    *,
    status: str = "ok",
    skip_reason: str | None = None,
    result_count: int = 1,
) -> None:
    try:
        await record_manager_memory_operation(
            manager,
            group_id,
            finish_operation(
                status=status,
                skip_reason=skip_reason,
                result_count=result_count,
            ),
        )
    except Exception:
        logger.debug("failed to record capture memory operation", exc_info=True)


def _camel_stage_name(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def _capture_stage_timings(manager: Any) -> dict[str, float]:
    getter = getattr(manager, "get_last_capture_stage_timings", None)
    if not callable(getter):
        return {}
    timings = getter()
    if inspect.isawaitable(timings):
        close = getattr(timings, "close", None)
        if callable(close):
            close()
        return {}
    if not isinstance(timings, Mapping):
        return {}
    return {
        str(key): float(value)
        for key, value in timings.items()
        if isinstance(value, int | float)
    }


def attach_api_capture_diagnostics(response: dict[str, Any], manager: Any) -> dict[str, Any]:
    timings = _capture_stage_timings(manager)
    if timings:
        response.setdefault("diagnostics", {})["stageTimingsMs"] = {
            _camel_stage_name(key): value for key, value in timings.items()
        }
        _mark_deferred_raw_capture_lifecycle(response, timings, camel_case=True)
    return response


def attach_mcp_capture_diagnostics(response: dict[str, Any], manager: Any) -> dict[str, Any]:
    timings = _capture_stage_timings(manager)
    if timings:
        response.setdefault("diagnostics", {})["stage_timings_ms"] = timings
        _mark_deferred_raw_capture_lifecycle(response, timings, camel_case=False)
    return response


def _mark_deferred_raw_capture_lifecycle(
    response: dict[str, Any],
    timings: Mapping[str, float],
    *,
    camel_case: bool,
) -> None:
    if "capture_store_timeout" not in timings or "capture_store" in timings:
        return
    lifecycle = response.get("lifecycle")
    if not isinstance(lifecycle, dict):
        return
    if camel_case:
        lifecycle["captureStatus"] = "deferred"
        lifecycle["projectionStatus"] = "pending"
    else:
        lifecycle["capture_status"] = "deferred"
        lifecycle["projection_status"] = "pending"


def attach_mcp_side_effect_diagnostics(
    response: dict[str, Any],
    timings: Mapping[str, float],
) -> dict[str, Any]:
    if timings:
        response.setdefault("diagnostics", {}).setdefault("stage_timings_ms", {}).update(
            {
                str(key): float(value)
                for key, value in timings.items()
                if isinstance(value, int | float)
            }
        )
    return response


def _close_awaitable(value: Any) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _cache_recent_observation_packet(
    manager: Any,
    *,
    group_id: str,
    episode_id: str,
    content: str,
    source: str,
    packet_source: str,
) -> None:
    cache_packets = getattr(manager, "cache_memory_packets", None)
    if not callable(cache_packets) or not content.strip():
        return
    summary = " ".join(content.split())[:240]
    why_now = "Recent observe turn captured in this session."
    packet = {
        "packet_type": "recent_observation",
        "title": f"Recent Observation: {episode_id}",
        "summary": summary,
        "why_now": why_now,
        "confidence": 0.9,
        "entity_ids": [],
        "relationship_ids": [],
        "episode_ids": [episode_id],
        "evidence_lines": [summary],
        "provenance": [f"episode:{episode_id}", f"source:{source or 'mcp'}"],
        "supporting_intents": ["session_recent_observation"],
        "trust": {
            "freshness": "fresh",
            "source": packet_source,
            "confidence": 0.9,
            "why_now": why_now,
            "provenance_count": 2,
            "evidence_count": 1,
            "belief_status": "unknown",
            "confirmed_count": 0,
            "corrected_count": 0,
            "dismissed_count": 0,
            "last_confirmed_at": None,
            "last_corrected_at": None,
            "last_dismissed_at": None,
        },
    }
    packets = _rolling_session_recent_packets(
        manager,
        group_id=group_id,
        newest_packet=packet,
    )
    try:
        result = cache_packets(
            group_id,
            scope=_SESSION_RECENT_PACKET_SCOPE,
            topic_hint=None,
            project_path=None,
            packets=packets,
            persist=False,
        )
        if inspect.isawaitable(result):
            _close_awaitable(result)
    except Exception:
        logger.debug("Failed to cache recent observation packet", exc_info=True)


def _rolling_session_recent_packets(
    manager: Any,
    *,
    group_id: str,
    newest_packet: Mapping[str, Any],
) -> list[dict[str, Any]]:
    packets = [_strip_cache_scope(dict(newest_packet))]
    get_recent = getattr(manager, "get_recent_cached_memory_packets", None)
    if callable(get_recent):
        try:
            existing = get_recent(
                group_id,
                scopes=(_SESSION_RECENT_PACKET_SCOPE,),
                limit_packets=_SESSION_RECENT_PACKET_LIMIT - 1,
                sync_persistent=False,
            )
        except Exception:
            existing = []
        if inspect.isawaitable(existing):
            _close_awaitable(existing)
            existing = []
        if isinstance(existing, list):
            packets.extend(
                _strip_cache_scope(dict(packet))
                for packet in existing
                if isinstance(packet, Mapping)
            )
    return _dedupe_recent_observation_packets(
        packets,
        limit=_SESSION_RECENT_PACKET_LIMIT,
    )


def _strip_cache_scope(packet: dict[str, Any]) -> dict[str, Any]:
    packet.pop("_cache_scope", None)
    return packet


def _dedupe_recent_observation_packets(
    packets: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for packet in packets:
        fingerprint = _recent_packet_fingerprint(packet)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        selected.append(packet)
        if len(selected) >= limit:
            break
    return selected


def _recent_packet_fingerprint(packet: Mapping[str, Any]) -> str:
    episode_ids = packet.get("episode_ids") or packet.get("episodeIds") or []
    if isinstance(episode_ids, list | tuple) and episode_ids:
        return "episode:" + "|".join(str(item) for item in episode_ids)
    provenance = packet.get("provenance") or []
    if isinstance(provenance, list | tuple) and provenance:
        return "provenance:" + "|".join(str(item) for item in provenance)
    packet_type = packet.get("packet_type") or packet.get("packetType")
    return f"{packet_type}:{packet.get('title')}:{packet.get('summary')}"


async def _run_mcp_write_side_effect(
    name: str,
    awaitable: Any,
    timings: dict[str, float],
    *,
    background_on_timeout: bool = False,
    timeout_seconds: float | None = None,
) -> bool:
    """Run non-capture MCP write side effects without making writes wait on them."""
    if not inspect.isawaitable(awaitable):
        return True
    timeout = (
        _MCP_WRITE_SIDE_EFFECT_TIMEOUT_SECONDS
        if timeout_seconds is None
        else max(0.0, float(timeout_seconds))
    )
    started = time.perf_counter()
    task: asyncio.Task[Any] | None = None
    try:
        if background_on_timeout:
            task = asyncio.create_task(awaitable)
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=timeout,
            )
        else:
            await asyncio.wait_for(
                awaitable,
                timeout=timeout,
            )
        timings[name] = round((time.perf_counter() - started) * 1000, 4)
        return True
    except TimeoutError:
        timings[f"{name}_timeout"] = round((time.perf_counter() - started) * 1000, 4)
        logger.debug(
            "MCP write side effect %s exceeded %.0fms",
            name,
            timeout * 1000,
        )
        if task is not None:
            task.add_done_callback(_log_mcp_write_side_effect_failure(name))
        return False
    except Exception:
        timings[f"{name}_error"] = round((time.perf_counter() - started) * 1000, 4)
        logger.debug("MCP write side effect %s failed", name, exc_info=True)
        return False


async def _run_mcp_write_side_effects(
    items: list[tuple[str, Any, bool] | tuple[str, Any, bool, float | None]],
    timings: dict[str, float],
) -> None:
    """Run independent MCP write side effects in one bounded wait window."""
    await asyncio.gather(
        *(
            _run_mcp_write_side_effect(
                item[0],
                item[1],
                timings,
                background_on_timeout=item[2],
                timeout_seconds=item[3] if len(item) > 3 else None,
            )
            for item in items
        )
    )


def _log_mcp_write_side_effect_failure(name: str):
    def _done(task: asyncio.Task[Any]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.debug("Background MCP write side effect %s failed", name, exc_info=True)

    return _done


def build_observation_attachment(
    *,
    mime_type: str,
    data_url: str,
    description: str | None = None,
) -> Attachment:
    """Build the attachment payload used by image/file observe surfaces."""
    return Attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )


async def store_observation(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str | None = None,
    session_id: str | None = None,
    conversation_date: datetime | None = None,
    attachments: list[Attachment] | None = None,
    pass_session_id: bool = False,
    pass_conversation_date: bool = False,
    pass_attachments: bool = False,
    capture_store_timeout_ms: int | None = None,
) -> str:
    """Store a raw Capture-stage observation through the manager facade."""
    kwargs: dict[str, Any] = {
        "content": content,
        "group_id": group_id,
        "source": source,
    }
    if session_id is not None or pass_session_id:
        kwargs["session_id"] = session_id
    if conversation_date is not None or pass_conversation_date:
        kwargs["conversation_date"] = conversation_date
    if attachments is not None or pass_attachments:
        kwargs["attachments"] = attachments
    if capture_store_timeout_ms is not None:
        kwargs["capture_store_timeout_ms"] = capture_store_timeout_ms
    return await manager.store_episode(**kwargs)


async def ingest_projecting_memory(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str | None = None,
    session_id: str | None = None,
    conversation_date: datetime | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    attachments: list[Attachment] | None = None,
    pass_session_id: bool = False,
    pass_attachments: bool = False,
) -> str:
    """Run the Capture -> Project compatibility write through the manager facade."""
    kwargs: dict[str, Any] = {
        "content": content,
        "group_id": group_id,
        "source": source,
        "conversation_date": conversation_date,
        "proposed_entities": proposed_entities,
        "proposed_relationships": proposed_relationships,
        "model_tier": model_tier,
    }
    if session_id is not None or pass_session_id:
        kwargs["session_id"] = session_id
    if attachments is not None or pass_attachments:
        kwargs["attachments"] = attachments
    return await manager.ingest_episode(**kwargs)


async def build_api_auto_observe_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    session_id: str | None = None,
    conversation_date: str | None = None,
    auto_observe_enabled: bool = True,
    dedup_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Run the REST auto-observe Capture policy behind an ingestion boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="api_auto_observe",
        mode="api_auto_observe",
    )
    if not auto_observe_enabled:
        await _record_write_operation(
            manager,
            group_id,
            finish_operation,
            status="skipped",
            skip_reason="disabled",
            result_count=0,
        )
        return present_api_observe_skip("skipped", reason="disabled")

    content = normalize_harness_observe_content(content)

    if not content or len(content.strip()) < 10:
        await _record_write_operation(
            manager,
            group_id,
            finish_operation,
            status="skipped",
            skip_reason="too_short",
            result_count=0,
        )
        return present_api_observe_skip("skipped", reason="too_short")

    if dedup_check is not None and dedup_check(content):
        await _record_write_operation(
            manager,
            group_id,
            finish_operation,
            status="skipped",
            skip_reason="dedup_skipped",
            result_count=0,
        )
        return present_api_observe_skip("dedup_skipped")

    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session_id,
        conversation_date=parse_conversation_date(conversation_date),
        pass_conversation_date=True,
        capture_store_timeout_ms=_AGENT_WRITE_CAPTURE_STORE_TIMEOUT_MS,
    )
    _cache_recent_observation_packet(
        manager,
        group_id=group_id,
        episode_id=episode_id,
        content=content,
        source=source,
        packet_source="api_auto_observe",
    )
    await _record_write_operation(manager, group_id, finish_operation)
    return attach_api_capture_diagnostics(present_api_memory_write(
        memory_write_contract("observe", episode_id),
        status="observed",
    ), manager)


async def build_api_auto_observe_request_surface(
    manager: Any,
    *,
    request: Any,
    group_id: str,
    auto_observe_enabled: bool = True,
    dedup_check: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Parse and run the REST auto-observe Capture policy behind one boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="api_auto_observe",
        mode="api_auto_observe",
    )
    try:
        raw_body = await request.json()
    except JSONDecodeError:
        await _record_write_operation(
            manager,
            group_id,
            finish_operation,
            status="skipped",
            skip_reason="invalid_json",
            result_count=0,
        )
        return build_api_auto_observe_skip_surface("invalid_json")

    if not isinstance(raw_body, Mapping):
        await _record_write_operation(
            manager,
            group_id,
            finish_operation,
            status="skipped",
            skip_reason="unsupported_payload",
            result_count=0,
        )
        return build_api_auto_observe_skip_surface("unsupported_payload")

    body = normalize_auto_observe_payload(raw_body)
    return await build_api_auto_observe_surface(
        manager,
        content=body["content"] or "",
        group_id=group_id,
        source=body["source"] or "auto:hook",
        session_id=body["session_id"],
        conversation_date=body["conversation_date"],
        auto_observe_enabled=auto_observe_enabled,
        dedup_check=dedup_check,
    )


async def build_api_observe_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    conversation_date: str | None = None,
    events: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the REST observe write path behind a Capture-stage surface boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="api_observe",
        mode="api_observe",
    )
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        conversation_date=parse_conversation_date(conversation_date),
        pass_conversation_date=True,
    )
    # observe stays cheap: persist event annotations as deferred evidence (CQRS
    # bridge) without projecting. Consolidation later promotes them into dated
    # Event nodes once the threshold/corroboration bar is met.
    await persist_observe_event_annotations(
        manager,
        episode_id=episode_id,
        group_id=group_id,
        content=content,
        events=events,
        conversation_date=parse_conversation_date(conversation_date),
    )
    await _record_write_operation(manager, group_id, finish_operation)
    return attach_api_capture_diagnostics(present_api_memory_write(
        memory_write_contract("observe", episode_id),
        status="observed",
    ), manager)


async def build_api_attachment_observe_write_surface(
    manager: Any,
    *,
    data_url: str,
    mime_type: str,
    attachment_kind: str,
    fallback_content: str,
    group_id: str,
    description: str = "",
    source: str = "api",
) -> dict[str, Any]:
    """Run a REST image/file observe write behind a Capture-stage boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="api_observe_attachment",
        mode="api_observe_attachment",
    )
    attachment = build_observation_attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )
    episode_id = await store_observation(
        manager,
        content=description or fallback_content,
        group_id=group_id,
        source=source,
        attachments=[attachment],
    )
    await _record_write_operation(manager, group_id, finish_operation)
    return attach_api_capture_diagnostics(present_api_memory_write(
        memory_write_contract("observe", episode_id, attachment_kind=attachment_kind),
        status="stored",
        include_legacy_episode_id=True,
    ), manager)


async def build_api_remember_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    source: str,
    conversation_date: str | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    events: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the REST remember write path behind a Capture -> Project boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="remember",
        source="api_remember",
        mode="api_remember",
    )
    # Dated event annotations materialize as first-class Event nodes + OCCURRED_ON
    # edges by flowing through the existing client-proposal evidence pipeline.
    proposed_entities, proposed_relationships = merge_event_proposals(
        events,
        proposed_entities,
        proposed_relationships,
    )
    episode_id = await ingest_projecting_memory(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        conversation_date=parse_conversation_date(conversation_date),
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
        model_tier=model_tier,
    )
    adjudications = await load_client_enabled_episode_adjudication_requests(
        manager,
        episode_id=episode_id,
        group_id=group_id,
    )
    await _record_write_operation(manager, group_id, finish_operation)
    return attach_api_capture_diagnostics(present_api_memory_write(
        memory_write_contract(
            "remember",
            episode_id,
            adjudication_requests=adjudications,
        ),
        status="remembered",
    ), manager)


def merge_event_proposals(
    events: list[dict] | None,
    proposed_entities: list[dict] | None,
    proposed_relationships: list[dict] | None,
) -> tuple[list[dict] | None, list[dict] | None]:
    """Fold dated event annotations into the client-proposal payloads.

    Returns the merged (proposed_entities, proposed_relationships) so events ride
    the existing evidence pipeline and materialize as Event nodes + OCCURRED_ON
    edges. Leaves the inputs untouched when there are no events.
    """
    if not events:
        return proposed_entities, proposed_relationships
    event_entities, event_rels = events_to_proposals(events)
    if not event_entities and not event_rels:
        return proposed_entities, proposed_relationships
    merged_entities = list(proposed_entities or []) + event_entities
    merged_rels = list(proposed_relationships or []) + event_rels
    return merged_entities, merged_rels


def _evidence_candidate_to_storage_row(candidate: Any, status: str) -> dict[str, Any]:
    """Serialize an EvidenceCandidate into an episode_evidence storage row."""
    created_at = candidate.created_at
    return {
        "evidence_id": candidate.evidence_id,
        "episode_id": candidate.episode_id,
        "fact_class": candidate.fact_class,
        "confidence": candidate.confidence,
        "source_type": candidate.source_type,
        "extractor_name": candidate.extractor_name,
        "payload": candidate.payload,
        "source_span": candidate.source_span,
        "corroborating_signals": candidate.corroborating_signals,
        "ambiguity_tags": candidate.ambiguity_tags,
        "ambiguity_score": candidate.ambiguity_score,
        "adjudication_request_id": candidate.adjudication_request_id,
        "created_at": (
            created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)
        ),
        "status": status,
        "commit_reason": "observe_event_annotation",
    }


async def persist_observe_event_annotations(
    manager: Any,
    *,
    episode_id: str,
    group_id: str,
    content: str,
    events: list[dict] | None,
    conversation_date: datetime | None = None,
) -> None:
    """Persist observe-time event annotations as deferred evidence (no projection).

    This is the CQRS bridge for cheap capture: events are span-verified, turned
    into Event/OCCURRED_ON evidence, and stored with status='deferred'. The
    consolidation evidence-adjudication phase later promotes them once corroborated.
    """
    if not events:
        return
    graph = getattr(manager, "_graph", None)
    store_evidence = getattr(graph, "store_evidence", None)
    if not callable(store_evidence):
        logger.debug("observe events skipped: graph store lacks store_evidence")
        return
    event_entities, event_rels = events_to_proposals(events)
    if not event_entities and not event_rels:
        return
    candidates = proposals_to_evidence(
        event_entities,
        event_rels,
        episode_id,
        group_id,
        episode_content=content,
        reference_date=conversation_date,
        verify_spans=True,
    )
    if not candidates:
        return
    rows = [_evidence_candidate_to_storage_row(c, status="deferred") for c in candidates]
    try:
        await store_evidence(rows, group_id=group_id, default_status="deferred")
    except Exception:
        logger.debug("Failed to persist observe event annotations", exc_info=True)


def record_mcp_memory_write_activity(session: Any) -> None:
    """Update MCP session activity after a successful Capture write."""
    session.episode_count += 1
    session.last_activity = utc_now()


async def build_mcp_remember_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session: Any,
    source: str = "mcp",
    conversation_date: str | None = None,
    proposed_entities: list[dict] | None = None,
    proposed_relationships: list[dict] | None = None,
    model_tier: str = "default",
    image_data: str | None = None,
    image_mime: str = "image/png",
    events: list[dict] | None = None,
    activation_cfg: ActivationConfig | None = None,
    ingest_live_turn: Callable[..., Any],
    recall_middleware: Callable[..., Any],
) -> dict[str, Any]:
    """Run the MCP remember write path behind a Capture-stage surface boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="remember",
        source="mcp_remember",
        mode="mcp_remember",
    )
    attachments = []
    if image_data:
        attachments.append(
            build_observation_attachment(
                mime_type=image_mime,
                data_url=image_data,
                description=content[:200] if content else "",
            )
        )

    # Dated event annotations materialize as first-class Event nodes + OCCURRED_ON
    # edges by flowing through the existing client-proposal evidence pipeline.
    proposed_entities, proposed_relationships = merge_event_proposals(
        events,
        proposed_entities,
        proposed_relationships,
    )

    episode_id = await ingest_projecting_memory(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=parse_conversation_date(conversation_date),
        proposed_entities=proposed_entities,
        proposed_relationships=proposed_relationships,
        model_tier=model_tier,
        attachments=attachments or None,
        pass_session_id=True,
        pass_attachments=True,
    )
    record_mcp_memory_write_activity(session)
    side_effect_timings: dict[str, float] = {}
    await _run_mcp_write_side_effect(
        "live_turn",
        ingest_live_turn(manager, content, source="remember"),
        side_effect_timings,
        background_on_timeout=True,
        timeout_seconds=_MCP_WRITE_LIVE_TURN_TIMEOUT_SECONDS,
    )

    message = "Memory received. Entities and relationships extracted."
    adjudications: list[dict] = []
    if activation_cfg and activation_cfg.evidence_extraction_enabled:
        message = "Memory received. Evidence extracted and evaluated."
        adjudications = await load_client_enabled_episode_adjudication_requests(
            manager,
            episode_id=episode_id,
            group_id=group_id,
            activation_cfg=activation_cfg,
        )

    response = attach_mcp_capture_diagnostics(present_mcp_memory_write(
        memory_write_contract(
            "remember",
            episode_id,
            adjudication_requests=adjudications,
        ),
        message=message,
    ), manager)
    await _run_mcp_write_side_effect(
        "recall_middleware",
        recall_middleware(content, response, tool_name="remember"),
        side_effect_timings,
    )
    attach_mcp_side_effect_diagnostics(response, side_effect_timings)
    await _record_write_operation(manager, group_id, finish_operation)
    return response


async def build_mcp_observe_write_surface(
    manager: Any,
    *,
    content: str,
    group_id: str,
    session: Any,
    source: str = "mcp",
    conversation_date: str | None = None,
    events: list[dict] | None = None,
    ingest_live_turn: Callable[..., Any],
    recall_middleware: Callable[..., Any],
) -> dict[str, Any]:
    """Run the MCP observe write path behind a Capture-stage surface boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="mcp_observe",
        mode="mcp_observe",
    )
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        conversation_date=parse_conversation_date(conversation_date),
        pass_session_id=True,
        pass_conversation_date=True,
        capture_store_timeout_ms=_AGENT_WRITE_CAPTURE_STORE_TIMEOUT_MS,
    )
    # observe stays cheap: persist event annotations as deferred evidence (CQRS
    # bridge) without projecting. Consolidation later promotes them into dated
    # Event nodes once the threshold/corroboration bar is met.
    await persist_observe_event_annotations(
        manager,
        episode_id=episode_id,
        group_id=group_id,
        content=content,
        events=events,
        conversation_date=parse_conversation_date(conversation_date),
    )
    record_mcp_memory_write_activity(session)
    _cache_recent_observation_packet(
        manager,
        group_id=group_id,
        episode_id=episode_id,
        content=content,
        source=source,
        packet_source="mcp_observe",
    )
    side_effect_timings: dict[str, float] = {}
    response = attach_mcp_capture_diagnostics(present_mcp_memory_write(
        memory_write_contract("observe", episode_id),
        message="Stored for background processing.",
    ), manager)
    await _run_mcp_write_side_effect(
        "live_turn",
        ingest_live_turn(manager, content, source="observe"),
        side_effect_timings,
        background_on_timeout=True,
        timeout_seconds=_MCP_WRITE_LIVE_TURN_TIMEOUT_SECONDS,
    )
    attach_mcp_side_effect_diagnostics(response, side_effect_timings)
    await _record_write_operation(manager, group_id, finish_operation)
    return response


async def build_mcp_observe_recall_surface(
    *,
    content: str,
    response: dict[str, Any],
    recall_middleware: Callable[..., Any],
) -> dict[str, Any]:
    """Attach cheap auto-recall context to an observe response.

    Kept separate from ``build_mcp_observe_write_surface`` so the Capture-stage
    write path stays pure (and is timed in isolation), while the recall
    attachment is an explicit, awaited surface step the tool layer composes.
    For observe this is the cache-only path (no deep recall / get_context), so
    it stays within observe's bulk-capture latency budget. ``recall_middleware``
    mutates ``response`` in place; we return it to fit the ``response = await
    build_*`` tool idiom and the public-surface boundary contract.
    """
    await recall_middleware(content, response, tool_name="observe")
    return response


async def build_mcp_attachment_observe_write_surface(
    manager: Any,
    *,
    data_url: str,
    mime_type: str,
    attachment_kind: str,
    fallback_content: str,
    group_id: str,
    session: Any,
    description: str = "",
    source: str = "mcp",
) -> dict[str, Any]:
    """Run the MCP image/file observe path behind a Capture-stage boundary."""
    _started, finish_operation = measured_memory_operation(
        operation="observe",
        source="mcp_observe_attachment",
        mode="mcp_observe_attachment",
    )
    content = description or fallback_content
    attachment = build_observation_attachment(
        mime_type=mime_type,
        data_url=data_url,
        description=description,
    )
    episode_id = await store_observation(
        manager,
        content=content,
        group_id=group_id,
        source=source,
        session_id=session.session_id,
        attachments=[attachment],
        pass_session_id=True,
    )
    record_mcp_memory_write_activity(session)
    await _record_write_operation(manager, group_id, finish_operation)
    return attach_mcp_capture_diagnostics(present_mcp_memory_write(
        memory_write_contract("observe", episode_id, attachment_kind=attachment_kind),
        message=f"{attachment_kind.title()} stored for background processing.",
    ), manager)
