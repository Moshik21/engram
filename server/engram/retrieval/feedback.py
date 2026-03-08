"""Recall telemetry, interaction, and usage-detection helpers."""

from __future__ import annotations

import re

from engram.events.bus import EventBus
from engram.models.recall import MemoryInteractionEvent, MemoryNeed


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


def extract_recall_targets(recall_results: list[dict]) -> list[dict]:
    """Extract deduplicated feedback targets from raw recall results."""
    targets: list[dict] = []
    seen_ids: set[str] = set()

    for result in recall_results:
        result_type = result.get("result_type")
        cue = result.get("cue")
        if result_type == "cue_episode" or (
            result_type is None and isinstance(cue, dict)
        ):
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
        token
        for token in normalized_text.split()
        if len(token) >= 4 and token not in label_tokens
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
