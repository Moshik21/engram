"""Build tiered memory context for agent prompt loading."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.models.entity import Entity
from engram.storage.protocols import ActivationStore, GraphStore

logger = logging.getLogger(__name__)


class MemoryContextBuilder:
    """Assemble the active memory context exposed through REST and MCP."""

    def __init__(
        self,
        *,
        graph_store: GraphStore,
        activation_store: ActivationStore,
        cfg: ActivationConfig,
        recall: Callable[..., Awaitable[list[dict]]],
        list_intentions: Callable[..., Awaitable[list]],
        resolve_entity_name: Callable[[str, str], Awaitable[str]],
        publish_access_event: Callable[[str, str, str, str, str], Awaitable[None]],
        briefing_cache: dict[tuple[str, str | None], tuple[float, str]] | None = None,
    ) -> None:
        self._graph = graph_store
        self._activation = activation_store
        self._cfg = cfg
        self._recall = recall
        self._list_intentions = list_intentions
        self._resolve_entity_name = resolve_entity_name
        self._publish_access_event = publish_access_event
        self._briefing_cache = briefing_cache if briefing_cache is not None else {}

    @property
    def briefing_cache(self) -> dict[tuple[str, str | None], tuple[float, str]]:
        return self._briefing_cache

    async def entity_to_context_data(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        summary: str,
        group_id: str,
        now: float,
        detail_level: str = "full",
    ) -> dict:
        """Build context data dict for a single entity with activation and facts."""
        result: dict = {
            "name": name,
            "type": entity_type,
            "detail_level": detail_level,
            "id": entity_id,
        }

        if detail_level == "mention":
            result["activation"] = 0.0
            result["summary"] = None
            result["facts"] = []
            result["attributes"] = None
            return result

        from engram.activation.engine import compute_activation

        state = await self._activation.get_activation(entity_id)
        act = 0.0
        if state:
            act = compute_activation(state.access_history, now, self._cfg)
        result["activation"] = act
        result["summary"] = summary

        max_facts = 5 if detail_level == "full" else 2
        facts: list[str] = []
        rels = await self._graph.get_relationships(
            entity_id,
            active_only=True,
            group_id=group_id,
        )
        for relationship in rels[:max_facts]:
            src = await self._resolve_entity_name(relationship.source_id, group_id)
            tgt = await self._resolve_entity_name(relationship.target_id, group_id)
            facts.append(f"{src} {relationship.predicate} {tgt}")
        result["facts"] = facts

        if detail_level == "full":
            entity = await self._graph.get_entity(entity_id, group_id)
            result["attributes"] = entity.attributes if entity else None
        else:
            result["attributes"] = None

        return result

    @staticmethod
    def render_tier(header: str, entities: list[dict], facts: list[str]) -> str:
        """Render a single context tier as markdown with variable resolution."""
        lines = [header, ""]
        for entity_data in entities:
            detail = entity_data.get("detail_level", "full")

            if detail == "mention":
                lines.append(f"- {entity_data['name']} ({entity_data['type']})")
                continue

            summary_part = f" — {entity_data['summary']}" if entity_data.get("summary") else ""
            if detail == "full":
                attrs = entity_data.get("attributes")
                if attrs:
                    attr_parts = [f"{k}: {v}" for k, v in list(attrs.items())[:5]]
                    summary_part += f" [{', '.join(attr_parts)}]"
            lines.append(
                f"- {entity_data['name']} ({entity_data['type']}, "
                f"act={entity_data['activation']:.2f}){summary_part}"
            )
            for fact in entity_data.get("facts", []):
                lines.append(f"  - {fact}")

        entity_facts = set()
        for entity_data in entities:
            entity_facts.update(entity_data.get("facts", []))
        extra_facts = [fact for fact in facts if fact not in entity_facts]
        if extra_facts:
            for fact in extra_facts:
                lines.append(f"  - {fact}")
        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return len(text) // 4

    def invalidate_briefing_cache(self, group_id: str) -> None:
        """Clear briefing cache entries for the given group."""
        keys_to_remove = [key for key in self._briefing_cache if key[0] == group_id]
        for key in keys_to_remove:
            del self._briefing_cache[key]

    def template_briefing(
        self,
        structured_context: str,
        group_id: str,
        topic_hint: str | None,
    ) -> str:
        """Render a brief deterministic narrative from structured context."""
        cache_key = (group_id, topic_hint)
        now = time.time()
        if cache_key in self._briefing_cache:
            timestamp, text = self._briefing_cache[cache_key]
            if now - timestamp < self._cfg.briefing_cache_ttl_seconds:
                return text

        sentences: list[str] = []
        tier1_lines: list[str] = []
        tier2_lines: list[str] = []
        tier3_lines: list[str] = []
        current_tier: list[str] | None = None

        for line in structured_context.split("\n"):
            stripped = line.strip()
            if "Identity" in stripped and stripped.startswith("#"):
                current_tier = tier1_lines
            elif "Project" in stripped and stripped.startswith("#"):
                current_tier = tier2_lines
            elif ("Recent" in stripped or "Activity" in stripped) and stripped.startswith("#"):
                current_tier = tier3_lines
            elif "Intention" in stripped and stripped.startswith("#"):
                current_tier = None
            elif current_tier is not None and stripped.startswith("- "):
                current_tier.append(stripped[2:].strip())

        if tier1_lines:
            sentences.append("Known context: " + "; ".join(tier1_lines[:3]) + ".")
        if tier2_lines:
            prefix = f"Currently working on {topic_hint}: " if topic_hint else "Current focus: "
            sentences.append(prefix + "; ".join(tier2_lines[:3]) + ".")
        if tier3_lines:
            sentences.append("Recent activity: " + "; ".join(tier3_lines[:3]) + ".")

        briefing = " ".join(sentences) if sentences else structured_context
        self._briefing_cache[cache_key] = (now, briefing)
        return briefing

    async def get_context(
        self,
        group_id: str = "default",
        max_tokens: int = 2000,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        """Build a tiered markdown context summary of the most activated memories."""
        from engram.activation.engine import compute_activation

        now = time.time()
        seen_ids: set[str] = set()

        project_entity_id: str | None = None
        if project_path:
            project_dir = Path(project_path).expanduser()
            if project_dir.name and str(project_dir) != str(Path.home()):
                if not topic_hint:
                    topic_hint = project_dir.name
                existing_projects = await self._graph.find_entities(
                    name=project_dir.name,
                    entity_type="Project",
                    group_id=group_id,
                    limit=1,
                )
                if existing_projects:
                    project_entity_id = existing_projects[0].id
                else:
                    project_entity_id = f"ent_{uuid.uuid4().hex[:12]}"
                    project_entity = Entity(
                        id=project_entity_id,
                        name=project_dir.name,
                        entity_type="Project",
                        summary=f"Software project at {project_path}",
                        attributes={"project_path": str(project_dir)},
                        group_id=group_id,
                    )
                    await self._graph.create_entity(project_entity)
                    await self._activation.record_access(
                        project_entity_id,
                        now,
                        group_id=group_id,
                    )

        layer1_entities: list[dict] = []
        layer1_facts: list[str] = []
        if self._cfg.identity_core_enabled and hasattr(self._graph, "get_identity_core_entities"):
            try:
                core_entities = await self._graph.get_identity_core_entities(group_id)
                for core_entity in core_entities:
                    entity_data = await self.entity_to_context_data(
                        core_entity.id,
                        core_entity.name,
                        core_entity.entity_type,
                        core_entity.summary or "",
                        group_id,
                        now,
                        detail_level="full",
                    )
                    layer1_entities.append(entity_data)
                    layer1_facts.extend(entity_data["facts"])
                    seen_ids.add(core_entity.id)
            except Exception:
                logger.debug("Identity core lookup failed (non-fatal)", exc_info=True)
        layer1_entities.sort(key=lambda item: item["activation"], reverse=True)
        layer1_text = self.render_tier("## Identity", layer1_entities, layer1_facts)

        layer2_entities: list[dict] = []
        layer2_facts: list[str] = []
        if topic_hint:
            results = await self._recall(query=topic_hint, group_id=group_id, limit=15)
            for result in results:
                if result.get("result_type") in {"episode", "cue_episode"}:
                    continue
                entity = result.get("entity")
                if not entity or entity["id"] in seen_ids:
                    continue
                hop = result.get("score_breakdown", {}).get("hop_distance")
                if hop is None or hop == 0:
                    detail = "full"
                elif hop == 1:
                    detail = "summary"
                else:
                    detail = "mention"
                entity_data = await self.entity_to_context_data(
                    entity["id"],
                    entity["name"],
                    entity["type"],
                    entity.get("summary") or "",
                    group_id,
                    now,
                    detail_level=detail,
                )
                layer2_entities.append(entity_data)
                layer2_facts.extend(entity_data["facts"])
                seen_ids.add(entity["id"])

        if project_entity_id:
            try:
                neighbors = await self._graph.get_neighbors(
                    project_entity_id,
                    hops=1,
                    group_id=group_id,
                )
                for neighbor_entity, _relationship in neighbors:
                    if neighbor_entity.id in seen_ids:
                        continue
                    entity_data = await self.entity_to_context_data(
                        neighbor_entity.id,
                        neighbor_entity.name,
                        neighbor_entity.entity_type,
                        neighbor_entity.summary or "",
                        group_id,
                        now,
                        detail_level="summary",
                    )
                    layer2_entities.append(entity_data)
                    layer2_facts.extend(entity_data["facts"])
                    seen_ids.add(neighbor_entity.id)
            except Exception:
                logger.debug("Project neighbor injection failed (non-fatal)", exc_info=True)

        if layer2_entities:
            layer2_entities.sort(key=lambda item: item["activation"], reverse=True)
            layer2_text = self.render_tier(
                f"## Project Context ({topic_hint})",
                layer2_entities,
                layer2_facts,
            )
        else:
            layer2_text = ""

        layer3_entities: list[dict] = []
        layer3_facts: list[str] = []
        top = await self._activation.get_top_activated(group_id=group_id, limit=20)
        for entity_id, state in top:
            if entity_id in seen_ids:
                continue
            entity = await self._graph.get_entity(entity_id, group_id)
            if not entity:
                continue
            activation = compute_activation(state.access_history, now, self._cfg)
            entity_data = await self.entity_to_context_data(
                entity.id,
                entity.name,
                entity.entity_type,
                entity.summary or "",
                group_id,
                now,
                detail_level="summary",
            )
            entity_data["activation"] = activation
            layer3_entities.append(entity_data)
            layer3_facts.extend(entity_data["facts"])
            seen_ids.add(entity_id)

        layer3_entities.sort(key=lambda item: item["activation"], reverse=True)
        layer3_text = self.render_tier("## Recent Activity", layer3_entities, layer3_facts)

        layer4_text = ""
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            try:
                from engram.models.prospective import IntentionMeta

                intention_entities = await self._list_intentions(group_id)
                intention_lines: list[str] = []
                for intention_entity in intention_entities:
                    attrs = intention_entity.attributes or {}
                    try:
                        meta = IntentionMeta(**attrs)
                    except Exception:
                        continue

                    intention_state: ActivationState | None = (
                        await self._activation.get_activation(intention_entity.id)
                    )
                    activation = 0.0
                    if intention_state:
                        activation = compute_activation(
                            intention_state.access_history,
                            now,
                            self._cfg,
                        )
                    if meta.activation_threshold > 0:
                        warmth_ratio = activation / meta.activation_threshold
                    else:
                        warmth_ratio = 0.0
                    levels = self._cfg.prospective_warmth_levels
                    if warmth_ratio < levels[0]:
                        continue

                    if warmth_ratio >= 1.0:
                        label = "HOT"
                    elif warmth_ratio >= levels[2]:
                        label = "warm"
                    elif warmth_ratio >= levels[1]:
                        label = "warming"
                    else:
                        label = "cool"

                    intention_lines.append(
                        f"- [{label}] {meta.trigger_text} → {meta.action_text} "
                        f"(fires: {meta.fire_count}/{meta.max_fires})"
                    )
                    seen_ids.add(intention_entity.id)

                if intention_lines:
                    layer4_text = "## Active Intentions\n\n" + "\n".join(intention_lines)
            except Exception:
                logger.debug("Intention tier in get_context failed (non-fatal)", exc_info=True)

        layer5_text = ""
        pinned_contexts: list[dict] = []
        if self._cfg.prospective_memory_enabled and self._cfg.prospective_graph_embedded:
            try:
                from engram.models.prospective import IntentionMeta

                pinned_entities = await self._list_intentions(group_id, enabled_only=True)
                pinned_lines: list[str] = []
                for pinned_entity in pinned_entities:
                    attrs = pinned_entity.attributes or {}
                    try:
                        pinned_meta = IntentionMeta(**attrs)
                    except Exception:
                        continue
                    if (
                        pinned_meta.trigger_type != "refresh_context"
                        or not pinned_meta.pinned_result
                    ):
                        continue
                    pinned_contexts.append(
                        {
                            "topic": pinned_meta.trigger_text,
                            "result": pinned_meta.pinned_result,
                            "last_refreshed": pinned_meta.last_refreshed,
                        }
                    )
                    pinned_lines.append(
                        f"### {pinned_meta.trigger_text}\n{pinned_meta.pinned_result}"
                    )
                if pinned_lines:
                    layer5_text = "## Pinned Contexts\n\n" + "\n\n".join(pinned_lines)
            except Exception:
                logger.debug("Pinned context tier in get_context failed (non-fatal)", exc_info=True)

        all_entities = layer1_entities + layer2_entities + layer3_entities
        all_facts = layer1_facts + layer2_facts + layer3_facts
        seen_facts: set[str] = set()
        unique_facts: list[str] = []
        for fact in all_facts:
            if fact in seen_facts:
                continue
            seen_facts.add(fact)
            unique_facts.append(fact)

        sections = [
            section for section in [layer1_text, layer2_text, layer3_text, layer4_text, layer5_text]
            if section
        ]
        context_text = (
            "\n\n".join(sections) if sections else "## Active Memory Context\n\nNo memories loaded."
        )

        token_estimate = self.estimate_tokens(context_text)
        if token_estimate > max_tokens:
            char_budget = max_tokens * 4
            context_text = context_text[:char_budget]
            token_estimate = max_tokens

        for entity_data in all_entities:
            await self._activation.record_access(entity_data["id"], now, group_id=group_id)
            await self._publish_access_event(
                entity_data["id"],
                entity_data["name"],
                entity_data["type"],
                group_id,
                "context",
            )

        if format == "briefing" and self._cfg.briefing_enabled and all_entities:
            briefing = self.template_briefing(context_text, group_id, topic_hint)
            result = {
                "context": briefing,
                "entity_count": len(all_entities),
                "fact_count": len(unique_facts),
                "token_estimate": self.estimate_tokens(briefing),
                "format": "briefing",
            }
            if pinned_contexts:
                result["pinned_contexts"] = pinned_contexts
            return result

        result = {
            "context": context_text,
            "entity_count": len(all_entities),
            "fact_count": len(unique_facts),
            "token_estimate": token_estimate,
            "format": "structured",
        }
        if pinned_contexts:
            result["pinned_contexts"] = pinned_contexts
        return result
