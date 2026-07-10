"""Epistemic question routing service."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from engram.config import ActivationConfig
from engram.retrieval.epistemic import (
    apply_answer_contract_to_evidence_plan,
    build_evidence_plan,
    resolve_answer_contract,
)
from engram.retrieval.epistemic import (
    route_question as build_question_frame,
)


async def build_question_route_surface(
    manager: Any,
    *,
    group_id: str,
    question: str,
    project_path: str | None,
    history: list[Any] | None,
    session_entity_names: list[str] | None,
    surface: str,
) -> dict:
    """Build the deterministic question-route payload for REST or MCP."""
    return await manager.route_question(
        question,
        group_id=group_id,
        project_path=project_path,
        recent_turns=recent_route_turn_contents(history, limit=6),
        session_entity_names=session_entity_names or [],
        surface=surface,
    )


async def build_mcp_question_route_tool_surface(
    manager: Any,
    *,
    group_id: str,
    question: str,
    project_path: str | None,
    history: list[Any] | None,
    session_entity_names: list[str] | None,
    recall_middleware: Callable[..., Awaitable[None]],
) -> dict:
    """Build the MCP question-route tool payload and run read-tool middleware."""
    result = await build_question_route_surface(
        manager,
        group_id=group_id,
        question=question,
        project_path=project_path,
        history=history,
        session_entity_names=session_entity_names,
        surface="mcp",
    )
    await recall_middleware(question, result, tool_name="route_question", auto_observe=True)
    return result


def recent_route_turn_contents(history: list[Any] | None, *, limit: int) -> list[str]:
    """Return recent non-empty route history strings from strings or chat messages."""
    values: list[str] = []
    for item in history or []:
        content = getattr(item, "content", item)
        if isinstance(content, str) and content.strip():
            values.append(content)
    return values[-limit:]


class EpistemicRouteService:
    """Build question frames, evidence plans, and answer contracts."""

    def __init__(
        self,
        *,
        cfg: ActivationConfig,
        conv_context: object | None,
        get_graph_probe: Callable[[], object | None],
        get_recall_need_thresholds: Callable[[str], object],
        record_route: Callable[[str, str, str, list[str]], None],
    ) -> None:
        self._cfg = cfg
        self._conv_context = conv_context
        self._get_graph_probe = get_graph_probe
        self._get_recall_need_thresholds = get_recall_need_thresholds
        self._record_route = record_route

    async def build_route(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need: Any = None,
    ):
        """Create the question frame and evidence plan for a turn."""
        if memory_need is None:
            from engram.retrieval.need import analyze_memory_need

            memory_need = await analyze_memory_need(
                question,
                recent_turns=recent_turns or [],
                session_entity_names=session_entity_names or [],
                mode="chat" if surface == "rest" else "auto_recall",
                graph_probe=(
                    self._get_graph_probe() if self._cfg.recall_need_graph_probe_enabled else None
                ),
                group_id=group_id,
                conv_context=self._conv_context,
                cfg=self._cfg,
                thresholds=self._get_recall_need_thresholds(group_id),
            )

        surface_capabilities = self.surface_capabilities(surface, project_path)
        frame = build_question_frame(
            question,
            memory_need=memory_need,
            recent_turns=recent_turns,
            project_path=project_path,
            surface_capabilities=surface_capabilities,
        )
        plan = build_evidence_plan(
            frame,
            surface_capabilities=surface_capabilities,
            cfg=self._cfg,
        )
        answer_contract = resolve_answer_contract(
            question,
            frame=frame,
            plan=plan,
            claims=[],
        )
        plan = apply_answer_contract_to_evidence_plan(
            question,
            frame=frame,
            plan=plan,
            answer_contract=answer_contract,
            memory_need=memory_need,
        )
        self._record_route(
            group_id,
            frame.mode,
            answer_contract.operator,
            answer_contract.relevant_scopes,
        )
        return frame, plan, memory_need, answer_contract

    async def route_question(
        self,
        question: str,
        *,
        group_id: str = "default",
        project_path: str | None = None,
        recent_turns: list[str] | None = None,
        session_entity_names: list[str] | None = None,
        surface: str = "rest",
        memory_need: Any = None,
    ) -> dict:
        """Return a routed question frame and evidence plan."""
        frame, plan, routed_need, answer_contract = await self.build_route(
            question,
            group_id=group_id,
            project_path=project_path,
            recent_turns=recent_turns,
            session_entity_names=session_entity_names,
            surface=surface,
            memory_need=memory_need,
        )
        payload = {
            "questionFrame": frame.to_dict(),
            "evidencePlan": plan.to_dict(),
            "answerContract": answer_contract.to_dict(),
            "recommendedNextSources": plan.recommended_next_sources,
        }
        if routed_need is not None:
            payload["memoryNeed"] = routed_need.to_payload(
                source="epistemic_route",
                mode=surface,
                turn_preview=question[:160],
            )
        return payload

    @staticmethod
    def surface_capabilities(surface: str, project_path: str | None) -> dict[str, bool]:
        return {
            "workspace_available": surface == "mcp" and bool(project_path),
            "native_workspace_search": surface == "mcp" and bool(project_path),
            "artifact_bootstrap": bool(project_path),
        }
