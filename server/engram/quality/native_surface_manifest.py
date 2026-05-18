"""Native-first public surface coverage manifest.

This manifest is intentionally static. Runtime parity tests prove behavior;
this file makes the intended PyO3-native coverage boundary explicit so new
REST/MCP surfaces must be classified instead of hiding in the route table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SurfaceKind = Literal[
    "rest",
    "websocket",
    "mcp_transport",
    "mcp_tool",
    "mcp_resource",
    "mcp_prompt",
    "dashboard",
    "operator",
]
CoverageKind = Literal[
    "native_runtime_parity",
    "native_fixture_parity",
    "native_operator_gate",
    "native_operator_smoke",
    "static_not_data_bound",
]


@dataclass(frozen=True, slots=True)
class NativeSurface:
    """Public surface plus the native-path evidence expected to cover it."""

    kind: SurfaceKind
    identifier: str
    coverage: CoverageKind
    evidence: str
    note: str = ""


def _rest(method: str, path: str, evidence: str, note: str = "") -> NativeSurface:
    return NativeSurface(
        kind="rest",
        identifier=f"{method} {path}",
        coverage="native_runtime_parity",
        evidence=evidence,
        note=note,
    )


def _mcp_tool(name: str, evidence: str, note: str = "") -> NativeSurface:
    return NativeSurface(
        kind="mcp_tool",
        identifier=name,
        coverage="native_runtime_parity",
        evidence=evidence,
        note=note,
    )


NATIVE_SURFACE_MANIFEST: tuple[NativeSurface, ...] = (
    # REST and dashboard transport surfaces.
    _rest("GET", "/health", "_assert_native_rest_health_surface"),
    _rest("GET", "/api/lifecycle/summary", "_assert_native_rest_surfaces"),
    _rest("POST", "/api/admin/load-benchmark", "_assert_native_rest_admin_benchmark_surface"),
    _rest(
        "POST",
        "/api/consolidation/trigger",
        "_assert_native_rest_consolidation_trigger_surface",
    ),
    _rest("GET", "/api/consolidation/status", "_assert_native_rest_consolidation_surfaces"),
    _rest("GET", "/api/consolidation/history", "_assert_native_rest_consolidation_surfaces"),
    _rest(
        "GET",
        "/api/consolidation/cycle/{cycle_id}",
        "_assert_native_rest_consolidation_surfaces",
    ),
    _rest("POST", "/api/evaluation/recall-samples", "_record_native_rest_evaluation_labels"),
    _rest("POST", "/api/evaluation/session-samples", "_record_native_rest_evaluation_labels"),
    _rest("GET", "/api/evaluation/brain-loop/report", "_assert_native_rest_surfaces"),
    _rest("GET", "/api/stats", "_assert_native_rest_dashboard_read_surfaces"),
    _rest("GET", "/api/activation/snapshot", "_assert_native_rest_dashboard_read_surfaces"),
    _rest(
        "GET",
        "/api/activation/{entity_id}/curve",
        "_assert_native_rest_dashboard_read_surfaces",
    ),
    _rest("GET", "/api/graph/atlas", "_assert_native_rest_atlas_surfaces"),
    _rest("GET", "/api/graph/atlas/history", "_assert_native_rest_atlas_surfaces"),
    _rest("GET", "/api/graph/regions/{region_id}", "_assert_native_rest_atlas_surfaces"),
    _rest("GET", "/api/graph/neighborhood", "_assert_native_rest_dashboard_read_surfaces"),
    _rest("GET", "/api/graph/at", "_assert_native_rest_dashboard_read_surfaces"),
    _rest("GET", "/api/entities/search", "_assert_native_rest_entity_fact_lookup_surface"),
    _rest("GET", "/api/entities/{entity_id}", "_assert_native_rest_entity_fact_lookup_surface"),
    _rest(
        "GET",
        "/api/entities/{entity_id}/neighbors",
        "_assert_native_rest_entity_fact_lookup_surface",
    ),
    _rest("PATCH", "/api/entities/{entity_id}", "_assert_native_rest_entity_mutation_surface"),
    _rest("DELETE", "/api/entities/{entity_id}", "_assert_native_rest_entity_mutation_surface"),
    _rest("GET", "/api/episodes", "_assert_native_rest_dashboard_read_surfaces"),
    _rest("GET", "/api/knowledge/notifications", "_assert_native_rest_notification_surfaces"),
    _rest(
        "POST",
        "/api/knowledge/notifications/dismiss",
        "_assert_native_rest_notification_surfaces",
    ),
    _rest("POST", "/api/knowledge/observe", "_assert_native_rest_observe_surface"),
    _rest("POST", "/api/knowledge/auto-observe", "_assert_native_rest_auto_observe_surface"),
    _rest("POST", "/api/knowledge/observe-image", "_assert_native_rest_attachment_surfaces"),
    _rest("POST", "/api/knowledge/observe-file", "_assert_native_rest_attachment_surfaces"),
    _rest(
        "POST",
        "/api/knowledge/replay-queue",
        "test_native_helix_rest_surfaces_handle_bounded_remember_recall_load",
    ),
    _rest(
        "POST",
        "/api/knowledge/remember",
        "test_native_helix_rest_surfaces_handle_bounded_remember_recall_load",
    ),
    _rest("POST", "/api/knowledge/adjudicate", "_assert_native_rest_adjudication_surface"),
    _rest("GET", "/api/knowledge/recall", "_assert_native_rest_surfaces"),
    _rest("GET", "/api/knowledge/facts", "_assert_native_rest_entity_fact_lookup_surface"),
    _rest("GET", "/api/knowledge/context", "_assert_native_rest_context_surface"),
    _rest("POST", "/api/knowledge/forget", "_assert_native_rest_forget_surface"),
    _rest("POST", "/api/knowledge/feedback", "_assert_native_rest_feedback_surface"),
    _rest(
        "POST",
        "/api/knowledge/bootstrap",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest("POST", "/api/knowledge/route", "_assert_native_rest_route_surface"),
    _rest(
        "GET",
        "/api/knowledge/artifacts/search",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest("GET", "/api/knowledge/runtime", "_assert_native_rest_runtime_surface"),
    _rest("POST", "/api/knowledge/intentions", "_assert_native_rest_intention_surfaces"),
    _rest("GET", "/api/knowledge/intentions", "_assert_native_rest_intention_surfaces"),
    _rest(
        "DELETE",
        "/api/knowledge/intentions/{intention_id}",
        "_assert_native_rest_intention_surfaces",
    ),
    _rest(
        "POST",
        "/api/knowledge/chat",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "GET",
        "/api/conversations/",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "POST",
        "/api/conversations/",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "GET",
        "/api/conversations/{conversation_id}/messages",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "POST",
        "/api/conversations/{conversation_id}/messages",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "PATCH",
        "/api/conversations/{conversation_id}",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _rest(
        "DELETE",
        "/api/conversations/{conversation_id}",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    NativeSurface(
        kind="websocket",
        identifier="/ws/dashboard",
        coverage="native_runtime_parity",
        evidence="test_native_helix_dashboard_websocket_uses_native_group",
        note="Dashboard WebSocket transport is route-table distinct from REST.",
    ),
    NativeSurface(
        kind="mcp_transport",
        identifier="/mcp",
        coverage="native_runtime_parity",
        evidence="test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
        note="Streamable HTTP transport hosts the same MCP runtime tools.",
    ),
    # MCP tools.
    _mcp_tool("remember", "_assert_native_mcp_write_surfaces"),
    _mcp_tool("feedback", "_assert_native_mcp_feedback_surface"),
    _mcp_tool("adjudicate_evidence", "_assert_native_mcp_adjudication_surface"),
    _mcp_tool("observe", "_assert_native_mcp_write_surfaces"),
    _mcp_tool("observe_image", "_assert_native_mcp_write_surfaces"),
    _mcp_tool("observe_file", "_assert_native_mcp_write_surfaces"),
    _mcp_tool("recall", "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces"),
    _mcp_tool("search_entities", "_assert_native_mcp_entity_fact_lookup_surface"),
    _mcp_tool("search_facts", "_assert_native_mcp_entity_fact_lookup_surface"),
    _mcp_tool("forget", "_assert_native_mcp_forget_surface"),
    _mcp_tool("get_context", "_assert_native_mcp_context_surface"),
    _mcp_tool("bootstrap_project", "_assert_native_mcp_project_bootstrap_surface"),
    _mcp_tool("route_question", "_assert_native_mcp_route_surface"),
    _mcp_tool("search_artifacts", "_assert_native_mcp_project_artifact_surface"),
    _mcp_tool("get_runtime_state", "_assert_native_mcp_runtime_surface"),
    _mcp_tool("claim_authority", "_assert_native_mcp_memory_authority_surface"),
    _mcp_tool("get_graph_state", "_assert_native_mcp_entity_fact_lookup_surface"),
    _mcp_tool(
        "get_lifecycle_summary",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _mcp_tool("record_recall_evaluation", "_assert_native_mcp_evaluation_write_surface"),
    _mcp_tool(
        "record_session_continuity_evaluation",
        "_assert_native_mcp_evaluation_write_surface",
    ),
    _mcp_tool(
        "get_evaluation_report",
        "test_native_helix_populated_brain_reaches_rest_and_mcp_surfaces",
    ),
    _mcp_tool("mark_identity_core", "_assert_native_mcp_identity_core_surface"),
    _mcp_tool("trigger_consolidation", "_assert_native_mcp_consolidation_control_surface"),
    _mcp_tool("get_consolidation_status", "_assert_native_mcp_consolidation_control_surface"),
    _mcp_tool("intend", "_assert_native_mcp_intention_surfaces"),
    _mcp_tool("dismiss_intention", "_assert_native_mcp_intention_surfaces"),
    _mcp_tool("list_intentions", "_assert_native_mcp_intention_surfaces"),
    # MCP resources and prompts.
    NativeSurface(
        kind="mcp_resource",
        identifier="engram://graph/stats",
        coverage="native_runtime_parity",
        evidence="_assert_native_mcp_graph_stats_resource",
    ),
    NativeSurface(
        kind="mcp_resource",
        identifier="engram://entity/{entity_id}",
        coverage="native_runtime_parity",
        evidence="_assert_native_mcp_entity_fact_lookup_surface",
    ),
    NativeSurface(
        kind="mcp_resource",
        identifier="engram://entity/{entity_id}/neighbors",
        coverage="native_runtime_parity",
        evidence="_assert_native_mcp_entity_fact_lookup_surface",
    ),
    NativeSurface(
        kind="mcp_prompt",
        identifier="engram_system",
        coverage="static_not_data_bound",
        evidence="server/engram/mcp/prompts.py",
    ),
    NativeSurface(
        kind="mcp_prompt",
        identifier="engram_context_loader",
        coverage="static_not_data_bound",
        evidence="server/engram/mcp/prompts.py",
    ),
    # Dashboard and operator native path gates.
    NativeSurface(
        kind="dashboard",
        identifier="nativeDashboardSmoke.test.tsx",
        coverage="native_fixture_parity",
        evidence="dashboard/src/test/nativeDashboardSmoke.test.tsx",
        note="Default no-bind fixture; live REST smoke remains opt-in.",
    ),
    NativeSurface(
        kind="operator",
        identifier="engram evaluate --smoke --mode helix",
        coverage="native_operator_smoke",
        evidence="server/engram/evaluation/smoke.py",
    ),
    NativeSurface(
        kind="operator",
        identifier="engram evaluate --mode helix --require-evaluation-signals",
        coverage="native_operator_gate",
        evidence="server/engram/evaluation/cli.py",
        note="Hard gate for measured evaluation signals on live or saved native reports.",
    ),
    NativeSurface(
        kind="operator",
        identifier="engram doctor --mode helix",
        coverage="native_operator_smoke",
        evidence="server/engram/doctor.py",
        note=(
            "Native diagnostic smoke reports coverage gaps and six-signal "
            "evaluation readiness."
        ),
    ),
    NativeSurface(
        kind="operator",
        identifier="make up-native / make mcp-native",
        coverage="native_operator_smoke",
        evidence="Makefile",
    ),
)


def surfaces_by_kind(kind: SurfaceKind) -> tuple[NativeSurface, ...]:
    return tuple(surface for surface in NATIVE_SURFACE_MANIFEST if surface.kind == kind)


def identifiers_by_kind(kind: SurfaceKind) -> set[str]:
    return {surface.identifier for surface in surfaces_by_kind(kind)}
