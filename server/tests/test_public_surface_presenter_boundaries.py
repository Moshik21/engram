from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PRESENTER_BOUNDARIES = {
    ("engram/api/knowledge.py", "observe"): {
        "memory_write_contract",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "auto_observe"): {
        "memory_write_contract",
        "present_api_memory_write",
        "present_api_observe_skip",
    },
    ("engram/api/knowledge.py", "observe_image"): {
        "memory_write_contract",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "observe_file"): {
        "memory_write_contract",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "remember"): {
        "memory_write_contract",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "recall"): {
        "build_api_recall_surface",
    },
    ("engram/mcp/server.py", "remember"): {
        "memory_write_contract",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "observe"): {
        "memory_write_contract",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "observe_image"): {
        "memory_write_contract",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "observe_file"): {
        "memory_write_contract",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "recall"): {
        "build_mcp_recall_surface",
    },
}

PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES = {
    ("engram/api/websocket.py", "dashboard_ws"): {
        "get_config",
        "get_manager",
        "get_notification_surface_service",
    },
    ("engram/api/websocket.py", "forward_events"): {"flatten_dashboard_event"},
    ("engram/api/knowledge.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/api/knowledge.py", "observe"): {
        "parse_conversation_date",
        "store_observation",
    },
    ("engram/api/knowledge.py", "auto_observe"): {
        "parse_conversation_date",
        "store_observation",
    },
    ("engram/api/knowledge.py", "observe_image"): {
        "build_observation_attachment",
        "store_observation",
    },
    ("engram/api/knowledge.py", "observe_file"): {
        "build_observation_attachment",
        "store_observation",
    },
    ("engram/api/knowledge.py", "replay_queue"): {
        "build_api_manager_offline_replay_surface",
    },
    ("engram/api/knowledge.py", "remember"): {
        "ingest_projecting_memory",
        "load_client_enabled_episode_adjudication_requests",
        "memory_write_contract",
        "parse_conversation_date",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "adjudicate"): {
        "build_api_adjudication_resolution_surface",
    },
    ("engram/api/knowledge.py", "forget"): {
        "build_api_forget_response_surface",
    },
    ("engram/api/knowledge.py", "post_feedback"): {
        "build_api_explicit_feedback_surface",
    },
    ("engram/api/knowledge.py", "search_facts"): {
        "build_api_fact_search_surface",
    },
    ("engram/api/knowledge.py", "get_context"): {
        "build_api_context_surface",
    },
    ("engram/api/knowledge.py", "bootstrap_project"): {
        "build_project_bootstrap_surface",
        "project_bootstrap_http_status",
    },
    ("engram/api/knowledge.py", "recall"): {
        "build_api_recall_surface",
    },
    ("engram/api/knowledge.py", "_execute_tool"): {
        "execute_chat_tool",
    },
    ("engram/api/knowledge.py", "_build_tool_events"): {
        "_emit_tool",
        "build_chat_tool_events",
    },
    ("engram/api/knowledge.py", "_retry_memory_grounded_response"): {
        "build_memory_grounding_retry_system_prompt",
    },
    ("engram/api/knowledge.py", "chat"): {
        "apply_chat_recall_feedback",
        "build_api_chat_rate_limit_surface",
        "build_chat_runtime_policy",
        "build_chat_messages",
        "build_chat_system_prompt_surface",
        "get_rate_limiter",
        "analyze_chat_memory_need",
        "build_chat_context_surface",
        "chat_conversation_not_found_payload",
        "gather_chat_epistemic_evidence",
        "hydrate_chat_context",
        "persist_chat_turn",
        "accumulate_chat_tool_result",
        "build_chat_tool_result_message",
        "record_chat_assistant_turn",
        "resolve_chat_conversation",
        "should_retry_chat_response",
    },
    ("engram/api/health.py", "health_check"): {
        "build_api_health_surface",
        "get_config",
        "get_graph_store",
        "get_mode",
    },
    ("engram/api/consolidation.py", "trigger_consolidation"): {
        "build_api_consolidation_trigger_surface",
        "run_api_consolidation_cycle",
    },
    ("engram/api/consolidation.py", "consolidation_status"): {
        "build_api_consolidation_status_surface",
    },
    ("engram/api/consolidation.py", "consolidation_history"): {
        "build_api_consolidation_history_surface",
    },
    ("engram/api/consolidation.py", "consolidation_cycle_detail"): {
        "build_api_consolidation_cycle_detail_surface",
    },
    ("engram/api/conversations.py", "list_conversations"): {
        "build_api_conversation_list_surface",
    },
    ("engram/api/conversations.py", "create_conversation"): {
        "build_api_conversation_create_surface",
    },
    ("engram/api/conversations.py", "get_messages"): {
        "build_api_conversation_messages_response_surface",
    },
    ("engram/api/conversations.py", "append_messages"): {
        "build_api_conversation_append_messages_response_surface",
    },
    ("engram/api/conversations.py", "update_conversation"): {
        "build_api_conversation_update_response_surface",
    },
    ("engram/api/conversations.py", "delete_conversation"): {
        "build_api_conversation_delete_response_surface",
    },
    ("engram/api/entities.py", "search_entities"): {
        "build_api_entity_search_surface",
    },
    ("engram/api/evaluation.py", "brain_loop_evaluation_report"): {
        "build_brain_loop_evaluation_surface",
        "get_recent_evaluation_context",
    },
    ("engram/api/evaluation.py", "create_recall_sample"): {
        "build_recall_evaluation_write_surface",
    },
    ("engram/api/evaluation.py", "create_session_sample"): {
        "build_session_continuity_evaluation_write_surface",
    },
    ("engram/api/knowledge.py", "route_knowledge_question"): {
        "_get_conv_top_entity_names",
        "build_question_route_surface",
    },
    ("engram/api/knowledge.py", "search_artifacts"): {
        "build_api_artifact_search_surface",
    },
    ("engram/api/knowledge.py", "get_runtime_state"): {
        "build_runtime_state_surface",
    },
    ("engram/api/knowledge.py", "get_notifications"): {
        "build_api_notifications_surface",
        "get_notification_surface_service",
    },
    ("engram/api/knowledge.py", "dismiss_notifications"): {
        "build_api_notification_dismiss_surface",
        "get_notification_surface_service",
    },
    ("engram/api/entities.py", "get_entity"): {
        "build_api_entity_detail_response_surface",
    },
    ("engram/api/entities.py", "get_entity_neighbors"): {
        "build_api_graph_neighborhood_surface",
    },
    ("engram/api/entities.py", "patch_entity"): {
        "build_api_entity_update_response_surface",
    },
    ("engram/api/entities.py", "delete_entity"): {
        "build_api_entity_delete_response_surface",
    },
    ("engram/api/admin.py", "load_benchmark"): {"build_api_benchmark_load_surface"},
    ("engram/api/graph.py", "get_atlas"): {"build_api_atlas_surface"},
    ("engram/api/graph.py", "get_atlas_history"): {
        "build_api_atlas_history_surface",
    },
    ("engram/api/graph.py", "get_region"): {"build_api_atlas_region_surface"},
    ("engram/api/graph.py", "get_neighborhood"): {"build_api_graph_neighborhood_surface"},
    ("engram/api/graph.py", "get_graph_at"): {"build_api_temporal_graph_surface"},
    ("engram/api/stats.py", "get_stats"): {"build_api_dashboard_stats_surface"},
    ("engram/api/episodes.py", "list_episodes"): {"build_api_episode_list_surface"},
    ("engram/api/lifecycle.py", "lifecycle_summary"): {
        "build_api_lifecycle_summary_surface"
    },
    ("engram/api/websocket.py", "activation_snapshot_loop"): {
        "build_api_activation_snapshot_surface",
        "build_dashboard_activation_snapshot_message",
    },
    ("engram/api/websocket.py", "receive_commands"): {
        "build_dashboard_pong_surface",
        "build_dashboard_resync_surface",
        "dismiss_dashboard_notification_command",
    },
    ("engram/api/activation.py", "get_activation_snapshot"): {
        "build_api_activation_snapshot_surface",
    },
    ("engram/api/activation.py", "get_activation_curve"): {
        "build_api_activation_curve_surface",
    },
    ("engram/mcp/server.py", "_serialize_notifications"): {
        "build_mcp_notifications_surface_from_state",
    },
    ("engram/mcp/server.py", "_should_recall"): {"should_recall_for_tool"},
    ("engram/mcp/server.py", "_auto_recall_lite"): {
        "build_lite_auto_recall_surface",
    },
    ("engram/mcp/server.py", "_auto_recall_full"): {
        "build_full_auto_recall_surface",
    },
    ("engram/mcp/server.py", "_session_prime"): {
        "build_session_prime_surface",
    },
    ("engram/mcp/server.py", "_recall_middleware"): {
        "apply_mcp_recall_enrichment",
        "drain_mcp_triggered_intentions",
        "plan_mcp_recall_middleware",
        "store_mcp_auto_observe_turn",
    },
    ("engram/mcp/server.py", "recall"): {
        "build_mcp_recall_surface",
    },
    ("engram/mcp/server.py", "_get_conv_context"): {"manager_conversation_context"},
    ("engram/mcp/server.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/mcp/server.py", "_ingest_live_turn"): {
        "_get_conv_context",
        "ingest_manager_conversation_turn",
    },
    ("engram/mcp/server.py", "get_lifecycle_summary"): {
        "build_mcp_lifecycle_summary_surface",
    },
    ("engram/mcp/server.py", "get_consolidation_status"): {
        "build_mcp_consolidation_status_surface",
    },
    ("engram/mcp/server.py", "get_evaluation_report"): {
        "build_mcp_evaluation_report_surface",
    },
    ("engram/mcp/server.py", "record_recall_evaluation"): {
        "build_recall_evaluation_write_surface",
    },
    ("engram/mcp/server.py", "record_session_continuity_evaluation"): {
        "build_session_continuity_evaluation_write_surface",
    },
    ("engram/mcp/server.py", "remember"): {
        "build_observation_attachment",
        "ingest_projecting_memory",
        "load_client_enabled_episode_adjudication_requests",
        "memory_write_contract",
        "parse_conversation_date",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "observe"): {
        "parse_conversation_date",
        "store_observation",
    },
    ("engram/mcp/server.py", "observe_image"): {
        "build_observation_attachment",
        "store_observation",
    },
    ("engram/mcp/server.py", "observe_file"): {
        "build_observation_attachment",
        "store_observation",
    },
    ("engram/mcp/server.py", "adjudicate_evidence"): {
        "build_mcp_adjudication_resolution_surface",
    },
    ("engram/mcp/server.py", "forget"): {
        "build_mcp_forget_surface",
    },
    ("engram/mcp/server.py", "feedback"): {
        "build_mcp_explicit_feedback_surface",
    },
    ("engram/mcp/server.py", "search_entities"): {
        "build_mcp_entity_search_surface",
    },
    ("engram/mcp/server.py", "search_facts"): {
        "build_mcp_fact_search_surface",
    },
    ("engram/mcp/server.py", "get_context"): {
        "build_mcp_context_surface",
    },
    ("engram/mcp/server.py", "bootstrap_project"): {
        "build_project_bootstrap_surface",
    },
    ("engram/mcp/server.py", "get_runtime_state"): {
        "build_runtime_state_surface",
    },
    ("engram/mcp/server.py", "get_graph_state"): {
        "build_mcp_graph_state_surface",
    },
    ("engram/mcp/server.py", "mark_identity_core"): {"build_mcp_identity_core_surface"},
    ("engram/mcp/server.py", "trigger_consolidation"): {
        "build_mcp_consolidation_trigger_surface",
        "resolve_mcp_consolidation_trigger_store",
    },
    ("engram/mcp/server.py", "graph_stats_resource"): {
        "build_mcp_graph_stats_resource_surface",
    },
    ("engram/mcp/server.py", "entity_profile_resource"): {
        "build_mcp_entity_profile_resource_surface",
    },
    ("engram/mcp/server.py", "entity_neighbors_resource"): {
        "build_mcp_entity_neighbors_resource_surface",
    },
    ("engram/api/knowledge.py", "create_intention"): {
        "build_api_create_intention_response_surface",
    },
    ("engram/api/knowledge.py", "list_intentions"): {"build_intention_list_surface"},
    ("engram/api/knowledge.py", "dismiss_intention"): {
        "build_api_dismiss_intention_response_surface",
    },
    ("engram/mcp/server.py", "list_intentions"): {"build_intention_list_surface"},
    ("engram/mcp/server.py", "intend"): {
        "build_mcp_create_intention_response_surface",
    },
    ("engram/mcp/server.py", "dismiss_intention"): {
        "build_mcp_dismiss_intention_response_surface",
    },
    ("engram/mcp/server.py", "route_question"): {
        "_get_conv_top_entity_names",
        "build_question_route_surface",
    },
    ("engram/mcp/server.py", "search_artifacts"): {
        "build_mcp_artifact_search_surface",
    },
}

PUBLIC_ROUTE_FORBIDDEN_IDENTIFIERS = {
    ("engram/api/knowledge.py", "forget"): {
        "api_forget_missing_target_payload",
        "build_api_forget_surface",
    },
    ("engram/api/entities.py", "get_entity"): {
        "build_api_entity_detail_surface",
        "entity_not_found_payload",
    },
    ("engram/api/entities.py", "patch_entity"): {
        "build_api_entity_update_surface",
        "entity_not_found_payload",
    },
    ("engram/api/entities.py", "delete_entity"): {
        "build_api_entity_delete_surface",
        "entity_not_found_payload",
    },
    ("engram/api/conversations.py", "get_messages"): {
        "build_api_conversation_messages_surface",
        "conversation_not_found_payload",
    },
    ("engram/api/conversations.py", "append_messages"): {
        "build_api_conversation_append_messages_surface",
        "conversation_not_found_payload",
    },
    ("engram/api/conversations.py", "update_conversation"): {
        "build_api_conversation_update_surface",
        "conversation_not_found_payload",
    },
    ("engram/api/conversations.py", "delete_conversation"): {
        "build_api_conversation_delete_surface",
        "conversation_not_found_payload",
    },
    ("engram/api/entities.py", "get_entity_neighbors"): {
        "get_neighborhood",
    },
    ("engram/api/admin.py", "load_benchmark"): {
        "load_benchmark_corpus",
    },
    ("engram/api/stats.py", "get_stats"): {
        "get_dashboard_stats",
    },
    ("engram/api/episodes.py", "list_episodes"): {
        "list_episode_summaries",
    },
    ("engram/api/activation.py", "get_activation_snapshot"): {
        "get_activation_snapshot",
    },
    ("engram/api/websocket.py", "activation_snapshot_loop"): {
        "get_activation_snapshot",
    },
    ("engram/api/websocket.py", "receive_commands"): {
        "dismiss_notifications",
        "get_events_since",
    },
    ("engram/api/health.py", "health_check"): {"get_stats"},
    ("engram/api/activation.py", "get_activation_curve"): {
        "get_activation_curve",
        "HTTPException",
    },
    ("engram/api/lifecycle.py", "lifecycle_summary"): {
        "get_lifecycle_summary",
    },
    ("engram/api/knowledge.py", "chat"): {
        "gather_epistemic_evidence",
        "get_context",
        "get_chat_runtime_policy",
    },
    ("engram/api/knowledge.py", "replay_queue"): {
        "build_api_offline_replay_surface",
        "store_episode",
    },
    ("engram/api/knowledge.py", "remember"): {
        "edge_adjudication_client_enabled",
        "load_episode_adjudication_requests",
    },
    ("engram/mcp/server.py", "remember"): {
        "edge_adjudication_client_enabled",
        "load_episode_adjudication_requests",
    },
    ("engram/api/knowledge.py", "create_intention"): {
        "api_intention_validation_error_payload",
        "build_api_create_intention_surface",
    },
    ("engram/api/knowledge.py", "dismiss_intention"): {
        "api_intention_not_found_payload",
        "build_api_dismiss_intention_surface",
    },
    ("engram/mcp/server.py", "intend"): {
        "build_mcp_create_intention_surface",
        "mcp_intention_error_payload",
    },
    ("engram/mcp/server.py", "dismiss_intention"): {
        "build_mcp_dismiss_intention_surface",
        "mcp_intention_error_payload",
    },
    ("engram/mcp/server.py", "recall"): {
        "get_recall_item_access_count",
        "resolve_entity_name",
    },
    ("engram/mcp/server.py", "trigger_consolidation"): {
        "get_consolidation_shared_db",
    },
    ("engram/mcp/server.py", "_auto_recall_lite"): {
        "recall_lite",
        "recall_medium",
    },
    ("engram/mcp/server.py", "_auto_recall_full"): {
        "analyze_memory_need",
        "assemble_memory_packets",
        "recall",
        "record_manager_memory_need_analysis",
        "resolve_manager_recall_need_thresholds",
    },
    ("engram/mcp/server.py", "_session_prime"): {
        "get_context",
        "plan_session_prime",
    },
    ("engram/mcp/server.py", "_recall_middleware"): {
        "drain_triggered_intention_views",
        "store_episode",
    },
    ("engram/mcp/server.py", "_serialize_notifications"): {
        "get_notification_surface_service_from_state",
        "mcp_notifications",
    },
    ("engram/api/graph.py", "get_atlas"): {
        "get_snapshot",
        "represented_entity_count",
        "represented_edge_count",
        "displayed_node_count",
        "displayed_edge_count",
        "total_entities",
        "total_relationships",
        "total_regions",
    },
    ("engram/api/graph.py", "get_atlas_history"): {
        "list_snapshots",
        "represented_entity_count",
        "represented_edge_count",
        "displayed_node_count",
        "displayed_edge_count",
        "total_entities",
        "total_relationships",
        "total_regions",
    },
    ("engram/api/graph.py", "get_region"): {
        "get_region_payload",
        "atlas_region_not_found_payload",
        "atlas_region_or_snapshot_not_found_payload",
    },
}

PUBLIC_SURFACES_WITHOUT_APP_STATE_READS = tuple(
    sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "engram/api").glob("*.py")
        if path.name not in {"__init__.py", "deps.py"}
    )
)


def _function_names_used(relative_path: str, function_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == function_name:
            return {name.id for name in ast.walk(node) if isinstance(name, ast.Name)}
    raise AssertionError(f"Function not found: {relative_path}:{function_name}")


def _function_identifiers_used(relative_path: str, function_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == function_name:
            return {
                *{name.id for name in ast.walk(node) if isinstance(name, ast.Name)},
                *{attr.attr for attr in ast.walk(node) if isinstance(attr, ast.Attribute)},
            }
    raise AssertionError(f"Function not found: {relative_path}:{function_name}")


def _private_attrs_used(relative_path: str, function_name: str, object_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        if node.name != function_name:
            continue
        return {
            attr.attr
            for attr in ast.walk(node)
            if isinstance(attr, ast.Attribute)
            and isinstance(attr.value, ast.Name)
            and attr.value.id == object_name
            and attr.attr.startswith("_")
        }
    raise AssertionError(f"Function not found: {relative_path}:{function_name}")


def _manager_private_attrs_used(relative_path: str, function_name: str) -> set[str]:
    return _private_attrs_used(relative_path, function_name, "manager")


@pytest.mark.parametrize(
    ("surface", "expected_names"),
    PRESENTER_BOUNDARIES.items(),
)
def test_public_memory_surfaces_use_shared_presenters(
    surface: tuple[str, str],
    expected_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_names_used(relative_path, function_name)
    missing = expected_names - names_used
    assert missing == set()


@pytest.mark.parametrize(
    ("surface", "expected_names"),
    PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES.items(),
)
def test_public_mutation_surfaces_use_manager_facades(
    surface: tuple[str, str],
    expected_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_identifiers_used(relative_path, function_name)
    missing = expected_names - names_used
    assert missing == set()
    assert _manager_private_attrs_used(relative_path, function_name) == set()
    assert _private_attrs_used(relative_path, function_name, "engine") == set()


@pytest.mark.parametrize(
    ("surface", "forbidden_names"),
    PUBLIC_ROUTE_FORBIDDEN_IDENTIFIERS.items(),
)
def test_public_routes_do_not_reassemble_delegated_payloads(
    surface: tuple[str, str],
    forbidden_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_identifiers_used(relative_path, function_name)
    assert names_used & forbidden_names == set()


@pytest.mark.parametrize("relative_path", PUBLIC_SURFACES_WITHOUT_APP_STATE_READS)
def test_public_surface_routes_do_not_read_app_state_directly(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text()
    assert "_app_state" not in source
