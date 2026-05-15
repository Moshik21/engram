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
        "present_api_recall_items",
    },
    ("engram/api/knowledge.py", "_execute_tool"): {
        "present_chat_recall_items",
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
        "present_mcp_recall_items",
    },
}

PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES = {
    ("engram/api/websocket.py", "dashboard_ws"): {
        "get_config",
        "get_manager",
        "get_notification_surface_service",
    },
    ("engram/api/knowledge.py", "_get_graph_probe"): {"get_recall_need_graph_probe"},
    ("engram/api/knowledge.py", "_get_conv_context"): {"manager_conversation_context"},
    ("engram/api/knowledge.py", "_get_conv_embed_fn"): {"manager_conversation_embed_fn"},
    ("engram/api/knowledge.py", "_get_conv_turn_count"): {
        "manager_conversation_turn_count",
    },
    ("engram/api/knowledge.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/api/knowledge.py", "_ingest_conversation_turn"): {
        "ingest_manager_conversation_turn",
    },
    ("engram/api/knowledge.py", "_hydrate_chat_context"): {
        "_get_conv_context",
        "_get_conv_turn_count",
        "_ingest_conversation_turn",
    },
    ("engram/api/knowledge.py", "_record_chat_assistant_turn"): {
        "_get_conv_context",
        "_ingest_conversation_turn",
    },
    ("engram/api/knowledge.py", "_analyze_chat_memory_need"): {
        "get_memory_need_config",
        "recall_need_graph_probe_enabled",
    },
    ("engram/api/knowledge.py", "remember"): {
        "edge_adjudication_client_enabled",
        "ingest_episode",
        "load_episode_adjudication_requests",
        "memory_write_contract",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "recall"): {
        "get_explicit_recall_packet_policy",
        "get_memory_need_config",
        "present_api_recall_items",
    },
    ("engram/api/knowledge.py", "_execute_tool"): {
        "get_chat_tool_recall_policy",
        "get_memory_need_config",
        "present_chat_recall_items",
    },
    ("engram/api/knowledge.py", "_build_tool_events"): {
        "_emit_tool",
        "build_chat_tool_events",
    },
    ("engram/api/knowledge.py", "_apply_chat_recall_feedback"): {
        "apply_memory_interaction",
        "recall_usage_feedback_enabled",
    },
    ("engram/api/knowledge.py", "_should_retry_chat_response"): {
        "recall_need_post_response_safety_net_enabled",
    },
    ("engram/api/knowledge.py", "chat"): {
        "get_chat_runtime_policy",
        "get_context",
        "get_rate_limiter",
        "_hydrate_chat_context",
        "persist_chat_turn",
        "raw_recall_from_chat_item",
        "resolve_chat_conversation",
    },
    ("engram/api/health.py", "health_check"): {
        "get_config",
        "get_graph_store",
        "get_mode",
    },
    ("engram/api/consolidation.py", "consolidation_status"): {
        "get_latest_cycle",
        "serialize_cycle_summary",
    },
    ("engram/api/consolidation.py", "consolidation_history"): {
        "get_recent_cycles",
        "serialize_cycle_summary",
    },
    ("engram/api/consolidation.py", "consolidation_cycle_detail"): {
        "audit_store_available",
        "get_cycle_detail",
        "serialize_cycle_detail",
    },
    ("engram/api/conversations.py", "list_conversations"): {
        "list_group_conversations",
    },
    ("engram/api/conversations.py", "create_conversation"): {
        "create_group_conversation",
    },
    ("engram/api/conversations.py", "get_messages"): {
        "get_group_conversation_messages",
    },
    ("engram/api/conversations.py", "append_messages"): {
        "append_group_conversation_messages",
    },
    ("engram/api/conversations.py", "update_conversation"): {
        "update_group_conversation_title",
    },
    ("engram/api/conversations.py", "delete_conversation"): {
        "delete_group_conversation",
    },
    ("engram/api/evaluation.py", "brain_loop_evaluation_report"): {
        "build_brain_loop_evaluation_surface",
        "get_recent_evaluation_context",
    },
    ("engram/api/evaluation.py", "create_recall_sample"): {
        "persist_recall_eval_sample",
        "present_recall_sample_write",
    },
    ("engram/api/evaluation.py", "create_session_sample"): {
        "persist_session_continuity_sample",
        "present_session_sample_write",
    },
    ("engram/api/knowledge.py", "route_knowledge_question"): {
        "_get_conv_top_entity_names",
        "route_question",
    },
    ("engram/api/knowledge.py", "get_notifications"): {"get_notification_surface_service"},
    ("engram/api/knowledge.py", "dismiss_notifications"): {
        "get_notification_surface_service",
    },
    ("engram/api/entities.py", "get_entity"): {"get_entity_detail"},
    ("engram/api/entities.py", "patch_entity"): {"update_entity_profile"},
    ("engram/api/entities.py", "delete_entity"): {"delete_entity_by_id"},
    ("engram/api/admin.py", "load_benchmark"): {"load_benchmark_corpus"},
    ("engram/api/graph.py", "get_neighborhood"): {"get_graph_neighborhood"},
    ("engram/api/graph.py", "get_graph_at"): {"get_temporal_graph"},
    ("engram/api/stats.py", "get_stats"): {"get_dashboard_stats"},
    ("engram/api/episodes.py", "list_episodes"): {"list_episode_summaries"},
    ("engram/api/lifecycle.py", "lifecycle_summary"): {"get_lifecycle_summary"},
    ("engram/api/websocket.py", "activation_snapshot_loop"): {"get_activation_snapshot"},
    ("engram/api/websocket.py", "receive_commands"): {"dismiss_notifications"},
    ("engram/api/activation.py", "get_activation_snapshot"): {"get_activation_snapshot"},
    ("engram/api/activation.py", "get_activation_curve"): {"get_activation_curve"},
    ("engram/mcp/server.py", "_serialize_intentions"): {"drain_triggered_intention_views"},
    ("engram/mcp/server.py", "_serialize_notifications"): {
        "get_notification_surface_service_from_state",
    },
    ("engram/mcp/server.py", "_should_recall"): {"should_recall_for_tool"},
    ("engram/mcp/server.py", "_auto_recall_lite"): {
        "compact_lite_auto_recall_surface",
    },
    ("engram/mcp/server.py", "_auto_recall_full"): {
        "compact_auto_recall_surface",
    },
    ("engram/mcp/server.py", "_session_prime"): {
        "plan_session_prime",
    },
    ("engram/mcp/server.py", "_recall_middleware"): {
        "apply_mcp_recall_enrichment",
        "plan_mcp_recall_middleware",
    },
    ("engram/mcp/server.py", "recall"): {
        "get_recall_item_access_count",
        "get_last_near_miss_views",
        "get_surprise_connection_views",
    },
    ("engram/mcp/server.py", "_get_graph_probe"): {"get_recall_need_graph_probe"},
    ("engram/mcp/server.py", "_get_conv_context"): {"manager_conversation_context"},
    ("engram/mcp/server.py", "_get_conv_embed_fn"): {"manager_conversation_embed_fn"},
    ("engram/mcp/server.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/mcp/server.py", "_get_conv_recent_turns"): {
        "manager_conversation_recent_turns",
    },
    ("engram/mcp/server.py", "_ingest_live_turn"): {
        "_get_conv_context",
        "ingest_manager_conversation_turn",
    },
    ("engram/mcp/server.py", "get_lifecycle_summary"): {
        "ConsolidationAuditReader",
        "get_lifecycle_summary",
    },
    ("engram/mcp/server.py", "get_consolidation_status"): {
        "ConsolidationAuditReader",
        "serialize_cycle_summary",
    },
    ("engram/mcp/server.py", "get_evaluation_report"): {
        "_load_consolidation_evaluation_inputs",
        "build_brain_loop_evaluation_surface",
    },
    ("engram/mcp/server.py", "record_recall_evaluation"): {
        "persist_recall_eval_sample",
        "present_recall_sample_write",
    },
    ("engram/mcp/server.py", "record_session_continuity_evaluation"): {
        "persist_session_continuity_sample",
        "present_session_sample_write",
    },
    ("engram/mcp/server.py", "remember"): {
        "load_episode_adjudication_requests",
        "memory_write_contract",
        "present_mcp_memory_write",
    },
    ("engram/mcp/server.py", "mark_identity_core"): {"mark_identity_core"},
    ("engram/mcp/server.py", "trigger_consolidation"): {
        "serialize_cycle_summary",
        "trigger_consolidation_cycle",
    },
    ("engram/mcp/server.py", "entity_profile_resource"): {"get_entity_profile"},
    ("engram/mcp/server.py", "entity_neighbors_resource"): {"get_entity_neighbors"},
    ("engram/api/knowledge.py", "list_intentions"): {"list_intention_views"},
    ("engram/mcp/server.py", "list_intentions"): {"list_intention_views"},
    ("engram/mcp/server.py", "intend"): {
        "create_intention",
        "effective_intention_threshold",
    },
    ("engram/mcp/server.py", "route_question"): {
        "_get_conv_top_entity_names",
        "route_question",
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


@pytest.mark.parametrize("relative_path", PUBLIC_SURFACES_WITHOUT_APP_STATE_READS)
def test_public_surface_routes_do_not_read_app_state_directly(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text()
    assert "_app_state" not in source
