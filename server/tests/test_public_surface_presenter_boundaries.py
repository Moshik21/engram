from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PRESENTER_BOUNDARIES = {
    ("engram/api/knowledge.py", "observe"): {
        "build_api_observe_write_surface",
    },
    ("engram/api/knowledge.py", "auto_observe"): {
        "build_api_auto_observe_request_surface",
    },
    ("engram/api/knowledge.py", "observe_image"): {
        "build_api_attachment_observe_write_surface",
    },
    ("engram/api/knowledge.py", "observe_file"): {
        "build_api_attachment_observe_write_surface",
    },
    ("engram/api/knowledge.py", "remember"): {
        "build_api_remember_write_surface",
    },
    ("engram/api/knowledge.py", "recall"): {
        "build_api_recall_surface",
    },
    ("engram/mcp/server.py", "remember"): {
        "build_mcp_remember_write_surface",
    },
    ("engram/mcp/server.py", "observe"): {
        "build_mcp_observe_write_surface",
        "build_mcp_observe_recall_surface",
    },
    ("engram/mcp/server.py", "observe_image"): {
        "build_mcp_attachment_observe_write_surface",
    },
    ("engram/mcp/server.py", "observe_file"): {
        "build_mcp_attachment_observe_write_surface",
    },
    ("engram/mcp/server.py", "recall"): {
        "build_mcp_explicit_recall_tool_surface",
    },
}

PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES = {
    ("engram/api/websocket.py", "dashboard_ws"): {
        "close_dashboard_websocket_auth_failure",
        "get_event_bus",
        "get_manager",
        "get_notification_surface_service",
        "resolve_dashboard_websocket_tenant",
        "run_dashboard_websocket_session",
    },
    ("engram/api/knowledge.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/api/knowledge.py", "observe"): {
        "build_api_observe_write_surface",
    },
    ("engram/api/knowledge.py", "auto_observe"): {
        "build_api_auto_observe_request_surface",
        "get_config",
    },
    ("engram/api/knowledge.py", "observe_image"): {
        "build_api_attachment_observe_write_surface",
    },
    ("engram/api/knowledge.py", "observe_file"): {
        "build_api_attachment_observe_write_surface",
    },
    ("engram/api/knowledge.py", "replay_queue"): {
        "build_api_manager_offline_replay_surface",
    },
    ("engram/api/knowledge.py", "remember"): {
        "build_api_remember_write_surface",
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
    ("engram/api/knowledge.py", "chat"): {
        "build_api_chat_stream_response_surface",
        "get_event_bus",
        "get_rate_limiter",
        "get_optional_conversation_store",
    },
    ("engram/api/health.py", "health_check"): {
        "build_api_health_response",
    },
    ("engram/api/consolidation.py", "trigger_consolidation"): {
        "build_api_consolidation_trigger_response_surface",
    },
    ("engram/api/consolidation.py", "consolidation_status"): {
        "build_api_consolidation_status_response_surface",
        "get_config",
        "get_consolidation_engine",
        "get_consolidation_scheduler",
        "get_pressure_accumulator",
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
        "build_api_brain_loop_evaluation_surface",
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
    ("engram/api/knowledge.py", "get_fast_runtime_packet"): {
        "build_fast_runtime_packet",
        "load_packet_cache_summary",
        "schedule_project_file_prefix_warmup",
        "get_config",
        "get_mode",
    },
    ("engram/api/knowledge.py", "get_packet_cache"): {
        "build_api_packet_cache_summary_surface",
    },
    ("engram/api/knowledge.py", "clear_packet_cache"): {
        "build_api_packet_cache_clear_surface",
    },
    ("engram/api/knowledge.py", "get_adjudications"): {
        "build_api_adjudications_list_surface",
    },
    ("engram/api/storage.py", "storage_summary"): {
        "get_storage_diagnostics",
    },
    ("engram/api/ingest_ws.py", "ingest_ws"): {
        "get_event_bus",
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
    ("engram/api/graph.py", "get_atlas"): {
        "build_api_atlas_json_response",
        "build_api_atlas_surface",
    },
    ("engram/api/graph.py", "get_atlas_history"): {
        "build_api_atlas_history_surface",
    },
    ("engram/api/graph.py", "get_region"): {
        "build_api_atlas_json_response",
        "build_api_atlas_region_surface",
    },
    ("engram/api/graph.py", "get_neighborhood"): {"build_api_graph_neighborhood_surface"},
    ("engram/api/graph.py", "get_graph_at"): {"build_api_temporal_graph_surface"},
    ("engram/api/stats.py", "get_stats"): {"build_api_dashboard_stats_surface"},
    ("engram/api/episodes.py", "list_episodes"): {"build_api_episode_list_surface"},
    ("engram/api/lifecycle.py", "lifecycle_summary"): {"build_api_lifecycle_summary_surface"},
    ("engram/api/activation.py", "get_activation_snapshot"): {
        "build_api_activation_snapshot_surface",
    },
    ("engram/api/activation.py", "get_activation_curve"): {
        "build_api_activation_curve_surface",
    },
    ("engram/mcp/server.py", "_serialize_notifications"): {
        "build_mcp_notifications_surface",
        "get_notification_surface_service",
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
        "_ingest_live_tool_turn",
        "run_mcp_recall_middleware",
    },
    ("engram/mcp/server.py", "recall"): {
        "build_mcp_explicit_recall_tool_surface",
    },
    ("engram/mcp/server.py", "_get_conv_context"): {"manager_conversation_context"},
    ("engram/mcp/server.py", "_get_conv_top_entity_names"): {
        "manager_conversation_top_entity_names",
    },
    ("engram/mcp/server.py", "_ingest_live_turn"): {
        "_get_conv_context",
        "ingest_manager_conversation_turn",
    },
    ("engram/mcp/server.py", "_ingest_live_tool_turn"): {
        "_ingest_live_turn",
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
        "build_mcp_remember_write_surface",
    },
    ("engram/mcp/server.py", "observe"): {
        "build_mcp_observe_write_surface",
        "build_mcp_observe_recall_surface",
    },
    ("engram/mcp/server.py", "observe_image"): {
        "build_mcp_attachment_observe_write_surface",
    },
    ("engram/mcp/server.py", "observe_file"): {
        "build_mcp_attachment_observe_write_surface",
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
        "build_mcp_entity_search_tool_surface",
    },
    ("engram/mcp/server.py", "search_facts"): {
        "build_mcp_fact_search_tool_surface",
    },
    ("engram/mcp/server.py", "timeline"): {
        "build_mcp_timeline_tool_surface",
    },
    ("engram/mcp/server.py", "get_context"): {
        "build_mcp_context_tool_surface",
    },
    ("engram/mcp/server.py", "bootstrap_project"): {
        "build_project_bootstrap_surface",
    },
    ("engram/mcp/server.py", "get_runtime_state"): {
        "build_runtime_state_surface",
    },
    ("engram/mcp/server.py", "claim_authority"): {
        "build_mcp_memory_authority_surface",
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
        "build_mcp_question_route_tool_surface",
    },
    ("engram/mcp/server.py", "search_artifacts"): {
        "build_mcp_artifact_search_tool_surface",
    },
    # Loop Steward + hygiene REST surfaces (TTL-clamped adjustment overlay,
    # read-only debt scoreboard). These delegate to loop_adjustment /
    # hygiene_debt module helpers rather than manager/engine facades.
    ("engram/api/loop.py", "get_loop_status"): {
        "status_payload",
    },
    ("engram/api/loop.py", "post_loop_apply"): {
        "clamp_loop_adjustment",
        "save_active_adjustment_async",
        "status_payload",
    },
    ("engram/api/loop.py", "clear_loop"): {
        "clear_active_adjustment_async",
    },
    ("engram/api/hygiene.py", "hygiene_debt"): {
        "collect_hygiene_debt_from_store",
        "get_config",
        "get_graph_store",
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
    ("engram/api/health.py", "health_check"): {
        "build_api_health_surface",
        "get_config",
        "get_graph_store",
        "get_mode",
        "get_stats",
    },
    ("engram/api/activation.py", "get_activation_curve"): {
        "get_activation_curve",
        "HTTPException",
    },
    ("engram/api/websocket.py", "dashboard_ws"): {
        "AuthConfig",
        "Headers",
        "query_params",
        "resolve_tenant_from_scope",
    },
    ("engram/api/evaluation.py", "brain_loop_evaluation_report"): {
        "build_brain_loop_evaluation_surface",
        "get_recent_evaluation_context",
    },
    ("engram/api/consolidation.py", "trigger_consolidation"): {
        "add_task",
        "build_api_consolidation_trigger_surface",
        "run_api_consolidation_cycle",
    },
    ("engram/api/consolidation.py", "consolidation_status"): {
        "activation",
        "build_api_consolidation_status_surface",
    },
    ("engram/api/lifecycle.py", "lifecycle_summary"): {
        "get_lifecycle_summary",
    },
    ("engram/api/knowledge.py", "chat"): {
        "analyze_chat_memory_need",
        "anthropic",
        "AsyncAnthropic",
        "apply_chat_recall_feedback",
        "build_chat_context_surface",
        "build_chat_messages",
        "build_chat_runtime_policy",
        "build_chat_system_prompt_surface",
        "build_chat_tool_stream_events",
        "CHAT_TOOLS",
        "check_api_chat_rate_limit",
        "chat_conversation_not_found_payload",
        "create_task",
        "event_stream",
        "extract_message_text",
        "gather_epistemic_evidence",
        "get_context",
        "get_chat_runtime_policy",
        "hydrate_chat_context",
        "json",
        "MAX_HISTORY_MESSAGES",
        "MAX_TOOL_TURNS",
        "record_chat_assistant_turn",
        "persist_chat_turn",
        "retry_memory_grounded_response",
        "resolve_chat_conversation",
        "run_chat_response_turn",
        "run_chat_tool_use_loop",
        "schedule_chat_turn_persistence",
        "should_retry_chat_response",
        "stream_api_chat_sse_events",
        "_sse",
    },
    ("engram/api/knowledge.py", "replay_queue"): {
        "build_api_offline_replay_surface",
        "store_episode",
    },
    ("engram/api/knowledge.py", "auto_observe"): {
        "memory_write_contract",
        "parse_conversation_date",
        "present_api_memory_write",
        "present_api_observe_skip",
        "store_observation",
    },
    ("engram/api/knowledge.py", "remember"): {
        "edge_adjudication_client_enabled",
        "ingest_projecting_memory",
        "load_client_enabled_episode_adjudication_requests",
        "load_episode_adjudication_requests",
        "memory_write_contract",
        "parse_conversation_date",
        "present_api_memory_write",
    },
    ("engram/api/knowledge.py", "observe"): {
        "memory_write_contract",
        "parse_conversation_date",
        "present_api_memory_write",
        "store_observation",
    },
    ("engram/api/knowledge.py", "observe_image"): {
        "build_observation_attachment",
        "memory_write_contract",
        "present_api_memory_write",
        "store_observation",
    },
    ("engram/api/knowledge.py", "observe_file"): {
        "build_observation_attachment",
        "memory_write_contract",
        "present_api_memory_write",
        "store_observation",
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

# loop.py deliberately bridges to main._app_state to reach the app-held
# consolidation store (Helix native sidecar path) when no DI dependency exists
# for it yet. Tracked debt: give it a get_consolidation_store() dependency and
# drop this exemption. Every other route file must stay _app_state-free.
PUBLIC_SURFACES_WITHOUT_APP_STATE_READS = tuple(
    sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "engram/api").glob("*.py")
        if path.name not in {"__init__.py", "deps.py", "loop.py"}
    )
)

SHARED_RUNTIME_MODULES_WITHOUT_APP_STATE_READS = (
    "engram/api/websocket_runtime.py",
    "engram/consolidation/scheduler.py",
    "engram/notifications/surface.py",
)

RUNTIME_SHUTDOWN_BOUNDARIES = {
    ("engram/main.py", "_shutdown"): {
        "close_if_supported",
        "close_runtime_resources",
        "run_shutdown_consolidation",
        "stop_if_supported",
    },
    ("engram/mcp/server.py", "_shutdown"): {
        "close_if_supported",
        "close_runtime_resources",
        "stop_if_supported",
    },
}

RUNTIME_SHUTDOWN_FORBIDDEN_IDENTIFIERS = {
    ("engram/main.py", "_shutdown"): {
        "aclose",
        "close",
        # "cancel" is intentionally permitted here: in shell (non-monolith)
        # role, _shutdown cancels a running consolidation engine instead of
        # running a shutdown cycle, to avoid racing the cold brain's exclusive
        # graph open. Running a live cycle here produced zombie 'shutdown'
        # cycles. It still routes stop/close through the shared helpers below.
        "is_running",
        "run_cycle",
        "stop",
    },
    ("engram/mcp/server.py", "_shutdown"): {
        "_maybe_close",
        "close",
        "stop",
    },
}

DASHBOARD_WEBSOCKET_RUNTIME_BOUNDARIES = {
    ("engram/api/websocket_runtime.py", "_forward_dashboard_events"): {
        "flatten_dashboard_event",
    },
    ("engram/api/websocket_runtime.py", "_run_dashboard_activation_snapshots"): {
        "build_api_activation_snapshot_surface",
        "build_dashboard_activation_snapshot_message",
    },
    ("engram/api/websocket_runtime.py", "_receive_dashboard_commands"): {
        "build_dashboard_pong_surface",
        "build_dashboard_resync_surface",
        "dismiss_dashboard_notification_command",
    },
    ("engram/api/websocket_runtime.py", "run_dashboard_websocket_session"): {
        "_forward_dashboard_events",
        "_receive_dashboard_commands",
        "subscribe",
        "unsubscribe",
    },
}

PUBLIC_API_ROUTE_ALLOWED_CONTROL_FLOW = Counter(
    {
        ("engram/api/knowledge.py", "chat", "If"): 1,
        ("engram/api/websocket.py", "dashboard_ws", "ExceptHandler"): 2,
        ("engram/api/websocket.py", "dashboard_ws", "Try"): 2,
        ("engram/api/ingest_ws.py", "ingest_ws", "ExceptHandler"): 3,
        ("engram/api/ingest_ws.py", "ingest_ws", "If"): 2,
        ("engram/api/ingest_ws.py", "ingest_ws", "Try"): 2,
        ("engram/api/ingest_ws.py", "ingest_ws", "While"): 1,
        # Shell-role 409 guard: consolidation runs in the cold brain, not the
        # hot shell, so trigger_consolidation early-returns on the wrong role.
        ("engram/api/consolidation.py", "trigger_consolidation", "If"): 1,
        # hygiene_debt: read-only debt scoreboard, guards store collection.
        ("engram/api/hygiene.py", "hygiene_debt", "ExceptHandler"): 1,
        ("engram/api/hygiene.py", "hygiene_debt", "Try"): 1,
        # Loop Steward routes: file-first status + store-optional apply/clear
        # with _app_state/config fallbacks guarded by try/except. Branchier
        # than the presenter ideal; counts are pinned so any further growth
        # trips this test.
        ("engram/api/loop.py", "clear_loop", "BoolOp"): 1,
        ("engram/api/loop.py", "clear_loop", "ExceptHandler"): 1,
        ("engram/api/loop.py", "clear_loop", "Try"): 1,
        ("engram/api/loop.py", "get_loop_status", "BoolOp"): 1,
        ("engram/api/loop.py", "get_loop_status", "ExceptHandler"): 1,
        ("engram/api/loop.py", "get_loop_status", "Try"): 1,
        ("engram/api/loop.py", "post_loop_apply", "BoolOp"): 2,
        ("engram/api/loop.py", "post_loop_apply", "ExceptHandler"): 2,
        ("engram/api/loop.py", "post_loop_apply", "If"): 1,
        ("engram/api/loop.py", "post_loop_apply", "Try"): 2,
    }
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


def _functions_directly_awaiting(relative_path: str, call_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    callers: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == call_name:
                callers.add(function.name)
    return callers


def _direct_runtime_method_calls(
    relative_paths: list[str],
    *,
    owner_names: set[str],
) -> set[tuple[str, str, str, str]]:
    calls: set[tuple[str, str, str, str]] = set()
    for relative_path in relative_paths:
        tree = ast.parse((ROOT / relative_path).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                owner = node.func.value
                if not isinstance(owner, ast.Name):
                    continue
                if owner.id not in owner_names:
                    continue
                calls.add((relative_path, function.name, owner.id, node.func.attr))
    return calls


def _decorated_mcp_function_names(relative_path: str) -> set[str]:
    tree = ast.parse((ROOT / relative_path).read_text())
    function_names: set[str] = set()
    for function in ast.walk(tree):
        if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for decorator in function.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "mcp"
                and func.attr in {"tool", "resource", "prompt"}
            ):
                function_names.add(function.name)
    return function_names


def _direct_runtime_method_calls_in_functions(
    relative_path: str,
    *,
    function_names: set[str],
    owner_names: set[str],
) -> set[tuple[str, str, str, str]]:
    return {
        call
        for call in _direct_runtime_method_calls([relative_path], owner_names=owner_names)
        if call[1] in function_names
    }


def _direct_awaited_name_calls(relative_paths: list[str]) -> set[tuple[str, str, str]]:
    calls: set[tuple[str, str, str]] = set()
    for relative_path in relative_paths:
        tree = ast.parse((ROOT / relative_path).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
                    continue
                func = node.value.func
                if isinstance(func, ast.Name):
                    calls.add((relative_path, function.name, func.id))
    return calls


def _public_api_route_paths() -> list[str]:
    return [
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "engram/api").glob("*.py")
        if path.name not in {"__init__.py", "deps.py"} and "APIRouter" in path.read_text()
    ]


def _decorated_api_route_surfaces() -> set[tuple[str, str]]:
    surfaces: set[tuple[str, str]] = set()
    for relative_path in _public_api_route_paths():
        tree = ast.parse((ROOT / relative_path).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            for decorator in function.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(func, ast.Attribute) and func.attr in {
                    "delete",
                    "get",
                    "patch",
                    "post",
                    "websocket",
                }:
                    surfaces.add((relative_path, function.name))
    return surfaces


def _decorated_api_route_nested_functions() -> set[tuple[str, str, str]]:
    nested: set[tuple[str, str, str]] = set()
    for relative_path in _public_api_route_paths():
        tree = ast.parse((ROOT / relative_path).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            is_route = False
            for decorator in function.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(func, ast.Attribute) and func.attr in {
                    "delete",
                    "get",
                    "patch",
                    "post",
                    "websocket",
                }:
                    is_route = True
                    break
            if not is_route:
                continue
            for node in ast.walk(function):
                if node is function:
                    continue
                if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
                    nested.add((relative_path, function.name, node.name))
    return nested


def _decorated_api_route_control_flow() -> Counter[tuple[str, str, str]]:
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.BoolOp,
        ast.ExceptHandler,
        ast.For,
        ast.If,
        ast.IfExp,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )
    counter: Counter[tuple[str, str, str]] = Counter()
    for relative_path in _public_api_route_paths():
        tree = ast.parse((ROOT / relative_path).read_text())
        for function in ast.walk(tree):
            if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
                continue
            is_route = False
            for decorator in function.decorator_list:
                func = decorator.func if isinstance(decorator, ast.Call) else decorator
                if isinstance(func, ast.Attribute) and func.attr in {
                    "delete",
                    "get",
                    "patch",
                    "post",
                    "websocket",
                }:
                    is_route = True
                    break
            if not is_route:
                continue
            for node in ast.walk(function):
                if isinstance(node, control_flow_nodes):
                    counter[(relative_path, function.name, type(node).__name__)] += 1
    return counter


def _nested_functions_in(surface: tuple[str, str]) -> set[tuple[str, str, str]]:
    relative_path, function_name = surface
    tree = ast.parse((ROOT / relative_path).read_text())
    for function in ast.walk(tree):
        if not isinstance(function, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        if function.name != function_name:
            continue
        return {
            (relative_path, function.name, node.name)
            for node in ast.walk(function)
            if node is not function and isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
        }
    raise AssertionError(f"Function not found: {relative_path}:{function_name}")


def _decorated_mcp_nested_functions(relative_path: str) -> set[tuple[str, str, str]]:
    return {
        nested
        for function_name in _decorated_mcp_function_names(relative_path)
        for nested in _nested_functions_in((relative_path, function_name))
    }


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


@pytest.mark.parametrize("relative_path", SHARED_RUNTIME_MODULES_WITHOUT_APP_STATE_READS)
def test_shared_runtime_modules_do_not_read_app_state_directly(relative_path: str) -> None:
    source = (ROOT / relative_path).read_text()
    assert "_app_state" not in source


@pytest.mark.parametrize(
    ("surface", "expected_names"),
    RUNTIME_SHUTDOWN_BOUNDARIES.items(),
)
def test_runtime_shutdown_uses_shared_stop_close_boundaries(
    surface: tuple[str, str],
    expected_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_identifiers_used(relative_path, function_name)
    assert expected_names - names_used == set()


@pytest.mark.parametrize(
    ("surface", "forbidden_names"),
    RUNTIME_SHUTDOWN_FORBIDDEN_IDENTIFIERS.items(),
)
def test_runtime_shutdown_does_not_reintroduce_local_shutdown_logic(
    surface: tuple[str, str],
    forbidden_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_identifiers_used(relative_path, function_name)
    assert names_used & forbidden_names == set()


@pytest.mark.parametrize(
    ("surface", "expected_names"),
    DASHBOARD_WEBSOCKET_RUNTIME_BOUNDARIES.items(),
)
def test_dashboard_websocket_runtime_uses_shared_boundaries(
    surface: tuple[str, str],
    expected_names: set[str],
) -> None:
    relative_path, function_name = surface
    names_used = _function_identifiers_used(relative_path, function_name)
    assert expected_names - names_used == set()


def test_mcp_tool_handlers_do_not_directly_await_recall_middleware() -> None:
    assert _functions_directly_awaiting("engram/mcp/server.py", "_recall_middleware") == set()


def test_mcp_recall_middleware_does_not_embed_nested_runtime_callbacks() -> None:
    assert _nested_functions_in(("engram/mcp/server.py", "_recall_middleware")) == set()


def test_mcp_public_surfaces_do_not_embed_nested_orchestration() -> None:
    assert _decorated_mcp_nested_functions("engram/mcp/server.py") == set()


def test_all_public_api_routes_have_orchestration_boundary_entries() -> None:
    uncovered = _decorated_api_route_surfaces() - set(PUBLIC_MUTATION_ORCHESTRATION_BOUNDARIES)
    assert uncovered == set()


def test_public_api_route_handlers_do_not_embed_nested_orchestration() -> None:
    assert _decorated_api_route_nested_functions() == set()


def test_public_api_route_handlers_only_keep_known_transport_branching() -> None:
    assert _decorated_api_route_control_flow() == PUBLIC_API_ROUTE_ALLOWED_CONTROL_FLOW


def test_public_routes_do_not_dispatch_manager_methods_directly() -> None:
    allowed = {
        ("engram/mcp/server.py", "_shutdown", "_manager", "close_runtime_resources"),
        # Lightweight adoption-debt metrics snapshot on session middleware.
        (
            "engram/mcp/server.py",
            "_session_adoption_debt",
            "manager",
            "get_memory_operation_metrics",
        ),
    }
    assert (
        _direct_runtime_method_calls(
            [*_public_api_route_paths(), "engram/mcp/server.py"],
            owner_names={"manager", "_manager"},
        )
        == allowed
    )


def test_public_routes_do_not_dispatch_engine_methods_directly() -> None:
    assert _direct_runtime_method_calls(_public_api_route_paths(), owner_names={"engine"}) == set()


def test_public_routes_do_not_dispatch_store_or_service_methods_directly() -> None:
    assert (
        _direct_runtime_method_calls(
            _public_api_route_paths(),
            owner_names={
                "atlas_service",
                "conv_store",
                "conversation_store",
                "evaluation_store",
                "graph_store",
                "notification_surface",
                "rate_limiter",
                "service",
                "store",
            },
        )
        == set()
    )


def test_public_api_routes_only_await_route_facing_helpers() -> None:
    allowed_names = {
        "close_dashboard_websocket_auth_failure",
        "resolve_dashboard_websocket_tenant",
        "run_dashboard_websocket_session",
        # Loop Steward / hygiene async domain helpers awaited directly by their
        # routes (no presenter surface wraps them yet — tracked debt).
        "clear_active_adjustment_async",
        "collect_hygiene_debt_from_store",
        "load_active_adjustment_async",
        "save_active_adjustment_async",
    }
    allowed_prefixes = (
        "build_",
        "dismiss_",
    )
    violations = {
        call
        for call in _direct_awaited_name_calls(_public_api_route_paths())
        if call[2] not in allowed_names and not call[2].startswith(allowed_prefixes)
    }
    assert violations == set()


def test_mcp_public_surfaces_do_not_dispatch_store_or_session_methods_directly() -> None:
    assert (
        _direct_runtime_method_calls_in_functions(
            "engram/mcp/server.py",
            function_names=_decorated_mcp_function_names("engram/mcp/server.py"),
            owner_names={
                "_consolidation_store",
                "_evaluation_store",
                "_session",
                "consolidation_store",
                "evaluation_store",
                "session",
                "store",
            },
        )
        == set()
    )


def test_mcp_public_surfaces_only_await_route_facing_helpers() -> None:
    allowed_names = {
        "_get_consolidation_store",
        "_get_evaluation_store",
        "resolve_mcp_consolidation_trigger_store",
        # Loop Steward operator MCP tools await the loop_adjustment async
        # helpers directly (same domain helpers as the REST routes).
        "clear_active_adjustment_async",
        "load_active_adjustment_async",
        "save_active_adjustment_async",
    }
    allowed_prefixes = ("build_",)
    violations = {
        call
        for call in _direct_awaited_name_calls(["engram/mcp/server.py"])
        if call[1] in _decorated_mcp_function_names("engram/mcp/server.py")
        and call[2] not in allowed_names
        and not call[2].startswith(allowed_prefixes)
    }
    assert violations == set()
