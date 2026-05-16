from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

from engram.graph_manager import GraphManager

CORE_LIFECYCLE_DELEGATES = {
    "store_episode": ("_capture_service", "store_episode"),
    "project_episode": ("_projection_service", "project_episode"),
    "ingest_episode": ("_episode_ingestion_service", "ingest_episode"),
    "bootstrap_project": ("_project_bootstrap_service", "bootstrap_project"),
    "get_runtime_state": ("_runtime_state_service", "get_runtime_state"),
    "route_question": ("_epistemic_route_service", "route_question"),
    "gather_epistemic_evidence": (
        "_epistemic_evidence_service",
        "gather_epistemic_evidence",
    ),
    "recall": ("_recall_service", "recall"),
    "recall_lite": ("_entity_probe_recall_service", "recall_lite"),
    "recall_medium": ("_entity_probe_recall_service", "recall_medium"),
    "get_context": ("_context_builder", "get_context"),
    "get_graph_state": ("_graph_state_service", "get_graph_state"),
    "get_dashboard_stats": ("_graph_state_service", "get_dashboard_stats"),
    "list_episode_summaries": ("_graph_state_service", "list_episode_summaries"),
    "get_lifecycle_summary": ("_lifecycle_summary_service", "get_lifecycle_summary"),
    "get_activation_snapshot": ("_graph_state_service", "get_activation_snapshot"),
    "get_activation_curve": ("_graph_state_service", "get_activation_curve"),
    "get_graph_neighborhood": ("_graph_state_service", "get_graph_neighborhood"),
    "get_temporal_graph": ("_graph_state_service", "get_temporal_graph"),
    "get_lifecycle_graph_store": ("_graph_state_service", "get_graph_store"),
    "get_entity_profile": ("_graph_state_service", "get_entity_profile"),
    "get_entity_detail": ("_graph_state_service", "get_entity_detail"),
    "get_entity_neighbors": ("_graph_state_service", "get_entity_neighbors"),
    "trigger_consolidation_cycle": (
        "_consolidation_trigger_service",
        "trigger_consolidation_cycle",
    ),
}

COMPATIBILITY_FACADE_DELEGATES = {
    "get_episode_adjudications": (
        "_evidence_adjudication_service",
        "get_episode_adjudications",
    ),
    "_index_materialized_bundle": (
        "_evidence_adjudication_service",
        "index_materialized_bundle",
    ),
    "materialize_evidence": ("_evidence_adjudication_service", "materialize_evidence"),
    "materialize_stored_evidence": (
        "_evidence_adjudication_service",
        "materialize_stored_evidence",
    ),
    "submit_adjudication_resolution": (
        "_evidence_adjudication_service",
        "submit_adjudication_resolution",
    ),
    "get_activation_config": ("_public_surface_policy_service", "activation_config"),
    "get_memory_need_config": ("_public_surface_policy_service", "activation_config"),
    "load_benchmark_corpus": ("_benchmark_load_service", "load_benchmark"),
    "recall_need_graph_probe_enabled": (
        "_public_surface_policy_service",
        "recall_need_graph_probe_enabled",
    ),
    "edge_adjudication_client_enabled": (
        "_public_surface_policy_service",
        "edge_adjudication_client_enabled",
    ),
    "get_explicit_recall_packet_policy": (
        "_public_surface_policy_service",
        "explicit_recall_packet_policy",
    ),
    "get_chat_tool_recall_policy": (
        "_public_surface_policy_service",
        "chat_tool_recall_policy",
    ),
    "recall_usage_feedback_enabled": (
        "_public_surface_policy_service",
        "recall_usage_feedback_enabled",
    ),
    "recall_need_post_response_safety_net_enabled": (
        "_public_surface_policy_service",
        "recall_need_post_response_safety_net_enabled",
    ),
    "get_chat_runtime_policy": ("_public_surface_policy_service", "chat_runtime_policy"),
    "apply_memory_interaction": ("_recall_memory_interaction_applier", "apply"),
    "drain_triggered_intention_views": (
        "_recall_response_state_service",
        "triggered_intention_views",
    ),
    "get_last_near_miss_views": ("_recall_response_state_service", "near_miss_views"),
    "get_recall_item_access_count": ("_recall_response_state_service", "get_access_count"),
    "get_surprise_connection_views": (
        "_recall_response_state_service",
        "surprise_connection_views",
    ),
    "get_conversation_context": ("_conversation_runtime_service", "get_context"),
    "get_conversation_embed_fn": ("_conversation_runtime_service", "get_embed_fn"),
    "get_conversation_turn_count": ("_conversation_runtime_service", "get_turn_count"),
    "get_conversation_top_entity_names": (
        "_conversation_runtime_service",
        "get_top_entity_names",
    ),
    "get_conversation_recent_turns": (
        "_conversation_runtime_service",
        "get_recent_turns",
    ),
    "ingest_conversation_turn": ("_conversation_runtime_service", "ingest_turn"),
    "_publish_access_event": ("_recall_access_recorder", "publish_access_event"),
    "_record_entity_access": ("_recall_access_recorder", "record_entity_access"),
    "_observe_project_files": ("_project_bootstrap_service", "observe_project_files"),
    "_iter_bootstrap_files": ("_project_bootstrap_service", "iter_bootstrap_files"),
    "_resolve_project_entity_id": (
        "_project_bootstrap_service",
        "resolve_project_entity_id",
    ),
    "_upsert_artifact_entity": ("_project_bootstrap_service", "upsert_artifact_entity"),
    "_list_project_artifacts": ("_artifact_search_service", "list_project_artifacts"),
    "search_artifacts": ("_artifact_search_service", "search_artifacts"),
    "_build_epistemic_route": ("_epistemic_route_service", "build_route"),
    "_materialize_artifact_decisions": (
        "_decision_materializer",
        "materialize_artifact_decisions",
    ),
    "_materialize_conversation_decisions": (
        "_decision_materializer",
        "materialize_conversation_decisions",
    ),
    "_upsert_conversation_artifact": (
        "_decision_materializer",
        "upsert_conversation_artifact",
    ),
    "_upsert_decision_entity": ("_decision_materializer", "upsert_decision_entity"),
    "_ensure_relationship": ("_decision_materializer", "ensure_relationship"),
    "_index_entity_with_structure": ("_entity_indexer", "index_entity"),
    "resolve_entity_name": ("_lookup_service", "resolve_entity_name"),
    "search_entities": ("_lookup_service", "search_entities"),
    "search_facts": ("_lookup_service", "search_facts"),
    "_relationship_is_epistemic": ("_lookup_service", "relationship_is_epistemic"),
    "forget_entity": ("_forgetting_service", "forget_entity"),
    "forget_fact": ("_forgetting_service", "forget_fact"),
    "update_entity_profile": ("_entity_mutation_service", "update_entity_profile"),
    "delete_entity_by_id": ("_entity_mutation_service", "delete_entity_by_id"),
    "mark_identity_core": ("_identity_core_service", "mark_identity_core"),
    "get_consolidation_shared_db": (
        "_consolidation_trigger_service",
        "shared_sqlite_db",
    ),
    "create_intention": ("_prospective_memory_service", "create_intention"),
    "_create_intention_v1": ("_prospective_memory_service", "_create_intention_v1"),
    "list_intentions": ("_prospective_memory_service", "list_intentions"),
    "list_intention_views": ("_prospective_memory_service", "list_intention_views"),
    "effective_intention_threshold": (
        "_prospective_memory_service",
        "effective_activation_threshold",
    ),
    "dismiss_intention": ("_prospective_memory_service", "dismiss_intention"),
    "delete_intention": ("_prospective_memory_service", "delete_intention"),
    "migrate_flat_intentions": (
        "_prospective_memory_service",
        "migrate_flat_intentions",
    ),
    "_update_intention_fire": ("_prospective_memory_service", "update_intention_fire"),
    "update_intention_meta": ("_prospective_memory_service", "update_intention_meta"),
    "_entity_to_context_data": ("_context_builder", "entity_to_context_data"),
    "invalidate_briefing_cache": ("_context_builder", "invalidate_briefing_cache"),
    "_template_briefing": ("_context_builder", "template_briefing"),
}


def _method_calls_service(method_name: str, service_attr: str, service_method: str) -> bool:
    source = textwrap.dedent(inspect.getsource(getattr(GraphManager, method_name)))
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != service_method:
            continue
        receiver = node.func.value
        if not isinstance(receiver, ast.Attribute):
            continue
        if receiver.attr != service_attr:
            continue
        if isinstance(receiver.value, ast.Name) and receiver.value.id == "self":
            return True
    return False


def _method_calls_function_for_self_attrs(
    method_name: str,
    function_name: str,
    expected_attrs: tuple[str, ...],
) -> bool:
    source = textwrap.dedent(inspect.getsource(getattr(GraphManager, method_name)))
    tree = ast.parse(source)
    seen_attrs: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != function_name:
            continue
        if len(node.args) != 1 or not isinstance(node.args[0], ast.Attribute):
            continue
        receiver = node.args[0]
        if isinstance(receiver.value, ast.Name) and receiver.value.id == "self":
            seen_attrs.append(receiver.attr)
    return tuple(seen_attrs) == expected_attrs


@pytest.mark.parametrize(
    ("method_name", "expected_delegate"),
    CORE_LIFECYCLE_DELEGATES.items(),
)
def test_core_lifecycle_facades_delegate_to_services(
    method_name: str,
    expected_delegate: tuple[str, str],
) -> None:
    service_attr, service_method = expected_delegate
    assert _method_calls_service(method_name, service_attr, service_method)


def test_close_runtime_resources_closes_owned_runtime_stores() -> None:
    assert _method_calls_function_for_self_attrs(
        "close_runtime_resources",
        "close_if_supported",
        ("_search", "_activation", "_graph"),
    )


@pytest.mark.parametrize(
    ("method_name", "expected_delegate"),
    COMPATIBILITY_FACADE_DELEGATES.items(),
)
def test_compatibility_facades_delegate_to_services(
    method_name: str,
    expected_delegate: tuple[str, str],
) -> None:
    service_attr, service_method = expected_delegate
    assert _method_calls_service(method_name, service_attr, service_method)
