from __future__ import annotations

from engram.config import ActivationConfig
from engram.public_surface_policy import PublicSurfacePolicyService


def test_public_surface_policy_exposes_explicit_recall_packet_settings() -> None:
    cfg = ActivationConfig(
        recall_packets_enabled=True,
        recall_packet_explicit_limit=4,
    )
    service = PublicSurfacePolicyService(cfg)

    policy = service.explicit_recall_packet_policy()

    assert service.activation_config() is cfg
    assert policy.enabled is True
    assert policy.max_packets == 4


def test_public_surface_policy_chat_tool_usage_feedback_semantics() -> None:
    service = PublicSurfacePolicyService(
        ActivationConfig(
            recall_usage_feedback_enabled=True,
            recall_telemetry_enabled=True,
            recall_packets_enabled=False,
        )
    )

    policy = service.chat_tool_recall_policy()

    assert policy.record_access is False
    assert policy.interaction_type == "selected"
    assert policy.interaction_source == "chat_tool_select"
    assert policy.packets_enabled is False


def test_public_surface_policy_chat_tool_telemetry_semantics() -> None:
    service = PublicSurfacePolicyService(
        ActivationConfig(
            recall_usage_feedback_enabled=False,
            recall_telemetry_enabled=True,
            recall_packets_enabled=True,
            recall_packet_chat_limit=3,
        )
    )

    policy = service.chat_tool_recall_policy()

    assert policy.record_access is True
    assert policy.interaction_type == "used"
    assert policy.interaction_source == "chat_tool_use"
    assert policy.packets_enabled is True
    assert policy.packet_limit == 3


def test_public_surface_policy_chat_runtime_flags() -> None:
    service = PublicSurfacePolicyService(
        ActivationConfig(
            recall_need_analyzer_enabled=True,
            recall_telemetry_enabled=True,
            epistemic_routing_enabled=True,
        )
    )

    policy = service.chat_runtime_policy()

    assert policy.recall_need_analyzer_enabled is True
    assert policy.recall_telemetry_enabled is True
    assert policy.epistemic_routing_enabled is True
