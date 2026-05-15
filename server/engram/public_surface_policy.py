"""Runtime policy snapshots for REST and MCP public surfaces."""

from __future__ import annotations

from dataclasses import dataclass

from engram.config import ActivationConfig


@dataclass(frozen=True)
class ExplicitRecallPacketPolicy:
    enabled: bool
    max_packets: int


@dataclass(frozen=True)
class ChatToolRecallPolicy:
    usage_feedback_enabled: bool
    telemetry_enabled: bool
    packets_enabled: bool
    packet_limit: int
    record_access: bool
    interaction_type: str | None
    interaction_source: str


@dataclass(frozen=True)
class ChatRuntimePolicy:
    recall_need_analyzer_enabled: bool
    recall_telemetry_enabled: bool
    epistemic_routing_enabled: bool


class PublicSurfacePolicyService:
    """Expose route-facing runtime policy without leaking manager internals."""

    def __init__(self, cfg: ActivationConfig) -> None:
        self._cfg = cfg

    def activation_config(self) -> ActivationConfig:
        return self._cfg

    def recall_need_graph_probe_enabled(self) -> bool:
        return self._cfg.recall_need_graph_probe_enabled

    def edge_adjudication_client_enabled(self) -> bool:
        return self._cfg.edge_adjudication_client_enabled

    def explicit_recall_packet_policy(self) -> ExplicitRecallPacketPolicy:
        return ExplicitRecallPacketPolicy(
            enabled=self._cfg.recall_packets_enabled,
            max_packets=self._cfg.recall_packet_explicit_limit,
        )

    def chat_tool_recall_policy(self) -> ChatToolRecallPolicy:
        usage_feedback_enabled = self._cfg.recall_usage_feedback_enabled
        telemetry_enabled = self._cfg.recall_telemetry_enabled
        record_access = True
        interaction_type = None
        interaction_source = "chat_tool_use"
        if usage_feedback_enabled:
            record_access = False
            interaction_type = "selected"
            interaction_source = "chat_tool_select"
        elif telemetry_enabled:
            interaction_type = "used"

        return ChatToolRecallPolicy(
            usage_feedback_enabled=usage_feedback_enabled,
            telemetry_enabled=telemetry_enabled,
            packets_enabled=self._cfg.recall_packets_enabled,
            packet_limit=self._cfg.recall_packet_chat_limit,
            record_access=record_access,
            interaction_type=interaction_type,
            interaction_source=interaction_source,
        )

    def recall_usage_feedback_enabled(self) -> bool:
        return self._cfg.recall_usage_feedback_enabled

    def recall_need_post_response_safety_net_enabled(self) -> bool:
        return self._cfg.recall_need_post_response_safety_net_enabled

    def chat_runtime_policy(self) -> ChatRuntimePolicy:
        return ChatRuntimePolicy(
            recall_need_analyzer_enabled=self._cfg.recall_need_analyzer_enabled,
            recall_telemetry_enabled=self._cfg.recall_telemetry_enabled,
            epistemic_routing_enabled=self._cfg.epistemic_routing_enabled,
        )
