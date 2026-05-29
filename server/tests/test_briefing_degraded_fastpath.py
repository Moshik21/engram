"""Fast-path briefing degradation flagging.

When ``get_context(format="briefing")`` is served from a fast-path payload
(cache hit / session-recent), the briefing is silently rendered as
``structured``.  These tests assert the swap is now visible via the
``briefing_degraded`` flag that the deep builder already sets.
"""

from types import SimpleNamespace

from engram.config import ActivationConfig
from engram.retrieval.budgets import recall_budget_for_profile
from engram.retrieval.context_builder import _cached_context_payload_from_manager


def _make_manager(cfg: ActivationConfig, packet: dict) -> SimpleNamespace:
    def get_cached_memory_packets(group_id, *, scope, topic_hint, project_path, sync_persistent):
        if scope == "session_recent":
            return SimpleNamespace(packets=[packet])
        return SimpleNamespace(packets=[])

    return SimpleNamespace(
        get_activation_config=lambda: cfg,
        get_cached_memory_packets=get_cached_memory_packets,
    )


def test_cache_fast_path_flags_briefing_degradation() -> None:
    cfg = ActivationConfig(recall_packet_cache_enabled=True)
    packet = {
        "packet_type": "session_recent",
        "title": "Recent session note",
        "summary": "Session-recent context that answers the topic.",
        "trust": {"source": "session", "freshness": "fresh"},
    }
    manager = _make_manager(cfg, packet)
    budget = recall_budget_for_profile(profile="chat", cfg=cfg)

    payload = _cached_context_payload_from_manager(
        manager,
        group_id="default",
        topic_hint=None,
        project_path=None,
        format="briefing",
        budget=budget,
        status="ok",
        duration_ms=1.0,
        skip_reason=None,
        timeout=False,
    )

    assert payload is not None
    # Briefing is served as structured on the fast path...
    assert payload["format"] == "structured"
    # ...but the degradation is now visible instead of silent.
    assert payload["briefing_degraded"] is True
    assert payload["briefing_degraded_reason"] == "cache_fast_path"


def test_cache_fast_path_structured_request_not_flagged() -> None:
    cfg = ActivationConfig(recall_packet_cache_enabled=True)
    packet = {
        "packet_type": "session_recent",
        "title": "Recent session note",
        "summary": "Session-recent context that answers the topic.",
        "trust": {"source": "session", "freshness": "fresh"},
    }
    manager = _make_manager(cfg, packet)
    budget = recall_budget_for_profile(profile="chat", cfg=cfg)

    payload = _cached_context_payload_from_manager(
        manager,
        group_id="default",
        topic_hint=None,
        project_path=None,
        format="structured",
        budget=budget,
        status="ok",
        duration_ms=1.0,
        skip_reason=None,
        timeout=False,
    )

    assert payload is not None
    assert payload["format"] == "structured"
    assert "briefing_degraded" not in payload
