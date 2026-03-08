"""Dependency helpers for FastAPI endpoints."""

from __future__ import annotations

from engram.config import EngramConfig
from engram.consolidation.engine import ConsolidationEngine
from engram.consolidation.pressure import PressureAccumulator
from engram.consolidation.scheduler import ConsolidationScheduler
from engram.events.bus import EventBus
from engram.graph_manager import GraphManager


def get_manager() -> GraphManager:
    """Retrieve the GraphManager from app state."""
    from engram.main import _app_state

    manager = _app_state.get("graph_manager")
    if not manager:
        raise RuntimeError("GraphManager not initialized")
    return manager


def get_graph_store():
    """Retrieve the GraphStore from app state."""
    from engram.main import _app_state

    store = _app_state.get("graph_store")
    if not store:
        raise RuntimeError("GraphStore not initialized")
    return store


def get_atlas_service():
    """Retrieve the AtlasService from app state."""
    from engram.main import _app_state

    service = _app_state.get("atlas_service")
    if not service:
        raise RuntimeError("AtlasService not initialized")
    return service


def get_event_bus() -> EventBus:
    """Retrieve the EventBus from app state."""
    from engram.main import _app_state

    bus = _app_state.get("event_bus")
    if not bus:
        raise RuntimeError("EventBus not initialized")
    return bus


def get_consolidation_engine() -> ConsolidationEngine:
    """Retrieve the ConsolidationEngine from app state."""
    from engram.main import _app_state

    engine = _app_state.get("consolidation_engine")
    if not engine:
        raise RuntimeError("ConsolidationEngine not initialized")
    return engine


def get_consolidation_scheduler() -> ConsolidationScheduler | None:
    """Retrieve the ConsolidationScheduler from app state (may be None)."""
    from engram.main import _app_state

    return _app_state.get("consolidation_scheduler")


def get_pressure_accumulator() -> PressureAccumulator | None:
    """Retrieve the PressureAccumulator from app state (may be None)."""
    from engram.main import _app_state

    return _app_state.get("pressure_accumulator")


def get_conversation_store():
    """Retrieve the ConversationStore from app state."""
    from engram.main import _app_state

    store = _app_state.get("conversation_store")
    if not store:
        raise RuntimeError("ConversationStore not initialized")
    return store


def get_config() -> EngramConfig:
    """Retrieve the EngramConfig from app state."""
    from engram.main import _app_state

    config = _app_state.get("config")
    if not config:
        raise RuntimeError("EngramConfig not initialized")
    return config
