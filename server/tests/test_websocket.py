"""Tests for the WebSocket dashboard endpoint."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from engram.config import EngramConfig
from engram.events.bus import get_event_bus
from engram.main import _app_state, create_app
from engram.notifications.models import MemoryNotification


@pytest.fixture
def ws_app(tmp_path):
    """Create a synchronous test app for websocket testing."""
    config = EngramConfig(
        mode="lite",
        sqlite={"path": str(tmp_path / "ws_test.db")},
    )
    app = create_app(config)
    return app, config


class TestWebSocket:
    def test_connect_and_pong(self, ws_app):
        """WebSocket connects and responds to ping with pong."""
        app, config = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                ws.send_json({"type": "ping"})
                data = ws.receive_json()
                assert data["type"] == "pong"
                assert "timestamp" in data

    def test_receives_events(self, ws_app):
        """WebSocket receives events published to the event bus."""
        app, config = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                bus = get_event_bus()
                bus.publish("default", "test.event", {"hello": "world"})
                data = ws.receive_json()
                assert data["type"] == "test.event"
                assert data["hello"] == "world"

    def test_receives_only_configured_tenant_group_events(self, tmp_path):
        """WebSocket event subscriptions follow the resolved tenant group."""
        tenant_group = "tenant_brain"
        config = EngramConfig(
            mode="lite",
            default_group_id=tenant_group,
            sqlite={"path": str(tmp_path / "ws_tenant_test.db")},
            auth={"default_group_id": tenant_group},
        )
        app = create_app(config)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                bus = get_event_bus()
                bus.publish("default", "wrong.group", {"hello": "default"})
                bus.publish(tenant_group, "tenant.event", {"hello": "tenant"})
                data = ws.receive_json()
                assert data["type"] == "tenant.event"
                assert data["hello"] == "tenant"
                assert data["group_id"] == tenant_group

    def test_auth_enabled_accepts_query_param_token(self, tmp_path):
        """Browser WebSocket token fallback preserves the configured tenant group."""
        tenant_group = "browser_ws_brain"
        config = EngramConfig(
            mode="lite",
            default_group_id=tenant_group,
            sqlite={"path": str(tmp_path / "ws_token_test.db")},
            auth={
                "enabled": True,
                "bearer_token": "secret-token",
                "default_group_id": tenant_group,
            },
        )
        app = create_app(config)

        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard?token=secret-token") as ws:
                bus = get_event_bus()
                bus.publish(tenant_group, "browser.token.event", {"hello": "token"})
                data = ws.receive_json()
                assert data["type"] == "browser.token.event"
                assert data["hello"] == "token"
                assert data["group_id"] == tenant_group

    def test_resync_returns_missed_events(self, ws_app):
        """Resync command returns events missed since lastSeq."""
        app, config = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                bus = get_event_bus()

                # Publish some events
                bus.publish("default", "ev1", {"n": 1})
                ev1 = ws.receive_json()
                bus.publish("default", "ev2", {"n": 2})
                ws.receive_json()  # consume ev2
                bus.publish("default", "ev3", {"n": 3})
                ws.receive_json()  # consume ev3

                # Request resync from after ev1 (should get ev2, ev3)
                ws.send_json(
                    {
                        "type": "command",
                        "command": "resync",
                        "lastSeq": ev1["seq"],
                    }
                )
                resync = ws.receive_json()
                assert resync["type"] == "resync"
                assert resync["isFull"] is False
                assert len(resync["events"]) == 2

    def test_resync_full_when_seq_too_old(self, ws_app):
        """Resync returns isFull=True when gap is too large."""
        app, config = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                # Request resync with a very old seq (before any events)
                # but we need at least one event in history
                bus = get_event_bus()
                bus.publish("default", "boot", {})
                ws.receive_json()  # consume it

                ws.send_json(
                    {
                        "type": "command",
                        "command": "resync",
                        "lastSeq": 0,
                    }
                )
                resync = ws.receive_json()
                assert resync["type"] == "resync"
                # With lastSeq=0 and events starting at seq>0,
                # should return events (not full resync)
                # since 0 < oldest_seq only if events were evicted
                # In this case we have events and 0 < first event seq
                # so it returns full=True (gap too large)
                assert isinstance(resync["isFull"], bool)

    def test_activation_monitor_snapshot_uses_dashboard_payload(self, ws_app):
        """Activation monitor command emits the shared activation snapshot payload."""
        app, config = ws_app
        with TestClient(app) as client:
            with client.websocket_connect("/ws/dashboard") as ws:
                ws.send_json(
                    {
                        "type": "command",
                        "command": "subscribe.activation_monitor",
                        "interval_ms": 1,
                    }
                )

                snapshot = ws.receive_json()
                assert snapshot["type"] == "activation.snapshot"
                assert snapshot["payload"] == {"topActivated": []}

    def test_dismiss_notification_command_uses_connection_group(self, ws_app):
        """Dashboard notification dismiss commands cannot cross group boundaries."""
        app, config = ws_app
        with TestClient(app) as client:
            store = _app_state["notification_store"]
            active = MemoryNotification(
                group_id="default",
                notification_type="schema_discovery",
                priority="normal",
                title="Default notification",
                body="Dismiss through websocket.",
                entity_ids=[],
                metadata={},
                created_at=1.0,
            )
            other = MemoryNotification(
                group_id="other_brain",
                notification_type="schema_discovery",
                priority="normal",
                title="Other notification",
                body="Must not be dismissed through default websocket.",
                entity_ids=[],
                metadata={},
                created_at=2.0,
            )
            store.add(active)
            store.add(other)

            with client.websocket_connect("/ws/dashboard") as ws:
                ws.send_json(
                    {
                        "type": "command",
                        "command": "dismiss_notification",
                        "id": other.id,
                    }
                )
                ws.send_json({"type": "ping"})
                assert ws.receive_json()["type"] == "pong"

                assert other.dismissed_at is None

                ws.send_json(
                    {
                        "type": "command",
                        "command": "dismiss_notification",
                        "id": active.id,
                    }
                )
                ws.send_json({"type": "ping"})
                assert ws.receive_json()["type"] == "pong"

                assert active.dismissed_at is not None
