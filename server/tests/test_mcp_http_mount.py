from __future__ import annotations

from starlette.testclient import TestClient

from engram.config import EngramConfig
from engram.main import create_app


def test_rest_app_serves_streamable_http_mcp_at_advertised_path(monkeypatch, tmp_path):
    db_path = tmp_path / "engram-mcp-http.db"
    monkeypatch.setenv("ENGRAM_MODE", "lite")
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(db_path))
    monkeypatch.setenv("ENGRAM_EMBEDDING__PROVIDER", "noop")
    monkeypatch.setenv("ENGRAM_MCP_ENABLED", "1")

    app = create_app(EngramConfig(_env_file=None))
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "mount-smoke", "version": "0"},
        },
    }

    with TestClient(app, base_url="http://127.0.0.1:8100") as client:
        health_response = client.get("/mcp", follow_redirects=False)
        response = client.post(
            "/mcp",
            json=payload,
            headers={"Accept": "application/json, text/event-stream"},
            follow_redirects=False,
        )

    assert health_response.status_code == 200
    assert health_response.json() == {
        "status": "ok",
        "transport": "streamable-http",
        "path": "/mcp",
    }
    assert response.status_code == 200
    assert response.headers.get("location") is None
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"serverInfo":{"name":"engram"' in response.text
