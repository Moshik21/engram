from __future__ import annotations

from starlette.testclient import TestClient

from engram.config import EngramConfig
from engram.main import create_app
from engram.mcp import server as mcp_server_module


def test_rest_app_serves_streamable_http_mcp_at_advertised_path(monkeypatch, tmp_path):
    db_path = tmp_path / "engram-mcp-http.db"
    monkeypatch.setenv("ENGRAM_MODE", "lite")
    monkeypatch.setenv("ENGRAM_SQLITE__PATH", str(db_path))
    monkeypatch.setenv("ENGRAM_EMBEDDING__PROVIDER", "noop")
    monkeypatch.setenv("ENGRAM_MCP_ENABLED", "1")
    toggle_calls: list[bool] = []
    original_set_background_runtime = mcp_server_module.set_background_runtime_managed_externally

    def record_background_runtime_owner(enabled: bool = True) -> None:
        toggle_calls.append(enabled)
        original_set_background_runtime(enabled)

    monkeypatch.setattr(
        mcp_server_module,
        "set_background_runtime_managed_externally",
        record_background_runtime_owner,
    )
    init_calls: list[str] = []

    async def fail_if_lazy_mcp_runtime_starts() -> None:
        init_calls.append("init")
        raise AssertionError("mounted MCP should reuse the REST runtime")

    monkeypatch.setattr(
        mcp_server_module,
        "_init",
        fail_if_lazy_mcp_runtime_starts,
    )

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
    assert toggle_calls[0] is True
    assert toggle_calls[-1] is False
    assert init_calls == []
    assert mcp_server_module._background_runtime_managed_externally is False
    assert mcp_server_module._external_runtime_attached is False
