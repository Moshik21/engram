from __future__ import annotations

import urllib.error

import pytest

from engram.axi.client import AxiRestClient, AxiRestError


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def close(self) -> None:
        return None


def test_rest_client_parses_json_response(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout: float):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["auth"] = request.headers.get("Authorization")
        seen["client"] = (
            request.get_header("X-Engram-Client")
            or request.get_header("X-engram-client")
        )
        return FakeResponse(b'{"status":"healthy"}')

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(
        server_url="http://localhost:8100/",
        timeout_seconds=1.5,
        auth_token="secret",
    )
    payload = client.request_json("GET", "/health", query={"q": "Engram AXI"})

    assert payload == {"status": "healthy"}
    assert seen["url"] == "http://localhost:8100/health?q=Engram+AXI"
    assert seen["timeout"] == 1.5
    assert seen["auth"] == "Bearer secret"
    assert seen["client"] == "axi"


def test_rest_client_can_clone_with_shorter_timeout() -> None:
    client = AxiRestClient(
        server_url="http://localhost:8100/",
        timeout_seconds=10,
        auth_token="secret",
    )

    clone = client.with_timeout(0.75)

    assert clone.server_url == "http://localhost:8100"
    assert clone.timeout_seconds == 0.75
    assert clone.auth_token == "secret"


def test_rest_client_clears_packet_cache(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout: float):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["data"] = request.data
        return FakeResponse(b'{"status":"cleared","clearedCount":2}')

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    payload = client.clear_packet_cache()

    assert payload == {"status": "cleared", "clearedCount": 2}
    assert seen["url"] == "http://localhost:8100/api/knowledge/packet-cache/clear"
    assert seen["method"] == "POST"
    assert seen["data"] is None


def test_rest_client_requests_fast_runtime_packet(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout: float):
        seen["url"] = request.full_url
        return FakeResponse(b'{"runtime":{"surface":"fast_packet"}}')

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    payload = client.runtime_fast(project_path="/tmp/Engram")

    assert payload == {"runtime": {"surface": "fast_packet"}}
    assert (
        seen["url"]
        == "http://localhost:8100/api/knowledge/runtime/fast?project_path=%2Ftmp%2FEngram"
    )


def test_rest_client_can_request_live_storage(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout: float):
        seen["url"] = request.full_url
        return FakeResponse(b'{"backend":"helix_native"}')

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    payload = client.storage(live=True, timeout_seconds=3)

    assert payload == {"backend": "helix_native"}
    assert seen["url"] == "http://localhost:8100/api/storage?live=True&timeoutSeconds=3"


def test_rest_client_can_request_live_cost_evaluation_report(monkeypatch) -> None:
    seen = {}

    def fake_urlopen(request, timeout: float):
        seen["url"] = request.full_url
        return FakeResponse(b'{"memory_value":{"status":"measured"}}')

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    payload = client.evaluation_report(live_cost=True)

    assert payload == {"memory_value": {"status": "measured"}}
    assert seen["url"] == "http://localhost:8100/api/evaluation/brain-loop/report?liveCost=True"


def test_rest_client_rejects_malformed_json(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: float):
        return FakeResponse(b"not json")

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    with pytest.raises(AxiRestError, match="malformed JSON"):
        client.health()


def test_rest_client_formats_http_errors(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: float):
        raise urllib.error.HTTPError(
            "http://localhost:8100/api",
            401,
            "Unauthorized",
            hdrs=None,
            fp=FakeResponse(b'{"detail":"auth required"}'),
        )

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=1)
    with pytest.raises(AxiRestError) as exc:
        client.health()

    assert exc.value.status == 401
    assert exc.value.message == "auth required"


def test_rest_client_reports_timeout_separately(monkeypatch) -> None:
    def fake_urlopen(_request, timeout: float):
        raise TimeoutError("timed out")

    monkeypatch.setattr("engram.axi.client.urllib.request.urlopen", fake_urlopen)

    client = AxiRestClient(server_url="http://localhost:8100", timeout_seconds=2)
    with pytest.raises(AxiRestError) as exc:
        client.health()

    assert exc.value.message == "Engram REST request timed out after 2s"
