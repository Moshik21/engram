from __future__ import annotations

from types import SimpleNamespace


def test_mcp_runtime_builds_reranker_from_activation_config(monkeypatch) -> None:
    from engram.mcp import server as mcp_server

    calls: list[dict[str, object]] = []
    reranker = object()

    def fake_create_reranker(**kwargs):
        calls.append(kwargs)
        return reranker

    monkeypatch.setenv("COHERE_API_KEY", "cohere-test-key")
    monkeypatch.setattr("engram.retrieval.reranker.create_reranker", fake_create_reranker)

    config = SimpleNamespace(
        activation=SimpleNamespace(
            reranker_enabled=True,
            reranker_provider="local",
            reranker_local_model="local-cross-encoder",
        ),
    )

    assert mcp_server._create_configured_reranker(config) is reranker
    assert calls == [
        {
            "api_key": "cohere-test-key",
            "provider": "local",
            "local_model": "local-cross-encoder",
        }
    ]


def test_mcp_runtime_leaves_reranker_unbuilt_when_disabled() -> None:
    from engram.mcp import server as mcp_server

    config = SimpleNamespace(
        activation=SimpleNamespace(
            reranker_enabled=False,
            reranker_provider="local",
            reranker_local_model="local-cross-encoder",
        ),
    )

    assert mcp_server._create_configured_reranker(config) is None
