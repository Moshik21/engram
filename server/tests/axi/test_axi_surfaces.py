from __future__ import annotations

import pytest

from engram.axi.client import AxiRestError
from engram.axi.surfaces import (
    build_context_payload,
    build_doctor_payload,
    build_home_payload,
    build_packet_cache_clear_payload,
    build_recall_payload,
    build_storage_payload,
    build_value_payload,
    build_write_payload,
)


class FakeClient:
    server_url = "http://127.0.0.1:8100"

    def __init__(self, *, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.observed: list[dict] = []
        self.remembered: list[dict] = []
        self.calls: list[str] = []
        self.timeouts: list[float] = []
        self.timeout_seconds = 10.0

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail or (name == "runtime_fast" and "runtime" in self.fail):
            raise AxiRestError(f"{name} failed", url=f"{self.server_url}/{name}")

    def with_timeout(self, timeout_seconds: float):
        self.timeouts.append(timeout_seconds)
        return self

    def health(self) -> dict:
        self.calls.append("health")
        self._maybe_fail("health")
        return {"status": "healthy", "mode": "helix"}

    def runtime(self, *, project_path: str | None = None) -> dict:
        self.calls.append("runtime")
        self._maybe_fail("runtime")
        return {
            "projectName": "Engram",
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "projectPath": project_path,
                "artifactCount": 3,
            },
            "agentAdoption": {
                "status": "ready",
                "requiredNextTools": ["get_context", "recall"],
            },
            "stats": {"packetCache": {"fresh_count": 2, "hit_count": 5}},
        }

    def runtime_fast(self, *, project_path: str | None = None) -> dict:
        self.calls.append("runtime_fast")
        self._maybe_fail("runtime_fast")
        return {
            "projectName": "Engram",
            "runtime": {
                "mode": "helix",
                "surface": "fast_packet",
                "loadedGraphTouched": False,
            },
            "artifactBootstrap": {
                "projectPath": project_path,
                "artifactCount": 3,
            },
            "agentAdoption": {
                "status": "ready",
                "requiredNextTools": ["get_context", "recall"],
            },
            "stats": {"packetCache": {"fresh_count": 2, "hit_count": 5}},
        }

    def storage(
        self,
        *,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict:
        self.calls.append("storage")
        self._maybe_fail("storage")
        return {
            "mode": "helix",
            "backend": "helix_native",
            "counts": {
                "episodes": 2,
                "entities": 4,
                "relationships": 3,
                "cues": 2,
            },
            "disk": {"humanSize": "10.0 MB"},
            "paths": [
                {
                    "label": "Helix native data",
                    "path": "/tmp/helix",
                    "exists": True,
                    "humanSize": "10.0 MB",
                }
            ],
            "diagnostics": {
                "live": live,
                "countsStatus": "live" if live else "cached",
                "countsAgeSeconds": 0,
                "pathsStatus": "live" if live else "cached",
                "pathsAgeSeconds": 0,
                "timeoutSeconds": timeout_seconds,
            },
        }

    def context(
        self,
        *,
        max_tokens: int,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict:
        self.calls.append("context")
        self._maybe_fail("context")
        return {
            "context": "## Active Memory\nEngram owns portable cross-context memory.",
            "entityCount": 1,
            "factCount": 2,
            "tokenEstimate": 16,
            "format": format,
        }

    def recall(self, query: str, *, limit: int) -> dict:
        self.calls.append("recall")
        self._maybe_fail("recall")
        return {
            "status": "ok",
            "results": [
                {
                    "result_type": "entity",
                    "name": "Engram",
                    "summary": "Portable memory for agents.",
                    "score": 0.9,
                }
            ][:limit],
        }

    def evaluation_report(self) -> dict:
        self.calls.append("evaluation_report")
        self._maybe_fail("evaluation_report")
        return {
            "memory_value": {
                "status": "measured",
                "cost": {
                    "operation_count": 12,
                    "p95_added_latency_ms": 42,
                    "cache_hit_rate": 0.5,
                    "budget_miss_rate": 0.1,
                    "skipped_count": 3,
                },
                "benefit": {
                    "memory_need_precision": 0.75,
                    "useful_packet_rate": 0.6,
                    "session_continuity_lift": 0.25,
                },
            }
        }

    def clear_packet_cache(self) -> dict:
        self.calls.append("clear_packet_cache")
        self._maybe_fail("clear_packet_cache")
        return {
            "operation": "packet_cache.clear",
            "status": "cleared",
            "clearedCount": 3,
            "packetCache": {
                "entryCount": 0,
                "freshCount": 0,
                "hitCount": 7,
                "persistent": True,
                "path": "/tmp/engram-packet-cache.sqlite3",
            },
        }

    def observe(self, **kwargs) -> dict:
        self.calls.append("observe")
        self._maybe_fail("observe")
        self.observed.append(kwargs)
        return {
            "status": "stored",
            "episode_id": "ep_1",
            "lifecycle": {
                "capture_status": "stored",
                "projection_mode": "background",
                "projection_status": "queued",
            },
        }

    def remember(self, **kwargs) -> dict:
        self.calls.append("remember")
        self._maybe_fail("remember")
        self.remembered.append(kwargs)
        return {"status": "stored", "episode_id": "ep_2"}


def test_home_payload_compacts_healthy_runtime() -> None:
    client = FakeClient()

    result = build_home_payload(
        client,
        project_path="/Users/konnermoshier/Engram",
        topic_hint="AXI",
        budget=800,
    )

    assert result.exit_code == 0
    assert result.payload["status"] == "healthy"
    assert result.payload["mode"] == "helix"
    assert result.payload["transport"] == "native"
    assert result.payload["storage"]["data_dir"] == "/tmp/helix"
    assert result.payload["brain"]["artifact_status"] == "ready"
    assert result.payload["brain"]["packet_cache"] == {
        "status": "warm",
        "fresh": 2,
        "hits": 5,
    }
    assert result.payload["context"]["status"] == "available"
    assert result.payload["context"]["cmd"].startswith("engram axi context")
    assert result.payload["next"][0]["cmd"].startswith("engram axi context")
    assert client.timeouts == [2.5]
    assert set(client.calls) == {"health", "runtime_fast", "storage"}


def test_home_payload_adds_metadata_trace_flags_to_read_followups() -> None:
    client = FakeClient()

    result = build_home_payload(
        client,
        project_path="/Users/konnermoshier/Engram",
        topic_hint="AXI",
        budget=800,
        trace_file="/tmp/axi-runs.jsonl",
        trace_client="codex",
        followup_trace_origin="agent-followup",
    )

    context_cmd = result.payload["context"]["cmd"]
    recall_cmd = result.payload["next"][1]["cmd"]
    observe_cmd = result.payload["next"][2]["cmd"]
    assert "--trace-file /tmp/axi-runs.jsonl" in context_cmd
    assert "--trace-client codex" in context_cmd
    assert "--trace-origin agent-followup" in context_cmd
    assert "--timeout 10" in context_cmd
    assert "--trace-origin agent-followup" in recall_cmd
    assert "--trace-origin" not in observe_cmd


def test_home_payload_degrades_when_rest_is_offline() -> None:
    result = build_home_payload(
        FakeClient(fail={"health", "runtime", "storage"}),
        project_path=None,
        topic_hint=None,
        budget=800,
    )

    assert result.exit_code == 0
    assert result.payload["status"] == "offline"
    assert result.payload["error"] == "health failed"
    assert result.payload["next"][0]["cmd"] == "engramctl start"


def test_home_payload_reports_busy_runtime_timeout_as_degraded() -> None:
    client = FakeClient()

    def slow_health() -> dict:
        raise AxiRestError("Engram REST request timed out after 0.75s")

    client.health = slow_health  # type: ignore[method-assign]

    result = build_home_payload(
        client,
        project_path=None,
        topic_hint=None,
        budget=800,
    )

    assert result.exit_code == 0
    assert result.payload["status"] == "degraded"
    assert result.payload["mode"] == "helix"
    assert result.payload["context"]["cmd"] == "engram axi context --budget 800 --timeout 10"
    assert result.payload["next"][0]["cmd"] == "engram axi context --budget 800 --timeout 10"
    assert result.payload["error"] == "Engram REST request timed out after 0.75s"


def test_context_payload_respects_budget() -> None:
    client = FakeClient()

    def long_context(**_kwargs) -> dict:
        return {
            "context": "memory " * 500,
            "entityCount": 1,
            "factCount": 1,
            "tokenEstimate": 500,
            "format": "structured",
        }

    client.context = long_context  # type: ignore[method-assign]

    result = build_context_payload(
        client,
        topic_hint="Engram",
        project_path=None,
        budget=20,
    )

    assert result.payload["truncated"] is True
    assert "[truncated;" in result.payload["context"]
    assert result.payload["next"][0]["cmd"] == "engram axi context --full"


def test_context_payload_includes_cached_packet_trust() -> None:
    client = FakeClient()

    def cached_context(**_kwargs) -> dict:
        return {
            "context": "## Cached Memory Packets\n- Project Home",
            "entityCount": 1,
            "factCount": 1,
            "tokenEstimate": 16,
            "format": "structured",
            "packet_cache": {
                "hit": True,
                "packet_count": 1,
                "scopes": {"project_home": 1},
            },
            "cached_packets": [
                {
                    "packet_type": "project_home",
                    "title": "Project Home: Engram",
                    "summary": "Cached project context.",
                    "trust": {
                        "freshness": "recent",
                        "source": "cache",
                        "confidence": 0.8,
                        "why_now": "Cached project context for the current workspace.",
                        "provenance_count": 1,
                        "evidence_count": 2,
                        "belief_status": "unknown",
                    },
                }
            ],
        }

    client.context = cached_context  # type: ignore[method-assign]

    result = build_context_payload(
        client,
        topic_hint="Engram",
        project_path="/Users/konnermoshier/Engram",
        budget=200,
    )

    assert result.payload["packet_cache"] == {
        "hit": True,
        "packet_count": 1,
        "scopes": {"project_home": 1},
    }
    assert result.payload["packets"][0]["trust"] == {
        "freshness": "recent",
        "source": "cache",
        "confidence": 0.8,
        "why": "Cached project context for the current workspace.",
        "provenance": 1,
        "evidence": 2,
        "belief": "unknown",
        "confirmed": 0,
        "corrected": 0,
        "dismissed": 0,
        "last_confirmed": None,
        "last_corrected": None,
    }


def test_recall_payload_compacts_results() -> None:
    result = build_recall_payload(
        FakeClient(),
        query="Engram",
        limit=5,
        budget=200,
    )

    assert result.exit_code == 0
    assert result.payload["operation"] == "recall"
    assert result.payload["result_count"] == 1
    assert result.payload["results"][0]["name"] == "Engram"


def test_recall_payload_compacts_rest_items_shape() -> None:
    client = FakeClient()

    def recall_items(_query: str, *, limit: int) -> dict:
        return {
            "operation": "recall",
            "status": "degraded",
            "lifecycle": {
                "resultCount": 2,
                "degraded": True,
                "timeout": True,
                "skipReason": "recall_timeout",
            },
            "budget": {
                "profile": "explicit",
                "surface": "axi",
                "mode": "axi_recall",
                "maxWallMs": 2000,
                "maxSearchMs": 1200,
                "durationMs": 1201.0,
                "timeout": True,
                "degraded": True,
            },
            "items": [
                {
                    "resultType": "entity",
                    "entity": {"name": "AXI", "summary": "Agent interface"},
                    "score": 0.9,
                },
                {
                    "resultType": "episode",
                    "episode": {"source": "codex", "content": "AXI planning notes"},
                    "score": 0.8,
                },
            ][:limit],
        }

    client.recall = recall_items  # type: ignore[method-assign]

    result = build_recall_payload(
        client,
        query="AXI",
        limit=2,
        budget=200,
    )

    assert result.payload["result_count"] == 2
    assert result.payload["status"] == "degraded"
    assert result.payload["lifecycle"]["skipReason"] == "recall_timeout"
    assert result.payload["budget"]["surface"] == "axi"
    assert result.payload["results"][0]["type"] == "entity"
    assert result.payload["results"][0]["name"] == "AXI"
    assert result.payload["results"][0]["text"] == "Agent interface"
    assert result.payload["results"][1]["type"] == "episode"
    assert result.payload["results"][1]["name"] == "codex"
    assert result.payload["results"][1]["text"] == "AXI planning notes"


def test_storage_payload_includes_paths() -> None:
    client = FakeClient()
    result = build_storage_payload(client)

    assert result.payload["operation"] == "storage"
    assert result.payload["backend"] == "helix_native"
    assert result.payload["paths"][0]["path"] == "/tmp/helix"
    assert result.payload["counts"]["episodes"] == 2


def test_value_payload_compacts_memory_value_report() -> None:
    result = build_value_payload(FakeClient())

    assert result.exit_code == 0
    assert result.payload["operation"] == "value"
    assert result.payload["status"] == "measured"
    assert result.payload["cost"]["p95_added_latency_ms"] == 42
    assert result.payload["cost"]["skipped_count"] == 3
    assert result.payload["benefit"]["continuity_lift"] == 0.25
    assert result.payload["next"][0]["cmd"].startswith("engram axi context")


def test_value_payload_timeout_suggests_value_report_timeout() -> None:
    client = FakeClient()

    def timeout_report() -> dict:
        raise AxiRestError(
            "Engram REST request timed out after 10s",
            url="http://127.0.0.1:8100/api/evaluation/brain-loop/report",
        )

    client.evaluation_report = timeout_report  # type: ignore[method-assign]

    result = build_value_payload(client)

    assert result.exit_code == 1
    assert result.payload["status"] == "error"
    assert result.payload["next"][0]["cmd"] == "engram axi value --timeout 20"


def test_packet_cache_clear_payload_reports_post_clear_summary() -> None:
    result = build_packet_cache_clear_payload(FakeClient())

    assert result.exit_code == 0
    assert result.payload["operation"] == "packet-cache.clear"
    assert result.payload["status"] == "cleared"
    assert result.payload["cleared_count"] == 3
    assert result.payload["packet_cache"] == {
        "entry_count": 0,
        "fresh_count": 0,
        "hit_count": 7,
        "persistent": True,
        "path": "/tmp/engram-packet-cache.sqlite3",
    }
    assert (
        result.payload["next"][0]["cmd"]
        == 'engram axi context --project "$PWD" --timeout 10'
    )


def test_doctor_payload_fails_when_required_probe_fails() -> None:
    result = build_doctor_payload(FakeClient(fail={"storage"}))

    assert result.exit_code == 1
    assert result.payload["status"] == "fail"
    assert {check["name"]: check["status"] for check in result.payload["checks"]} == {
        "health": "pass",
        "runtime": "pass",
        "storage": "fail",
    }


@pytest.mark.parametrize("operation", ["observe", "remember"])
def test_write_payload_uses_explicit_stdin_content(operation: str) -> None:
    client = FakeClient()

    result = build_write_payload(
        client,
        operation=operation,
        content="Remember this.",
        source="codex",
        conversation_date=None,
    )

    assert result.exit_code == 0
    assert result.payload["operation"] == operation
    assert result.payload["status"] == "stored"
    if operation == "observe":
        assert client.observed[0]["content"] == "Remember this."
    else:
        assert client.remembered[0]["source"] == "codex"
