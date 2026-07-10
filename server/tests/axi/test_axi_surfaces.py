from __future__ import annotations

import pytest

from engram.axi.client import AxiRestError
from engram.axi.surfaces import (
    build_context_payload,
    build_doctor_payload,
    build_home_payload,
    build_packet_cache_clear_payload,
    build_packet_cache_summary_payload,
    build_recall_payload,
    build_storage_payload,
    build_value_payload,
    build_write_payload,
)


class FakeClient:
    server_url = "http://127.0.0.1:8100"

    def __init__(self, *, fail: set[str] | None = None, fresh_runtime: bool = False) -> None:
        self.fail = fail or set()
        self.observed: list[dict] = []
        self.remembered: list[dict] = []
        self.calls: list[str] = []
        self.timeouts: list[float] = []
        self.timeout_seconds = 10.0
        self.fresh_runtime = fresh_runtime
        self.bootstrapped: list[str] = []

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

    def runtime(
        self,
        *,
        project_path: str | None = None,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict:
        self.calls.append("runtime")
        self._maybe_fail("runtime")
        if live and project_path in self.bootstrapped:
            observed = 2
            return {
                "projectName": "Engram",
                "runtime": {"mode": "helix"},
                "artifactBootstrap": {
                    "projectPath": project_path,
                    "artifactCount": observed,
                    "lastObservedAt": "2026-06-30T15:00:00Z",
                },
                "agentAdoption": {
                    "status": "ready",
                    "requiredNextTools": ["get_context"],
                },
                "stats": {"packetCache": {"fresh_count": 1, "hit_count": 0}},
            }
        return {
            "projectName": "Engram",
            "runtime": {"mode": "helix"},
            "artifactBootstrap": {
                "projectPath": project_path,
                "artifactCount": 3,
                "lastObservedAt": "2026-01-01T00:00:00Z",
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
        if self.fresh_runtime and not self.bootstrapped:
            return {
                "projectName": "Engram",
                "runtime": {
                    "mode": "helix",
                    "surface": "fast_packet",
                    "loadedGraphTouched": False,
                },
                "artifactBootstrap": {
                    "projectPath": project_path,
                    "artifactCount": 0,
                    "lastObservedAt": None,
                },
                "agentAdoption": {
                    "status": "fresh_runtime",
                    "doNotTreatEmptyAsFailure": True,
                    "requiredNextTools": ["claim_authority", "get_context"],
                },
                "stats": {"packetCache": {"fresh_count": 0, "hit_count": 0}},
            }
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
                "lastObservedAt": "2026-01-01T00:00:00Z",
            },
            "agentAdoption": {
                "status": "ready",
                "requiredNextTools": ["get_context", "recall"],
            },
            "stats": {"packetCache": {"fresh_count": 2, "hit_count": 5}},
        }

    def bootstrap(
        self,
        *,
        project_path: str,
        include_patterns: list[str] | None = None,
    ) -> dict:
        self.calls.append("bootstrap")
        self._maybe_fail("bootstrap")
        self.bootstrapped.append(project_path)
        self.fresh_runtime = False
        return {"status": "ok", "observed": 2, "skipped": 0}

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

    def search_artifacts(
        self,
        query_text: str,
        *,
        project_path: str | None = None,
        limit: int = 5,
    ) -> dict:
        self.calls.append("search_artifacts")
        self._maybe_fail("search_artifacts")
        return {
            "items": [
                {
                    "path": "docs/design/extraction-rework.md",
                    "snippet": "Progressive Projection and Cue-First Memory",
                    "score": 0.8,
                }
            ]
        }

    def recall(
        self,
        query: str,
        *,
        limit: int,
        project_path: str | None = None,
    ) -> dict:
        self.calls.append("recall")
        self._maybe_fail("recall")
        if project_path:
            self.calls.append(f"recall_project:{project_path}")
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

    def evaluation_report(self, *, live_cost: bool = False) -> dict:
        self.calls.append("evaluation_report")
        if live_cost:
            self.calls.append("evaluation_report_live_cost")
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
                    "recent_problem_samples": [
                        {
                            "operation": "auto_recall_gate",
                            "source": "auto_recall",
                            "mode": "medium",
                            "status": "degraded",
                            "skip_reason": "recall_timeout",
                            "duration_ms": 75.0,
                            "timeout": True,
                        }
                    ],
                    "by_mode": {
                        "mcp_context": {
                            "operation_count": 3,
                            "p95_added_latency_ms": 20,
                            "cache_hit_count": 2,
                            "cache_miss_count": 1,
                            "status_counts": {"ok": 3},
                            "operation_counts": {"context": 3},
                            "source_counts": {"mcp_context": 3},
                        },
                        "api_auto_observe": {
                            "operation_count": 2,
                            "p95_added_latency_ms": 8000,
                            "skipped_count": 1,
                            "skip_reason_counts": {"background_capture": 1},
                        },
                    },
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

    def packet_cache(self) -> dict:
        self.calls.append("packet_cache")
        self._maybe_fail("packet_cache")
        return {
            "operation": "packet_cache.summary",
            "status": "ok",
            "packetCache": {
                "entryCount": 4,
                "freshCount": 3,
                "invalidatedCount": 1,
                "expiredCount": 0,
                "hitCount": 9,
                "scopes": {"project_home": 2, "session_recent": 1},
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


def test_home_payload_auto_bootstraps_fresh_runtime() -> None:
    client = FakeClient(fresh_runtime=True)

    result = build_home_payload(
        client,
        project_path="/tmp/engram-followup-test",
        topic_hint=None,
        budget=800,
    )

    assert result.exit_code == 0
    bootstrap = result.payload["bootstrap"]
    brain = result.payload["brain"]
    assert bootstrap["auto"] is True
    assert bootstrap["observed"] == 2
    assert brain["artifact_count"] == bootstrap["observed"]
    assert brain["artifact_status"] == "ready"
    assert brain["required_next_tools"] == ["get_context"]
    assert client.calls.count("runtime_fast") == 1
    assert client.calls.count("runtime") == 1
    assert "bootstrap" in client.calls


def test_bootstrap_then_live_runtime_matches_observed_artifact_count() -> None:
    """Canonical bootstrap path must return a live runtime consistent with observed files."""
    from engram.axi.surfaces import bootstrap_then_live_runtime

    client = FakeClient(fresh_runtime=True)
    summary, runtime = bootstrap_then_live_runtime(
        client,
        project_path="/tmp/engram-followup-test",
    )

    assert summary["observed"] == 2
    assert runtime["artifactBootstrap"]["artifactCount"] == summary["observed"]
    assert runtime["agentAdoption"]["status"] == "ready"
    assert runtime["agentAdoption"]["requiredNextTools"] == ["get_context"]


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
    assert result.payload["briefing"]
    assert "Memory growth:" in result.payload["briefing"]
    assert result.payload["artifactHits"]
    assert result.payload["injection"]["status"] == "ok"
    assert client.timeouts == [2.5, 8.0]
    assert set(client.calls) == {
        "health",
        "runtime_fast",
        "storage",
        "context",
        "search_artifacts",
    }


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
    assert '--project "$PWD"' in recall_cmd
    assert "--trace-origin agent-followup" in recall_cmd
    assert "--trace-origin" not in observe_cmd
    assert "--source codex" in observe_cmd


def test_home_payload_uses_trace_client_for_capture_source() -> None:
    result = build_home_payload(
        FakeClient(),
        project_path="/Users/konnermoshier/Engram",
        topic_hint=None,
        budget=800,
        trace_client="claude-code",
    )

    assert result.payload["next"][2]["cmd"] == ("engram axi observe --stdin --source claude-code")


def test_home_payload_uses_generic_capture_source_without_trace_client() -> None:
    result = build_home_payload(
        FakeClient(),
        project_path="/Users/konnermoshier/Engram",
        topic_hint=None,
        budget=800,
    )

    assert result.payload["next"][2]["cmd"] == "engram axi observe --stdin --source axi"


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


def test_context_payload_preserves_runtime_diagnostics() -> None:
    client = FakeClient()

    def degraded_context(**_kwargs) -> dict:
        return {
            "status": "degraded",
            "context": "## Cached Memory Packets\n- Project Home",
            "entityCount": 1,
            "factCount": 1,
            "tokenEstimate": 16,
            "format": "structured",
            "budget": {
                "profile": "explicit",
                "surface": "axi",
                "mode": "axi_context",
                "durationMs": 1201.0,
                "budgetMiss": True,
                "skipReason": "context_timeout",
            },
            "lifecycle": {
                "stage": "recall",
                "degraded": True,
                "skipReason": "context_timeout",
            },
            "diagnostics": {
                "stageTimingsMs": {
                    "packetCache": 0.5,
                    "projectFileFallback": 42.0,
                }
            },
        }

    client.context = degraded_context  # type: ignore[method-assign]

    result = build_context_payload(
        client,
        topic_hint="Engram",
        project_path="/Users/konnermoshier/Engram",
        budget=200,
    )

    assert result.payload["status"] == "degraded"
    assert result.payload["budget"]["skipReason"] == "context_timeout"
    assert result.payload["lifecycle"]["skipReason"] == "context_timeout"
    assert result.payload["diagnostics"]["stageTimingsMs"]["projectFileFallback"] == 42.0


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


def test_recall_payload_passes_project_path() -> None:
    client = FakeClient()

    result = build_recall_payload(
        client,
        query="Engram",
        limit=5,
        budget=200,
        project_path="/Users/konnermoshier/Engram",
    )

    assert result.exit_code == 0
    assert "recall_project:/Users/konnermoshier/Engram" in client.calls


def test_recall_payload_compacts_rest_items_shape() -> None:
    client = FakeClient()

    def recall_items(
        _query: str,
        *,
        limit: int,
        project_path: str | None = None,
    ) -> dict:
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


def test_recall_payload_includes_degraded_cached_packets() -> None:
    client = FakeClient()

    def recall_items(
        _query: str,
        *,
        limit: int,
        project_path: str | None = None,
    ) -> dict:
        return {
            "operation": "recall",
            "status": "degraded",
            "lifecycle": {
                "resultCount": 0,
                "packetCount": 1,
                "degraded": True,
                "timeout": True,
                "skipReason": "recall_timeout",
            },
            "diagnostics": {
                "stageTimingsMs": {
                    "packetCache": 0.4,
                    "recallSearch": 650.2,
                    "recallFallback": 91.3,
                }
            },
            "items": [],
            "packets": [
                {
                    "packet_type": "project_home",
                    "title": "Project Home: Engram",
                    "summary": "Cached project context survives timeout.",
                    "trust": {
                        "freshness": "recent",
                        "source": "cache",
                        "confidence": 0.8,
                        "why_now": "Cached project context for the current workspace.",
                    },
                }
            ][:limit],
        }

    client.recall = recall_items  # type: ignore[method-assign]

    result = build_recall_payload(
        client,
        query="Engram performance dogfood runtime",
        limit=5,
        budget=200,
    )

    assert result.payload["status"] == "degraded"
    assert result.payload["result_count"] == 0
    assert result.payload["packet_count"] == 1
    assert result.payload["packets"][0]["title"] == "Project Home: Engram"
    assert result.payload["packets"][0]["summary"] == "Cached project context survives timeout."
    assert result.payload["packets"][0]["trust"]["source"] == "cache"
    assert result.payload["diagnostics"]["stageTimingsMs"]["recallSearch"] == 650.2


def test_storage_payload_includes_paths() -> None:
    client = FakeClient()
    result = build_storage_payload(client)

    assert result.payload["operation"] == "storage"
    assert result.payload["backend"] == "helix_native"
    assert result.payload["paths"][0]["path"] == "/tmp/helix"
    assert result.payload["counts"]["episodes"] == 2


def test_value_payload_compacts_memory_value_report() -> None:
    client = FakeClient()
    result = build_value_payload(client)

    assert result.exit_code == 0
    assert result.payload["operation"] == "value"
    assert result.payload["status"] == "measured"
    assert result.payload["cost"]["source"] == "live_runtime"
    assert result.payload["cost"]["p95_added_latency_ms"] == 42
    assert result.payload["cost"]["skipped_count"] == 3
    assert result.payload["cost"]["read_path"]["operation_count"] == 3
    assert result.payload["cost"]["read_path"]["p95_added_latency_ms"] == 20
    assert result.payload["cost"]["read_path"]["cache_hit_rate"] == 0.6667
    assert result.payload["cost"]["recent_problem_samples"][0]["mode"] == "medium"
    assert result.payload["cost"]["mode_breakdown"]["mcp_context"]["operation_counts"] == {
        "context": 3
    }
    assert result.payload["cost"]["mode_breakdown"]["api_auto_observe"]["skip_reason_counts"] == {
        "background_capture": 1
    }
    assert result.payload["cost"]["write_path"]["operation_count"] == 2
    assert result.payload["cost"]["write_path"]["p95_added_latency_ms"] == 8000
    assert result.payload["cost"]["top_modes_by_p95"][0]["mode"] == "api_auto_observe"
    assert result.payload["benefit"]["continuity_lift"] == 0.25
    assert result.payload["next"][0]["cmd"].startswith("engram axi context")
    assert "evaluation_report_live_cost" in client.calls


def test_value_payload_timeout_suggests_value_report_timeout() -> None:
    client = FakeClient()

    def timeout_report(*, live_cost: bool = False) -> dict:
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
    assert result.payload["next"][0]["cmd"] == 'engram axi context --project "$PWD" --timeout 10'


def test_packet_cache_summary_payload_reports_cache_diagnostics() -> None:
    result = build_packet_cache_summary_payload(FakeClient())

    assert result.exit_code == 0
    assert result.payload["operation"] == "packet-cache.summary"
    assert result.payload["status"] == "ok"
    assert result.payload["packet_cache"] == {
        "entry_count": 4,
        "fresh_count": 3,
        "invalidated_count": 1,
        "expired_count": 0,
        "hit_count": 9,
        "scopes": {"project_home": 2, "session_recent": 1},
        "persistent": True,
        "path": "/tmp/engram-packet-cache.sqlite3",
    }
    assert result.payload["next"][0]["cmd"] == ('engram axi context --project "$PWD" --timeout 10')
    assert result.payload["next"][1]["cmd"] == "engram axi packet-cache clear"


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
