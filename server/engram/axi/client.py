"""Bounded REST client used by AXI commands."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

DEFAULT_SERVER_URL = "http://127.0.0.1:8100"
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass
class AxiRestError(Exception):
    """A REST call failed in a way AXI should report compactly."""

    message: str
    status: int | None = None
    url: str | None = None

    def __str__(self) -> str:
        if self.status is not None:
            return f"{self.message} (status={self.status})"
        return self.message


class AxiRestClient:
    """Small sync HTTP client for local Engram REST endpoints."""

    def __init__(
        self,
        *,
        server_url: str | None = None,
        timeout_seconds: float | None = None,
        auth_token: str | None = None,
    ) -> None:
        resolved_url = server_url or os.environ.get("ENGRAM_AXI_SERVER_URL") or DEFAULT_SERVER_URL
        self.server_url = resolved_url.rstrip("/")
        self.timeout_seconds = timeout_seconds or DEFAULT_TIMEOUT_SECONDS
        self.auth_token = (
            auth_token if auth_token is not None else os.environ.get("ENGRAM_AUTH__BEARER_TOKEN")
        )

    def health(self) -> dict[str, Any]:
        return self.request_json("GET", "/health")

    def with_timeout(self, timeout_seconds: float) -> AxiRestClient:
        """Return a client for the same runtime with a different request timeout."""
        return AxiRestClient(
            server_url=self.server_url,
            timeout_seconds=timeout_seconds,
            auth_token=self.auth_token,
        )

    def runtime(
        self,
        *,
        project_path: str | None = None,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if project_path:
            query["project_path"] = project_path
        if live:
            query["live"] = True
        if timeout_seconds is not None:
            query["timeoutSeconds"] = timeout_seconds
        return self.request_json(
            "GET",
            "/api/knowledge/runtime",
            query=query or None,
        )

    def runtime_fast(self, *, project_path: str | None = None) -> dict[str, Any]:
        query = {"project_path": project_path} if project_path else None
        return self.request_json("GET", "/api/knowledge/runtime/fast", query=query)

    def storage(
        self,
        *,
        live: bool = False,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] | None = None
        if live or timeout_seconds is not None:
            query = {"live": live, "timeoutSeconds": timeout_seconds}
        return self.request_json("GET", "/api/storage", query=query)

    def context(
        self,
        *,
        max_tokens: int,
        topic_hint: str | None = None,
        project_path: str | None = None,
        format: str = "structured",
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"max_tokens": max_tokens, "format": format}
        if topic_hint:
            query["topic_hint"] = topic_hint
        if project_path:
            query["project_path"] = project_path
        return self.request_json("GET", "/api/knowledge/context", query=query)

    def search_artifacts(
        self,
        query_text: str,
        *,
        project_path: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"q": query_text, "limit": limit}
        if project_path:
            query["project_path"] = project_path
        return self.request_json("GET", "/api/knowledge/artifacts/search", query=query)

    def recall(
        self,
        query_text: str,
        *,
        limit: int,
        project_path: str | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {"q": query_text, "limit": limit}
        if project_path:
            query["project_path"] = project_path
        return self.request_json(
            "GET",
            "/api/knowledge/recall",
            query=query,
        )

    def evaluation_report(
        self,
        *,
        live_cost: bool = False,
        cycle_limit: int | None = None,
        sample_limit: int | None = None,
    ) -> dict[str, Any]:
        query: dict[str, Any] = {}
        if live_cost:
            query["liveCost"] = True
        if cycle_limit is not None:
            query["cycleLimit"] = cycle_limit
        if sample_limit is not None:
            query["sampleLimit"] = sample_limit
        return self.request_json("GET", "/api/evaluation/brain-loop/report", query=query)

    def clear_packet_cache(self) -> dict[str, Any]:
        return self.request_json("POST", "/api/knowledge/packet-cache/clear")

    def packet_cache(self) -> dict[str, Any]:
        return self.request_json("GET", "/api/knowledge/packet-cache")

    def observe(
        self,
        *,
        content: str,
        source: str,
        conversation_date: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content, "source": source}
        if conversation_date:
            body["conversation_date"] = conversation_date
        return self.request_json("POST", "/api/knowledge/observe", body=body)

    def remember(
        self,
        *,
        content: str,
        source: str,
        conversation_date: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content, "source": source}
        if conversation_date:
            body["conversation_date"] = conversation_date
        return self.request_json("POST", "/api/knowledge/remember", body=body)

    def bootstrap(
        self,
        *,
        project_path: str,
        include_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"project_path": project_path}
        if include_patterns:
            body["include_patterns"] = include_patterns
        return self.request_json("POST", "/api/knowledge/bootstrap", body=body)

    def request_json(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = self._url(path, query=query)
        data = None
        headers = {"Accept": "application/json", "X-Engram-Client": "axi"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise AxiRestError(
                _http_error_message(raw_error) or exc.reason or "Engram REST request failed",
                status=exc.code,
                url=url,
            ) from exc
        except TimeoutError as exc:
            raise AxiRestError(
                f"Engram REST request timed out after {self.timeout_seconds:g}s",
                url=url,
            ) from exc
        except urllib.error.URLError as exc:
            if _is_timeout_reason(exc.reason):
                raise AxiRestError(
                    f"Engram REST request timed out after {self.timeout_seconds:g}s",
                    url=url,
                ) from exc
            raise AxiRestError(
                f"Engram REST API is unreachable at {self.server_url}",
                url=url,
            ) from exc
        except OSError as exc:
            raise AxiRestError(
                f"Engram REST API is unreachable at {self.server_url}",
                url=url,
            ) from exc

        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError as exc:
            raise AxiRestError("Engram REST API returned malformed JSON", url=url) from exc
        if not isinstance(payload, dict):
            raise AxiRestError("Engram REST API returned a non-object JSON payload", url=url)
        return payload

    def _url(self, path: str, *, query: dict[str, Any] | None = None) -> str:
        url = f"{self.server_url}/{path.lstrip('/')}"
        if query:
            clean_query = {key: value for key, value in query.items() if value is not None}
            url = f"{url}?{urllib.parse.urlencode(clean_query)}"
        return url


def _http_error_message(raw: str) -> str:
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip()
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if detail:
            return json.dumps(detail, ensure_ascii=False)
    return raw.strip()


def _is_timeout_reason(reason: Any) -> bool:
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(reason).lower()
