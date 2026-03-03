# 01 - Configuration Schema & Database Indexes

## Overview

This document defines the complete configuration model for Engram and the FalkorDB index DDL required for performant graph operations. All configuration is managed through a single Pydantic Settings model that supports environment variables, `.env` files, and a YAML config file with layered overrides.

---

## 1. Environment Variable Naming Convention

Engram supports two env var naming styles that coexist:

### 1.1 Pydantic Nested Style (canonical)

Used by `pydantic-settings` for automatic field mapping. Double-underscore (`__`) delimits nesting:

```
ENGRAM_<SECTION>__<FIELD>=value
```

Examples:
```bash
ENGRAM_CLAUDE__API_KEY=sk-ant-...
ENGRAM_REDIS__HOST=localhost
ENGRAM_FALKORDB__PORT=6380
ENGRAM_ACTIVATION__DECAY_EXPONENT=0.5
ENGRAM_SERVER__LOG_LEVEL=debug
```

### 1.2 Flat Style (Docker/DevOps convenience)

For Docker Compose and `.env` files, flat single-underscore names are also accepted.
These are mapped to their nested equivalents by explicit `validation_alias` on each
field. See 08_devops_infrastructure.md Section 7 for the full reference table.

| Flat env var | Maps to config field |
|---|---|
| `ENGRAM_ANTHROPIC_API_KEY` | `claude.api_key` |
| `ENGRAM_FALKORDB_URL` | `falkordb.url` (parsed into host/port/password) |
| `ENGRAM_FALKORDB_PASSWORD` | `falkordb.password` |
| `ENGRAM_FALKORDB_PORT` | `falkordb.port` |
| `ENGRAM_REDIS_URL` | `redis.url` (parsed into host/port/db/password) |
| `ENGRAM_REDIS_PASSWORD` | `redis.password` |
| `ENGRAM_REDIS_PORT` | `redis.port` |
| `ENGRAM_SERVER_PORT` | `server.port` |
| `ENGRAM_SERVER_HOST` | `server.host` |
| `ENGRAM_AUTH_ENABLED` | `auth.enabled` |
| `ENGRAM_AUTH_SECRET` | `auth.bearer_token` |
| `ENGRAM_MASTER_KEY` | `encryption.master_key` |
| `ENGRAM_JWT_SECRET` | `auth.jwt.secret` |
| `ENGRAM_LOG_LEVEL` | `server.log_level` |
| `ENGRAM_ENV` | `server.env` |
| `ENGRAM_EMBEDDING_API_KEY` | `embedding.api_key` |
| `ENGRAM_MODE` | `mode` |
| `ENGRAM_SQLITE_PATH` | `sqlite.path` |

Both styles work simultaneously. If both are set, the nested style takes precedence
(Pydantic resolves it first).

### 1.3 Third-party keys

The standard `ANTHROPIC_API_KEY` (without `ENGRAM_` prefix) is also accepted as a
fallback for `claude.api_key`. This avoids forcing users to rename a key they
already have set.

### 1.4 Precedence (highest to lowest)

1. Environment variables (nested `ENGRAM_X__Y` style)
2. Environment variables (flat `ENGRAM_X_Y` style)
3. Environment variables (third-party, e.g. `ANTHROPIC_API_KEY`)
4. `.env` file (project root)
5. `config.yaml` file
6. Compiled defaults

---

## 2. Pydantic Settings Model

File: `server/engram/config.py`

```python
"""
Engram configuration.

Loads from env vars (ENGRAM_ prefix), .env file, and config.yaml.
Env vars override .env, which overrides YAML, which overrides defaults.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models (plain BaseModel -- not settings, just structure)
# ---------------------------------------------------------------------------


class ClaudeConfig(BaseModel):
    """Claude API configuration for entity extraction."""

    api_key: str = Field(
        default="",
        description=(
            "Anthropic API key. Required for entity extraction. "
            "Accepts ENGRAM_CLAUDE__API_KEY, ENGRAM_ANTHROPIC_API_KEY, "
            "or ANTHROPIC_API_KEY (in that priority order)."
        ),
        # validation_alias handled in Settings.load() -- see note below.
    )
    extraction_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description=(
            "Model used for entity/relationship extraction. "
            "Haiku keeps per-episode costs low (~$0.001/episode). "
            "Use claude-sonnet-4-6 for higher extraction quality at ~10x cost."
        ),
    )
    extraction_max_tokens: int = Field(
        default=4096,
        description="Max tokens for extraction response.",
    )
    extraction_temperature: float = Field(
        default=0.0,
        description="Temperature for extraction. 0 for deterministic output.",
    )
    batch_size: int = Field(
        default=5,
        description=(
            "Number of episodes to batch into a single extraction call "
            "when processing a backlog. Reduces API calls."
        ),
    )


class RedisConfig(BaseModel):
    """Redis connection for activation state hot storage.

    Connection can be specified either via individual fields (host, port, etc.)
    or via a single URL (ENGRAM_REDIS_URL). If URL is provided, it takes
    precedence and is parsed into the individual fields.
    """

    host: str = Field(default="localhost")
    port: int = Field(
        default=6381,
        description=(
            "Redis host-exposed port. Default 6381 avoids collision with "
            "FalkorDB on 6380 and any system Redis on 6379. "
            "Inside Docker, containers always connect on 6379."
        ),
    )
    db: int = Field(
        default=0,
        description="Redis DB number. Keep separate from FalkorDB if co-located.",
    )
    password: str = Field(default="", description="Redis AUTH password. Empty = no auth.")
    activation_key_prefix: str = Field(
        default="engram:activation:",
        description="Key prefix for activation state hashes in Redis.",
    )
    activation_ttl_seconds: int = Field(
        default=604_800,
        description=(
            "TTL for activation keys (default 7 days). "
            "Expired keys are lazily rebuilt from FalkorDB snapshots."
        ),
    )
    max_connections: int = Field(
        default=20,
        description="Connection pool size.",
    )

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    @model_validator(mode="before")
    @classmethod
    def _parse_url(cls, data: Any) -> Any:
        """If ENGRAM_REDIS_URL is set, parse it into host/port/db/password."""
        if isinstance(data, dict):
            url = data.pop("url", None) or os.environ.get("ENGRAM_REDIS_URL")
            if url:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                data.setdefault("host", parsed.hostname or "localhost")
                data.setdefault("port", parsed.port or 6379)
                data.setdefault("password", parsed.password or "")
                if parsed.path and parsed.path.strip("/").isdigit():
                    data.setdefault("db", int(parsed.path.strip("/")))
        return data


class FalkorDBConfig(BaseModel):
    """FalkorDB (RedisGraph module) connection for the knowledge graph.

    Connection can be specified either via individual fields or via
    ENGRAM_FALKORDB_URL. If URL is provided, it takes precedence.
    """

    host: str = Field(default="localhost")
    port: int = Field(
        default=6380,
        description=(
            "FalkorDB host-exposed port. Using 6380 avoids collision with "
            "standalone Redis on 6381 and system Redis on 6379. "
            "Inside Docker, containers always connect on 6379."
        ),
    )
    password: str = Field(default="")
    graph_name: str = Field(
        default="engram",
        description="Name of the FalkorDB graph.",
    )
    max_connections: int = Field(default=20)

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}"

    @model_validator(mode="before")
    @classmethod
    def _parse_url(cls, data: Any) -> Any:
        """If ENGRAM_FALKORDB_URL is set, parse it into host/port/password."""
        if isinstance(data, dict):
            url = data.pop("url", None) or os.environ.get("ENGRAM_FALKORDB_URL")
            if url:
                from urllib.parse import urlparse
                parsed = urlparse(url)
                data.setdefault("host", parsed.hostname or "localhost")
                data.setdefault("port", parsed.port or 6379)
                data.setdefault("password", parsed.password or "")
        return data


class ActivationConfig(BaseModel):
    """Tunable parameters for the ACT-R based activation engine.

    These directly control memory retrieval behavior. Changing them
    alters which memories surface and how aggressively the graph
    "forgets" vs. "remembers".

    Activation is computed lazily from stored access timestamps using the
    ACT-R base-level learning equation. There is no background decay sweep.
    See 02_activation_engine.md for the full design.
    """

    # -- ACT-R Decay --
    decay_exponent: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description=(
            "ACT-R decay exponent (d). Higher = faster forgetting. "
            "Formula: B_i(t) = ln(sum((t - t_j)^{-d})). "
            "0.5 is the standard ACT-R value validated across decades "
            "of cognitive modeling."
        ),
    )
    min_age_seconds: float = Field(
        default=1.0,
        ge=0.01,
        description=(
            "Minimum age for access timestamps to avoid division by zero "
            "in the power-law formula."
        ),
    )
    max_history_size: int = Field(
        default=200,
        ge=10,
        le=10000,
        description=(
            "Maximum access timestamps stored per node. Older entries "
            "are evicted. Bounds computation cost (200 terms per node "
            "per retrieval). Older accesses contribute negligibly to "
            "the power-law sum anyway."
        ),
    )

    # -- Sigmoid Normalization --
    B_mid: float = Field(
        default=-0.5,
        description=(
            "Raw B_i value that maps to activation 0.5. Calibrated so "
            "a node accessed once ~1 hour ago sits at activation 0.5."
        ),
    )
    B_scale: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Sigmoid steepness for normalizing raw B_i to [0, 1]. "
            "Larger = more gradual transition."
        ),
    )

    # -- Spreading Activation --
    spread_max_hops: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Maximum BFS depth for spreading activation. "
            "2 = direct neighbors and their neighbors. "
            "3+ hops pulls in too much noise for personal-scale graphs."
        ),
    )
    spread_decay_per_hop: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description=(
            "Energy multiplier per hop. 0.5 = halve each hop. "
            "Hop-2 neighbors get 25% of seed energy at most."
        ),
    )
    spread_firing_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum energy to propagate to a neighbor. Below this, "
            "the spread stops. Prevents negligible activations from "
            "expanding the BFS frontier."
        ),
    )
    spread_energy_budget: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Total energy that can be distributed in one retrieval. "
            "Bounds worst-case spreading cost. With typical seed energies "
            "of 0.5-0.8, this allows spreading to ~20-30 nodes."
        ),
    )

    # -- Retrieval Scoring Weights --
    weight_semantic: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for semantic similarity in composite score. "
            "Primary signal -- content relevance to the query."
        ),
    )
    weight_activation: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for activation level in composite score. "
            "Encodes recency + frequency + associative priming "
            "via the ACT-R formula + spreading bonus."
        ),
    )
    weight_edge_proximity: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description=(
            "Weight for edge proximity (structural closeness in graph) "
            "in composite score. Tiebreaker that favors structurally "
            "close nodes."
        ),
    )
    seed_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum semantic similarity for a candidate to become a "
            "spreading activation seed node."
        ),
    )

    # -- Retrieval Limits --
    retrieval_top_k: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of candidates from vector search before scoring.",
    )
    retrieval_top_n: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results returned after composite scoring.",
    )

    @model_validator(mode="after")
    def _validate_scoring_weights(self) -> "ActivationConfig":
        total = (
            self.weight_semantic
            + self.weight_activation
            + self.weight_edge_proximity
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.3f}. "
                f"(semantic={self.weight_semantic}, activation={self.weight_activation}, "
                f"edge_proximity={self.weight_edge_proximity})"
            )
        return self


_MODEL_DIMENSIONS: dict[str, int] = {
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "nomic-ai/nomic-embed-text-v1.5": 768,
}


class EmbeddingConfig(BaseModel):
    """Embedding model and vector index configuration.

    See 04_embedding_strategy.md for rationale, benchmarks, and
    the full provider abstraction layer.
    """

    # -- Provider selection --
    provider: str = Field(
        default="voyage",
        description=(
            "Embedding provider. Options: 'voyage', 'openai', 'local'. "
            "Voyage voyage-3-lite is the default for best cost/quality. "
            "'local' uses sentence-transformers (no API key needed)."
        ),
    )
    model: str = Field(
        default="voyage-3-lite",
        description=(
            "Model identifier passed to the provider. "
            "Default: voyage-3-lite (512 dim, $0.02/M tokens). "
            "Upgrade: voyage-3 (1024 dim, $0.06/M tokens)."
        ),
    )
    api_key: str = Field(
        default="",
        description=(
            "API key for the embedding provider. Not needed for 'local'. "
            "Accepts ENGRAM_EMBEDDING__API_KEY or ENGRAM_EMBEDDING_API_KEY."
        ),
    )

    # -- Vector index parameters --
    dimensions: int = Field(
        default=0,
        description=(
            "Embedding vector dimensions. If 0 (default), auto-detected "
            "from the model name. Override only if using a custom model."
        ),
    )
    distance_metric: str = Field(
        default="COSINE",
        description="Distance metric for vector search: 'COSINE', 'IP', or 'L2'.",
    )
    hnsw_m: int = Field(
        default=16,
        ge=2,
        le=64,
        description=(
            "HNSW connectivity parameter. Higher = better recall at "
            "the cost of more memory and slower indexing."
        ),
    )
    hnsw_ef_construction: int = Field(
        default=200,
        ge=10,
        le=500,
        description="HNSW build-time search depth. Higher = better index quality.",
    )
    hnsw_ef_runtime: int = Field(
        default=50,
        ge=10,
        le=500,
        description="HNSW query-time search depth. Higher = better recall, slower queries.",
    )

    # -- Retrieval parameters --
    candidate_top_k: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Number of candidates from vector search before composite scoring.",
    )
    hybrid_search: bool = Field(
        default=False,
        description="Enable BM25 + semantic hybrid search.",
    )
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic score in hybrid search.",
    )
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword (BM25) score in hybrid search.",
    )

    # -- Operational --
    batch_size: int = Field(
        default=64,
        description="Max texts per embedding API call.",
    )
    allow_reindex: bool = Field(
        default=False,
        description=(
            "Must be true to allow dimension changes that require "
            "re-embedding all existing vectors. Safety guard."
        ),
    )

    # -- Local provider settings --
    local_model_name: str = Field(
        default="nomic-ai/nomic-embed-text-v1.5",
        description="HuggingFace model ID for local embedding provider.",
    )
    local_device: str = Field(
        default="cpu",
        description="Device for local model inference: 'cpu', 'cuda', or 'mps'.",
    )

    @model_validator(mode="after")
    def _auto_set_dimensions_and_validate(self) -> "EmbeddingConfig":
        # Auto-detect dimensions from model if not explicitly set
        if self.dimensions == 0:
            self.dimensions = _MODEL_DIMENSIONS.get(self.model, 512)

        # Validate provider
        if self.provider not in ("voyage", "openai", "local"):
            raise ValueError(
                f"embedding.provider must be 'voyage', 'openai', or 'local', "
                f"got '{self.provider}'"
            )

        # Validate distance metric
        if self.distance_metric not in ("COSINE", "IP", "L2"):
            raise ValueError(
                f"embedding.distance_metric must be 'COSINE', 'IP', or 'L2', "
                f"got '{self.distance_metric}'"
            )

        # Validate local device
        if self.local_device not in ("cpu", "cuda", "mps"):
            raise ValueError(
                f"embedding.local_device must be 'cpu', 'cuda', or 'mps', "
                f"got '{self.local_device}'"
            )

        # Validate hybrid search weights sum to 1.0
        if self.hybrid_search:
            total = self.semantic_weight + self.keyword_weight
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"Hybrid search weights must sum to 1.0, got {total:.3f}. "
                    f"(semantic={self.semantic_weight}, keyword={self.keyword_weight})"
                )

        return self


class SQLiteConfig(BaseModel):
    """SQLite configuration for lite mode.

    In lite mode, SQLite replaces FalkorDB (graph store) and Redis
    (activation state). A single file provides zero-infrastructure
    persistence. See 09_lite_mode.md for the full design.
    """

    path: str = Field(
        default="~/.engram/engram.db",
        description=(
            "Path to the SQLite database file. Tilde-expanded at runtime. "
            "Created automatically if it does not exist."
        ),
    )

    @property
    def resolved_path(self) -> Path:
        """Return the tilde-expanded, absolute path."""
        return Path(self.path).expanduser().resolve()


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ServerConfig(BaseModel):
    """FastAPI server and MCP transport configuration."""

    env: str = Field(
        default="production",
        description="Environment: 'production', 'development', or 'test'.",
    )
    host: str = Field(default="0.0.0.0")
    port: int = Field(
        default=8100,
        description="REST API + SSE port.",
    )
    mcp_port: int = Field(
        default=8101,
        description="Dedicated MCP SSE transport port (separate from REST API).",
    )
    ws_port: int = Field(
        default=8102,
        description="WebSocket port for dashboard live updates.",
    )
    log_level: LogLevel = Field(default=LogLevel.INFO)
    workers: int = Field(
        default=1,
        description=(
            "Uvicorn workers. Keep at 1 unless running behind a load balancer, "
            "since activation state is in-process cached."
        ),
    )


class AdditionalTokenEntry(BaseModel):
    """A named bearer token mapped to a specific group_id."""

    name: str = Field(description="Human-readable label, e.g. 'claude_desktop'.")
    token: str = Field(description="Bearer token value. Should come from env var.")
    group_id: str = Field(
        default="default",
        description="Tenant group this token authenticates into.",
    )


class JWTConfig(BaseModel):
    """JWT configuration for SaaS dashboard auth."""

    secret: str = Field(
        default="",
        description=(
            "JWT signing key. Required when auth.mode='saas'. "
            "Must come from env var ENGRAM_AUTH__JWT__SECRET."
        ),
    )
    algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm.",
    )
    access_token_ttl_seconds: int = Field(
        default=900,
        description="Access token lifetime (default 15 min).",
    )
    refresh_token_ttl_seconds: int = Field(
        default=604_800,
        description="Refresh token lifetime (default 7 days).",
    )


class APIKeyConfig(BaseModel):
    """API key format and storage configuration for SaaS mode."""

    prefix: str = Field(
        default="ek_live_",
        description="All issued API keys start with this prefix.",
    )
    hash_algorithm: str = Field(
        default="argon2id",
        description="Algorithm used to hash stored API keys.",
    )


class AuthConfig(BaseModel):
    """Authentication configuration.

    Two deployment modes:
    - self_hosted: single bearer token (or multiple via additional_tokens).
    - saas: API key auth for MCP clients + JWT auth for dashboard.

    See 05_security_model.md for the full security design.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Enable authentication. "
            "Disabled by default for frictionless local development."
        ),
    )
    mode: str = Field(
        default="self_hosted",
        description="Deployment mode: 'self_hosted' or 'saas'.",
    )
    bearer_token: str = Field(
        default="",
        description=(
            "Primary bearer token for self-hosted auth. "
            "When auth is enabled and this is empty, a random token "
            "is generated at startup and printed to stdout. "
            "Must come from env var ENGRAM_AUTH__BEARER_TOKEN."
        ),
    )
    default_group_id: str = Field(
        default="default",
        description=(
            "Default group_id for multi-tenant isolation. "
            "All entities and episodes are scoped to a group_id. "
            "In self-hosted mode this is the static tenant. "
            "In SaaS mode, group_id comes from the JWT/API-key."
        ),
    )
    additional_tokens: list[AdditionalTokenEntry] = Field(
        default_factory=list,
        description=(
            "Optional extra bearer tokens for multi-client setups. "
            "Each maps a token to a group_id."
        ),
    )
    jwt: JWTConfig = Field(default_factory=JWTConfig)
    api_key: APIKeyConfig = Field(default_factory=APIKeyConfig)

    @model_validator(mode="after")
    def _validate_mode(self) -> "AuthConfig":
        if self.mode not in ("self_hosted", "saas"):
            raise ValueError(
                f"auth.mode must be 'self_hosted' or 'saas', got '{self.mode}'"
            )
        return self


class EncryptionConfig(BaseModel):
    """Field-level encryption for PII at rest.

    Uses AES-256-GCM with per-tenant keys derived from a master key via HKDF.
    See 05_security_model.md Section 7 for details.
    """

    enabled: bool = Field(
        default=True,
        description="Enable field-level encryption for episode content and entity summaries.",
    )
    master_key: str = Field(
        default="",
        description=(
            "32-byte hex master key for encryption. "
            "Must come from env var ENGRAM_ENCRYPTION__MASTER_KEY."
        ),
    )
    encrypt_all_content: bool = Field(
        default=True,
        description="Encrypt episode.content and entity.summary unconditionally.",
    )
    encrypt_pii_only: bool = Field(
        default=False,
        description=(
            "Alternative: only encrypt fields flagged as PII by the extractor. "
            "Mutually exclusive with encrypt_all_content."
        ),
    )


class CORSConfig(BaseModel):
    """CORS configuration for the REST API."""

    allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins. Dashboard dev servers by default.",
    )


class RateLimitBucket(BaseModel):
    """Rate limit for a single endpoint class."""

    max_requests: int = Field(description="Max requests per window.")
    window_seconds: int = Field(default=60, description="Sliding window size.")


class WebSocketRateLimitBucket(BaseModel):
    """Rate limit for WebSocket connections."""

    max_connections: int = Field(default=5, description="Max concurrent WS connections per tenant.")


class RateLimitConfig(BaseModel):
    """Sliding-window rate limiting per API-key (SaaS) or per-IP (self-hosted)."""

    enabled: bool = Field(default=True)
    ingestion: RateLimitBucket = Field(
        default_factory=lambda: RateLimitBucket(max_requests=60, window_seconds=60),
        description="Limits for the 'remember' tool / ingestion endpoints.",
    )
    retrieval: RateLimitBucket = Field(
        default_factory=lambda: RateLimitBucket(max_requests=120, window_seconds=60),
        description="Limits for recall, search_entities, search_facts.",
    )
    admin: RateLimitBucket = Field(
        default_factory=lambda: RateLimitBucket(max_requests=30, window_seconds=60),
        description="Limits for GDPR, config, and admin endpoints.",
    )
    websocket: WebSocketRateLimitBucket = Field(
        default_factory=WebSocketRateLimitBucket,
        description="WebSocket connection limits.",
    )


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------


_SECRET_FIELD_PATHS = [
    ("auth", "bearer_token"),
    ("auth", "jwt", "secret"),
    ("encryption", "master_key"),
    ("claude", "api_key"),
    ("embedding", "api_key"),
    ("redis", "password"),
    ("falkordb", "password"),
]


def _check_plaintext_secrets(data: dict[str, Any]) -> None:
    """Refuse to start if plaintext secrets are found in YAML.

    Secret fields should reference env vars via ${...} syntax or be
    empty (meaning the env var override is expected). A non-empty value
    that does NOT start with '${' is treated as a plaintext secret.
    """
    for path in _SECRET_FIELD_PATHS:
        node = data
        for key in path:
            if not isinstance(node, dict):
                break
            node = node.get(key)
            if node is None:
                break
        if isinstance(node, str) and node and not node.startswith("${"):
            dotted = ".".join(path)
            raise ValueError(
                f"Secret field '{dotted}' appears to contain a plaintext value "
                f"in config.yaml. Use an env var reference (e.g., "
                f"'${{ENGRAM_{path[-1].upper()}}}') or set it via environment "
                f"variable instead. See 05_security_model.md Section 11.2."
            )


def _load_yaml_config(yaml_path: str | Path | None) -> dict[str, Any]:
    """Load config from YAML file, returning empty dict if not found."""
    if yaml_path is None:
        # Check common locations
        candidates = [
            Path("config.yaml"),
            Path("config.yml"),
            Path(os.environ.get("ENGRAM_CONFIG_PATH", "config.yaml")),
        ]
        for candidate in candidates:
            if candidate.is_file():
                yaml_path = candidate
                break
    if yaml_path is None:
        return {}
    path = Path(yaml_path)
    if not path.is_file():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    _check_plaintext_secrets(data)
    return data


class Settings(BaseSettings):
    """Root Engram configuration.

    Loading order (highest priority first):
        1. Environment variables  (ENGRAM_CLAUDE__API_KEY, etc.)
        2. .env file
        3. config.yaml / config.yml
        4. Field defaults defined here
    """

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: str = Field(
        default="auto",
        description=(
            "Infrastructure mode: 'lite', 'full', or 'auto'. "
            "'auto' probes Redis and FalkorDB with 2-second timeouts -- "
            "if both are available, uses full mode; otherwise falls back to lite. "
            "'lite' uses SQLite only (zero infrastructure). "
            "'full' requires Redis + FalkorDB. "
            "Can also be set via ENGRAM_MODE env var or --mode CLI arg. "
            "See 09_lite_mode.md for the full design."
        ),
    )
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    falkordb: FalkorDBConfig = Field(default_factory=FalkorDBConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    activation: ActivationConfig = Field(default_factory=ActivationConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    encryption: EncryptionConfig = Field(default_factory=EncryptionConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    rate_limiting: RateLimitConfig = Field(default_factory=RateLimitConfig)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        if v not in ("lite", "full", "auto"):
            raise ValueError(
                f"mode must be 'lite', 'full', or 'auto', got '{v}'"
            )
        return v

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Settings":
        """Load settings with YAML as the base, overridden by env vars.

        Supports flat DevOps-style env vars (ENGRAM_ANTHROPIC_API_KEY) in
        addition to the nested Pydantic-style (ENGRAM_CLAUDE__API_KEY).
        See Section 1 for the full mapping table.

        Usage:
            settings = Settings.load()                    # auto-discover config.yaml
            settings = Settings.load("my-config.yaml")    # explicit path
        """
        yaml_data = _load_yaml_config(yaml_path)
        settings = cls(**yaml_data)

        # --- Flat env var fallbacks (DevOps convenience) ---
        # These resolve flat ENGRAM_X vars that don't map through
        # pydantic-settings' nested delimiter. Applied only if the
        # canonical nested var was not set.

        # Claude API key: ENGRAM_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY
        if not settings.claude.api_key:
            settings.claude.api_key = (
                os.environ.get("ENGRAM_ANTHROPIC_API_KEY", "")
                or os.environ.get("ANTHROPIC_API_KEY", "")
            )

        # Auth secret: ENGRAM_AUTH_SECRET -> auth.bearer_token
        if not settings.auth.bearer_token:
            settings.auth.bearer_token = os.environ.get("ENGRAM_AUTH_SECRET", "")

        # JWT secret: ENGRAM_JWT_SECRET -> auth.jwt.secret
        if not settings.auth.jwt.secret:
            settings.auth.jwt.secret = os.environ.get("ENGRAM_JWT_SECRET", "")

        # Master key: ENGRAM_MASTER_KEY -> encryption.master_key
        if not settings.encryption.master_key:
            settings.encryption.master_key = os.environ.get("ENGRAM_MASTER_KEY", "")

        # Embedding API key: ENGRAM_EMBEDDING_API_KEY
        if not settings.embedding.api_key:
            settings.embedding.api_key = os.environ.get("ENGRAM_EMBEDDING_API_KEY", "")

        # Log level: ENGRAM_LOG_LEVEL -> server.log_level
        flat_log = os.environ.get("ENGRAM_LOG_LEVEL")
        if flat_log and not os.environ.get("ENGRAM_SERVER__LOG_LEVEL"):
            settings.server.log_level = LogLevel(flat_log.lower())

        # Env: ENGRAM_ENV -> server.env
        flat_env = os.environ.get("ENGRAM_ENV")
        if flat_env and not os.environ.get("ENGRAM_SERVER__ENV"):
            settings.server.env = flat_env

        # Mode: ENGRAM_MODE -> mode (lite/full/auto)
        flat_mode = os.environ.get("ENGRAM_MODE")
        if flat_mode and not os.environ.get("ENGRAM_MODE"):
            settings.mode = flat_mode

        # SQLite path: ENGRAM_SQLITE_PATH -> sqlite.path
        flat_sqlite = os.environ.get("ENGRAM_SQLITE_PATH")
        if flat_sqlite and not os.environ.get("ENGRAM_SQLITE__PATH"):
            settings.sqlite.path = flat_sqlite

        return settings


# Module-level singleton -- import and use throughout the app.
# Lazy-loaded on first access to avoid import-time side effects.
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
```

### 2.1 Defaults Rationale

| Field | Default | Rationale |
|-------|---------|-----------|
| `claude.extraction_model` | `claude-haiku-4-5-20251001` | Haiku is 10x cheaper than Sonnet. Extraction is structured output, not creative -- Haiku handles it well. Users can upgrade to Sonnet for tricky domains. |
| `claude.extraction_temperature` | `0.0` | Deterministic extraction. We want the same input to produce the same entities every time. |
| `claude.batch_size` | `5` | Balances throughput vs. context window usage. 5 short episodes fit comfortably in Haiku's window. |
| `redis.port` | `6381` | Host-exposed port. 6381 avoids collision with FalkorDB on 6380 and any system Redis on 6379. Inside Docker, containers connect on 6379. |
| `redis.activation_ttl_seconds` | `604800` (7d) | Nodes not accessed in a week have fully decayed. Their state can be cheaply rebuilt from FalkorDB snapshots on next access. |
| `falkordb.port` | `6380` | Avoids collision with standalone Redis on 6379. FalkorDB is a Redis module but runs as its own process in Docker. |
| `activation.decay_exponent` | `0.5` | Standard ACT-R value. With power-law decay `t^{-0.5}`, a single access 1h ago gives activation ~0.50, 1 day ago ~0.10, 1 week ago ~0.03. Multiple accesses sum, so frequency compensates for recency. |
| `activation.min_age_seconds` | `1.0` | Floor to avoid div-by-zero in `t^{-d}`. 1 second is generous. |
| `activation.max_history_size` | `200` | 200 timestamp terms is sub-millisecond to compute. Older accesses contribute negligibly to the power-law sum. |
| `activation.B_mid` | `-0.5` | Calibrated so a node accessed once ~1 hour ago maps to activation 0.5. |
| `activation.B_scale` | `1.0` | Smooth sigmoid without extreme clipping. |
| `activation.spread_max_hops` | `2` | 2 hops captures direct + one level of indirect association. 3+ hops pulls in too much noise. |
| `activation.spread_decay_per_hop` | `0.5` | Halving per hop means hop-2 neighbors get 25% of seed energy at most. |
| `activation.spread_firing_threshold` | `0.05` | Prevents negligible activations from expanding BFS frontier. |
| `activation.spread_energy_budget` | `5.0` | With typical seed energies of 0.5-0.8, allows spreading to ~20-30 nodes. |
| `activation.weight_*` | `0.50/0.35/0.15` | Three orthogonal signals: semantic similarity dominates; activation provides meaningful re-ranking; edge proximity is a structural tiebreaker. See 02_activation_engine.md Section 7. |
| `activation.seed_threshold` | `0.3` | Minimum semantic similarity to become a spreading seed. Prevents low-relevance candidates from seeding. |
| `activation.retrieval_top_k` | `50` | Vector search candidate count before scoring. Balances recall with compute cost. |
| `activation.retrieval_top_n` | `10` | Final returned results. Suitable for MCP tool responses. |
| `mode` | `auto` | Auto-detect probes Redis + FalkorDB with 2s timeouts. Falls back to lite if unavailable. Frictionless for new users. |
| `sqlite.path` | `~/.engram/engram.db` | XDG-style home directory location. Created automatically. Only used in lite mode. |
| `embedding.provider` | `voyage` | Voyage AI voyage-3-lite offers the best cost/quality tradeoff ($0.02/M tokens). Alternative: `local` for zero-API-key setup. |
| `embedding.model` | `voyage-3-lite` | 512 dimensions keeps vectors compact (2KB/vector). Quality is sufficient because the activation engine re-ranks results. Upgrade to `voyage-3` (1024d) for higher recall. |
| `embedding.dimensions` | auto (512) | Auto-detected from model name. Override only for custom models. |
| `embedding.distance_metric` | `COSINE` | Standard for normalized embeddings. All three supported providers output unit-normalized vectors. |
| `embedding.hnsw_m` | `16` | Standard HNSW connectivity. Good balance of recall and memory. |
| `embedding.candidate_top_k` | `20` | Candidates before activation scoring. Lower than `activation.retrieval_top_k` (50) because vector search runs first; activation scoring then expands via spreading. |
| `embedding.hybrid_search` | `false` | Disabled by default. Enable for domains with specific terminology where keyword matching adds value. |
| `embedding.local_model_name` | `nomic-embed-text-v1.5` | Best open-source embedding model for retrieval (768d, 8K context). |
| `server.port` | `8100` | Avoids conflict with common dev servers (3000, 5173, 8000, 8080). |
| `auth.enabled` | `false` | Frictionless quickstart. Security doc (05) defines when/how to enable. |
| `auth.mode` | `self_hosted` | Self-hosted uses bearer tokens. SaaS uses API keys + JWT. |
| `auth.default_group_id` | `default` | Single-user self-hosted mode uses one group. SaaS overrides per request via JWT. |
| `auth.jwt.algorithm` | `HS256` | Simpler than RS256 for single-service deployments. SaaS can switch to RS256. |
| `auth.jwt.access_token_ttl_seconds` | `900` (15 min) | Short-lived access tokens limit exposure from stolen tokens. |
| `auth.jwt.refresh_token_ttl_seconds` | `604800` (7 days) | Balances security with UX -- users re-login weekly. |
| `auth.api_key.prefix` | `ek_live_` | Human-readable prefix makes keys easy to identify and revoke. |
| `auth.api_key.hash_algorithm` | `argon2id` | Memory-hard hash. Resistant to GPU/ASIC attacks. |
| `encryption.enabled` | `true` | Engram stores personal memories -- encryption at rest is on by default. |
| `encryption.encrypt_all_content` | `true` | Conservative default. All content encrypted, not just PII-flagged. |
| `rate_limiting.enabled` | `true` | Prevents abuse. Self-hosted users can disable if desired. |
| `rate_limiting.ingestion.max_requests` | `60/min` | Ingestion is expensive (Claude API call per episode). |
| `rate_limiting.retrieval.max_requests` | `120/min` | Retrieval is cheaper but still involves graph + Redis. |

---

## 3. Example `config.yaml`

File: `config.example.yaml` (committed to repo root)

```yaml
# Engram Configuration
# Copy to config.yaml and fill in your values.
# Environment variables (ENGRAM_ prefix) override these values.

# Infrastructure mode: "auto" (default), "lite", or "full"
# auto: probes Redis + FalkorDB, falls back to lite if unavailable
# lite: SQLite only, zero infrastructure (pip install engram)
# full: requires Redis + FalkorDB (Docker Compose)
mode: "auto"

claude:
  api_key: ""  # Required: your Anthropic API key
  extraction_model: "claude-haiku-4-5-20251001"

redis:
  host: "localhost"
  port: 6381

falkordb:
  host: "localhost"
  port: 6380
  graph_name: "engram"

sqlite:
  path: "~/.engram/engram.db"  # lite mode only

activation:
  # ACT-R decay (see 02_activation_engine.md)
  decay_exponent: 0.5
  min_age_seconds: 1.0
  max_history_size: 200
  # Sigmoid normalization
  B_mid: -0.5
  B_scale: 1.0
  # Spreading activation
  spread_max_hops: 2
  spread_decay_per_hop: 0.5
  spread_firing_threshold: 0.05
  spread_energy_budget: 5.0
  # Scoring weights (must sum to 1.0)
  weight_semantic: 0.50
  weight_activation: 0.35
  weight_edge_proximity: 0.15
  seed_threshold: 0.3
  # Retrieval limits
  retrieval_top_k: 50
  retrieval_top_n: 10

embedding:
  provider: "voyage"  # or "openai" or "local"
  model: "voyage-3-lite"
  api_key: ""  # Set via ENGRAM_EMBEDDING_API_KEY env var
  # dimensions: auto-detected from model (512 for voyage-3-lite)
  distance_metric: "COSINE"
  hnsw_m: 16
  hnsw_ef_construction: 200
  hnsw_ef_runtime: 50
  candidate_top_k: 20
  hybrid_search: false
  semantic_weight: 0.7
  keyword_weight: 0.3
  batch_size: 64
  allow_reindex: false
  local_model_name: "nomic-ai/nomic-embed-text-v1.5"
  local_device: "cpu"

server:
  host: "0.0.0.0"
  port: 8100
  mcp_port: 8101
  ws_port: 8102
  log_level: "info"

auth:
  enabled: false
  mode: "self_hosted"
  bearer_token: "${ENGRAM_AUTH__BEARER_TOKEN}"
  default_group_id: "default"
  # additional_tokens:
  #   - name: "claude_desktop"
  #     token: "${ENGRAM_TOKEN_CLAUDE_DESKTOP}"
  #     group_id: "default"
  jwt:  # SaaS mode only
    secret: "${ENGRAM_AUTH__JWT__SECRET}"
    algorithm: "HS256"
    access_token_ttl_seconds: 900
    refresh_token_ttl_seconds: 604800
  api_key:  # SaaS mode only
    prefix: "ek_live_"
    hash_algorithm: "argon2id"

encryption:
  enabled: true
  master_key: "${ENGRAM_ENCRYPTION__MASTER_KEY}"
  encrypt_all_content: true
  encrypt_pii_only: false

cors:
  allowed_origins:
    - "http://localhost:3000"
    - "http://localhost:5173"

rate_limiting:
  enabled: true
  ingestion:
    max_requests: 60
    window_seconds: 60
  retrieval:
    max_requests: 120
    window_seconds: 60
  admin:
    max_requests: 30
    window_seconds: 60
  websocket:
    max_connections: 5
```

---

## 4. Example `.env`

```bash
# Minimal .env -- secrets only, structure in config.yaml
# config.yaml is gitignored; only config.example.yaml is committed.
#
# Both flat (ENGRAM_X) and nested (ENGRAM_X__Y) styles work.
# Flat style shown here for Docker/DevOps compatibility.

# Required -- Claude API key for entity extraction
ENGRAM_ANTHROPIC_API_KEY=sk-ant-api03-xxxxxxxxxxxx

# Embedding provider key (not needed if using provider='local')
ENGRAM_EMBEDDING__API_KEY=pa-xxxxxxxxxxxxxxxx

# Data store passwords (change from defaults in production)
ENGRAM_FALKORDB_PASSWORD=changeme
ENGRAM_REDIS_PASSWORD=changeme

# Auth (disabled by default for local dev)
# ENGRAM_AUTH_ENABLED=true
# ENGRAM_AUTH_SECRET=my-secret-token     # generate: openssl rand -hex 32

# PII encryption master key (optional -- only when encryption.enabled=true)
# ENGRAM_MASTER_KEY=0a1b2c3d4e5f...      # generate: openssl rand -hex 32

# SaaS mode only:
# ENGRAM_JWT_SECRET=your-jwt-signing-secret
```

---

## 5. Docker Compose Environment Mapping

For `docker-compose.yml`, services reference the same env vars:

```yaml
services:
  engram-server:
    environment:
      # Connection URLs (container-internal addresses, port 6379)
      - ENGRAM_FALKORDB_URL=redis://:${ENGRAM_FALKORDB_PASSWORD:-changeme}@falkordb:6379
      - ENGRAM_REDIS_URL=redis://:${ENGRAM_REDIS_PASSWORD:-changeme}@redis:6379/0
      - ENGRAM_SERVER_HOST=0.0.0.0
      # Secrets -- passed through from host .env
      - ENGRAM_ANTHROPIC_API_KEY=${ENGRAM_ANTHROPIC_API_KEY}
      - ENGRAM_AUTH_ENABLED=${ENGRAM_AUTH_ENABLED:-false}
      - ENGRAM_AUTH_SECRET=${ENGRAM_AUTH_SECRET:-}
      - ENGRAM_MASTER_KEY=${ENGRAM_MASTER_KEY:-}
    env_file:
      - .env
```

Note: Inside Docker Compose, connection URLs point to container service names
(`falkordb`, `redis`) on their internal port 6379. The host-exposed ports
(`ENGRAM_FALKORDB_PORT=6380`, `ENGRAM_REDIS_PORT=6381`) are only used for
external access (CLI tools, debugging). See 08_devops_infrastructure.md Section 1.

---

## 6. FalkorDB Index DDL

File: `server/engram/graph/indexes.cypher`

These indexes support the core query patterns: entity lookup, episode timeline queries, relationship filtering, and multi-tenant isolation.

```cypher
// =============================================================================
// Engram FalkorDB Indexes
// =============================================================================
// Run once on graph initialization. FalkorDB silently no-ops if an index
// already exists, so this script is idempotent.
//
// FalkorDB index syntax:
//   CREATE INDEX FOR (n:Label) ON (n.property)
//   CREATE INDEX FOR ()-[r:TYPE]-() ON (r.property)
// =============================================================================

// ---------------------------------------------------------------------------
// Entity indexes
// ---------------------------------------------------------------------------

// Multi-tenant isolation. Every entity query filters by group_id first.
// This is the most critical index -- without it, every query scans all nodes.
CREATE INDEX FOR (n:Entity) ON (n.group_id)

// Entity lookup by name within a group. Used for dedup, search, and linking.
CREATE INDEX FOR (n:Entity) ON (n.name)

// Filter/sort by entity type (Person, Project, Concept, etc.).
CREATE INDEX FOR (n:Entity) ON (n.entity_type)

// Temporal queries: "entities created after X", "recently updated entities".
CREATE INDEX FOR (n:Entity) ON (n.created_at)
CREATE INDEX FOR (n:Entity) ON (n.updated_at)

// Access recency: used by dashboard "recently active" queries and
// the activation engine's top-activated leaderboard (which first
// filters by recently-accessed nodes, then computes activation lazily).
CREATE INDEX FOR (n:Entity) ON (n.last_accessed)

// ---------------------------------------------------------------------------
// Episode indexes
// ---------------------------------------------------------------------------

// Multi-tenant + chronological. The primary episode query pattern is
// "episodes for group X ordered by time" for the Memory Feed view.
CREATE INDEX FOR (n:Episode) ON (n.group_id)
CREATE INDEX FOR (n:Episode) ON (n.created_at)

// Filter episodes by source (claude_desktop, claude_code, api, etc.).
CREATE INDEX FOR (n:Episode) ON (n.source)

// ---------------------------------------------------------------------------
// Relationship (edge) indexes
// ---------------------------------------------------------------------------

// Filter relationships by predicate ("works_at", "interested_in", etc.).
// Used in search_facts and graph explorer filtering.
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.predicate)

// Temporal validity. Queries like "facts valid at time T" use these.
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_from)
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_to)

// Edge weight for spreading activation traversal -- retrieve high-weight
// edges first during BFS spread.
CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.weight)
```

### 6.1 Index Rationale

| Index | Query Pattern | Why It Matters |
|-------|--------------|----------------|
| `Entity(group_id)` | `WHERE n.group_id = $gid` on every query | Without this, FalkorDB scans all entities across all tenants. This is the partition key. |
| `Entity(name)` | Entity dedup (`MATCH (n:Entity {name: $name, group_id: $gid})`) | Called on every ingestion to check for existing entities before creating new ones. |
| `Entity(entity_type)` | `search_entities` tool with type filter | Dashboard type-distribution queries, filtered graph views. |
| `Entity(created_at)` | Timeline view, "entities created this week" | Temporal exploration is a core dashboard feature. |
| `Entity(updated_at)` | "Recently changed" feed | Memory Feed sorts by update time. |
| `Entity(last_accessed)` | `ORDER BY n.last_accessed DESC LIMIT $n` | Dashboard "recently active" queries. Also used to find candidate nodes for the top-activated leaderboard (lazy activation is then computed from Redis access_history). |
| `Episode(group_id)` | All episode queries scoped by tenant | Same rationale as Entity(group_id). |
| `Episode(created_at)` | Memory Feed reverse-chronological listing | Primary sort for the feed view, paginated. |
| `Episode(source)` | Filter by conversation source | "Show me what Claude Desktop learned" vs. "Claude Code". |
| `RELATES_TO(predicate)` | `search_facts` by relationship type | "Find all 'works_at' relationships" is a common query. |
| `RELATES_TO(valid_from/to)` | Temporal validity window queries | "What was true as of date X?" -- core to the bi-temporal model. |
| `RELATES_TO(weight)` | Spreading activation edge traversal | Prioritize high-weight edges during BFS spread for performance. |

### 6.2 Python Bootstrap Function

The indexes should be applied on server startup. Here is the initialization function:

```python
"""Graph initialization -- create indexes and ensure schema."""

from falkordb import FalkorDB

from engram.config import get_settings


INDEX_STATEMENTS = [
    # Entity indexes
    "CREATE INDEX FOR (n:Entity) ON (n.group_id)",
    "CREATE INDEX FOR (n:Entity) ON (n.name)",
    "CREATE INDEX FOR (n:Entity) ON (n.entity_type)",
    "CREATE INDEX FOR (n:Entity) ON (n.created_at)",
    "CREATE INDEX FOR (n:Entity) ON (n.updated_at)",
    "CREATE INDEX FOR (n:Entity) ON (n.last_accessed)",
    # Episode indexes
    "CREATE INDEX FOR (n:Episode) ON (n.group_id)",
    "CREATE INDEX FOR (n:Episode) ON (n.created_at)",
    "CREATE INDEX FOR (n:Episode) ON (n.source)",
    # Edge indexes
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.predicate)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_from)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.valid_to)",
    "CREATE INDEX FOR ()-[r:RELATES_TO]-() ON (r.weight)",
]


def ensure_indexes() -> None:
    """Create all indexes idempotently. Safe to call on every startup."""
    settings = get_settings()
    db = FalkorDB(
        host=settings.falkordb.host,
        port=settings.falkordb.port,
        password=settings.falkordb.password or None,
    )
    graph = db.select_graph(settings.falkordb.graph_name)
    for stmt in INDEX_STATEMENTS:
        try:
            graph.query(stmt)
        except Exception:
            # FalkorDB raises if index already exists; safe to ignore.
            pass
```

### 6.3 Composite Index Note

FalkorDB does not support composite indexes (multi-property). The common pattern `WHERE n.group_id = $gid AND n.name = $name` will use the `group_id` index to filter first, then scan the reduced set for `name`. At personal-graph scale (<100K nodes), this is fast enough. If it becomes a bottleneck at scale, the mitigation is to include `group_id` as a prefix in the `name` field (`f"{group_id}:{name}"`), but this is not recommended for v0.1.

---

## 7. Validation & Startup Checks

The server should validate config on startup before accepting requests:

```python
def validate_config_on_startup(settings: Settings) -> list[str]:
    """Return a list of warnings. Raise on fatal misconfiguration."""
    warnings: list[str] = []

    # Fatal: no Claude API key
    if not settings.claude.api_key:
        raise ValueError(
            "Claude API key is required. Set one of: "
            "ENGRAM_CLAUDE__API_KEY, ENGRAM_ANTHROPIC_API_KEY, or ANTHROPIC_API_KEY."
        )

    # Fatal: embedding API key required for non-local providers
    if settings.embedding.provider != "local" and not settings.embedding.api_key:
        raise ValueError(
            f"Embedding API key is required for provider '{settings.embedding.provider}'. "
            "Set ENGRAM_EMBEDDING__API_KEY or ENGRAM_EMBEDDING_API_KEY, "
            "or use provider='local' for no-API embeddings."
        )

    # Fatal: SaaS mode requires JWT secret
    if settings.auth.mode == "saas" and not settings.auth.jwt.secret:
        raise ValueError(
            "ENGRAM_AUTH__JWT__SECRET is required when auth.mode='saas'. "
            "Set it in .env or as an environment variable."
        )

    # Fatal: encryption enabled but no master key
    if settings.encryption.enabled and not settings.encryption.master_key:
        raise ValueError(
            "ENGRAM_ENCRYPTION__MASTER_KEY is required when encryption is enabled. "
            "Set it in .env or disable encryption with ENGRAM_ENCRYPTION__ENABLED=false."
        )

    # Warning: auth disabled
    if not settings.auth.enabled:
        warnings.append(
            "Auth is disabled. API endpoints are unauthenticated. "
            "Set ENGRAM_AUTH__ENABLED=true for production deployments."
        )

    # Warning: default group_id in production with auth enabled
    if settings.auth.enabled and settings.auth.default_group_id == "default":
        warnings.append(
            "Auth is enabled but default_group_id is still 'default'. "
            "For multi-tenant setups, group_id should come from the JWT."
        )

    # Warning: encryption disabled
    if not settings.encryption.enabled:
        warnings.append(
            "Encryption is disabled. Episode content and entity summaries "
            "will be stored in plaintext. Enable for production."
        )

    # Info: lite mode
    if settings.mode == "lite":
        warnings.append(
            f"Running in lite mode. Using SQLite at "
            f"{settings.sqlite.resolved_path}. "
            "Redis and FalkorDB config sections are ignored."
        )
    elif settings.mode == "auto":
        warnings.append(
            "Running in auto mode. Will probe Redis and FalkorDB on startup."
        )

    return warnings
```

---

## 8. Cross-Cutting Concerns for Other Agents

### For Auth Agent (Task #5)
- **Updated to match 05_security_model.md.** All fields from the auth agent's request are now included.
- `AuthConfig.enabled`: controls whether the FastAPI middleware checks bearer tokens.
- `AuthConfig.mode`: `"self_hosted"` or `"saas"` -- determines which auth flow is active.
- `AuthConfig.bearer_token`: primary token for self-hosted mode. Must come from env var.
- `AuthConfig.default_group_id`: the tenant isolation key. In self-hosted mode it is static. In SaaS mode, group_id comes from JWT/API-key.
- `AuthConfig.additional_tokens`: list of `{name, token, group_id}` for multi-client setups.
- `AuthConfig.jwt.*`: JWT config for SaaS dashboard auth (secret, algorithm, TTLs).
- `AuthConfig.api_key.*`: API key config for SaaS MCP clients (prefix, hash algorithm).
- `EncryptionConfig.*`: field-level encryption at rest (enabled, master_key, encrypt_all_content, encrypt_pii_only).
- `CORSConfig.allowed_origins`: replaces the old `server.cors_origins` field.
- `RateLimitConfig.*`: per-endpoint-class rate limiting (ingestion, retrieval, admin, websocket).
- Secret-in-YAML validation: the config loader now refuses to start if plaintext secrets are found in the YAML file. See `_check_plaintext_secrets()`.
- The auth middleware must inject `group_id` into request state so all downstream graph queries are scoped.

### For Activation Engine Agent (Task #2)
- **Updated to match 02_activation_engine.md ACT-R model.** All old parameters replaced.
- ACT-R decay: `decay_exponent` (d=0.5), `min_age_seconds`, `max_history_size`.
- Sigmoid normalization: `B_mid`, `B_scale`.
- Spreading activation: `spread_max_hops`, `spread_decay_per_hop`, `spread_firing_threshold`, `spread_energy_budget`.
- Scoring weights: `weight_semantic` (0.50), `weight_activation` (0.35), `weight_edge_proximity` (0.15) -- three orthogonal signals. Model validator enforces sum to 1.0.
- Retrieval: `seed_threshold`, `retrieval_top_k`, `retrieval_top_n`.
- **Removed**: `decay_rate`, `decay_sweep_interval_seconds` (lazy decay, no background sweep), `spread_factor`, `contextual_boost`, `reinforcement_delta`, `max_activated_nodes`, `firing_threshold`, `weight_recency`, `weight_frequency`.
- Redis config (`activation_key_prefix`, `activation_ttl_seconds`) still defines how activation state (access_history timestamps) is stored.

### For Embedding Agent (Task #4)
- **Updated to match 04_embedding_strategy.md.** All fields from the embedding agent's request are now included.
- `EmbeddingConfig.provider`: `"voyage"`, `"openai"`, or `"local"`. Note: changed from `"voyageai"` to `"voyage"` to match the doc.
- `EmbeddingConfig.model`: default `"voyage-3-lite"` (512d). Changed from `"voyage-3-large"`.
- `EmbeddingConfig.dimensions`: auto-detected from model name via `_MODEL_DIMENSIONS` lookup. Default 0 means auto-detect. Override for custom models.
- Vector index params: `distance_metric` (COSINE/IP/L2), `hnsw_m` (2-64), `hnsw_ef_construction` (10-500), `hnsw_ef_runtime` (10-500).
- Retrieval params: `candidate_top_k` (20), `hybrid_search` (false), `semantic_weight` (0.7), `keyword_weight` (0.3).
- Hybrid search weights validated to sum to 1.0 when `hybrid_search=true`.
- Operational: `batch_size` (64), `allow_reindex` (false).
- Local provider: `local_model_name` (`nomic-ai/nomic-embed-text-v1.5`), `local_device` (cpu/cuda/mps).
- Flat env var: `ENGRAM_EMBEDDING_API_KEY` -> `embedding.api_key` (handled by existing nested `ENGRAM_EMBEDDING__API_KEY`).
- The embedding module should read these from `get_settings().embedding`.

### For Ingestion Agent (Task #3)
- `ClaudeConfig.extraction_model`, `extraction_max_tokens`, `extraction_temperature` control the extraction call.
- `ClaudeConfig.batch_size` determines how many episodes can be batched into one extraction request.
- The ingestion pipeline should use `get_settings().claude` for all API configuration.

### For MCP Agent (Task #6)
- `ServerConfig.mcp_port` is the dedicated port for the MCP SSE transport.
- `AuthConfig` fields determine whether MCP tools require authentication.
- `AuthConfig.group_id` scopes all MCP tool operations.

### For Lite Mode Agent (Task #9)
- `Settings.mode`: top-level field, `Literal["lite", "full", "auto"]`, default `"auto"`. Validated by `_validate_mode()`.
- `SQLiteConfig.path`: default `"~/.engram/engram.db"`. `resolved_path` property returns tilde-expanded absolute `Path`.
- Flat env var aliases: `ENGRAM_MODE` -> `mode`, `ENGRAM_SQLITE_PATH` -> `sqlite.path`.
- The mode resolver (`engram/storage/resolver.py`) should consume the validated `Settings` object via `get_settings()`, not read raw env vars.
- CLI `--mode` override: the entrypoint should call `settings.mode = cli_args.mode` before passing settings to the resolver.
- In lite mode, `redis.*` and `falkordb.*` config sections are ignored (but still present with defaults for schema completeness).

### For DevOps Agent (Task #8)
- **Updated to align with 08_devops_infrastructure.md env var conventions.**
- Docker Compose uses URL-based connection strings: `ENGRAM_FALKORDB_URL`, `ENGRAM_REDIS_URL`. These are parsed by `_parse_url()` validators on `FalkorDBConfig` and `RedisConfig`.
- Flat env var aliases supported: `ENGRAM_ANTHROPIC_API_KEY`, `ENGRAM_AUTH_SECRET`, `ENGRAM_MASTER_KEY`, `ENGRAM_JWT_SECRET`, `ENGRAM_LOG_LEVEL`, `ENGRAM_ENV`. See Section 1.2 for the full mapping table.
- Default ports aligned: FalkorDB=6380, Redis=6381, Server=8100 (host-exposed). Inside Docker, containers use 6379.
- `ANTHROPIC_API_KEY` (without prefix) accepted as fallback for Claude API key.
- Health check endpoint should verify Redis and FalkorDB connectivity using the config.
- `ServerConfig.workers` should stay at 1 for single-node deployments.
- `ServerConfig.env` field added for environment detection (`production`, `development`, `test`).

---

## 9. Dependencies

The config module requires these Python packages in `pyproject.toml`:

```toml
[project]
dependencies = [
    "pydantic>=2.6,<3",
    "pydantic-settings>=2.2,<3",
    "pyyaml>=6.0,<7",
]
```

---

## 10. Testing the Config

```python
"""Tests for config loading."""

import os
import pytest
from engram.config import (
    Settings, ActivationConfig, AuthConfig, EncryptionConfig,
    _check_plaintext_secrets,
)


def test_defaults_load():
    """Settings load with all defaults when no env/yaml is present."""
    s = Settings()
    assert s.redis.port == 6381
    assert s.falkordb.port == 6380
    assert s.activation.decay_exponent == 0.5
    assert s.activation.spread_max_hops == 2
    assert s.activation.weight_semantic == 0.50
    assert s.server.port == 8100
    assert s.auth.enabled is False
    assert s.auth.mode == "self_hosted"
    assert s.auth.default_group_id == "default"
    assert s.encryption.enabled is True
    assert s.rate_limiting.enabled is True
    assert s.cors.allowed_origins == [
        "http://localhost:3000", "http://localhost:5173"
    ]


def test_env_override(monkeypatch):
    """Environment variables override defaults."""
    monkeypatch.setenv("ENGRAM_REDIS__PORT", "7777")
    monkeypatch.setenv("ENGRAM_ACTIVATION__DECAY_EXPONENT", "0.7")
    monkeypatch.setenv("ENGRAM_AUTH__MODE", "saas")
    s = Settings()
    assert s.redis.port == 7777
    assert s.activation.decay_exponent == 0.7
    assert s.auth.mode == "saas"


def test_scoring_weights_must_sum_to_one():
    """Scoring weights that don't sum to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ActivationConfig(
            weight_semantic=0.5,
            weight_activation=0.5,
            weight_edge_proximity=0.5,
        )


def test_scoring_weights_valid():
    """Valid scoring weights pass validation."""
    cfg = ActivationConfig(
        weight_semantic=0.60,
        weight_activation=0.30,
        weight_edge_proximity=0.10,
    )
    assert cfg.weight_semantic == 0.60


def test_redis_url():
    """Redis URL is correctly constructed."""
    from engram.config import RedisConfig
    r = RedisConfig(host="myhost", port=9999, db=2, password="secret")
    assert r.url == "redis://:secret@myhost:9999/2"

    r2 = RedisConfig()
    assert r2.url == "redis://localhost:6381/0"


def test_redis_url_parsing():
    """Redis URL env var is parsed into individual fields."""
    from engram.config import RedisConfig
    r = RedisConfig(**{"url": "redis://:mypass@myhost:7777/3"})
    assert r.host == "myhost"
    assert r.port == 7777
    assert r.password == "mypass"
    assert r.db == 3


def test_auth_mode_validation():
    """Invalid auth mode raises ValueError."""
    with pytest.raises(ValueError, match="must be 'self_hosted' or 'saas'"):
        AuthConfig(mode="invalid")


def test_plaintext_secret_in_yaml_rejected():
    """Config loader rejects plaintext secrets in YAML data."""
    data = {"auth": {"bearer_token": "my-actual-secret"}}
    with pytest.raises(ValueError, match="plaintext value"):
        _check_plaintext_secrets(data)


def test_env_var_reference_in_yaml_accepted():
    """Config loader accepts ${...} env var references in YAML."""
    data = {"auth": {"bearer_token": "${ENGRAM_AUTH__BEARER_TOKEN}"}}
    _check_plaintext_secrets(data)  # should not raise


def test_empty_secret_in_yaml_accepted():
    """Empty secrets in YAML are fine -- env var override expected."""
    data = {"auth": {"bearer_token": ""}}
    _check_plaintext_secrets(data)  # should not raise


def test_additional_tokens():
    """Additional tokens parse correctly."""
    cfg = AuthConfig(
        additional_tokens=[
            {"name": "desktop", "token": "tok1", "group_id": "grp1"},
            {"name": "code", "token": "tok2", "group_id": "grp2"},
        ]
    )
    assert len(cfg.additional_tokens) == 2
    assert cfg.additional_tokens[0].name == "desktop"
    assert cfg.additional_tokens[1].group_id == "grp2"
```
