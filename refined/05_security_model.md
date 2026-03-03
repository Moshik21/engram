# 05 -- Auth Layer & Security Model

> Engram stores personal memories. Every design decision in this document starts
> from a single axiom: **an unauthenticated or cross-tenant memory leak is a
> ship-stopping bug, not a feature request.**

---

## 1. Threat Model

| Threat                              | Impact | Primary Control                        |
|-------------------------------------|--------|----------------------------------------|
| Unauthenticated access to memories  | Critical | Bearer token / API key on every request |
| Cross-tenant data leak (graph)      | Critical | Mandatory `TenantContext` + Cypher injection of `group_id` |
| Cross-tenant data leak (Redis)      | Critical | `{group_id}:` key prefix, keyspace isolation |
| PII exposure from disk/backup       | High   | Field-level encryption at rest          |
| Token/key theft                     | High   | Short-lived JWTs, key rotation, HTTPS-only |
| Brute-force / abuse                 | Medium | Rate limiting per key + IP              |
| XSS via dashboard                   | Medium | CSP headers, CORS allowlist             |
| GDPR Article 17 non-compliance      | High   | Hard-delete path with cascade           |

---

## 2. Authentication Models

Engram supports two deployment modes with different auth strategies.

### 2.1 Self-Hosted Mode (Bearer Token)

The simplest model: a single secret token in the config, validated on every request.

**Config fields** (coordinate with config schema):

```yaml
auth:
  mode: "self_hosted"           # "self_hosted" | "saas"
  bearer_token: "${ENGRAM_AUTH_SECRET}"   # loaded from env, never hardcoded
  # Optional: multiple tokens for multi-client setups
  additional_tokens:
    - name: "claude_desktop"
      token: "${ENGRAM_TOKEN_CLAUDE_DESKTOP}"
    - name: "claude_code"
      token: "${ENGRAM_TOKEN_CLAUDE_CODE}"
```

**Validation middleware:**

```python
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer_scheme = HTTPBearer()

async def verify_bearer_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    """Validate bearer token, return group_id.

    In self-hosted mode the bearer token maps directly to a group_id.
    The token-to-group mapping is loaded from config at startup.
    """
    token = credentials.credentials
    group_id = token_registry.get_group_id(token)
    if group_id is None:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return group_id
```

**Token registry** (in-memory, loaded from config at startup):

```python
class TokenRegistry:
    """Maps bearer tokens to group_ids. Loaded once at startup."""

    def __init__(self, config: AuthConfig):
        self._token_map: dict[str, str] = {}
        # Primary token always maps to "default" group
        self._token_map[config.bearer_token] = config.default_group_id
        for entry in config.additional_tokens:
            self._token_map[entry.token] = entry.group_id

    def get_group_id(self, token: str) -> str | None:
        return self._token_map.get(token)
```

### 2.2 SaaS Mode (API Key + JWT)

For the hosted service (Month 3+), two auth flows coexist:

| Client             | Auth Method        | Token Lifetime |
|--------------------|--------------------|----------------|
| MCP clients (Claude Desktop, etc.) | API key (header: `X-API-Key`) | Long-lived, rotatable |
| Dashboard (browser) | JWT (cookie: `engram_session`) | 15 min access, 7 day refresh |

**API Key flow:**

```
MCP Client                          Engram API
    │                                    │
    │── X-API-Key: ek_live_abc123 ──────>│
    │                                    │── lookup key in DB
    │                                    │── resolve group_id + permissions
    │                                    │── inject TenantContext
    │<── 200 OK ─────────────────────────│
```

**JWT flow (dashboard):**

```
Browser                   Engram API              Auth Provider
    │                          │                        │
    │── POST /auth/login ─────>│                        │
    │   {email, password}      │── verify credentials ─>│
    │                          │<── user_id, group_id ──│
    │                          │── sign JWT ────────────>│ (self-signed)
    │<── Set-Cookie: engram_session=ey... ──────────────│
    │                          │                        │
    │── GET /graph ───────────>│                        │
    │   Cookie: engram_session │── verify JWT signature  │
    │                          │── extract group_id      │
    │                          │── inject TenantContext   │
    │<── 200 OK ───────────────│                        │
```

**JWT payload structure:**

```json
{
  "sub": "user_abc123",
  "group_id": "grp_xyz789",
  "role": "owner",
  "iat": 1740700000,
  "exp": 1740700900
}
```

**Config fields for SaaS mode:**

```yaml
auth:
  mode: "saas"
  jwt:
    secret: "${ENGRAM_JWT_SECRET}"       # RS256 or HS256
    algorithm: "HS256"
    access_token_ttl_seconds: 900        # 15 minutes
    refresh_token_ttl_seconds: 604800    # 7 days
  api_key:
    prefix: "ek_live_"                   # all keys start with this
    hash_algorithm: "argon2id"           # keys stored hashed, never plaintext
```

---

## 3. Mandatory TenantContext Middleware

This is the single most important security control. **No request reaches a
handler without a resolved `TenantContext`.** The graph store, Redis client, and
retrieval engine all require it as a parameter -- they cannot be called without it.

### 3.1 TenantContext Model

```python
from pydantic import BaseModel

class TenantContext(BaseModel):
    """Immutable per-request tenant scope. Injected by auth middleware."""
    group_id: str
    user_id: str | None = None          # None in self-hosted mode
    role: str = "owner"                  # "owner" | "member" | "readonly"
    auth_method: str                     # "bearer" | "api_key" | "jwt"

    class Config:
        frozen = True                    # immutable after creation
```

### 3.2 Middleware Chain

```python
from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware

class TenantContextMiddleware(BaseHTTPMiddleware):
    """Resolves auth credentials into a TenantContext.

    EVERY request except health checks must pass through this.
    If auth fails, the request is rejected before reaching any handler.
    """

    EXEMPT_PATHS = {"/health", "/metrics", "/auth/login", "/auth/refresh"}

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Resolve auth -- raises HTTPException(401) on failure
        tenant = await self._resolve_tenant(request)

        # Attach to request state -- handlers access via get_tenant(request)
        request.state.tenant = tenant

        response = await call_next(request)
        return response

    async def _resolve_tenant(self, request: Request) -> TenantContext:
        """Try each auth method in order. First success wins."""
        # 1. Bearer token (self-hosted)
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return await self._resolve_bearer(auth_header[7:])

        # 2. API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._resolve_api_key(api_key)

        # 3. JWT cookie (dashboard)
        session_cookie = request.cookies.get("engram_session")
        if session_cookie:
            return await self._resolve_jwt(session_cookie)

        raise HTTPException(status_code=401, detail="No valid credentials")
```

### 3.3 Dependency Injection Pattern

Handlers never access auth directly. They receive `TenantContext` via FastAPI dependency injection:

```python
from fastapi import Depends

def get_tenant(request: Request) -> TenantContext:
    """Extract TenantContext from request. Fails loud if missing."""
    tenant = getattr(request.state, "tenant", None)
    if tenant is None:
        raise RuntimeError(
            "TenantContext missing -- this handler was not protected by "
            "TenantContextMiddleware. This is a bug."
        )
    return tenant

# Usage in route handlers:
@router.get("/graph")
async def get_graph(tenant: TenantContext = Depends(get_tenant)):
    return await graph_store.get_full_graph(tenant.group_id)
```

**Key guarantee:** If a handler receives a `TenantContext`, the request has been
authenticated. If a handler tries to query without one, it gets a `RuntimeError`
at development time, not a silent data leak.

---

## 4. Tenant Isolation: Graph Store

### 4.1 group_id in Every Cypher Query

The `GraphStore` class wraps all FalkorDB operations. It accepts `group_id` as a
required parameter and injects it into every Cypher query.

```python
class GraphStore:
    """All graph operations are tenant-scoped. group_id is mandatory."""

    async def get_entities(self, group_id: str, limit: int = 100) -> list[Entity]:
        query = """
            MATCH (e:Entity)
            WHERE e.group_id = $group_id
            RETURN e
            ORDER BY e.activation_current DESC
            LIMIT $limit
        """
        return await self._execute(query, {"group_id": group_id, "limit": limit})

    async def create_entity(self, group_id: str, entity: EntityCreate) -> Entity:
        query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                entity_type: $entity_type,
                summary: $summary,
                group_id: $group_id,
                created_at: datetime(),
                updated_at: datetime(),
                activation_base: 0.1,
                activation_current: 0.1,
                access_count: 0
            })
            RETURN e
        """
        params = {"group_id": group_id, **entity.model_dump(), "id": generate_id()}
        return await self._execute(query, params)

    async def get_neighbors(
        self, group_id: str, entity_id: str, hops: int = 1
    ) -> list[Entity]:
        query = """
            MATCH (e:Entity {id: $entity_id, group_id: $group_id})
                  -[r*1..$hops]-
                  (n:Entity {group_id: $group_id})
            RETURN DISTINCT n
        """
        return await self._execute(
            query, {"entity_id": entity_id, "group_id": group_id, "hops": hops}
        )

    async def delete_entity_hard(self, group_id: str, entity_id: str) -> bool:
        """GDPR Article 17 hard delete. Removes node and all edges."""
        query = """
            MATCH (e:Entity {id: $entity_id, group_id: $group_id})
            DETACH DELETE e
        """
        result = await self._execute(
            query, {"entity_id": entity_id, "group_id": group_id}
        )
        return result.nodes_deleted > 0
```

**Design rule:** The `GraphStore` class has no method that omits `group_id`.
There is no `get_all_entities()` without a group scope. A developer who tries
to query across tenants must bypass the entire class, which code review catches.

### 4.2 FalkorDB Index for group_id

To ensure `WHERE group_id = $group_id` is fast and not a full scan:

```cypher
CREATE INDEX FOR (e:Entity) ON (e.group_id)
CREATE INDEX FOR (ep:Episode) ON (ep.group_id)
```

These are created at startup by the migration runner.

---

## 5. Tenant Isolation: Redis Activation State

### 5.1 Key Prefix Strategy

All Redis keys are prefixed with `{group_id}:` to achieve both isolation and
Redis cluster hash-slot locality (the `{}` ensures keys for the same tenant
land on the same shard).

```python
class ActivationStateStore:
    """Redis-backed activation state with mandatory tenant isolation."""

    def __init__(self, redis: Redis):
        self._redis = redis

    def _key(self, group_id: str, entity_id: str) -> str:
        """Build a tenant-scoped Redis key.

        Format: {group_id}:activation:{entity_id}
        The {group_id} braces ensure cluster hash-slot locality.
        """
        return f"{{{group_id}}}:activation:{entity_id}"

    def _pattern(self, group_id: str) -> str:
        """Pattern for scanning all activation keys for a tenant."""
        return f"{{{group_id}}}:activation:*"

    async def get_activation(
        self, group_id: str, entity_id: str
    ) -> ActivationState | None:
        key = self._key(group_id, entity_id)
        data = await self._redis.hgetall(key)
        if not data:
            return None
        return ActivationState.from_redis(data)

    async def set_activation(
        self, group_id: str, entity_id: str, state: ActivationState
    ) -> None:
        key = self._key(group_id, entity_id)
        await self._redis.hset(key, mapping=state.to_redis())

    async def get_top_activated(
        self, group_id: str, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Get top-N activated entities for a tenant.

        Uses a sorted set per tenant for O(log N) ranked lookups.
        """
        zset_key = f"{{{group_id}}}:activation_rank"
        return await self._redis.zrevrange(zset_key, 0, limit - 1, withscores=True)

    async def delete_all_for_tenant(self, group_id: str) -> int:
        """GDPR: delete all activation state for a tenant."""
        pattern = self._pattern(group_id)
        deleted = 0
        async for key in self._redis.scan_iter(match=pattern, count=500):
            await self._redis.delete(key)
            deleted += 1
        # Also delete the ranked sorted set
        await self._redis.delete(f"{{{group_id}}}:activation_rank")
        return deleted
```

### 5.2 No Cross-Tenant Key Access

The `_key()` method always requires `group_id`. There is no way to construct
a key without it. A missing or empty `group_id` would produce an invalid key
format caught by validation:

```python
class TenantContext(BaseModel):
    group_id: str

    @field_validator("group_id")
    @classmethod
    def group_id_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("group_id cannot be empty")
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("group_id must be alphanumeric (with hyphens/underscores)")
        return v
```

---

## 6. MCP Transport Auth

MCP clients (Claude Desktop, Claude Code) connect via HTTP/SSE. Auth must be
injected into the transport layer.

### 6.1 HTTP/SSE Transport

```
MCP Client ──(HTTP)──> Engram MCP Server
                        │
                        ├─ Authorization: Bearer <token>
                        │  (injected via MCP client config)
                        │
                        └─ Same TenantContextMiddleware applies
```

**Claude Desktop `claude_desktop_config.json`:**

```json
{
  "mcpServers": {
    "engram": {
      "url": "http://localhost:8787/mcp",
      "headers": {
        "Authorization": "Bearer ${ENGRAM_AUTH_SECRET}"
      }
    }
  }
}
```

### 6.2 stdio Transport (via mcp-remote)

For clients that only support stdio, `mcp-remote` bridges to HTTP/SSE.
The bearer token is configured in the mcp-remote invocation:

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:8787/mcp",
        "--header", "Authorization: Bearer ${ENGRAM_AUTH_SECRET}"
      ]
    }
  }
}
```

### 6.3 MCP Server Auth Enforcement

The MCP server handler extracts the `TenantContext` before dispatching any tool call:

```python
from mcp.server import Server
from mcp.types import TextContent

app = Server("engram")

@app.tool()
async def recall(query: str, limit: int = 5, *, request: Request) -> list[TextContent]:
    """Activation-aware memory retrieval."""
    tenant = get_tenant(request)  # same dependency as REST routes
    results = await retrieval_engine.recall(
        group_id=tenant.group_id,
        query=query,
        limit=limit,
    )
    return [TextContent(type="text", text=format_results(results))]
```

**Every MCP tool receives the tenant context.** A tool that tries to operate
without it fails at compile time (missing required parameter).

---

## 7. PII Handling & Encryption at Rest

### 7.1 Field-Level Encryption

Episode content and entity summaries may contain PII. These fields are encrypted
at rest using AES-256-GCM with per-tenant encryption keys.

```python
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class FieldEncryptor:
    """Encrypts/decrypts individual fields with per-tenant keys."""

    def __init__(self, master_key: bytes):
        """master_key: 32-byte key loaded from ENGRAM_MASTER_KEY env var."""
        self._master_key = master_key

    def _derive_tenant_key(self, group_id: str) -> bytes:
        """Derive a per-tenant key from the master key using HKDF."""
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes

        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=f"engram-tenant-{group_id}".encode(),
        )
        return hkdf.derive(self._master_key)

    def encrypt(self, group_id: str, plaintext: str) -> bytes:
        """Encrypt a field value. Returns nonce || ciphertext."""
        key = self._derive_tenant_key(group_id)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        return nonce + ciphertext

    def decrypt(self, group_id: str, data: bytes) -> str:
        """Decrypt a field value."""
        key = self._derive_tenant_key(group_id)
        aesgcm = AESGCM(key)
        nonce = data[:12]
        ciphertext = data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
```

**Encrypted fields:**

| Model    | Field          | Encrypted |
|----------|----------------|-----------|
| Episode  | `content`      | Yes       |
| Entity   | `summary`      | Yes       |
| Entity   | `name`         | No (needed for graph queries, not PII by itself) |
| Entity   | `entity_type`  | No        |
| Edge     | `predicate`    | No        |

### 7.2 PII Detection Flag

During entity extraction, the Claude extraction prompt includes a PII classification step:

```python
EXTRACTION_PROMPT_SUFFIX = """
For each entity and fact extracted, also classify:
- pii_detected: boolean (true if the content contains names, addresses,
  phone numbers, health info, financial info, or other personally
  identifiable information)
- pii_categories: list[string] (e.g., ["name", "health", "financial"])
"""
```

The `pii_detected` flag enables:
- Dashboard UI warnings ("This entity contains PII")
- Selective encryption (only encrypt PII-flagged content in lite mode)
- GDPR data subject access requests (quickly find all PII for a user)

### 7.3 Config for Encryption

```yaml
encryption:
  enabled: true                              # false for dev/testing
  master_key: "${ENGRAM_MASTER_KEY}"         # 32-byte hex, loaded from env
  encrypt_all_content: true                  # encrypt episode.content always
  encrypt_pii_only: false                    # alternative: only encrypt pii_detected fields
```

---

## 8. GDPR Compliance

### 8.1 Article 17: Right to Erasure (Hard Delete)

Engram implements a complete hard-delete cascade. When a user requests erasure:

```python
class GDPRErasureService:
    """Implements GDPR Article 17 hard delete across all stores."""

    async def erase_tenant(self, group_id: str) -> ErasureReport:
        """Delete ALL data for a tenant. Irreversible."""
        report = ErasureReport(group_id=group_id)

        # 1. Delete all graph data (entities, episodes, edges)
        report.graph_nodes_deleted = await self.graph_store.delete_all_for_tenant(
            group_id
        )

        # 2. Delete all activation state from Redis
        report.redis_keys_deleted = await self.activation_store.delete_all_for_tenant(
            group_id
        )

        # 3. Delete all embedding vectors
        report.vectors_deleted = await self.embedding_store.delete_all_for_tenant(
            group_id
        )

        # 4. Delete API keys and auth records (SaaS mode)
        report.auth_records_deleted = await self.auth_store.delete_all_for_tenant(
            group_id
        )

        # 5. Log the erasure event (keep audit log, but no PII in it)
        await self.audit_log.record_erasure(
            group_id=group_id,
            timestamp=datetime.utcnow(),
            summary=report.model_dump(),
        )

        return report

    async def erase_entity(self, group_id: str, entity_id: str) -> ErasureReport:
        """Delete a single entity and its edges. For selective erasure."""
        report = ErasureReport(group_id=group_id)

        # Hard delete from graph (DETACH DELETE removes edges too)
        report.graph_nodes_deleted = await self.graph_store.delete_entity_hard(
            group_id, entity_id
        )

        # Delete activation state
        await self.activation_store.delete_activation(group_id, entity_id)
        report.redis_keys_deleted = 1

        # Delete embedding vector
        await self.embedding_store.delete_vector(group_id, entity_id)
        report.vectors_deleted = 1

        return report
```

### 8.2 Article 15: Right of Access (Data Export)

```python
async def export_tenant_data(self, group_id: str) -> TenantDataExport:
    """Export all data for a tenant in machine-readable format."""
    return TenantDataExport(
        entities=await self.graph_store.get_entities(group_id, limit=None),
        episodes=await self.graph_store.get_episodes(group_id, limit=None),
        relationships=await self.graph_store.get_relationships(group_id, limit=None),
        activation_states=await self.activation_store.get_all(group_id),
        exported_at=datetime.utcnow(),
    )
```

### 8.3 GDPR Compliance Checklist

| Requirement                         | Implementation                                    | Status    |
|-------------------------------------|---------------------------------------------------|-----------|
| Lawful basis for processing         | Explicit consent at signup / self-hosted = user controls own data | Design |
| Data minimization                   | Only extract entities relevant to memory function | Design |
| Right to access (Art. 15)           | `GET /gdpr/export` returns all tenant data as JSON | To build |
| Right to erasure (Art. 17)          | `DELETE /gdpr/erase` hard-deletes across all stores | To build |
| Right to rectification (Art. 16)    | Dashboard entity edit + `PATCH /entities/{id}`    | To build |
| Data portability (Art. 20)          | JSON export of full graph                         | To build |
| Encryption at rest                  | AES-256-GCM per-tenant key derivation             | Design |
| Breach notification                 | Audit log + alerting (SaaS mode)                  | To build |
| Data processing agreement           | Required for SaaS tier, template in docs          | To draft |
| Privacy policy                      | Required for SaaS, hosted dashboard               | To draft |

---

## 9. CORS Policy

### 9.1 Self-Hosted

In self-hosted mode, the dashboard and API typically run on the same host.
CORS is configured to allow the dashboard origin only:

```python
from fastapi.middleware.cors import CORSMiddleware

def configure_cors(app: FastAPI, config: CORSConfig):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,  # e.g., ["http://localhost:3000"]
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "X-API-Key", "Content-Type"],
        max_age=3600,
    )
```

**Config fields:**

```yaml
cors:
  allowed_origins:
    - "http://localhost:3000"       # dashboard dev
    - "http://localhost:5173"       # vite dev server
  # In production, restrict to the actual dashboard URL
```

### 9.2 SaaS Mode

```yaml
cors:
  allowed_origins:
    - "https://app.engram.dev"
    - "https://engram.dev"
```

### 9.3 Additional Security Headers

Applied via middleware on all responses:

```python
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"  # CSP is preferred
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' ws://localhost:* wss://localhost:*; "
            "img-src 'self' data: blob:; "
        )
        return response
```

---

## 10. Rate Limiting

### 10.1 Strategy

Rate limiting is per-API-key (SaaS) or per-IP (self-hosted) using a sliding
window counter in Redis.

```python
from fastapi import Request
import time

class RateLimiter:
    """Sliding window rate limiter backed by Redis."""

    def __init__(self, redis: Redis, config: RateLimitConfig):
        self._redis = redis
        self._config = config

    async def check(self, identifier: str, endpoint_class: str) -> bool:
        """Returns True if request is allowed, False if rate-limited."""
        limits = self._config.limits[endpoint_class]
        key = f"ratelimit:{identifier}:{endpoint_class}"
        now = time.time()
        window = limits.window_seconds

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, window)
        _, _, count, _ = await pipe.execute()

        return count <= limits.max_requests
```

### 10.2 Default Limits

```yaml
rate_limiting:
  enabled: true
  limits:
    ingestion:
      max_requests: 60          # remember tool
      window_seconds: 60
    retrieval:
      max_requests: 120         # recall, search tools
      window_seconds: 60
    admin:
      max_requests: 30          # GDPR, config endpoints
      window_seconds: 60
    websocket:
      max_connections: 5        # per tenant
```

### 10.3 Rate Limit Headers

```
X-RateLimit-Limit: 120
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1740700060
Retry-After: 12                 # only on 429 responses
```

---

## 11. Secrets Management

### 11.1 Hierarchy: Environment Variables Over Config Files

```
Priority (highest to lowest):
1. Environment variables (ENGRAM_AUTH_SECRET, ENGRAM_JWT_SECRET, etc.)
2. .env file (loaded via python-dotenv, gitignored)
3. config.yaml (non-secret values only)
```

**Sensitive fields that MUST come from env vars:**

| Variable                   | Purpose                           |
|----------------------------|-----------------------------------|
| `ENGRAM_AUTH_SECRET`      | Self-hosted auth token            |
| `ENGRAM_JWT_SECRET`        | JWT signing key (SaaS)           |
| `ENGRAM_MASTER_KEY`        | PII encryption master key        |
| `ENGRAM_ANTHROPIC_API_KEY` | Claude API key for extraction    |
| `ENGRAM_REDIS_PASSWORD`    | Redis auth (if enabled)          |

### 11.2 Config Loader Validation

```python
class ConfigLoader:
    """Loads config with env var interpolation. Refuses to start with secrets in YAML."""

    @staticmethod
    def load(config_path: str) -> EngramConfig:
        raw = yaml.safe_load(open(config_path))
        config = EngramConfig(**raw)

        # Validate no plaintext secrets in YAML
        secret_fields = ["bearer_token", "jwt_secret", "master_key", "anthropic_api_key"]
        for field in secret_fields:
            value = config.get_nested(field)
            if value and not value.startswith("${"):
                raise ConfigError(
                    f"Secret field '{field}' must reference an env var "
                    f"(e.g., '${{ENGRAM_{field.upper()}}}'), not a plaintext value."
                )

        # Resolve env var references
        config = config.resolve_env_vars()
        return config
```

### 11.3 Docker Secrets Integration

For Docker Compose deployments, secrets are passed via environment:

```yaml
# docker-compose.yml
services:
  engram-server:
    image: engram/server:latest
    environment:
      - ENGRAM_AUTH_SECRET=${ENGRAM_AUTH_SECRET}
      - ENGRAM_ANTHROPIC_API_KEY=${ENGRAM_ANTHROPIC_API_KEY}
      - ENGRAM_MASTER_KEY=${ENGRAM_MASTER_KEY}
    env_file:
      - .env        # gitignored, contains secrets
```

For production Docker Swarm / Kubernetes:

```yaml
# docker-compose.prod.yml (swarm secrets)
services:
  engram-server:
    secrets:
      - engram_auth_secret
      - engram_master_key
    environment:
      - ENGRAM_AUTH_SECRET_FILE=/run/secrets/engram_auth_secret
      - ENGRAM_MASTER_KEY_FILE=/run/secrets/engram_master_key

secrets:
  engram_auth_secret:
    external: true
  engram_master_key:
    external: true
```

The config loader supports both `ENGRAM_X` and `ENGRAM_X_FILE` (Docker secrets file mount pattern):

```python
def resolve_secret(name: str) -> str:
    """Resolve a secret from env var or Docker secret file."""
    # Try direct env var first
    value = os.environ.get(name)
    if value:
        return value
    # Try Docker secrets file mount
    file_path = os.environ.get(f"{name}_FILE")
    if file_path and os.path.exists(file_path):
        return open(file_path).read().strip()
    raise ConfigError(f"Required secret '{name}' not found in env or Docker secrets")
```

### 11.4 .gitignore Protections

The repo `.gitignore` must include:

```
# Secrets -- never commit
.env
.env.*
config.yaml
config.local.yaml
*.pem
*.key

# Only config.example.yaml should be committed
!config.example.yaml
```

---

## 12. Middleware Stack Order

The order of middleware matters. FastAPI executes middleware in reverse registration
order (last registered = outermost = first to run):

```python
def create_app(config: EngramConfig) -> FastAPI:
    app = FastAPI(title="Engram", version="0.1.0")

    # 1. Security headers (outermost -- always applied)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. CORS (must be before auth to handle preflight OPTIONS)
    configure_cors(app, config.cors)

    # 3. Rate limiting (before auth to prevent brute force)
    app.add_middleware(RateLimitMiddleware, config=config.rate_limiting)

    # 4. Tenant context / auth (innermost -- closest to handlers)
    app.add_middleware(TenantContextMiddleware, config=config.auth)

    return app
```

Execution order for an incoming request:
```
Request → SecurityHeaders → CORS → RateLimit → TenantContext → Handler
Response ← SecurityHeaders ← CORS ← RateLimit ← TenantContext ← Handler
```

---

## 13. Audit Logging

Security-relevant events are logged for both self-hosted (stdout/file) and
SaaS (structured log aggregation) deployments:

```python
import structlog

audit_log = structlog.get_logger("engram.audit")

# Events to log:
audit_log.info("auth.success", group_id=tenant.group_id, method="bearer")
audit_log.warning("auth.failure", ip=request.client.host, reason="invalid_token")
audit_log.info("data.created", group_id=group_id, entity_id=entity_id)
audit_log.info("data.deleted", group_id=group_id, entity_id=entity_id, gdpr=True)
audit_log.warning("rate_limit.exceeded", identifier=api_key_prefix, endpoint="recall")
```

**Audit log entries never contain PII.** They reference `group_id` and `entity_id`
but never entity names, content, or user data.

---

## 14. Summary of Config Fields for Auth & Security

These fields should be added to the Pydantic config schema:

```yaml
auth:
  mode: "self_hosted"                          # "self_hosted" | "saas"
  bearer_token: "${ENGRAM_AUTH_SECRET}"
  default_group_id: "default"
  additional_tokens: []                        # list of {name, token, group_id}
  jwt:                                         # SaaS mode only
    secret: "${ENGRAM_JWT_SECRET}"
    algorithm: "HS256"
    access_token_ttl_seconds: 900
    refresh_token_ttl_seconds: 604800
  api_key:                                     # SaaS mode only
    prefix: "ek_live_"
    hash_algorithm: "argon2id"

encryption:
  enabled: true
  master_key: "${ENGRAM_MASTER_KEY}"
  encrypt_all_content: true

cors:
  allowed_origins:
    - "http://localhost:3000"

rate_limiting:
  enabled: true
  limits:
    ingestion: { max_requests: 60, window_seconds: 60 }
    retrieval: { max_requests: 120, window_seconds: 60 }
    admin: { max_requests: 30, window_seconds: 60 }
    websocket: { max_connections: 5 }
```

---

## 15. Implementation Priority

| Priority | Component                        | Week  |
|----------|----------------------------------|-------|
| P0       | TenantContext middleware          | 1     |
| P0       | group_id in all Cypher queries   | 1     |
| P0       | Bearer token auth (self-hosted)  | 1     |
| P0       | Redis key prefix isolation       | 3     |
| P0       | .gitignore / secrets from env    | 1     |
| P1       | CORS configuration               | 5     |
| P1       | Rate limiting                    | 4     |
| P1       | Security headers                 | 5     |
| P1       | PII encryption at rest           | 2     |
| P2       | GDPR hard-delete path            | 6     |
| P2       | GDPR data export                 | 6     |
| P2       | API key auth (SaaS)              | Month 3 |
| P2       | JWT auth (SaaS dashboard)        | Month 3 |
| P2       | Audit logging                    | 7     |

P0 items are Week 1 blockers. Nothing ships without tenant isolation and auth.

---

## 16. Cross-Document Alignment: DevOps Conventions

The DevOps design (`08_devops_infrastructure.md`) establishes specific env var
naming conventions. This section reconciles auth/security env vars with those
conventions.

### 16.1 Env Var Name Mapping (Aligned)

Both documents now use `ENGRAM_AUTH_SECRET` as the canonical env var for the
self-hosted auth token. This was the devops-established convention; the security
doc adopted it for consistency. All names below are the final, agreed-upon names.

| Env Var                      | Purpose                              | Default       | Required          |
|------------------------------|--------------------------------------|---------------|-------------------|
| `ENGRAM_AUTH_SECRET`        | Self-hosted auth token               | *(empty)*     | When auth enabled |
| `ENGRAM_AUTH_ENABLED`        | Enable/disable auth middleware       | `false`       | No                |
| `ENGRAM_MASTER_KEY`          | PII encryption master key (32B hex)  | *(empty)*     | When encryption enabled |
| `ENGRAM_JWT_SECRET`          | JWT signing key (SaaS mode)          | *(empty)*     | SaaS only         |
| `ENGRAM_ANTHROPIC_API_KEY`   | Claude API key for extraction        | *(required)*  | Yes               |
| `ENGRAM_FALKORDB_PASSWORD`   | FalkorDB auth                        | `changeme`    | Yes               |
| `ENGRAM_REDIS_PASSWORD`      | Redis auth                           | `changeme`    | Yes               |

All secret vars support the `_FILE` suffix for Docker Swarm mounts (see section 11.3).

### 16.2 Auth Disabled by Default in Development

The devops doc sets `ENGRAM_AUTH_ENABLED=false` by default, which means the
`TenantContextMiddleware` must handle this case:

```python
async def _resolve_tenant(self, request: Request) -> TenantContext:
    if not self._config.auth_enabled:
        # Auth disabled -- use default tenant for local dev
        return TenantContext(
            group_id=self._config.default_group_id,  # "default"
            user_id=None,
            role="owner",
            auth_method="none",
        )
    # ... normal auth resolution ...
```

This preserves the zero-config developer experience (`docker compose up` just
works) while ensuring the middleware always produces a `TenantContext`. The
`group_id` is still injected into all queries even when auth is off, so tenant
isolation logic is always exercised.

### 16.3 Production Secrets Migration Path

The devops doc (section 10) flagged a gap: for hosted/SaaS deployments, `.env`
files are insufficient. Here is the migration path from `.env` to a production
secrets manager:

**Tier 1: `.env` file (self-hosted, personal use)**
- Secrets in `.env` file, gitignored
- Acceptable for single-user, local-only deployments
- The `config.example.yaml` / `.env.example` files document all required vars

**Tier 2: Docker Swarm secrets (small team / self-hosted production)**
- Secrets mounted as files at `/run/secrets/<name>`
- The config loader supports `ENGRAM_<VAR>_FILE` convention (section 11.3)
- No code changes needed -- the `resolve_secret()` function handles both patterns

```yaml
# docker-compose.prod.yml
services:
  server:
    secrets:
      - engram_auth_secret
      - engram_master_key
      - anthropic_api_key
    environment:
      - ENGRAM_AUTH_SECRET_FILE=/run/secrets/engram_auth_secret
      - ENGRAM_MASTER_KEY_FILE=/run/secrets/engram_master_key
      - ANTHROPIC_API_KEY_FILE=/run/secrets/anthropic_api_key
secrets:
  engram_auth_secret:
    external: true
  engram_master_key:
    external: true
  anthropic_api_key:
    external: true
```

**Tier 3: HashiCorp Vault / cloud-native (SaaS production)**
- Secrets fetched at startup from Vault, AWS Secrets Manager, or GCP Secret Manager
- Implemented as a pluggable `SecretProvider` interface:

```python
from abc import ABC, abstractmethod

class SecretProvider(ABC):
    @abstractmethod
    async def get_secret(self, name: str) -> str: ...

class EnvSecretProvider(SecretProvider):
    """Tier 1 & 2: env vars and Docker secret files."""
    async def get_secret(self, name: str) -> str:
        return resolve_secret(name)  # existing function from section 11.3

class VaultSecretProvider(SecretProvider):
    """Tier 3: HashiCorp Vault."""
    def __init__(self, vault_addr: str, vault_token: str):
        self._client = hvac.Client(url=vault_addr, token=vault_token)

    async def get_secret(self, name: str) -> str:
        result = self._client.secrets.kv.v2.read_secret_version(
            path=f"engram/{name}"
        )
        return result["data"]["data"]["value"]
```

The `SecretProvider` is selected by config:

```yaml
secrets:
  provider: "env"      # "env" | "vault" | "aws_sm" | "gcp_sm"
  vault:               # only when provider=vault
    addr: "https://vault.internal:8200"
    auth_method: "kubernetes"  # or "token", "approle"
```

**Implementation timeline:**
- Tier 1 (`.env`): Week 1 -- ships with MVP
- Tier 2 (Docker Swarm): Week 7 -- ships with production Docker polish
- Tier 3 (Vault/cloud): Month 3 -- ships with SaaS launch
