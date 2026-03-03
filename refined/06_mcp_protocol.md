# 06 -- MCP Protocol Design

> Refined spec for Engram's MCP server: tool schemas, resources, prompts, session scoping, transport, automatic ingestion, and error handling.

---

## 1. Transport Architecture

### 1.1 Native stdio (Primary)

Engram ships as a **native stdio MCP server** using the official `mcp` Python SDK (`mcp.server.fastmcp`). No `mcp-remote` bridge required.

```
Claude Desktop / Claude Code / Cursor
         |  stdin/stdout (JSON-RPC 2.0)
         v
   engram-mcp (Python process)
         |
    FastAPI internal bus
         |
   +-----+------+--------+
   |      |      |        |
FalkorDB Redis  Embed   Claude API
```

**Entry point:** `server/engram/mcp/server.py`

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "engram",
    version="0.1.0",
    capabilities={
        "tools": {},
        "resources": {"subscribe": True},
        "prompts": {},
    },
)

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

### 1.2 Claude Desktop Configuration (Local stdio -- Development)

For local development, native stdio with no auth. The user's filesystem access is the trust boundary.

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
      "env": {
        "ENGRAM_GROUP_ID": "personal",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

### 1.3 Claude Desktop Configuration (HTTP/SSE -- Authenticated)

For hosted or multi-user deployments, Claude Desktop connects via HTTP/SSE with bearer token auth. Uses `mcp-remote` to bridge stdio-only clients to the HTTP transport.

```json
{
  "mcpServers": {
    "engram": {
      "command": "npx",
      "args": [
        "mcp-remote",
        "http://localhost:8787/mcp",
        "--header",
        "Authorization: Bearer ${ENGRAM_BEARER_TOKEN}"
      ]
    }
  }
}
```

### 1.4 Claude Code Configuration (Local stdio -- Development)

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "python", "-m", "engram.mcp.server"],
      "env": {
        "ENGRAM_GROUP_ID": "work",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

### 1.5 Streamable HTTP (Hosted / Dashboard / Multi-User)

For the hosted SaaS tier, multi-user deployments, and dashboard live-query features, Engram exposes the MCP protocol over Streamable HTTP at `/mcp`. Auth uses the same `TenantContextMiddleware` as the REST API -- a single security surface for both MCP and REST.

```
Dashboard / Remote Client / mcp-remote bridge
         |  HTTP POST + SSE (JSON-RPC 2.0)
         |  Authorization: Bearer <token>
         v
   FastAPI /mcp endpoint
         |  TenantContextMiddleware (same as REST)
         |  -> resolves TenantContext { group_id, scopes }
         v
   (same internal bus as stdio)
```

### 1.6 Transport Decision Matrix

| Scenario | Transport | Auth | group_id Source |
|----------|-----------|------|-----------------|
| Local dev (single user) | Native stdio | None (filesystem trust) | `ENGRAM_GROUP_ID` env var |
| Hosted / multi-user | HTTP/SSE | Bearer token | JWT `group_id` claim |
| stdio client -> hosted server | mcp-remote bridge -> HTTP/SSE | Bearer token via `--header` | JWT `group_id` claim |

---

## 2. Session and Conversation Scoping

### 2.1 group_id Flow

`group_id` isolates memory graphs per tenant/context. It never appears as a tool parameter -- the LLM does not manage scoping. Resolution depends on transport:

```
Local stdio:
   ENGRAM_GROUP_ID env var -> server reads on startup -> injected into all ops

HTTP/SSE (hosted):
   Bearer token -> TenantContextMiddleware -> JWT group_id claim -> injected into all ops
```

Both paths converge to the same internal `TenantContext` that is passed to every graph/Redis/retrieval operation. All Cypher queries include `WHERE group_id = $gid`.

| Source | Transport | Mechanism | Example |
|--------|-----------|-----------|---------|
| Local dev | stdio | `ENGRAM_GROUP_ID` env var | `"ENGRAM_GROUP_ID": "personal"` |
| Hosted | HTTP/SSE | JWT `group_id` claim via `TenantContextMiddleware` | `group_id = token.group_id` |
| stdio -> hosted | mcp-remote bridge | Bearer token `--header` -> same JWT path | `--header "Authorization: Bearer $TOKEN"` |

### 2.2 Conversation Tracking

Each MCP connection corresponds to one conversation session. The server generates a `session_id` (UUID v4) on initialization and attaches it to every episode created during that session.

```python
class SessionState:
    session_id: str          # UUID v4, generated on init
    group_id: str            # from env or auth token
    started_at: datetime
    episode_count: int       # episodes created this session
    last_activity: datetime
```

This lets the dashboard show "which conversation produced which memories" and enables per-session replay.

---

## 3. MCP Tool Definitions (7 Tools)

All tools use JSON Schema for `inputSchema` per MCP spec. Response shapes use `TextContent` with JSON-serialized payloads.

### 3.1 `remember`

Ingest an episode. Triggers async entity/relationship extraction.

**Definition:**

```json
{
  "name": "remember",
  "description": "Store a memory from the current conversation. Extracts entities, relationships, and facts automatically. Call this after each meaningful exchange to build persistent memory.",
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "The text content to remember. Include both the user message and your response for full context."
      },
      "source": {
        "type": "string",
        "description": "Source identifier for this memory.",
        "enum": ["claude_desktop", "claude_code", "cursor", "api", "other"],
        "default": "claude_desktop"
      },
      "metadata": {
        "type": "object",
        "description": "Optional key-value metadata to attach to this episode.",
        "additionalProperties": {
          "type": "string"
        }
      }
    },
    "required": ["content"]
  }
}
```

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "remember",
    "arguments": {
      "content": "User: I've been working on ReadyCheck, trying to get the Stripe integration done this week.\nAssistant: That sounds like good progress on ReadyCheck! Stripe integration is a solid next step for your meeting prep SaaS.",
      "source": "claude_desktop"
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"queued\",\"episode_id\":\"ep_01HXYZ9ABCDEF1234567890\",\"message\":\"Memory received and queued for processing.\",\"status_url\":\"/api/episodes/ep_01HXYZ9ABCDEF1234567890/status\"}"
      }
    ]
  }
}
```

The response returns immediately (~10ms) after the episode is persisted. Entity extraction runs asynchronously via Redis Streams (see ingestion pipeline spec 03). The LLM does not wait for extraction.

**Response JSON shape:**

```json
{
  "status": "queued | duplicate",
  "episode_id": "string (ULID, ep_ prefixed)",
  "message": "string",
  "status_url": "string (REST endpoint for polling extraction progress, dashboard use only)"
}
```

Episode IDs use ULIDs (Universally Unique Lexicographically Sortable Identifiers) for natural time-ordering. Processing flows through states: `queued -> extracting -> resolving -> writing -> activating -> embedding -> completed` (or `dead_letter` on failure). The `status_url` is a REST endpoint for the dashboard, not an MCP tool.

---

### 3.2 `recall`

Activation-aware memory retrieval. The core differentiator.

**Definition:**

```json
{
  "name": "recall",
  "description": "Retrieve relevant memories using activation-based scoring. Combines semantic similarity, recency, frequency, and spreading activation to surface the most contextually relevant memories.",
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query describing what to remember."
      },
      "top_k": {
        "type": "integer",
        "description": "Maximum number of results to return.",
        "default": 5,
        "minimum": 1,
        "maximum": 20
      },
      "min_score": {
        "type": "number",
        "description": "Minimum composite score threshold (0.0 to 1.0). Results below this are excluded.",
        "default": 0.1,
        "minimum": 0.0,
        "maximum": 1.0
      }
    },
    "required": ["query"]
  }
}
```

Note: `readOnlyHint` is `false` because recall updates activation state (access counts, timestamps, spreading activation) as a side effect of retrieval.

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "recall",
    "arguments": {
      "query": "What was I doing with payments?",
      "top_k": 5
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"results\":[{\"entity\":\"Stripe\",\"entity_type\":\"Tool/Technology\",\"summary\":\"Payment processing platform being integrated into ReadyCheck\",\"activation_score\":0.82,\"composite_score\":0.78,\"score_breakdown\":{\"semantic_similarity\":0.85,\"activation\":0.82,\"recency\":0.71,\"frequency\":0.65},\"related_facts\":[{\"subject\":\"ReadyCheck\",\"predicate\":\"INTEGRATES\",\"object\":\"Stripe\",\"valid_from\":\"2026-02-20T00:00:00Z\",\"valid_to\":null,\"source_episode\":\"ep_a1b2c3d4\"}],\"last_accessed\":\"2026-02-26T14:30:00Z\",\"access_count\":7},{\"entity\":\"ReadyCheck\",\"entity_type\":\"Project\",\"summary\":\"Meeting prep SaaS with 8-week launch target\",\"activation_score\":0.75,\"composite_score\":0.72,\"score_breakdown\":{\"semantic_similarity\":0.60,\"activation\":0.75,\"recency\":0.80,\"frequency\":0.70},\"related_facts\":[{\"subject\":\"User\",\"predicate\":\"WORKS_ON\",\"object\":\"ReadyCheck\",\"valid_from\":\"2026-01-15T00:00:00Z\",\"valid_to\":null,\"source_episode\":\"ep_e5f6g7h8\"}],\"last_accessed\":\"2026-02-27T09:00:00Z\",\"access_count\":12}],\"total_candidates\":14,\"activation_hops\":2,\"query_time_ms\":45}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "results": [
    {
      "entity": "string",
      "entity_type": "string (Person | Organization | Project | Concept | Preference | Location | Event | Tool/Technology)",
      "summary": "string",
      "activation_score": "number (0.0-1.0)",
      "composite_score": "number (0.0-1.0)",
      "score_breakdown": {
        "semantic_similarity": "number (0.0-1.0, weight 0.4)",
        "activation": "number (0.0-1.0, weight 0.3)",
        "recency": "number (0.0-1.0, weight 0.2)",
        "frequency": "number (0.0-1.0, weight 0.1)"
      },
      "related_facts": [
        {
          "subject": "string",
          "predicate": "string",
          "object": "string",
          "valid_from": "string (ISO 8601)",
          "valid_to": "string | null",
          "source_episode": "string"
        }
      ],
      "last_accessed": "string (ISO 8601)",
      "access_count": "integer"
    }
  ],
  "total_candidates": "integer",
  "activation_hops": "integer",
  "query_time_ms": "integer"
}
```

---

### 3.3 `search_entities`

Direct entity lookup by name or type. Does not trigger spreading activation.

**Definition:**

```json
{
  "name": "search_entities",
  "description": "Search for specific entities by name or type. Returns matching entities with their current activation state. Use this for direct lookups rather than contextual recall.",
  "annotations": {
    "readOnlyHint": true,
    "destructiveHint": false,
    "idempotentHint": true,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Entity name to search for. Supports fuzzy matching."
      },
      "entity_type": {
        "type": "string",
        "description": "Filter by entity type.",
        "enum": ["Person", "Organization", "Project", "Concept", "Preference", "Location", "Event", "Tool/Technology"]
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of results.",
        "default": 10,
        "minimum": 1,
        "maximum": 50
      }
    },
    "required": []
  }
}
```

At least one of `name` or `entity_type` should be provided. The server returns an error if both are omitted.

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search_entities",
    "arguments": {
      "name": "ReadyCheck",
      "entity_type": "Project"
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"entities\":[{\"id\":\"ent_x1y2z3\",\"name\":\"ReadyCheck\",\"entity_type\":\"Project\",\"summary\":\"Meeting prep SaaS with 8-week launch target\",\"activation_score\":0.75,\"created_at\":\"2026-01-15T10:00:00Z\",\"updated_at\":\"2026-02-26T14:30:00Z\",\"access_count\":12}],\"total\":1}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "entities": [
    {
      "id": "string (ent_ prefixed)",
      "name": "string",
      "entity_type": "string",
      "summary": "string",
      "activation_score": "number (0.0-1.0)",
      "created_at": "string (ISO 8601)",
      "updated_at": "string (ISO 8601)",
      "access_count": "integer"
    }
  ],
  "total": "integer"
}
```

---

### 3.4 `search_facts`

Query relationships/facts in the graph.

**Definition:**

```json
{
  "name": "search_facts",
  "description": "Search for facts and relationships in the knowledge graph. Returns temporally-aware results showing when facts were valid.",
  "annotations": {
    "readOnlyHint": true,
    "destructiveHint": false,
    "idempotentHint": true,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language query about facts or relationships."
      },
      "subject": {
        "type": "string",
        "description": "Filter facts by subject entity name."
      },
      "predicate": {
        "type": "string",
        "description": "Filter by relationship type (e.g., 'works_at', 'interested_in')."
      },
      "include_expired": {
        "type": "boolean",
        "description": "Include facts that are no longer valid (have a valid_to date).",
        "default": false
      },
      "limit": {
        "type": "integer",
        "description": "Maximum number of facts to return.",
        "default": 10,
        "minimum": 1,
        "maximum": 50
      }
    },
    "required": ["query"]
  }
}
```

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "search_facts",
    "arguments": {
      "query": "where does the user work",
      "include_expired": true
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"facts\":[{\"subject\":\"User\",\"predicate\":\"WORKS_ON\",\"object\":\"ReadyCheck\",\"valid_from\":\"2026-01-15T00:00:00Z\",\"valid_to\":null,\"confidence\":0.95,\"source_episode\":\"ep_a1b2c3d4\",\"created_at\":\"2026-01-15T10:00:00Z\"},{\"subject\":\"User\",\"predicate\":\"WORKED_AT\",\"object\":\"Acme Corp\",\"valid_from\":\"2024-03-01T00:00:00Z\",\"valid_to\":\"2025-12-15T00:00:00Z\",\"confidence\":0.90,\"source_episode\":\"ep_m9n0o1p2\",\"created_at\":\"2025-06-10T08:00:00Z\"}],\"total\":2}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "facts": [
    {
      "subject": "string",
      "predicate": "string",
      "object": "string",
      "valid_from": "string (ISO 8601)",
      "valid_to": "string | null",
      "confidence": "number (0.0-1.0)",
      "source_episode": "string",
      "created_at": "string (ISO 8601)"
    }
  ],
  "total": "integer"
}
```

---

### 3.5 `forget`

Soft-delete a memory or mark a fact as expired.

**Definition:**

```json
{
  "name": "forget",
  "description": "Mark a memory, entity, or fact as forgotten. Performs a soft delete by setting valid_to to now and decaying activation to zero. The data remains in the graph for audit but will not appear in future recall results.",
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": true,
    "idempotentHint": true,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "entity_name": {
        "type": "string",
        "description": "Name of the entity to forget."
      },
      "fact": {
        "type": "object",
        "description": "Specific fact/relationship to forget.",
        "properties": {
          "subject": {
            "type": "string",
            "description": "Subject entity name."
          },
          "predicate": {
            "type": "string",
            "description": "Relationship type."
          },
          "object": {
            "type": "string",
            "description": "Object entity name."
          }
        },
        "required": ["subject", "predicate", "object"]
      },
      "reason": {
        "type": "string",
        "description": "Why this memory should be forgotten. Stored for audit."
      }
    },
    "required": []
  }
}
```

Exactly one of `entity_name` or `fact` must be provided. The server returns an error if both or neither are given.

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tools/call",
  "params": {
    "name": "forget",
    "arguments": {
      "fact": {
        "subject": "User",
        "predicate": "LIVES_IN",
        "object": "Mesa"
      },
      "reason": "User corrected: they moved to Denver."
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"forgotten\",\"target_type\":\"fact\",\"target\":\"User -[LIVES_IN]-> Mesa\",\"valid_to\":\"2026-02-27T12:00:00Z\",\"message\":\"Fact marked as expired. It will no longer appear in recall results.\"}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "status": "forgotten",
  "target_type": "entity | fact",
  "target": "string (human-readable description)",
  "valid_to": "string (ISO 8601, when it was expired)",
  "message": "string"
}
```

---

### 3.6 `get_context`

Pre-assembled context blob of the most activated memories. Designed for "attach to every turn" usage.

**Definition:**

```json
{
  "name": "get_context",
  "description": "Return a pre-assembled context string of the most activated memories for the current conversation. Use this at the start of a conversation or when you need a broad overview of what you know about the user.",
  "annotations": {
    "readOnlyHint": false,
    "destructiveHint": false,
    "idempotentHint": false,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "max_tokens": {
        "type": "integer",
        "description": "Approximate maximum token budget for the returned context.",
        "default": 2000,
        "minimum": 100,
        "maximum": 8000
      },
      "topic_hint": {
        "type": "string",
        "description": "Optional topic hint to bias activation towards relevant memories."
      }
    },
    "required": []
  }
}
```

Note: `readOnlyHint` is `false` because calling `get_context` triggers activation updates on the nodes included in the context.

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "get_context",
    "arguments": {
      "max_tokens": 2000,
      "topic_hint": "coding projects"
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"context\":\"## Active Memory Context\\n\\n**Key entities (by activation):**\\n- ReadyCheck (Project, activation: 0.82): Meeting prep SaaS, 8-week launch target. Currently integrating Stripe for payments.\\n- Stripe (Tool/Technology, activation: 0.75): Payment platform being integrated into ReadyCheck.\\n- Engram (Project, activation: 0.70): Open-source memory layer for AI agents. Uses activation-based retrieval.\\n\\n**Recent facts:**\\n- User is building ReadyCheck (since Jan 2026)\\n- ReadyCheck is integrating Stripe (since Feb 2026)\\n- User moved to Denver (Feb 2026)\\n\\n**Active topics:** SaaS development, payment integration, AI tooling\",\"entity_count\":8,\"fact_count\":12,\"token_estimate\":185}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "context": "string (markdown-formatted context block)",
  "entity_count": "integer (entities included)",
  "fact_count": "integer (facts included)",
  "token_estimate": "integer (approximate token count)"
}
```

---

### 3.7 `get_graph_state`

Graph statistics and top-activated nodes. Primary data source for the dashboard.

**Definition:**

```json
{
  "name": "get_graph_state",
  "description": "Return current graph statistics and top-activated nodes. Used by the dashboard for overview displays and activation monitoring.",
  "annotations": {
    "readOnlyHint": true,
    "destructiveHint": false,
    "idempotentHint": true,
    "openWorldHint": false
  },
  "inputSchema": {
    "type": "object",
    "properties": {
      "top_n": {
        "type": "integer",
        "description": "Number of top-activated nodes to include.",
        "default": 20,
        "minimum": 1,
        "maximum": 100
      },
      "include_edges": {
        "type": "boolean",
        "description": "Include edge/relationship data in the response.",
        "default": false
      },
      "entity_types": {
        "type": "array",
        "description": "Filter to specific entity types.",
        "items": {
          "type": "string",
          "enum": ["Person", "Organization", "Project", "Concept", "Preference", "Location", "Event", "Tool/Technology"]
        }
      }
    },
    "required": []
  }
}
```

**Example request:**

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "get_graph_state",
    "arguments": {
      "top_n": 10,
      "include_edges": true
    }
  }
}
```

**Success response:**

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"stats\":{\"total_entities\":47,\"total_relationships\":128,\"total_episodes\":83,\"entity_type_distribution\":{\"Person\":8,\"Organization\":5,\"Project\":6,\"Concept\":12,\"Preference\":4,\"Location\":3,\"Event\":5,\"Tool/Technology\":4},\"active_entities\":18,\"dormant_entities\":29},\"top_activated\":[{\"id\":\"ent_x1y2z3\",\"name\":\"ReadyCheck\",\"entity_type\":\"Project\",\"activation_score\":0.82,\"summary\":\"Meeting prep SaaS with 8-week launch target\",\"last_accessed\":\"2026-02-27T09:00:00Z\",\"access_count\":12,\"neighbor_count\":8},{\"id\":\"ent_a4b5c6\",\"name\":\"Stripe\",\"entity_type\":\"Tool/Technology\",\"activation_score\":0.75,\"summary\":\"Payment processing platform\",\"last_accessed\":\"2026-02-26T14:30:00Z\",\"access_count\":7,\"neighbor_count\":3}],\"edges\":[{\"source\":\"ent_x1y2z3\",\"target\":\"ent_a4b5c6\",\"predicate\":\"INTEGRATES\",\"weight\":0.85,\"valid_from\":\"2026-02-20T00:00:00Z\",\"valid_to\":null}],\"group_id\":\"personal\"}"
      }
    ]
  }
}
```

**Response JSON shape:**

```json
{
  "stats": {
    "total_entities": "integer",
    "total_relationships": "integer",
    "total_episodes": "integer",
    "entity_type_distribution": {
      "Person": "integer",
      "Organization": "integer",
      "Project": "integer",
      "Concept": "integer",
      "Preference": "integer",
      "Location": "integer",
      "Event": "integer",
      "Tool/Technology": "integer"
    },
    "active_entities": "integer (activation > 0.3)",
    "dormant_entities": "integer (activation <= 0.3)"
  },
  "top_activated": [
    {
      "id": "string",
      "name": "string",
      "entity_type": "string",
      "activation_score": "number (0.0-1.0)",
      "summary": "string",
      "last_accessed": "string (ISO 8601)",
      "access_count": "integer",
      "neighbor_count": "integer"
    }
  ],
  "edges": [
    {
      "source": "string (entity id)",
      "target": "string (entity id)",
      "predicate": "string",
      "weight": "number (0.0-1.0)",
      "valid_from": "string (ISO 8601)",
      "valid_to": "string | null"
    }
  ],
  "group_id": "string"
}
```

`edges` is only populated when `include_edges` is `true`. When `false`, the field is omitted to reduce response size.

---

## 4. MCP Resources

Resources expose read-only graph state for application-controlled context attachment. Clients can subscribe to updates.

### 4.1 Static Resources

```json
{
  "resources": [
    {
      "uri": "engram://graph/stats",
      "name": "Graph Statistics",
      "description": "Current graph statistics: entity counts, relationship counts, type distribution, activation summary.",
      "mimeType": "application/json"
    }
  ]
}
```

**`engram://graph/stats` read response:**

```json
{
  "contents": [
    {
      "uri": "engram://graph/stats",
      "mimeType": "application/json",
      "text": "{\"total_entities\":47,\"total_relationships\":128,\"total_episodes\":83,\"entity_type_distribution\":{\"Person\":8,\"Organization\":5,\"Project\":6,\"Concept\":12,\"Preference\":4,\"Location\":3,\"Event\":5,\"Tool/Technology\":4},\"active_entities\":18,\"dormant_entities\":29,\"last_episode_at\":\"2026-02-27T09:00:00Z\"}"
    }
  ]
}
```

### 4.2 Resource Templates

```json
{
  "resourceTemplates": [
    {
      "uriTemplate": "engram://entity/{entity_id}",
      "name": "Entity Profile",
      "description": "Full profile for a specific entity including activation state, related facts, and connection graph.",
      "mimeType": "application/json"
    },
    {
      "uriTemplate": "engram://entity/{entity_id}/neighbors",
      "name": "Entity Neighbors",
      "description": "Entities directly connected to the specified entity with relationship details.",
      "mimeType": "application/json"
    },
    {
      "uriTemplate": "engram://episodes/{date}",
      "name": "Episodes by Date",
      "description": "All episodes ingested on a given date (YYYY-MM-DD format).",
      "mimeType": "application/json"
    }
  ]
}
```

**`engram://entity/{entity_id}` read response:**

```json
{
  "contents": [
    {
      "uri": "engram://entity/ent_x1y2z3",
      "mimeType": "application/json",
      "text": "{\"id\":\"ent_x1y2z3\",\"name\":\"ReadyCheck\",\"entity_type\":\"Project\",\"summary\":\"Meeting prep SaaS with 8-week launch target\",\"activation\":{\"base\":0.65,\"current\":0.82,\"last_accessed\":\"2026-02-27T09:00:00Z\",\"access_count\":12,\"decay_rate\":0.1},\"facts\":[{\"predicate\":\"INTEGRATES\",\"object\":\"Stripe\",\"valid_from\":\"2026-02-20T00:00:00Z\",\"valid_to\":null},{\"predicate\":\"BUILT_BY\",\"object\":\"User\",\"valid_from\":\"2026-01-15T00:00:00Z\",\"valid_to\":null}],\"created_at\":\"2026-01-15T10:00:00Z\",\"updated_at\":\"2026-02-26T14:30:00Z\"}"
    }
  ]
}
```

### 4.3 Resource Subscriptions

Clients can subscribe to resource updates using `resources/subscribe`. The server emits `notifications/resources/updated` when graph state changes.

```json
{
  "jsonrpc": "2.0",
  "id": 10,
  "method": "resources/subscribe",
  "params": {
    "uri": "engram://graph/stats"
  }
}
```

After a `remember` call completes extraction, the server sends:

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "uri": "engram://graph/stats"
  }
}
```

---

## 5. MCP Prompts

### 5.1 `engram-system` -- System Prompt Template

This is the primary prompt. It instructs Claude on how to use Engram tools naturally in conversation.

**Prompt definition:**

```json
{
  "name": "engram-system",
  "description": "System prompt instructions for using Engram memory tools in conversation. Attach this to give Claude the ability to remember and recall naturally.",
  "arguments": [
    {
      "name": "persona",
      "description": "Optional persona name for the assistant (e.g., 'my coding assistant', 'my research partner').",
      "required": false
    },
    {
      "name": "auto_remember",
      "description": "Whether to automatically remember after each turn. 'always', 'important', or 'never'.",
      "required": false
    }
  ]
}
```

**`prompts/get` response (auto_remember = "important"):**

```json
{
  "description": "System prompt for Engram memory integration",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "You have access to Engram, a persistent memory system. Use it to remember information about the user and recall it in future conversations.\n\n## Memory Tools\n\nYou have these memory tools available:\n\n- **remember**: Store important information from the conversation. Call this when the user shares personal details, preferences, project updates, decisions, or any information they would expect you to know later.\n- **recall**: Retrieve relevant memories. Call this when the user references something from a previous conversation, asks 'do you remember...', or when context from past interactions would improve your response.\n- **search_entities**: Look up specific people, projects, or concepts the user has mentioned before.\n- **search_facts**: Find specific facts or relationships (e.g., 'where does the user work?').\n- **forget**: Remove outdated or incorrect information when the user corrects you or asks you to forget something.\n- **get_context**: Get a broad overview of what you know about the user. Use this at the start of conversations.\n\n## When to Remember\n\nCall `remember` after a turn when the user shares:\n- Personal information (name, location, job, preferences)\n- Project details (what they are working on, technologies, deadlines)\n- Decisions or opinions\n- Relationships (people, organizations they mention)\n- Corrections to things you previously got wrong\n- Goals, plans, or intentions\n\nInclude both the user's message and your response as the content so the full context is preserved.\n\n## When to Recall\n\nCall `recall` when:\n- Starting a new conversation (use get_context for broad overview)\n- The user references something from the past\n- You need context that might exist in memory to give a better answer\n- The user asks 'do you remember...' or 'what do you know about...'\n\n## Guidelines\n\n- Do not tell the user you are storing memories unless they ask. Memory should feel natural, not transactional.\n- When recalling, integrate the information smoothly into your response. Do not say 'According to my memory system...'.\n- If recall returns no results, do not mention it. Just respond normally.\n- If you are uncertain whether something is worth remembering, remember it. It is better to have too much context than too little.\n- Always prioritize the user's most recent statements over older memories if there is a conflict."
      }
    }
  ]
}
```

### 5.2 `engram-context-loader` -- Conversation Start Prompt

**Prompt definition:**

```json
{
  "name": "engram-context-loader",
  "description": "Load memory context at the start of a conversation. Returns a prompt that triggers get_context to pre-load relevant memories.",
  "arguments": [
    {
      "name": "topic",
      "description": "Optional topic hint to bias which memories are loaded.",
      "required": false
    }
  ]
}
```

**`prompts/get` response:**

```json
{
  "description": "Pre-load memory context for a new conversation",
  "messages": [
    {
      "role": "user",
      "content": {
        "type": "text",
        "text": "Before responding, call get_context to load what you know about the user. If a topic was mentioned, pass it as the topic_hint."
      }
    }
  ]
}
```

---

## 6. Automatic Ingestion Strategy

### 6.1 Architecture

Automatic ingestion is driven by the **system prompt** (via the `engram-system` prompt above), not by protocol-level hooks. The LLM decides when to call `remember` based on the instructions.

This approach was chosen over alternatives because:

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| System prompt instruction | Works with any MCP client, no custom protocol needed | LLM may forget to call it | **Selected** |
| Client-side hook (every turn) | Guaranteed capture | Requires client modification, noisy | Rejected |
| Server-side sampling | Server controls ingestion timing | Sampling is not widely supported yet | Future option |

### 6.2 Ingestion Behavior

When the LLM calls `remember`, the server:

1. **Immediately** persists the raw episode to FalkorDB (~10ms, synchronous)
2. Publishes extraction job to **Redis Streams** consumer group
3. Returns `{"status": "queued", "episode_id": "ep_..."}` to the LLM
4. **Asynchronously**, the ingestion pipeline processes through stages:
   - `queued` -- episode persisted, awaiting extraction
   - `extracting` -- Claude API call for entity/relationship extraction (Haiku for simple, Sonnet for complex)
   - `resolving` -- entity dedup, conflict resolution
   - `writing` -- graph mutations (FalkorDB)
   - `activating` -- activation state updates (Redis)
   - `embedding` -- vector embedding generation
   - `completed` -- fully processed
5. On completion, `notifications/resources/updated` is emitted for subscribed MCP clients

The LLM does not wait for extraction. Memory is fire-and-forget from the tool-calling perspective. Failed extractions move to `dead_letter` state and are retried by the pipeline automatically.

### 6.3 Deduplication

If the same content is submitted twice (exact match on content hash), the server returns the existing episode ID without re-queuing extraction:

```json
{
  "status": "duplicate",
  "episode_id": "ep_01HXYZ9ABCDEF0000000000",
  "message": "This content has already been remembered.",
  "status_url": "/api/episodes/ep_01HXYZ9ABCDEF0000000000/status"
}
```

---

## 7. Error Response Shapes

All errors follow MCP's standard JSON-RPC error format with an additional `data` field for structured error details.

### 7.1 Error Code Registry

| Code | Name | When |
|------|------|------|
| -32602 | InvalidParams | Missing required params, invalid types, constraint violations |
| -32603 | InternalError | Server-side failure (DB down, extraction crash) |
| 1001 | ExtractionFailed | Claude API call for entity extraction failed |
| 1002 | EntityNotFound | Referenced entity does not exist in graph |
| 1003 | GraphUnavailable | FalkorDB or Redis is unreachable |
| 1004 | RateLimited | Too many requests (hosted tier) |
| 1005 | GroupNotConfigured | ENGRAM_GROUP_ID is not set and no auth token provided |
| 1006 | InvalidForgetTarget | Must provide exactly one of entity_name or fact |

### 7.2 Error Response Examples

**Invalid parameters (missing required field):**

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "field": "query",
      "reason": "Required field 'query' is missing.",
      "tool": "recall"
    }
  }
}
```

**Extraction failure (Claude API error):**

Extraction failures are **not** surfaced to the MCP client at all. The `remember` tool always returns `"status": "queued"` as long as the episode was persisted. If extraction later fails, the episode moves to `dead_letter` state in the pipeline -- visible via the `status_url` REST endpoint and dashboard, but invisible to the LLM. The ingestion pipeline handles retries with exponential backoff internally.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"queued\",\"episode_id\":\"ep_01HXYZ9ABCDEF9999999999\",\"message\":\"Memory received and queued for processing.\",\"status_url\":\"/api/episodes/ep_01HXYZ9ABCDEF9999999999/status\"}"
      }
    ],
    "isError": false
  }
}
```

Note: The only scenario where `remember` returns an error is if the episode itself cannot be persisted (e.g., FalkorDB is down). Extraction failures happen asynchronously and are handled by the ingestion pipeline's retry/dead-letter mechanism.

**Entity not found:**

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"status\":\"error\",\"code\":1002,\"message\":\"Entity 'FooBar' not found in the knowledge graph.\"}"
      }
    ],
    "isError": true
  }
}
```

**Graph unavailable:**

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": {
      "code": 1003,
      "detail": "FalkorDB is unreachable at localhost:6379. Is the Docker container running?",
      "retryable": true
    }
  }
}
```

**Group not configured:**

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "code": 1005,
      "detail": "ENGRAM_GROUP_ID environment variable is not set. Add it to your MCP server configuration.",
      "docs": "https://github.com/engram-dev/engram#configuration"
    }
  }
}
```

### 7.3 Tool-Level Error Convention

Tools that succeed at the protocol level but encounter application-level issues use the MCP `isError` flag:

- `isError: false` (default) -- tool executed successfully
- `isError: true` -- tool ran but the operation failed logically (e.g., entity not found, validation error)

Protocol-level failures (DB down, auth failure) use JSON-RPC error responses, not `isError`.

---

## 8. Auth in MCP Context

**Key design decision (aligned with security model spec 05):** MCP auth uses the exact same `TenantContextMiddleware` as the REST API. There is no separate auth system for MCP. This keeps the security surface small and consistent.

### 8.1 Local (stdio) -- Filesystem Trust Boundary

For local stdio transport, the user's filesystem access IS the auth boundary. If you can start the MCP server process, you have access. `group_id` is set via `ENGRAM_GROUP_ID` environment variable.

No bearer token is needed. Every MCP tool handler resolves `TenantContext` from the env var on this path.

### 8.2 Hosted (HTTP/SSE) -- Bearer Token

For hosted and multi-user deployments, every HTTP request to `/mcp` includes a Bearer token:

```
Authorization: Bearer <token>
```

The same `TenantContextMiddleware` that protects REST routes also protects the `/mcp` endpoint. It validates the JWT and resolves `TenantContext`:

```json
{
  "sub": "user_abc123",
  "group_id": "personal",
  "scopes": ["read", "write"],
  "exp": 1740700800
}
```

### 8.3 stdio Clients Connecting to Hosted Server

Clients that only support stdio (e.g., some Claude Desktop versions) use `mcp-remote` to bridge to the authenticated HTTP/SSE endpoint:

```
npx mcp-remote http://localhost:8787/mcp --header "Authorization: Bearer ${ENGRAM_BEARER_TOKEN}"
```

The token flows through `mcp-remote` as an HTTP header on every request, hitting the same `TenantContextMiddleware`.

### 8.4 MCP Tool Auth Enforcement

Every MCP tool handler receives `TenantContext` via the same `get_tenant(request)` dependency used by REST routes. No MCP tool can execute without a resolved tenant scope. The tenant's `group_id` is passed to all graph, Redis, and retrieval operations.

### 8.5 Scope Mapping

| Scope | Allowed Tools |
|-------|--------------|
| `read` | recall, search_entities, search_facts, get_context, get_graph_state |
| `write` | remember, forget |
| `admin` | All tools + resource management |

Scope enforcement is handled by `TenantContextMiddleware` before the tool handler runs. A `write`-scoped token attempting `recall` will succeed (read is implicit in write). A `read`-scoped token attempting `remember` will receive a 403 error.

See the security model spec (05_security_model.md) for full JWT structure, token rotation, OAuth flow, and `TenantContextMiddleware` implementation details.

---

## 9. Server Capability Negotiation

On connection initialization, the Engram MCP server advertises its capabilities:

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": "2025-11-25",
    "serverInfo": {
      "name": "engram",
      "version": "0.1.0"
    },
    "capabilities": {
      "tools": {
        "listChanged": true
      },
      "resources": {
        "subscribe": true,
        "listChanged": true
      },
      "prompts": {
        "listChanged": true
      }
    }
  }
}
```

- `listChanged: true` means the server may emit `notifications/tools/list_changed` if tools are dynamically updated (future: plugin system).
- `resources.subscribe: true` enables clients to subscribe to resource updates.

---

## 10. Implementation Skeleton

Reference implementation structure for `server/engram/mcp/server.py`:

```python
from mcp.server.fastmcp import FastMCP
from engram.config import get_settings
from engram.graph.store import GraphStore
from engram.activation.engine import ActivationEngine
from engram.retrieval.scorer import CompositeScorer
from engram.extraction.pipeline import ExtractionPipeline

settings = get_settings()
mcp = FastMCP("engram", version="0.1.0")

graph = GraphStore(settings)
activation = ActivationEngine(settings)
scorer = CompositeScorer(settings)
pipeline = ExtractionPipeline(settings)


@mcp.tool()
async def remember(
    content: str,
    source: str = "claude_desktop",
    metadata: dict[str, str] | None = None,
) -> dict:
    """Store a memory from the current conversation. Extracts entities,
    relationships, and facts automatically. Call this after each meaningful
    exchange to build persistent memory."""
    episode = await graph.create_episode(
        content=content,
        source=source,
        group_id=settings.group_id,
        metadata=metadata,
    )
    await pipeline.queue_extraction(episode.id)
    return {
        "status": "queued",
        "episode_id": episode.id,
        "message": "Memory received and queued for processing.",
        "status_url": f"/api/episodes/{episode.id}/status",
    }


@mcp.tool()
async def recall(
    query: str,
    top_k: int = 5,
    min_score: float = 0.1,
) -> dict:
    """Retrieve relevant memories using activation-based scoring."""
    results = await scorer.query(
        query=query,
        group_id=settings.group_id,
        top_k=top_k,
        min_score=min_score,
    )
    return results.to_dict()


@mcp.tool()
async def search_entities(
    name: str | None = None,
    entity_type: str | None = None,
    limit: int = 10,
) -> dict:
    """Search for specific entities by name or type."""
    if not name and not entity_type:
        raise ValueError("At least one of 'name' or 'entity_type' is required.")
    entities = await graph.search_entities(
        name=name,
        entity_type=entity_type,
        group_id=settings.group_id,
        limit=limit,
    )
    return {"entities": [e.to_dict() for e in entities], "total": len(entities)}


@mcp.tool()
async def search_facts(
    query: str,
    subject: str | None = None,
    predicate: str | None = None,
    include_expired: bool = False,
    limit: int = 10,
) -> dict:
    """Search for facts and relationships in the knowledge graph."""
    facts = await graph.search_facts(
        query=query,
        subject=subject,
        predicate=predicate,
        include_expired=include_expired,
        group_id=settings.group_id,
        limit=limit,
    )
    return {"facts": [f.to_dict() for f in facts], "total": len(facts)}


@mcp.tool()
async def forget(
    entity_name: str | None = None,
    fact: dict | None = None,
    reason: str | None = None,
) -> dict:
    """Mark a memory, entity, or fact as forgotten (soft delete)."""
    if (entity_name is None) == (fact is None):
        raise ValueError("Provide exactly one of 'entity_name' or 'fact'.")
    result = await graph.forget(
        entity_name=entity_name,
        fact=fact,
        reason=reason,
        group_id=settings.group_id,
    )
    return result.to_dict()


@mcp.tool()
async def get_context(
    max_tokens: int = 2000,
    topic_hint: str | None = None,
) -> dict:
    """Return a pre-assembled context string of the most activated memories."""
    context = await scorer.build_context(
        group_id=settings.group_id,
        max_tokens=max_tokens,
        topic_hint=topic_hint,
    )
    return context.to_dict()


@mcp.tool()
async def get_graph_state(
    top_n: int = 20,
    include_edges: bool = False,
    entity_types: list[str] | None = None,
) -> dict:
    """Return current graph statistics and top-activated nodes."""
    state = await graph.get_state(
        group_id=settings.group_id,
        top_n=top_n,
        include_edges=include_edges,
        entity_types=entity_types,
    )
    return state.to_dict()


# --- Resources ---

@mcp.resource("engram://graph/stats")
async def graph_stats() -> str:
    """Current graph statistics."""
    import json
    stats = await graph.get_stats(group_id=settings.group_id)
    return json.dumps(stats)


@mcp.resource("engram://entity/{entity_id}")
async def entity_profile(entity_id: str) -> str:
    """Full profile for a specific entity."""
    import json
    entity = await graph.get_entity(entity_id, group_id=settings.group_id)
    return json.dumps(entity.to_full_dict())


@mcp.resource("engram://entity/{entity_id}/neighbors")
async def entity_neighbors(entity_id: str) -> str:
    """Entities connected to the specified entity."""
    import json
    neighbors = await graph.get_neighbors(entity_id, group_id=settings.group_id)
    return json.dumps([n.to_dict() for n in neighbors])


# --- Prompts ---

@mcp.prompt()
def engram_system(
    persona: str = "assistant",
    auto_remember: str = "important",
) -> str:
    """System prompt instructions for Engram memory integration."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        auto_remember=auto_remember,
    )


@mcp.prompt()
def engram_context_loader(topic: str | None = None) -> str:
    """Pre-load memory context at conversation start."""
    hint = f' Pass topic_hint="{topic}".' if topic else ""
    return f"Before responding, call get_context to load what you know about the user.{hint}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 11. Cross-Agent Coordination Notes

### For Auth Agent (05)
- **ALIGNED** with security model spec (05_security_model.md)
- MCP `/mcp` endpoint uses the same `TenantContextMiddleware` as REST routes -- single security surface
- Local stdio: no auth, `group_id` from env var `ENGRAM_GROUP_ID`
- Hosted HTTP/SSE: Bearer token via `Authorization` header, JWT with `sub`, `group_id`, `scopes`
- stdio -> hosted: `mcp-remote --header "Authorization: Bearer $TOKEN"` bridges to HTTP auth
- Scopes: `read` (recall, search, get_context, get_graph_state), `write` (remember, forget), `admin` (all)
- Every MCP tool handler resolves `TenantContext` via `get_tenant(request)` before execution

### For Frontend Agent (07)
- `get_graph_state` response shape is defined in Section 3.7 above
- Key fields for the dashboard: `stats`, `top_activated[]`, `edges[]` (when `include_edges=true`)
- Entity type distribution is in `stats.entity_type_distribution`
- Resource `engram://graph/stats` provides the same data via resource read
- WebSocket updates triggered by `notifications/resources/updated` after ingestion

### For Ingestion Agent (03)
- **ALIGNED** with ingestion pipeline spec (03_async_ingestion.md)
- `remember` response: returns immediately (~10ms) with `{"status": "queued", "episode_id": "ep_<ULID>", "status_url": "..."}`
- Duplicate detection returns `{"status": "duplicate"}` with existing episode ID
- Episode IDs are ULIDs (ep_ prefixed)
- Extraction failures are invisible to MCP client; handled by pipeline retry/dead-letter mechanism
- `status_url` is REST-only (dashboard), not exposed as MCP tool
- Processing states: queued -> extracting -> resolving -> writing -> activating -> embedding -> completed | dead_letter
