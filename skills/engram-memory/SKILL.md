---
name: engram-memory
description: Persistent long-term memory powered by knowledge graphs, ACT-R activation, and 12-phase consolidation. Remember conversations across sessions, recall relevant context automatically, and build a private brain that improves over time.
version: 1.0.0
metadata:
  openclaw:
    requires:
      env: []
    optionalEnv:
      - ANTHROPIC_API_KEY
    emoji: "\U0001F9E0"
    homepage: https://github.com/Moshik21/engram
---

# Engram Memory

You have access to Engram, a persistent memory system that builds a temporal knowledge graph from conversations. It uses ACT-R cognitive architecture for activation-aware retrieval and runs offline consolidation inspired by biological memory.

Works with both lite and full Engram installs.

## Setup

The Engram server must be running locally.

### Lite install (recommended — no Docker)

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

Select **Lite** when prompted, then answer **yes** to "Install OpenClaw skill?".

```bash
engramctl start
```

### Full install (Docker)

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw
```

### Already have Engram running?

Add the OpenClaw skill with:

```bash
engramctl install-openclaw
```

### Advanced fallback (manual source)

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram/server
uv sync
uv run engram setup
```

### Standalone MCP Server (Source)

For developers who want to run the MCP server from source without the installer:

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram/server
uv sync
export ANTHROPIC_API_KEY=sk-ant-...
uv run engram mcp
```

**Transport modes:**
- `uv run engram mcp` — stdio (default, for Claude Desktop / Claude Code)
- `uv run engram mcp --transport streamable-http --port 8200` — HTTP (for remote clients)

**Claude Desktop configuration** (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "engram", "mcp"]
    }
  }
}
```

The REST API is available at `http://127.0.0.1:8100`. Check status with
`engramctl status`.

If you know the current project path, bootstrap it once at session start so
artifact-backed routing has parity with memory:
```
POST http://localhost:8100/api/knowledge/bootstrap
Content-Type: application/json

{"project_path": "<absolute project path>", "session_id": "<optional session id>"}
```

## When to Observe vs Remember

**Default to observe for most content.** Use remember only for high-signal items.

Use **observe** when:
- General conversation context or topics discussed
- Information that might be useful later but is not critical
- Bulk context from a long conversation
- You are uncertain whether something is worth a full remember

Use **remember** when:
- The user explicitly asks you to remember something
- Personal identity facts (name, location, job title)
- Explicit preferences or corrections to prior knowledge
- Key decisions that will affect future interactions
- Goals, plans, or deadlines with concrete details

## How to Store Memories

To observe (fast, cheap, no extraction):
```
POST http://localhost:8100/api/knowledge/observe
Content-Type: application/json

{"content": "<text to store>", "source": "openclaw"}
```

To remember (full extraction with entities and relationships):
```
POST http://localhost:8100/api/knowledge/remember
Content-Type: application/json

{"content": "<important text>", "source": "openclaw"}
```

To forget (soft delete outdated information):
```
POST http://localhost:8100/api/knowledge/forget
Content-Type: application/json

{"entity_name": "<entity to forget>"}
```

## How to Recall Memories

At the start of every conversation, get broad context:
```
GET http://localhost:8100/api/knowledge/context
```

When the user references something from the past or you need relevant context:
```
GET http://localhost:8100/api/knowledge/recall?q=<query>&limit=5
```

For project-truth questions, route first:
```
POST http://localhost:8100/api/knowledge/route
Content-Type: application/json

{"question": "<user question>", "project_path": "<optional project path>"}
```

Use the returned `answerContract` as response policy, not just source routing.
If the route says `inspect` or `reconcile`, treat `evidencePlan.requiredNextSources`
as mandatory. Carry the same `project_path` into artifact/runtime calls before
answering:
```
GET http://localhost:8100/api/knowledge/artifacts/search?q=<query>&project_path=<optional path>&limit=5
GET http://localhost:8100/api/knowledge/runtime?project_path=<optional path>
```

To search for specific entities:
```
GET http://localhost:8100/api/entities/search?q=<name>
```

To search for specific facts and relationships:
```
GET http://localhost:8100/api/knowledge/facts?q=<query>
```

`search_facts` is user-facing by default. Internal decision/artifact graph edges
stay hidden unless you explicitly opt into debug mode with
`include_epistemic=true`.

## Guidelines

- Call the context endpoint once at the start of each new conversation
- For personal continuity turns like "my son did great today" or "talked to Sarah about it", recall first.
- For install/config/current-truth questions like "how do we install the OpenClaw skill?" or "is full mode rework by default?", call `route`, then satisfy `requiredNextSources` before answering.
- For decision/history questions like "what did we decide about launching Engram publicly?", treat it as reconciliation: use memory plus artifacts/runtime before answering, and do not use `search_facts` as a substitute for artifact inspection.
- If `answerContract.operator` is `compare`, contrast raw defaults, shipped install defaults, repo posture, and runtime state when relevant.
- If `answerContract.operator` is `reconcile` or `unresolved_state_report`, preserve earlier discussion versus current documented or implemented truth.
- If `answerContract.operator` is `recommend` or `plan`, state the evidence first and then give advice or next steps.
- When recalling, integrate information naturally. Do not say "my memory system found..."
- If recall returns no results, do not mention it. Just respond normally.
- If uncertain whether something is worth remembering, observe it
- Always prioritize the user's most recent statements over older memories if there is a conflict
- When the user corrects previously stored information, forget the old info then remember the corrected version

## Memory Features

- **Activation-aware retrieval**: Memories accessed more frequently and recently rank higher
- **Knowledge graph**: Entities and relationships are extracted and connected
- **12-phase consolidation**: Offline cycles merge duplicates, infer missing links, prune noise, and discover patterns
- **Memory maturation**: Entities graduate from episodic (recent) to semantic (durable) over time
- **Prospective memory**: Set intentions that fire when related topics come up
- **Dream associations**: Cross-domain creative connections discovered during consolidation

## Prospective Memory (Intentions)

To set a reminder that fires when a related topic comes up:
```
POST http://localhost:8100/api/knowledge/intentions
Content-Type: application/json

{"query": "<topic to watch for>", "action": "<what to do when triggered>", "entity_names": ["<related entity>"]}
```

To list active intentions:
```
GET http://localhost:8100/api/knowledge/intentions
```

When an intention fires during recall, act on it naturally without announcing it was triggered.

## Consolidation

Engram runs 12 offline consolidation phases that improve memory quality over time:
triage, merge, infer, replay, prune, compact, mature, semanticize, schema, reindex, graph_embed, dream.

To trigger a consolidation cycle manually:
```
POST http://localhost:8100/api/consolidation/trigger
Content-Type: application/json

{"profile": "standard"}
```

To check consolidation status:
```
GET http://localhost:8100/api/consolidation/status
```
