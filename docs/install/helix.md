# HelixDB Install

Engram's helix mode uses [HelixDB](https://github.com/helixdb/helix) — a Rust
graph-vector database that combines graph traversals and vector search in a
single service, replacing both FalkorDB and Redis. All Engram features are
available, including the full 15-phase consolidation pipeline.

> For zero-infra local use, see [Lite Install](lite.md).
> For the FalkorDB + Redis stack, see [Full Docker Install](full-docker.md).

## Prerequisites

- **Docker** and **Docker Compose** (v2)
- **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
- An **Anthropic API key** for entity extraction (`ANTHROPIC_API_KEY`)

## Quick Start with Docker Compose

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram

# Copy and edit config
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY (required), optionally VOYAGE_API_KEY

# Start the Helix stack (HelixDB + server + dashboard)
make up-helix
```

Or use Docker Compose directly:

```bash
docker compose -f docker-compose.helix.yml up -d --build
```

This starts three containers:

| Container | Port | Description |
|-----------|------|-------------|
| `engram-helixdb` | 6969 | HelixDB graph-vector database |
| `engram-server` | 8100 | Engram REST API |
| `engram-dashboard` | 3000 | React dashboard |

Verify:

```bash
make status           # or: docker compose -f docker-compose.helix.yml ps
curl http://localhost:8100/health
```

## Manual / Native HelixDB Setup

To run HelixDB outside Docker, install from [helixdb/helix releases](https://github.com/helixdb/helix/releases), deploy the schema (`helix deploy --path server/engram/storage/helix/`), start HelixDB on port 6969, then:

```bash
cd server && ENGRAM_MODE=helix ENGRAM_HELIX__HOST=localhost ENGRAM_HELIX__PORT=6969 uv run engram serve
```

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...           # Claude Haiku for entity extraction

# Helix connection (defaults shown)
ENGRAM_MODE=helix                      # Force helix mode (auto-detect also works)
ENGRAM_HELIX__HOST=localhost           # HelixDB host ("helixdb" inside Docker)
ENGRAM_HELIX__PORT=6969                # HelixDB port

# Optional
VOYAGE_API_KEY=pa-...                  # Voyage AI embeddings
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard
ENGRAM_ACTIVATION__INTEGRATION_PROFILE=rework
```

When `ENGRAM_MODE=auto` (the default), Engram probes HelixDB at the configured
host and port. If reachable, it selects helix mode; otherwise falls back to lite.

## Lifecycle Commands

```bash
make up-helix          # Start Helix stack (build + start)
make down-helix        # Stop Helix stack
make restart-helix     # Rebuild and restart
make logs-helix        # Tail all container logs
```

## MCP Server

Run the MCP server on your host, connecting to the Dockerized HelixDB:

```bash
# stdio transport (for Claude Desktop / Claude Code)
cd server && ENGRAM_MODE=helix ENGRAM_HELIX__HOST=localhost ENGRAM_HELIX__PORT=6969 uv run engram mcp

# streamable HTTP transport (port 8200)
make mcp-helix
```

Claude Desktop config (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "engram": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/engram/server", "engram", "mcp"],
      "env": {
        "ENGRAM_MODE": "helix",
        "ENGRAM_HELIX__HOST": "localhost",
        "ENGRAM_HELIX__PORT": "6969",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

## Architecture Notes

Helix mode replaces the two-service FalkorDB + Redis stack with a single HelixDB
instance. HelixQL (a compiled query language) handles both graph traversals and
vector similarity search natively -- no separate search index needed. The schema
file (`schema.hx`) is mounted into the container and compiled at startup. Like
lite mode, the event bridge is in-process (not Redis pub/sub) and activation is
in-memory, rebuilt from access history on restart.

## Troubleshooting

**HelixDB container won't start** — Check logs for schema compilation errors:
`docker compose -f docker-compose.helix.yml logs helixdb`. The schema file at
`server/engram/storage/helix/schema.hx` must exist and be valid.

**Mode auto-detects as "lite"** — Engram probes the configured host/port with a
2s timeout. Verify the container is healthy, port `6969` is mapped, and env vars
are set. Or force it: `ENGRAM_MODE=helix`.

**"Connection refused" from host MCP** — Use `localhost:6969` (host port), not
the Docker-internal address.

**HelixDB Python SDK not found** — Install the extra:
`pip install engram[helix]` or `cd server && uv sync --extra helix`.

**Reset all data**:

```bash
docker compose -f docker-compose.helix.yml down -v   # WARNING: deletes everything
```
