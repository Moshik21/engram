# HelixDB Install

Engram's helix mode uses [HelixDB](https://github.com/helixdb/helix) — a Rust
graph-vector database that combines graph traversals and vector search in a
single engine, replacing both FalkorDB and Redis. All Engram features are
available, including the full 16-phase consolidation pipeline.

The recommended local path is **Helix native**: the HelixDB engine runs
in-process through the PyO3 `helix_native` binding, so you get the full Helix
graph/vector/BM25 backend without Docker or a network hop.

> For zero-infra local use, see [Lite Install](lite.md).
> For the FalkorDB + Redis stack, see [Full Docker Install](full-docker.md).

## Prerequisites

- **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
- An **Anthropic API key** for entity extraction (`ANTHROPIC_API_KEY`)
- Native mode: Rust/Cargo and the local Helix PyO3 source used by `make build-native`
- Docker mode: **Docker** and **Docker Compose** (v2)

## Quick Start with Native PyO3

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram

cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, ENGRAM_MODE=helix,
# and ENGRAM_HELIX__TRANSPORT=native

make build-native
make up-native
```

The public one-click path (`scripts/install.sh helix`) also selects native
Helix, but it depends on the installed package exposing the `helix_native` PyO3
extension. The installer requests native extras and runs a no-smoke
`engram doctor --mode helix` check; if that verification fails, use this source
workflow and `make build-native`.

For MCP instead of REST:

```bash
make mcp-native
```

Verify the configured native lifecycle without starting Docker:

```bash
cd server
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram lifecycle --mode helix
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram doctor --skip-server --no-smoke
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --smoke --mode helix --format json
```

When you want to inspect a specific native data directory, pass it explicitly:

```bash
uv run engram serve --mode helix --helix-data-dir /path/to/native-data
uv run engram mcp --mode helix --helix-data-dir /path/to/native-data
uv run engram lifecycle --mode helix --helix-data-dir /path/to/native-data
uv run engram doctor --mode helix --helix-data-dir /path/to/native-data --skip-server
```

The Makefile shortcuts accept the same directory as `NATIVE_DATA_DIR`:

```bash
make up-native NATIVE_DATA_DIR=/path/to/native-data
make mcp-native NATIVE_DATA_DIR=/path/to/native-data
```

`serve`, `mcp`, and the Makefile shortcuts set native transport and run against that directory.
Doctor reads that directory for the lifecycle snapshot, then runs its
projected/consolidated smoke against disposable native storage.

The native smoke creates a disposable PyO3 Helix brain, captures three
episodes, projects them through triage, persists a consolidation cycle and
calibration snapshot, stores local recall/continuity labels, and returns a
brain-loop report with no coverage gaps.

## Quick Start with Docker Compose

Use Docker when you specifically want HelixDB as a separate local service:

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

To run HelixDB as an external HTTP service outside Docker, install from
[helixdb/helix releases](https://github.com/helixdb/helix/releases), deploy the
schema (`helix deploy --path server/engram/storage/helix/`), start HelixDB on
port 6969, then:

```bash
cd server && ENGRAM_MODE=helix ENGRAM_HELIX__HOST=localhost ENGRAM_HELIX__PORT=6969 uv run engram serve
```

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...           # Claude Haiku for entity extraction

# Helix connection (defaults shown)
ENGRAM_MODE=helix                      # Force helix mode (auto-detect also works)
ENGRAM_DEFAULT_GROUP_ID=default        # Your brain ID (one per person, not per project)
# ENGRAM_AUTH__DEFAULT_GROUP_ID=default  # Optional override; omit to follow ENGRAM_DEFAULT_GROUP_ID
ENGRAM_HELIX__TRANSPORT=native         # Native PyO3 recommended; also http or grpc
ENGRAM_HELIX__DATA_DIR=/absolute/path/to/engram-native-data  # Optional native data dir
ENGRAM_HELIX__HOST=localhost           # HelixDB host ("helixdb" inside Docker)
ENGRAM_HELIX__PORT=6969                # HelixDB port

# Optional
VOYAGE_API_KEY=pa-...                  # Voyage AI embeddings
ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE=standard
ENGRAM_ACTIVATION__INTEGRATION_PROFILE=rework
```

When `ENGRAM_MODE=auto` (the default), Engram checks for `helix_native` first,
then probes HelixDB at the configured host and port. If neither native nor an
external Helix service is available, it falls back to lite.

## Lifecycle Commands

```bash
make build-native      # Build the PyO3 native extension
make up-native         # Start REST with native in-process HelixDB
make mcp-native        # Start MCP with native in-process HelixDB
make up-helix          # Start Docker Helix stack (build + start)
make down-helix        # Stop Helix stack
make restart-helix     # Rebuild and restart
make logs-helix        # Tail all container logs
```

## MCP Server

Run the MCP server on your host with native Helix:

```bash
cd server && ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram mcp
```

Or connect MCP to a Dockerized HelixDB:

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
        "ENGRAM_HELIX__TRANSPORT": "native",
        "ANTHROPIC_API_KEY": "sk-ant-..."
      }
    }
  }
}
```

## Architecture Notes

Helix mode replaces the two-service FalkorDB + Redis stack with one HelixDB
engine. In native mode that engine is loaded in-process through PyO3. In HTTP or
Docker mode it runs as a separate service. HelixQL handles graph traversals,
BM25, and vector similarity search natively, so no separate search index is
needed. Like lite mode, the event bridge is in-process and activation is
in-memory, rebuilt from access history on restart.

## Troubleshooting

**HelixDB container won't start** — Check logs for schema compilation errors:
`docker compose -f docker-compose.helix.yml logs helixdb`. The schema file at
`server/engram/storage/helix/schema.hx` must exist and be valid.

**Mode auto-detects as "lite"** — Engram probes the configured host/port with a
2s timeout after checking `helix_native`. Verify the native extension is built
or the container is healthy, port `6969` is mapped, and env vars are set. Or
force native explicitly: `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native`.

**Native mode says `helix_native` is not importable** — Build the PyO3 extension
from the repo root with `make build-native`, or use `make up-native` /
`make mcp-native` so the extension is built before startup. If you meant to use
an external HelixDB service instead, set `ENGRAM_HELIX__TRANSPORT=http`.

**"Connection refused" from host MCP** — Use `localhost:6969` (host port), not
the Docker-internal address.

**HelixDB Python SDK not found** — Install the extra:
`pip install engram[helix]` or `cd server && uv sync --extra helix`.

**Reset all data**:

```bash
docker compose -f docker-compose.helix.yml down -v   # WARNING: deletes everything
```
