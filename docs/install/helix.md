# HelixDB Install

Engram's helix mode uses [HelixDB](https://github.com/helixdb/helix) — a Rust
graph-vector database that combines graph traversals and vector search in a
single engine, replacing both FalkorDB and Redis. All Engram features are
available, including the full 17-phase consolidation pipeline.

The recommended local path is **Helix native**: the HelixDB engine runs
in-process through the PyO3 `helix_native` binding, so you get the full Helix
graph/vector/BM25 backend without Docker or a network hop.

> For zero-infra local use, see [Lite Install](lite.md).
> For the FalkorDB + Redis stack, see [Full Docker Install](full-docker.md).

## Prerequisites

- **Python 3.10+** with [uv](https://docs.astral.sh/uv/)
- Optional **Anthropic API key** for richer entity extraction (`ANTHROPIC_API_KEY`)
- Rust/Cargo only if no compatible `helix-native` release wheel exists and the installer must build from Engram's bundled source
- Docker and Docker Compose only for explicit Docker mode

## Quick Start with Native PyO3

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- helix
engramctl status
engramctl doctor
engramctl connect claude-code
engramctl bootstrap /path/to/project
engramctl bootstrap /path/to/project --include 'notes/**/*.md' --include 'exports/**/*.json'
```

The one-click path selects native Helix, installs Engram, adds the
`helix-native` PyO3 runtime to Engram's uv tool environment, and checks that the
runtime is importable before accepting the configuration. Bootstrap indexes the
selected project metadata plus generic docs, notes, and memory-export folders;
use `--include` for any additional user-approved folders or export globs.

Release wheels are preferred. If no compatible wheel is available for the
current platform, the installer builds `helix-native` from Engram's bundled
custom Helix source and reports Rust/Cargo as the only extra prerequisite. It
does not silently switch to Docker.

Verify the configured native lifecycle without starting Docker:

```bash
cd server
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram lifecycle --mode helix
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram doctor --skip-server --no-smoke
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --smoke --mode helix --format json
ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native uv run engram evaluate --mode helix --require-evaluation-signals --format json
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
projected/consolidated smoke against disposable native storage. The doctor smoke
section includes MCP endpoint readiness and evaluation-signal readiness, so JSON
and Markdown output show whether the runtime transport is reachable and the six
hard-gate signals are measured.

The native smoke creates a disposable PyO3 Helix brain, captures three
episodes, projects them through triage, persists a consolidation cycle and
calibration snapshot, stores local recall/continuity labels, and returns a
brain-loop report with no coverage gaps.
Use `engram evaluate --mode helix --require-evaluation-signals` as the hard
operator gate for a live or reusable native data directory. It exits non-zero
unless cue usefulness, projection yield, recall quality, false recall, triage
calibration, and consolidation effect are all measured with evidence and a
metric.

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

Installed-user lifecycle:

```bash
engramctl quickstart --mode helix
engramctl start
engramctl status
engramctl doctor
engramctl connect claude-code
engramctl bootstrap /path/to/project
engramctl bootstrap /path/to/project --include 'notes/**/*.md' --include 'exports/**/*.json'
engramctl logs
engramctl stop
engramctl update
```

Developer/source lifecycle:

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

For installed users, connect an MCP client to the local HTTP runtime:

```bash
engramctl connect claude-code
engramctl connect cursor
engramctl connect windsurf
engramctl connect claude-desktop
```

For source installs, run the MCP server on your host with native Helix:

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
2s timeout after checking `helix_native`. Run `engramctl update` to refresh the
native runtime, verify the container is healthy if you selected HTTP transport,
and force native explicitly with `ENGRAM_MODE=helix ENGRAM_HELIX__TRANSPORT=native`.

**Native mode says `helix_native` is not importable** — Rerun the public
installer or run `engramctl update`. If no compatible release wheel exists for
your platform, install Rust/Cargo so Engram can build `helix-native` from
source. Developer source checkouts can still use `make build-native`,
`make up-native`, or `make mcp-native`. If you meant to use an external HelixDB
service instead, set `ENGRAM_HELIX__TRANSPORT=http`.

**"Connection refused" from host MCP** — Use `localhost:6969` (host port), not
the Docker-internal address.

**HelixDB Python SDK not found** — Install the extra:
`pip install engram[helix]` or `cd server && uv sync --extra helix`.

**Reset all data**:

```bash
docker compose -f docker-compose.helix.yml down -v   # WARNING: deletes everything
```
