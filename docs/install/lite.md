# Lite Install

Engram's lite mode runs entirely on SQLite — no Docker, no Redis, no FalkorDB.
All features are available, including the full 16-phase consolidation pipeline,
graph embeddings, and schema formation.

> For full HelixDB graph/vector/BM25 without Docker, use
> [Helix native install](helix.md). For the legacy FalkorDB + Redis stack, see
> [Full Docker Install](full-docker.md).

## One-click install

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

Select **[1] Lite** when prompted. Or skip the prompt:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- lite
```

What it does:

- checks Python 3.10+ (guides to install if missing)
- installs `uv` if not present
- installs `engram[local]` via `uv tool install`
- installs `engramctl` to `~/.local/bin/`
- runs `engramctl setup` (API keys, profiles, etc.)
- writes config to `~/.engram/.env`

## Lifecycle commands

```bash
engramctl start        # start server in background
engramctl status       # show running state, port, DB path
engramctl logs         # tail server log
engramctl stop         # stop background server
engramctl update       # upgrade to latest version
engramctl uninstall    # remove (preserves data by default)
engramctl uninstall --purge-data  # remove everything
```

## Endpoints

- **API**: `http://127.0.0.1:8100`
- **Health**: `http://127.0.0.1:8100/health`

## Local diagnostics

Run the local doctor before trusting a lite install:

```bash
engram doctor
```

The doctor loads configuration, resolves the engine mode, checks the local API
health endpoint when the server is running, includes the current local
`Capture -> Cue -> Project -> Recall -> Consolidate` lifecycle snapshot, and
runs the disposable projected/consolidated smoke. The smoke section reports both
coverage gaps and evaluation-signal readiness for the six hard-gate signals. For
a JSON gate:

```bash
engram doctor --mode lite --skip-server --format json
```

For a fast config plus lifecycle check without the heavier smoke:

```bash
engram doctor --mode lite --skip-server --no-smoke
```

To inspect the current brain-loop state without starting the dashboard, print
the same lifecycle summary used by REST, MCP, and the Brain Loop view:

```bash
engram lifecycle
engram lifecycle --format json
```

To inspect evaluation readiness for the current lite brain, print the same
Capture -> Cue -> Project -> Recall -> Consolidate report used by REST, MCP,
and native Helix:

```bash
engram evaluate --mode lite
engram evaluate --mode lite --format json
engram evaluate --mode lite --require-evaluation-signals --format json
```

Use `--require-evaluation-signals` only when the lite brain has enough cue
feedback, projection yield, recall labels, triage calibration, and consolidation
history to be treated as a gate. Fresh disposable demo brains usually need the
doctor/smoke path first.

For clean local dashboard or demo smokes, disable hook-driven auto-capture so
external `/api/knowledge/auto-observe` traffic cannot write unrelated episodes
into the demo brain:

```bash
ENGRAM_MODE=lite ENGRAM_SERVER__AUTO_OBSERVE_ENABLED=false engram serve
```

## File locations

| Path | Description |
|------|-------------|
| `~/.engram/.env` | Configuration |
| `~/.engram/engram.db` | SQLite database |
| `~/.engram/engram.pid` | Server PID file |
| `~/.engram/logs/engram.log` | Server log |

## Feature comparison: Lite vs Full

Both modes get the complete Engram feature set:

- Full 16-phase consolidation (triage, merge, calibrate, infer, evidence_adjudication, edge_adjudication, replay, prune, compact, mature, semanticize, schema, reindex, graph_embed, microglia, dream)
- Entity extraction + resolution (Claude Haiku)
- Multi-signal merge/infer scorers (zero LLM cost)
- Graph embeddings (Node2Vec / TransE / GNN)
- Schema formation (recurring structural motifs)
- Dream spreading + dream associations
- Memory maturation (episodic -> transitional -> semantic)
- Prospective memory (intentions)
- Activation-aware retrieval with spreading activation
- Hybrid search (FTS5 + vectors, RRF fusion)
- MCP server (26 tools) + REST API
- WebSocket API

**Full mode adds** (scale/perf/infrastructure):

- FalkorDB graph database (Cypher queries, optimized traversals)
- Redis HNSW vector indices (faster ANN at >100K entities)
- Persistent distributed activation (Redis, 7-day TTL)
- Cross-process event bridge (Redis pub/sub)
- Pre-built Docker dashboard (React frontend)
- Multi-process rate limiting
- Operational headroom for large graphs

## Moving beyond lite

When you outgrow lite mode, use Helix native first if you want the full
graph/vector/BM25 backend without Docker:

```bash
cd server
make build-native
make up-native NATIVE_DATA_DIR=/path/to/native-data
```

The legacy Docker full stack is still available:

```bash
engramctl upgrade
```

This downloads the Docker bundle, starts the full stack, and switches lifecycle
commands to Docker. Your SQLite data at `~/.engram/engram.db` is preserved on
disk for reference but is **not** auto-imported into FalkorDB/Redis — full mode
starts with a fresh graph store.
