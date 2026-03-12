# Lite Install

Engram's lite mode runs entirely on SQLite — no Docker, no Redis, no FalkorDB.
All features are available, including the full 12-phase consolidation pipeline,
graph embeddings, and schema formation.

> For scale/perf with Docker, see [Full Docker Install](full-docker.md).

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

## File locations

| Path | Description |
|------|-------------|
| `~/.engram/.env` | Configuration |
| `~/.engram/engram.db` | SQLite database |
| `~/.engram/engram.pid` | Server PID file |
| `~/.engram/logs/engram.log` | Server log |

## Feature comparison: Lite vs Full

Both modes get the complete Engram feature set:

- Full 12-phase consolidation (triage, merge, infer, replay, prune, compact, mature, semanticize, schema, reindex, graph_embed, dream)
- Entity extraction + resolution (Claude Haiku)
- Multi-signal merge/infer scorers (zero LLM cost)
- Graph embeddings (Node2Vec / TransE / GNN)
- Schema formation (recurring structural motifs)
- Dream spreading + dream associations
- Memory maturation (episodic -> transitional -> semantic)
- Prospective memory (intentions)
- Activation-aware retrieval with spreading activation
- Hybrid search (FTS5 + vectors, RRF fusion)
- MCP server (15 tools) + REST API
- WebSocket API

**Full mode adds** (scale/perf/infrastructure):

- FalkorDB graph database (Cypher queries, optimized traversals)
- Redis HNSW vector indices (faster ANN at >100K entities)
- Persistent distributed activation (Redis, 7-day TTL)
- Cross-process event bridge (Redis pub/sub)
- Pre-built Docker dashboard (React frontend)
- Multi-process rate limiting
- Operational headroom for large graphs

## Upgrading to full mode

When you outgrow lite mode:

```bash
engramctl upgrade
```

This downloads the Docker bundle, starts the full stack, and switches lifecycle
commands to Docker. Your SQLite data at `~/.engram/engram.db` is preserved on
disk for reference but is **not** auto-imported into FalkorDB/Redis — full mode
starts with a fresh graph store.
