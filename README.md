<p align="center">
  <img src="docs/assets/readme/hero-banner.svg" alt="Engram: long-term memory for AI agents" width="100%">
</p>

<p align="center">
  <a href="https://engram-roan.vercel.app"><strong>Website</strong></a> &nbsp;·&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp;
  <a href="#harness-flow">How It Works</a> &nbsp;·&nbsp;
  <a href="#connect-your-agent">Connect</a> &nbsp;·&nbsp;
  <a href="docs/REFERENCE.md">Full Reference</a> &nbsp;·&nbsp;
  <a href="#dashboard">Dashboard</a>
</p>

<p align="center">
  <a href="https://engram-roan.vercel.app"><img src="https://img.shields.io/badge/website-engram-8b5cf6?style=flat-square" alt="Website"></a>
  <img src="https://github.com/Moshik21/engram/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/MCP-27_tools-6366f1?style=flat-square" alt="MCP Tools">
  <img src="https://img.shields.io/badge/license-Apache_2.0-22c55e?style=flat-square" alt="License">
</p>

<br>

**Engram is long-term memory for AI agents.** It stores conversations as episodes, builds a temporal knowledge graph of people, facts, and relationships, ranks recall with ACT-R activation, and runs consolidation in the background to merge duplicates and prune stale entries.

<table>
<tr>
<td width="33%" valign="top">

### Harness captures
Session-start injection, auto-observe hooks, offline queue replay. Routine turns stay in the harness; no agent tokens spent on capture.

</td>
<td width="33%" valign="top">

### Agent recalls
`get_context` once per session, `recall` when cues match, `remember` for durable facts the harness should not infer on its own.

</td>
<td width="33%" valign="top">

### Brain consolidates
Triage, projection, 16-phase consolidation, dream associations. Graph maintenance runs while you work.

</td>
</tr>
</table>

<br>

## Brain Loop

<p align="center">
  <img src="docs/assets/readme/brain-loop.svg" alt="Capture to Cue to Project to Recall to Consolidate" width="920">
</p>

| Stage | What happens |
|-------|----------------|
| **Capture** | `observe`, harness `auto:*`, or `remember`: fast store, optional cue trace |
| **Cue** | Deterministic latent memory; recall before full LLM extraction |
| **Project** | Entity resolution, embedding, graph write (background or immediate) |
| **Recall** | Activation-aware packets instead of raw chat history |
| **Consolidate** | Merge, infer, mature, prune, dream (offline) |

<br>

## Harness flow

<p align="center">
  <img src="docs/assets/readme/harness-flow.svg" alt="Harness, Agent, and Brain layers" width="920">
</p>

The harness handles capture and session injection. The agent spends tokens on recall and explicit `remember` calls. Consolidation runs in the background.

| Tier | Trigger | Action | Budget |
|------|---------|--------|--------|
| **Hot** | Session start, project open | AXI home, bootstrap, session prime | &lt; 3s |
| **Warm** | Read tools, proper nouns | auto_recall lite, packet cache | &lt; 350ms |
| **Cold** | Identity query, deep prior work | `recall`, `search_artifacts` | &lt; 2s |
| **Background** | Session end, schedule | triage, merge, dream | async |

<p align="center">
  <img src="docs/assets/readme/memory-tiers.svg" alt="Episodic to Transitional to Semantic memory tiers" width="760">
</p>

Entities move from episodic to transitional to semantic as they mature. Semantic memories and identity-core entities are pruned on longer horizons than fresh episodic entries.

<br>

## Quickstart

### One command

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

Installs native Helix (recommended), runs `engramctl quickstart`, and verifies the runtime.

```bash
engramctl start
engramctl status
engramctl doctor
```

### Connect your agent

```bash
# MCP + priming (Cursor, Windsurf, Grok Build)
engramctl connect cursor --project "$PWD"

# Claude Code / Codex: read-only session injection
engramctl connect claude-code --project "$PWD" --axi

# Passive transcript capture (Claude AutoCapture hooks)
engramctl connect claude-code --project "$PWD" --capture-transcript

# Bootstrap project docs into memory (idempotent)
engramctl bootstrap "$PWD"
```

<details>
<summary><strong>Developer / source install</strong></summary>

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram/server && uv sync
uv run engram setup --mode helix
make mcp-native    # MCP with native HelixDB (no Docker)
```

Modes: **Helix native** (best quality, no Docker) · **Lite** (SQLite, zero infra) · **Helix HTTP** (Docker) · **Full** (FalkorDB + Redis, legacy)

See [docs/install/](docs/install/) and [docs/REFERENCE.md](docs/REFERENCE.md#storage-modes) for full install paths.

</details>

<br>

## Try Engram (showcase)

Replay a bundled lite demo brain with no API keys:

```bash
cd server && uv run engram showcase run
```

<p align="center">
  <img src="docs/assets/showcase/showcase-demo.gif" alt="engram showcase run: 3/3 Liam continuity beats passed" width="920">
</p>

Three scripted beats: Liam continuity ("He had a great game today"), a correction (soccer to baseball), and a cross-session `get_context` briefing. Runs copy the bundled `demo.db` to `~/.engram/showcase/demo-run.db` so the packaged brain stays pristine.

```bash
# Export static beats for the website theater
cd server && uv run engram showcase export --out ../website/public/showcase-export.json

# Reseed the bundled demo.db from source
cd server && uv run engram showcase seed
cd server && uv run python scripts/simulate.py --showcase-seed engram/data/demo.db
```

Regenerate the recording anytime:

```bash
cd server && uv run python ../scripts/record_showcase.py
```

Outputs `docs/assets/showcase/showcase-demo.mp4` and `.gif` (also copied to `website/public/` for the site).

Step through the same beats in the [website scenario theater](https://engram-roan.vercel.app/#try-it). It reads `website/public/showcase-export.json`.

<br>

## Connect Your Agent

| Client | MCP | Priming rules | AXI inject | Transcript capture |
|--------|-----|---------------|------------|-------------------|
| **Claude Code** | yes | - | default | `--capture-transcript` |
| **Codex** | - | - | default | future |
| **Cursor** | yes | yes | - | - |
| **Windsurf** | yes | yes | - | - |
| **Grok Build** | yes | yes | - | - |
| **OpenClaw** | yes | skill | - | - |

**Agent protocol (harness-first):**

1. `claim_authority`, then follow `required_tools_before_answer`
2. `get_context` once per session (or on project switch / adoption debt)
3. `recall` when people, projects, or prior work appear in the question
4. `remember` for preferences, corrections, identity, durable decisions
5. Skip per-turn `observe` when harness capture is active

Validate adoption:

```bash
engram adoption --authority claim.json --calls trace.jsonl --expect-harness-capture
```

Design doc: [docs/harness-memory-adoption-plan.md](docs/harness-memory-adoption-plan.md)

<br>

## Key Capabilities

| Capability | Summary |
|------------|---------|
| **Temporal graph** | People, orgs, concepts, relationships with valid-from / valid-to |
| **ACT-R activation** | Recency and frequency scoring; lazy recompute from access history |
| **Spreading activation** | Recalling one entity boosts related nodes via graph proximity |
| **16-phase consolidation** | Triage, merge, infer, dream, prune (offline cycle) |
| **Memory packets** | Compact recall output with provenance for the current turn |
| **Prospective memory** | `intend`: graph-embedded intentions triggered by related topics |
| **Atlas** | Multi-scale graph exploration from region clusters to neighborhoods |
| **3D dashboard** | Graph visualization, consolidation monitor, evaluation gates |
| **Topological immunity** | Prunes low-connectivity noise and weak edges |
| **Auto-neuroplasticity** | Tunes activation parameters from retrieval feedback |

<br>

## MCP tools

**27 tools**, **3 resources**, **2 prompts**. Stdio or streamable HTTP.

| Tool | Purpose |
|------|---------|
| `observe` / `remember` | Capture (fast queue vs immediate extraction) |
| `recall` / `get_context` | Activation-aware retrieval and briefing |
| `bootstrap_project` | Ingest project docs as artifacts |
| `claim_authority` | Memory ownership and tool protocol for the turn |
| `route_question` | Route to memory, artifacts, or runtime state |
| `intend` / `list_intentions` | Prospective memory |
| `trigger_consolidation` | Run a consolidation cycle |

MCP instructions tell compatible clients to call memory tools on session start. Harness auto-capture handles routine turns; agents focus on recall.

Full tool list: [docs/REFERENCE.md#mcp-integration](docs/REFERENCE.md#mcp-integration)

<br>

## Storage Modes

| Mode | Backend | Docker | Best for |
|------|---------|--------|----------|
| **Helix native** | PyO3 in-process | None | Recommended (~97ms search, nDCG@10 0.448) |
| **Lite** | SQLite + FTS5 | None | Zero setup, demos |
| **Helix HTTP** | HelixDB container | 1 | Production multi-service |
| **Full** | FalkorDB + Redis | 2 | Legacy throughput |

HelixDB combines graph, HNSW vector search, and field-level BM25 in one engine (171 compiled HelixQL queries).

Details: [docs/install/helix.md](docs/install/helix.md) · [Benchmarks](docs/REFERENCE.md#benchmarks)

<br>

## Dashboard

3D graph view plus Atlas, Timeline, Feed, Activation, Consolidation, Evaluate, and Knowledge chat.

```bash
cd dashboard && pnpm install && pnpm dev   # http://localhost:5173
```

With the full stack, the dashboard updates over WebSocket when episodes are stored.

<br>

## Documentation

| Doc | Contents |
|-----|----------|
| [**REFERENCE.md**](docs/REFERENCE.md) | Installation, architecture, MCP, API, benchmarks, config, dev, troubleshooting |
| [**harness-memory-adoption-plan.md**](docs/harness-memory-adoption-plan.md) | Harness-first adoption design |
| [**axi-interface-plan.md**](docs/axi-interface-plan.md) | AXI session-start injection |
| [**install/**](docs/install/) | Lite, Helix, Docker, OpenClaw guides |
| [**vision/**](docs/vision/) | Product narrative and science |

<br>

## Project Structure

```
server/engram/
  activation/       ACT-R engine, BFS, PPR
  consolidation/    16-phase engine + scheduler
  ingestion/        CQRS: store_episode / project_episode
  mcp/              27 tools, resources, prompts
  retrieval/        Hybrid search, packets, auto_recall
  storage/          Helix native, SQLite, FalkorDB

dashboard/src/      React 19 + Three.js brain + Zustand
```

<br>

## Development

```bash
cd server && uv run pytest -m "not requires_helix" -v   # Backend tests
cd server && uv run ruff check .                        # Lint
cd server && uv run engram mcp                          # MCP stdio
make up-native                                          # Full stack, native Helix
```

Full commands: [docs/REFERENCE.md#development](docs/REFERENCE.md#development)

<br>

## License

Engram is **Apache 2.0**. See [LICENSE](LICENSE).

**HelixDB note:** HTTP/gRPC transport (default Docker) has no license impact on Engram. **Native PyO3** links AGPL-3.0 in-process; review terms for proprietary network services. Lite and Full backends have no AGPL concerns.