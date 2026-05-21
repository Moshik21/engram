# OpenClaw + Engram

Engram gives OpenClaw agents a local long-term brain. The release path is native
Helix through PyO3: no Docker, no cloud memory service, and no API key required
for basic deterministic capture and recall.

## Install

Install the public OpenClaw skill:

```bash
openclaw skills install engram-brain
```

Then install and start the native Engram runtime:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw
```

The `openclaw` installer mode:

- installs Engram and adds the Helix native runtime to the same uv tool environment
- runs `engramctl quickstart --mode helix`
- installs the shared OpenClaw skill at `~/.openclaw/skills/engram-brain`
- writes OpenClaw MCP config for `http://127.0.0.1:8100/mcp`
- runs readiness checks through `engramctl doctor`

Release wheels are preferred for `helix-native`. If no compatible wheel exists
for the current platform, the installer builds `helix-native` from Engram's
bundled custom Helix source and reports Rust/Cargo as the only extra
prerequisite. It does not silently switch OpenClaw users to Docker.

## Existing Engram Install

If Engram is already installed, wire it into OpenClaw directly:

```bash
engramctl quickstart --mode helix --install-openclaw --connect openclaw
```

If the server is already running:

```bash
engramctl install-openclaw
engramctl connect openclaw
engramctl doctor
```

## MCP Contract

OpenClaw should use Engram through MCP when available. The expected OpenClaw
MCP entry is:

```json
{
  "url": "http://127.0.0.1:8100/mcp",
  "transport": "streamable-http"
}
```

`engramctl connect openclaw` writes this with:

```bash
openclaw mcp set engram '{"url":"http://127.0.0.1:8100/mcp","transport":"streamable-http"}'
```

`engramctl` prefers a global `openclaw` command. If it is not installed but
`npx` is available, it uses `npx -y openclaw` against the same `~/.openclaw`
registry. To force a specific command, set `ENGRAM_OPENCLAW_COMMAND`, for
example:

```bash
ENGRAM_OPENCLAW_COMMAND="npx -y openclaw" engramctl connect openclaw --verify
```

OpenClaw's `mcp set` / `mcp show` commands manage OpenClaw's saved MCP
registry. They do not open a live session to Engram or prove the target server
is reachable; use `engramctl doctor` and the dogfood validator for Engram-side
readiness and tool-catalog checks.

If neither `openclaw` nor `npx` is available, `engramctl` reports the blocker
and prints the manual command.

## AXI Shell Fallback

OpenClaw should remain MCP-first when its MCP client is available. For agents
with shell access, Engram also exposes a compact AXI surface:

```bash
engram axi --project "$PWD"
engram axi context --project "$PWD" --budget 800 --timeout 5
engram axi recall "current task" --limit 5 --timeout 5
```

This is a lightweight context and fallback layer, not a replacement for the
OpenClaw MCP configuration above. Hook installation for OpenClaw is still
planned separately after the current OpenClaw hook mechanism is confirmed.
Codex and Claude Code can enable the same read-only startup packet with
`engramctl connect codex --axi` or `engramctl connect claude-code --axi`; do
not translate those hook semantics to OpenClaw until OpenClaw exposes a stable
hook mechanism.

## Verify

```bash
engramctl status
engramctl doctor
openclaw skills list --eligible
openclaw mcp show engram --json
```

If you use the npx path:

```bash
npx -y openclaw mcp show engram --json
```

Then start a new OpenClaw session so its skill snapshot includes Engram and the
runtime adapter can consume the saved MCP registry entry.

## Bootstrap A Project

Bootstrap useful project memory once per workspace:

```bash
engramctl bootstrap /path/to/project --include 'docs/**/*.md' --include 'memory/**/*.md' --include 'exports/**/*.json'
```

The defaults already cover common docs, notes, memory, and export folders.
Use explicit `--include` patterns only when you want to narrow or extend that
set.

## Lite And Docker Fallbacks

Lite SQLite mode is available for disposable local testing:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- lite
```

Docker full mode is explicit:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- full
```

Do not use Docker as the default OpenClaw path.

## ClawHub Release

The intended public skill slug is:

```bash
openclaw skills install engram-brain
```

Publish the skill with the ClawHub CLI after release artifacts are generated:

```bash
mkdir -p dist/clawhub
tar -xzf dist/release/engram-brain-skill.tar.gz -C dist/clawhub
clawhub sync --dry-run --workdir dist/clawhub --root dist/clawhub --no-input

clawhub publish dist/clawhub/engram-brain \
  --slug engram-brain \
  --name "Engram Brain" \
  --version 0.3.4 \
  --changelog "Align Project Synapse with the 17-phase consolidation contract, including immunity, dashboard phase constants, and OpenClaw skill docs." \
  --tags latest,memory,knowledge-graph,mcp,recall,long-term-memory,cognitive-architecture
```
