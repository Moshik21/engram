# OpenClaw + Engram

> **Note:** The OpenClaw integration has not been fully tested yet. These instructions are provided as-is and may require adjustments. Please report issues if you encounter problems.

The OpenClaw skill connects to the local Engram REST API on `127.0.0.1:8100`.
It works with both lite and full Engram installs.

## One-click install

### Lite (recommended — no Docker)

```bash
curl -sSL https://engram.run/install | bash
```

Select **[1] Lite** when prompted, then answer **yes** to "Install OpenClaw skill?".

### Full (Docker)

```bash
curl -sSL https://engram.run/install | bash -s -- full
```

Then install the skill:

```bash
engramctl install-openclaw
```

Or use the combined openclaw mode:

```bash
curl -sSL https://engram.run/install | bash -s -- openclaw
```

## Runtime contract

- OpenClaw talks to the local Engram REST API on `127.0.0.1:8100`
- Works with both lite (SQLite) and full (FalkorDB + Redis) backends
- The same `engramctl` lifecycle commands manage the stack

## Manual follow-up

If you already installed Engram (either mode), add the skill later with:

```bash
engramctl install-openclaw
```

## Advanced fallback

Manual source-based setup remains available for development:

```bash
git clone https://github.com/Moshik21/engram.git ~/engram
cd ~/engram/server
uv sync
uv run engram setup
```
