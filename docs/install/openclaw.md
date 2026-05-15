# OpenClaw + Engram

> **Note:** The OpenClaw integration has not been fully tested yet. These instructions are provided as-is and may require adjustments. Please report issues if you encounter problems.

The OpenClaw skill connects to the local Engram REST API on `127.0.0.1:8100`.
It works with native Helix, lite, and Docker-backed Engram installs.

## One-click install

### Native Helix (recommended — no Docker)

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- helix
```

The Helix path verifies that the native `helix_native` PyO3 runtime is
available. If the public package source does not include native support yet,
install from source and run `make build-native` before starting Engram.

This uses HelixDB in-process through PyO3, giving OpenClaw the full graph/vector/BM25
backend without Docker. Answer **yes** to "Install OpenClaw skill?" during setup.

### Lite fallback

Use lite only when you want a disposable SQLite-only install:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- lite
```

Then install the skill:

```bash
engramctl install-openclaw
```

### Full Docker fallback

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- full
```

Then install the skill:

```bash
engramctl install-openclaw
```

Or use the combined openclaw mode:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- openclaw
```

## Runtime contract

- OpenClaw talks to the local Engram REST API on `127.0.0.1:8100`
- Works with native Helix, lite (SQLite), and Docker-backed backends
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
