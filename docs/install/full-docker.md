# Full Docker Install

> For a lighter install without Docker, see [Lite Install](lite.md).

Engram's full mode uses FalkorDB + Redis for scale and performance. It runs as
a prebuilt local Docker stack — no repo clone, no image builds.

## One-click install

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash
```

Select **[2] Full** when prompted. Or skip the prompt:

```bash
curl -sSL https://raw.githubusercontent.com/Moshik21/engram/main/scripts/install.sh | bash -s -- full
```

What it does:

- checks Docker and Docker Compose availability
- downloads the latest release install bundle
- writes local config into `~/.engram/full/.env`
- creates bind-mounted data dirs under `~/.engram/full/data/`
- pulls prebuilt images and starts the full stack
- opens the dashboard at `http://127.0.0.1:3000`

Result:

- Dashboard: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8100`
- Runtime: `ENGRAM_MODE=full`
- Profiles: `consolidation_profile=standard`, `recall_profile=all`, `integration_profile=rework`

## Lifecycle commands

The installer adds `engramctl` to `~/.local/bin` when possible.

```bash
engramctl status
engramctl logs
engramctl stop
engramctl start
engramctl update
engramctl uninstall
engramctl uninstall --purge-data
```

`engramctl uninstall` preserves `~/.engram/full/.env` and `~/.engram/full/data`
by default. Use `--purge-data` to remove everything.

## Advanced / developer path

If you want to build from local source instead, use the repo-root
[`docker-compose.yml`](../../docker-compose.yml) or the developer installer in
[`scripts/dev-install.sh`](../../scripts/dev-install.sh).
