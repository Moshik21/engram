#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
MODE="${1:-full}"
SKIP_RUNTIME=0

if [ "$MODE" = "--skip-runtime" ]; then
  SKIP_RUNTIME=1
  MODE="${2:-openclaw}"
fi

TMP_ROOT="$(mktemp -d)"
trap 'rm -rf "$TMP_ROOT"' EXIT

export HOME="$TMP_ROOT/home"
mkdir -p "$HOME"

INSTALL_ROOT="$TMP_ROOT/install-root"
BIN_DIR="$TMP_ROOT/bin"
OUTPUT_DIR="$TMP_ROOT/assets"
mkdir -p "$BIN_DIR" "$OUTPUT_DIR"

if [ "$SKIP_RUNTIME" != "1" ]; then
  docker build -t engramci/engram-server:smoke -f "$ROOT_DIR/server/Dockerfile" "$ROOT_DIR"
  docker build -t engramci/engram-dashboard:smoke "$ROOT_DIR/dashboard"
fi

python3 "$ROOT_DIR/scripts/build_install_bundle.py" \
  --version smoke \
  --output-dir "$OUTPUT_DIR" \
  --image-namespace engramci

export ENGRAM_INSTALL_ROOT="$INSTALL_ROOT"
export ENGRAM_INSTALL_BIN_DIR="$BIN_DIR"
export ENGRAM_INSTALL_SKIP_OPEN=1
export ENGRAM_INSTALL_NONINTERACTIVE=1
export ENGRAM_ANTHROPIC_API_KEY="test-anthropic-key"
export ENGRAM_BUNDLE_URL="file://$OUTPUT_DIR/engram-install-bundle.tar.gz"
export ENGRAM_BUNDLE_SHA256_URL="file://$OUTPUT_DIR/engram-install-bundle.sha256"
export PATH="$BIN_DIR:$PATH"

if [ "$SKIP_RUNTIME" = "1" ]; then
  export ENGRAM_INSTALL_SKIP_RUNTIME=1
fi

bash "$ROOT_DIR/scripts/install.sh" "$MODE"
test -L "$BIN_DIR/engramctl"
test -f "$INSTALL_ROOT/.env"
test -f "$INSTALL_ROOT/current/compose.yaml"

if [ "$MODE" = "openclaw" ] || [ "$SKIP_RUNTIME" = "1" ]; then
  test -f "$HOME/.openclaw/skills/engram-memory/SKILL.md"
fi

if [ "$SKIP_RUNTIME" != "1" ]; then
  curl -fsS "http://127.0.0.1:8100/health" >/dev/null
  curl -fsS "http://127.0.0.1:3000/health" >/dev/null
  engramctl status >/dev/null
  engramctl update >/dev/null
  engramctl stop >/dev/null
  engramctl start >/dev/null
  engramctl uninstall --purge-data >/dev/null
else
  engramctl uninstall --purge-data >/dev/null
fi

echo "Installer smoke test passed."
