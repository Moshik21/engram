#!/usr/bin/env bash
# Lite install path smoke test — exercises the full lite happy path end-to-end.
# No Docker required. Uses ENGRAM_PACKAGE_SOURCE and ENGRAM_SKILL_SOURCE to
# install from the local checkout instead of PyPI/GitHub.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

pass() { echo -e "  ${GREEN}✓${RESET} $1"; }
fail() { echo -e "  ${RED}✗${RESET} $1"; exit 1; }
phase() { echo -e "\n  ${BOLD}Phase $1: $2${RESET}"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ─── Phase 0: Environment Setup ──────────────────────────────────────

phase 0 "Environment setup"

TMP_ROOT="$(mktemp -d)"
cleanup() {
  # Stop any running engram process
  if [ -f "$TMP_ROOT/home/.engram/engram.pid" ]; then
    local pid
    pid="$(cat "$TMP_ROOT/home/.engram/engram.pid" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

export HOME="$TMP_ROOT/home"
mkdir -p "$HOME"

export BIN_DIR="$TMP_ROOT/bin"
export ENGRAM_HOME="$HOME/.engram"
export ENGRAM_INSTALL_BIN_DIR="$BIN_DIR"
export ENGRAM_INSTALL_NONINTERACTIVE=1
export ENGRAM_ANTHROPIC_API_KEY="test-key-smoke"
export ENGRAM_PACKAGE_SOURCE="$ROOT_DIR/server[local]"

# Ensure uv tool bin and our bin dir are on PATH
export PATH="$BIN_DIR:$HOME/.local/bin:$PATH"

pass "Temp root: $TMP_ROOT"

# ─── Phase 1: Bootstrap ──────────────────────────────────────────────

phase 1 "Bootstrap (install.sh lite)"

bash "$ROOT_DIR/scripts/install.sh" lite

pass "install.sh lite completed"

# ─── Phase 2: Post-Install Assertions ────────────────────────────────

phase 2 "Post-install assertions"

# engramctl should be installed (file or symlink)
if [ -f "$BIN_DIR/engramctl" ] || [ -L "$BIN_DIR/engramctl" ]; then
  pass "engramctl exists at $BIN_DIR/engramctl"
else
  fail "engramctl not found at $BIN_DIR/engramctl"
fi

# .env file should exist
if [ -f "$ENGRAM_HOME/.env" ]; then
  pass ".env file exists"
else
  fail ".env file not found at $ENGRAM_HOME/.env"
fi

# Variant should be lite
if grep -q 'ENGRAM_INSTALL_VARIANT=lite' "$ENGRAM_HOME/.env"; then
  pass "Variant is lite"
else
  fail "ENGRAM_INSTALL_VARIANT=lite not found in .env"
fi

# API key should be present
if grep -q 'ANTHROPIC_API_KEY=' "$ENGRAM_HOME/.env"; then
  pass "API key present in .env"
else
  fail "ANTHROPIC_API_KEY not found in .env"
fi

# engram command should be on PATH
if command -v engram >/dev/null 2>&1; then
  pass "engram command on PATH"
else
  fail "engram command not found on PATH"
fi

# ─── Phase 3: Runtime Lifecycle ──────────────────────────────────────

phase 3 "Runtime lifecycle"

engramctl start

# PID file should exist
if [ -f "$ENGRAM_HOME/engram.pid" ]; then
  pass "PID file created"
else
  fail "PID file not found after start"
fi

# Health check with retry
PORT="${ENGRAM_API_PORT:-8100}"
healthy=0
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" = "1" ]; then
  pass "Health check passed"
else
  # Print logs for debugging
  echo "--- engram.log ---"
  cat "$ENGRAM_HOME/logs/engram.log" 2>/dev/null || true
  echo "--- end ---"
  fail "Health check failed after 30s"
fi

# Status should mention lite
if engramctl status 2>&1 | grep -qi "lite"; then
  pass "Status reports lite mode"
else
  fail "Status does not mention lite"
fi

# Log file should exist
if [ -f "$ENGRAM_HOME/logs/engram.log" ]; then
  pass "Log file exists"
else
  fail "Log file not found"
fi

# Stop
engramctl stop

# PID file should be removed
if [ ! -f "$ENGRAM_HOME/engram.pid" ]; then
  pass "PID file removed after stop"
else
  fail "PID file still present after stop"
fi

# Restart cycle
engramctl start
healthy=0
for i in $(seq 1 30); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  sleep 1
done
if [ "$healthy" = "1" ]; then
  pass "Restart health check passed"
else
  fail "Restart health check failed"
fi
engramctl stop
pass "Restart cycle complete"

# ─── Phase 4: OpenClaw Skill Install ─────────────────────────────────

phase 4 "OpenClaw skill install"

export ENGRAM_SKILL_SOURCE="$ROOT_DIR/skills/engram-memory"

engramctl install-openclaw

if [ -f "$HOME/.openclaw/skills/engram-memory/SKILL.md" ]; then
  pass "OpenClaw skill installed"
else
  fail "SKILL.md not found after install-openclaw"
fi

# ─── Phase 5: Uninstall ──────────────────────────────────────────────

phase 5 "Uninstall"

engramctl uninstall --purge-data

# engramctl symlink should be removed
if [ ! -f "$BIN_DIR/engramctl" ] && [ ! -L "$BIN_DIR/engramctl" ]; then
  pass "engramctl removed"
else
  fail "engramctl still present after uninstall"
fi

# ENGRAM_HOME should be purged
if [ ! -d "$ENGRAM_HOME" ]; then
  pass "ENGRAM_HOME purged"
else
  fail "ENGRAM_HOME still exists after uninstall --purge-data"
fi

echo ""
echo -e "  ${BOLD}${GREEN}All lite smoke tests passed!${RESET}"
echo ""
