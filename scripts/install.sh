#!/usr/bin/env bash
# Engram installer — sets up persistent memory for AI agents.
# Usage: curl -sSL https://engram.run/install | bash
set -euo pipefail

REPO="https://github.com/Moshik21/engram.git"
INSTALL_DIR="${ENGRAM_DIR:-$HOME/engram}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "  ${GREEN}*${RESET} $1"; }
warn()  { echo -e "  ${RED}!${RESET} $1"; }
dim()   { echo -e "  ${DIM}$1${RESET}"; }

echo ""
echo -e "  ${BOLD}Engram Installer${RESET}"
echo -e "  ${DIM}Persistent memory for AI agents${RESET}"
echo ""

# --- Check Python ---
if ! command -v python3 &>/dev/null; then
    warn "Python 3 is required but not found."
    warn "Install it from https://python.org or your package manager."
    exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    warn "Python 3.10+ required, found $PY_VERSION"
    exit 1
fi
info "Python $PY_VERSION"

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
    info "Installing uv (Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv &>/dev/null; then
        warn "uv installation failed. Install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
fi
info "uv $(uv --version 2>/dev/null | head -1)"

# --- Clone or update repo ---
if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing install at $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --quiet
else
    info "Cloning Engram to $INSTALL_DIR"
    git clone --quiet "$REPO" "$INSTALL_DIR"
fi

# --- Install dependencies ---
info "Installing dependencies..."
cd "$INSTALL_DIR/server"
uv sync --quiet 2>/dev/null || uv sync

info "Engram installed at $INSTALL_DIR"
echo ""

# --- Run setup wizard ---
echo -e "  ${BOLD}Running setup wizard...${RESET}"
echo ""
uv run python -m engram setup

echo ""
echo -e "  ${BOLD}${GREEN}Installation complete!${RESET}"
echo ""
echo -e "  ${BOLD}Quick reference:${RESET}"
dim "  cd $INSTALL_DIR/server"
echo ""
dim "  uv run engram serve          # Start REST API (port 8100)"
dim "  uv run engram mcp            # Start MCP server (stdio)"
dim "  uv run engram config         # Edit settings"
dim "  uv run engram health         # Check if server is running"
echo ""
