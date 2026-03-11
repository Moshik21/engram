#!/usr/bin/env bash
# Engram developer installer — repo clone + local uv environment.
set -euo pipefail

REPO="https://github.com/engram-labs/engram.git"
INSTALL_DIR="${ENGRAM_DIR:-$HOME/engram}"

RED='\033[0;31m'
GREEN='\033[0;32m'
DIM='\033[2m'
BOLD='\033[1m'
RESET='\033[0m'

info()  { echo -e "  ${GREEN}*${RESET} $1"; }
warn()  { echo -e "  ${RED}!${RESET} $1"; }
dim()   { echo -e "  ${DIM}$1${RESET}"; }

echo ""
echo -e "  ${BOLD}Engram Developer Installer${RESET}"
echo -e "  ${DIM}Repo clone + local uv environment${RESET}"
echo ""

if ! command -v python3 &>/dev/null; then
    warn "Python 3 is required but not found."
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

if ! command -v uv &>/dev/null; then
    info "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv &>/dev/null || warn "uv installation failed."
fi
info "uv $(uv --version 2>/dev/null | head -1)"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing install at $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --quiet
else
    info "Cloning Engram to $INSTALL_DIR"
    git clone --quiet "$REPO" "$INSTALL_DIR"
fi

info "Installing dependencies..."
cd "$INSTALL_DIR/server"
uv sync --quiet 2>/dev/null || uv sync

echo ""
echo -e "  ${BOLD}Running setup wizard...${RESET}"
echo ""
uv run python -m engram setup

echo ""
echo -e "  ${BOLD}${GREEN}Developer install complete!${RESET}"
echo ""
dim "  cd $INSTALL_DIR/server"
dim "  uv run engram serve"
dim "  uv run engram mcp"
dim "  uv run engram config"
echo ""
