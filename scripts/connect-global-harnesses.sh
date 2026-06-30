#!/usr/bin/env bash
# Wire Engram MCP globally across Codex, Claude Desktop, Windsurf, Cursor, Grok, Claude Code.
# Memory is global; this script connects each harness to http://127.0.0.1:8100/mcp.
#
# Usage:
#   bash scripts/connect-global-harnesses.sh
#   ENGRAM_PROJECTS="$HOME/Engram $HOME/MachineShopScheduler" bash scripts/connect-global-harnesses.sh
#
# Optional env:
#   ENGRAMCTL          path to engramctl (default: engramctl on PATH)
#   ENGRAM_CLI         path to engram CLI for harness priming (auto-detected)
#   ENGRAM_API_PORT    default 8100
#   ENGRAM_PROJECTS    space-separated project dirs for cursor/grok/claude-code + bootstrap
#   SKIP_BOOTSTRAP=1   skip engramctl bootstrap for each project
#   SKIP_AXI=1         skip AXI hooks for codex/claude-code

set -euo pipefail

ENGRAMCTL="${ENGRAMCTL:-engramctl}"
PORT="${ENGRAM_API_PORT:-8100}"
MCP_URL="http://127.0.0.1:${PORT}/mcp"
PROJECTS="${ENGRAM_PROJECTS:-$HOME/Engram $HOME/MachineShopScheduler}"
CONNECT_PROJECT="${ENGRAM_CONNECT_PROJECT:-$HOME/Engram}"

die() { echo "error: $*" >&2; exit 1; }
info() { echo "==> $*"; }
dim() { echo "    $*"; }

command -v "$ENGRAMCTL" >/dev/null 2>&1 || die "engramctl not found. Install Engram first."

if ! curl --connect-timeout 2 --max-time 5 -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  die "Engram API not reachable at http://127.0.0.1:${PORT}. Run: engramctl start"
fi

resolve_engram_cli() {
  if [ -n "${ENGRAM_CLI:-}" ] && command -v "$ENGRAM_CLI" >/dev/null 2>&1; then
    echo "$ENGRAM_CLI"
    return
  fi
  for candidate in \
    "$HOME/Engram/server/.venv/bin/engram" \
    "$HOME/.local/bin/engram" \
    engram; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done
  echo ""
}

ENGRAM_CLI="$(resolve_engram_cli)"

merge_mcp_http_config() {
  local target="$1"
  local url="$2"
  mkdir -p "$(dirname "$target")"
  python3 - "$target" "$url" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
url = sys.argv[2]
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text())
        data = {}
data.setdefault("mcpServers", {})["engram"] = {"type": "http", "url": url}
path.write_text(json.dumps(data, indent=2) + "\n")
print(f"    engram -> {url} ({path})")
PY
}

write_priming() {
  local client="$1"
  local project="$2"
  [ -n "$ENGRAM_CLI" ] || { dim "Skipping priming for $project ($client): engram CLI not found"; return 0; }
  "$ENGRAM_CLI" harness write-priming --client "$client" --project "$project" >/dev/null
  dim "Priming: $project/.$( [ "$client" = cursor ] && echo cursor/rules/engram-memory.mdc || echo grok/rules/engram-memory.md )"
}

[ -d "$CONNECT_PROJECT" ] || die "CONNECT_PROJECT does not exist: $CONNECT_PROJECT"

info "Engram healthy on port ${PORT}"

# --- Global MCP ---
info "Connecting Codex (global ~/.codex/config.toml)"
if [ "${SKIP_AXI:-0}" = "1" ]; then
  (cd "$CONNECT_PROJECT" && "$ENGRAMCTL" connect codex)
else
  (cd "$CONNECT_PROJECT" && "$ENGRAMCTL" connect codex --axi)
fi

if [ -d "$HOME/Library/Application Support/Claude" ]; then
  info "Connecting Claude Desktop (global)"
  (cd "$CONNECT_PROJECT" && "$ENGRAMCTL" connect claude-desktop)
else
  dim "Skipping Claude Desktop (not installed)"
fi

if [ -d "$HOME/.codeium/windsurf" ] || command -v windsurf >/dev/null 2>&1; then
  info "Connecting Windsurf (global)"
  (cd "$CONNECT_PROJECT" && "$ENGRAMCTL" connect windsurf)
else
  dim "Skipping Windsurf (not detected)"
fi

if command -v openclaw >/dev/null 2>&1 || command -v npx >/dev/null 2>&1; then
  info "Connecting OpenClaw (global)"
  (cd "$CONNECT_PROJECT" && "$ENGRAMCTL" connect openclaw) || dim "OpenClaw connect skipped or failed"
else
  dim "Skipping OpenClaw (openclaw/npx not found)"
fi

# --- Cursor global ---
CURSOR_MCP="$HOME/.cursor/mcp.json"
info "Merging Engram into global Cursor MCP: $CURSOR_MCP"
merge_mcp_http_config "$CURSOR_MCP" "$MCP_URL"

# --- Per-project ---
for project in $PROJECTS; do
  [ -d "$project" ] || { dim "Skipping missing project: $project"; continue; }
  info "Project setup: $project"

  info "  Cursor project MCP + priming"
  (cd "$project" && "$ENGRAMCTL" connect cursor)
  write_priming cursor "$project"

  info "  Grok Build project MCP + priming"
  merge_mcp_http_config "$project/.grok/mcp.json" "$MCP_URL"
  write_priming grok-build "$project"

  if command -v claude >/dev/null 2>&1; then
    info "  Claude Code project MCP"
    if [ "${SKIP_AXI:-0}" = "1" ]; then
      (cd "$project" && "$ENGRAMCTL" connect claude-code)
    else
      (cd "$project" && "$ENGRAMCTL" connect claude-code --axi)
    fi
  fi

  if [ "${SKIP_BOOTSTRAP:-0}" != "1" ]; then
    info "  Bootstrap artifacts"
    "$ENGRAMCTL" bootstrap "$project" || dim "  bootstrap skipped or already done"
  fi
done

info "Done. Restart agents or refresh MCP tools."
dim "Cursor: Settings -> MCP -> confirm 'engram' is enabled (global + per-project)"
dim "Codex: new session picks up ~/.codex/config.toml + AXI hooks"
dim "Test: ask any agent 'What do you know about me from Engram?' before answering"