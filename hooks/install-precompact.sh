#!/usr/bin/env bash
# Install Engram PreCompact hook for Claude Code (promotion window reset).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/hooks/pre-compact.sh"
DEST_DIR="${ENGRAM_HOOKS_DIR:-$HOME/.engram/hooks}"
DEST="$DEST_DIR/pre-compact.sh"
SETTINGS="${CLAUDE_SETTINGS:-$HOME/.claude/settings.json}"

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST"
chmod +x "$DEST"

/usr/bin/python3 - "$DEST" "$SETTINGS" <<'PY'
import json
import sys
from pathlib import Path

hook_path, settings_path = sys.argv[1:3]
path = Path(settings_path)
if path.exists():
    data = json.loads(path.read_text())
else:
    data = {}
hooks = data.setdefault("hooks", {})
entry = {
    "matcher": "",
    "hooks": [
        {
            "type": "command",
            "command": hook_path,
            "timeout": 5000,
            "async": True,
        }
    ],
}
pre = hooks.get("PreCompact")
managed = hook_path
if not isinstance(pre, list):
    hooks["PreCompact"] = [entry]
else:
    kept = []
    for block in pre:
        cmds = [
            h
            for h in (block.get("hooks") or [])
            if h.get("command") != managed
        ]
        if cmds:
            block = dict(block)
            block["hooks"] = cmds
            kept.append(block)
    kept.append(entry)
    hooks["PreCompact"] = kept
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2) + "\n")
print(f"Installed PreCompact hook -> {hook_path}")
print(f"Updated {settings_path}")
PY

echo "Done. Compaction will refresh the 0–5 remember promotion window."
