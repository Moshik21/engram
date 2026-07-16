#!/usr/bin/env bash
# Engram — SessionEnd nudge for Loop Steward (silent subconscious ritual)
#
# Writes a short checklist. Does NOT call engram loop apply — harness owns that.
#
# Install (optional SessionEnd hook alongside session-promote-nudge):
#   cp hooks/session-steward-nudge.sh ~/.engram/hooks/ && chmod +x ...
set -euo pipefail

NUDGE_FILE="${ENGRAM_SESSION_STEWARD_NUDGE_FILE:-$HOME/.engram/session-steward-nudge.md}"

INPUT=$(cat 2>/dev/null || true)
if [ -n "${INPUT}" ]; then
    export ENGRAM_HOOK_INPUT="$INPUT"
else
    export ENGRAM_HOOK_INPUT="{}"
fi

eval "$(
  /usr/bin/python3 - "$NUDGE_FILE" <<'PY'
import datetime
import json
import os
import sys

nudge_path = sys.argv[1]
raw = os.environ.get("ENGRAM_HOOK_INPUT") or "{}"
try:
    payload = json.loads(raw)
except Exception:
    payload = {}

session_id = str(
    payload.get("session_id") or payload.get("sessionId") or "unknown"
)
cwd = str(payload.get("cwd") or payload.get("cwd_path") or "")
project = os.path.basename(cwd.rstrip("/")) if cwd else "unknown"
trigger = str(
    payload.get("hook_event_name") or payload.get("trigger") or "session_end"
)
now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

os.makedirs(os.path.dirname(nudge_path) or ".", exist_ok=True)
body = f"""# Engram Loop Steward nudge

- written_at: {now}
- session_id: {session_id}
- project: {project}
- trigger: {trigger}

## Silent harness checklist (do not narrate to the user)

**Rate limit:** at most **one** steward pass per session (or per 4h). Prefer a subagent.

**Preferred one-shot (operator/CLI/AXI only — never public MCP):**
```bash
engram loop steward-once --format json
# or: engram axi steward-once --format json
# dry-run: engram loop steward-once --dry-run --format json
```

1. Ensure runtime: `engramctl status` / health OK (start if needed).
2. Run one-shot above. If `healthy_noop=true` / regime healthy → stop.
3. Optional heavier: `engram loop steward-once --mop --budget 200`.
4. Verify lived path: get_context / continuity when server up.
5. Never expand public MCP. Never dump phases to the user.

If scoreboard is quiet — **do nothing**.
"""
with open(nudge_path, "w", encoding="utf-8") as fh:
    fh.write(body)

def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

print(f"NUDGE_FILE={sh_quote(nudge_path)}")
print(f"SESSION_ID={sh_quote(session_id)}")
print(f"PROJECT={sh_quote(project)}")
PY
)"

exit 0
