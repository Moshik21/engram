#!/usr/bin/env bash
# Engram — Claude Code PreCompact / context-compression hook
#
# When the harness is about to compact context, stamp a new promotion window so
# sparse remember() budgets (0–5 durable facts) reset for the next context era.
# Also records a cheap auto-observe marker for the brain trail.
#
# Install:
#   cp hooks/pre-compact.sh ~/.engram/hooks/pre-compact.sh && chmod +x ...
#   Add PreCompact command hook pointing at that path (see docs/CURRENT_HANDOFF.md).
set -euo pipefail

ENGRAM_URL="${ENGRAM_URL:-http://localhost:8100}"
WINDOW_FILE="${ENGRAM_PROMOTION_WINDOW_FILE:-$HOME/.engram/promotion-window.json}"
TRACE_FILE="${ENGRAM_ADOPTION_TRACE_FILE:-$HOME/.engram/adoption-trace.jsonl}"
QUEUE_FILE="${ENGRAM_CAPTURE_QUEUE_FILE:-$HOME/.engram/capture-queue.jsonl}"

INPUT=$(cat)
export ENGRAM_HOOK_INPUT="$INPUT"

# All parsing + window stamp in one Python pass (hook JSON shapes vary).
# NOTE: must not use a heredoc as stdin for the hook payload — payload is in env.
eval "$(
  /usr/bin/python3 - "$WINDOW_FILE" <<'PY'
import datetime
import hashlib
import json
import os
import sys
import time

window_path = sys.argv[1]
raw_input = os.environ.get("ENGRAM_HOOK_INPUT") or "{}"
try:
    payload = json.loads(raw_input)
except Exception:
    payload = {}

session_id = str(
    payload.get("session_id")
    or payload.get("sessionId")
    or "unknown"
)
cwd = str(payload.get("cwd") or payload.get("cwd_path") or "")
trigger = str(
    payload.get("trigger")
    or payload.get("reason")
    or payload.get("hook_event_name")
    or "precompact"
)
project = os.path.basename(cwd.rstrip("/")) if cwd else "unknown"
raw = f"{session_id}:{time.time():.3f}:precompact"
compaction_id = "compact_" + hashlib.sha1(raw.encode()).hexdigest()[:16]
reset_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

os.makedirs(os.path.dirname(window_path) or ".", exist_ok=True)
doc = {
    "compaction_id": compaction_id,
    "session_id": session_id,
    "trigger": trigger,
    "project": project,
    "source": "claude:precompact",
    "reset_at": reset_at,
}
with open(window_path, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2)
    fh.write("\n")

def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"

print(f"SESSION_ID={sh_quote(session_id)}")
print(f"PROJECT={sh_quote(project)}")
print(f"TRIGGER={sh_quote(trigger)}")
print(f"COMPACTION_ID={sh_quote(compaction_id)}")
print(f"WINDOW_FILE={sh_quote(window_path)}")
PY
)"

write_trace() {
    local phase="$1"
    local tool="$2"
    local source="$3"
    local session_id="$4"
    mkdir -p "$(dirname "$TRACE_FILE")"
    /usr/bin/python3 - "$phase" "$tool" "$source" "$session_id" >> "$TRACE_FILE" <<'PY'
import datetime
import json
import sys

phase, tool, source, session_id = sys.argv[1:5]
captured_at = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
print(json.dumps({
    "phase": phase,
    "tool": tool,
    "source": source,
    "client": "Claude Code",
    "capturedAt": captured_at,
    "session_id": session_id,
}))
PY
}

CONTENT="[compaction|${PROJECT}|${SESSION_ID}] Context compaction; promotion window ${COMPACTION_ID}"
PAYLOAD=$(/usr/bin/python3 -c "
import json, sys
print(json.dumps({
    'content': sys.argv[1],
    'source': 'claude:precompact',
    'project': sys.argv[2],
}))
" "$CONTENT" "$PROJECT")

if curl -sf "${ENGRAM_URL}/health" --connect-timeout 1 --max-time 2 > /dev/null 2>&1; then
    if curl -sf -X POST "${ENGRAM_URL}/api/knowledge/auto-observe" \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD" \
        --connect-timeout 1 \
        --max-time 3 \
        > /dev/null 2>&1; then
        write_trace "capture" "auto_observe" "claude:precompact" "$SESSION_ID"
    else
        mkdir -p "$(dirname "$QUEUE_FILE")"
        echo "$PAYLOAD" >> "$QUEUE_FILE"
    fi
else
    mkdir -p "$(dirname "$QUEUE_FILE")"
    echo "$PAYLOAD" >> "$QUEUE_FILE"
fi

# Never block compaction on Engram failure.
exit 0
