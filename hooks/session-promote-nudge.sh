#!/usr/bin/env bash
# Engram — SessionEnd / Stop nudge for sparse promotion
#
# Writes a short promotion checklist the agent (or human) can act on.
# Does NOT call remember() — agents own extraction + sparse promotion.
#
# Install (optional second SessionEnd hook, or run from Stop):
#   cp hooks/session-promote-nudge.sh ~/.engram/hooks/ && chmod +x ...
set -euo pipefail

NUDGE_FILE="${ENGRAM_SESSION_PROMOTE_NUDGE_FILE:-$HOME/.engram/session-promote-nudge.md}"
WINDOW_FILE="${ENGRAM_PROMOTION_WINDOW_FILE:-$HOME/.engram/promotion-window.json}"
CAP="${ENGRAM_PROMOTE_CAP:-5}"

INPUT=$(cat 2>/dev/null || true)
# Quote default "{}" — bare ${VAR:-{}} leaves a trailing } when VAR is set.
if [ -n "${INPUT}" ]; then
    export ENGRAM_HOOK_INPUT="$INPUT"
else
    export ENGRAM_HOOK_INPUT="{}"
fi

eval "$(
  /usr/bin/python3 - "$NUDGE_FILE" "$WINDOW_FILE" "$CAP" <<'PY'
import datetime
import json
import os
import sys

nudge_path, window_path, cap_raw = sys.argv[1:4]
try:
    cap = max(0, min(5, int(cap_raw)))
except Exception:
    cap = 5

raw = os.environ.get("ENGRAM_HOOK_INPUT") or "{}"
try:
    payload = json.loads(raw)
except Exception:
    payload = {}

session_id = str(
    payload.get("session_id")
    or payload.get("sessionId")
    or "unknown"
)
cwd = str(payload.get("cwd") or payload.get("cwd_path") or "")
project = os.path.basename(cwd.rstrip("/")) if cwd else "unknown"
trigger = str(
    payload.get("hook_event_name")
    or payload.get("trigger")
    or "session_end"
)
now = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")

compaction_id = ""
if os.path.isfile(window_path):
    try:
        with open(window_path, encoding="utf-8") as fh:
            doc = json.load(fh)
        compaction_id = str(doc.get("compaction_id") or "")
    except Exception:
        compaction_id = ""

os.makedirs(os.path.dirname(nudge_path) or ".", exist_ok=True)
body = f"""# Engram session promote nudge

- written_at: {now}
- session_id: {session_id}
- project: {project}
- trigger: {trigger}
- compaction_id: {compaction_id or "(none — PreCompact will stamp one)"}
- promote_cap: {cap}

## Agent checklist (≤{cap})

1. Invoke skill `engram-session-promote` or follow `docs/GOLDEN_LOOP.md`.
2. Propose **0–{cap}** high-signal Decision/Preference/Person/Correction/Goal/Commitment facts.
3. Call `remember()` with `proposed_entities` + `proposed_relationships` + verbatim `source_span`.
4. Do **not** dump session recaps. Prefer silence over noise.
5. Confirm cold continuity later with `get_context` / `recall`.

Product KPI: fresh agent surfaces ≥1 Decision without a handoff doc.
Open_work / adjudication queue is **hygiene**, not success.
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

# Best-effort: also append a trace line for adoption evidence.
TRACE_FILE="${ENGRAM_ADOPTION_TRACE_FILE:-$HOME/.engram/adoption-trace.jsonl}"
mkdir -p "$(dirname "$TRACE_FILE")"
/usr/bin/python3 - "$TRACE_FILE" "${SESSION_ID:-unknown}" <<'PY' 2>/dev/null || true
import datetime, json, sys
path, session_id = sys.argv[1], sys.argv[2]
with open(path, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({
        "phase": "promote",
        "tool": "session_promote_nudge",
        "source": "hook_session_promote_nudge",
        "client": "Claude Code",
        "capturedAt": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "session_id": session_id,
    }) + "\n")
PY

exit 0
