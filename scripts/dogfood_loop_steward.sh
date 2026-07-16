#!/usr/bin/env bash
# Dogfood checklist for Loop Steward control plane.
# apply → status → (optional) continuity; honest SKIP when server down.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER_URL="${ENGRAM_SERVER_URL:-http://127.0.0.1:8100}"
WORKDIR="${ENGRAM_DOGFOOD_LOOP_DIR:-$(mktemp -d /tmp/engram-loop-dogfood-XXXXXX)}"
export ENGRAM_LOOP_ADJUSTMENT_FILE="${ENGRAM_LOOP_ADJUSTMENT_FILE:-$WORKDIR/loop-adjustment.json}"
export ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE="${ENGRAM_LOOP_ADJUSTMENT_AUDIT_FILE:-$WORKDIR/loop-adjustments.jsonl}"
mkdir -p "$WORKDIR"

cd "$ROOT/server"

echo "== Loop Steward dogfood =="
echo "workdir=$WORKDIR"
echo "adjustment_file=$ENGRAM_LOOP_ADJUSTMENT_FILE"

# 1) Deterministic propose-from-report (no write)
DEBT_JSON="$WORKDIR/debt.json"
cat >"$DEBT_JSON" <<'JSON'
{
  "debt": {
    "deferred_evidence": 2500,
    "cue_only_episodes": 400,
    "open_work": 3000
  },
  "pressure": {
    "should_trigger_mop": true,
    "total": 200,
    "threshold": 100
  }
}
JSON

echo "-- propose-from-report"
uv run engram loop propose-from-report \
  --debt-json "$DEBT_JSON" \
  --format json \
  --created-by "dogfood:loop-steward" \
  | tee "$WORKDIR/propose.json"

# Extract adjustment for apply
python3 - "$WORKDIR/propose.json" "$WORKDIR/adj.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
doc = json.load(open(src, encoding="utf-8"))
adj = doc.get("adjustment") or doc
json.dump(adj, open(dst, "w", encoding="utf-8"), indent=2)
print("wrote", dst, "regime=", adj.get("regime"))
PY

echo "-- status before apply (expect none)"
uv run engram loop status --format json | tee "$WORKDIR/status-before.json"

echo "-- apply"
uv run engram loop apply \
  --file "$WORKDIR/adj.json" \
  --skip-continuity-check \
  --format json \
  | tee "$WORKDIR/apply.json"

echo "-- status after apply (expect active)"
uv run engram loop status --format json | tee "$WORKDIR/status-after.json"

# Continuity if server up
if curl -sf -m 3 "$SERVER_URL/health" >/dev/null; then
  echo "-- continuity --against-live"
  uv run engram continuity --against-live --max-recall-ms 4000 \
    | tee "$WORKDIR/continuity.txt"
  echo "LIVE_CONTINUITY=ran" | tee -a "$WORKDIR/summary.txt"
else
  echo "SKIP: server not reachable at $SERVER_URL for continuity" \
    | tee "$WORKDIR/continuity.txt"
  echo "LIVE_CONTINUITY=skip" | tee -a "$WORKDIR/summary.txt"
fi

echo "-- clear"
uv run engram loop clear --format json | tee "$WORKDIR/clear.json"

echo "OK dogfood complete workdir=$WORKDIR"
echo "WORKDIR=$WORKDIR"
