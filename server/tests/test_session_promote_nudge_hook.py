"""Session-end promote nudge hook writes a checklist file."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


def test_session_promote_nudge_writes_checklist(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    hook = repo_root / "hooks" / "session-promote-nudge.sh"
    assert hook.is_file()

    nudge = tmp_path / "session-promote-nudge.md"
    window = tmp_path / "promotion-window.json"
    window.write_text(
        json.dumps({"compaction_id": "compact_test123"}),
        encoding="utf-8",
    )
    trace = tmp_path / "adoption-trace.jsonl"

    env = os.environ.copy()
    env["ENGRAM_SESSION_PROMOTE_NUDGE_FILE"] = str(nudge)
    env["ENGRAM_PROMOTION_WINDOW_FILE"] = str(window)
    env["ENGRAM_ADOPTION_TRACE_FILE"] = str(trace)
    env["ENGRAM_PROMOTE_CAP"] = "5"

    payload = json.dumps(
        {
            "session_id": "sess_abc",
            "cwd": "/Users/konnermoshier/Engram",
            "hook_event_name": "SessionEnd",
        }
    )
    result = subprocess.run(
        ["bash", str(hook)],
        input=payload,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert nudge.is_file()
    text = nudge.read_text(encoding="utf-8")
    assert "engram-session-promote" in text
    assert "compact_test123" in text
    assert "promote_cap: 5" in text
    assert "sess_abc" in text
