"""I3: concurrent open of one native Helix data dir (two processes / one process).

The MCP stdio path (engram/mcp/server.py::_init) builds stores directly with no
lock; the brain flock (engram/brain_runtime.py) only serializes brain-vs-shell.
These tests pin what ACTUALLY happens when a second opener targets the same
``--helix-data-dir``:

- Cross-process: helix-db opens LMDB via heed3 with default flags (no NOLOCK —
  native/helix-repo/helix-db/src/helix_engine/storage_core/mod.rs:84-90), so
  LMDB's lock.mdb multi-process protocol applies: second open SUCCEEDS, writers
  serialize, committed writes are visible both ways.
- Same-process: heed3 keeps a global registry of opened env paths and returns
  ``EnvAlreadyOpened`` — a clean, deterministic failure (this is why
  NativeTransport caches engines per data dir until process exit).

Never touches the live brain: every opener uses a pytest tmp_path data dir and
the second opener runs in a subprocess with HOME pointed at a scratch dir.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import textwrap

import pytest

from engram.config import HelixDBConfig
from engram.storage.helix.graph import HelixGraphStore

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("helix_native") is None,
    reason="helix_native PyO3 extension is not installed",
)

GROUP_ID = "concurrent_open"

_CHILD_SCRIPT = textwrap.dedent(
    """
    import asyncio
    import json
    import sys

    from engram.config import HelixDBConfig
    from engram.storage.helix.graph import HelixGraphStore


    def _evidence(evidence_id: str) -> dict:
        return {
            "evidence_id": evidence_id,
            "episode_id": "ep_concurrent_open",
            "fact_class": "relationship",
            "confidence": 0.5,
            "source_type": "test",
            "extractor_name": "concurrent-open-test",
            "payload": {"subject": evidence_id},
            "source_span": f"Evidence {evidence_id}",
            "corroborating_signals": [],
            "ambiguity_tags": [],
            "ambiguity_score": 0.0,
            "status": "pending",
        }


    async def main() -> None:
        data_dir, result_path, group_id = sys.argv[1], sys.argv[2], sys.argv[3]
        store = HelixGraphStore(
            HelixDBConfig(transport="native", data_dir=data_dir)
        )
        await store.initialize()
        try:
            before = await store.get_pending_evidence(group_id)
            await store.store_evidence([_evidence("ev_child")], group_id=group_id)
            after = await store.get_pending_evidence(group_id)
            payload = {
                "open_ok": True,
                "seen_before": sorted(item["evidence_id"] for item in before),
                "seen_after": sorted(item["evidence_id"] for item in after),
            }
        finally:
            await store.close()
        with open(result_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)


    asyncio.run(main())
    """
)


def _evidence(evidence_id: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "episode_id": "ep_concurrent_open",
        "fact_class": "relationship",
        "confidence": 0.5,
        "source_type": "test",
        "extractor_name": "concurrent-open-test",
        "payload": {"subject": evidence_id},
        "source_span": f"Evidence {evidence_id}",
        "corroborating_signals": [],
        "ambiguity_tags": [],
        "ambiguity_score": 0.0,
        "status": "pending",
    }


@pytest.mark.asyncio
async def test_second_process_open_succeeds_and_shares_committed_state(
    tmp_path,
) -> None:
    """Two processes on one native dir: second open works, writes interleave."""
    data_dir = tmp_path / "native-concurrent-open"
    result_path = tmp_path / "child-result.json"
    fake_home = tmp_path / "fakehome"
    fake_home.mkdir()

    store = HelixGraphStore(HelixDBConfig(transport="native", data_dir=str(data_dir)))
    await store.initialize()
    try:
        await store.store_evidence([_evidence("ev_parent")], group_id=GROUP_ID)

        env = dict(os.environ)
        env["HOME"] = str(fake_home)
        # While THIS process still holds the LMDB env open, a second process
        # opens the same data dir.
        proc = subprocess.run(
            [sys.executable, "-c", _CHILD_SCRIPT, str(data_dir), str(result_path), GROUP_ID],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, (
            f"second-process open failed\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )
        child = json.loads(result_path.read_text(encoding="utf-8"))

        # The child saw the parent's committed write on open...
        assert child["open_ok"] is True
        assert child["seen_before"] == ["ev_parent"]
        assert child["seen_after"] == ["ev_child", "ev_parent"]

        # ...and the parent sees the child's committed write afterwards
        # (fresh read txns observe the latest committed state — no exclusion,
        # no corruption, LMDB serialized the writers via lock.mdb).
        after = await store.get_pending_evidence(GROUP_ID)
        assert sorted(item["evidence_id"] for item in after) == [
            "ev_child",
            "ev_parent",
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_same_process_second_open_fails_cleanly(tmp_path) -> None:
    """Same-process double open is rejected by heed3's env registry."""
    import helix_native

    data_dir = tmp_path / "native-same-process-open"

    store = HelixGraphStore(HelixDBConfig(transport="native", data_dir=str(data_dir)))
    await store.initialize()
    try:
        # Bypass NativeTransport's per-process engine cache: a raw second
        # engine on the same canonical path must fail with EnvAlreadyOpened,
        # not corrupt or hang.
        with pytest.raises(RuntimeError, match="(?i)already open"):
            helix_native.HelixEngine(data_dir=str(data_dir))
    finally:
        await store.close()
