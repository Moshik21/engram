"""M5.3 — Thompson Sampling KILL contract (F4 resolved: KILL).

Pins the deletion: no TS scorer, no TS knobs, no TS posterior fields, no
activation/feedback module — and snapshot/journal loaders TOLERATE + DROP
the old ts_alpha/ts_beta fields instead of crashing or resurrecting them.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, fields

import pytest

from engram.config import ActivationConfig
from engram.models.activation import ActivationState
from engram.storage.memory.activation import MemoryActivationStore


class TestTsSurfaceDeleted:
    def test_scorer_module_has_no_thompson_scorer(self):
        import engram.retrieval.scorer as scorer

        assert not hasattr(scorer, "score_candidates_thompson")

    def test_activation_package_has_no_feedback_module(self):
        with pytest.raises(ImportError):
            from engram.activation import feedback  # noqa: F401

    def test_config_has_no_ts_knobs(self):
        cfg_fields = set(ActivationConfig.model_fields)
        for knob in ("ts_enabled", "ts_weight", "ts_positive_increment", "ts_negative_increment"):
            assert knob not in cfg_fields

    def test_activation_state_has_no_ts_fields(self):
        names = {f.name for f in fields(ActivationState)}
        assert "ts_alpha" not in names
        assert "ts_beta" not in names


class TestSnapshotToleratesAndDropsTsFields:
    def test_v2_snapshot_with_ts_fields_loads_and_drops(self, tmp_path):
        now = time.time()
        payload = {
            "version": 2,
            "saved_at": now,
            "states": {
                "ent-1": {
                    "node_id": "ent-1",
                    "access_history": [now - 60.0],
                    "last_accessed": now - 60.0,
                    "access_count": 1,
                    "consolidated_strength": 0.1,
                    "last_compacted": 0.0,
                    # pre-M5.3 posterior fields — must be tolerated + dropped
                    "ts_alpha": 7.0,
                    "ts_beta": 3.0,
                    "usage_events": [[now - 60.0, 1.0]],
                    "usage_weight_sum": 1.0,
                    "usage_last_ts": now - 60.0,
                    "group_id": "default",
                }
            },
        }
        snap = tmp_path / "activation-snapshot.json"
        snap.write_text(json.dumps(payload), encoding="utf-8")

        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        loaded = store.load_from_file(snap)

        assert loaded == 1
        state = store._states["ent-1"]
        assert not hasattr(state, "ts_alpha")
        assert not hasattr(state, "ts_beta")
        assert state.n_eff == pytest.approx(1.0)
        # Round-trip: the re-saved snapshot no longer carries ts fields.
        assert "ts_alpha" not in asdict(state)

    def test_resaved_snapshot_carries_no_ts_fields(self, tmp_path):
        store = MemoryActivationStore(journal_path=tmp_path / "journal.jsonl")
        state = ActivationState(node_id="ent-2", access_history=[1.0], access_count=1)
        store._states["ent-2"] = state
        out = tmp_path / "snap.json"
        assert store.save_to_file(out) == 1
        payload = json.loads(out.read_text(encoding="utf-8"))
        entry = payload["states"]["ent-2"]
        assert "ts_alpha" not in entry
        assert "ts_beta" not in entry
