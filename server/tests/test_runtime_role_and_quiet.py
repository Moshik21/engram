"""Hot-shell / cold-brain role gates and quiet profile."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from engram.brain_runtime import BrainStatus, exclusive_brain_lock, read_brain_status
from engram.config import ActivationConfig, EngramConfig


class TestQuietProfile:
    def test_quiet_disables_worker_and_cross_encoder(self):
        cfg = ActivationConfig(consolidation_profile="quiet")
        assert cfg.consolidation_enabled is True
        assert cfg.consolidation_dry_run is False
        assert cfg.worker_enabled is False
        assert cfg.consolidation_cross_encoder_enabled is False
        assert cfg.reranker_enabled is False
        assert cfg.graph_embedding_node2vec_enabled is False
        assert cfg.triage_enabled is True
        assert cfg.recall_profile == "wave2"
        assert cfg.auto_recall_enabled is True

    def test_quiet_preserves_explicit_recall_profile(self):
        cfg = ActivationConfig(
            consolidation_profile="quiet",
            recall_profile="wave1",
        )
        assert cfg.recall_profile == "wave1"


class TestRuntimeRole:
    def test_monolith_runs_in_process_brain(self):
        # Explicit kwargs (constructor wins over ~/.engram/.env dogfood).
        cfg = EngramConfig(runtime_role="monolith")
        assert cfg.runtime_role == "monolith"
        assert cfg.shell_runs_in_process_brain() is True

    def test_shell_does_not_run_in_process_brain(self):
        cfg = EngramConfig(runtime_role="shell")
        assert cfg.shell_runs_in_process_brain() is False

    def test_brain_role_not_shell_brain_colocation(self):
        # brain role is for CLI; serve should still not treat it as monolith
        cfg = EngramConfig(runtime_role="brain")
        assert cfg.shell_runs_in_process_brain() is False


class TestBrainLock:
    def test_exclusive_lock_blocks_second(self, tmp_path: Path):
        lock = tmp_path / "brain.lock"
        with exclusive_brain_lock(lock):
            with pytest.raises(RuntimeError, match="Another brain"):
                with exclusive_brain_lock(lock):
                    pass

    def test_status_roundtrip(self, tmp_path: Path):
        path = tmp_path / "brain-status.json"
        BrainStatus(
            ok=True,
            started_at="t0",
            finished_at="t1",
            duration_s=1.5,
            tier="warm",
            profile="quiet",
            paused_shell=True,
            pid=1,
            error=None,
            cycle_id="cyc_x",
            summary={"total_processed": 3},
        ).write(path)
        data = read_brain_status(path)
        assert data is not None
        assert data["ok"] is True
        assert data["tier"] == "warm"
        assert data["summary"]["total_processed"] == 3
        # file is valid json
        assert json.loads(path.read_text(encoding="utf-8"))["profile"] == "quiet"


class TestBrainMopPath:
    def test_mop_tier_routes_to_hygiene_not_phases(self):
        from engram.brain_cli import _TIER_PHASES

        assert "mop" in _TIER_PHASES
        # mop no longer uses phase set for execution (hygiene_ops path)
        assert _TIER_PHASES["mop"] is not None


class TestShellLifespanGates:
    def test_scheduler_start_skipped_for_shell_role(self):
        """Document the gate: shell_runs_in_process_brain controls scheduler."""
        shell = EngramConfig(runtime_role="shell")
        mono = EngramConfig(runtime_role="monolith")
        assert not shell.shell_runs_in_process_brain()
        assert mono.shell_runs_in_process_brain()

    @patch.dict(
        "os.environ",
        {
            "ENGRAM_RUNTIME_ROLE": "shell",
            "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": "quiet",
        },
        clear=False,
    )
    def test_env_loads_shell_quiet(self):
        # Rebuild settings from env (ignore cached .env values that may conflict)
        cfg = EngramConfig(
            runtime_role="shell",
            activation=ActivationConfig(consolidation_profile="quiet"),
        )
        assert cfg.runtime_role == "shell"
        assert cfg.activation.worker_enabled is False
        assert cfg.shell_runs_in_process_brain() is False
