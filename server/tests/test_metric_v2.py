"""M4 honest metric v2: aged-organic Decision selection, precision@5 scrap,
shell availability reconstruction, brain anomaly detection."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from engram.evaluation.continuity import (
    count_decision_scrap_top5,
    select_aged_organic_decision,
)
from engram.ops_metrics import brain_status_anomalies, compute_shell_availability

NOW = datetime(2026, 7, 16, 12, 0, 0, tzinfo=timezone.utc)


def _decision(name: str, days_old: float, entity_type: str = "Decision") -> dict:
    return {
        "name": name,
        "entity_type": entity_type,
        "created_at": (NOW - timedelta(days=days_old)).isoformat(),
    }


class TestAgedOrganicSelection:
    def test_picks_newest_aged_decision(self):
        entities = [
            _decision("Too fresh decision", 2),
            _decision("Old strategy decision", 30),
            _decision("Recent enough decision", 10),
        ]
        got = select_aged_organic_decision(entities, min_age_days=7.0, now=NOW)
        assert got is not None and got["name"] == "Recent enough decision"

    def test_excludes_synthetic_gate_name(self):
        entities = [_decision("Cold Decision hit requires healthy search index", 30)]
        got = select_aged_organic_decision(
            entities,
            min_age_days=7.0,
            exclude_names={"Cold Decision hit requires healthy search index"},
            now=NOW,
        )
        assert got is None

    def test_excludes_non_decisions_and_missing_created_at(self):
        entities = [
            _decision("A person", 30, entity_type="Person"),
            {"name": "No timestamp decision", "entity_type": "Decision"},
        ]
        assert select_aged_organic_decision(entities, min_age_days=7.0, now=NOW) is None

    def test_no_self_satisfaction_when_all_fresh(self):
        """The exact v1 failure mode: only just-written Decisions exist."""
        entities = [_decision("Written moments ago by the gate", 0.001)]
        assert select_aged_organic_decision(entities, min_age_days=7.0, now=NOW) is None


class TestDecisionScrapTop5:
    def test_counts_noise_names_only_in_top5(self):
        from engram.extraction.promotion import is_decision_statement_noise

        # Find a name the shared noise filter actually flags, so the test
        # tracks the real filter rather than a parallel definition.
        noise_candidates = [
            "We decided to do the thing today",
            "Decision: let's sync tomorrow",
            "decided to keep going with it",
        ]
        noise = next((n for n in noise_candidates if is_decision_statement_noise(n)), None)
        payload = {
            "items": [
                {"entity": {"name": "Helix native is the preferred backend"}},
                *([{"entity": {"name": noise}}] if noise else []),
            ]
        }
        expected = 1 if noise else 0
        assert count_decision_scrap_top5(payload) == expected

    def test_empty_payload(self):
        assert count_decision_scrap_top5({}) == 0


SAMPLE_LOG = "\n".join(
    [
        "2026-07-16 08:51:07,743 [INFO] engram.main: Engram v0.1.0 started in helix mode",
        "2026-07-16 10:51:11,109 [INFO] mcp.server.streamable_http_manager: "
        "StreamableHTTP session manager shutting down",
        "2026-07-16 10:58:40,575 [INFO] engram.main: Engram v0.1.0 started in helix mode",
    ]
)


class TestShellAvailability:
    def test_reconstructs_outage_windows(self):
        report = compute_shell_availability(
            SAMPLE_LOG,
            window_hours=24.0,
            now=datetime(2026, 7, 16, 12, 0, 0),
        )
        assert report.outage_count == 1
        # 10:51:11 -> 10:58:40 = 449s
        assert 440 <= report.max_outage_seconds <= 460
        assert report.open_outage is False
        assert report.availability_pct is not None
        assert report.availability_pct > 99.0

    def test_open_outage_counts_to_now(self):
        log = (
            "2026-07-16 10:51:11,109 [INFO] mcp.server.streamable_http_manager: "
            "StreamableHTTP session manager shutting down\n"
        )
        report = compute_shell_availability(
            log,
            window_hours=24.0,
            now=datetime(2026, 7, 16, 11, 51, 11),
        )
        assert report.open_outage is True
        assert 3590 <= report.max_outage_seconds <= 3610

    def test_no_events_reports_unknown(self):
        report = compute_shell_availability("", window_hours=24.0, now=NOW.replace(tzinfo=None))
        assert report.availability_pct is None
        assert report.outage_count == 0


class TestBrainAnomalies:
    def test_missing_status(self):
        assert "no brain runs recorded" in brain_status_anomalies(None)[0]

    def test_sleep_and_failure_flagged(self):
        status = {
            "ok": False,
            "error": "boom",
            "system_slept": True,
            "duration_s": 38690.0,
            "duration_monotonic_s": 620.0,
            "finished_at": NOW.isoformat(),
        }
        anomalies = brain_status_anomalies(status, now=NOW)
        joined = " | ".join(anomalies)
        assert "failed" in joined
        assert "slept" in joined

    def test_stale_launchagent_flagged(self):
        status = {
            "ok": True,
            "finished_at": (NOW - timedelta(hours=40)).isoformat(),
            "duration_s": 60.0,
        }
        anomalies = brain_status_anomalies(status, now=NOW)
        assert any("may not be firing" in a for a in anomalies)

    def test_healthy_recent_run_clean(self):
        status = {
            "ok": True,
            "finished_at": (NOW - timedelta(hours=1)).isoformat(),
            "duration_s": 120.0,
            "duration_monotonic_s": 119.0,
            "system_slept": False,
        }
        assert brain_status_anomalies(status, now=NOW) == []
