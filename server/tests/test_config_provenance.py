"""Ticket 24 / AUDIT-6: the CLI and the running service must not disagree silently.

The bug: `~/Library/LaunchAgents/dev.engram.local.plist` runs
``set -a; source ~/.engram/.env; set +a`` before ``python -m engram serve``, so every
key in that file becomes a real process environment variable — and process env outranks
every dotenv file in pydantic-settings. A CLI run exports nothing, so the dotenv chain
applies, and `DEFAULT_ENV_FILES` puts the cwd-relative ``.env`` LAST, i.e. highest file
precedence. Same machine, same venv, three answers.

Live values captured on the dogfood machine 2026-07-24 and used as fixture seeds
(STANDING_GOAL §2.7 — fixtures come from live data, not imagination):

    ~/.engram/.env      CONSOLIDATION_PROFILE=quiet     RECALL_PROFILE=wave2
    <repo>/.env         CONSOLIDATION_PROFILE=standard  WORKER_ENABLED=true
    <repo>/server/.env  CONSOLIDATION_PROFILE=standard  RECALL_PROFILE=all
    live service        consolidationProfile=quiet      recallProfile=wave2

These tests do three jobs:

1. Prove the divergence is real and reproducible (`test_launcher_and_cli_diverge_*`).
2. Prove `resolve_provenance` models pydantic-settings' *actual* precedence — if the
   model drifts from the resolver, the instrument becomes another liar
   (`test_provenance_matches_pydantic_settings_*`).
3. Prove the detector cannot go quietly inert: it must fire on divergence, stay silent
   on agreement, and report UNKNOWN rather than PASS when the shell is unreachable.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from engram.config import EngramConfig
from engram.config_provenance import (
    PROCESS_ENV,
    Resolution,
    compare_to_runtime,
    contested_keys,
    format_contest_banner,
    local_effective_values,
    resolve_provenance,
    warn_if_config_resolution_is_ambiguous,
)

CONSOLIDATION = "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE"
RECALL = "ENGRAM_ACTIVATION__RECALL_PROFILE"

# Verbatim shapes from the three live .env files (values only; secrets omitted).
LIVE_GLOBAL_ENV = "\n".join(
    [
        "ENGRAM_MODE=helix",
        f"{CONSOLIDATION}=quiet",
        f"{RECALL}=wave2",
        "ENGRAM_ACTIVATION__INTEGRATION_PROFILE=off",
        "ENGRAM_ACTIVATION__WORKER_ENABLED=false",
        "ENGRAM_RUNTIME_ROLE=shell",
    ]
)
LIVE_REPO_ENV = "\n".join(
    [
        "ENGRAM_MODE=helix",
        f"{CONSOLIDATION}=standard",
        "ENGRAM_ACTIVATION__WORKER_ENABLED=true",
    ]
)
LIVE_SERVER_ENV = "\n".join(
    [
        "ENGRAM_MODE=helix",
        f"{CONSOLIDATION}=standard",
        f"{RECALL}=all",
    ]
)

# What GET /api/knowledge/runtime/fast returned from the live shell, 2026-07-24.
LIVE_RUNTIME_PAYLOAD = {
    "runtime": {"mode": "helix", "surface": "fast_packet"},
    "activation": {
        "consolidationProfile": "quiet",
        "recallProfile": "wave2",
        "integrationProfile": "off",
    },
}


@pytest.fixture
def live_env_chain(tmp_path: Path) -> tuple[str, ...]:
    """The real three-file chain, in DEFAULT_ENV_FILES order (lowest first)."""
    global_env = tmp_path / "home" / ".engram" / ".env"
    repo_env = tmp_path / "repo" / ".env"
    server_env = tmp_path / "repo" / "server" / ".env"
    for path, body in (
        (global_env, LIVE_GLOBAL_ENV),
        (repo_env, LIVE_REPO_ENV),
        (server_env, LIVE_SERVER_ENV),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body + "\n", encoding="utf-8")
    return (str(global_env), str(repo_env), str(server_env))


def _launcher_environ(env_files: tuple[str, ...]) -> dict[str, str]:
    """What `set -a; source ~/.engram/.env; set +a` leaves in the process env."""
    from dotenv import dotenv_values

    return {k: v for k, v in dotenv_values(env_files[0]).items() if v is not None}


# --------------------------------------------------------------------------------
# 1. The divergence is real
# --------------------------------------------------------------------------------


def test_launcher_and_cli_diverge_on_the_live_env_chain(live_env_chain):
    """The launcher and a CLI run resolve the SAME key to DIFFERENT values."""
    cli = resolve_provenance(env_files=live_env_chain, environ={})
    launcher = resolve_provenance(
        env_files=live_env_chain, environ=_launcher_environ(live_env_chain)
    )

    assert cli[CONSOLIDATION].value == "standard"
    assert cli[CONSOLIDATION].source.endswith("server/.env")
    assert launcher[CONSOLIDATION].value == "quiet"
    assert launcher[CONSOLIDATION].source == PROCESS_ENV
    assert cli[CONSOLIDATION].value != launcher[CONSOLIDATION].value

    assert cli[RECALL].value == "all"
    assert launcher[RECALL].value == "wave2"


def test_contest_is_detectable_offline_with_no_server(live_env_chain):
    """No network needed: two files disagreeing already means launcher-dependence."""
    contested = contested_keys(resolve_provenance(env_files=live_env_chain, environ={}))
    assert CONSOLIDATION in contested
    assert RECALL in contested
    assert "ENGRAM_ACTIVATION__WORKER_ENABLED" in contested

    banner = format_contest_banner(resolve_provenance(env_files=live_env_chain, environ={}))
    assert "ENGRAM CONFIG IS AMBIGUOUS" in banner
    assert "shadowed: quiet" in banner


def test_unambiguous_chain_produces_no_banner(tmp_path: Path):
    """The banner must stay silent when the chain agrees — otherwise it is noise."""
    only = tmp_path / ".env"
    only.write_text(f"{CONSOLIDATION}=quiet\n", encoding="utf-8")
    resolutions = resolve_provenance(env_files=(str(only),), environ={})
    assert resolutions[CONSOLIDATION].value == "quiet"
    assert resolutions[CONSOLIDATION].contested is False
    assert format_contest_banner(resolutions) == ""


def test_agreeing_files_are_not_contested(tmp_path: Path):
    """Two files setting the SAME value is not launcher-dependence."""
    low = tmp_path / "low.env"
    high = tmp_path / "high.env"
    low.write_text(f"{CONSOLIDATION}=quiet\n", encoding="utf-8")
    high.write_text(f"{CONSOLIDATION}=quiet\n", encoding="utf-8")
    resolutions = resolve_provenance(env_files=(str(low), str(high)), environ={})
    assert resolutions[CONSOLIDATION].contested is False


def test_missing_key_resolves_to_default_not_a_guess(tmp_path: Path):
    empty = tmp_path / ".env"
    empty.write_text("# nothing\n", encoding="utf-8")
    resolutions = resolve_provenance(env_files=(str(empty),), environ={})
    assert resolutions[CONSOLIDATION].value is None
    assert resolutions[CONSOLIDATION].source == "default"


# --------------------------------------------------------------------------------
# 2. The provenance model must match the real resolver
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("file_count", "with_process_env"),
    [(1, False), (2, False), (3, False), (1, True), (3, True)],
)
def test_provenance_matches_pydantic_settings_resolution(
    live_env_chain, monkeypatch, file_count: int, with_process_env: bool
):
    """`resolve_provenance` must predict what pydantic-settings actually picks.

    This is the anti-drift guard. If the replay ever disagrees with the resolver, the
    doctor's provenance report becomes a plausible-but-wrong metric — exactly the class
    of failure this whole ledger exists to stop.
    """
    for key in (CONSOLIDATION, RECALL, "ENGRAM_MODE"):
        monkeypatch.delenv(key, raising=False)

    files = live_env_chain[:file_count]
    environ: dict[str, str] = {}
    if with_process_env:
        environ = _launcher_environ(live_env_chain)
        for key, value in environ.items():
            monkeypatch.setenv(key, value)

    predicted = resolve_provenance(env_files=files, environ=environ or {})[CONSOLIDATION]
    actual = EngramConfig(_env_file=files).activation.consolidation_profile

    assert predicted.value == actual, (
        f"provenance replay says {predicted.value} (from {predicted.source}) "
        f"but pydantic-settings resolved {actual}"
    )


def test_process_env_outranks_every_dotenv_file(live_env_chain, monkeypatch):
    """The exact mechanism the LaunchAgent exploits, asserted against the real loader."""
    monkeypatch.setenv(CONSOLIDATION, "observe")
    resolved = EngramConfig(_env_file=live_env_chain).activation.consolidation_profile
    assert resolved == "observe"
    predicted = resolve_provenance(env_files=live_env_chain, environ={CONSOLIDATION: "observe"})
    assert predicted[CONSOLIDATION].source == PROCESS_ENV


def test_last_env_file_wins_among_files(live_env_chain, monkeypatch):
    monkeypatch.delenv(RECALL, raising=False)
    resolved = EngramConfig(_env_file=live_env_chain).activation.recall_profile
    assert resolved == "all"  # server/.env, the last entry
    predicted = resolve_provenance(env_files=live_env_chain, environ={})
    assert predicted[RECALL].source.endswith("server/.env")


# --------------------------------------------------------------------------------
# 3. The detector must fire, must stay quiet, and must never fake a PASS
# --------------------------------------------------------------------------------


def _check_config_resolution(*args, **kwargs):
    """Import `engram.doctor` lazily.

    Deliberately NOT `importorskip`: if the doctor cannot be imported these tests must
    fail loudly, not vanish. A skipped divergence check is the same disease as a
    divergence that goes undetected.
    """
    from engram.doctor import _check_config_resolution as impl

    return impl(*args, **kwargs)


def _doctor_args(**overrides) -> argparse.Namespace:
    base = {
        "mode": None,
        "server_url": "http://127.0.0.1:8100",
        "timeout": 1.0,
        "skip_server": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _config_with(consolidation: str, recall: str) -> EngramConfig:
    return EngramConfig(
        mode="helix",
        activation={"consolidation_profile": consolidation, "recall_profile": recall},
    )


def _run_doctor_check(monkeypatch, config, payload, url="http://x/runtime/fast"):
    monkeypatch.setattr(
        "engram.doctor._fetch_runtime_payload",
        lambda _args: (payload, url),
    )
    checks: list[dict] = []
    _check_config_resolution(config, _doctor_args(), checks)
    return checks[0]


def test_doctor_fails_when_cli_and_service_disagree(monkeypatch):
    """The live shape: CLI resolved `standard`/`all`, service reports `quiet`/`wave2`."""
    check = _run_doctor_check(monkeypatch, _config_with("standard", "all"), LIVE_RUNTIME_PAYLOAD)
    assert check["name"] == "config_resolution"
    assert check["status"] == "fail"
    assert "CONFIG DIVERGENCE" in check["detail"]
    keys = {d["key"] for d in check["metadata"]["divergences"]}
    assert keys == {CONSOLIDATION, RECALL}


def test_doctor_passes_when_cli_and_service_agree(monkeypatch):
    check = _run_doctor_check(monkeypatch, _config_with("quiet", "wave2"), LIVE_RUNTIME_PAYLOAD)
    assert check["status"] == "pass"
    assert check["metadata"]["divergences"] == []
    # Absence is reported, not rounded down to agreement.
    assert check["metadata"]["unverifiable_keys"]


def test_doctor_reports_unknown_not_pass_when_shell_is_unreachable(monkeypatch):
    monkeypatch.setattr(
        "engram.doctor._fetch_runtime_payload",
        lambda _args: (None, "connection refused"),
    )
    checks: list[dict] = []
    _check_config_resolution(_config_with("standard", "all"), _doctor_args(), checks)
    assert checks[0]["status"] == "warn"
    assert "UNKNOWN" in checks[0]["detail"]
    assert checks[0]["status"] != "pass"


def test_doctor_does_not_silently_skip_when_server_checks_are_skipped():
    checks: list[dict] = []
    _check_config_resolution(
        _config_with("standard", "all"), _doctor_args(skip_server=True), checks
    )
    assert checks[0]["status"] == "skipped"
    assert "NOT verified" in checks[0]["detail"]


def test_mode_override_is_reported_unverifiable_not_compared(monkeypatch):
    """`--mode` intentionally overrides config, so comparing it would be a false alarm."""
    monkeypatch.setattr(
        "engram.doctor._fetch_runtime_payload",
        lambda _args: ({"runtime": {"mode": "lite"}, "activation": {}}, "http://x"),
    )
    checks: list[dict] = []
    _check_config_resolution(_config_with("quiet", "wave2"), _doctor_args(mode="helix"), checks)
    assert checks[0]["status"] == "pass"
    assert "ENGRAM_MODE" in checks[0]["metadata"]["unverifiable_keys"]


# --------------------------------------------------------------------------------
# Canary — the detector must be capable of both answers on the same input shape
# --------------------------------------------------------------------------------


def test_comparator_is_not_inert():
    """A comparator that always returns [] would make every check above pass.

    Assert both directions on the same call shape, so a neutered comparator cannot go
    green: divergent input must yield exactly one divergence naming the winning source,
    and identical input must yield none.
    """
    resolutions = {
        CONSOLIDATION: Resolution(key=CONSOLIDATION, value="standard", source="/repo/server/.env")
    }
    divergent, _ = compare_to_runtime(
        local_values={CONSOLIDATION: "standard"},
        runtime_payload=LIVE_RUNTIME_PAYLOAD,
        resolutions=resolutions,
    )
    assert [d.key for d in divergent] == [CONSOLIDATION]
    assert divergent[0].local_value == "standard"
    assert divergent[0].runtime_value == "quiet"
    assert divergent[0].local_source == "/repo/server/.env"

    agreeing, _ = compare_to_runtime(
        local_values={CONSOLIDATION: "quiet"},
        runtime_payload=LIVE_RUNTIME_PAYLOAD,
        resolutions=resolutions,
    )
    assert agreeing == []


def test_absent_runtime_field_is_unverifiable_never_agreement():
    """A key the payload omits must not be counted as a match."""
    divergences, unverifiable = compare_to_runtime(
        local_values={CONSOLIDATION: "standard"},
        runtime_payload={"runtime": {}, "activation": {}},
    )
    assert divergences == []
    assert CONSOLIDATION in unverifiable


def test_local_effective_values_reads_the_constructed_config():
    values = local_effective_values(_config_with("observe", "wave1"))
    assert values[CONSOLIDATION] == "observe"
    assert values[RECALL] == "wave1"


def _arm_banner(monkeypatch, env_files):
    import engram.config_provenance as provenance

    monkeypatch.setattr(provenance, "_AMBIGUITY_BANNER_EMITTED", False)
    monkeypatch.setattr(
        provenance,
        "resolve_provenance",
        lambda **_kwargs: resolve_provenance(env_files=env_files, environ={}),
    )


def test_banner_emits_once_per_process(live_env_chain, monkeypatch):
    """Loud, but not once per EngramConfig() — that would train people to ignore it."""
    _arm_banner(monkeypatch, live_env_chain)
    first = warn_if_config_resolution_is_ambiguous()
    second = warn_if_config_resolution_is_ambiguous()
    assert "ENGRAM CONFIG IS AMBIGUOUS" in first
    assert second == ""


def test_banner_actually_writes_to_the_stream(live_env_chain, monkeypatch):
    """The print side channel must work, not just the return value."""
    _arm_banner(monkeypatch, live_env_chain)
    sink = io.StringIO()
    warn_if_config_resolution_is_ambiguous(stream=sink)
    written = sink.getvalue()
    assert "ENGRAM CONFIG IS AMBIGUOUS" in written
    assert "shadowed: quiet" in written


def test_banner_is_silent_on_an_unambiguous_chain(tmp_path: Path, monkeypatch):
    only = tmp_path / ".env"
    only.write_text(f"{CONSOLIDATION}=quiet\n", encoding="utf-8")
    _arm_banner(monkeypatch, (str(only),))
    sink = io.StringIO()
    assert warn_if_config_resolution_is_ambiguous(stream=sink) == ""
    assert sink.getvalue() == ""
