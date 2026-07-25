"""Which config is actually live, and which file or env var decided it.

Ticket 24 / `INSTRUMENT_AUDIT.md` AUDIT-6. The LaunchAgent runs
``set -a; source ~/.engram/.env; set +a`` before ``python -m engram serve``, so every
key in that file becomes a **real process environment variable** — and in
pydantic-settings process env outranks every dotenv file. A CLI run exports nothing, so
the dotenv chain applies instead, and ``DEFAULT_ENV_FILES`` puts the *cwd-relative*
``.env`` last, i.e. highest file precedence.

Measured on the dogfood machine, same venv, same day::

    cwd=server/        consolidation=standard  recall=all      (server/.env wins)
    cwd=repo root      consolidation=standard  recall=wave2    (repo .env wins)
    launchd service    consolidation=quiet     recall=wave2    (process env wins)

Three answers. Every CLI-run measurement read a different machine than the service
under test, and ``engram doctor`` could not detect it because it resolved config the
same ambient way.

This module makes the resolution **explicit and reportable**:

* :func:`resolve_provenance` replays the precedence chain and records which source won
  each key and which sources were shadowed.
* :func:`contested_keys` flags keys whose answer depends on where you launched from —
  detectable offline, with no server.
* :func:`compare_to_runtime` checks a local resolution against the live shell's
  ``GET /api/knowledge/runtime`` payload and reports divergence per key, plus the keys
  it could **not** verify (the payload does not carry every setting; saying so is the
  point — see `INSTRUMENT_AUDIT.md` "accurate or absent").
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROCESS_ENV = "process-env"
DEFAULT_SOURCE = "default"

# Keys the live runtime packet exposes, so divergence is checkable rather than assumed.
# `local_attr` is the dotted path on EngramConfig; `payload_path` is the dotted path in
# the GET /api/knowledge/runtime[/fast] JSON.
TRACKED_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "ENGRAM_MODE": {
        "local_attr": ("mode",),
        "payload_path": ("runtime", "mode"),
    },
    "ENGRAM_ACTIVATION__CONSOLIDATION_PROFILE": {
        "local_attr": ("activation", "consolidation_profile"),
        "payload_path": ("activation", "consolidationProfile"),
    },
    "ENGRAM_ACTIVATION__RECALL_PROFILE": {
        "local_attr": ("activation", "recall_profile"),
        "payload_path": ("activation", "recallProfile"),
    },
    "ENGRAM_ACTIVATION__INTEGRATION_PROFILE": {
        "local_attr": ("activation", "integration_profile"),
        "payload_path": ("activation", "integrationProfile"),
    },
}

# Set by a dotenv file or the launcher but NOT carried on the runtime packet, so a
# divergence in these cannot be detected by comparison. Reported as unverifiable rather
# than silently omitted.
UNVERIFIABLE_KEYS: tuple[str, ...] = (
    "ENGRAM_ACTIVATION__WORKER_ENABLED",
    "ENGRAM_RUNTIME_ROLE",
    "ENGRAM_HELIX__TRANSPORT",
    "ENGRAM_HELIX__DATA_DIR",
)


@dataclass(frozen=True)
class Resolution:
    """Where one env key's value came from, and what it beat."""

    key: str
    value: str | None
    source: str
    # Losing (source, value) pairs, highest precedence first.
    shadowed: tuple[tuple[str, str], ...] = ()

    @property
    def contested(self) -> bool:
        """True when a shadowed source disagrees with the winner.

        A contested key is launcher-dependent by construction: whichever source the
        launcher happens to promote changes the answer.
        """
        return any(value != self.value for _source, value in self.shadowed)


@dataclass(frozen=True)
class Divergence:
    """One key where this process and the running shell disagree."""

    key: str
    local_value: str
    local_source: str
    runtime_value: str


def _env_file_sources(env_files: Iterable[str]) -> list[tuple[str, dict[str, str]]]:
    """Read the dotenv chain lowest-precedence first, labelled by absolute path."""
    from dotenv import dotenv_values

    sources: list[tuple[str, dict[str, str]]] = []
    for entry in env_files:
        path = Path(entry).expanduser()
        # A cwd-relative entry is exactly the ambient dependence this module exists to
        # expose, so resolve it and show the reader the file it actually landed on.
        label = str(path if path.is_absolute() else (Path.cwd() / path).resolve())
        if not Path(label).is_file():
            continue
        values = {
            str(key).upper(): value
            for key, value in dotenv_values(label).items()
            if value is not None
        }
        sources.append((label, values))
    return sources


def resolve_provenance(
    *,
    env_files: Iterable[str] | None = None,
    environ: Mapping[str, str] | None = None,
    keys: Iterable[str] | None = None,
) -> dict[str, Resolution]:
    """Replay pydantic-settings precedence and record who won each key.

    Precedence, lowest to highest: dotenv files in ``env_files`` order, then real
    process environment variables.
    """
    if env_files is None:
        from engram.config import DEFAULT_ENV_FILES

        env_files = DEFAULT_ENV_FILES
    if environ is None:
        environ = os.environ
    tracked = tuple(keys) if keys is not None else (*TRACKED_KEYS, *UNVERIFIABLE_KEYS)

    layers = _env_file_sources(env_files)
    layers.append((PROCESS_ENV, {str(k).upper(): v for k, v in environ.items()}))

    resolved: dict[str, Resolution] = {}
    for key in tracked:
        upper = key.upper()
        hits = [(source, values[upper]) for source, values in layers if upper in values]
        if not hits:
            resolved[key] = Resolution(key=key, value=None, source=DEFAULT_SOURCE)
            continue
        winner_source, winner_value = hits[-1]
        losers = tuple(reversed(hits[:-1]))
        resolved[key] = Resolution(
            key=key,
            value=winner_value,
            source=winner_source,
            shadowed=losers,
        )
    return resolved


def contested_keys(resolutions: Mapping[str, Resolution]) -> dict[str, Resolution]:
    """Keys whose effective value depends on which source the launcher promotes."""
    return {key: res for key, res in resolutions.items() if res.contested}


def _dig(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = payload
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    return node


def local_effective_values(config: Any) -> dict[str, str]:
    """Read the tracked settings off a constructed config object."""
    values: dict[str, str] = {}
    for key, spec in TRACKED_KEYS.items():
        node: Any = config
        for part in spec["local_attr"]:
            node = getattr(node, part, None)
            if node is None:
                break
        if node is not None:
            values[key] = str(node)
    return values


def compare_to_runtime(
    *,
    local_values: Mapping[str, str],
    runtime_payload: Mapping[str, Any],
    resolutions: Mapping[str, Resolution] | None = None,
    skip_keys: Iterable[str] = (),
) -> tuple[list[Divergence], list[str]]:
    """Compare this process's effective config against the live shell's.

    Returns ``(divergences, unverifiable)``. ``unverifiable`` names every tracked key
    the runtime payload did not carry plus the keys explicitly skipped — absence is
    reported, never rounded down to agreement.
    """
    skipped = {key.upper() for key in skip_keys}
    divergences: list[Divergence] = []
    unverifiable: list[str] = []
    for key in TRACKED_KEYS:
        if key.upper() in skipped:
            unverifiable.append(key)
            continue
        local = local_values.get(key)
        remote = _dig(runtime_payload, TRACKED_KEYS[key]["payload_path"])
        if local is None or remote is None:
            unverifiable.append(key)
            continue
        if str(local) != str(remote):
            source = DEFAULT_SOURCE
            if resolutions is not None and key in resolutions:
                source = resolutions[key].source
            divergences.append(
                Divergence(
                    key=key,
                    local_value=str(local),
                    local_source=source,
                    runtime_value=str(remote),
                )
            )
    unverifiable.extend(UNVERIFIABLE_KEYS)
    return divergences, unverifiable


def format_contest_banner(resolutions: Mapping[str, Resolution]) -> str:
    """Render the offline warning for launcher-dependent keys. Empty when clean."""
    contested = contested_keys(resolutions)
    if not contested:
        return ""
    lines = [
        "!! ENGRAM CONFIG IS AMBIGUOUS — the effective value depends on how you launched.",
    ]
    for key in sorted(contested):
        res = contested[key]
        lines.append(f"   {key} = {res.value}   <- {res.source}")
        for source, value in res.shadowed:
            if value != res.value:
                lines.append(f"       shadowed: {value} <- {source}")
    lines.append(
        "   The launchd service exports ~/.engram/.env as real process env vars, which outrank"
    )
    lines.append(
        "   every dotenv file, so the running shell may be on the shadowed values. Verify with:"
    )
    lines.append("       engram doctor      # compares this process to the live shell")
    return "\n".join(lines)


_AMBIGUITY_BANNER_EMITTED = False


def warn_if_config_resolution_is_ambiguous(*, stream: Any | None = None) -> str:
    """Shout, once per process, when the effective config is launcher-dependent.

    Called from ``EngramConfig.model_post_init`` so that *every* entrypoint — CLI,
    ``python -m engram serve`` under launchd, and the ad-hoc ``python -c`` one-liners
    agents use to check config — reports the ambiguity instead of silently picking a
    winner. The condition needs no server, so it is detected even when the shell is
    down.

    The banner text is always computed and returned — the print is a side channel, so
    the mechanism stays fully testable. Under pytest the default print is suppressed
    (an explicit ``stream`` still writes): the developer's own repo `.env` legitimately
    overrides `~/.engram/.env`, and letting whichever test happens to build the first
    config emit to stderr makes any "stderr is empty" assertion order-dependent.
    """
    global _AMBIGUITY_BANNER_EMITTED
    if _AMBIGUITY_BANNER_EMITTED:
        return ""
    _AMBIGUITY_BANNER_EMITTED = True
    try:
        banner = format_contest_banner(resolve_provenance())
    except Exception:  # pragma: no cover - a diagnostic must never break startup
        return ""
    if banner and (stream is not None or "PYTEST_CURRENT_TEST" not in os.environ):
        print(banner, file=stream if stream is not None else sys.stderr)
    return banner


def reset_ambiguity_banner() -> None:
    """Re-arm the once-per-process banner. Test-only."""
    global _AMBIGUITY_BANNER_EMITTED
    _AMBIGUITY_BANNER_EMITTED = False
