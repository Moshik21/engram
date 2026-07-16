"""Helix backend availability for tests and operator probes.

Product truth: **native PyO3 + data-dir** is first-class. HTTP :6969 is a
secondary compatibility path for Docker/HTTP Helix.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any


def helix_native_importable() -> bool:
    """Return True when the custom helix_native PyO3 package imports."""
    try:
        import helix_native  # noqa: F401

        return True
    except Exception:
        return False


def helix_http_available(
    host: str = "localhost",
    port: int = 6969,
    *,
    timeout: float = 2.0,
) -> bool:
    """Return True when a Helix HTTP listener accepts TCP connections."""
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except Exception:
        return False


def resolve_helix_data_dir(explicit: str | Path | None = None) -> Path:
    """Resolve native data directory from arg, env, or default."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("ENGRAM_HELIX__DATA_DIR") or os.environ.get("ENGRAM_HELIX_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".helix" / "engram-native").resolve()


def helix_native_available(
    data_dir: str | Path | None = None,
    *,
    require_writable: bool = True,
) -> bool:
    """True when native PyO3 is importable and data-dir is usable."""
    if not helix_native_importable():
        return False
    path = resolve_helix_data_dir(data_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    if require_writable and not os.access(path, os.W_OK):
        return False
    return True


def helix_available(
    *,
    prefer_native: bool = True,
    host: str = "localhost",
    port: int = 6969,
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Probe Helix backends; native is preferred product path.

    Returns a JSON-serializable dict for tests/doctor:
    ``{available, backend, native, http, data_dir, detail}``.
    """
    native = helix_native_available(data_dir)
    http = helix_http_available(host, port)
    path = str(resolve_helix_data_dir(data_dir))
    if prefer_native and native:
        return {
            "available": True,
            "backend": "native",
            "native": True,
            "http": http,
            "data_dir": path,
            "detail": "helix_native importable; data-dir writable",
        }
    if http:
        return {
            "available": True,
            "backend": "http",
            "native": native,
            "http": True,
            "data_dir": path,
            "detail": f"Helix HTTP listening on {host}:{port}",
        }
    if native:
        return {
            "available": True,
            "backend": "native",
            "native": True,
            "http": False,
            "data_dir": path,
            "detail": "helix_native importable; data-dir writable",
        }
    return {
        "available": False,
        "backend": None,
        "native": False,
        "http": False,
        "data_dir": path,
        "detail": "neither helix_native nor HTTP Helix available",
    }
