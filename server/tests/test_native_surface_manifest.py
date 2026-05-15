from __future__ import annotations

import ast
from pathlib import Path

from fastapi.routing import APIRoute
from starlette.routing import Mount, WebSocketRoute

from engram.config import EngramConfig
from engram.main import create_app
from engram.quality.native_surface_manifest import (
    NATIVE_SURFACE_MANIFEST,
    identifiers_by_kind,
)

ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER = ROOT / "engram" / "mcp" / "server.py"
NATIVE_PARITY_TEST = ROOT / "tests" / "test_native_surface_parity.py"


def _public_rest_routes() -> set[str]:
    app = create_app(EngramConfig(_env_file=None))
    routes: set[str] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in route.methods or set():
            if method in {"HEAD", "OPTIONS"}:
                continue
            routes.add(f"{method} {route.path}")
    return routes


def _public_non_rest_transports() -> set[str]:
    app = create_app(EngramConfig(_env_file=None))
    transports: set[str] = set()
    for route in app.routes:
        if isinstance(route, WebSocketRoute):
            transports.add(route.path)
        elif isinstance(route, Mount):
            transports.add(route.path)
    return transports


def _decorated_mcp_surfaces() -> dict[str, set[str]]:
    tree = ast.parse(MCP_SERVER.read_text())
    surfaces = {
        "mcp_tool": set(),
        "mcp_resource": set(),
        "mcp_prompt": set(),
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            func = decorator.func
            if (
                not isinstance(func, ast.Attribute)
                or not isinstance(func.value, ast.Name)
                or func.value.id != "mcp"
            ):
                continue
            if func.attr == "tool":
                surfaces["mcp_tool"].add(node.name)
            elif func.attr == "prompt":
                surfaces["mcp_prompt"].add(node.name)
            elif func.attr == "resource":
                first_arg = decorator.args[0] if decorator.args else None
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    surfaces["mcp_resource"].add(first_arg.value)
    return surfaces


def test_native_surface_manifest_classifies_every_rest_route() -> None:
    assert _public_rest_routes() == identifiers_by_kind("rest")


def test_native_surface_manifest_classifies_non_rest_transports() -> None:
    expected = identifiers_by_kind("websocket") | identifiers_by_kind("mcp_transport")
    assert _public_non_rest_transports() == expected


def test_native_surface_manifest_classifies_mcp_surfaces() -> None:
    decorated = _decorated_mcp_surfaces()
    assert decorated["mcp_tool"] == identifiers_by_kind("mcp_tool")
    assert decorated["mcp_resource"] == identifiers_by_kind("mcp_resource")
    assert decorated["mcp_prompt"] == identifiers_by_kind("mcp_prompt")


def test_native_runtime_evidence_points_to_existing_parity_helpers() -> None:
    parity_source = NATIVE_PARITY_TEST.read_text()
    missing: list[str] = []
    for surface in NATIVE_SURFACE_MANIFEST:
        if surface.coverage != "native_runtime_parity":
            continue
        if surface.evidence not in parity_source:
            missing.append(f"{surface.identifier}: {surface.evidence}")
    assert missing == []
