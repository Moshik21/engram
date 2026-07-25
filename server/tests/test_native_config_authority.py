"""Contract: the declared HelixDB config must equal the effective one.

Ticket #27 (T0, measurement integrity). Three tracked copies of
``config.hx.json`` declared ``ef_search 512`` / ``ef_construction 200`` /
``db_max_size_gb 50`` / ``mcp false``. The value the engine actually runs on is
the generated Rust literal in ``fn config()`` at
``native/helix-repo/helix-python/src/queries.rs`` -- 768 / 128 / 20 / true --
which ``HelixEngine::new`` reads directly (``helix-python/src/lib.rs:53``).
``NativeTransport.initialize()`` passes no config path at all, so the JSON has
never had an effect on the native path, and the last ``helix push dev`` staged a
container config that does not match the tracked copies either.

Two concrete harms, both of which this test now makes impossible to repeat
silently:

1. **Reading.** An operator sizing growth off the JSON believed there were 50 GB
   of headroom against a real 20 GB -- 2.5x optimistic on the exact axis that
   causes ``MAP_FULL``. The live brain was measured at 84.8% of the *real* 20 GB
   before compaction.
2. **Editing.** Ticket #4 carries an "ef_search rebuild" item, and
   ``RECALL_PERFORMANCE_PLAN.md`` M1 aims an edit at ``config.hx.json:5``. That
   edit is a no-op. With this test, the same edit is a red build that names the
   file to change and the rebuild it needs.

Note that 768/128/16 are not choices -- they are the helix-db library defaults
(``helix-db/src/helix_engine/traversal_core/config.rs:17-21``). Nobody ever
picked them for this workload. Recording them here states what the machine does,
not what it should do; ``RECALL_PERFORMANCE_PLAN.md`` argues ef_search should be
48-128 for this N, and that change must land in ``queries.rs`` plus a
``make build-native``.

This test reads only git-tracked files. It never skips: every path it needs is
committed, so a missing one is a failure, not an excuse.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# The single source of truth at runtime.
GENERATED_QUERIES = REPO_ROOT / "native/helix-repo/helix-python/src/queries.rs"
# The struct definitions the generated literal is an instance of.
HELIX_CONFIG_RS = REPO_ROOT / "native/helix-repo/helix-db/src/helix_engine/traversal_core/config.rs"

# Every tracked copy of the declared config. A new copy must be added here
# deliberately -- test_no_untracked_config_copies fails when one appears.
DECLARED_CONFIGS = (
    REPO_ROOT / "helixdb-cfg/config.hx.json",
    REPO_ROOT / "helixdb-cfg/db/config.hx.json",
    REPO_ROOT / "server/engram/storage/helix/config.hx.json",
)

# Directories that hold build output or vendored trees; copies there are
# artifacts, not declarations.
_PRUNED_DIRS = {".git", ".helix", "target", "node_modules", ".venv", "__pycache__"}

# Keys the JSON is allowed to carry beyond the Rust struct fields.
_MARKER_KEY = "_authority"

# Config fields that are populated by codegen, not by this file.
_CODEGEN_OWNED = {"schema", "graphvis_node_label", "bm25_field_filters"}


def _read(path: Path) -> str:
    assert path.exists(), (
        f"{path} is git-tracked and must exist; a missing file here means this "
        "contract cannot check anything and must fail loudly rather than skip"
    )
    return path.read_text()


def parse_rust_config(text: str) -> dict[str, Any]:
    """Extract the scalar fields of the generated ``fn config()`` literal.

    The embedded schema blob is stripped first: it is ~3000 lines of JSON full
    of numbers and would poison every regex below.
    """
    start = text.find("pub fn config() -> Option<Config> {")
    assert start != -1, "generated queries.rs has no fn config() -- codegen shape changed"
    end = text.find("bm25_field_filters:", start)
    assert end != -1, "fn config() has no bm25_field_filters terminator"
    block = text[start:end]

    schema_start = block.find('schema: Some(r#"')
    if schema_start != -1:
        schema_end = block.find('"#.to_string()),', schema_start)
        assert schema_end != -1, "unterminated schema blob in fn config()"
        block = block[:schema_start] + block[schema_end:]

    parsed: dict[str, Any] = {}
    for field in ("m", "ef_construction", "ef_search", "db_max_size_gb"):
        match = re.search(rf"^{field}: Some\((\d+)\),$", block, re.M)
        assert match, f"fn config() does not set {field}"
        parsed[field] = int(match.group(1))
    for field in ("mcp", "bm25"):
        match = re.search(rf"^{field}: Some\((true|false)\),$", block, re.M)
        assert match, f"fn config() does not set {field}"
        parsed[field] = match.group(1) == "true"

    match = re.search(r'^embedding_model: (?:Some\("(.*?)"\.to_string\(\)\)|None),$', block, re.M)
    assert match, "fn config() does not set embedding_model"
    parsed["embedding_model"] = match.group(1)

    match = re.search(r"^secondary_indices: (None|Some\(vec!\[)", block, re.M)
    assert match, "fn config() does not set secondary_indices"
    parsed["secondary_indices_are_codegen_derived"] = match.group(1) != "None"
    return parsed


def parse_rust_struct_fields(text: str, struct_name: str) -> set[str]:
    match = re.search(
        rf"pub struct {struct_name} \{{(?P<body>.*?)\n\}}",
        text,
        re.S,
    )
    assert match, f"missing Rust struct {struct_name}"
    return {
        line.strip().removeprefix("pub ").split(":", 1)[0].strip()
        for line in match.group("body").splitlines()
        if line.strip().startswith("pub ")
    }


def _declared(path: Path) -> dict[str, Any]:
    return json.loads(_read(path))


def _effective() -> dict[str, Any]:
    return parse_rust_config(_read(GENERATED_QUERIES))


def test_declared_hnsw_and_map_size_match_the_effective_rust_literal() -> None:
    """The bug this file exists for: JSON said 512/200/50/false, Rust ran 768/128/20/true."""
    effective = _effective()
    for path in DECLARED_CONFIGS:
        declared = _declared(path)
        vector = declared["vector_config"]
        where = f"{path.relative_to(REPO_ROOT)} vs {GENERATED_QUERIES.relative_to(REPO_ROOT)}"
        assert vector["m"] == effective["m"], f"m diverged: {where}"
        assert vector["ef_construction"] == effective["ef_construction"], (
            f"ef_construction diverged: {where}"
        )
        assert vector["ef_search"] == effective["ef_search"], f"ef_search diverged: {where}"
        assert declared["db_max_size_gb"] == effective["db_max_size_gb"], (
            f"db_max_size_gb diverged: {where} -- an operator sizing growth off the "
            "JSON would be wrong by this ratio on the axis that causes MAP_FULL"
        )
        assert declared["mcp"] == effective["mcp"], f"mcp diverged: {where}"
        assert declared["bm25"] == effective["bm25"], f"bm25 diverged: {where}"
        assert declared["embedding_model"] == effective["embedding_model"], (
            f"embedding_model diverged: {where}"
        )


def test_declared_config_declares_no_key_the_runtime_cannot_honour() -> None:
    """A knob that deserialises into nothing is a knob that reads as real.

    ``vector_config.db_max_size: 50`` was exactly that: ``VectorConfig`` has no
    such field, so serde dropped it silently on every parse.
    """
    config_rs = _read(HELIX_CONFIG_RS)
    top_level = parse_rust_struct_fields(config_rs, "Config") | {_MARKER_KEY}
    vector_fields = parse_rust_struct_fields(config_rs, "VectorConfig")
    graph_fields = parse_rust_struct_fields(config_rs, "GraphConfig")

    assert "db_max_size" not in vector_fields, (
        "VectorConfig gained a db_max_size field; re-check the phantom-knob claim"
    )

    for path in DECLARED_CONFIGS:
        declared = _declared(path)
        rel = path.relative_to(REPO_ROOT)
        unknown = set(declared) - top_level
        assert not unknown, f"{rel} declares keys the Rust Config cannot honour: {sorted(unknown)}"
        codegen_owned = set(declared) & _CODEGEN_OWNED
        assert not codegen_owned, (
            f"{rel} declares codegen-owned keys: {sorted(codegen_owned)}; these are "
            "written by the Helix compiler, not read from this file"
        )
        unknown_vector = set(declared.get("vector_config", {})) - vector_fields
        assert not unknown_vector, (
            f"{rel} vector_config declares keys VectorConfig cannot honour: "
            f"{sorted(unknown_vector)}"
        )
        unknown_graph = set(declared.get("graph_config", {})) - graph_fields
        assert not unknown_graph, (
            f"{rel} graph_config declares keys GraphConfig cannot honour: {sorted(unknown_graph)}"
        )


def test_secondary_indices_cannot_be_declared_here() -> None:
    """Indices come from ``INDEX`` in schema.hx, never from this file.

    ``Config::fmt_with_schema`` takes ``secondary_indices`` as a parameter from
    schema introspection and ignores ``self.graph_config`` entirely, so a list
    typed here is discarded. Pin it empty so nobody types one.
    """
    effective = _effective()
    assert effective["secondary_indices_are_codegen_derived"], (
        "fn config() has no secondary indices; schema.hx INDEX declarations are "
        "not reaching codegen"
    )
    for path in DECLARED_CONFIGS:
        declared = _declared(path)
        assert declared["graph_config"]["secondary_indices"] == [], (
            f"{path.relative_to(REPO_ROOT)} declares secondary_indices; this file "
            "cannot express them -- add INDEX to schema.hx instead"
        )


def test_every_declared_copy_is_byte_identical() -> None:
    """Three copies that may differ are three chances to read the wrong one."""
    texts = {path: _read(path) for path in DECLARED_CONFIGS}
    first_path, first_text = next(iter(texts.items()))
    for path, text in texts.items():
        assert text == first_text, (
            f"{path.relative_to(REPO_ROOT)} differs from "
            f"{first_path.relative_to(REPO_ROOT)}; the copies must move together"
        )


def test_every_declared_copy_names_its_authority() -> None:
    """A reader opening any copy must learn, in the file, that it is not the authority."""
    for path in DECLARED_CONFIGS:
        declared = _declared(path)
        marker = declared.get(_MARKER_KEY)
        rel = path.relative_to(REPO_ROOT)
        assert isinstance(marker, str) and marker, f"{rel} has no {_MARKER_KEY} marker"
        assert "queries.rs" in marker, f"{rel} {_MARKER_KEY} does not name queries.rs"
        assert "make build-native" in marker, (
            f"{rel} {_MARKER_KEY} does not say a rebuild is required"
        )
        assert next(iter(declared)) == _MARKER_KEY, (
            f"{rel} {_MARKER_KEY} is not the first key; it must be unmissable"
        )


def test_no_untracked_config_copies() -> None:
    """A fourth copy must be a deliberate act, not a silent one."""
    found: list[Path] = []
    stack = [REPO_ROOT]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in _PRUNED_DIRS:
                    stack.append(entry)
            elif entry.name == "config.hx.json":
                found.append(entry)
    assert sorted(found) == sorted(DECLARED_CONFIGS), (
        "config.hx.json copies changed. Found "
        f"{sorted(str(p.relative_to(REPO_ROOT)) for p in found)}; expected "
        f"{sorted(str(p.relative_to(REPO_ROOT)) for p in DECLARED_CONFIGS)}. "
        "Every copy must be pinned to queries.rs by this contract."
    )


_SYNTHETIC_QUERIES_RS = (
    "pub fn config() -> Option<Config> {\n"
    "return Some(Config {\n"
    "vector_config: Some(VectorConfig {\n"
    "m: Some(7),\n"
    "ef_construction: Some(11),\n"
    "ef_search: Some(13),\n"
    "}),\n"
    "graph_config: Some(GraphConfig {\n"
    'secondary_indices: Some(vec![SecondaryIndex::Index("name".to_string())]),\n'
    "}),\n"
    "db_max_size_gb: Some(17),\n"
    "mcp: Some(false),\n"
    "bm25: Some(true),\n"
    'schema: Some(r#"{"ef_search": 999, "db_max_size_gb": 999}"#.to_string()),\n'
    'embedding_model: Some("synthetic-model".to_string()),\n'
    "graphvis_node_label: None,\n"
    "bm25_field_filters: None,\n"
    "})\n"
    "}\n"
)


def test_effective_values_are_read_from_the_rust_and_nowhere_else(monkeypatch) -> None:
    """Canary against the worst vacuous form: an instrument that measures itself.

    An ``_effective()`` that quietly sourced its numbers from ``config.hx.json``
    would make every assertion above a tautology and the whole file would go
    green while checking nothing -- and once the two files agree, no value
    comparison can detect it. So force the issue: serve a *synthetic* queries.rs
    and require ``_effective()`` to report its values, and make any read of the
    JSON an error.

    This canary was written after a weaker one passed on exactly this neuter.
    """
    seen: list[Path] = []

    def only_the_generated_rust(path: Path) -> str:
        seen.append(path)
        assert path == GENERATED_QUERIES, (
            f"_effective() read {path}; the effective config must come from "
            f"{GENERATED_QUERIES} alone"
        )
        return _SYNTHETIC_QUERIES_RS

    monkeypatch.setitem(globals(), "_read", only_the_generated_rust)
    effective = _effective()

    assert seen == [GENERATED_QUERIES], f"_effective() touched {seen}"
    assert effective["ef_search"] == 13
    assert effective["ef_construction"] == 11
    assert effective["m"] == 7
    assert effective["db_max_size_gb"] == 17
    assert effective["mcp"] is False
    assert effective["embedding_model"] == "synthetic-model"


def test_parsers_are_not_inert() -> None:
    """Canary. A contract test whose parser stops matching is the project's own
    dominant bug class wearing a green checkmark.

    Asserts the Rust parser reads real values off a synthetic literal, and that
    it detects a divergence rather than shrugging at one.
    """
    synthetic = (
        "pub fn config() -> Option<Config> {\n"
        "return Some(Config {\n"
        "vector_config: Some(VectorConfig {\n"
        "m: Some(7),\n"
        "ef_construction: Some(11),\n"
        "ef_search: Some(13),\n"
        "}),\n"
        "graph_config: Some(GraphConfig {\n"
        'secondary_indices: Some(vec![SecondaryIndex::Index("name".to_string())]),\n'
        "}),\n"
        "db_max_size_gb: Some(17),\n"
        "mcp: Some(false),\n"
        "bm25: Some(true),\n"
        'schema: Some(r#"{"ef_search": 999, "db_max_size_gb": 999}"#.to_string()),\n'
        'embedding_model: Some("synthetic-model".to_string()),\n'
        "graphvis_node_label: None,\n"
        "bm25_field_filters: None,\n"
        "})\n"
        "}\n"
    )
    parsed = parse_rust_config(synthetic)
    assert parsed == {
        "m": 7,
        "ef_construction": 11,
        "ef_search": 13,
        "db_max_size_gb": 17,
        "mcp": False,
        "bm25": True,
        "embedding_model": "synthetic-model",
        "secondary_indices_are_codegen_derived": True,
    }, "the Rust config parser is not reading the values it claims to read"

    # The schema blob carries decoys; if stripping ever breaks, these flip.
    assert parsed["ef_search"] != 999
    assert parsed["db_max_size_gb"] != 999

    # And it must disagree with a divergent declaration rather than pass anything.
    declared = _declared(DECLARED_CONFIGS[0])
    assert declared["vector_config"]["ef_search"] != parsed["ef_search"], (
        "the synthetic value collided with the real one; pick different decoys"
    )

    # The struct-field parser must return real field names, not an empty set
    # (an empty allowlist would make the unknown-key test vacuously pass).
    config_rs = _read(HELIX_CONFIG_RS)
    assert parse_rust_struct_fields(config_rs, "VectorConfig") == {
        "m",
        "ef_construction",
        "ef_search",
    }
    assert {"db_max_size_gb", "mcp", "bm25", "vector_config"} <= parse_rust_struct_fields(
        config_rs, "Config"
    )
