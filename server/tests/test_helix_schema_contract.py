from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_ROOT = Path(__file__).resolve().parents[1]
SERVER_SCHEMA = SERVER_ROOT / "engram/storage/helix/schema.hx"
NATIVE_SCHEMA = REPO_ROOT / "helixdb-cfg/db/schema.hx"
NATIVE_GENERATED_QUERIES = (
    REPO_ROOT
    / "helixdb-cfg/.helix/dev/helix-repo-copy/helix-container/src/queries.rs"
)

ENTITY_HX_FIELDS = {
    "name": "String",
    "group_id": "String",
    "entity_type": "String",
    "canonical_identifier": "String",
    "entity_id": "String",
    "summary": "String",
    "attributes_json": "String",
    "created_at": "String",
    "updated_at": "String",
    "is_deleted": "Boolean",
    "deleted_at": "String",
    "identity_core": "Boolean",
    "mat_tier": "String",
    "recon_count": "I32",
    "lexical_regime": "String",
    "identifier_label": "String",
    "pii_detected": "Boolean",
    "pii_categories_json": "String",
    "access_count": "I64",
    "last_accessed": "String",
    "source_episode_ids": "String",
    "evidence_count": "I64",
    "evidence_span_start": "String",
    "evidence_span_end": "String",
}
ENTITY_RUST_FIELDS = {
    field: {
        "Boolean": "bool",
        "I32": "i32",
        "I64": "i64",
    }.get(field_type, "String")
    for field, field_type in ENTITY_HX_FIELDS.items()
}
ENTITY_MUTABLE_FIELDS = [
    field
    for field in ENTITY_HX_FIELDS
    if field not in {"entity_id", "group_id", "created_at"}
]

EPISODE_CUE_HX_FIELDS = {
    "episode_id": "String",
    "group_id": "String",
    "cue_version": "I32",
    "discourse_class": "String",
    "cue_text": "String",
    "supporting_spans_json": "String",
    "temporal_markers_json": "String",
    "quote_spans_json": "String",
    "contradiction_keys_json": "String",
    "first_spans_json": "String",
    "projection_state": "String",
    "cue_score": "F64",
    "salience_score": "F64",
    "projection_priority": "F64",
    "route_reason": "String",
    "hit_count": "I32",
    "surfaced_count": "I32",
    "selected_count": "I32",
    "used_count": "I32",
    "near_miss_count": "I32",
    "policy_score": "F64",
    "projection_attempts": "I32",
    "last_hit_at": "String",
    "last_feedback_at": "String",
    "last_projected_at": "String",
    "created_at": "String",
    "updated_at": "String",
}
EPISODE_CUE_RUST_FIELDS = {
    field: {"I32": "i32", "F64": "f64"}.get(field_type, "String")
    for field, field_type in EPISODE_CUE_HX_FIELDS.items()
}
EPISODE_CUE_MUTABLE_FIELDS = [
    field
    for field in EPISODE_CUE_HX_FIELDS
    if field not in {"episode_id", "group_id", "created_at"}
]


def test_entity_schema_is_synced_across_helix_sources() -> None:
    """PyO3 native must preserve entity evidence/provenance fields."""
    server_fields = _helix_node_fields(SERVER_SCHEMA.read_text(), "Entity")
    native_fields = _helix_node_fields(NATIVE_SCHEMA.read_text(), "Entity")
    generated = NATIVE_GENERATED_QUERIES.read_text()
    generated_fields = _rust_struct_fields(generated, "Entity")

    assert server_fields == ENTITY_HX_FIELDS
    assert native_fields == ENTITY_HX_FIELDS
    assert generated_fields == ENTITY_RUST_FIELDS


def test_entity_queries_cover_provenance_fields() -> None:
    """Create/update/read paths must round-trip projected evidence lineage."""
    for schema_path in (SERVER_SCHEMA, NATIVE_SCHEMA):
        text = schema_path.read_text()
        create_signature = _helix_query_signature(text, "create_entity")
        update_signature = _helix_query_signature(text, "update_entity_full")

        for field in ENTITY_HX_FIELDS:
            assert field in create_signature
        for field in ENTITY_MUTABLE_FIELDS:
            assert field in update_signature

    generated = NATIVE_GENERATED_QUERIES.read_text()
    create_input = _rust_struct_fields(generated, "create_entityInput")
    update_input = _rust_struct_fields(generated, "update_entity_fullInput")
    get_return = _rust_struct_fields(generated, "Get_entityEntityReturnType")
    group_return = _rust_struct_fields(generated, "Find_entities_by_groupEntitiesReturnType")

    for field, field_type in ENTITY_RUST_FIELDS.items():
        assert create_input[field] == field_type
    for field, field_type in ENTITY_RUST_FIELDS.items():
        if field not in ENTITY_MUTABLE_FIELDS:
            continue
        assert update_input[field] == field_type
    for return_fields in (get_return, group_return):
        for field in ENTITY_HX_FIELDS:
            assert return_fields[field] == "Option<&'a Value>"


def test_episode_cue_schema_is_synced_across_helix_sources() -> None:
    """Keep native PyO3 and server Helix cue contracts from drifting."""
    server_fields = _helix_node_fields(SERVER_SCHEMA.read_text(), "EpisodeCue")
    native_fields = _helix_node_fields(NATIVE_SCHEMA.read_text(), "EpisodeCue")
    generated_fields = _rust_struct_fields(
        NATIVE_GENERATED_QUERIES.read_text(),
        "EpisodeCue",
    )

    assert server_fields == EPISODE_CUE_HX_FIELDS
    assert native_fields == EPISODE_CUE_HX_FIELDS
    assert generated_fields == EPISODE_CUE_RUST_FIELDS


def test_episode_cue_update_queries_cover_feedback_fields() -> None:
    """Cue usefulness depends on feedback fields reaching both Helix runtimes."""
    for schema_path in (SERVER_SCHEMA, NATIVE_SCHEMA):
        text = schema_path.read_text()
        assert "QUERY update_cue_by_episode" in text
        create_signature = _helix_query_signature(text, "create_episode_cue")
        update_signature = _helix_query_signature(text, "update_cue")
        key_update_signature = _helix_query_signature(text, "update_cue_by_episode")

        for field in EPISODE_CUE_HX_FIELDS:
            assert field in create_signature
        for field in EPISODE_CUE_MUTABLE_FIELDS:
            assert field in update_signature
            assert field in key_update_signature
        assert "ep_id: String" in key_update_signature
        assert "gid: String" in key_update_signature

    generated = NATIVE_GENERATED_QUERIES.read_text()
    assert "pub fn update_cue_by_episode" in generated
    create_input = _rust_struct_fields(generated, "create_episode_cueInput")
    update_input = _rust_struct_fields(generated, "update_cueInput")
    key_update_input = _rust_struct_fields(generated, "update_cue_by_episodeInput")

    for field, field_type in EPISODE_CUE_RUST_FIELDS.items():
        assert create_input[field] == field_type
    for field, field_type in EPISODE_CUE_RUST_FIELDS.items():
        if field not in EPISODE_CUE_MUTABLE_FIELDS:
            continue
        assert update_input[field] == field_type
        assert key_update_input[field] == field_type
    assert key_update_input["ep_id"] == "String"
    assert key_update_input["gid"] == "String"


def test_graph_embed_delete_route_is_available_to_native_helix() -> None:
    """Graph embedding cleanup must not disappear from the PyO3 route map."""
    for schema_path in (SERVER_SCHEMA, NATIVE_SCHEMA):
        text = schema_path.read_text()
        assert "QUERY delete_graph_embed_vector" in text
        signature = _helix_query_signature(text, "delete_graph_embed_vector")
        assert signature.strip() == "id: ID"

    generated = NATIVE_GENERATED_QUERIES.read_text()
    assert "pub fn delete_graph_embed_vector" in generated
    assert _rust_struct_fields(generated, "delete_graph_embed_vectorInput") == {
        "id": "ID"
    }


def test_open_adjudication_status_queries_are_native_available() -> None:
    """Native consolidation must see all lite-mode open adjudication work states."""
    for schema_path in (SERVER_SCHEMA, NATIVE_SCHEMA):
        text = schema_path.read_text()
        evidence_signature = _helix_query_signature(text, "find_evidence_by_status")
        adjudication_signature = _helix_query_signature(
            text,
            "find_adjudications_by_status",
        )

        assert evidence_signature.strip() == "gid: String, st: String"
        assert adjudication_signature.strip() == "gid: String, st: String"

    generated = NATIVE_GENERATED_QUERIES.read_text()
    assert "pub fn find_evidence_by_status" in generated
    assert "pub fn find_adjudications_by_status" in generated
    assert _rust_struct_fields(generated, "find_evidence_by_statusInput") == {
        "gid": "String",
        "st": "String",
    }
    assert _rust_struct_fields(generated, "find_adjudications_by_statusInput") == {
        "gid": "String",
        "st": "String",
    }


def _helix_node_fields(text: str, node_name: str) -> dict[str, str]:
    match = re.search(rf"N::{node_name}\s*\{{(?P<body>.*?)\n\}}", text, re.S)
    assert match, f"missing Helix node {node_name}"

    fields: dict[str, str] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip().rstrip(",")
        if not line:
            continue
        if line.startswith("INDEX "):
            line = line.removeprefix("INDEX ").strip()
        name, field_type = [part.strip() for part in line.split(":", 1)]
        fields[name] = field_type
    return fields


def _helix_query_signature(text: str, query_name: str) -> str:
    match = re.search(rf"QUERY {query_name}\((?P<signature>.*?)\)\s*=>", text, re.S)
    assert match, f"missing Helix query {query_name}"
    return match.group("signature")


def _rust_struct_fields(text: str, struct_name: str) -> dict[str, str]:
    match = re.search(
        rf"pub struct {struct_name}(?:<[^>]+>)?\s*\{{(?P<body>.*?)\n\}}",
        text,
        re.S,
    )
    assert match, f"missing Rust struct {struct_name}"

    fields: dict[str, str] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or not line.startswith("pub "):
            continue
        name, field_type = [part.strip() for part in line.removeprefix("pub ").split(":", 1)]
        fields[name] = field_type
    return fields
