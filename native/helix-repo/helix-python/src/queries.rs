
// DEFAULT CODE
// use helix_db::helix_engine::traversal_core::config::Config;

// pub fn config() -> Option<Config> {
//     None
// }



use bumpalo::Bump;
use heed3::RoTxn;
use helix_macros::{handler, tool_call, mcp_handler, migration};
use helix_db::{
    helix_engine::{
        reranker::{
            RerankAdapter,
            fusion::{RRFReranker, MMRReranker, DistanceMethod},
        },
        traversal_core::{
            config::{Config, GraphConfig, VectorConfig},
            ops::{
                bm25::search_bm25::SearchBM25Adapter,
                g::G,
                in_::{in_::InAdapter, in_e::InEdgesAdapter, to_n::ToNAdapter, to_v::ToVAdapter},
                out::{
                    from_n::FromNAdapter, from_v::FromVAdapter, out::OutAdapter, out_e::OutEdgesAdapter,
                },
                source::{
                    add_e::AddEAdapter,
                    add_n::AddNAdapter,
                    e_from_id::EFromIdAdapter,
                    e_from_type::EFromTypeAdapter,
                    n_from_id::NFromIdAdapter,
                    n_from_index::NFromIndexAdapter,
                    n_from_type::NFromTypeAdapter,
                    v_from_id::VFromIdAdapter,
                    v_from_type::VFromTypeAdapter
                },
                util::{
                    dedup::DedupAdapter, drop::Drop, exist::Exist, filter_mut::FilterMut,
                    filter_ref::FilterRefAdapter, intersect::IntersectAdapter, map::MapAdapter, paths::{PathAlgorithm, ShortestPathAdapter},
                    range::RangeAdapter, update::UpdateAdapter, order::OrderByAdapter,
                    aggregate::AggregateAdapter, group_by::GroupByAdapter, count::CountAdapter,
                    upsert::UpsertAdapter,
                },
                vectors::{
                    brute_force_search::BruteForceSearchVAdapter, insert::InsertVAdapter,
                    search::SearchVAdapter,
                },
            },
            traversal_value::TraversalValue,
        },
        types::{GraphError, SecondaryIndex},
        vector_core::vector::HVector,
    },
    helix_gateway::{
        embedding_providers::{EmbeddingModel, get_embedding_model},
        router::router::{HandlerInput, IoContFn},
        mcp::mcp::{MCPHandlerSubmission, MCPToolInput, MCPHandler}
    },
    node_matches, props, embed, embed_async,
    field_addition_from_old_field, field_type_cast, field_addition_from_value,
    protocol::{
        response::Response,
        value::{casting::{cast, CastType}, Value},
        date::Date,
        format::Format,
    },
    utils::{
        id::{ID, uuid_str},
        items::{Edge, Node},
        properties::ImmutablePropertiesMap,
    },
};
use sonic_rs::{Deserialize, Serialize, json};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use chrono::{DateTime, Utc};

// Re-export scalar types for generated code
type I8 = i8;
type I16 = i16;
type I32 = i32;
type I64 = i64;
type U8 = u8;
type U16 = u16;
type U32 = u32;
type U64 = u64;
type U128 = u128;
type F32 = f32;
type F64 = f64;

pub fn config() -> Option<Config> {
return Some(Config {
vector_config: Some(VectorConfig {
m: Some(16),
ef_construction: Some(128),
ef_search: Some(768),
}),
graph_config: Some(GraphConfig {
secondary_indices: Some(vec![SecondaryIndex::Index("name".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("entity_type".to_string()), SecondaryIndex::Index("canonical_identifier".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("status".to_string()), SecondaryIndex::Index("session_id".to_string()), SecondaryIndex::Index("episode_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("episode_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("status".to_string()), SecondaryIndex::Index("episode_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("status".to_string()), SecondaryIndex::Index("schema_entity_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("conversation_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("snapshot_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("snapshot_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("snapshot_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("region_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("target_id".to_string()), SecondaryIndex::Index("group_id".to_string()), SecondaryIndex::Index("cycle_id".to_string()), SecondaryIndex::Index("group_id".to_string())]),
}),
db_max_size_gb: Some(20),
mcp: Some(true),
bm25: Some(true),
schema: Some(r#"{
  "schema": {
    "nodes": [
      {
        "name": "Entity",
        "properties": {
          "entity_id": "String",
          "updated_at": "String",
          "group_id": "String",
          "lexical_regime": "String",
          "pii_categories_json": "String",
          "mat_tier": "String",
          "entity_type": "String",
          "created_at": "String",
          "recon_count": "I32",
          "access_count": "I64",
          "id": "ID",
          "label": "String",
          "identifier_label": "String",
          "last_accessed": "String",
          "source_episode_ids": "String",
          "evidence_count": "I64",
          "evidence_span_start": "String",
          "evidence_span_end": "String",
          "identity_core": "Boolean",
          "name": "String",
          "summary": "String",
          "canonical_identifier": "String",
          "pii_detected": "Boolean",
          "attributes_json": "String",
          "is_deleted": "Boolean",
          "deleted_at": "String"
        }
      },
      {
        "name": "Episode",
        "properties": {
          "content": "String",
          "projection_state": "String",
          "last_projection_reason": "String",
          "entity_coverage": "F64",
          "last_projected_at": "String",
          "status": "String",
          "session_id": "String",
          "processing_duration_ms": "I64",
          "id": "ID",
          "encoding_context_json": "String",
          "episode_id": "String",
          "skipped_triage": "Boolean",
          "created_at": "String",
          "updated_at": "String",
          "retry_count": "I32",
          "memory_tier": "String",
          "error": "String",
          "conversation_date": "String",
          "attachments_json": "String",
          "label": "String",
          "source": "String",
          "group_id": "String",
          "skipped_meta": "Boolean",
          "consolidation_cycles": "I32"
        }
      },
      {
        "name": "EpisodeCue",
        "properties": {
          "contradiction_keys_json": "String",
          "cue_score": "F64",
          "cue_text": "String",
          "cue_version": "I32",
          "created_at": "String",
          "discourse_class": "String",
          "episode_id": "String",
          "first_spans_json": "String",
          "group_id": "String",
          "hit_count": "I32",
          "id": "ID",
          "label": "String",
          "last_feedback_at": "String",
          "last_hit_at": "String",
          "last_projected_at": "String",
          "near_miss_count": "I32",
          "policy_score": "F64",
          "projection_attempts": "I32",
          "projection_priority": "F64",
          "projection_state": "String",
          "quote_spans_json": "String",
          "route_reason": "String",
          "salience_score": "F64",
          "selected_count": "I32",
          "supporting_spans_json": "String",
          "surfaced_count": "I32",
          "temporal_markers_json": "String",
          "updated_at": "String",
          "used_count": "I32"
        }
      },
      {
        "name": "Intention",
        "properties": {
          "enabled": "Boolean",
          "max_fires": "I32",
          "id": "ID",
          "entity_names_json": "String",
          "trigger_text": "String",
          "created_at": "String",
          "group_id": "String",
          "fire_count": "I32",
          "deleted_at": "String",
          "is_deleted": "Boolean",
          "context_json": "String",
          "intention_id": "String",
          "label": "String",
          "updated_at": "String",
          "action_text": "String"
        }
      },
      {
        "name": "Evidence",
        "properties": {
          "created_at": "String",
          "label": "String",
          "source_span": "String",
          "fact_class": "String",
          "committed_id": "String",
          "payload_json": "String",
          "adjudication_request_id": "String",
          "signals_json": "String",
          "ambiguity_score": "F64",
          "group_id": "String",
          "evidence_id": "String",
          "id": "ID",
          "status": "String",
          "commit_reason": "String",
          "ambiguity_tags_json": "String",
          "source_type": "String",
          "extractor_name": "String",
          "episode_id": "String",
          "deferred_cycles": "I32",
          "confidence": "F64",
          "resolved_at": "String"
        }
      },
      {
        "name": "AdjudicationRequest",
        "properties": {
          "request_id": "String",
          "created_at": "String",
          "label": "String",
          "group_id": "String",
          "episode_id": "String",
          "ambiguity_tags_json": "String",
          "selected_text": "String",
          "resolved_at": "String",
          "resolution_source": "String",
          "request_reason": "String",
          "id": "ID",
          "status": "String",
          "resolution_payload_json": "String",
          "evidence_ids_json": "String",
          "attempt_count": "I32"
        }
      },
      {
        "name": "SchemaMember",
        "properties": {
          "role_label": "String",
          "id": "ID",
          "label": "String",
          "member_entity_id": "String",
          "schema_entity_id": "String",
          "group_id": "String"
        }
      },
      {
        "name": "Conversation",
        "properties": {
          "title": "String",
          "session_date": "String",
          "conversation_id": "String",
          "label": "String",
          "group_id": "String",
          "created_at": "String",
          "updated_at": "String",
          "id": "ID"
        }
      },
      {
        "name": "ConversationMessage",
        "properties": {
          "content": "String",
          "parts_json": "String",
          "id": "ID",
          "message_id": "String",
          "role": "String",
          "conversation_id": "String",
          "label": "String",
          "created_at": "String"
        }
      },
      {
        "name": "AtlasSnapshot",
        "properties": {
          "total_entities": "I32",
          "generated_at": "String",
          "group_id": "String",
          "displayed_node_count": "I32",
          "total_relationships": "I32",
          "fastest_growing_region_id": "String",
          "represented_edge_count": "I32",
          "represented_entity_count": "I32",
          "displayed_edge_count": "I32",
          "total_regions": "I32",
          "label": "String",
          "id": "ID",
          "snapshot_id": "String",
          "hottest_region_id": "String",
          "truncated": "Boolean"
        }
      },
      {
        "name": "AtlasRegion",
        "properties": {
          "represented_edge_count": "I32",
          "region_id": "String",
          "member_count": "I32",
          "label": "String",
          "activation_score": "F64",
          "hub_entity_ids_json": "String",
          "dominant_entity_types_json": "String",
          "center_entity_id": "String",
          "subtitle": "String",
          "latest_entity_created_at": "String",
          "z": "F64",
          "region_label": "String",
          "id": "ID",
          "y": "F64",
          "group_id": "String",
          "kind": "String",
          "growth_7d": "I32",
          "x": "F64",
          "snapshot_id": "String",
          "growth_30d": "I32"
        }
      },
      {
        "name": "AtlasRegionEdge",
        "properties": {
          "source_region_id": "String",
          "edge_id": "String",
          "label": "String",
          "group_id": "String",
          "snapshot_id": "String",
          "target_region_id": "String",
          "weight": "F64",
          "relationship_count": "I32",
          "id": "ID"
        }
      },
      {
        "name": "AtlasRegionMember",
        "properties": {
          "entity_id": "String",
          "id": "ID",
          "group_id": "String",
          "label": "String",
          "snapshot_id": "String",
          "region_id": "String"
        }
      },
      {
        "name": "ConsolCycle",
        "properties": {
          "label": "String",
          "cycle_id": "String",
          "phase_results_json": "String",
          "total_duration_ms": "F64",
          "trigger": "String",
          "id": "ID",
          "status": "String",
          "error": "String",
          "started_at": "F64",
          "group_id": "String",
          "completed_at": "F64",
          "dry_run": "Boolean"
        }
      },
      {
        "name": "ConsolMerge",
        "properties": {
          "decision_reason": "String",
          "id": "ID",
          "label": "String",
          "group_id": "String",
          "timestamp": "F64",
          "keep_id": "String",
          "decision_confidence": "F64",
          "remove_id": "String",
          "merge_id": "String",
          "cycle_id": "String",
          "decision_source": "String",
          "keep_name": "String",
          "relationships_transferred": "I32",
          "remove_name": "String",
          "similarity": "F64"
        }
      },
      {
        "name": "ConsolIdentifierReview",
        "properties": {
          "entity_b_name": "String",
          "raw_similarity": "F64",
          "id": "ID",
          "entity_b_type": "String",
          "decision_reason": "String",
          "review_id": "String",
          "entity_a_id": "String",
          "entity_a_regime": "String",
          "entity_b_id": "String",
          "entity_a_type": "String",
          "metadata_json": "String",
          "cycle_id": "String",
          "group_id": "String",
          "entity_a_name": "String",
          "entity_b_regime": "String",
          "label": "String",
          "canonical_identifier_b": "String",
          "canonical_identifier_a": "String",
          "adjusted_similarity": "F64",
          "review_status": "String",
          "decision_source": "String",
          "timestamp": "F64"
        }
      },
      {
        "name": "ConsolInferredEdge",
        "properties": {
          "cycle_id": "String",
          "target_id": "String",
          "pmi_score": "F64",
          "co_occurrence_count": "I32",
          "infer_type": "String",
          "source_id": "String",
          "target_name": "String",
          "label": "String",
          "group_id": "String",
          "llm_verdict": "String",
          "timestamp": "F64",
          "edge_id": "String",
          "confidence": "F64",
          "id": "ID",
          "source_name": "String",
          "relationship_id": "String"
        }
      },
      {
        "name": "ConsolPrune",
        "properties": {
          "entity_id": "String",
          "reason": "String",
          "label": "String",
          "cycle_id": "String",
          "entity_name": "String",
          "prune_id": "String",
          "entity_type": "String",
          "id": "ID",
          "timestamp": "F64",
          "group_id": "String"
        }
      },
      {
        "name": "ConsolReindex",
        "properties": {
          "id": "ID",
          "cycle_id": "String",
          "entity_id": "String",
          "source_phase": "String",
          "entity_name": "String",
          "group_id": "String",
          "label": "String",
          "timestamp": "F64",
          "reindex_id": "String"
        }
      },
      {
        "name": "ConsolReplay",
        "properties": {
          "new_entities_found": "I32",
          "cycle_id": "String",
          "skipped_reason": "String",
          "group_id": "String",
          "id": "ID",
          "episode_id": "String",
          "new_relationships_found": "I32",
          "timestamp": "F64",
          "replay_id": "String",
          "entities_updated": "I32",
          "label": "String"
        }
      },
      {
        "name": "ConsolDream",
        "properties": {
          "target_entity_id": "String",
          "timestamp": "F64",
          "dream_id": "String",
          "cycle_id": "String",
          "weight_delta": "F64",
          "id": "ID",
          "label": "String",
          "seed_entity_id": "String",
          "group_id": "String",
          "source_entity_id": "String"
        }
      },
      {
        "name": "ConsolTriage",
        "properties": {
          "label": "String",
          "score": "F64",
          "score_breakdown_json": "String",
          "id": "ID",
          "triage_id": "String",
          "episode_id": "String",
          "decision": "String",
          "timestamp": "F64",
          "group_id": "String",
          "cycle_id": "String"
        }
      },
      {
        "name": "ConsolDreamAssociation",
        "properties": {
          "target_domain": "String",
          "source_domain": "String",
          "source_entity_id": "String",
          "group_id": "String",
          "surprise_score": "F64",
          "structural_proximity": "F64",
          "timestamp": "F64",
          "embedding_similarity": "F64",
          "target_entity_name": "String",
          "label": "String",
          "relationship_id": "String",
          "id": "ID",
          "cycle_id": "String",
          "assoc_id": "String",
          "source_entity_name": "String",
          "target_entity_id": "String"
        }
      },
      {
        "name": "ConsolGraphEmbed",
        "properties": {
          "method": "String",
          "entities_trained": "I32",
          "full_retrain": "Boolean",
          "id": "ID",
          "training_duration_ms": "F64",
          "dimensions": "I32",
          "group_id": "String",
          "label": "String",
          "cycle_id": "String",
          "embed_id": "String",
          "timestamp": "F64"
        }
      },
      {
        "name": "ConsolMaturation",
        "properties": {
          "group_id": "String",
          "temporal_span_days": "F64",
          "timestamp": "F64",
          "old_tier": "String",
          "label": "String",
          "source_diversity": "I32",
          "relationship_richness": "I32",
          "maturity_score": "F64",
          "cycle_id": "String",
          "mat_id": "String",
          "access_regularity": "F64",
          "id": "ID",
          "entity_id": "String",
          "entity_name": "String",
          "new_tier": "String"
        }
      },
      {
        "name": "ConsolSemanticTransition",
        "properties": {
          "new_tier": "String",
          "entity_coverage": "F64",
          "group_id": "String",
          "id": "ID",
          "timestamp": "F64",
          "consolidation_cycles": "I32",
          "episode_id": "String",
          "label": "String",
          "cycle_id": "String",
          "trans_id": "String",
          "old_tier": "String"
        }
      },
      {
        "name": "ConsolSchema",
        "properties": {
          "timestamp": "F64",
          "schema_entity_id": "String",
          "cycle_id": "String",
          "predicate_count": "I32",
          "schema_id": "String",
          "schema_name": "String",
          "group_id": "String",
          "action": "String",
          "id": "ID",
          "label": "String",
          "instance_count": "I32"
        }
      },
      {
        "name": "ConsolDecisionTrace",
        "properties": {
          "cycle_id": "String",
          "confidence": "F64",
          "phase": "String",
          "decision": "String",
          "candidate_id": "String",
          "label": "String",
          "decision_source": "String",
          "metadata_json": "String",
          "constraints_json": "String",
          "features_json": "String",
          "threshold_band": "String",
          "group_id": "String",
          "trace_id": "String",
          "candidate_type": "String",
          "policy_version": "String",
          "id": "ID",
          "timestamp": "F64"
        }
      },
      {
        "name": "ConsolDecisionOutcome",
        "properties": {
          "label": "String",
          "decision_trace_id": "String",
          "group_id": "String",
          "outcome_label": "String",
          "cycle_id": "String",
          "id": "ID",
          "outcome_value": "F64",
          "outcome_id": "String",
          "metadata_json": "String",
          "phase": "String",
          "outcome_type": "String",
          "timestamp": "F64"
        }
      },
      {
        "name": "ConsolDistillation",
        "properties": {
          "student_confidence": "F64",
          "decision_trace_id": "String",
          "candidate_id": "String",
          "phase": "String",
          "student_decision": "String",
          "distill_id": "String",
          "features_json": "String",
          "cycle_id": "String",
          "timestamp": "F64",
          "candidate_type": "String",
          "teacher_source": "String",
          "group_id": "String",
          "label": "String",
          "id": "ID",
          "correct": "Boolean",
          "teacher_label": "String",
          "threshold_band": "String",
          "metadata_json": "String"
        }
      },
      {
        "name": "ConsolCalibration",
        "properties": {
          "id": "ID",
          "oracle_examples": "I32",
          "mean_confidence": "F64",
          "expected_calibration_error": "F64",
          "cycle_id": "String",
          "summary_json": "String",
          "calibration_id": "String",
          "timestamp": "F64",
          "accuracy": "F64",
          "window_cycles": "I32",
          "label": "String",
          "labeled_examples": "I32",
          "total_traces": "I32",
          "group_id": "String",
          "phase": "String",
          "abstain_count": "I32"
        }
      },
      {
        "name": "ConsolEvidenceAdj",
        "properties": {
          "cycle_id": "String",
          "id": "ID",
          "reason": "String",
          "evidence_id": "String",
          "timestamp": "F64",
          "label": "String",
          "new_confidence": "F64",
          "group_id": "String",
          "adj_id": "String",
          "action": "String"
        }
      },
      {
        "name": "ComplementTag",
        "properties": {
          "id": "ID",
          "updated_at": "String",
          "target_id": "String",
          "label": "String",
          "score": "F64",
          "group_id": "String",
          "tag_type": "String",
          "target_type": "String",
          "cycle_tagged": "I32",
          "cycle_confirmed": "I32",
          "cleared": "Boolean",
          "created_at": "String"
        }
      },
      {
        "name": "ConsolMicroglia",
        "properties": {
          "timestamp": "F64",
          "detail": "String",
          "id": "ID",
          "label": "String",
          "action": "String",
          "score": "F64",
          "target_type": "String",
          "cycle_id": "String",
          "microglia_id": "String",
          "target_id": "String",
          "group_id": "String",
          "tag_type": "String"
        }
      }
    ],
    "vectors": [
      {
        "name": "EntityVec",
        "properties": {
          "score": "F64",
          "content_type": "String",
          "group_id": "String",
          "embed_model": "String",
          "data": "Array(F64)",
          "label": "String",
          "id": "ID",
          "entity_id": "String",
          "embed_provider": "String"
        }
      },
      {
        "name": "EpisodeVec",
        "properties": {
          "data": "Array(F64)",
          "episode_id": "String",
          "label": "String",
          "group_id": "String",
          "score": "F64",
          "content_type": "String",
          "id": "ID"
        }
      },
      {
        "name": "CueVec",
        "properties": {
          "group_id": "String",
          "id": "ID",
          "label": "String",
          "data": "Array(F64)",
          "score": "F64",
          "content_type": "String",
          "episode_id": "String"
        }
      },
      {
        "name": "GraphEmbedVec",
        "properties": {
          "group_id": "String",
          "method": "String",
          "model_version": "String",
          "data": "Array(F64)",
          "score": "F64",
          "id": "ID",
          "entity_id": "String",
          "label": "String"
        }
      },
      {
        "name": "EpisodeChunk",
        "properties": {
          "chunk_text": "String",
          "data": "Array(F64)",
          "chunk_index": "I32",
          "score": "F64",
          "label": "String",
          "episode_id": "String",
          "id": "ID",
          "group_id": "String",
          "content_type": "String"
        }
      }
    ],
    "edges": [
      {
        "name": "RelatesTo",
        "from": "Entity",
        "to": "Entity",
        "properties": {
          "rel_id": "String",
          "group_id": "String",
          "polarity": "String",
          "valid_from": "String",
          "valid_to": "String",
          "predicate": "String",
          "weight": "F64",
          "is_expired": "Boolean",
          "created_at": "String",
          "source_episode_id": "String"
        }
      },
      {
        "name": "HasConversationEntity",
        "from": "Conversation",
        "to": "Entity",
        "properties": {}
      },
      {
        "name": "HasEpisodeChunk",
        "from": "Episode",
        "to": "EpisodeChunk",
        "properties": {}
      },
      {
        "name": "HasEntity",
        "from": "Episode",
        "to": "Entity",
        "properties": {}
      },
      {
        "name": "HasSchemaMember",
        "from": "Entity",
        "to": "SchemaMember",
        "properties": {}
      }
    ]
  },
  "queries": [
    {
      "name": "find_consol_merges_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "merges"
      ]
    },
    {
      "name": "update_evidence",
      "parameters": {
        "status": "String",
        "resolved_at": "String",
        "commit_reason": "String",
        "committed_id": "String",
        "id": "ID"
      },
      "returns": [
        "evidence"
      ]
    },
    {
      "name": "update_entity_full",
      "parameters": {
        "summary": "String",
        "pii_categories_json": "String",
        "mat_tier": "String",
        "updated_at": "String",
        "id": "ID",
        "name": "String",
        "entity_type": "String",
        "deleted_at": "String",
        "identifier_label": "String",
        "identity_core": "Boolean",
        "pii_detected": "Boolean",
        "access_count": "I64",
        "last_accessed": "String",
        "source_episode_ids": "String",
        "evidence_count": "I64",
        "evidence_span_start": "String",
        "evidence_span_end": "String",
        "is_deleted": "Boolean",
        "canonical_identifier": "String",
        "lexical_regime": "String",
        "attributes_json": "String",
        "recon_count": "I32"
      },
      "returns": [
        "entity"
      ]
    },
    {
      "name": "find_consol_cycles_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "cycles"
      ]
    },
    {
      "name": "find_consol_cycles_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "cycles"
      ]
    },
    {
      "name": "create_consol_cycle",
      "parameters": {
        "phase_results_json": "String",
        "completed_at": "F64",
        "started_at": "F64",
        "error": "String",
        "cycle_id": "String",
        "dry_run": "Boolean",
        "group_id": "String",
        "total_duration_ms": "F64",
        "status": "String",
        "trigger": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_atlas_region_edges",
      "parameters": {
        "gid": "String",
        "snap_id": "String"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "get_incoming_neighbors",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "neighbors"
      ]
    },
    {
      "name": "get_consol_cycle",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "cycle"
      ]
    },
    {
      "name": "create_atlas_region_member",
      "parameters": {
        "snapshot_id": "String",
        "group_id": "String",
        "entity_id": "String",
        "region_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_consol_calibration",
      "parameters": {
        "summary_json": "String",
        "cycle_id": "String",
        "oracle_examples": "I32",
        "timestamp": "F64",
        "accuracy": "F64",
        "total_traces": "I32",
        "group_id": "String",
        "window_cycles": "I32",
        "mean_confidence": "F64",
        "expected_calibration_error": "F64",
        "abstain_count": "I32",
        "labeled_examples": "I32",
        "calibration_id": "String",
        "phase": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "update_episode_full",
      "parameters": {
        "updated_at": "String",
        "id": "ID",
        "error": "String",
        "last_projection_reason": "String",
        "status": "String",
        "encoding_context_json": "String",
        "skipped_triage": "Boolean",
        "entity_coverage": "F64",
        "conversation_date": "String",
        "attachments_json": "String",
        "memory_tier": "String",
        "processing_duration_ms": "I64",
        "projection_state": "String",
        "retry_count": "I32",
        "last_projected_at": "String",
        "skipped_meta": "Boolean",
        "consolidation_cycles": "I32",
        "content": "String"
      },
      "returns": [
        "episode"
      ]
    },
    {
      "name": "find_consol_distillations_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "distillations"
      ]
    },
    {
      "name": "create_consol_prune",
      "parameters": {
        "group_id": "String",
        "prune_id": "String",
        "entity_type": "String",
        "reason": "String",
        "cycle_id": "String",
        "timestamp": "F64",
        "entity_id": "String",
        "entity_name": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_complement_tags_by_target",
      "parameters": {
        "target_id": "String"
      },
      "returns": [
        "tags"
      ]
    },
    {
      "name": "delete_conversation",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "hard_delete_schema_member",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "hard_delete_conversation_message",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "find_confirmed_complement_tags",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "tags"
      ]
    },
    {
      "name": "find_consol_microglias_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "records"
      ]
    },
    {
      "name": "get_conversation",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "conversation"
      ]
    },
    {
      "name": "find_consol_dream_associations_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "assocs"
      ]
    },
    {
      "name": "create_atlas_region",
      "parameters": {
        "region_label": "String",
        "group_id": "String",
        "x": "F64",
        "z": "F64",
        "center_entity_id": "String",
        "subtitle": "String",
        "growth_7d": "I32",
        "kind": "String",
        "activation_score": "F64",
        "region_id": "String",
        "member_count": "I32",
        "represented_edge_count": "I32",
        "growth_30d": "I32",
        "hub_entity_ids_json": "String",
        "y": "F64",
        "latest_entity_created_at": "String",
        "snapshot_id": "String",
        "dominant_entity_types_json": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_episodes_by_status",
      "parameters": {
        "st": "String",
        "gid": "String"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "create_consol_reindex",
      "parameters": {
        "timestamp": "F64",
        "cycle_id": "String",
        "group_id": "String",
        "entity_id": "String",
        "entity_name": "String",
        "reindex_id": "String",
        "source_phase": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_schema_member",
      "parameters": {
        "member_entity_id": "String",
        "group_id": "String",
        "role_label": "String",
        "schema_entity_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "get_incoming_edges_by_predicate",
      "parameters": {
        "pred": "String",
        "id": "ID"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "create_episode_cue",
      "parameters": {
        "contradiction_keys_json": "String",
        "cue_score": "F64",
        "cue_text": "String",
        "cue_version": "I32",
        "created_at": "String",
        "discourse_class": "String",
        "episode_id": "String",
        "first_spans_json": "String",
        "group_id": "String",
        "hit_count": "I32",
        "last_feedback_at": "String",
        "last_hit_at": "String",
        "last_projected_at": "String",
        "near_miss_count": "I32",
        "policy_score": "F64",
        "projection_attempts": "I32",
        "projection_priority": "F64",
        "projection_state": "String",
        "quote_spans_json": "String",
        "route_reason": "String",
        "salience_score": "F64",
        "selected_count": "I32",
        "supporting_spans_json": "String",
        "surfaced_count": "I32",
        "temporal_markers_json": "String",
        "updated_at": "String",
        "used_count": "I32"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_consol_evidence_adj",
      "parameters": {
        "action": "String",
        "group_id": "String",
        "cycle_id": "String",
        "timestamp": "F64",
        "reason": "String",
        "evidence_id": "String",
        "new_confidence": "F64",
        "adj_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "hard_delete_entity",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "search_episode_chunks_vec",
      "parameters": {
        "vec": "Array(F64)",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "create_consol_decision_trace",
      "parameters": {
        "trace_id": "String",
        "cycle_id": "String",
        "group_id": "String",
        "decision_source": "String",
        "threshold_band": "String",
        "phase": "String",
        "constraints_json": "String",
        "metadata_json": "String",
        "policy_version": "String",
        "candidate_type": "String",
        "confidence": "F64",
        "timestamp": "F64",
        "decision": "String",
        "features_json": "String",
        "candidate_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_entity_ids_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": []
    },
    {
      "name": "get_entity_cooccurrences",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "cooccurring"
      ]
    },
    {
      "name": "get_two_hop_neighbors",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "neighbors"
      ]
    },
    {
      "name": "create_entity",
      "parameters": {
        "last_accessed": "String",
        "updated_at": "String",
        "pii_categories_json": "String",
        "summary": "String",
        "deleted_at": "String",
        "lexical_regime": "String",
        "canonical_identifier": "String",
        "entity_id": "String",
        "pii_detected": "Boolean",
        "entity_type": "String",
        "group_id": "String",
        "access_count": "I64",
        "created_at": "String",
        "recon_count": "I32",
        "mat_tier": "String",
        "attributes_json": "String",
        "is_deleted": "Boolean",
        "name": "String",
        "identifier_label": "String",
        "identity_core": "Boolean",
        "source_episode_ids": "String",
        "evidence_count": "I64",
        "evidence_span_start": "String",
        "evidence_span_end": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_episode",
      "parameters": {
        "skipped_meta": "Boolean",
        "updated_at": "String",
        "memory_tier": "String",
        "processing_duration_ms": "I64",
        "skipped_triage": "Boolean",
        "content": "String",
        "session_id": "String",
        "source": "String",
        "group_id": "String",
        "error": "String",
        "last_projection_reason": "String",
        "consolidation_cycles": "I32",
        "entity_coverage": "F64",
        "status": "String",
        "attachments_json": "String",
        "encoding_context_json": "String",
        "episode_id": "String",
        "created_at": "String",
        "last_projected_at": "String",
        "projection_state": "String",
        "retry_count": "I32",
        "conversation_date": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_consol_dreams_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "dreams"
      ]
    },
    {
      "name": "invalidate_edge",
      "parameters": {
        "valid_to": "String",
        "id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "find_consol_reindexes_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "reindexes"
      ]
    },
    {
      "name": "hard_delete_atlas_region_edge",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "create_consol_triage",
      "parameters": {
        "group_id": "String",
        "cycle_id": "String",
        "triage_id": "String",
        "timestamp": "F64",
        "episode_id": "String",
        "decision": "String",
        "score": "F64",
        "score_breakdown_json": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "search_entities_embed",
      "parameters": {
        "k": "I32",
        "query": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "hard_delete_cue",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "hard_delete_episode",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "get_edge",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "get_intention",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "intention"
      ]
    },
    {
      "name": "find_entities_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "create_evidence",
      "parameters": {
        "extractor_name": "String",
        "source_span": "String",
        "committed_id": "String",
        "adjudication_request_id": "String",
        "deferred_cycles": "I32",
        "group_id": "String",
        "resolved_at": "String",
        "ambiguity_score": "F64",
        "evidence_id": "String",
        "confidence": "F64",
        "source_type": "String",
        "payload_json": "String",
        "fact_class": "String",
        "created_at": "String",
        "episode_id": "String",
        "signals_json": "String",
        "ambiguity_tags_json": "String",
        "commit_reason": "String",
        "status": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_pending_adjudications",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "requests"
      ]
    },
    {
      "name": "create_consol_distillation",
      "parameters": {
        "distill_id": "String",
        "candidate_type": "String",
        "student_confidence": "F64",
        "correct": "Boolean",
        "decision_trace_id": "String",
        "teacher_label": "String",
        "student_decision": "String",
        "features_json": "String",
        "metadata_json": "String",
        "phase": "String",
        "threshold_band": "String",
        "timestamp": "F64",
        "candidate_id": "String",
        "cycle_id": "String",
        "group_id": "String",
        "teacher_source": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "add_cue_vector",
      "parameters": {
        "vec": "Array(F64)",
        "group_id": "String",
        "content_type": "String",
        "episode_id": "String"
      },
      "returns": [
        "v"
      ]
    },
    {
      "name": "find_entities_by_group_limited",
      "parameters": {
        "gid": "String",
        "limit": "I64"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "create_conversation",
      "parameters": {
        "created_at": "String",
        "group_id": "String",
        "updated_at": "String",
        "conversation_id": "String",
        "session_date": "String",
        "title": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_messages_by_conversation",
      "parameters": {
        "conv_id": "String"
      },
      "returns": [
        "messages"
      ]
    },
    {
      "name": "search_entity_vectors_filtered",
      "parameters": {
        "vec": "Array(F64)",
        "k": "I32",
        "gid": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "get_episode_chunks",
      "parameters": {
        "ep_id": "String"
      },
      "returns": [
        "chunks"
      ]
    },
    {
      "name": "find_consol_decision_traces_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "traces"
      ]
    },
    {
      "name": "get_outgoing_edges",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "find_evidence_by_episode",
      "parameters": {
        "ep_id": "String",
        "gid": "String"
      },
      "returns": [
        "evidence"
      ]
    },
    {
      "name": "search_entities_bm25",
      "parameters": {
        "query": "String",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "search_cue_vectors",
      "parameters": {
        "vec": "Array(F64)",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "get_episodes_for_entity",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "update_conversation",
      "parameters": {
        "title": "String",
        "updated_at": "String",
        "id": "ID"
      },
      "returns": [
        "conversation"
      ]
    },
    {
      "name": "update_adjudication",
      "parameters": {
        "resolution_payload_json": "String",
        "resolved_at": "String",
        "attempt_count": "I32",
        "resolution_source": "String",
        "status": "String",
        "id": "ID"
      },
      "returns": [
        "request"
      ]
    },
    {
      "name": "find_entities_by_type",
      "parameters": {
        "gid": "String",
        "etype": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "hard_delete_atlas_region_member",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "find_consol_inferred_edges_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "find_consol_triages_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "triages"
      ]
    },
    {
      "name": "hard_delete_evidence",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "find_consol_calibrations_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "calibrations"
      ]
    },
    {
      "name": "search_episode_vectors_filtered",
      "parameters": {
        "gid": "String",
        "vec": "Array(F64)",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "hard_delete_episode_chunk",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "create_relationship",
      "parameters": {
        "source_id": "ID",
        "source_episode_id": "String",
        "weight": "F64",
        "group_id": "String",
        "polarity": "String",
        "rel_id": "String",
        "valid_from": "String",
        "is_expired": "Boolean",
        "target_id": "ID",
        "predicate": "String",
        "valid_to": "String",
        "created_at": "String"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "find_unconfirmed_complement_tags",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "tags"
      ]
    },
    {
      "name": "get_incoming_edges",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "create_consol_maturation",
      "parameters": {
        "old_tier": "String",
        "temporal_span_days": "F64",
        "access_regularity": "F64",
        "group_id": "String",
        "maturity_score": "F64",
        "timestamp": "F64",
        "cycle_id": "String",
        "entity_id": "String",
        "mat_id": "String",
        "entity_name": "String",
        "new_tier": "String",
        "source_diversity": "I32",
        "relationship_richness": "I32"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_atlas_regions",
      "parameters": {
        "snap_id": "String",
        "gid": "String"
      },
      "returns": [
        "regions"
      ]
    },
    {
      "name": "find_enabled_intentions",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "intentions"
      ]
    },
    {
      "name": "get_outgoing_neighbors",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "neighbors"
      ]
    },
    {
      "name": "find_consol_graph_embeds_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "embeds"
      ]
    },
    {
      "name": "find_episodes_by_session",
      "parameters": {
        "sid": "String"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "search_episode_vectors",
      "parameters": {
        "k": "I32",
        "vec": "Array(F64)"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "create_consol_dream",
      "parameters": {
        "seed_entity_id": "String",
        "cycle_id": "String",
        "weight_delta": "F64",
        "source_entity_id": "String",
        "group_id": "String",
        "dream_id": "String",
        "timestamp": "F64",
        "target_entity_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "get_evidence",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "evidence"
      ]
    },
    {
      "name": "create_consol_graph_embed",
      "parameters": {
        "method": "String",
        "cycle_id": "String",
        "timestamp": "F64",
        "group_id": "String",
        "training_duration_ms": "F64",
        "full_retrain": "Boolean",
        "embed_id": "String",
        "entities_trained": "I32",
        "dimensions": "I32"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "update_intention_full",
      "parameters": {
        "is_deleted": "Boolean",
        "id": "ID",
        "entity_names_json": "String",
        "enabled": "Boolean",
        "trigger_text": "String",
        "deleted_at": "String",
        "updated_at": "String",
        "context_json": "String",
        "action_text": "String",
        "max_fires": "I32",
        "fire_count": "I32"
      },
      "returns": [
        "intention"
      ]
    },
    {
      "name": "find_consol_schemas_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "schemas"
      ]
    },
    {
      "name": "find_consol_evidence_adjs_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "adjs"
      ]
    },
    {
      "name": "search_episodes_embed",
      "parameters": {
        "query": "String",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "soft_delete_intention",
      "parameters": {
        "id": "ID",
        "deleted_at": "String"
      },
      "returns": [
        "intention"
      ]
    },
    {
      "name": "get_episode_entities",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "create_atlas_region_edge",
      "parameters": {
        "target_region_id": "String",
        "relationship_count": "I32",
        "weight": "F64",
        "source_region_id": "String",
        "group_id": "String",
        "edge_id": "String",
        "snapshot_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "add_entity_vector",
      "parameters": {
        "content_type": "String",
        "group_id": "String",
        "embed_provider": "String",
        "vec": "Array(F64)",
        "entity_id": "String",
        "embed_model": "String"
      },
      "returns": [
        "v"
      ]
    },
    {
      "name": "find_schema_members",
      "parameters": {
        "gid": "String",
        "schema_id": "String"
      },
      "returns": [
        "members"
      ]
    },
    {
      "name": "search_episode_chunks_embed",
      "parameters": {
        "k": "I32",
        "query": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "find_consol_semantic_transitions_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "transitions"
      ]
    },
    {
      "name": "find_cue_by_episode",
      "parameters": {
        "gid": "String",
        "ep_id": "String"
      },
      "returns": [
        "cues"
      ]
    },
    {
      "name": "find_adjudications_by_episode",
      "parameters": {
        "ep_id": "String",
        "gid": "String"
      },
      "returns": [
        "requests"
      ]
    },
    {
      "name": "find_identity_core_entities",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "link_episode_chunk",
      "parameters": {
        "episode_id": "ID",
        "chunk_id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "link_schema_member",
      "parameters": {
        "member_id": "ID",
        "entity_id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "search_episode_chunks_embed_filtered",
      "parameters": {
        "gid": "String",
        "query": "String",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "find_consol_decision_outcomes_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "outcomes"
      ]
    },
    {
      "name": "find_entities_by_name",
      "parameters": {
        "name_query": "String",
        "gid": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "search_episodes_bm25",
      "parameters": {
        "k": "I32",
        "query": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "find_entities_by_name_and_type",
      "parameters": {
        "name_query": "String",
        "etype": "String",
        "gid": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "add_graph_embed_vector",
      "parameters": {
        "entity_id": "String",
        "vec": "Array(F64)",
        "model_version": "String",
        "method": "String",
        "group_id": "String"
      },
      "returns": [
        "v"
      ]
    },
    {
      "name": "delete_graph_embed_vector",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "find_consol_replays_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "replays"
      ]
    },
    {
      "name": "find_active_complement_tags",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "tags"
      ]
    },
    {
      "name": "add_episode_vector",
      "parameters": {
        "vec": "Array(F64)",
        "content_type": "String",
        "episode_id": "String",
        "group_id": "String"
      },
      "returns": [
        "v"
      ]
    },
    {
      "name": "find_atlas_snapshots_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "snapshots"
      ]
    },
    {
      "name": "create_episode_chunk_vec",
      "parameters": {
        "group_id": "String",
        "chunk_text": "String",
        "episode_id": "String",
        "chunk_index": "I32",
        "vec": "Array(F64)",
        "content_type": "String"
      },
      "returns": [
        "chunk"
      ]
    },
    {
      "name": "create_atlas_snapshot",
      "parameters": {
        "generated_at": "String",
        "fastest_growing_region_id": "String",
        "truncated": "Boolean",
        "group_id": "String",
        "snapshot_id": "String",
        "total_regions": "I32",
        "represented_entity_count": "I32",
        "represented_edge_count": "I32",
        "displayed_edge_count": "I32",
        "total_relationships": "I32",
        "displayed_node_count": "I32",
        "total_entities": "I32",
        "hottest_region_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_pending_evidence",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "evidence"
      ]
    },
    {
      "name": "find_episodes_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "get_adjudication",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "request"
      ]
    },
    {
      "name": "update_cue",
      "parameters": {
        "contradiction_keys_json": "String",
        "cue_score": "F64",
        "cue_text": "String",
        "cue_version": "I32",
        "discourse_class": "String",
        "first_spans_json": "String",
        "hit_count": "I32",
        "id": "ID",
        "last_feedback_at": "String",
        "last_hit_at": "String",
        "last_projected_at": "String",
        "near_miss_count": "I32",
        "policy_score": "F64",
        "projection_attempts": "I32",
        "projection_priority": "F64",
        "projection_state": "String",
        "quote_spans_json": "String",
        "route_reason": "String",
        "salience_score": "F64",
        "selected_count": "I32",
        "supporting_spans_json": "String",
        "surfaced_count": "I32",
        "temporal_markers_json": "String",
        "updated_at": "String",
        "used_count": "I32"
      },
      "returns": [
        "cue"
      ]
    },
    {
      "name": "update_cue_by_episode",
      "parameters": {
        "contradiction_keys_json": "String",
        "cue_score": "F64",
        "cue_text": "String",
        "cue_version": "I32",
        "discourse_class": "String",
        "ep_id": "String",
        "first_spans_json": "String",
        "gid": "String",
        "hit_count": "I32",
        "last_feedback_at": "String",
        "last_hit_at": "String",
        "last_projected_at": "String",
        "near_miss_count": "I32",
        "policy_score": "F64",
        "projection_attempts": "I32",
        "projection_priority": "F64",
        "projection_state": "String",
        "quote_spans_json": "String",
        "route_reason": "String",
        "salience_score": "F64",
        "selected_count": "I32",
        "supporting_spans_json": "String",
        "surfaced_count": "I32",
        "temporal_markers_json": "String",
        "updated_at": "String",
        "used_count": "I32"
      },
      "returns": [
        "cue"
      ]
    },
    {
      "name": "find_atlas_region_members",
      "parameters": {
        "snap_id": "String",
        "gid": "String",
        "region_id": "String"
      },
      "returns": [
        "members"
      ]
    },
    {
      "name": "search_cues_embed",
      "parameters": {
        "k": "I32",
        "query": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "find_conversation_entities",
      "parameters": {
        "conv_id": "ID"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "update_consol_cycle",
      "parameters": {
        "phase_results_json": "String",
        "completed_at": "F64",
        "total_duration_ms": "F64",
        "error": "String",
        "status": "String",
        "id": "ID"
      },
      "returns": [
        "cycle"
      ]
    },
    {
      "name": "find_intentions_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "intentions"
      ]
    },
    {
      "name": "create_complement_tag",
      "parameters": {
        "group_id": "String",
        "target_id": "String",
        "tag_type": "String",
        "updated_at": "String",
        "created_at": "String",
        "cycle_confirmed": "I32",
        "target_type": "String",
        "cycle_tagged": "I32",
        "score": "F64",
        "cleared": "Boolean"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "get_atlas_snapshot",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "snapshot"
      ]
    },
    {
      "name": "create_consol_replay",
      "parameters": {
        "episode_id": "String",
        "timestamp": "F64",
        "replay_id": "String",
        "cycle_id": "String",
        "new_relationships_found": "I32",
        "group_id": "String",
        "skipped_reason": "String",
        "new_entities_found": "I32",
        "entities_updated": "I32"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_consol_dream_association",
      "parameters": {
        "cycle_id": "String",
        "target_entity_name": "String",
        "source_domain": "String",
        "assoc_id": "String",
        "surprise_score": "F64",
        "group_id": "String",
        "source_entity_id": "String",
        "target_entity_id": "String",
        "target_domain": "String",
        "structural_proximity": "F64",
        "relationship_id": "String",
        "timestamp": "F64",
        "source_entity_name": "String",
        "embedding_similarity": "F64"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "get_entity_neighborhood",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "search_cues_bm25",
      "parameters": {
        "query": "String",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "find_episodes_by_group_limited",
      "parameters": {
        "gid": "String",
        "limit": "I64"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "shortest_path_bfs",
      "parameters": {
        "end": "ID",
        "start": "ID"
      },
      "returns": [
        "path"
      ]
    },
    {
      "name": "create_consol_semantic_transition",
      "parameters": {
        "old_tier": "String",
        "trans_id": "String",
        "entity_coverage": "F64",
        "timestamp": "F64",
        "episode_id": "String",
        "group_id": "String",
        "new_tier": "String",
        "consolidation_cycles": "I32",
        "cycle_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_entities_by_canonical",
      "parameters": {
        "canon": "String",
        "gid": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "create_consol_decision_outcome",
      "parameters": {
        "cycle_id": "String",
        "decision_trace_id": "String",
        "metadata_json": "String",
        "outcome_type": "String",
        "outcome_id": "String",
        "phase": "String",
        "outcome_label": "String",
        "outcome_value": "F64",
        "group_id": "String",
        "timestamp": "F64"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_consol_identifier_reviews_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "reviews"
      ]
    },
    {
      "name": "shortest_path_weighted",
      "parameters": {
        "start": "ID",
        "end": "ID"
      },
      "returns": [
        "path"
      ]
    },
    {
      "name": "search_cue_vectors_filtered",
      "parameters": {
        "gid": "String",
        "vec": "Array(F64)",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "create_adjudication",
      "parameters": {
        "episode_id": "String",
        "ambiguity_tags_json": "String",
        "resolution_payload_json": "String",
        "group_id": "String",
        "evidence_ids_json": "String",
        "selected_text": "String",
        "request_id": "String",
        "resolution_source": "String",
        "created_at": "String",
        "resolved_at": "String",
        "request_reason": "String",
        "attempt_count": "I32",
        "status": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "get_episode",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "episode"
      ]
    },
    {
      "name": "link_episode_entity",
      "parameters": {
        "episode_id": "ID",
        "entity_id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "hard_delete_intention",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "hard_delete_atlas_region",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "hard_delete_adjudication",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "create_conversation_message",
      "parameters": {
        "message_id": "String",
        "parts_json": "String",
        "created_at": "String",
        "role": "String",
        "content": "String",
        "conversation_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "soft_delete_entity",
      "parameters": {
        "id": "ID",
        "deleted_at": "String"
      },
      "returns": [
        "entity"
      ]
    },
    {
      "name": "drop_edge",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "create_episode_chunk_embed",
      "parameters": {
        "episode_id": "String",
        "chunk_index": "I32",
        "content_type": "String",
        "group_id": "String",
        "chunk_text": "String"
      },
      "returns": [
        "chunk"
      ]
    },
    {
      "name": "find_episodes_by_source",
      "parameters": {
        "gid": "String",
        "src": "String"
      },
      "returns": [
        "episodes"
      ]
    },
    {
      "name": "create_intention",
      "parameters": {
        "fire_count": "I32",
        "updated_at": "String",
        "is_deleted": "Boolean",
        "created_at": "String",
        "trigger_text": "String",
        "intention_id": "String",
        "action_text": "String",
        "max_fires": "I32",
        "entity_names_json": "String",
        "group_id": "String",
        "enabled": "Boolean",
        "deleted_at": "String",
        "context_json": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "link_conversation_entity",
      "parameters": {
        "conv_id": "ID",
        "entity_id": "ID"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "delete_atlas_snapshot",
      "parameters": {
        "id": "ID"
      },
      "returns": []
    },
    {
      "name": "search_episode_chunks_filtered",
      "parameters": {
        "k": "I32",
        "vec": "Array(F64)",
        "gid": "String"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "search_entity_vectors",
      "parameters": {
        "vec": "Array(F64)",
        "k": "I32"
      },
      "returns": [
        "results"
      ]
    },
    {
      "name": "create_consol_microglia",
      "parameters": {
        "target_type": "String",
        "cycle_id": "String",
        "target_id": "String",
        "detail": "String",
        "microglia_id": "String",
        "group_id": "String",
        "action": "String",
        "timestamp": "F64",
        "tag_type": "String",
        "score": "F64"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_entities_exact_name",
      "parameters": {
        "gid": "String",
        "name_exact": "String"
      },
      "returns": [
        "entities"
      ]
    },
    {
      "name": "get_outgoing_edges_by_predicate",
      "parameters": {
        "pred": "String",
        "id": "ID"
      },
      "returns": [
        "edges"
      ]
    },
    {
      "name": "create_consol_identifier_review",
      "parameters": {
        "adjusted_similarity": "F64",
        "entity_b_name": "String",
        "metadata_json": "String",
        "entity_b_regime": "String",
        "review_status": "String",
        "timestamp": "F64",
        "entity_a_regime": "String",
        "entity_a_id": "String",
        "entity_b_id": "String",
        "raw_similarity": "F64",
        "decision_reason": "String",
        "review_id": "String",
        "entity_a_name": "String",
        "cycle_id": "String",
        "entity_a_type": "String",
        "canonical_identifier_a": "String",
        "group_id": "String",
        "decision_source": "String",
        "canonical_identifier_b": "String",
        "entity_b_type": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "find_conversations_by_group",
      "parameters": {
        "gid": "String"
      },
      "returns": [
        "conversations"
      ]
    },
    {
      "name": "find_consol_maturations_by_cycle",
      "parameters": {
        "gid": "String",
        "cycle_id": "String"
      },
      "returns": [
        "maturations"
      ]
    },
    {
      "name": "create_consol_inferred_edge",
      "parameters": {
        "target_name": "String",
        "co_occurrence_count": "I32",
        "source_name": "String",
        "infer_type": "String",
        "confidence": "F64",
        "relationship_id": "String",
        "pmi_score": "F64",
        "llm_verdict": "String",
        "group_id": "String",
        "timestamp": "F64",
        "target_id": "String",
        "edge_id": "String",
        "cycle_id": "String",
        "source_id": "String"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "create_consol_schema",
      "parameters": {
        "group_id": "String",
        "timestamp": "F64",
        "action": "String",
        "cycle_id": "String",
        "schema_name": "String",
        "schema_id": "String",
        "schema_entity_id": "String",
        "instance_count": "I32",
        "predicate_count": "I32"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "update_complement_tag",
      "parameters": {
        "id": "ID",
        "cleared": "Boolean",
        "updated_at": "String",
        "cycle_confirmed": "I32"
      },
      "returns": [
        "tag"
      ]
    },
    {
      "name": "get_entity",
      "parameters": {
        "id": "ID"
      },
      "returns": [
        "entity"
      ]
    },
    {
      "name": "update_edge",
      "parameters": {
        "id": "ID",
        "valid_to": "String",
        "weight": "F64",
        "is_expired": "Boolean"
      },
      "returns": [
        "edge"
      ]
    },
    {
      "name": "find_consol_prunes_by_cycle",
      "parameters": {
        "cycle_id": "String",
        "gid": "String"
      },
      "returns": [
        "prunes"
      ]
    },
    {
      "name": "create_consol_merge",
      "parameters": {
        "decision_source": "String",
        "decision_reason": "String",
        "timestamp": "F64",
        "keep_name": "String",
        "group_id": "String",
        "keep_id": "String",
        "remove_id": "String",
        "remove_name": "String",
        "similarity": "F64",
        "cycle_id": "String",
        "relationships_transferred": "I32",
        "merge_id": "String",
        "decision_confidence": "F64"
      },
      "returns": [
        "node"
      ]
    },
    {
      "name": "search_graph_embed_vectors",
      "parameters": {
        "k": "I32",
        "vec": "Array(F64)"
      },
      "returns": [
        "results"
      ]
    }
  ]
}"#.to_string()),
embedding_model: Some("gemini:gemini-embedding-2-preview:RETRIEVAL_DOCUMENT".to_string()),
graphvis_node_label: None,
bm25_field_filters: None,
})
}
pub struct Entity {
    pub name: String,
    pub group_id: String,
    pub entity_type: String,
    pub canonical_identifier: String,
    pub entity_id: String,
    pub summary: String,
    pub attributes_json: String,
    pub created_at: String,
    pub updated_at: String,
    pub is_deleted: bool,
    pub deleted_at: String,
    pub identity_core: bool,
    pub mat_tier: String,
    pub recon_count: i32,
    pub lexical_regime: String,
    pub identifier_label: String,
    pub pii_detected: bool,
    pub pii_categories_json: String,
    pub access_count: i64,
    pub last_accessed: String,
    pub source_episode_ids: String,
    pub evidence_count: i64,
    pub evidence_span_start: String,
    pub evidence_span_end: String,
}

pub struct Episode {
    pub group_id: String,
    pub status: String,
    pub session_id: String,
    pub episode_id: String,
    pub content: String,
    pub source: String,
    pub created_at: String,
    pub updated_at: String,
    pub error: String,
    pub retry_count: i32,
    pub processing_duration_ms: i64,
    pub skipped_meta: bool,
    pub skipped_triage: bool,
    pub encoding_context_json: String,
    pub memory_tier: String,
    pub consolidation_cycles: i32,
    pub entity_coverage: f64,
    pub projection_state: String,
    pub last_projection_reason: String,
    pub last_projected_at: String,
    pub conversation_date: String,
    pub attachments_json: String,
}

pub struct EpisodeCue {
    pub episode_id: String,
    pub group_id: String,
    pub cue_version: i32,
    pub discourse_class: String,
    pub cue_text: String,
    pub supporting_spans_json: String,
    pub temporal_markers_json: String,
    pub quote_spans_json: String,
    pub contradiction_keys_json: String,
    pub first_spans_json: String,
    pub projection_state: String,
    pub cue_score: f64,
    pub salience_score: f64,
    pub projection_priority: f64,
    pub route_reason: String,
    pub hit_count: i32,
    pub surfaced_count: i32,
    pub selected_count: i32,
    pub used_count: i32,
    pub near_miss_count: i32,
    pub policy_score: f64,
    pub projection_attempts: i32,
    pub last_hit_at: String,
    pub last_feedback_at: String,
    pub last_projected_at: String,
    pub created_at: String,
    pub updated_at: String,
}

pub struct Intention {
    pub group_id: String,
    pub intention_id: String,
    pub trigger_text: String,
    pub action_text: String,
    pub entity_names_json: String,
    pub enabled: bool,
    pub fire_count: i32,
    pub max_fires: i32,
    pub created_at: String,
    pub updated_at: String,
    pub deleted_at: String,
    pub is_deleted: bool,
    pub context_json: String,
}

pub struct Evidence {
    pub episode_id: String,
    pub group_id: String,
    pub status: String,
    pub evidence_id: String,
    pub fact_class: String,
    pub confidence: f64,
    pub source_type: String,
    pub extractor_name: String,
    pub payload_json: String,
    pub source_span: String,
    pub signals_json: String,
    pub ambiguity_tags_json: String,
    pub ambiguity_score: f64,
    pub adjudication_request_id: String,
    pub commit_reason: String,
    pub committed_id: String,
    pub deferred_cycles: i32,
    pub created_at: String,
    pub resolved_at: String,
}

pub struct AdjudicationRequest {
    pub episode_id: String,
    pub group_id: String,
    pub status: String,
    pub request_id: String,
    pub ambiguity_tags_json: String,
    pub evidence_ids_json: String,
    pub selected_text: String,
    pub request_reason: String,
    pub resolution_source: String,
    pub resolution_payload_json: String,
    pub attempt_count: i32,
    pub created_at: String,
    pub resolved_at: String,
}

pub struct SchemaMember {
    pub schema_entity_id: String,
    pub group_id: String,
    pub role_label: String,
    pub member_entity_id: String,
}

pub struct Conversation {
    pub group_id: String,
    pub conversation_id: String,
    pub title: String,
    pub session_date: String,
    pub created_at: String,
    pub updated_at: String,
}

pub struct ConversationMessage {
    pub conversation_id: String,
    pub message_id: String,
    pub role: String,
    pub content: String,
    pub parts_json: String,
    pub created_at: String,
}

pub struct AtlasSnapshot {
    pub group_id: String,
    pub snapshot_id: String,
    pub generated_at: String,
    pub represented_entity_count: i32,
    pub represented_edge_count: i32,
    pub displayed_node_count: i32,
    pub displayed_edge_count: i32,
    pub total_entities: i32,
    pub total_relationships: i32,
    pub total_regions: i32,
    pub hottest_region_id: String,
    pub fastest_growing_region_id: String,
    pub truncated: bool,
}

pub struct AtlasRegion {
    pub snapshot_id: String,
    pub group_id: String,
    pub region_id: String,
    pub region_label: String,
    pub subtitle: String,
    pub kind: String,
    pub member_count: i32,
    pub represented_edge_count: i32,
    pub activation_score: f64,
    pub growth_7d: i32,
    pub growth_30d: i32,
    pub dominant_entity_types_json: String,
    pub hub_entity_ids_json: String,
    pub center_entity_id: String,
    pub latest_entity_created_at: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct AtlasRegionEdge {
    pub snapshot_id: String,
    pub group_id: String,
    pub edge_id: String,
    pub source_region_id: String,
    pub target_region_id: String,
    pub weight: f64,
    pub relationship_count: i32,
}

pub struct AtlasRegionMember {
    pub snapshot_id: String,
    pub group_id: String,
    pub region_id: String,
    pub entity_id: String,
}

pub struct ConsolCycle {
    pub group_id: String,
    pub cycle_id: String,
    pub trigger: String,
    pub dry_run: bool,
    pub status: String,
    pub phase_results_json: String,
    pub started_at: f64,
    pub completed_at: f64,
    pub total_duration_ms: f64,
    pub error: String,
}

pub struct ConsolMerge {
    pub cycle_id: String,
    pub group_id: String,
    pub merge_id: String,
    pub keep_id: String,
    pub remove_id: String,
    pub keep_name: String,
    pub remove_name: String,
    pub similarity: f64,
    pub decision_confidence: f64,
    pub decision_source: String,
    pub decision_reason: String,
    pub relationships_transferred: i32,
    pub timestamp: f64,
}

pub struct ConsolIdentifierReview {
    pub cycle_id: String,
    pub group_id: String,
    pub review_id: String,
    pub entity_a_id: String,
    pub entity_b_id: String,
    pub entity_a_name: String,
    pub entity_b_name: String,
    pub entity_a_type: String,
    pub entity_b_type: String,
    pub raw_similarity: f64,
    pub adjusted_similarity: f64,
    pub decision_source: String,
    pub decision_reason: String,
    pub entity_a_regime: String,
    pub entity_b_regime: String,
    pub canonical_identifier_a: String,
    pub canonical_identifier_b: String,
    pub review_status: String,
    pub metadata_json: String,
    pub timestamp: f64,
}

pub struct ConsolInferredEdge {
    pub cycle_id: String,
    pub group_id: String,
    pub edge_id: String,
    pub source_id: String,
    pub target_id: String,
    pub source_name: String,
    pub target_name: String,
    pub co_occurrence_count: i32,
    pub confidence: f64,
    pub infer_type: String,
    pub pmi_score: f64,
    pub llm_verdict: String,
    pub relationship_id: String,
    pub timestamp: f64,
}

pub struct ConsolPrune {
    pub cycle_id: String,
    pub group_id: String,
    pub prune_id: String,
    pub entity_id: String,
    pub entity_name: String,
    pub entity_type: String,
    pub reason: String,
    pub timestamp: f64,
}

pub struct ConsolReindex {
    pub cycle_id: String,
    pub group_id: String,
    pub reindex_id: String,
    pub entity_id: String,
    pub entity_name: String,
    pub source_phase: String,
    pub timestamp: f64,
}

pub struct ConsolReplay {
    pub cycle_id: String,
    pub group_id: String,
    pub replay_id: String,
    pub episode_id: String,
    pub new_entities_found: i32,
    pub new_relationships_found: i32,
    pub entities_updated: i32,
    pub skipped_reason: String,
    pub timestamp: f64,
}

pub struct ConsolDream {
    pub cycle_id: String,
    pub group_id: String,
    pub dream_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub weight_delta: f64,
    pub seed_entity_id: String,
    pub timestamp: f64,
}

pub struct ConsolTriage {
    pub cycle_id: String,
    pub group_id: String,
    pub triage_id: String,
    pub episode_id: String,
    pub score: f64,
    pub decision: String,
    pub score_breakdown_json: String,
    pub timestamp: f64,
}

pub struct ConsolDreamAssociation {
    pub cycle_id: String,
    pub group_id: String,
    pub assoc_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub source_entity_name: String,
    pub target_entity_name: String,
    pub source_domain: String,
    pub target_domain: String,
    pub surprise_score: f64,
    pub embedding_similarity: f64,
    pub structural_proximity: f64,
    pub relationship_id: String,
    pub timestamp: f64,
}

pub struct ConsolGraphEmbed {
    pub cycle_id: String,
    pub group_id: String,
    pub embed_id: String,
    pub method: String,
    pub entities_trained: i32,
    pub dimensions: i32,
    pub training_duration_ms: f64,
    pub full_retrain: bool,
    pub timestamp: f64,
}

pub struct ConsolMaturation {
    pub cycle_id: String,
    pub group_id: String,
    pub mat_id: String,
    pub entity_id: String,
    pub entity_name: String,
    pub old_tier: String,
    pub new_tier: String,
    pub maturity_score: f64,
    pub source_diversity: i32,
    pub temporal_span_days: f64,
    pub relationship_richness: i32,
    pub access_regularity: f64,
    pub timestamp: f64,
}

pub struct ConsolSemanticTransition {
    pub cycle_id: String,
    pub group_id: String,
    pub trans_id: String,
    pub episode_id: String,
    pub old_tier: String,
    pub new_tier: String,
    pub entity_coverage: f64,
    pub consolidation_cycles: i32,
    pub timestamp: f64,
}

pub struct ConsolSchema {
    pub cycle_id: String,
    pub group_id: String,
    pub schema_id: String,
    pub schema_entity_id: String,
    pub schema_name: String,
    pub instance_count: i32,
    pub predicate_count: i32,
    pub action: String,
    pub timestamp: f64,
}

pub struct ConsolDecisionTrace {
    pub cycle_id: String,
    pub group_id: String,
    pub trace_id: String,
    pub phase: String,
    pub candidate_type: String,
    pub candidate_id: String,
    pub decision: String,
    pub decision_source: String,
    pub confidence: f64,
    pub threshold_band: String,
    pub features_json: String,
    pub constraints_json: String,
    pub policy_version: String,
    pub metadata_json: String,
    pub timestamp: f64,
}

pub struct ConsolDecisionOutcome {
    pub cycle_id: String,
    pub group_id: String,
    pub outcome_id: String,
    pub phase: String,
    pub decision_trace_id: String,
    pub outcome_type: String,
    pub outcome_label: String,
    pub outcome_value: f64,
    pub metadata_json: String,
    pub timestamp: f64,
}

pub struct ConsolDistillation {
    pub cycle_id: String,
    pub group_id: String,
    pub distill_id: String,
    pub phase: String,
    pub candidate_type: String,
    pub candidate_id: String,
    pub decision_trace_id: String,
    pub teacher_label: String,
    pub teacher_source: String,
    pub student_decision: String,
    pub student_confidence: f64,
    pub threshold_band: String,
    pub features_json: String,
    pub correct: bool,
    pub metadata_json: String,
    pub timestamp: f64,
}

pub struct ConsolCalibration {
    pub cycle_id: String,
    pub group_id: String,
    pub calibration_id: String,
    pub phase: String,
    pub window_cycles: i32,
    pub total_traces: i32,
    pub labeled_examples: i32,
    pub oracle_examples: i32,
    pub abstain_count: i32,
    pub accuracy: f64,
    pub mean_confidence: f64,
    pub expected_calibration_error: f64,
    pub summary_json: String,
    pub timestamp: f64,
}

pub struct ConsolEvidenceAdj {
    pub cycle_id: String,
    pub group_id: String,
    pub adj_id: String,
    pub evidence_id: String,
    pub action: String,
    pub new_confidence: f64,
    pub reason: String,
    pub timestamp: f64,
}

pub struct ComplementTag {
    pub target_id: String,
    pub group_id: String,
    pub target_type: String,
    pub tag_type: String,
    pub score: f64,
    pub cycle_tagged: i32,
    pub cycle_confirmed: i32,
    pub cleared: bool,
    pub created_at: String,
    pub updated_at: String,
}

pub struct ConsolMicroglia {
    pub cycle_id: String,
    pub group_id: String,
    pub microglia_id: String,
    pub target_type: String,
    pub target_id: String,
    pub action: String,
    pub tag_type: String,
    pub score: f64,
    pub detail: String,
    pub timestamp: f64,
}

pub struct RelatesTo {
    pub from: Entity,
    pub to: Entity,
    pub rel_id: String,
    pub group_id: String,
    pub predicate: String,
    pub weight: f64,
    pub polarity: String,
    pub valid_from: String,
    pub valid_to: String,
    pub is_expired: bool,
    pub created_at: String,
    pub source_episode_id: String,
}

pub struct HasEntity {
    pub from: Episode,
    pub to: Entity,
}

pub struct HasSchemaMember {
    pub from: Entity,
    pub to: SchemaMember,
}

pub struct HasConversationEntity {
    pub from: Conversation,
    pub to: Entity,
}

pub struct HasEpisodeChunk {
    pub from: Episode,
    pub to: EpisodeChunk,
}

pub struct EntityVec {
    pub entity_id: String,
    pub group_id: String,
    pub content_type: String,
    pub embed_provider: String,
    pub embed_model: String,
}

pub struct EpisodeVec {
    pub episode_id: String,
    pub group_id: String,
    pub content_type: String,
}

pub struct CueVec {
    pub episode_id: String,
    pub group_id: String,
    pub content_type: String,
}

pub struct GraphEmbedVec {
    pub entity_id: String,
    pub group_id: String,
    pub method: String,
    pub model_version: String,
}

pub struct EpisodeChunk {
    pub episode_id: String,
    pub group_id: String,
    pub chunk_text: String,
    pub chunk_index: i32,
    pub content_type: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_merges_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_merges_by_cycleMergesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub merge_id: Option<&'a Value>,
    pub keep_id: Option<&'a Value>,
    pub remove_id: Option<&'a Value>,
    pub keep_name: Option<&'a Value>,
    pub remove_name: Option<&'a Value>,
    pub similarity: Option<&'a Value>,
    pub decision_confidence: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub decision_reason: Option<&'a Value>,
    pub relationships_transferred: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_merges_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_merges_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let merges = G::new(&db, &txn, &arena)
.n_from_type("ConsolMerge")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "merges": merges.iter().map(|merge| Find_consol_merges_by_cycleMergesReturnType {
        id: uuid_str(merge.id(), &arena),
        label: merge.label(),
        cycle_id: merge.get_property("cycle_id"),
        group_id: merge.get_property("group_id"),
        merge_id: merge.get_property("merge_id"),
        keep_id: merge.get_property("keep_id"),
        remove_id: merge.get_property("remove_id"),
        keep_name: merge.get_property("keep_name"),
        remove_name: merge.get_property("remove_name"),
        similarity: merge.get_property("similarity"),
        decision_confidence: merge.get_property("decision_confidence"),
        decision_source: merge.get_property("decision_source"),
        decision_reason: merge.get_property("decision_reason"),
        relationships_transferred: merge.get_property("relationships_transferred"),
        timestamp: merge.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_evidenceInput {

pub id: ID,
pub status: String,
pub resolved_at: String,
pub commit_reason: String,
pub committed_id: String
}
#[derive(Serialize, Default)]
pub struct Update_evidenceEvidenceReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub fact_class: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub source_type: Option<&'a Value>,
    pub extractor_name: Option<&'a Value>,
    pub payload_json: Option<&'a Value>,
    pub source_span: Option<&'a Value>,
    pub signals_json: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub ambiguity_score: Option<&'a Value>,
    pub adjudication_request_id: Option<&'a Value>,
    pub commit_reason: Option<&'a Value>,
    pub committed_id: Option<&'a Value>,
    pub deferred_cycles: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_evidence (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_evidenceInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let evidence = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("status", Value::from(&data.status)), ("resolved_at", Value::from(&data.resolved_at)), ("commit_reason", Value::from(&data.commit_reason)), ("committed_id", Value::from(&data.committed_id))])
    .collect_to_obj()?};
let response = json!({
    "evidence": Update_evidenceEvidenceReturnType {
        id: uuid_str(evidence.id(), &arena),
        label: evidence.label(),
        episode_id: evidence.get_property("episode_id"),
        group_id: evidence.get_property("group_id"),
        status: evidence.get_property("status"),
        evidence_id: evidence.get_property("evidence_id"),
        fact_class: evidence.get_property("fact_class"),
        confidence: evidence.get_property("confidence"),
        source_type: evidence.get_property("source_type"),
        extractor_name: evidence.get_property("extractor_name"),
        payload_json: evidence.get_property("payload_json"),
        source_span: evidence.get_property("source_span"),
        signals_json: evidence.get_property("signals_json"),
        ambiguity_tags_json: evidence.get_property("ambiguity_tags_json"),
        ambiguity_score: evidence.get_property("ambiguity_score"),
        adjudication_request_id: evidence.get_property("adjudication_request_id"),
        commit_reason: evidence.get_property("commit_reason"),
        committed_id: evidence.get_property("committed_id"),
        deferred_cycles: evidence.get_property("deferred_cycles"),
        created_at: evidence.get_property("created_at"),
        resolved_at: evidence.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_entity_fullInput {

pub id: ID,
pub name: String,
pub entity_type: String,
pub summary: String,
pub attributes_json: String,
pub updated_at: String,
pub is_deleted: bool,
pub deleted_at: String,
pub identity_core: bool,
pub mat_tier: String,
pub recon_count: i32,
pub lexical_regime: String,
pub canonical_identifier: String,
pub identifier_label: String,
pub pii_detected: bool,
pub pii_categories_json: String,
pub access_count: i64,
pub last_accessed: String,
pub source_episode_ids: String,
pub evidence_count: i64,
pub evidence_span_start: String,
pub evidence_span_end: String
}
#[derive(Serialize, Default)]
pub struct Update_entity_fullEntityReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_entity_full (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_entity_fullInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let entity = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("name", Value::from(&data.name)), ("entity_type", Value::from(&data.entity_type)), ("summary", Value::from(&data.summary)), ("attributes_json", Value::from(&data.attributes_json)), ("updated_at", Value::from(&data.updated_at)), ("is_deleted", Value::from(&data.is_deleted)), ("deleted_at", Value::from(&data.deleted_at)), ("identity_core", Value::from(&data.identity_core)), ("mat_tier", Value::from(&data.mat_tier)), ("recon_count", Value::from(&data.recon_count)), ("lexical_regime", Value::from(&data.lexical_regime)), ("canonical_identifier", Value::from(&data.canonical_identifier)), ("identifier_label", Value::from(&data.identifier_label)), ("pii_detected", Value::from(&data.pii_detected)), ("pii_categories_json", Value::from(&data.pii_categories_json)), ("access_count", Value::from(&data.access_count)), ("last_accessed", Value::from(&data.last_accessed)), ("source_episode_ids", Value::from(&data.source_episode_ids)), ("evidence_count", Value::from(&data.evidence_count)), ("evidence_span_start", Value::from(&data.evidence_span_start)), ("evidence_span_end", Value::from(&data.evidence_span_end))])
    .collect_to_obj()?};
let response = json!({
    "entity": Update_entity_fullEntityReturnType {
        id: uuid_str(entity.id(), &arena),
        label: entity.label(),
        name: entity.get_property("name"),
        group_id: entity.get_property("group_id"),
        entity_type: entity.get_property("entity_type"),
        canonical_identifier: entity.get_property("canonical_identifier"),
        entity_id: entity.get_property("entity_id"),
        summary: entity.get_property("summary"),
        attributes_json: entity.get_property("attributes_json"),
        created_at: entity.get_property("created_at"),
        updated_at: entity.get_property("updated_at"),
        is_deleted: entity.get_property("is_deleted"),
        deleted_at: entity.get_property("deleted_at"),
        identity_core: entity.get_property("identity_core"),
        mat_tier: entity.get_property("mat_tier"),
        recon_count: entity.get_property("recon_count"),
        lexical_regime: entity.get_property("lexical_regime"),
        identifier_label: entity.get_property("identifier_label"),
        pii_detected: entity.get_property("pii_detected"),
        pii_categories_json: entity.get_property("pii_categories_json"),
        access_count: entity.get_property("access_count"),
        last_accessed: entity.get_property("last_accessed"),
        source_episode_ids: entity.get_property("source_episode_ids"),
        evidence_count: entity.get_property("evidence_count"),
        evidence_span_start: entity.get_property("evidence_span_start"),
        evidence_span_end: entity.get_property("evidence_span_end"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_cycles_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_cycles_by_cycleCyclesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub cycle_id: Option<&'a Value>,
    pub trigger: Option<&'a Value>,
    pub dry_run: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub phase_results_json: Option<&'a Value>,
    pub started_at: Option<&'a Value>,
    pub completed_at: Option<&'a Value>,
    pub total_duration_ms: Option<&'a Value>,
    pub error: Option<&'a Value>,
}

#[handler]
pub fn find_consol_cycles_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_cycles_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let cycles = G::new(&db, &txn, &arena)
.n_from_type("ConsolCycle")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "cycles": cycles.iter().map(|cycle| Find_consol_cycles_by_cycleCyclesReturnType {
        id: uuid_str(cycle.id(), &arena),
        label: cycle.label(),
        group_id: cycle.get_property("group_id"),
        cycle_id: cycle.get_property("cycle_id"),
        trigger: cycle.get_property("trigger"),
        dry_run: cycle.get_property("dry_run"),
        status: cycle.get_property("status"),
        phase_results_json: cycle.get_property("phase_results_json"),
        started_at: cycle.get_property("started_at"),
        completed_at: cycle.get_property("completed_at"),
        total_duration_ms: cycle.get_property("total_duration_ms"),
        error: cycle.get_property("error"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_cycles_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_cycles_by_groupCyclesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub cycle_id: Option<&'a Value>,
    pub trigger: Option<&'a Value>,
    pub dry_run: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub phase_results_json: Option<&'a Value>,
    pub started_at: Option<&'a Value>,
    pub completed_at: Option<&'a Value>,
    pub total_duration_ms: Option<&'a Value>,
    pub error: Option<&'a Value>,
}

#[handler]
pub fn find_consol_cycles_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_cycles_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let cycles = G::new(&db, &txn, &arena)
.n_from_type("ConsolCycle")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "cycles": cycles.iter().map(|cycle| Find_consol_cycles_by_groupCyclesReturnType {
        id: uuid_str(cycle.id(), &arena),
        label: cycle.label(),
        group_id: cycle.get_property("group_id"),
        cycle_id: cycle.get_property("cycle_id"),
        trigger: cycle.get_property("trigger"),
        dry_run: cycle.get_property("dry_run"),
        status: cycle.get_property("status"),
        phase_results_json: cycle.get_property("phase_results_json"),
        started_at: cycle.get_property("started_at"),
        completed_at: cycle.get_property("completed_at"),
        total_duration_ms: cycle.get_property("total_duration_ms"),
        error: cycle.get_property("error"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_cycleInput {

pub cycle_id: String,
pub group_id: String,
pub trigger: String,
pub dry_run: bool,
pub status: String,
pub phase_results_json: String,
pub started_at: f64,
pub completed_at: f64,
pub total_duration_ms: f64,
pub error: String
}
#[derive(Serialize, Default)]
pub struct Create_consol_cycleNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub cycle_id: Option<&'a Value>,
    pub trigger: Option<&'a Value>,
    pub dry_run: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub phase_results_json: Option<&'a Value>,
    pub started_at: Option<&'a Value>,
    pub completed_at: Option<&'a Value>,
    pub total_duration_ms: Option<&'a Value>,
    pub error: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolCycle", Some(ImmutablePropertiesMap::new(10, vec![("status", Value::from(&data.status)), ("group_id", Value::from(&data.group_id)), ("trigger", Value::from(&data.trigger)), ("phase_results_json", Value::from(&data.phase_results_json)), ("error", Value::from(&data.error)), ("started_at", Value::from(&data.started_at)), ("completed_at", Value::from(&data.completed_at)), ("total_duration_ms", Value::from(&data.total_duration_ms)), ("cycle_id", Value::from(&data.cycle_id)), ("dry_run", Value::from(&data.dry_run))].into_iter(), &arena)), Some(&["group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_cycleNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        group_id: node.get_property("group_id"),
        cycle_id: node.get_property("cycle_id"),
        trigger: node.get_property("trigger"),
        dry_run: node.get_property("dry_run"),
        status: node.get_property("status"),
        phase_results_json: node.get_property("phase_results_json"),
        started_at: node.get_property("started_at"),
        completed_at: node.get_property("completed_at"),
        total_duration_ms: node.get_property("total_duration_ms"),
        error: node.get_property("error"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_atlas_region_edgesInput {

pub snap_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_atlas_region_edgesEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub edge_id: Option<&'a Value>,
    pub source_region_id: Option<&'a Value>,
    pub target_region_id: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub relationship_count: Option<&'a Value>,
}

#[handler]
pub fn find_atlas_region_edges (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_atlas_region_edgesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_type("AtlasRegionEdge")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("snapshot_id")
                    .map_or(false, |v| *v == data.snap_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Find_atlas_region_edgesEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        snapshot_id: edge.get_property("snapshot_id"),
        group_id: edge.get_property("group_id"),
        edge_id: edge.get_property("edge_id"),
        source_region_id: edge.get_property("source_region_id"),
        target_region_id: edge.get_property("target_region_id"),
        weight: edge.get_property("weight"),
        relationship_count: edge.get_property("relationship_count"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_incoming_neighborsInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_incoming_neighborsNeighborsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_incoming_neighbors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_incoming_neighborsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let neighbors = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.in_node("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "neighbors": neighbors.iter().map(|neighbor| Get_incoming_neighborsNeighborsReturnType {
        id: uuid_str(neighbor.id(), &arena),
        label: neighbor.label(),
        name: neighbor.get_property("name"),
        group_id: neighbor.get_property("group_id"),
        entity_type: neighbor.get_property("entity_type"),
        canonical_identifier: neighbor.get_property("canonical_identifier"),
        entity_id: neighbor.get_property("entity_id"),
        summary: neighbor.get_property("summary"),
        attributes_json: neighbor.get_property("attributes_json"),
        created_at: neighbor.get_property("created_at"),
        updated_at: neighbor.get_property("updated_at"),
        is_deleted: neighbor.get_property("is_deleted"),
        deleted_at: neighbor.get_property("deleted_at"),
        identity_core: neighbor.get_property("identity_core"),
        mat_tier: neighbor.get_property("mat_tier"),
        recon_count: neighbor.get_property("recon_count"),
        lexical_regime: neighbor.get_property("lexical_regime"),
        identifier_label: neighbor.get_property("identifier_label"),
        pii_detected: neighbor.get_property("pii_detected"),
        pii_categories_json: neighbor.get_property("pii_categories_json"),
        access_count: neighbor.get_property("access_count"),
        last_accessed: neighbor.get_property("last_accessed"),
        source_episode_ids: neighbor.get_property("source_episode_ids"),
        evidence_count: neighbor.get_property("evidence_count"),
        evidence_span_start: neighbor.get_property("evidence_span_start"),
        evidence_span_end: neighbor.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_consol_cycleInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_consol_cycleCycleReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub cycle_id: Option<&'a Value>,
    pub trigger: Option<&'a Value>,
    pub dry_run: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub phase_results_json: Option<&'a Value>,
    pub started_at: Option<&'a Value>,
    pub completed_at: Option<&'a Value>,
    pub total_duration_ms: Option<&'a Value>,
    pub error: Option<&'a Value>,
}

#[handler]
pub fn get_consol_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_consol_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let cycle = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "cycle": Get_consol_cycleCycleReturnType {
        id: uuid_str(cycle.id(), &arena),
        label: cycle.label(),
        group_id: cycle.get_property("group_id"),
        cycle_id: cycle.get_property("cycle_id"),
        trigger: cycle.get_property("trigger"),
        dry_run: cycle.get_property("dry_run"),
        status: cycle.get_property("status"),
        phase_results_json: cycle.get_property("phase_results_json"),
        started_at: cycle.get_property("started_at"),
        completed_at: cycle.get_property("completed_at"),
        total_duration_ms: cycle.get_property("total_duration_ms"),
        error: cycle.get_property("error"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_atlas_region_memberInput {

pub snapshot_id: String,
pub group_id: String,
pub region_id: String,
pub entity_id: String
}
#[derive(Serialize, Default)]
pub struct Create_atlas_region_memberNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub region_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_atlas_region_member (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_atlas_region_memberInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("AtlasRegionMember", Some(ImmutablePropertiesMap::new(4, vec![("snapshot_id", Value::from(&data.snapshot_id)), ("group_id", Value::from(&data.group_id)), ("region_id", Value::from(&data.region_id)), ("entity_id", Value::from(&data.entity_id))].into_iter(), &arena)), Some(&["snapshot_id", "group_id", "region_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_atlas_region_memberNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        snapshot_id: node.get_property("snapshot_id"),
        group_id: node.get_property("group_id"),
        region_id: node.get_property("region_id"),
        entity_id: node.get_property("entity_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_calibrationInput {

pub calibration_id: String,
pub cycle_id: String,
pub group_id: String,
pub phase: String,
pub window_cycles: i32,
pub total_traces: i32,
pub labeled_examples: i32,
pub oracle_examples: i32,
pub abstain_count: i32,
pub accuracy: f64,
pub mean_confidence: f64,
pub expected_calibration_error: f64,
pub summary_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_calibrationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub calibration_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub window_cycles: Option<&'a Value>,
    pub total_traces: Option<&'a Value>,
    pub labeled_examples: Option<&'a Value>,
    pub oracle_examples: Option<&'a Value>,
    pub abstain_count: Option<&'a Value>,
    pub accuracy: Option<&'a Value>,
    pub mean_confidence: Option<&'a Value>,
    pub expected_calibration_error: Option<&'a Value>,
    pub summary_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_calibration (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_calibrationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolCalibration", Some(ImmutablePropertiesMap::new(14, vec![("abstain_count", Value::from(&data.abstain_count)), ("summary_json", Value::from(&data.summary_json)), ("phase", Value::from(&data.phase)), ("calibration_id", Value::from(&data.calibration_id)), ("timestamp", Value::from(&data.timestamp)), ("window_cycles", Value::from(&data.window_cycles)), ("group_id", Value::from(&data.group_id)), ("expected_calibration_error", Value::from(&data.expected_calibration_error)), ("labeled_examples", Value::from(&data.labeled_examples)), ("total_traces", Value::from(&data.total_traces)), ("accuracy", Value::from(&data.accuracy)), ("cycle_id", Value::from(&data.cycle_id)), ("mean_confidence", Value::from(&data.mean_confidence)), ("oracle_examples", Value::from(&data.oracle_examples))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_calibrationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        calibration_id: node.get_property("calibration_id"),
        phase: node.get_property("phase"),
        window_cycles: node.get_property("window_cycles"),
        total_traces: node.get_property("total_traces"),
        labeled_examples: node.get_property("labeled_examples"),
        oracle_examples: node.get_property("oracle_examples"),
        abstain_count: node.get_property("abstain_count"),
        accuracy: node.get_property("accuracy"),
        mean_confidence: node.get_property("mean_confidence"),
        expected_calibration_error: node.get_property("expected_calibration_error"),
        summary_json: node.get_property("summary_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_episode_fullInput {

pub id: ID,
pub status: String,
pub updated_at: String,
pub error: String,
pub retry_count: i32,
pub processing_duration_ms: i64,
pub content: String,
pub skipped_meta: bool,
pub skipped_triage: bool,
pub encoding_context_json: String,
pub memory_tier: String,
pub consolidation_cycles: i32,
pub entity_coverage: f64,
pub projection_state: String,
pub last_projection_reason: String,
pub last_projected_at: String,
pub conversation_date: String,
pub attachments_json: String
}
#[derive(Serialize, Default)]
pub struct Update_episode_fullEpisodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_episode_full (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_episode_fullInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let episode = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("status", Value::from(&data.status)), ("updated_at", Value::from(&data.updated_at)), ("error", Value::from(&data.error)), ("retry_count", Value::from(&data.retry_count)), ("processing_duration_ms", Value::from(&data.processing_duration_ms)), ("content", Value::from(&data.content)), ("skipped_meta", Value::from(&data.skipped_meta)), ("skipped_triage", Value::from(&data.skipped_triage)), ("encoding_context_json", Value::from(&data.encoding_context_json)), ("memory_tier", Value::from(&data.memory_tier)), ("consolidation_cycles", Value::from(&data.consolidation_cycles)), ("entity_coverage", Value::from(&data.entity_coverage)), ("projection_state", Value::from(&data.projection_state)), ("last_projection_reason", Value::from(&data.last_projection_reason)), ("last_projected_at", Value::from(&data.last_projected_at)), ("conversation_date", Value::from(&data.conversation_date)), ("attachments_json", Value::from(&data.attachments_json))])
    .collect_to_obj()?};
let response = json!({
    "episode": Update_episode_fullEpisodeReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_distillations_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_distillations_by_cycleDistillationsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub distill_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub candidate_type: Option<&'a Value>,
    pub candidate_id: Option<&'a Value>,
    pub decision_trace_id: Option<&'a Value>,
    pub teacher_label: Option<&'a Value>,
    pub teacher_source: Option<&'a Value>,
    pub student_decision: Option<&'a Value>,
    pub student_confidence: Option<&'a Value>,
    pub threshold_band: Option<&'a Value>,
    pub features_json: Option<&'a Value>,
    pub correct: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_distillations_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_distillations_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let distillations = G::new(&db, &txn, &arena)
.n_from_type("ConsolDistillation")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "distillations": distillations.iter().map(|distillation| Find_consol_distillations_by_cycleDistillationsReturnType {
        id: uuid_str(distillation.id(), &arena),
        label: distillation.label(),
        cycle_id: distillation.get_property("cycle_id"),
        group_id: distillation.get_property("group_id"),
        distill_id: distillation.get_property("distill_id"),
        phase: distillation.get_property("phase"),
        candidate_type: distillation.get_property("candidate_type"),
        candidate_id: distillation.get_property("candidate_id"),
        decision_trace_id: distillation.get_property("decision_trace_id"),
        teacher_label: distillation.get_property("teacher_label"),
        teacher_source: distillation.get_property("teacher_source"),
        student_decision: distillation.get_property("student_decision"),
        student_confidence: distillation.get_property("student_confidence"),
        threshold_band: distillation.get_property("threshold_band"),
        features_json: distillation.get_property("features_json"),
        correct: distillation.get_property("correct"),
        metadata_json: distillation.get_property("metadata_json"),
        timestamp: distillation.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_pruneInput {

pub prune_id: String,
pub cycle_id: String,
pub group_id: String,
pub entity_id: String,
pub entity_name: String,
pub entity_type: String,
pub reason: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_pruneNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub prune_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_prune (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_pruneInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolPrune", Some(ImmutablePropertiesMap::new(8, vec![("group_id", Value::from(&data.group_id)), ("entity_id", Value::from(&data.entity_id)), ("timestamp", Value::from(&data.timestamp)), ("prune_id", Value::from(&data.prune_id)), ("entity_type", Value::from(&data.entity_type)), ("reason", Value::from(&data.reason)), ("entity_name", Value::from(&data.entity_name)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_pruneNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        prune_id: node.get_property("prune_id"),
        entity_id: node.get_property("entity_id"),
        entity_name: node.get_property("entity_name"),
        entity_type: node.get_property("entity_type"),
        reason: node.get_property("reason"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_complement_tags_by_targetInput {

pub target_id: String
}
#[derive(Serialize, Default)]
pub struct Find_complement_tags_by_targetTagsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_complement_tags_by_target (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_complement_tags_by_targetInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let tags = G::new(&db, &txn, &arena)
.n_from_type("ComplementTag")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("target_id")
                    .map_or(false, |v| *v == data.target_id.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "tags": tags.iter().map(|tag| Find_complement_tags_by_targetTagsReturnType {
        id: uuid_str(tag.id(), &arena),
        label: tag.label(),
        target_id: tag.get_property("target_id"),
        group_id: tag.get_property("group_id"),
        target_type: tag.get_property("target_type"),
        tag_type: tag.get_property("tag_type"),
        cycle_tagged: tag.get_property("cycle_tagged"),
        cycle_confirmed: tag.get_property("cycle_confirmed"),
        cleared: tag.get_property("cleared"),
        created_at: tag.get_property("created_at"),
        updated_at: tag.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct delete_conversationInput {

pub id: ID
}
#[handler(is_write)]
pub fn delete_conversation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<delete_conversationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_schema_memberInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_schema_member (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_schema_memberInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_conversation_messageInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_conversation_message (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_conversation_messageInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_confirmed_complement_tagsInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_confirmed_complement_tagsTagsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_confirmed_complement_tags (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_confirmed_complement_tagsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let tags = G::new(&db, &txn, &arena)
.n_from_type("ComplementTag")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("cleared")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "tags": tags.iter().map(|tag| Find_confirmed_complement_tagsTagsReturnType {
        id: uuid_str(tag.id(), &arena),
        label: tag.label(),
        target_id: tag.get_property("target_id"),
        group_id: tag.get_property("group_id"),
        target_type: tag.get_property("target_type"),
        tag_type: tag.get_property("tag_type"),
        cycle_tagged: tag.get_property("cycle_tagged"),
        cycle_confirmed: tag.get_property("cycle_confirmed"),
        cleared: tag.get_property("cleared"),
        created_at: tag.get_property("created_at"),
        updated_at: tag.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_microglias_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_microglias_by_cycleRecordsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub microglia_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub target_id: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub detail: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_microglias_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_microglias_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let records = G::new(&db, &txn, &arena)
.n_from_type("ConsolMicroglia")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "records": records.iter().map(|record| Find_consol_microglias_by_cycleRecordsReturnType {
        id: uuid_str(record.id(), &arena),
        label: record.label(),
        cycle_id: record.get_property("cycle_id"),
        group_id: record.get_property("group_id"),
        microglia_id: record.get_property("microglia_id"),
        target_type: record.get_property("target_type"),
        target_id: record.get_property("target_id"),
        action: record.get_property("action"),
        tag_type: record.get_property("tag_type"),
        detail: record.get_property("detail"),
        timestamp: record.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_conversationInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_conversationConversationReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub conversation_id: Option<&'a Value>,
    pub title: Option<&'a Value>,
    pub session_date: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn get_conversation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_conversationInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let conversation = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "conversation": Get_conversationConversationReturnType {
        id: uuid_str(conversation.id(), &arena),
        label: conversation.label(),
        group_id: conversation.get_property("group_id"),
        conversation_id: conversation.get_property("conversation_id"),
        title: conversation.get_property("title"),
        session_date: conversation.get_property("session_date"),
        created_at: conversation.get_property("created_at"),
        updated_at: conversation.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_dream_associations_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_dream_associations_by_cycleAssocsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub assoc_id: Option<&'a Value>,
    pub source_entity_id: Option<&'a Value>,
    pub target_entity_id: Option<&'a Value>,
    pub source_entity_name: Option<&'a Value>,
    pub target_entity_name: Option<&'a Value>,
    pub source_domain: Option<&'a Value>,
    pub target_domain: Option<&'a Value>,
    pub surprise_score: Option<&'a Value>,
    pub embedding_similarity: Option<&'a Value>,
    pub structural_proximity: Option<&'a Value>,
    pub relationship_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_dream_associations_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_dream_associations_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let assocs = G::new(&db, &txn, &arena)
.n_from_type("ConsolDreamAssociation")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "assocs": assocs.iter().map(|assoc| Find_consol_dream_associations_by_cycleAssocsReturnType {
        id: uuid_str(assoc.id(), &arena),
        label: assoc.label(),
        cycle_id: assoc.get_property("cycle_id"),
        group_id: assoc.get_property("group_id"),
        assoc_id: assoc.get_property("assoc_id"),
        source_entity_id: assoc.get_property("source_entity_id"),
        target_entity_id: assoc.get_property("target_entity_id"),
        source_entity_name: assoc.get_property("source_entity_name"),
        target_entity_name: assoc.get_property("target_entity_name"),
        source_domain: assoc.get_property("source_domain"),
        target_domain: assoc.get_property("target_domain"),
        surprise_score: assoc.get_property("surprise_score"),
        embedding_similarity: assoc.get_property("embedding_similarity"),
        structural_proximity: assoc.get_property("structural_proximity"),
        relationship_id: assoc.get_property("relationship_id"),
        timestamp: assoc.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_atlas_regionInput {

pub snapshot_id: String,
pub group_id: String,
pub region_id: String,
pub region_label: String,
pub subtitle: String,
pub kind: String,
pub member_count: i32,
pub represented_edge_count: i32,
pub activation_score: f64,
pub growth_7d: i32,
pub growth_30d: i32,
pub dominant_entity_types_json: String,
pub hub_entity_ids_json: String,
pub center_entity_id: String,
pub latest_entity_created_at: String,
pub x: f64,
pub y: f64,
pub z: f64
}
#[derive(Serialize, Default)]
pub struct Create_atlas_regionNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub region_id: Option<&'a Value>,
    pub region_label: Option<&'a Value>,
    pub subtitle: Option<&'a Value>,
    pub kind: Option<&'a Value>,
    pub member_count: Option<&'a Value>,
    pub represented_edge_count: Option<&'a Value>,
    pub activation_score: Option<&'a Value>,
    pub growth_7d: Option<&'a Value>,
    pub growth_30d: Option<&'a Value>,
    pub dominant_entity_types_json: Option<&'a Value>,
    pub hub_entity_ids_json: Option<&'a Value>,
    pub center_entity_id: Option<&'a Value>,
    pub latest_entity_created_at: Option<&'a Value>,
    pub x: Option<&'a Value>,
    pub y: Option<&'a Value>,
    pub z: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_atlas_region (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_atlas_regionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("AtlasRegion", Some(ImmutablePropertiesMap::new(18, vec![("dominant_entity_types_json", Value::from(&data.dominant_entity_types_json)), ("center_entity_id", Value::from(&data.center_entity_id)), ("growth_7d", Value::from(&data.growth_7d)), ("represented_edge_count", Value::from(&data.represented_edge_count)), ("member_count", Value::from(&data.member_count)), ("x", Value::from(&data.x)), ("hub_entity_ids_json", Value::from(&data.hub_entity_ids_json)), ("z", Value::from(&data.z)), ("growth_30d", Value::from(&data.growth_30d)), ("subtitle", Value::from(&data.subtitle)), ("latest_entity_created_at", Value::from(&data.latest_entity_created_at)), ("activation_score", Value::from(&data.activation_score)), ("y", Value::from(&data.y)), ("kind", Value::from(&data.kind)), ("region_id", Value::from(&data.region_id)), ("group_id", Value::from(&data.group_id)), ("region_label", Value::from(&data.region_label)), ("snapshot_id", Value::from(&data.snapshot_id))].into_iter(), &arena)), Some(&["snapshot_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_atlas_regionNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        snapshot_id: node.get_property("snapshot_id"),
        group_id: node.get_property("group_id"),
        region_id: node.get_property("region_id"),
        region_label: node.get_property("region_label"),
        subtitle: node.get_property("subtitle"),
        kind: node.get_property("kind"),
        member_count: node.get_property("member_count"),
        represented_edge_count: node.get_property("represented_edge_count"),
        activation_score: node.get_property("activation_score"),
        growth_7d: node.get_property("growth_7d"),
        growth_30d: node.get_property("growth_30d"),
        dominant_entity_types_json: node.get_property("dominant_entity_types_json"),
        hub_entity_ids_json: node.get_property("hub_entity_ids_json"),
        center_entity_id: node.get_property("center_entity_id"),
        latest_entity_created_at: node.get_property("latest_entity_created_at"),
        x: node.get_property("x"),
        y: node.get_property("y"),
        z: node.get_property("z"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_episodes_by_statusInput {

pub gid: String,
pub st: String
}
#[derive(Serialize, Default)]
pub struct Find_episodes_by_statusEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn find_episodes_by_status (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_episodes_by_statusInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("status")
                    .map_or(false, |v| *v == data.st.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Find_episodes_by_statusEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_reindexInput {

pub reindex_id: String,
pub cycle_id: String,
pub group_id: String,
pub entity_id: String,
pub entity_name: String,
pub source_phase: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_reindexNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub reindex_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub source_phase: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_reindex (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_reindexInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolReindex", Some(ImmutablePropertiesMap::new(7, vec![("source_phase", Value::from(&data.source_phase)), ("group_id", Value::from(&data.group_id)), ("cycle_id", Value::from(&data.cycle_id)), ("entity_id", Value::from(&data.entity_id)), ("entity_name", Value::from(&data.entity_name)), ("timestamp", Value::from(&data.timestamp)), ("reindex_id", Value::from(&data.reindex_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_reindexNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        reindex_id: node.get_property("reindex_id"),
        entity_id: node.get_property("entity_id"),
        entity_name: node.get_property("entity_name"),
        source_phase: node.get_property("source_phase"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_schema_memberInput {

pub schema_entity_id: String,
pub group_id: String,
pub role_label: String,
pub member_entity_id: String
}
#[derive(Serialize, Default)]
pub struct Create_schema_memberNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub schema_entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub role_label: Option<&'a Value>,
    pub member_entity_id: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_schema_member (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_schema_memberInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("SchemaMember", Some(ImmutablePropertiesMap::new(4, vec![("member_entity_id", Value::from(&data.member_entity_id)), ("group_id", Value::from(&data.group_id)), ("role_label", Value::from(&data.role_label)), ("schema_entity_id", Value::from(&data.schema_entity_id))].into_iter(), &arena)), Some(&["schema_entity_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_schema_memberNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        schema_entity_id: node.get_property("schema_entity_id"),
        group_id: node.get_property("group_id"),
        role_label: node.get_property("role_label"),
        member_entity_id: node.get_property("member_entity_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_incoming_edges_by_predicateInput {

pub id: ID,
pub pred: String
}
#[derive(Serialize, Default)]
pub struct Get_incoming_edges_by_predicateEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_incoming_edges_by_predicate (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_incoming_edges_by_predicateInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.in_e("RelatesTo")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("predicate")
                    .map_or(false, |v| *v == data.pred.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Get_incoming_edges_by_predicateEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_episode_cueInput {

pub episode_id: String,
pub group_id: String,
pub cue_version: i32,
pub discourse_class: String,
pub cue_text: String,
pub supporting_spans_json: String,
pub temporal_markers_json: String,
pub quote_spans_json: String,
pub contradiction_keys_json: String,
pub first_spans_json: String,
pub projection_state: String,
pub cue_score: f64,
pub salience_score: f64,
pub projection_priority: f64,
pub route_reason: String,
pub hit_count: i32,
pub surfaced_count: i32,
pub selected_count: i32,
pub used_count: i32,
pub near_miss_count: i32,
pub policy_score: f64,
pub projection_attempts: i32,
pub last_hit_at: String,
pub last_feedback_at: String,
pub last_projected_at: String,
pub created_at: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Create_episode_cueNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub cue_version: Option<&'a Value>,
    pub discourse_class: Option<&'a Value>,
    pub cue_text: Option<&'a Value>,
    pub supporting_spans_json: Option<&'a Value>,
    pub temporal_markers_json: Option<&'a Value>,
    pub quote_spans_json: Option<&'a Value>,
    pub contradiction_keys_json: Option<&'a Value>,
    pub first_spans_json: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub cue_score: Option<&'a Value>,
    pub salience_score: Option<&'a Value>,
    pub projection_priority: Option<&'a Value>,
    pub route_reason: Option<&'a Value>,
    pub hit_count: Option<&'a Value>,
    pub surfaced_count: Option<&'a Value>,
    pub selected_count: Option<&'a Value>,
    pub used_count: Option<&'a Value>,
    pub near_miss_count: Option<&'a Value>,
    pub policy_score: Option<&'a Value>,
    pub projection_attempts: Option<&'a Value>,
    pub last_hit_at: Option<&'a Value>,
    pub last_feedback_at: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_episode_cue (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_episode_cueInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("EpisodeCue", Some(ImmutablePropertiesMap::new(27, vec![("episode_id", Value::from(&data.episode_id)), ("group_id", Value::from(&data.group_id)), ("cue_version", Value::from(&data.cue_version)), ("discourse_class", Value::from(&data.discourse_class)), ("cue_text", Value::from(&data.cue_text)), ("supporting_spans_json", Value::from(&data.supporting_spans_json)), ("temporal_markers_json", Value::from(&data.temporal_markers_json)), ("quote_spans_json", Value::from(&data.quote_spans_json)), ("contradiction_keys_json", Value::from(&data.contradiction_keys_json)), ("first_spans_json", Value::from(&data.first_spans_json)), ("projection_state", Value::from(&data.projection_state)), ("cue_score", Value::from(&data.cue_score)), ("salience_score", Value::from(&data.salience_score)), ("projection_priority", Value::from(&data.projection_priority)), ("route_reason", Value::from(&data.route_reason)), ("hit_count", Value::from(&data.hit_count)), ("surfaced_count", Value::from(&data.surfaced_count)), ("selected_count", Value::from(&data.selected_count)), ("used_count", Value::from(&data.used_count)), ("near_miss_count", Value::from(&data.near_miss_count)), ("policy_score", Value::from(&data.policy_score)), ("projection_attempts", Value::from(&data.projection_attempts)), ("last_hit_at", Value::from(&data.last_hit_at)), ("last_feedback_at", Value::from(&data.last_feedback_at)), ("last_projected_at", Value::from(&data.last_projected_at)), ("created_at", Value::from(&data.created_at)), ("updated_at", Value::from(&data.updated_at))].into_iter(), &arena)), Some(&["episode_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_episode_cueNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        episode_id: node.get_property("episode_id"),
        group_id: node.get_property("group_id"),
        cue_version: node.get_property("cue_version"),
        discourse_class: node.get_property("discourse_class"),
        cue_text: node.get_property("cue_text"),
        supporting_spans_json: node.get_property("supporting_spans_json"),
        temporal_markers_json: node.get_property("temporal_markers_json"),
        quote_spans_json: node.get_property("quote_spans_json"),
        contradiction_keys_json: node.get_property("contradiction_keys_json"),
        first_spans_json: node.get_property("first_spans_json"),
        projection_state: node.get_property("projection_state"),
        cue_score: node.get_property("cue_score"),
        salience_score: node.get_property("salience_score"),
        projection_priority: node.get_property("projection_priority"),
        route_reason: node.get_property("route_reason"),
        hit_count: node.get_property("hit_count"),
        surfaced_count: node.get_property("surfaced_count"),
        selected_count: node.get_property("selected_count"),
        used_count: node.get_property("used_count"),
        near_miss_count: node.get_property("near_miss_count"),
        policy_score: node.get_property("policy_score"),
        projection_attempts: node.get_property("projection_attempts"),
        last_hit_at: node.get_property("last_hit_at"),
        last_feedback_at: node.get_property("last_feedback_at"),
        last_projected_at: node.get_property("last_projected_at"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_evidence_adjInput {

pub adj_id: String,
pub cycle_id: String,
pub group_id: String,
pub evidence_id: String,
pub action: String,
pub new_confidence: f64,
pub reason: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_evidence_adjNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub adj_id: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub new_confidence: Option<&'a Value>,
    pub reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_evidence_adj (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_evidence_adjInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolEvidenceAdj", Some(ImmutablePropertiesMap::new(8, vec![("action", Value::from(&data.action)), ("reason", Value::from(&data.reason)), ("group_id", Value::from(&data.group_id)), ("cycle_id", Value::from(&data.cycle_id)), ("timestamp", Value::from(&data.timestamp)), ("evidence_id", Value::from(&data.evidence_id)), ("new_confidence", Value::from(&data.new_confidence)), ("adj_id", Value::from(&data.adj_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_evidence_adjNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        adj_id: node.get_property("adj_id"),
        evidence_id: node.get_property("evidence_id"),
        action: node.get_property("action"),
        new_confidence: node.get_property("new_confidence"),
        reason: node.get_property("reason"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_entityInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_entityInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_chunks_vecInput {

pub vec: Vec<f64>,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_episode_chunks_vecResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_chunks_vec (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_chunks_vecInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EpisodeChunk", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_chunks_vecResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        chunk_text: result.get_property("chunk_text"),
        chunk_index: result.get_property("chunk_index"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_decision_traceInput {

pub trace_id: String,
pub cycle_id: String,
pub group_id: String,
pub phase: String,
pub candidate_type: String,
pub candidate_id: String,
pub decision: String,
pub decision_source: String,
pub confidence: f64,
pub threshold_band: String,
pub features_json: String,
pub constraints_json: String,
pub policy_version: String,
pub metadata_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_decision_traceNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub trace_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub candidate_type: Option<&'a Value>,
    pub candidate_id: Option<&'a Value>,
    pub decision: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub threshold_band: Option<&'a Value>,
    pub features_json: Option<&'a Value>,
    pub constraints_json: Option<&'a Value>,
    pub policy_version: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_decision_trace (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_decision_traceInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolDecisionTrace", Some(ImmutablePropertiesMap::new(15, vec![("timestamp", Value::from(&data.timestamp)), ("metadata_json", Value::from(&data.metadata_json)), ("confidence", Value::from(&data.confidence)), ("candidate_id", Value::from(&data.candidate_id)), ("candidate_type", Value::from(&data.candidate_type)), ("group_id", Value::from(&data.group_id)), ("decision_source", Value::from(&data.decision_source)), ("features_json", Value::from(&data.features_json)), ("constraints_json", Value::from(&data.constraints_json)), ("trace_id", Value::from(&data.trace_id)), ("threshold_band", Value::from(&data.threshold_band)), ("phase", Value::from(&data.phase)), ("decision", Value::from(&data.decision)), ("policy_version", Value::from(&data.policy_version)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_decision_traceNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        trace_id: node.get_property("trace_id"),
        phase: node.get_property("phase"),
        candidate_type: node.get_property("candidate_type"),
        candidate_id: node.get_property("candidate_id"),
        decision: node.get_property("decision"),
        decision_source: node.get_property("decision_source"),
        confidence: node.get_property("confidence"),
        threshold_band: node.get_property("threshold_band"),
        features_json: node.get_property("features_json"),
        constraints_json: node.get_property("constraints_json"),
        policy_version: node.get_property("policy_version"),
        metadata_json: node.get_property("metadata_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entity_ids_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entity_ids_by_groupEntitiesReturnType<'a> {
    pub entity_id: Option<&'a Value>,
    pub name: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
}

#[handler]
pub fn find_entity_ids_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entity_ids_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entity_ids_by_groupEntitiesReturnType {
        entity_id: entitie.get_property("entity_id"),
        name: entitie.get_property("name"),
        entity_type: entitie.get_property("entity_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_entity_cooccurrencesInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_entity_cooccurrencesCooccurringReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_entity_cooccurrences (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_entity_cooccurrencesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let cooccurring = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.in_node("HasEntity")

.out_node("HasEntity").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "cooccurring": cooccurring.iter().map(|cooccurring| Get_entity_cooccurrencesCooccurringReturnType {
        id: uuid_str(cooccurring.id(), &arena),
        label: cooccurring.label(),
        name: cooccurring.get_property("name"),
        group_id: cooccurring.get_property("group_id"),
        entity_type: cooccurring.get_property("entity_type"),
        canonical_identifier: cooccurring.get_property("canonical_identifier"),
        entity_id: cooccurring.get_property("entity_id"),
        summary: cooccurring.get_property("summary"),
        attributes_json: cooccurring.get_property("attributes_json"),
        created_at: cooccurring.get_property("created_at"),
        updated_at: cooccurring.get_property("updated_at"),
        is_deleted: cooccurring.get_property("is_deleted"),
        deleted_at: cooccurring.get_property("deleted_at"),
        identity_core: cooccurring.get_property("identity_core"),
        mat_tier: cooccurring.get_property("mat_tier"),
        recon_count: cooccurring.get_property("recon_count"),
        lexical_regime: cooccurring.get_property("lexical_regime"),
        identifier_label: cooccurring.get_property("identifier_label"),
        pii_detected: cooccurring.get_property("pii_detected"),
        pii_categories_json: cooccurring.get_property("pii_categories_json"),
        access_count: cooccurring.get_property("access_count"),
        last_accessed: cooccurring.get_property("last_accessed"),
        source_episode_ids: cooccurring.get_property("source_episode_ids"),
        evidence_count: cooccurring.get_property("evidence_count"),
        evidence_span_start: cooccurring.get_property("evidence_span_start"),
        evidence_span_end: cooccurring.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_two_hop_neighborsInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_two_hop_neighborsNeighborsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_two_hop_neighbors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_two_hop_neighborsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let neighbors = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_node("RelatesTo")

.out_node("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "neighbors": neighbors.iter().map(|neighbor| Get_two_hop_neighborsNeighborsReturnType {
        id: uuid_str(neighbor.id(), &arena),
        label: neighbor.label(),
        name: neighbor.get_property("name"),
        group_id: neighbor.get_property("group_id"),
        entity_type: neighbor.get_property("entity_type"),
        canonical_identifier: neighbor.get_property("canonical_identifier"),
        entity_id: neighbor.get_property("entity_id"),
        summary: neighbor.get_property("summary"),
        attributes_json: neighbor.get_property("attributes_json"),
        created_at: neighbor.get_property("created_at"),
        updated_at: neighbor.get_property("updated_at"),
        is_deleted: neighbor.get_property("is_deleted"),
        deleted_at: neighbor.get_property("deleted_at"),
        identity_core: neighbor.get_property("identity_core"),
        mat_tier: neighbor.get_property("mat_tier"),
        recon_count: neighbor.get_property("recon_count"),
        lexical_regime: neighbor.get_property("lexical_regime"),
        identifier_label: neighbor.get_property("identifier_label"),
        pii_detected: neighbor.get_property("pii_detected"),
        pii_categories_json: neighbor.get_property("pii_categories_json"),
        access_count: neighbor.get_property("access_count"),
        last_accessed: neighbor.get_property("last_accessed"),
        source_episode_ids: neighbor.get_property("source_episode_ids"),
        evidence_count: neighbor.get_property("evidence_count"),
        evidence_span_start: neighbor.get_property("evidence_span_start"),
        evidence_span_end: neighbor.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_entityInput {

pub entity_id: String,
pub name: String,
pub group_id: String,
pub entity_type: String,
pub summary: String,
pub attributes_json: String,
pub created_at: String,
pub updated_at: String,
pub is_deleted: bool,
pub deleted_at: String,
pub identity_core: bool,
pub mat_tier: String,
pub recon_count: i32,
pub lexical_regime: String,
pub canonical_identifier: String,
pub identifier_label: String,
pub pii_detected: bool,
pub pii_categories_json: String,
pub access_count: i64,
pub last_accessed: String,
pub source_episode_ids: String,
pub evidence_count: i64,
pub evidence_span_start: String,
pub evidence_span_end: String
}
#[derive(Serialize, Default)]
pub struct Create_entityNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_entityInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("Entity", Some(ImmutablePropertiesMap::new(24, vec![("updated_at", Value::from(&data.updated_at)), ("access_count", Value::from(&data.access_count)), ("name", Value::from(&data.name)), ("entity_type", Value::from(&data.entity_type)), ("lexical_regime", Value::from(&data.lexical_regime)), ("deleted_at", Value::from(&data.deleted_at)), ("canonical_identifier", Value::from(&data.canonical_identifier)), ("pii_detected", Value::from(&data.pii_detected)), ("summary", Value::from(&data.summary)), ("entity_id", Value::from(&data.entity_id)), ("created_at", Value::from(&data.created_at)), ("group_id", Value::from(&data.group_id)), ("attributes_json", Value::from(&data.attributes_json)), ("recon_count", Value::from(&data.recon_count)), ("identifier_label", Value::from(&data.identifier_label)), ("identity_core", Value::from(&data.identity_core)), ("pii_categories_json", Value::from(&data.pii_categories_json)), ("is_deleted", Value::from(&data.is_deleted)), ("last_accessed", Value::from(&data.last_accessed)), ("source_episode_ids", Value::from(&data.source_episode_ids)), ("evidence_count", Value::from(&data.evidence_count)), ("evidence_span_start", Value::from(&data.evidence_span_start)), ("evidence_span_end", Value::from(&data.evidence_span_end)), ("mat_tier", Value::from(&data.mat_tier))].into_iter(), &arena)), Some(&["name", "group_id", "entity_type", "canonical_identifier"])).collect_to_obj()?;
let response = json!({
    "node": Create_entityNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        name: node.get_property("name"),
        group_id: node.get_property("group_id"),
        entity_type: node.get_property("entity_type"),
        canonical_identifier: node.get_property("canonical_identifier"),
        entity_id: node.get_property("entity_id"),
        summary: node.get_property("summary"),
        attributes_json: node.get_property("attributes_json"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
        is_deleted: node.get_property("is_deleted"),
        deleted_at: node.get_property("deleted_at"),
        identity_core: node.get_property("identity_core"),
        mat_tier: node.get_property("mat_tier"),
        recon_count: node.get_property("recon_count"),
        lexical_regime: node.get_property("lexical_regime"),
        identifier_label: node.get_property("identifier_label"),
        pii_detected: node.get_property("pii_detected"),
        pii_categories_json: node.get_property("pii_categories_json"),
        access_count: node.get_property("access_count"),
        last_accessed: node.get_property("last_accessed"),
        source_episode_ids: node.get_property("source_episode_ids"),
        evidence_count: node.get_property("evidence_count"),
        evidence_span_start: node.get_property("evidence_span_start"),
        evidence_span_end: node.get_property("evidence_span_end"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_episodeInput {

pub episode_id: String,
pub group_id: String,
pub content: String,
pub source: String,
pub session_id: String,
pub status: String,
pub created_at: String,
pub updated_at: String,
pub error: String,
pub retry_count: i32,
pub processing_duration_ms: i64,
pub skipped_meta: bool,
pub skipped_triage: bool,
pub encoding_context_json: String,
pub memory_tier: String,
pub consolidation_cycles: i32,
pub entity_coverage: f64,
pub projection_state: String,
pub last_projection_reason: String,
pub last_projected_at: String,
pub conversation_date: String,
pub attachments_json: String
}
#[derive(Serialize, Default)]
pub struct Create_episodeNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("Episode", Some(ImmutablePropertiesMap::new(22, vec![("group_id", Value::from(&data.group_id)), ("status", Value::from(&data.status)), ("source", Value::from(&data.source)), ("consolidation_cycles", Value::from(&data.consolidation_cycles)), ("updated_at", Value::from(&data.updated_at)), ("projection_state", Value::from(&data.projection_state)), ("encoding_context_json", Value::from(&data.encoding_context_json)), ("last_projection_reason", Value::from(&data.last_projection_reason)), ("error", Value::from(&data.error)), ("attachments_json", Value::from(&data.attachments_json)), ("processing_duration_ms", Value::from(&data.processing_duration_ms)), ("skipped_meta", Value::from(&data.skipped_meta)), ("episode_id", Value::from(&data.episode_id)), ("entity_coverage", Value::from(&data.entity_coverage)), ("last_projected_at", Value::from(&data.last_projected_at)), ("session_id", Value::from(&data.session_id)), ("memory_tier", Value::from(&data.memory_tier)), ("conversation_date", Value::from(&data.conversation_date)), ("content", Value::from(&data.content)), ("skipped_triage", Value::from(&data.skipped_triage)), ("retry_count", Value::from(&data.retry_count)), ("created_at", Value::from(&data.created_at))].into_iter(), &arena)), Some(&["group_id", "status", "session_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_episodeNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        group_id: node.get_property("group_id"),
        status: node.get_property("status"),
        session_id: node.get_property("session_id"),
        episode_id: node.get_property("episode_id"),
        content: node.get_property("content"),
        source: node.get_property("source"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
        error: node.get_property("error"),
        retry_count: node.get_property("retry_count"),
        processing_duration_ms: node.get_property("processing_duration_ms"),
        skipped_meta: node.get_property("skipped_meta"),
        skipped_triage: node.get_property("skipped_triage"),
        encoding_context_json: node.get_property("encoding_context_json"),
        memory_tier: node.get_property("memory_tier"),
        consolidation_cycles: node.get_property("consolidation_cycles"),
        entity_coverage: node.get_property("entity_coverage"),
        projection_state: node.get_property("projection_state"),
        last_projection_reason: node.get_property("last_projection_reason"),
        last_projected_at: node.get_property("last_projected_at"),
        conversation_date: node.get_property("conversation_date"),
        attachments_json: node.get_property("attachments_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_dreams_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_dreams_by_cycleDreamsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub dream_id: Option<&'a Value>,
    pub source_entity_id: Option<&'a Value>,
    pub target_entity_id: Option<&'a Value>,
    pub weight_delta: Option<&'a Value>,
    pub seed_entity_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_dreams_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_dreams_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let dreams = G::new(&db, &txn, &arena)
.n_from_type("ConsolDream")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "dreams": dreams.iter().map(|dream| Find_consol_dreams_by_cycleDreamsReturnType {
        id: uuid_str(dream.id(), &arena),
        label: dream.label(),
        cycle_id: dream.get_property("cycle_id"),
        group_id: dream.get_property("group_id"),
        dream_id: dream.get_property("dream_id"),
        source_entity_id: dream.get_property("source_entity_id"),
        target_entity_id: dream.get_property("target_entity_id"),
        weight_delta: dream.get_property("weight_delta"),
        seed_entity_id: dream.get_property("seed_entity_id"),
        timestamp: dream.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct invalidate_edgeInput {

pub id: ID,
pub valid_to: String
}
#[derive(Serialize, Default)]
pub struct Invalidate_edgeEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler(is_write)]
pub fn invalidate_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<invalidate_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = {let update_tr = G::new(&db, &txn, &arena)
.e_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("is_expired", Value::from(true)), ("valid_to", Value::from(&data.valid_to))])
    .collect_to_obj()?};
let response = json!({
    "edge": Invalidate_edgeEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_reindexes_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_reindexes_by_cycleReindexesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub reindex_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub source_phase: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_reindexes_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_reindexes_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let reindexes = G::new(&db, &txn, &arena)
.n_from_type("ConsolReindex")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "reindexes": reindexes.iter().map(|reindexe| Find_consol_reindexes_by_cycleReindexesReturnType {
        id: uuid_str(reindexe.id(), &arena),
        label: reindexe.label(),
        cycle_id: reindexe.get_property("cycle_id"),
        group_id: reindexe.get_property("group_id"),
        reindex_id: reindexe.get_property("reindex_id"),
        entity_id: reindexe.get_property("entity_id"),
        entity_name: reindexe.get_property("entity_name"),
        source_phase: reindexe.get_property("source_phase"),
        timestamp: reindexe.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_atlas_region_edgeInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_atlas_region_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_atlas_region_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_triageInput {

pub triage_id: String,
pub cycle_id: String,
pub group_id: String,
pub episode_id: String,
pub score: f64,
pub decision: String,
pub score_breakdown_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_triageNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub triage_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub decision: Option<&'a Value>,
    pub score_breakdown_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_triage (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_triageInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolTriage", Some(ImmutablePropertiesMap::new(8, vec![("cycle_id", Value::from(&data.cycle_id)), ("triage_id", Value::from(&data.triage_id)), ("score", Value::from(&data.score)), ("episode_id", Value::from(&data.episode_id)), ("timestamp", Value::from(&data.timestamp)), ("group_id", Value::from(&data.group_id)), ("decision", Value::from(&data.decision)), ("score_breakdown_json", Value::from(&data.score_breakdown_json))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_triageNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        triage_id: node.get_property("triage_id"),
        episode_id: node.get_property("episode_id"),
        decision: node.get_property("decision"),
        score_breakdown_json: node.get_property("score_breakdown_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_entities_embedInput {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_entities_embedResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
    pub embed_provider: Option<&'a Value>,
    pub embed_model: Option<&'a Value>,
}

#[handler]
pub fn search_entities_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_entities_embedInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.query);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&__internal_embed_data_0, data.k.clone(), "EntityVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_entities_embedResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        entity_id: result.get_property("entity_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
        embed_provider: result.get_property("embed_provider"),
        embed_model: result.get_property("embed_model"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_cueInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_cue (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_cueInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_episodeInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_edgeInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_edgeEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edge = G::new(&db, &txn, &arena)
.e_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "edge": Get_edgeEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_intentionInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_intentionIntentionReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler]
pub fn get_intention (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_intentionInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let intention = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "intention": Get_intentionIntentionReturnType {
        id: uuid_str(intention.id(), &arena),
        label: intention.label(),
        group_id: intention.get_property("group_id"),
        intention_id: intention.get_property("intention_id"),
        trigger_text: intention.get_property("trigger_text"),
        action_text: intention.get_property("action_text"),
        entity_names_json: intention.get_property("entity_names_json"),
        enabled: intention.get_property("enabled"),
        fire_count: intention.get_property("fire_count"),
        max_fires: intention.get_property("max_fires"),
        created_at: intention.get_property("created_at"),
        updated_at: intention.get_property("updated_at"),
        deleted_at: intention.get_property("deleted_at"),
        is_deleted: intention.get_property("is_deleted"),
        context_json: intention.get_property("context_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_groupEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_groupEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_evidenceInput {

pub evidence_id: String,
pub episode_id: String,
pub group_id: String,
pub status: String,
pub fact_class: String,
pub confidence: f64,
pub source_type: String,
pub extractor_name: String,
pub payload_json: String,
pub source_span: String,
pub signals_json: String,
pub ambiguity_tags_json: String,
pub ambiguity_score: f64,
pub adjudication_request_id: String,
pub commit_reason: String,
pub committed_id: String,
pub deferred_cycles: i32,
pub created_at: String,
pub resolved_at: String
}
#[derive(Serialize, Default)]
pub struct Create_evidenceNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub fact_class: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub source_type: Option<&'a Value>,
    pub extractor_name: Option<&'a Value>,
    pub payload_json: Option<&'a Value>,
    pub source_span: Option<&'a Value>,
    pub signals_json: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub ambiguity_score: Option<&'a Value>,
    pub adjudication_request_id: Option<&'a Value>,
    pub commit_reason: Option<&'a Value>,
    pub committed_id: Option<&'a Value>,
    pub deferred_cycles: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_evidence (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_evidenceInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("Evidence", Some(ImmutablePropertiesMap::new(19, vec![("evidence_id", Value::from(&data.evidence_id)), ("resolved_at", Value::from(&data.resolved_at)), ("payload_json", Value::from(&data.payload_json)), ("commit_reason", Value::from(&data.commit_reason)), ("extractor_name", Value::from(&data.extractor_name)), ("status", Value::from(&data.status)), ("created_at", Value::from(&data.created_at)), ("deferred_cycles", Value::from(&data.deferred_cycles)), ("source_type", Value::from(&data.source_type)), ("adjudication_request_id", Value::from(&data.adjudication_request_id)), ("episode_id", Value::from(&data.episode_id)), ("source_span", Value::from(&data.source_span)), ("committed_id", Value::from(&data.committed_id)), ("confidence", Value::from(&data.confidence)), ("ambiguity_score", Value::from(&data.ambiguity_score)), ("ambiguity_tags_json", Value::from(&data.ambiguity_tags_json)), ("fact_class", Value::from(&data.fact_class)), ("signals_json", Value::from(&data.signals_json)), ("group_id", Value::from(&data.group_id))].into_iter(), &arena)), Some(&["episode_id", "group_id", "status"])).collect_to_obj()?;
let response = json!({
    "node": Create_evidenceNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        episode_id: node.get_property("episode_id"),
        group_id: node.get_property("group_id"),
        status: node.get_property("status"),
        evidence_id: node.get_property("evidence_id"),
        fact_class: node.get_property("fact_class"),
        confidence: node.get_property("confidence"),
        source_type: node.get_property("source_type"),
        extractor_name: node.get_property("extractor_name"),
        payload_json: node.get_property("payload_json"),
        source_span: node.get_property("source_span"),
        signals_json: node.get_property("signals_json"),
        ambiguity_tags_json: node.get_property("ambiguity_tags_json"),
        ambiguity_score: node.get_property("ambiguity_score"),
        adjudication_request_id: node.get_property("adjudication_request_id"),
        commit_reason: node.get_property("commit_reason"),
        committed_id: node.get_property("committed_id"),
        deferred_cycles: node.get_property("deferred_cycles"),
        created_at: node.get_property("created_at"),
        resolved_at: node.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_pending_adjudicationsInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_pending_adjudicationsRequestsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub request_id: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub evidence_ids_json: Option<&'a Value>,
    pub selected_text: Option<&'a Value>,
    pub request_reason: Option<&'a Value>,
    pub resolution_source: Option<&'a Value>,
    pub resolution_payload_json: Option<&'a Value>,
    pub attempt_count: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn find_pending_adjudications (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_pending_adjudicationsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let requests = G::new(&db, &txn, &arena)
.n_from_type("AdjudicationRequest")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("status")
                    .map_or(false, |v| *v == "pending")))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "requests": requests.iter().map(|request| Find_pending_adjudicationsRequestsReturnType {
        id: uuid_str(request.id(), &arena),
        label: request.label(),
        episode_id: request.get_property("episode_id"),
        group_id: request.get_property("group_id"),
        status: request.get_property("status"),
        request_id: request.get_property("request_id"),
        ambiguity_tags_json: request.get_property("ambiguity_tags_json"),
        evidence_ids_json: request.get_property("evidence_ids_json"),
        selected_text: request.get_property("selected_text"),
        request_reason: request.get_property("request_reason"),
        resolution_source: request.get_property("resolution_source"),
        resolution_payload_json: request.get_property("resolution_payload_json"),
        attempt_count: request.get_property("attempt_count"),
        created_at: request.get_property("created_at"),
        resolved_at: request.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_adjudications_by_statusInput {

pub gid: String,
pub st: String
}

#[handler]
pub fn find_adjudications_by_status (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_adjudications_by_statusInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let requests = G::new(&db, &txn, &arena)
.n_from_type("AdjudicationRequest")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("status")
                    .map_or(false, |v| *v == data.st.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "requests": requests.iter().map(|request| Find_pending_adjudicationsRequestsReturnType {
        id: uuid_str(request.id(), &arena),
        label: request.label(),
        episode_id: request.get_property("episode_id"),
        group_id: request.get_property("group_id"),
        status: request.get_property("status"),
        request_id: request.get_property("request_id"),
        ambiguity_tags_json: request.get_property("ambiguity_tags_json"),
        evidence_ids_json: request.get_property("evidence_ids_json"),
        selected_text: request.get_property("selected_text"),
        request_reason: request.get_property("request_reason"),
        resolution_source: request.get_property("resolution_source"),
        resolution_payload_json: request.get_property("resolution_payload_json"),
        attempt_count: request.get_property("attempt_count"),
        created_at: request.get_property("created_at"),
        resolved_at: request.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_distillationInput {

pub distill_id: String,
pub cycle_id: String,
pub group_id: String,
pub phase: String,
pub candidate_type: String,
pub candidate_id: String,
pub decision_trace_id: String,
pub teacher_label: String,
pub teacher_source: String,
pub student_decision: String,
pub student_confidence: f64,
pub threshold_band: String,
pub features_json: String,
pub correct: bool,
pub metadata_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_distillationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub distill_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub candidate_type: Option<&'a Value>,
    pub candidate_id: Option<&'a Value>,
    pub decision_trace_id: Option<&'a Value>,
    pub teacher_label: Option<&'a Value>,
    pub teacher_source: Option<&'a Value>,
    pub student_decision: Option<&'a Value>,
    pub student_confidence: Option<&'a Value>,
    pub threshold_band: Option<&'a Value>,
    pub features_json: Option<&'a Value>,
    pub correct: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_distillation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_distillationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolDistillation", Some(ImmutablePropertiesMap::new(16, vec![("teacher_source", Value::from(&data.teacher_source)), ("candidate_id", Value::from(&data.candidate_id)), ("features_json", Value::from(&data.features_json)), ("student_confidence", Value::from(&data.student_confidence)), ("teacher_label", Value::from(&data.teacher_label)), ("candidate_type", Value::from(&data.candidate_type)), ("cycle_id", Value::from(&data.cycle_id)), ("group_id", Value::from(&data.group_id)), ("timestamp", Value::from(&data.timestamp)), ("distill_id", Value::from(&data.distill_id)), ("student_decision", Value::from(&data.student_decision)), ("phase", Value::from(&data.phase)), ("metadata_json", Value::from(&data.metadata_json)), ("correct", Value::from(&data.correct)), ("decision_trace_id", Value::from(&data.decision_trace_id)), ("threshold_band", Value::from(&data.threshold_band))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_distillationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        distill_id: node.get_property("distill_id"),
        phase: node.get_property("phase"),
        candidate_type: node.get_property("candidate_type"),
        candidate_id: node.get_property("candidate_id"),
        decision_trace_id: node.get_property("decision_trace_id"),
        teacher_label: node.get_property("teacher_label"),
        teacher_source: node.get_property("teacher_source"),
        student_decision: node.get_property("student_decision"),
        student_confidence: node.get_property("student_confidence"),
        threshold_band: node.get_property("threshold_band"),
        features_json: node.get_property("features_json"),
        correct: node.get_property("correct"),
        metadata_json: node.get_property("metadata_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct add_cue_vectorInput {

pub episode_id: String,
pub group_id: String,
pub content_type: String,
pub vec: Vec<f64>
}
#[derive(Serialize, Default)]
pub struct Add_cue_vectorVReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler(is_write)]
pub fn add_cue_vector (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<add_cue_vectorInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let v = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&data.vec, "CueVec", Some(ImmutablePropertiesMap::new(3, vec![("content_type", Value::from(data.content_type.clone())), ("episode_id", Value::from(data.episode_id.clone())), ("group_id", Value::from(data.group_id.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "v": Add_cue_vectorVReturnType {
        id: uuid_str(v.id(), &arena),
        label: v.label(),
        data: v.data(),
        score: v.score(),
        episode_id: v.get_property("episode_id"),
        group_id: v.get_property("group_id"),
        content_type: v.get_property("content_type"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_group_limitedInput {

pub gid: String,
pub limit: i64
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_group_limitedEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_group_limited (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_group_limitedInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            })

.range(0, data.limit.clone()).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_group_limitedEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_conversationInput {

pub conversation_id: String,
pub group_id: String,
pub title: String,
pub session_date: String,
pub created_at: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Create_conversationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub conversation_id: Option<&'a Value>,
    pub title: Option<&'a Value>,
    pub session_date: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_conversation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_conversationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("Conversation", Some(ImmutablePropertiesMap::new(6, vec![("created_at", Value::from(&data.created_at)), ("conversation_id", Value::from(&data.conversation_id)), ("updated_at", Value::from(&data.updated_at)), ("group_id", Value::from(&data.group_id)), ("title", Value::from(&data.title)), ("session_date", Value::from(&data.session_date))].into_iter(), &arena)), Some(&["group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_conversationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        group_id: node.get_property("group_id"),
        conversation_id: node.get_property("conversation_id"),
        title: node.get_property("title"),
        session_date: node.get_property("session_date"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_messages_by_conversationInput {

pub conv_id: String
}
#[derive(Serialize, Default)]
pub struct Find_messages_by_conversationMessagesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub conversation_id: Option<&'a Value>,
    pub message_id: Option<&'a Value>,
    pub role: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub parts_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
}

#[handler]
pub fn find_messages_by_conversation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_messages_by_conversationInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let messages = G::new(&db, &txn, &arena)
.n_from_type("ConversationMessage")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("conversation_id")
                    .map_or(false, |v| *v == data.conv_id.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "messages": messages.iter().map(|message| Find_messages_by_conversationMessagesReturnType {
        id: uuid_str(message.id(), &arena),
        label: message.label(),
        conversation_id: message.get_property("conversation_id"),
        message_id: message.get_property("message_id"),
        role: message.get_property("role"),
        content: message.get_property("content"),
        parts_json: message.get_property("parts_json"),
        created_at: message.get_property("created_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_entity_vectors_filteredInput {

pub vec: Vec<f64>,
pub k: i32,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Search_entity_vectors_filteredResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
    pub embed_provider: Option<&'a Value>,
    pub embed_model: Option<&'a Value>,
}

#[handler]
pub fn search_entity_vectors_filtered (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_entity_vectors_filteredInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EntityVec", None)

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_entity_vectors_filteredResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        entity_id: result.get_property("entity_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
        embed_provider: result.get_property("embed_provider"),
        embed_model: result.get_property("embed_model"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_episode_chunksInput {

pub ep_id: String
}
#[derive(Serialize, Default)]
pub struct Get_episode_chunksChunksReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn get_episode_chunks (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_episode_chunksInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let chunks = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("episode_id")
                    .map_or(false, |v| *v == data.ep_id.clone()))
                } else {
                    Ok(false)
                }
            })

.out_vec("HasEpisodeChunk", false).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "chunks": chunks.iter().map(|chunk| Get_episode_chunksChunksReturnType {
        id: uuid_str(chunk.id(), &arena),
        label: chunk.label(),
        data: chunk.data(),
        score: chunk.score(),
        episode_id: chunk.get_property("episode_id"),
        group_id: chunk.get_property("group_id"),
        chunk_text: chunk.get_property("chunk_text"),
        chunk_index: chunk.get_property("chunk_index"),
        content_type: chunk.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_decision_traces_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_decision_traces_by_cycleTracesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub trace_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub candidate_type: Option<&'a Value>,
    pub candidate_id: Option<&'a Value>,
    pub decision: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub threshold_band: Option<&'a Value>,
    pub features_json: Option<&'a Value>,
    pub constraints_json: Option<&'a Value>,
    pub policy_version: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_decision_traces_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_decision_traces_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let traces = G::new(&db, &txn, &arena)
.n_from_type("ConsolDecisionTrace")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "traces": traces.iter().map(|trace| Find_consol_decision_traces_by_cycleTracesReturnType {
        id: uuid_str(trace.id(), &arena),
        label: trace.label(),
        cycle_id: trace.get_property("cycle_id"),
        group_id: trace.get_property("group_id"),
        trace_id: trace.get_property("trace_id"),
        phase: trace.get_property("phase"),
        candidate_type: trace.get_property("candidate_type"),
        candidate_id: trace.get_property("candidate_id"),
        decision: trace.get_property("decision"),
        decision_source: trace.get_property("decision_source"),
        confidence: trace.get_property("confidence"),
        threshold_band: trace.get_property("threshold_band"),
        features_json: trace.get_property("features_json"),
        constraints_json: trace.get_property("constraints_json"),
        policy_version: trace.get_property("policy_version"),
        metadata_json: trace.get_property("metadata_json"),
        timestamp: trace.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_outgoing_edgesInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_outgoing_edgesEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_outgoing_edges (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_outgoing_edgesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_e("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Get_outgoing_edgesEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_evidence_by_episodeInput {

pub ep_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_evidence_by_episodeEvidenceReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub fact_class: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub source_type: Option<&'a Value>,
    pub extractor_name: Option<&'a Value>,
    pub payload_json: Option<&'a Value>,
    pub source_span: Option<&'a Value>,
    pub signals_json: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub ambiguity_score: Option<&'a Value>,
    pub adjudication_request_id: Option<&'a Value>,
    pub commit_reason: Option<&'a Value>,
    pub committed_id: Option<&'a Value>,
    pub deferred_cycles: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn find_evidence_by_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_evidence_by_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let evidence = G::new(&db, &txn, &arena)
.n_from_type("Evidence")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("episode_id")
                    .map_or(false, |v| *v == data.ep_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "evidence": evidence.iter().map(|evidence| Find_evidence_by_episodeEvidenceReturnType {
        id: uuid_str(evidence.id(), &arena),
        label: evidence.label(),
        episode_id: evidence.get_property("episode_id"),
        group_id: evidence.get_property("group_id"),
        status: evidence.get_property("status"),
        evidence_id: evidence.get_property("evidence_id"),
        fact_class: evidence.get_property("fact_class"),
        confidence: evidence.get_property("confidence"),
        source_type: evidence.get_property("source_type"),
        extractor_name: evidence.get_property("extractor_name"),
        payload_json: evidence.get_property("payload_json"),
        source_span: evidence.get_property("source_span"),
        signals_json: evidence.get_property("signals_json"),
        ambiguity_tags_json: evidence.get_property("ambiguity_tags_json"),
        ambiguity_score: evidence.get_property("ambiguity_score"),
        adjudication_request_id: evidence.get_property("adjudication_request_id"),
        commit_reason: evidence.get_property("commit_reason"),
        committed_id: evidence.get_property("committed_id"),
        deferred_cycles: evidence.get_property("deferred_cycles"),
        created_at: evidence.get_property("created_at"),
        resolved_at: evidence.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_entities_bm25Input {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_entities_bm25ResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn search_entities_bm25 (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_entities_bm25Input>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_bm25("Entity", &data.query, data.k.clone())?.collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_entities_bm25ResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        name: result.get_property("name"),
        group_id: result.get_property("group_id"),
        entity_type: result.get_property("entity_type"),
        canonical_identifier: result.get_property("canonical_identifier"),
        entity_id: result.get_property("entity_id"),
        summary: result.get_property("summary"),
        attributes_json: result.get_property("attributes_json"),
        created_at: result.get_property("created_at"),
        updated_at: result.get_property("updated_at"),
        is_deleted: result.get_property("is_deleted"),
        deleted_at: result.get_property("deleted_at"),
        identity_core: result.get_property("identity_core"),
        mat_tier: result.get_property("mat_tier"),
        recon_count: result.get_property("recon_count"),
        lexical_regime: result.get_property("lexical_regime"),
        identifier_label: result.get_property("identifier_label"),
        pii_detected: result.get_property("pii_detected"),
        pii_categories_json: result.get_property("pii_categories_json"),
        access_count: result.get_property("access_count"),
        last_accessed: result.get_property("last_accessed"),
        source_episode_ids: result.get_property("source_episode_ids"),
        evidence_count: result.get_property("evidence_count"),
        evidence_span_start: result.get_property("evidence_span_start"),
        evidence_span_end: result.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_cue_vectorsInput {

pub vec: Vec<f64>,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_cue_vectorsResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_cue_vectors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_cue_vectorsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "CueVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_cue_vectorsResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_episodes_for_entityInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_episodes_for_entityEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn get_episodes_for_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_episodes_for_entityInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.in_node("HasEntity").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Get_episodes_for_entityEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_conversationInput {

pub id: ID,
pub title: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Update_conversationConversationReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub conversation_id: Option<&'a Value>,
    pub title: Option<&'a Value>,
    pub session_date: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_conversation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_conversationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let conversation = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("title", Value::from(&data.title)), ("updated_at", Value::from(&data.updated_at))])
    .collect_to_obj()?};
let response = json!({
    "conversation": Update_conversationConversationReturnType {
        id: uuid_str(conversation.id(), &arena),
        label: conversation.label(),
        group_id: conversation.get_property("group_id"),
        conversation_id: conversation.get_property("conversation_id"),
        title: conversation.get_property("title"),
        session_date: conversation.get_property("session_date"),
        created_at: conversation.get_property("created_at"),
        updated_at: conversation.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_adjudicationInput {

pub id: ID,
pub status: String,
pub resolution_source: String,
pub resolution_payload_json: String,
pub attempt_count: i32,
pub resolved_at: String
}
#[derive(Serialize, Default)]
pub struct Update_adjudicationRequestReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub request_id: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub evidence_ids_json: Option<&'a Value>,
    pub selected_text: Option<&'a Value>,
    pub request_reason: Option<&'a Value>,
    pub resolution_source: Option<&'a Value>,
    pub resolution_payload_json: Option<&'a Value>,
    pub attempt_count: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_adjudication (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_adjudicationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let request = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("status", Value::from(&data.status)), ("resolution_source", Value::from(&data.resolution_source)), ("resolution_payload_json", Value::from(&data.resolution_payload_json)), ("attempt_count", Value::from(&data.attempt_count)), ("resolved_at", Value::from(&data.resolved_at))])
    .collect_to_obj()?};
let response = json!({
    "request": Update_adjudicationRequestReturnType {
        id: uuid_str(request.id(), &arena),
        label: request.label(),
        episode_id: request.get_property("episode_id"),
        group_id: request.get_property("group_id"),
        status: request.get_property("status"),
        request_id: request.get_property("request_id"),
        ambiguity_tags_json: request.get_property("ambiguity_tags_json"),
        evidence_ids_json: request.get_property("evidence_ids_json"),
        selected_text: request.get_property("selected_text"),
        request_reason: request.get_property("request_reason"),
        resolution_source: request.get_property("resolution_source"),
        resolution_payload_json: request.get_property("resolution_payload_json"),
        attempt_count: request.get_property("attempt_count"),
        created_at: request.get_property("created_at"),
        resolved_at: request.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_typeInput {

pub etype: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_typeEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_type (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_typeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("entity_type")
                    .map_or(false, |v| *v == data.etype.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_typeEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_atlas_region_memberInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_atlas_region_member (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_atlas_region_memberInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_inferred_edges_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_inferred_edges_by_cycleEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub edge_id: Option<&'a Value>,
    pub source_id: Option<&'a Value>,
    pub target_id: Option<&'a Value>,
    pub source_name: Option<&'a Value>,
    pub target_name: Option<&'a Value>,
    pub co_occurrence_count: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub infer_type: Option<&'a Value>,
    pub pmi_score: Option<&'a Value>,
    pub llm_verdict: Option<&'a Value>,
    pub relationship_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_inferred_edges_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_inferred_edges_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_type("ConsolInferredEdge")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Find_consol_inferred_edges_by_cycleEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        cycle_id: edge.get_property("cycle_id"),
        group_id: edge.get_property("group_id"),
        edge_id: edge.get_property("edge_id"),
        source_id: edge.get_property("source_id"),
        target_id: edge.get_property("target_id"),
        source_name: edge.get_property("source_name"),
        target_name: edge.get_property("target_name"),
        co_occurrence_count: edge.get_property("co_occurrence_count"),
        confidence: edge.get_property("confidence"),
        infer_type: edge.get_property("infer_type"),
        pmi_score: edge.get_property("pmi_score"),
        llm_verdict: edge.get_property("llm_verdict"),
        relationship_id: edge.get_property("relationship_id"),
        timestamp: edge.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_triages_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_triages_by_cycleTriagesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub triage_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub decision: Option<&'a Value>,
    pub score_breakdown_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_triages_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_triages_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let triages = G::new(&db, &txn, &arena)
.n_from_type("ConsolTriage")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "triages": triages.iter().map(|triage| Find_consol_triages_by_cycleTriagesReturnType {
        id: uuid_str(triage.id(), &arena),
        label: triage.label(),
        cycle_id: triage.get_property("cycle_id"),
        group_id: triage.get_property("group_id"),
        triage_id: triage.get_property("triage_id"),
        episode_id: triage.get_property("episode_id"),
        decision: triage.get_property("decision"),
        score_breakdown_json: triage.get_property("score_breakdown_json"),
        timestamp: triage.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_evidenceInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_evidence (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_evidenceInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_calibrations_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_calibrations_by_cycleCalibrationsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub calibration_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub window_cycles: Option<&'a Value>,
    pub total_traces: Option<&'a Value>,
    pub labeled_examples: Option<&'a Value>,
    pub oracle_examples: Option<&'a Value>,
    pub abstain_count: Option<&'a Value>,
    pub accuracy: Option<&'a Value>,
    pub mean_confidence: Option<&'a Value>,
    pub expected_calibration_error: Option<&'a Value>,
    pub summary_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_calibrations_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_calibrations_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let calibrations = G::new(&db, &txn, &arena)
.n_from_type("ConsolCalibration")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "calibrations": calibrations.iter().map(|calibration| Find_consol_calibrations_by_cycleCalibrationsReturnType {
        id: uuid_str(calibration.id(), &arena),
        label: calibration.label(),
        cycle_id: calibration.get_property("cycle_id"),
        group_id: calibration.get_property("group_id"),
        calibration_id: calibration.get_property("calibration_id"),
        phase: calibration.get_property("phase"),
        window_cycles: calibration.get_property("window_cycles"),
        total_traces: calibration.get_property("total_traces"),
        labeled_examples: calibration.get_property("labeled_examples"),
        oracle_examples: calibration.get_property("oracle_examples"),
        abstain_count: calibration.get_property("abstain_count"),
        accuracy: calibration.get_property("accuracy"),
        mean_confidence: calibration.get_property("mean_confidence"),
        expected_calibration_error: calibration.get_property("expected_calibration_error"),
        summary_json: calibration.get_property("summary_json"),
        timestamp: calibration.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_vectors_filteredInput {

pub vec: Vec<f64>,
pub k: i32,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Search_episode_vectors_filteredResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_vectors_filtered (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_vectors_filteredInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EpisodeVec", None)

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_vectors_filteredResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_episode_chunkInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_episode_chunk (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_episode_chunkInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.v_from_id(&data.id, false).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_relationshipInput {

pub rel_id: String,
pub group_id: String,
pub predicate: String,
pub weight: f64,
pub polarity: String,
pub valid_from: String,
pub valid_to: String,
pub is_expired: bool,
pub created_at: String,
pub source_episode_id: String,
pub source_id: ID,
pub target_id: ID
}
#[derive(Serialize, Default)]
pub struct Create_relationshipEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_relationship (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_relationshipInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = G::new_mut(&db, &arena, &mut txn)
.add_edge("RelatesTo", Some(ImmutablePropertiesMap::new(10, vec![("valid_from", Value::from(data.valid_from.clone())), ("source_episode_id", Value::from(data.source_episode_id.clone())), ("weight", Value::from(data.weight.clone())), ("polarity", Value::from(data.polarity.clone())), ("created_at", Value::from(data.created_at.clone())), ("rel_id", Value::from(data.rel_id.clone())), ("group_id", Value::from(data.group_id.clone())), ("is_expired", Value::from(data.is_expired.clone())), ("predicate", Value::from(data.predicate.clone())), ("valid_to", Value::from(data.valid_to.clone()))].into_iter(), &arena)), *data.source_id, *data.target_id, false, false).collect_to_obj()?;
let response = json!({
    "edge": Create_relationshipEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_unconfirmed_complement_tagsInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_unconfirmed_complement_tagsTagsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_unconfirmed_complement_tags (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_unconfirmed_complement_tagsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let tags = G::new(&db, &txn, &arena)
.n_from_type("ComplementTag")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("cleared")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "tags": tags.iter().map(|tag| Find_unconfirmed_complement_tagsTagsReturnType {
        id: uuid_str(tag.id(), &arena),
        label: tag.label(),
        target_id: tag.get_property("target_id"),
        group_id: tag.get_property("group_id"),
        target_type: tag.get_property("target_type"),
        tag_type: tag.get_property("tag_type"),
        cycle_tagged: tag.get_property("cycle_tagged"),
        cycle_confirmed: tag.get_property("cycle_confirmed"),
        cleared: tag.get_property("cleared"),
        created_at: tag.get_property("created_at"),
        updated_at: tag.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_incoming_edgesInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_incoming_edgesEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_incoming_edges (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_incoming_edgesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.in_e("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Get_incoming_edgesEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_maturationInput {

pub mat_id: String,
pub cycle_id: String,
pub group_id: String,
pub entity_id: String,
pub entity_name: String,
pub old_tier: String,
pub new_tier: String,
pub maturity_score: f64,
pub source_diversity: i32,
pub temporal_span_days: f64,
pub relationship_richness: i32,
pub access_regularity: f64,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_maturationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub mat_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub old_tier: Option<&'a Value>,
    pub new_tier: Option<&'a Value>,
    pub maturity_score: Option<&'a Value>,
    pub source_diversity: Option<&'a Value>,
    pub temporal_span_days: Option<&'a Value>,
    pub relationship_richness: Option<&'a Value>,
    pub access_regularity: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_maturation (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_maturationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolMaturation", Some(ImmutablePropertiesMap::new(13, vec![("old_tier", Value::from(&data.old_tier)), ("access_regularity", Value::from(&data.access_regularity)), ("source_diversity", Value::from(&data.source_diversity)), ("mat_id", Value::from(&data.mat_id)), ("cycle_id", Value::from(&data.cycle_id)), ("temporal_span_days", Value::from(&data.temporal_span_days)), ("group_id", Value::from(&data.group_id)), ("new_tier", Value::from(&data.new_tier)), ("maturity_score", Value::from(&data.maturity_score)), ("entity_id", Value::from(&data.entity_id)), ("entity_name", Value::from(&data.entity_name)), ("timestamp", Value::from(&data.timestamp)), ("relationship_richness", Value::from(&data.relationship_richness))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_maturationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        mat_id: node.get_property("mat_id"),
        entity_id: node.get_property("entity_id"),
        entity_name: node.get_property("entity_name"),
        old_tier: node.get_property("old_tier"),
        new_tier: node.get_property("new_tier"),
        maturity_score: node.get_property("maturity_score"),
        source_diversity: node.get_property("source_diversity"),
        temporal_span_days: node.get_property("temporal_span_days"),
        relationship_richness: node.get_property("relationship_richness"),
        access_regularity: node.get_property("access_regularity"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_atlas_regionsInput {

pub snap_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_atlas_regionsRegionsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub region_id: Option<&'a Value>,
    pub region_label: Option<&'a Value>,
    pub subtitle: Option<&'a Value>,
    pub kind: Option<&'a Value>,
    pub member_count: Option<&'a Value>,
    pub represented_edge_count: Option<&'a Value>,
    pub activation_score: Option<&'a Value>,
    pub growth_7d: Option<&'a Value>,
    pub growth_30d: Option<&'a Value>,
    pub dominant_entity_types_json: Option<&'a Value>,
    pub hub_entity_ids_json: Option<&'a Value>,
    pub center_entity_id: Option<&'a Value>,
    pub latest_entity_created_at: Option<&'a Value>,
    pub x: Option<&'a Value>,
    pub y: Option<&'a Value>,
    pub z: Option<&'a Value>,
}

#[handler]
pub fn find_atlas_regions (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_atlas_regionsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let regions = G::new(&db, &txn, &arena)
.n_from_type("AtlasRegion")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("snapshot_id")
                    .map_or(false, |v| *v == data.snap_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "regions": regions.iter().map(|region| Find_atlas_regionsRegionsReturnType {
        id: uuid_str(region.id(), &arena),
        label: region.label(),
        snapshot_id: region.get_property("snapshot_id"),
        group_id: region.get_property("group_id"),
        region_id: region.get_property("region_id"),
        region_label: region.get_property("region_label"),
        subtitle: region.get_property("subtitle"),
        kind: region.get_property("kind"),
        member_count: region.get_property("member_count"),
        represented_edge_count: region.get_property("represented_edge_count"),
        activation_score: region.get_property("activation_score"),
        growth_7d: region.get_property("growth_7d"),
        growth_30d: region.get_property("growth_30d"),
        dominant_entity_types_json: region.get_property("dominant_entity_types_json"),
        hub_entity_ids_json: region.get_property("hub_entity_ids_json"),
        center_entity_id: region.get_property("center_entity_id"),
        latest_entity_created_at: region.get_property("latest_entity_created_at"),
        x: region.get_property("x"),
        y: region.get_property("y"),
        z: region.get_property("z"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_enabled_intentionsInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_enabled_intentionsIntentionsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler]
pub fn find_enabled_intentions (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_enabled_intentionsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let intentions = G::new(&db, &txn, &arena)
.n_from_type("Intention")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("enabled")
                    .map_or(false, |v| *v == true) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "intentions": intentions.iter().map(|intention| Find_enabled_intentionsIntentionsReturnType {
        id: uuid_str(intention.id(), &arena),
        label: intention.label(),
        group_id: intention.get_property("group_id"),
        intention_id: intention.get_property("intention_id"),
        trigger_text: intention.get_property("trigger_text"),
        action_text: intention.get_property("action_text"),
        entity_names_json: intention.get_property("entity_names_json"),
        enabled: intention.get_property("enabled"),
        fire_count: intention.get_property("fire_count"),
        max_fires: intention.get_property("max_fires"),
        created_at: intention.get_property("created_at"),
        updated_at: intention.get_property("updated_at"),
        deleted_at: intention.get_property("deleted_at"),
        is_deleted: intention.get_property("is_deleted"),
        context_json: intention.get_property("context_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_outgoing_neighborsInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_outgoing_neighborsNeighborsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_outgoing_neighbors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_outgoing_neighborsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let neighbors = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_node("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "neighbors": neighbors.iter().map(|neighbor| Get_outgoing_neighborsNeighborsReturnType {
        id: uuid_str(neighbor.id(), &arena),
        label: neighbor.label(),
        name: neighbor.get_property("name"),
        group_id: neighbor.get_property("group_id"),
        entity_type: neighbor.get_property("entity_type"),
        canonical_identifier: neighbor.get_property("canonical_identifier"),
        entity_id: neighbor.get_property("entity_id"),
        summary: neighbor.get_property("summary"),
        attributes_json: neighbor.get_property("attributes_json"),
        created_at: neighbor.get_property("created_at"),
        updated_at: neighbor.get_property("updated_at"),
        is_deleted: neighbor.get_property("is_deleted"),
        deleted_at: neighbor.get_property("deleted_at"),
        identity_core: neighbor.get_property("identity_core"),
        mat_tier: neighbor.get_property("mat_tier"),
        recon_count: neighbor.get_property("recon_count"),
        lexical_regime: neighbor.get_property("lexical_regime"),
        identifier_label: neighbor.get_property("identifier_label"),
        pii_detected: neighbor.get_property("pii_detected"),
        pii_categories_json: neighbor.get_property("pii_categories_json"),
        access_count: neighbor.get_property("access_count"),
        last_accessed: neighbor.get_property("last_accessed"),
        source_episode_ids: neighbor.get_property("source_episode_ids"),
        evidence_count: neighbor.get_property("evidence_count"),
        evidence_span_start: neighbor.get_property("evidence_span_start"),
        evidence_span_end: neighbor.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_graph_embeds_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_graph_embeds_by_cycleEmbedsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub embed_id: Option<&'a Value>,
    pub method: Option<&'a Value>,
    pub entities_trained: Option<&'a Value>,
    pub dimensions: Option<&'a Value>,
    pub training_duration_ms: Option<&'a Value>,
    pub full_retrain: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_graph_embeds_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_graph_embeds_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let embeds = G::new(&db, &txn, &arena)
.n_from_type("ConsolGraphEmbed")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "embeds": embeds.iter().map(|embed| Find_consol_graph_embeds_by_cycleEmbedsReturnType {
        id: uuid_str(embed.id(), &arena),
        label: embed.label(),
        cycle_id: embed.get_property("cycle_id"),
        group_id: embed.get_property("group_id"),
        embed_id: embed.get_property("embed_id"),
        method: embed.get_property("method"),
        entities_trained: embed.get_property("entities_trained"),
        dimensions: embed.get_property("dimensions"),
        training_duration_ms: embed.get_property("training_duration_ms"),
        full_retrain: embed.get_property("full_retrain"),
        timestamp: embed.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_episodes_by_sessionInput {

pub sid: String
}
#[derive(Serialize, Default)]
pub struct Find_episodes_by_sessionEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn find_episodes_by_session (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_episodes_by_sessionInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("session_id")
                    .map_or(false, |v| *v == data.sid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Find_episodes_by_sessionEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_vectorsInput {

pub vec: Vec<f64>,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_episode_vectorsResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_vectors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_vectorsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EpisodeVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_vectorsResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_dreamInput {

pub dream_id: String,
pub cycle_id: String,
pub group_id: String,
pub source_entity_id: String,
pub target_entity_id: String,
pub weight_delta: f64,
pub seed_entity_id: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_dreamNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub dream_id: Option<&'a Value>,
    pub source_entity_id: Option<&'a Value>,
    pub target_entity_id: Option<&'a Value>,
    pub weight_delta: Option<&'a Value>,
    pub seed_entity_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_dream (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_dreamInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolDream", Some(ImmutablePropertiesMap::new(8, vec![("group_id", Value::from(&data.group_id)), ("dream_id", Value::from(&data.dream_id)), ("timestamp", Value::from(&data.timestamp)), ("source_entity_id", Value::from(&data.source_entity_id)), ("weight_delta", Value::from(&data.weight_delta)), ("target_entity_id", Value::from(&data.target_entity_id)), ("seed_entity_id", Value::from(&data.seed_entity_id)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_dreamNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        dream_id: node.get_property("dream_id"),
        source_entity_id: node.get_property("source_entity_id"),
        target_entity_id: node.get_property("target_entity_id"),
        weight_delta: node.get_property("weight_delta"),
        seed_entity_id: node.get_property("seed_entity_id"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_evidenceInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_evidenceEvidenceReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub fact_class: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub source_type: Option<&'a Value>,
    pub extractor_name: Option<&'a Value>,
    pub payload_json: Option<&'a Value>,
    pub source_span: Option<&'a Value>,
    pub signals_json: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub ambiguity_score: Option<&'a Value>,
    pub adjudication_request_id: Option<&'a Value>,
    pub commit_reason: Option<&'a Value>,
    pub committed_id: Option<&'a Value>,
    pub deferred_cycles: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn get_evidence (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_evidenceInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let evidence = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "evidence": Get_evidenceEvidenceReturnType {
        id: uuid_str(evidence.id(), &arena),
        label: evidence.label(),
        episode_id: evidence.get_property("episode_id"),
        group_id: evidence.get_property("group_id"),
        status: evidence.get_property("status"),
        evidence_id: evidence.get_property("evidence_id"),
        fact_class: evidence.get_property("fact_class"),
        confidence: evidence.get_property("confidence"),
        source_type: evidence.get_property("source_type"),
        extractor_name: evidence.get_property("extractor_name"),
        payload_json: evidence.get_property("payload_json"),
        source_span: evidence.get_property("source_span"),
        signals_json: evidence.get_property("signals_json"),
        ambiguity_tags_json: evidence.get_property("ambiguity_tags_json"),
        ambiguity_score: evidence.get_property("ambiguity_score"),
        adjudication_request_id: evidence.get_property("adjudication_request_id"),
        commit_reason: evidence.get_property("commit_reason"),
        committed_id: evidence.get_property("committed_id"),
        deferred_cycles: evidence.get_property("deferred_cycles"),
        created_at: evidence.get_property("created_at"),
        resolved_at: evidence.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_graph_embedInput {

pub embed_id: String,
pub cycle_id: String,
pub group_id: String,
pub method: String,
pub entities_trained: i32,
pub dimensions: i32,
pub training_duration_ms: f64,
pub full_retrain: bool,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_graph_embedNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub embed_id: Option<&'a Value>,
    pub method: Option<&'a Value>,
    pub entities_trained: Option<&'a Value>,
    pub dimensions: Option<&'a Value>,
    pub training_duration_ms: Option<&'a Value>,
    pub full_retrain: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_graph_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_graph_embedInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolGraphEmbed", Some(ImmutablePropertiesMap::new(9, vec![("timestamp", Value::from(&data.timestamp)), ("embed_id", Value::from(&data.embed_id)), ("method", Value::from(&data.method)), ("cycle_id", Value::from(&data.cycle_id)), ("group_id", Value::from(&data.group_id)), ("training_duration_ms", Value::from(&data.training_duration_ms)), ("entities_trained", Value::from(&data.entities_trained)), ("full_retrain", Value::from(&data.full_retrain)), ("dimensions", Value::from(&data.dimensions))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_graph_embedNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        embed_id: node.get_property("embed_id"),
        method: node.get_property("method"),
        entities_trained: node.get_property("entities_trained"),
        dimensions: node.get_property("dimensions"),
        training_duration_ms: node.get_property("training_duration_ms"),
        full_retrain: node.get_property("full_retrain"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_intention_fullInput {

pub id: ID,
pub trigger_text: String,
pub action_text: String,
pub entity_names_json: String,
pub enabled: bool,
pub fire_count: i32,
pub max_fires: i32,
pub updated_at: String,
pub deleted_at: String,
pub is_deleted: bool,
pub context_json: String
}
#[derive(Serialize, Default)]
pub struct Update_intention_fullIntentionReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_intention_full (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_intention_fullInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let intention = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("trigger_text", Value::from(&data.trigger_text)), ("action_text", Value::from(&data.action_text)), ("entity_names_json", Value::from(&data.entity_names_json)), ("enabled", Value::from(&data.enabled)), ("fire_count", Value::from(&data.fire_count)), ("max_fires", Value::from(&data.max_fires)), ("updated_at", Value::from(&data.updated_at)), ("deleted_at", Value::from(&data.deleted_at)), ("is_deleted", Value::from(&data.is_deleted)), ("context_json", Value::from(&data.context_json))])
    .collect_to_obj()?};
let response = json!({
    "intention": Update_intention_fullIntentionReturnType {
        id: uuid_str(intention.id(), &arena),
        label: intention.label(),
        group_id: intention.get_property("group_id"),
        intention_id: intention.get_property("intention_id"),
        trigger_text: intention.get_property("trigger_text"),
        action_text: intention.get_property("action_text"),
        entity_names_json: intention.get_property("entity_names_json"),
        enabled: intention.get_property("enabled"),
        fire_count: intention.get_property("fire_count"),
        max_fires: intention.get_property("max_fires"),
        created_at: intention.get_property("created_at"),
        updated_at: intention.get_property("updated_at"),
        deleted_at: intention.get_property("deleted_at"),
        is_deleted: intention.get_property("is_deleted"),
        context_json: intention.get_property("context_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_schemas_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_schemas_by_cycleSchemasReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub schema_id: Option<&'a Value>,
    pub schema_entity_id: Option<&'a Value>,
    pub schema_name: Option<&'a Value>,
    pub instance_count: Option<&'a Value>,
    pub predicate_count: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_schemas_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_schemas_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let schemas = G::new(&db, &txn, &arena)
.n_from_type("ConsolSchema")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "schemas": schemas.iter().map(|schema| Find_consol_schemas_by_cycleSchemasReturnType {
        id: uuid_str(schema.id(), &arena),
        label: schema.label(),
        cycle_id: schema.get_property("cycle_id"),
        group_id: schema.get_property("group_id"),
        schema_id: schema.get_property("schema_id"),
        schema_entity_id: schema.get_property("schema_entity_id"),
        schema_name: schema.get_property("schema_name"),
        instance_count: schema.get_property("instance_count"),
        predicate_count: schema.get_property("predicate_count"),
        action: schema.get_property("action"),
        timestamp: schema.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_evidence_adjs_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_evidence_adjs_by_cycleAdjsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub adj_id: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub new_confidence: Option<&'a Value>,
    pub reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_evidence_adjs_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_evidence_adjs_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let adjs = G::new(&db, &txn, &arena)
.n_from_type("ConsolEvidenceAdj")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "adjs": adjs.iter().map(|adj| Find_consol_evidence_adjs_by_cycleAdjsReturnType {
        id: uuid_str(adj.id(), &arena),
        label: adj.label(),
        cycle_id: adj.get_property("cycle_id"),
        group_id: adj.get_property("group_id"),
        adj_id: adj.get_property("adj_id"),
        evidence_id: adj.get_property("evidence_id"),
        action: adj.get_property("action"),
        new_confidence: adj.get_property("new_confidence"),
        reason: adj.get_property("reason"),
        timestamp: adj.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episodes_embedInput {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_episodes_embedResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episodes_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episodes_embedInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.query);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&__internal_embed_data_0, data.k.clone(), "EpisodeVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episodes_embedResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct soft_delete_intentionInput {

pub id: ID,
pub deleted_at: String
}
#[derive(Serialize, Default)]
pub struct Soft_delete_intentionIntentionReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler(is_write)]
pub fn soft_delete_intention (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<soft_delete_intentionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let intention = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("is_deleted", Value::from(true)), ("deleted_at", Value::from(&data.deleted_at))])
    .collect_to_obj()?};
let response = json!({
    "intention": Soft_delete_intentionIntentionReturnType {
        id: uuid_str(intention.id(), &arena),
        label: intention.label(),
        group_id: intention.get_property("group_id"),
        intention_id: intention.get_property("intention_id"),
        trigger_text: intention.get_property("trigger_text"),
        action_text: intention.get_property("action_text"),
        entity_names_json: intention.get_property("entity_names_json"),
        enabled: intention.get_property("enabled"),
        fire_count: intention.get_property("fire_count"),
        max_fires: intention.get_property("max_fires"),
        created_at: intention.get_property("created_at"),
        updated_at: intention.get_property("updated_at"),
        deleted_at: intention.get_property("deleted_at"),
        is_deleted: intention.get_property("is_deleted"),
        context_json: intention.get_property("context_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_episode_entitiesInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_episode_entitiesEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_episode_entities (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_episode_entitiesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_node("HasEntity").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Get_episode_entitiesEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_atlas_region_edgeInput {

pub snapshot_id: String,
pub group_id: String,
pub edge_id: String,
pub source_region_id: String,
pub target_region_id: String,
pub weight: f64,
pub relationship_count: i32
}
#[derive(Serialize, Default)]
pub struct Create_atlas_region_edgeNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub edge_id: Option<&'a Value>,
    pub source_region_id: Option<&'a Value>,
    pub target_region_id: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub relationship_count: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_atlas_region_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_atlas_region_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("AtlasRegionEdge", Some(ImmutablePropertiesMap::new(7, vec![("group_id", Value::from(&data.group_id)), ("target_region_id", Value::from(&data.target_region_id)), ("edge_id", Value::from(&data.edge_id)), ("snapshot_id", Value::from(&data.snapshot_id)), ("source_region_id", Value::from(&data.source_region_id)), ("relationship_count", Value::from(&data.relationship_count)), ("weight", Value::from(&data.weight))].into_iter(), &arena)), Some(&["snapshot_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_atlas_region_edgeNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        snapshot_id: node.get_property("snapshot_id"),
        group_id: node.get_property("group_id"),
        edge_id: node.get_property("edge_id"),
        source_region_id: node.get_property("source_region_id"),
        target_region_id: node.get_property("target_region_id"),
        weight: node.get_property("weight"),
        relationship_count: node.get_property("relationship_count"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct add_entity_vectorInput {

pub entity_id: String,
pub group_id: String,
pub content_type: String,
pub embed_provider: String,
pub embed_model: String,
pub vec: Vec<f64>
}
#[derive(Serialize, Default)]
pub struct Add_entity_vectorVReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
    pub embed_provider: Option<&'a Value>,
    pub embed_model: Option<&'a Value>,
}

#[handler(is_write)]
pub fn add_entity_vector (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<add_entity_vectorInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let v = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&data.vec, "EntityVec", Some(ImmutablePropertiesMap::new(5, vec![("embed_model", Value::from(data.embed_model.clone())), ("group_id", Value::from(data.group_id.clone())), ("content_type", Value::from(data.content_type.clone())), ("entity_id", Value::from(data.entity_id.clone())), ("embed_provider", Value::from(data.embed_provider.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "v": Add_entity_vectorVReturnType {
        id: uuid_str(v.id(), &arena),
        label: v.label(),
        data: v.data(),
        score: v.score(),
        entity_id: v.get_property("entity_id"),
        group_id: v.get_property("group_id"),
        content_type: v.get_property("content_type"),
        embed_provider: v.get_property("embed_provider"),
        embed_model: v.get_property("embed_model"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_schema_membersInput {

pub schema_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_schema_membersMembersReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub schema_entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub role_label: Option<&'a Value>,
    pub member_entity_id: Option<&'a Value>,
}

#[handler]
pub fn find_schema_members (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_schema_membersInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let members = G::new(&db, &txn, &arena)
.n_from_type("SchemaMember")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("schema_entity_id")
                    .map_or(false, |v| *v == data.schema_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "members": members.iter().map(|member| Find_schema_membersMembersReturnType {
        id: uuid_str(member.id(), &arena),
        label: member.label(),
        schema_entity_id: member.get_property("schema_entity_id"),
        group_id: member.get_property("group_id"),
        role_label: member.get_property("role_label"),
        member_entity_id: member.get_property("member_entity_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_chunks_embedInput {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_episode_chunks_embedResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_chunks_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_chunks_embedInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.query);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&__internal_embed_data_0, data.k.clone(), "EpisodeChunk", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_chunks_embedResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        chunk_text: result.get_property("chunk_text"),
        chunk_index: result.get_property("chunk_index"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_semantic_transitions_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_semantic_transitions_by_cycleTransitionsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub trans_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub old_tier: Option<&'a Value>,
    pub new_tier: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_semantic_transitions_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_semantic_transitions_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let transitions = G::new(&db, &txn, &arena)
.n_from_type("ConsolSemanticTransition")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "transitions": transitions.iter().map(|transition| Find_consol_semantic_transitions_by_cycleTransitionsReturnType {
        id: uuid_str(transition.id(), &arena),
        label: transition.label(),
        cycle_id: transition.get_property("cycle_id"),
        group_id: transition.get_property("group_id"),
        trans_id: transition.get_property("trans_id"),
        episode_id: transition.get_property("episode_id"),
        old_tier: transition.get_property("old_tier"),
        new_tier: transition.get_property("new_tier"),
        entity_coverage: transition.get_property("entity_coverage"),
        consolidation_cycles: transition.get_property("consolidation_cycles"),
        timestamp: transition.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_cue_by_episodeInput {

pub ep_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_cue_by_episodeCuesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub cue_version: Option<&'a Value>,
    pub discourse_class: Option<&'a Value>,
    pub cue_text: Option<&'a Value>,
    pub supporting_spans_json: Option<&'a Value>,
    pub temporal_markers_json: Option<&'a Value>,
    pub quote_spans_json: Option<&'a Value>,
    pub contradiction_keys_json: Option<&'a Value>,
    pub first_spans_json: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub cue_score: Option<&'a Value>,
    pub salience_score: Option<&'a Value>,
    pub projection_priority: Option<&'a Value>,
    pub route_reason: Option<&'a Value>,
    pub hit_count: Option<&'a Value>,
    pub surfaced_count: Option<&'a Value>,
    pub selected_count: Option<&'a Value>,
    pub used_count: Option<&'a Value>,
    pub near_miss_count: Option<&'a Value>,
    pub policy_score: Option<&'a Value>,
    pub projection_attempts: Option<&'a Value>,
    pub last_hit_at: Option<&'a Value>,
    pub last_feedback_at: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_cue_by_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_cue_by_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let cues = G::new(&db, &txn, &arena)
.n_from_type("EpisodeCue")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("episode_id")
                    .map_or(false, |v| *v == data.ep_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "cues": cues.iter().map(|cue| Find_cue_by_episodeCuesReturnType {
        id: uuid_str(cue.id(), &arena),
        label: cue.label(),
        episode_id: cue.get_property("episode_id"),
        group_id: cue.get_property("group_id"),
        cue_version: cue.get_property("cue_version"),
        discourse_class: cue.get_property("discourse_class"),
        cue_text: cue.get_property("cue_text"),
        supporting_spans_json: cue.get_property("supporting_spans_json"),
        temporal_markers_json: cue.get_property("temporal_markers_json"),
        quote_spans_json: cue.get_property("quote_spans_json"),
        contradiction_keys_json: cue.get_property("contradiction_keys_json"),
        first_spans_json: cue.get_property("first_spans_json"),
        projection_state: cue.get_property("projection_state"),
        cue_score: cue.get_property("cue_score"),
        salience_score: cue.get_property("salience_score"),
        projection_priority: cue.get_property("projection_priority"),
        route_reason: cue.get_property("route_reason"),
        hit_count: cue.get_property("hit_count"),
        surfaced_count: cue.get_property("surfaced_count"),
        selected_count: cue.get_property("selected_count"),
        used_count: cue.get_property("used_count"),
        near_miss_count: cue.get_property("near_miss_count"),
        policy_score: cue.get_property("policy_score"),
        projection_attempts: cue.get_property("projection_attempts"),
        last_hit_at: cue.get_property("last_hit_at"),
        last_feedback_at: cue.get_property("last_feedback_at"),
        last_projected_at: cue.get_property("last_projected_at"),
        created_at: cue.get_property("created_at"),
        updated_at: cue.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_adjudications_by_episodeInput {

pub ep_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_adjudications_by_episodeRequestsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub request_id: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub evidence_ids_json: Option<&'a Value>,
    pub selected_text: Option<&'a Value>,
    pub request_reason: Option<&'a Value>,
    pub resolution_source: Option<&'a Value>,
    pub resolution_payload_json: Option<&'a Value>,
    pub attempt_count: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn find_adjudications_by_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_adjudications_by_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let requests = G::new(&db, &txn, &arena)
.n_from_type("AdjudicationRequest")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("episode_id")
                    .map_or(false, |v| *v == data.ep_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "requests": requests.iter().map(|request| Find_adjudications_by_episodeRequestsReturnType {
        id: uuid_str(request.id(), &arena),
        label: request.label(),
        episode_id: request.get_property("episode_id"),
        group_id: request.get_property("group_id"),
        status: request.get_property("status"),
        request_id: request.get_property("request_id"),
        ambiguity_tags_json: request.get_property("ambiguity_tags_json"),
        evidence_ids_json: request.get_property("evidence_ids_json"),
        selected_text: request.get_property("selected_text"),
        request_reason: request.get_property("request_reason"),
        resolution_source: request.get_property("resolution_source"),
        resolution_payload_json: request.get_property("resolution_payload_json"),
        attempt_count: request.get_property("attempt_count"),
        created_at: request.get_property("created_at"),
        resolved_at: request.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_identity_core_entitiesInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_identity_core_entitiesEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_identity_core_entities (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_identity_core_entitiesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("identity_core")
                    .map_or(false, |v| *v == true) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_identity_core_entitiesEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct link_episode_chunkInput {

pub episode_id: ID,
pub chunk_id: ID
}
#[derive(Serialize, Default)]
pub struct Link_episode_chunkEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
}

#[handler(is_write)]
pub fn link_episode_chunk (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<link_episode_chunkInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = G::new_mut(&db, &arena, &mut txn)
.add_edge("HasEpisodeChunk", None, *data.episode_id, *data.chunk_id, false, false).collect_to_obj()?;
let response = json!({
    "edge": Link_episode_chunkEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct link_schema_memberInput {

pub entity_id: ID,
pub member_id: ID
}
#[derive(Serialize, Default)]
pub struct Link_schema_memberEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
}

#[handler(is_write)]
pub fn link_schema_member (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<link_schema_memberInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = G::new_mut(&db, &arena, &mut txn)
.add_edge("HasSchemaMember", None, *data.entity_id, *data.member_id, false, false).collect_to_obj()?;
let response = json!({
    "edge": Link_schema_memberEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_chunks_embed_filteredInput {

pub query: String,
pub k: i32,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Search_episode_chunks_embed_filteredResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_chunks_embed_filtered (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_chunks_embed_filteredInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.query);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&__internal_embed_data_0, data.k.clone(), "EpisodeChunk", None)

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_chunks_embed_filteredResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        chunk_text: result.get_property("chunk_text"),
        chunk_index: result.get_property("chunk_index"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_decision_outcomes_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_decision_outcomes_by_cycleOutcomesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub outcome_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub decision_trace_id: Option<&'a Value>,
    pub outcome_type: Option<&'a Value>,
    pub outcome_label: Option<&'a Value>,
    pub outcome_value: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_decision_outcomes_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_decision_outcomes_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let outcomes = G::new(&db, &txn, &arena)
.n_from_type("ConsolDecisionOutcome")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "outcomes": outcomes.iter().map(|outcome| Find_consol_decision_outcomes_by_cycleOutcomesReturnType {
        id: uuid_str(outcome.id(), &arena),
        label: outcome.label(),
        cycle_id: outcome.get_property("cycle_id"),
        group_id: outcome.get_property("group_id"),
        outcome_id: outcome.get_property("outcome_id"),
        phase: outcome.get_property("phase"),
        decision_trace_id: outcome.get_property("decision_trace_id"),
        outcome_type: outcome.get_property("outcome_type"),
        outcome_label: outcome.get_property("outcome_label"),
        outcome_value: outcome.get_property("outcome_value"),
        metadata_json: outcome.get_property("metadata_json"),
        timestamp: outcome.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_nameInput {

pub name_query: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_nameEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_name (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_nameInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("name")
                    .map_or(false, |v| v.contains(&data.name_query)) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_nameEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episodes_bm25Input {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_episodes_bm25ResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn search_episodes_bm25 (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episodes_bm25Input>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_bm25("Episode", &data.query, data.k.clone())?.collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episodes_bm25ResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        group_id: result.get_property("group_id"),
        status: result.get_property("status"),
        session_id: result.get_property("session_id"),
        episode_id: result.get_property("episode_id"),
        content: result.get_property("content"),
        source: result.get_property("source"),
        created_at: result.get_property("created_at"),
        updated_at: result.get_property("updated_at"),
        error: result.get_property("error"),
        retry_count: result.get_property("retry_count"),
        processing_duration_ms: result.get_property("processing_duration_ms"),
        skipped_meta: result.get_property("skipped_meta"),
        skipped_triage: result.get_property("skipped_triage"),
        encoding_context_json: result.get_property("encoding_context_json"),
        memory_tier: result.get_property("memory_tier"),
        consolidation_cycles: result.get_property("consolidation_cycles"),
        entity_coverage: result.get_property("entity_coverage"),
        projection_state: result.get_property("projection_state"),
        last_projection_reason: result.get_property("last_projection_reason"),
        last_projected_at: result.get_property("last_projected_at"),
        conversation_date: result.get_property("conversation_date"),
        attachments_json: result.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_name_and_typeInput {

pub name_query: String,
pub etype: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_name_and_typeEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_name_and_type (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_name_and_typeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("name")
                    .map_or(false, |v| v.contains(&data.name_query)) && val
                    .get_property("entity_type")
                    .map_or(false, |v| *v == data.etype.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_name_and_typeEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct add_graph_embed_vectorInput {

pub entity_id: String,
pub group_id: String,
pub method: String,
pub model_version: String,
pub vec: Vec<f64>
}
#[derive(Serialize, Default)]
pub struct Add_graph_embed_vectorVReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub method: Option<&'a Value>,
    pub model_version: Option<&'a Value>,
}

#[handler(is_write)]
pub fn add_graph_embed_vector (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<add_graph_embed_vectorInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let v = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&data.vec, "GraphEmbedVec", Some(ImmutablePropertiesMap::new(4, vec![("group_id", Value::from(data.group_id.clone())), ("entity_id", Value::from(data.entity_id.clone())), ("model_version", Value::from(data.model_version.clone())), ("method", Value::from(data.method.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "v": Add_graph_embed_vectorVReturnType {
        id: uuid_str(v.id(), &arena),
        label: v.label(),
        data: v.data(),
        score: v.score(),
        entity_id: v.get_property("entity_id"),
        group_id: v.get_property("group_id"),
        method: v.get_property("method"),
        model_version: v.get_property("model_version"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct delete_graph_embed_vectorInput {

pub id: ID
}
#[handler(is_write)]
pub fn delete_graph_embed_vector (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<delete_graph_embed_vectorInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.v_from_id(&data.id, false).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_replays_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_replays_by_cycleReplaysReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub replay_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub new_entities_found: Option<&'a Value>,
    pub new_relationships_found: Option<&'a Value>,
    pub entities_updated: Option<&'a Value>,
    pub skipped_reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_replays_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_replays_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let replays = G::new(&db, &txn, &arena)
.n_from_type("ConsolReplay")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "replays": replays.iter().map(|replay| Find_consol_replays_by_cycleReplaysReturnType {
        id: uuid_str(replay.id(), &arena),
        label: replay.label(),
        cycle_id: replay.get_property("cycle_id"),
        group_id: replay.get_property("group_id"),
        replay_id: replay.get_property("replay_id"),
        episode_id: replay.get_property("episode_id"),
        new_entities_found: replay.get_property("new_entities_found"),
        new_relationships_found: replay.get_property("new_relationships_found"),
        entities_updated: replay.get_property("entities_updated"),
        skipped_reason: replay.get_property("skipped_reason"),
        timestamp: replay.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_active_complement_tagsInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_active_complement_tagsTagsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_active_complement_tags (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_active_complement_tagsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let tags = G::new(&db, &txn, &arena)
.n_from_type("ComplementTag")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("cleared")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "tags": tags.iter().map(|tag| Find_active_complement_tagsTagsReturnType {
        id: uuid_str(tag.id(), &arena),
        label: tag.label(),
        target_id: tag.get_property("target_id"),
        group_id: tag.get_property("group_id"),
        target_type: tag.get_property("target_type"),
        tag_type: tag.get_property("tag_type"),
        cycle_tagged: tag.get_property("cycle_tagged"),
        cycle_confirmed: tag.get_property("cycle_confirmed"),
        cleared: tag.get_property("cleared"),
        created_at: tag.get_property("created_at"),
        updated_at: tag.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct add_episode_vectorInput {

pub episode_id: String,
pub group_id: String,
pub content_type: String,
pub vec: Vec<f64>
}
#[derive(Serialize, Default)]
pub struct Add_episode_vectorVReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler(is_write)]
pub fn add_episode_vector (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<add_episode_vectorInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let v = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&data.vec, "EpisodeVec", Some(ImmutablePropertiesMap::new(3, vec![("content_type", Value::from(data.content_type.clone())), ("episode_id", Value::from(data.episode_id.clone())), ("group_id", Value::from(data.group_id.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "v": Add_episode_vectorVReturnType {
        id: uuid_str(v.id(), &arena),
        label: v.label(),
        data: v.data(),
        score: v.score(),
        episode_id: v.get_property("episode_id"),
        group_id: v.get_property("group_id"),
        content_type: v.get_property("content_type"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_atlas_snapshots_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_atlas_snapshots_by_groupSnapshotsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub snapshot_id: Option<&'a Value>,
    pub generated_at: Option<&'a Value>,
    pub represented_entity_count: Option<&'a Value>,
    pub represented_edge_count: Option<&'a Value>,
    pub displayed_node_count: Option<&'a Value>,
    pub displayed_edge_count: Option<&'a Value>,
    pub total_entities: Option<&'a Value>,
    pub total_relationships: Option<&'a Value>,
    pub total_regions: Option<&'a Value>,
    pub hottest_region_id: Option<&'a Value>,
    pub fastest_growing_region_id: Option<&'a Value>,
    pub truncated: Option<&'a Value>,
}

#[handler]
pub fn find_atlas_snapshots_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_atlas_snapshots_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let snapshots = G::new(&db, &txn, &arena)
.n_from_type("AtlasSnapshot")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "snapshots": snapshots.iter().map(|snapshot| Find_atlas_snapshots_by_groupSnapshotsReturnType {
        id: uuid_str(snapshot.id(), &arena),
        label: snapshot.label(),
        group_id: snapshot.get_property("group_id"),
        snapshot_id: snapshot.get_property("snapshot_id"),
        generated_at: snapshot.get_property("generated_at"),
        represented_entity_count: snapshot.get_property("represented_entity_count"),
        represented_edge_count: snapshot.get_property("represented_edge_count"),
        displayed_node_count: snapshot.get_property("displayed_node_count"),
        displayed_edge_count: snapshot.get_property("displayed_edge_count"),
        total_entities: snapshot.get_property("total_entities"),
        total_relationships: snapshot.get_property("total_relationships"),
        total_regions: snapshot.get_property("total_regions"),
        hottest_region_id: snapshot.get_property("hottest_region_id"),
        fastest_growing_region_id: snapshot.get_property("fastest_growing_region_id"),
        truncated: snapshot.get_property("truncated"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_episode_chunk_vecInput {

pub episode_id: String,
pub group_id: String,
pub chunk_text: String,
pub chunk_index: i32,
pub content_type: String,
pub vec: Vec<f64>
}
#[derive(Serialize, Default)]
pub struct Create_episode_chunk_vecChunkReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_episode_chunk_vec (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_episode_chunk_vecInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let chunk = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&data.vec, "EpisodeChunk", Some(ImmutablePropertiesMap::new(5, vec![("group_id", Value::from(data.group_id.clone())), ("episode_id", Value::from(data.episode_id.clone())), ("content_type", Value::from(data.content_type.clone())), ("chunk_index", Value::from(data.chunk_index.clone())), ("chunk_text", Value::from(data.chunk_text.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "chunk": Create_episode_chunk_vecChunkReturnType {
        id: uuid_str(chunk.id(), &arena),
        label: chunk.label(),
        data: chunk.data(),
        score: chunk.score(),
        episode_id: chunk.get_property("episode_id"),
        group_id: chunk.get_property("group_id"),
        chunk_text: chunk.get_property("chunk_text"),
        chunk_index: chunk.get_property("chunk_index"),
        content_type: chunk.get_property("content_type"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_atlas_snapshotInput {

pub snapshot_id: String,
pub group_id: String,
pub generated_at: String,
pub represented_entity_count: i32,
pub represented_edge_count: i32,
pub displayed_node_count: i32,
pub displayed_edge_count: i32,
pub total_entities: i32,
pub total_relationships: i32,
pub total_regions: i32,
pub hottest_region_id: String,
pub fastest_growing_region_id: String,
pub truncated: bool
}
#[derive(Serialize, Default)]
pub struct Create_atlas_snapshotNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub snapshot_id: Option<&'a Value>,
    pub generated_at: Option<&'a Value>,
    pub represented_entity_count: Option<&'a Value>,
    pub represented_edge_count: Option<&'a Value>,
    pub displayed_node_count: Option<&'a Value>,
    pub displayed_edge_count: Option<&'a Value>,
    pub total_entities: Option<&'a Value>,
    pub total_relationships: Option<&'a Value>,
    pub total_regions: Option<&'a Value>,
    pub hottest_region_id: Option<&'a Value>,
    pub fastest_growing_region_id: Option<&'a Value>,
    pub truncated: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_atlas_snapshot (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_atlas_snapshotInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("AtlasSnapshot", Some(ImmutablePropertiesMap::new(13, vec![("represented_edge_count", Value::from(&data.represented_edge_count)), ("fastest_growing_region_id", Value::from(&data.fastest_growing_region_id)), ("total_entities", Value::from(&data.total_entities)), ("displayed_edge_count", Value::from(&data.displayed_edge_count)), ("generated_at", Value::from(&data.generated_at)), ("total_relationships", Value::from(&data.total_relationships)), ("total_regions", Value::from(&data.total_regions)), ("truncated", Value::from(&data.truncated)), ("represented_entity_count", Value::from(&data.represented_entity_count)), ("group_id", Value::from(&data.group_id)), ("displayed_node_count", Value::from(&data.displayed_node_count)), ("hottest_region_id", Value::from(&data.hottest_region_id)), ("snapshot_id", Value::from(&data.snapshot_id))].into_iter(), &arena)), Some(&["group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_atlas_snapshotNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        group_id: node.get_property("group_id"),
        snapshot_id: node.get_property("snapshot_id"),
        generated_at: node.get_property("generated_at"),
        represented_entity_count: node.get_property("represented_entity_count"),
        represented_edge_count: node.get_property("represented_edge_count"),
        displayed_node_count: node.get_property("displayed_node_count"),
        displayed_edge_count: node.get_property("displayed_edge_count"),
        total_entities: node.get_property("total_entities"),
        total_relationships: node.get_property("total_relationships"),
        total_regions: node.get_property("total_regions"),
        hottest_region_id: node.get_property("hottest_region_id"),
        fastest_growing_region_id: node.get_property("fastest_growing_region_id"),
        truncated: node.get_property("truncated"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_pending_evidenceInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_pending_evidenceEvidenceReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub evidence_id: Option<&'a Value>,
    pub fact_class: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub source_type: Option<&'a Value>,
    pub extractor_name: Option<&'a Value>,
    pub payload_json: Option<&'a Value>,
    pub source_span: Option<&'a Value>,
    pub signals_json: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub ambiguity_score: Option<&'a Value>,
    pub adjudication_request_id: Option<&'a Value>,
    pub commit_reason: Option<&'a Value>,
    pub committed_id: Option<&'a Value>,
    pub deferred_cycles: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn find_pending_evidence (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_pending_evidenceInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let evidence = G::new(&db, &txn, &arena)
.n_from_type("Evidence")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("status")
                    .map_or(false, |v| *v == "pending")))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "evidence": evidence.iter().map(|evidence| Find_pending_evidenceEvidenceReturnType {
        id: uuid_str(evidence.id(), &arena),
        label: evidence.label(),
        episode_id: evidence.get_property("episode_id"),
        group_id: evidence.get_property("group_id"),
        status: evidence.get_property("status"),
        evidence_id: evidence.get_property("evidence_id"),
        fact_class: evidence.get_property("fact_class"),
        confidence: evidence.get_property("confidence"),
        source_type: evidence.get_property("source_type"),
        extractor_name: evidence.get_property("extractor_name"),
        payload_json: evidence.get_property("payload_json"),
        source_span: evidence.get_property("source_span"),
        signals_json: evidence.get_property("signals_json"),
        ambiguity_tags_json: evidence.get_property("ambiguity_tags_json"),
        ambiguity_score: evidence.get_property("ambiguity_score"),
        adjudication_request_id: evidence.get_property("adjudication_request_id"),
        commit_reason: evidence.get_property("commit_reason"),
        committed_id: evidence.get_property("committed_id"),
        deferred_cycles: evidence.get_property("deferred_cycles"),
        created_at: evidence.get_property("created_at"),
        resolved_at: evidence.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_evidence_by_statusInput {

pub gid: String,
pub st: String
}

#[handler]
pub fn find_evidence_by_status (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_evidence_by_statusInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let evidence = G::new(&db, &txn, &arena)
.n_from_type("Evidence")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("status")
                    .map_or(false, |v| *v == data.st.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "evidence": evidence.iter().map(|evidence| Find_pending_evidenceEvidenceReturnType {
        id: uuid_str(evidence.id(), &arena),
        label: evidence.label(),
        episode_id: evidence.get_property("episode_id"),
        group_id: evidence.get_property("group_id"),
        status: evidence.get_property("status"),
        evidence_id: evidence.get_property("evidence_id"),
        fact_class: evidence.get_property("fact_class"),
        confidence: evidence.get_property("confidence"),
        source_type: evidence.get_property("source_type"),
        extractor_name: evidence.get_property("extractor_name"),
        payload_json: evidence.get_property("payload_json"),
        source_span: evidence.get_property("source_span"),
        signals_json: evidence.get_property("signals_json"),
        ambiguity_tags_json: evidence.get_property("ambiguity_tags_json"),
        ambiguity_score: evidence.get_property("ambiguity_score"),
        adjudication_request_id: evidence.get_property("adjudication_request_id"),
        commit_reason: evidence.get_property("commit_reason"),
        committed_id: evidence.get_property("committed_id"),
        deferred_cycles: evidence.get_property("deferred_cycles"),
        created_at: evidence.get_property("created_at"),
        resolved_at: evidence.get_property("resolved_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_episodes_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_episodes_by_groupEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn find_episodes_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_episodes_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Find_episodes_by_groupEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_adjudicationInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_adjudicationRequestReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub request_id: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub evidence_ids_json: Option<&'a Value>,
    pub selected_text: Option<&'a Value>,
    pub request_reason: Option<&'a Value>,
    pub resolution_source: Option<&'a Value>,
    pub resolution_payload_json: Option<&'a Value>,
    pub attempt_count: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler]
pub fn get_adjudication (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_adjudicationInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let request = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "request": Get_adjudicationRequestReturnType {
        id: uuid_str(request.id(), &arena),
        label: request.label(),
        episode_id: request.get_property("episode_id"),
        group_id: request.get_property("group_id"),
        status: request.get_property("status"),
        request_id: request.get_property("request_id"),
        ambiguity_tags_json: request.get_property("ambiguity_tags_json"),
        evidence_ids_json: request.get_property("evidence_ids_json"),
        selected_text: request.get_property("selected_text"),
        request_reason: request.get_property("request_reason"),
        resolution_source: request.get_property("resolution_source"),
        resolution_payload_json: request.get_property("resolution_payload_json"),
        attempt_count: request.get_property("attempt_count"),
        created_at: request.get_property("created_at"),
        resolved_at: request.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_cueInput {

pub id: ID,
pub cue_version: i32,
pub discourse_class: String,
pub cue_text: String,
pub supporting_spans_json: String,
pub temporal_markers_json: String,
pub quote_spans_json: String,
pub contradiction_keys_json: String,
pub first_spans_json: String,
pub projection_state: String,
pub cue_score: f64,
pub salience_score: f64,
pub projection_priority: f64,
pub route_reason: String,
pub hit_count: i32,
pub surfaced_count: i32,
pub selected_count: i32,
pub used_count: i32,
pub near_miss_count: i32,
pub policy_score: f64,
pub projection_attempts: i32,
pub last_hit_at: String,
pub last_feedback_at: String,
pub last_projected_at: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Update_cueCueReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub cue_version: Option<&'a Value>,
    pub discourse_class: Option<&'a Value>,
    pub cue_text: Option<&'a Value>,
    pub supporting_spans_json: Option<&'a Value>,
    pub temporal_markers_json: Option<&'a Value>,
    pub quote_spans_json: Option<&'a Value>,
    pub contradiction_keys_json: Option<&'a Value>,
    pub first_spans_json: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub cue_score: Option<&'a Value>,
    pub salience_score: Option<&'a Value>,
    pub projection_priority: Option<&'a Value>,
    pub route_reason: Option<&'a Value>,
    pub hit_count: Option<&'a Value>,
    pub surfaced_count: Option<&'a Value>,
    pub selected_count: Option<&'a Value>,
    pub used_count: Option<&'a Value>,
    pub near_miss_count: Option<&'a Value>,
    pub policy_score: Option<&'a Value>,
    pub projection_attempts: Option<&'a Value>,
    pub last_hit_at: Option<&'a Value>,
    pub last_feedback_at: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_cue (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_cueInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let cue = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("cue_version", Value::from(&data.cue_version)), ("discourse_class", Value::from(&data.discourse_class)), ("cue_text", Value::from(&data.cue_text)), ("supporting_spans_json", Value::from(&data.supporting_spans_json)), ("temporal_markers_json", Value::from(&data.temporal_markers_json)), ("quote_spans_json", Value::from(&data.quote_spans_json)), ("contradiction_keys_json", Value::from(&data.contradiction_keys_json)), ("first_spans_json", Value::from(&data.first_spans_json)), ("projection_state", Value::from(&data.projection_state)), ("cue_score", Value::from(&data.cue_score)), ("salience_score", Value::from(&data.salience_score)), ("projection_priority", Value::from(&data.projection_priority)), ("route_reason", Value::from(&data.route_reason)), ("hit_count", Value::from(&data.hit_count)), ("surfaced_count", Value::from(&data.surfaced_count)), ("selected_count", Value::from(&data.selected_count)), ("used_count", Value::from(&data.used_count)), ("near_miss_count", Value::from(&data.near_miss_count)), ("policy_score", Value::from(&data.policy_score)), ("projection_attempts", Value::from(&data.projection_attempts)), ("last_hit_at", Value::from(&data.last_hit_at)), ("last_feedback_at", Value::from(&data.last_feedback_at)), ("last_projected_at", Value::from(&data.last_projected_at)), ("updated_at", Value::from(&data.updated_at))])
    .collect_to_obj()?};
let response = json!({
    "cue": Update_cueCueReturnType {
        id: uuid_str(cue.id(), &arena),
        label: cue.label(),
        episode_id: cue.get_property("episode_id"),
        group_id: cue.get_property("group_id"),
        cue_version: cue.get_property("cue_version"),
        discourse_class: cue.get_property("discourse_class"),
        cue_text: cue.get_property("cue_text"),
        supporting_spans_json: cue.get_property("supporting_spans_json"),
        temporal_markers_json: cue.get_property("temporal_markers_json"),
        quote_spans_json: cue.get_property("quote_spans_json"),
        contradiction_keys_json: cue.get_property("contradiction_keys_json"),
        first_spans_json: cue.get_property("first_spans_json"),
        projection_state: cue.get_property("projection_state"),
        cue_score: cue.get_property("cue_score"),
        salience_score: cue.get_property("salience_score"),
        projection_priority: cue.get_property("projection_priority"),
        route_reason: cue.get_property("route_reason"),
        hit_count: cue.get_property("hit_count"),
        surfaced_count: cue.get_property("surfaced_count"),
        selected_count: cue.get_property("selected_count"),
        used_count: cue.get_property("used_count"),
        near_miss_count: cue.get_property("near_miss_count"),
        policy_score: cue.get_property("policy_score"),
        projection_attempts: cue.get_property("projection_attempts"),
        last_hit_at: cue.get_property("last_hit_at"),
        last_feedback_at: cue.get_property("last_feedback_at"),
        last_projected_at: cue.get_property("last_projected_at"),
        created_at: cue.get_property("created_at"),
        updated_at: cue.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_cue_by_episodeInput {

pub ep_id: String,
pub gid: String,
pub cue_version: i32,
pub discourse_class: String,
pub cue_text: String,
pub supporting_spans_json: String,
pub temporal_markers_json: String,
pub quote_spans_json: String,
pub contradiction_keys_json: String,
pub first_spans_json: String,
pub projection_state: String,
pub cue_score: f64,
pub salience_score: f64,
pub projection_priority: f64,
pub route_reason: String,
pub hit_count: i32,
pub surfaced_count: i32,
pub selected_count: i32,
pub used_count: i32,
pub near_miss_count: i32,
pub policy_score: f64,
pub projection_attempts: i32,
pub last_hit_at: String,
pub last_feedback_at: String,
pub last_projected_at: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Update_cue_by_episodeCueReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub cue_version: Option<&'a Value>,
    pub discourse_class: Option<&'a Value>,
    pub cue_text: Option<&'a Value>,
    pub supporting_spans_json: Option<&'a Value>,
    pub temporal_markers_json: Option<&'a Value>,
    pub quote_spans_json: Option<&'a Value>,
    pub contradiction_keys_json: Option<&'a Value>,
    pub first_spans_json: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub cue_score: Option<&'a Value>,
    pub salience_score: Option<&'a Value>,
    pub projection_priority: Option<&'a Value>,
    pub route_reason: Option<&'a Value>,
    pub hit_count: Option<&'a Value>,
    pub surfaced_count: Option<&'a Value>,
    pub selected_count: Option<&'a Value>,
    pub used_count: Option<&'a Value>,
    pub near_miss_count: Option<&'a Value>,
    pub policy_score: Option<&'a Value>,
    pub projection_attempts: Option<&'a Value>,
    pub last_hit_at: Option<&'a Value>,
    pub last_feedback_at: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_cue_by_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_cue_by_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let cue = {let update_tr = G::new(&db, &txn, &arena)
.n_from_type("EpisodeCue")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("episode_id")
                    .map_or(false, |v| *v == data.ep_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("cue_version", Value::from(&data.cue_version)), ("discourse_class", Value::from(&data.discourse_class)), ("cue_text", Value::from(&data.cue_text)), ("supporting_spans_json", Value::from(&data.supporting_spans_json)), ("temporal_markers_json", Value::from(&data.temporal_markers_json)), ("quote_spans_json", Value::from(&data.quote_spans_json)), ("contradiction_keys_json", Value::from(&data.contradiction_keys_json)), ("first_spans_json", Value::from(&data.first_spans_json)), ("projection_state", Value::from(&data.projection_state)), ("cue_score", Value::from(&data.cue_score)), ("salience_score", Value::from(&data.salience_score)), ("projection_priority", Value::from(&data.projection_priority)), ("route_reason", Value::from(&data.route_reason)), ("hit_count", Value::from(&data.hit_count)), ("surfaced_count", Value::from(&data.surfaced_count)), ("selected_count", Value::from(&data.selected_count)), ("used_count", Value::from(&data.used_count)), ("near_miss_count", Value::from(&data.near_miss_count)), ("policy_score", Value::from(&data.policy_score)), ("projection_attempts", Value::from(&data.projection_attempts)), ("last_hit_at", Value::from(&data.last_hit_at)), ("last_feedback_at", Value::from(&data.last_feedback_at)), ("last_projected_at", Value::from(&data.last_projected_at)), ("updated_at", Value::from(&data.updated_at))])
    .collect_to_obj()?};
let response = json!({
    "cue": Update_cue_by_episodeCueReturnType {
        id: uuid_str(cue.id(), &arena),
        label: cue.label(),
        episode_id: cue.get_property("episode_id"),
        group_id: cue.get_property("group_id"),
        cue_version: cue.get_property("cue_version"),
        discourse_class: cue.get_property("discourse_class"),
        cue_text: cue.get_property("cue_text"),
        supporting_spans_json: cue.get_property("supporting_spans_json"),
        temporal_markers_json: cue.get_property("temporal_markers_json"),
        quote_spans_json: cue.get_property("quote_spans_json"),
        contradiction_keys_json: cue.get_property("contradiction_keys_json"),
        first_spans_json: cue.get_property("first_spans_json"),
        projection_state: cue.get_property("projection_state"),
        cue_score: cue.get_property("cue_score"),
        salience_score: cue.get_property("salience_score"),
        projection_priority: cue.get_property("projection_priority"),
        route_reason: cue.get_property("route_reason"),
        hit_count: cue.get_property("hit_count"),
        surfaced_count: cue.get_property("surfaced_count"),
        selected_count: cue.get_property("selected_count"),
        used_count: cue.get_property("used_count"),
        near_miss_count: cue.get_property("near_miss_count"),
        policy_score: cue.get_property("policy_score"),
        projection_attempts: cue.get_property("projection_attempts"),
        last_hit_at: cue.get_property("last_hit_at"),
        last_feedback_at: cue.get_property("last_feedback_at"),
        last_projected_at: cue.get_property("last_projected_at"),
        created_at: cue.get_property("created_at"),
        updated_at: cue.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_atlas_region_membersInput {

pub snap_id: String,
pub region_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_atlas_region_membersMembersReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub snapshot_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub region_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
}

#[handler]
pub fn find_atlas_region_members (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_atlas_region_membersInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let members = G::new(&db, &txn, &arena)
.n_from_type("AtlasRegionMember")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("snapshot_id")
                    .map_or(false, |v| *v == data.snap_id.clone()) && val
                    .get_property("region_id")
                    .map_or(false, |v| *v == data.region_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "members": members.iter().map(|member| Find_atlas_region_membersMembersReturnType {
        id: uuid_str(member.id(), &arena),
        label: member.label(),
        snapshot_id: member.get_property("snapshot_id"),
        group_id: member.get_property("group_id"),
        region_id: member.get_property("region_id"),
        entity_id: member.get_property("entity_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_cues_embedInput {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_cues_embedResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_cues_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_cues_embedInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.query);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&__internal_embed_data_0, data.k.clone(), "CueVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_cues_embedResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_conversation_entitiesInput {

pub conv_id: ID
}
#[derive(Serialize, Default)]
pub struct Find_conversation_entitiesEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_conversation_entities (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_conversation_entitiesInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_id(&data.conv_id)

.out_node("HasConversationEntity").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_conversation_entitiesEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_consol_cycleInput {

pub id: ID,
pub status: String,
pub phase_results_json: String,
pub completed_at: f64,
pub total_duration_ms: f64,
pub error: String
}
#[derive(Serialize, Default)]
pub struct Update_consol_cycleCycleReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub cycle_id: Option<&'a Value>,
    pub trigger: Option<&'a Value>,
    pub dry_run: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub phase_results_json: Option<&'a Value>,
    pub started_at: Option<&'a Value>,
    pub completed_at: Option<&'a Value>,
    pub total_duration_ms: Option<&'a Value>,
    pub error: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_consol_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_consol_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let cycle = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("status", Value::from(&data.status)), ("phase_results_json", Value::from(&data.phase_results_json)), ("completed_at", Value::from(&data.completed_at)), ("total_duration_ms", Value::from(&data.total_duration_ms)), ("error", Value::from(&data.error))])
    .collect_to_obj()?};
let response = json!({
    "cycle": Update_consol_cycleCycleReturnType {
        id: uuid_str(cycle.id(), &arena),
        label: cycle.label(),
        group_id: cycle.get_property("group_id"),
        cycle_id: cycle.get_property("cycle_id"),
        trigger: cycle.get_property("trigger"),
        dry_run: cycle.get_property("dry_run"),
        status: cycle.get_property("status"),
        phase_results_json: cycle.get_property("phase_results_json"),
        started_at: cycle.get_property("started_at"),
        completed_at: cycle.get_property("completed_at"),
        total_duration_ms: cycle.get_property("total_duration_ms"),
        error: cycle.get_property("error"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_intentions_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_intentions_by_groupIntentionsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler]
pub fn find_intentions_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_intentions_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let intentions = G::new(&db, &txn, &arena)
.n_from_type("Intention")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "intentions": intentions.iter().map(|intention| Find_intentions_by_groupIntentionsReturnType {
        id: uuid_str(intention.id(), &arena),
        label: intention.label(),
        group_id: intention.get_property("group_id"),
        intention_id: intention.get_property("intention_id"),
        trigger_text: intention.get_property("trigger_text"),
        action_text: intention.get_property("action_text"),
        entity_names_json: intention.get_property("entity_names_json"),
        enabled: intention.get_property("enabled"),
        fire_count: intention.get_property("fire_count"),
        max_fires: intention.get_property("max_fires"),
        created_at: intention.get_property("created_at"),
        updated_at: intention.get_property("updated_at"),
        deleted_at: intention.get_property("deleted_at"),
        is_deleted: intention.get_property("is_deleted"),
        context_json: intention.get_property("context_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_complement_tagInput {

pub target_type: String,
pub target_id: String,
pub tag_type: String,
pub score: f64,
pub cycle_tagged: i32,
pub cycle_confirmed: i32,
pub cleared: bool,
pub group_id: String,
pub created_at: String,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Create_complement_tagNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_complement_tag (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_complement_tagInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ComplementTag", Some(ImmutablePropertiesMap::new(10, vec![("cleared", Value::from(&data.cleared)), ("group_id", Value::from(&data.group_id)), ("tag_type", Value::from(&data.tag_type)), ("created_at", Value::from(&data.created_at)), ("cycle_tagged", Value::from(&data.cycle_tagged)), ("updated_at", Value::from(&data.updated_at)), ("cycle_confirmed", Value::from(&data.cycle_confirmed)), ("target_id", Value::from(&data.target_id)), ("target_type", Value::from(&data.target_type)), ("score", Value::from(&data.score))].into_iter(), &arena)), Some(&["target_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_complement_tagNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        target_id: node.get_property("target_id"),
        group_id: node.get_property("group_id"),
        target_type: node.get_property("target_type"),
        tag_type: node.get_property("tag_type"),
        cycle_tagged: node.get_property("cycle_tagged"),
        cycle_confirmed: node.get_property("cycle_confirmed"),
        cleared: node.get_property("cleared"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_atlas_snapshotInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_atlas_snapshotSnapshotReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub snapshot_id: Option<&'a Value>,
    pub generated_at: Option<&'a Value>,
    pub represented_entity_count: Option<&'a Value>,
    pub represented_edge_count: Option<&'a Value>,
    pub displayed_node_count: Option<&'a Value>,
    pub displayed_edge_count: Option<&'a Value>,
    pub total_entities: Option<&'a Value>,
    pub total_relationships: Option<&'a Value>,
    pub total_regions: Option<&'a Value>,
    pub hottest_region_id: Option<&'a Value>,
    pub fastest_growing_region_id: Option<&'a Value>,
    pub truncated: Option<&'a Value>,
}

#[handler]
pub fn get_atlas_snapshot (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_atlas_snapshotInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let snapshot = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "snapshot": Get_atlas_snapshotSnapshotReturnType {
        id: uuid_str(snapshot.id(), &arena),
        label: snapshot.label(),
        group_id: snapshot.get_property("group_id"),
        snapshot_id: snapshot.get_property("snapshot_id"),
        generated_at: snapshot.get_property("generated_at"),
        represented_entity_count: snapshot.get_property("represented_entity_count"),
        represented_edge_count: snapshot.get_property("represented_edge_count"),
        displayed_node_count: snapshot.get_property("displayed_node_count"),
        displayed_edge_count: snapshot.get_property("displayed_edge_count"),
        total_entities: snapshot.get_property("total_entities"),
        total_relationships: snapshot.get_property("total_relationships"),
        total_regions: snapshot.get_property("total_regions"),
        hottest_region_id: snapshot.get_property("hottest_region_id"),
        fastest_growing_region_id: snapshot.get_property("fastest_growing_region_id"),
        truncated: snapshot.get_property("truncated"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_replayInput {

pub replay_id: String,
pub cycle_id: String,
pub group_id: String,
pub episode_id: String,
pub new_entities_found: i32,
pub new_relationships_found: i32,
pub entities_updated: i32,
pub skipped_reason: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_replayNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub replay_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub new_entities_found: Option<&'a Value>,
    pub new_relationships_found: Option<&'a Value>,
    pub entities_updated: Option<&'a Value>,
    pub skipped_reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_replay (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_replayInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolReplay", Some(ImmutablePropertiesMap::new(9, vec![("cycle_id", Value::from(&data.cycle_id)), ("timestamp", Value::from(&data.timestamp)), ("new_relationships_found", Value::from(&data.new_relationships_found)), ("entities_updated", Value::from(&data.entities_updated)), ("episode_id", Value::from(&data.episode_id)), ("new_entities_found", Value::from(&data.new_entities_found)), ("replay_id", Value::from(&data.replay_id)), ("group_id", Value::from(&data.group_id)), ("skipped_reason", Value::from(&data.skipped_reason))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_replayNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        replay_id: node.get_property("replay_id"),
        episode_id: node.get_property("episode_id"),
        new_entities_found: node.get_property("new_entities_found"),
        new_relationships_found: node.get_property("new_relationships_found"),
        entities_updated: node.get_property("entities_updated"),
        skipped_reason: node.get_property("skipped_reason"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_dream_associationInput {

pub assoc_id: String,
pub cycle_id: String,
pub group_id: String,
pub source_entity_id: String,
pub target_entity_id: String,
pub source_entity_name: String,
pub target_entity_name: String,
pub source_domain: String,
pub target_domain: String,
pub surprise_score: f64,
pub embedding_similarity: f64,
pub structural_proximity: f64,
pub relationship_id: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_dream_associationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub assoc_id: Option<&'a Value>,
    pub source_entity_id: Option<&'a Value>,
    pub target_entity_id: Option<&'a Value>,
    pub source_entity_name: Option<&'a Value>,
    pub target_entity_name: Option<&'a Value>,
    pub source_domain: Option<&'a Value>,
    pub target_domain: Option<&'a Value>,
    pub surprise_score: Option<&'a Value>,
    pub embedding_similarity: Option<&'a Value>,
    pub structural_proximity: Option<&'a Value>,
    pub relationship_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_dream_association (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_dream_associationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolDreamAssociation", Some(ImmutablePropertiesMap::new(14, vec![("structural_proximity", Value::from(&data.structural_proximity)), ("cycle_id", Value::from(&data.cycle_id)), ("target_entity_name", Value::from(&data.target_entity_name)), ("target_domain", Value::from(&data.target_domain)), ("group_id", Value::from(&data.group_id)), ("source_domain", Value::from(&data.source_domain)), ("surprise_score", Value::from(&data.surprise_score)), ("timestamp", Value::from(&data.timestamp)), ("target_entity_id", Value::from(&data.target_entity_id)), ("relationship_id", Value::from(&data.relationship_id)), ("source_entity_name", Value::from(&data.source_entity_name)), ("embedding_similarity", Value::from(&data.embedding_similarity)), ("assoc_id", Value::from(&data.assoc_id)), ("source_entity_id", Value::from(&data.source_entity_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_dream_associationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        assoc_id: node.get_property("assoc_id"),
        source_entity_id: node.get_property("source_entity_id"),
        target_entity_id: node.get_property("target_entity_id"),
        source_entity_name: node.get_property("source_entity_name"),
        target_entity_name: node.get_property("target_entity_name"),
        source_domain: node.get_property("source_domain"),
        target_domain: node.get_property("target_domain"),
        surprise_score: node.get_property("surprise_score"),
        embedding_similarity: node.get_property("embedding_similarity"),
        structural_proximity: node.get_property("structural_proximity"),
        relationship_id: node.get_property("relationship_id"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_entity_neighborhoodInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_entity_neighborhoodNeighborsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[derive(Serialize, Default)]
pub struct Get_entity_neighborhoodEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_entity_neighborhood (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_entity_neighborhoodInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_e("RelatesTo").collect::<Result<Vec<_>, _>>()?;
    let neighbors = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_node("RelatesTo").collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "neighbors": neighbors.iter().map(|neighbor| Get_entity_neighborhoodNeighborsReturnType {
        id: uuid_str(neighbor.id(), &arena),
        label: neighbor.label(),
        name: neighbor.get_property("name"),
        group_id: neighbor.get_property("group_id"),
        entity_type: neighbor.get_property("entity_type"),
        canonical_identifier: neighbor.get_property("canonical_identifier"),
        entity_id: neighbor.get_property("entity_id"),
        summary: neighbor.get_property("summary"),
        attributes_json: neighbor.get_property("attributes_json"),
        created_at: neighbor.get_property("created_at"),
        updated_at: neighbor.get_property("updated_at"),
        is_deleted: neighbor.get_property("is_deleted"),
        deleted_at: neighbor.get_property("deleted_at"),
        identity_core: neighbor.get_property("identity_core"),
        mat_tier: neighbor.get_property("mat_tier"),
        recon_count: neighbor.get_property("recon_count"),
        lexical_regime: neighbor.get_property("lexical_regime"),
        identifier_label: neighbor.get_property("identifier_label"),
        pii_detected: neighbor.get_property("pii_detected"),
        pii_categories_json: neighbor.get_property("pii_categories_json"),
        access_count: neighbor.get_property("access_count"),
        last_accessed: neighbor.get_property("last_accessed"),
        source_episode_ids: neighbor.get_property("source_episode_ids"),
        evidence_count: neighbor.get_property("evidence_count"),
        evidence_span_start: neighbor.get_property("evidence_span_start"),
        evidence_span_end: neighbor.get_property("evidence_span_end"),
    }).collect::<Vec<_>>(),
    "edges": edges.iter().map(|edge| Get_entity_neighborhoodEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_cues_bm25Input {

pub query: String,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_cues_bm25ResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub cue_text: Option<&'a Value>,
    pub supporting_spans_json: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn search_cues_bm25 (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_cues_bm25Input>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_bm25("EpisodeCue", &data.query, data.k.clone())?.collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_cues_bm25ResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        cue_text: result.get_property("cue_text"),
        supporting_spans_json: result.get_property("supporting_spans_json"),
        projection_state: result.get_property("projection_state"),
        created_at: result.get_property("created_at"),
        updated_at: result.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_episodes_by_group_limitedInput {

pub gid: String,
pub limit: i64
}
#[derive(Serialize, Default)]
pub struct Find_episodes_by_group_limitedEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn find_episodes_by_group_limited (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_episodes_by_group_limitedInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            })

.range(0, data.limit.clone()).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Find_episodes_by_group_limitedEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct shortest_path_bfsInput {

pub start: ID,
pub end: ID
}
#[derive(Serialize, Default)]
pub struct Shortest_path_bfsPathReturnType {
}

#[handler]
pub fn shortest_path_bfs (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<shortest_path_bfsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let path = G::new(&db, &txn, &arena)
.n_from_id(&data.start)

.shortest_path_with_algorithm(Some("RelatesTo"), None, Some(&data.end), PathAlgorithm::BFS, helix_db::helix_engine::traversal_core::ops::util::paths::default_weight_fn).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "path": path.iter().map(|path| Shortest_path_bfsPathReturnType {
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_semantic_transitionInput {

pub trans_id: String,
pub cycle_id: String,
pub group_id: String,
pub episode_id: String,
pub old_tier: String,
pub new_tier: String,
pub entity_coverage: f64,
pub consolidation_cycles: i32,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_semantic_transitionNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub trans_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub old_tier: Option<&'a Value>,
    pub new_tier: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_semantic_transition (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_semantic_transitionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolSemanticTransition", Some(ImmutablePropertiesMap::new(9, vec![("entity_coverage", Value::from(&data.entity_coverage)), ("old_tier", Value::from(&data.old_tier)), ("new_tier", Value::from(&data.new_tier)), ("cycle_id", Value::from(&data.cycle_id)), ("group_id", Value::from(&data.group_id)), ("consolidation_cycles", Value::from(&data.consolidation_cycles)), ("trans_id", Value::from(&data.trans_id)), ("timestamp", Value::from(&data.timestamp)), ("episode_id", Value::from(&data.episode_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_semantic_transitionNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        trans_id: node.get_property("trans_id"),
        episode_id: node.get_property("episode_id"),
        old_tier: node.get_property("old_tier"),
        new_tier: node.get_property("new_tier"),
        entity_coverage: node.get_property("entity_coverage"),
        consolidation_cycles: node.get_property("consolidation_cycles"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_by_canonicalInput {

pub canon: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_by_canonicalEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_by_canonical (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_by_canonicalInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("canonical_identifier")
                    .map_or(false, |v| *v == data.canon.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_by_canonicalEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_decision_outcomeInput {

pub outcome_id: String,
pub cycle_id: String,
pub group_id: String,
pub phase: String,
pub decision_trace_id: String,
pub outcome_type: String,
pub outcome_label: String,
pub outcome_value: f64,
pub metadata_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_decision_outcomeNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub outcome_id: Option<&'a Value>,
    pub phase: Option<&'a Value>,
    pub decision_trace_id: Option<&'a Value>,
    pub outcome_type: Option<&'a Value>,
    pub outcome_label: Option<&'a Value>,
    pub outcome_value: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_decision_outcome (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_decision_outcomeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolDecisionOutcome", Some(ImmutablePropertiesMap::new(10, vec![("timestamp", Value::from(&data.timestamp)), ("metadata_json", Value::from(&data.metadata_json)), ("phase", Value::from(&data.phase)), ("decision_trace_id", Value::from(&data.decision_trace_id)), ("outcome_label", Value::from(&data.outcome_label)), ("group_id", Value::from(&data.group_id)), ("outcome_type", Value::from(&data.outcome_type)), ("outcome_value", Value::from(&data.outcome_value)), ("outcome_id", Value::from(&data.outcome_id)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_decision_outcomeNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        outcome_id: node.get_property("outcome_id"),
        phase: node.get_property("phase"),
        decision_trace_id: node.get_property("decision_trace_id"),
        outcome_type: node.get_property("outcome_type"),
        outcome_label: node.get_property("outcome_label"),
        outcome_value: node.get_property("outcome_value"),
        metadata_json: node.get_property("metadata_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_identifier_reviews_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_identifier_reviews_by_cycleReviewsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub review_id: Option<&'a Value>,
    pub entity_a_id: Option<&'a Value>,
    pub entity_b_id: Option<&'a Value>,
    pub entity_a_name: Option<&'a Value>,
    pub entity_b_name: Option<&'a Value>,
    pub entity_a_type: Option<&'a Value>,
    pub entity_b_type: Option<&'a Value>,
    pub raw_similarity: Option<&'a Value>,
    pub adjusted_similarity: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub decision_reason: Option<&'a Value>,
    pub entity_a_regime: Option<&'a Value>,
    pub entity_b_regime: Option<&'a Value>,
    pub canonical_identifier_a: Option<&'a Value>,
    pub canonical_identifier_b: Option<&'a Value>,
    pub review_status: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_identifier_reviews_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_identifier_reviews_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let reviews = G::new(&db, &txn, &arena)
.n_from_type("ConsolIdentifierReview")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "reviews": reviews.iter().map(|review| Find_consol_identifier_reviews_by_cycleReviewsReturnType {
        id: uuid_str(review.id(), &arena),
        label: review.label(),
        cycle_id: review.get_property("cycle_id"),
        group_id: review.get_property("group_id"),
        review_id: review.get_property("review_id"),
        entity_a_id: review.get_property("entity_a_id"),
        entity_b_id: review.get_property("entity_b_id"),
        entity_a_name: review.get_property("entity_a_name"),
        entity_b_name: review.get_property("entity_b_name"),
        entity_a_type: review.get_property("entity_a_type"),
        entity_b_type: review.get_property("entity_b_type"),
        raw_similarity: review.get_property("raw_similarity"),
        adjusted_similarity: review.get_property("adjusted_similarity"),
        decision_source: review.get_property("decision_source"),
        decision_reason: review.get_property("decision_reason"),
        entity_a_regime: review.get_property("entity_a_regime"),
        entity_b_regime: review.get_property("entity_b_regime"),
        canonical_identifier_a: review.get_property("canonical_identifier_a"),
        canonical_identifier_b: review.get_property("canonical_identifier_b"),
        review_status: review.get_property("review_status"),
        metadata_json: review.get_property("metadata_json"),
        timestamp: review.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct shortest_path_weightedInput {

pub start: ID,
pub end: ID
}
#[derive(Serialize, Default)]
pub struct Shortest_path_weightedPathReturnType {
}

#[handler]
pub fn shortest_path_weighted (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<shortest_path_weightedInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let path = G::new(&db, &txn, &arena)
.n_from_id(&data.start)

.shortest_path_with_algorithm(Some("RelatesTo"), None, Some(&data.end), PathAlgorithm::Dijkstra, |edge, src_node, dst_node| -> Result<f64, GraphError> { Ok((edge.get_property("weight").ok_or(GraphError::Default)?.as_f64())) }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "path": path.iter().map(|path| Shortest_path_weightedPathReturnType {
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_cue_vectors_filteredInput {

pub vec: Vec<f64>,
pub k: i32,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Search_cue_vectors_filteredResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_cue_vectors_filtered (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_cue_vectors_filteredInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "CueVec", None)

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_cue_vectors_filteredResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_adjudicationInput {

pub request_id: String,
pub episode_id: String,
pub group_id: String,
pub status: String,
pub ambiguity_tags_json: String,
pub evidence_ids_json: String,
pub selected_text: String,
pub request_reason: String,
pub resolution_source: String,
pub resolution_payload_json: String,
pub attempt_count: i32,
pub created_at: String,
pub resolved_at: String
}
#[derive(Serialize, Default)]
pub struct Create_adjudicationNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub request_id: Option<&'a Value>,
    pub ambiguity_tags_json: Option<&'a Value>,
    pub evidence_ids_json: Option<&'a Value>,
    pub selected_text: Option<&'a Value>,
    pub request_reason: Option<&'a Value>,
    pub resolution_source: Option<&'a Value>,
    pub resolution_payload_json: Option<&'a Value>,
    pub attempt_count: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub resolved_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_adjudication (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_adjudicationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("AdjudicationRequest", Some(ImmutablePropertiesMap::new(13, vec![("resolution_source", Value::from(&data.resolution_source)), ("attempt_count", Value::from(&data.attempt_count)), ("status", Value::from(&data.status)), ("resolution_payload_json", Value::from(&data.resolution_payload_json)), ("selected_text", Value::from(&data.selected_text)), ("ambiguity_tags_json", Value::from(&data.ambiguity_tags_json)), ("request_id", Value::from(&data.request_id)), ("resolved_at", Value::from(&data.resolved_at)), ("episode_id", Value::from(&data.episode_id)), ("created_at", Value::from(&data.created_at)), ("request_reason", Value::from(&data.request_reason)), ("evidence_ids_json", Value::from(&data.evidence_ids_json)), ("group_id", Value::from(&data.group_id))].into_iter(), &arena)), Some(&["episode_id", "group_id", "status"])).collect_to_obj()?;
let response = json!({
    "node": Create_adjudicationNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        episode_id: node.get_property("episode_id"),
        group_id: node.get_property("group_id"),
        status: node.get_property("status"),
        request_id: node.get_property("request_id"),
        ambiguity_tags_json: node.get_property("ambiguity_tags_json"),
        evidence_ids_json: node.get_property("evidence_ids_json"),
        selected_text: node.get_property("selected_text"),
        request_reason: node.get_property("request_reason"),
        resolution_source: node.get_property("resolution_source"),
        resolution_payload_json: node.get_property("resolution_payload_json"),
        attempt_count: node.get_property("attempt_count"),
        created_at: node.get_property("created_at"),
        resolved_at: node.get_property("resolved_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_episodeInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_episodeEpisodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn get_episode (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_episodeInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episode = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "episode": Get_episodeEpisodeReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct link_episode_entityInput {

pub episode_id: ID,
pub entity_id: ID
}
#[derive(Serialize, Default)]
pub struct Link_episode_entityEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
}

#[handler(is_write)]
pub fn link_episode_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<link_episode_entityInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = G::new_mut(&db, &arena, &mut txn)
.add_edge("HasEntity", None, *data.episode_id, *data.entity_id, false, false).collect_to_obj()?;
let response = json!({
    "edge": Link_episode_entityEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_intentionInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_intention (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_intentionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_atlas_regionInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_atlas_region (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_atlas_regionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct hard_delete_adjudicationInput {

pub id: ID
}
#[handler(is_write)]
pub fn hard_delete_adjudication (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<hard_delete_adjudicationInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_conversation_messageInput {

pub conversation_id: String,
pub message_id: String,
pub role: String,
pub content: String,
pub parts_json: String,
pub created_at: String
}
#[derive(Serialize, Default)]
pub struct Create_conversation_messageNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub conversation_id: Option<&'a Value>,
    pub message_id: Option<&'a Value>,
    pub role: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub parts_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_conversation_message (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_conversation_messageInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConversationMessage", Some(ImmutablePropertiesMap::new(6, vec![("role", Value::from(&data.role)), ("content", Value::from(&data.content)), ("message_id", Value::from(&data.message_id)), ("conversation_id", Value::from(&data.conversation_id)), ("parts_json", Value::from(&data.parts_json)), ("created_at", Value::from(&data.created_at))].into_iter(), &arena)), Some(&["conversation_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_conversation_messageNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        conversation_id: node.get_property("conversation_id"),
        message_id: node.get_property("message_id"),
        role: node.get_property("role"),
        content: node.get_property("content"),
        parts_json: node.get_property("parts_json"),
        created_at: node.get_property("created_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct soft_delete_entityInput {

pub id: ID,
pub deleted_at: String
}
#[derive(Serialize, Default)]
pub struct Soft_delete_entityEntityReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler(is_write)]
pub fn soft_delete_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<soft_delete_entityInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let entity = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("is_deleted", Value::from(true)), ("deleted_at", Value::from(&data.deleted_at))])
    .collect_to_obj()?};
let response = json!({
    "entity": Soft_delete_entityEntityReturnType {
        id: uuid_str(entity.id(), &arena),
        label: entity.label(),
        name: entity.get_property("name"),
        group_id: entity.get_property("group_id"),
        entity_type: entity.get_property("entity_type"),
        canonical_identifier: entity.get_property("canonical_identifier"),
        entity_id: entity.get_property("entity_id"),
        summary: entity.get_property("summary"),
        attributes_json: entity.get_property("attributes_json"),
        created_at: entity.get_property("created_at"),
        updated_at: entity.get_property("updated_at"),
        is_deleted: entity.get_property("is_deleted"),
        deleted_at: entity.get_property("deleted_at"),
        identity_core: entity.get_property("identity_core"),
        mat_tier: entity.get_property("mat_tier"),
        recon_count: entity.get_property("recon_count"),
        lexical_regime: entity.get_property("lexical_regime"),
        identifier_label: entity.get_property("identifier_label"),
        pii_detected: entity.get_property("pii_detected"),
        pii_categories_json: entity.get_property("pii_categories_json"),
        access_count: entity.get_property("access_count"),
        last_accessed: entity.get_property("last_accessed"),
        source_episode_ids: entity.get_property("source_episode_ids"),
        evidence_count: entity.get_property("evidence_count"),
        evidence_span_start: entity.get_property("evidence_span_start"),
        evidence_span_end: entity.get_property("evidence_span_end"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct drop_edgeInput {

pub id: ID
}
#[handler(is_write)]
pub fn drop_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<drop_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.e_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_episode_chunk_embedInput {

pub episode_id: String,
pub group_id: String,
pub chunk_text: String,
pub chunk_index: i32,
pub content_type: String
}
#[derive(Serialize, Default)]
pub struct Create_episode_chunk_embedChunkReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_episode_chunk_embed (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_episode_chunk_embedInput>(&input.request.body)?.into_owned();
Err(IoContFn::create_err(move |__internal_cont_tx, __internal_ret_chan| Box::pin(async move {
let __internal_embed_data_0 = embed_async!(db, &data.chunk_text);
__internal_cont_tx.send_async((__internal_ret_chan, Box::new(move || {
let __internal_embed_data_0: Vec<f64> = __internal_embed_data_0?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let chunk = G::new_mut(&db, &arena, &mut txn)
.insert_v::<fn(&HVector, &RoTxn) -> bool>(&__internal_embed_data_0, "EpisodeChunk", Some(ImmutablePropertiesMap::new(5, vec![("chunk_text", Value::from(data.chunk_text.clone())), ("content_type", Value::from(data.content_type.clone())), ("chunk_index", Value::from(data.chunk_index.clone())), ("episode_id", Value::from(data.episode_id.clone())), ("group_id", Value::from(data.group_id.clone()))].into_iter(), &arena))).collect_to_obj()?;
let response = json!({
    "chunk": Create_episode_chunk_embedChunkReturnType {
        id: uuid_str(chunk.id(), &arena),
        label: chunk.label(),
        data: chunk.data(),
        score: chunk.score(),
        episode_id: chunk.get_property("episode_id"),
        group_id: chunk.get_property("group_id"),
        chunk_text: chunk.get_property("chunk_text"),
        chunk_index: chunk.get_property("chunk_index"),
        content_type: chunk.get_property("content_type"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}))).await.expect("Cont Channel should be alive")
})))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_episodes_by_sourceInput {

pub gid: String,
pub src: String
}
#[derive(Serialize, Default)]
pub struct Find_episodes_by_sourceEpisodesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub status: Option<&'a Value>,
    pub session_id: Option<&'a Value>,
    pub episode_id: Option<&'a Value>,
    pub content: Option<&'a Value>,
    pub source: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub error: Option<&'a Value>,
    pub retry_count: Option<&'a Value>,
    pub processing_duration_ms: Option<&'a Value>,
    pub skipped_meta: Option<&'a Value>,
    pub skipped_triage: Option<&'a Value>,
    pub encoding_context_json: Option<&'a Value>,
    pub memory_tier: Option<&'a Value>,
    pub consolidation_cycles: Option<&'a Value>,
    pub entity_coverage: Option<&'a Value>,
    pub projection_state: Option<&'a Value>,
    pub last_projection_reason: Option<&'a Value>,
    pub last_projected_at: Option<&'a Value>,
    pub conversation_date: Option<&'a Value>,
    pub attachments_json: Option<&'a Value>,
}

#[handler]
pub fn find_episodes_by_source (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_episodes_by_sourceInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let episodes = G::new(&db, &txn, &arena)
.n_from_type("Episode")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("source")
                    .map_or(false, |v| *v == data.src.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "episodes": episodes.iter().map(|episode| Find_episodes_by_sourceEpisodesReturnType {
        id: uuid_str(episode.id(), &arena),
        label: episode.label(),
        group_id: episode.get_property("group_id"),
        status: episode.get_property("status"),
        session_id: episode.get_property("session_id"),
        episode_id: episode.get_property("episode_id"),
        content: episode.get_property("content"),
        source: episode.get_property("source"),
        created_at: episode.get_property("created_at"),
        updated_at: episode.get_property("updated_at"),
        error: episode.get_property("error"),
        retry_count: episode.get_property("retry_count"),
        processing_duration_ms: episode.get_property("processing_duration_ms"),
        skipped_meta: episode.get_property("skipped_meta"),
        skipped_triage: episode.get_property("skipped_triage"),
        encoding_context_json: episode.get_property("encoding_context_json"),
        memory_tier: episode.get_property("memory_tier"),
        consolidation_cycles: episode.get_property("consolidation_cycles"),
        entity_coverage: episode.get_property("entity_coverage"),
        projection_state: episode.get_property("projection_state"),
        last_projection_reason: episode.get_property("last_projection_reason"),
        last_projected_at: episode.get_property("last_projected_at"),
        conversation_date: episode.get_property("conversation_date"),
        attachments_json: episode.get_property("attachments_json"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_intentionInput {

pub intention_id: String,
pub group_id: String,
pub trigger_text: String,
pub action_text: String,
pub entity_names_json: String,
pub enabled: bool,
pub fire_count: i32,
pub max_fires: i32,
pub created_at: String,
pub updated_at: String,
pub deleted_at: String,
pub is_deleted: bool,
pub context_json: String
}
#[derive(Serialize, Default)]
pub struct Create_intentionNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub intention_id: Option<&'a Value>,
    pub trigger_text: Option<&'a Value>,
    pub action_text: Option<&'a Value>,
    pub entity_names_json: Option<&'a Value>,
    pub enabled: Option<&'a Value>,
    pub fire_count: Option<&'a Value>,
    pub max_fires: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub context_json: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_intention (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_intentionInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("Intention", Some(ImmutablePropertiesMap::new(13, vec![("action_text", Value::from(&data.action_text)), ("fire_count", Value::from(&data.fire_count)), ("updated_at", Value::from(&data.updated_at)), ("enabled", Value::from(&data.enabled)), ("context_json", Value::from(&data.context_json)), ("deleted_at", Value::from(&data.deleted_at)), ("is_deleted", Value::from(&data.is_deleted)), ("max_fires", Value::from(&data.max_fires)), ("created_at", Value::from(&data.created_at)), ("entity_names_json", Value::from(&data.entity_names_json)), ("group_id", Value::from(&data.group_id)), ("trigger_text", Value::from(&data.trigger_text)), ("intention_id", Value::from(&data.intention_id))].into_iter(), &arena)), Some(&["group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_intentionNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        group_id: node.get_property("group_id"),
        intention_id: node.get_property("intention_id"),
        trigger_text: node.get_property("trigger_text"),
        action_text: node.get_property("action_text"),
        entity_names_json: node.get_property("entity_names_json"),
        enabled: node.get_property("enabled"),
        fire_count: node.get_property("fire_count"),
        max_fires: node.get_property("max_fires"),
        created_at: node.get_property("created_at"),
        updated_at: node.get_property("updated_at"),
        deleted_at: node.get_property("deleted_at"),
        is_deleted: node.get_property("is_deleted"),
        context_json: node.get_property("context_json"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct link_conversation_entityInput {

pub conv_id: ID,
pub entity_id: ID
}
#[derive(Serialize, Default)]
pub struct Link_conversation_entityEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
}

#[handler(is_write)]
pub fn link_conversation_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<link_conversation_entityInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = G::new_mut(&db, &arena, &mut txn)
.add_edge("HasConversationEntity", None, *data.conv_id, *data.entity_id, false, false).collect_to_obj()?;
let response = json!({
    "edge": Link_conversation_entityEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct delete_atlas_snapshotInput {

pub id: ID
}
#[handler(is_write)]
pub fn delete_atlas_snapshot (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<delete_atlas_snapshotInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    Drop::drop_traversal(
                G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect::<Vec<_>>().into_iter(),
                &db,
                &mut txn,
            )?;;
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&()))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_episode_chunks_filteredInput {

pub vec: Vec<f64>,
pub k: i32,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Search_episode_chunks_filteredResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub episode_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub chunk_text: Option<&'a Value>,
    pub chunk_index: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
}

#[handler]
pub fn search_episode_chunks_filtered (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_episode_chunks_filteredInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EpisodeChunk", None)

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_episode_chunks_filteredResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        episode_id: result.get_property("episode_id"),
        group_id: result.get_property("group_id"),
        chunk_text: result.get_property("chunk_text"),
        chunk_index: result.get_property("chunk_index"),
        content_type: result.get_property("content_type"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_entity_vectorsInput {

pub vec: Vec<f64>,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_entity_vectorsResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub content_type: Option<&'a Value>,
    pub embed_provider: Option<&'a Value>,
    pub embed_model: Option<&'a Value>,
}

#[handler]
pub fn search_entity_vectors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_entity_vectorsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "EntityVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_entity_vectorsResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        entity_id: result.get_property("entity_id"),
        group_id: result.get_property("group_id"),
        content_type: result.get_property("content_type"),
        embed_provider: result.get_property("embed_provider"),
        embed_model: result.get_property("embed_model"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_microgliaInput {

pub microglia_id: String,
pub cycle_id: String,
pub group_id: String,
pub target_type: String,
pub target_id: String,
pub action: String,
pub tag_type: String,
pub score: f64,
pub detail: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_microgliaNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub microglia_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub target_id: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub detail: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_microglia (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_microgliaInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolMicroglia", Some(ImmutablePropertiesMap::new(10, vec![("target_type", Value::from(&data.target_type)), ("action", Value::from(&data.action)), ("group_id", Value::from(&data.group_id)), ("microglia_id", Value::from(&data.microglia_id)), ("detail", Value::from(&data.detail)), ("target_id", Value::from(&data.target_id)), ("tag_type", Value::from(&data.tag_type)), ("timestamp", Value::from(&data.timestamp)), ("score", Value::from(&data.score)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_microgliaNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        microglia_id: node.get_property("microglia_id"),
        target_type: node.get_property("target_type"),
        target_id: node.get_property("target_id"),
        action: node.get_property("action"),
        tag_type: node.get_property("tag_type"),
        detail: node.get_property("detail"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_entities_exact_nameInput {

pub name_exact: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_entities_exact_nameEntitiesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn find_entities_exact_name (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_entities_exact_nameInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entities = G::new(&db, &txn, &arena)
.n_from_type("Entity")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()) && val
                    .get_property("name")
                    .map_or(false, |v| *v == data.name_exact.clone()) && val
                    .get_property("is_deleted")
                    .map_or(false, |v| *v == false)))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "entities": entities.iter().map(|entitie| Find_entities_exact_nameEntitiesReturnType {
        id: uuid_str(entitie.id(), &arena),
        label: entitie.label(),
        name: entitie.get_property("name"),
        group_id: entitie.get_property("group_id"),
        entity_type: entitie.get_property("entity_type"),
        canonical_identifier: entitie.get_property("canonical_identifier"),
        entity_id: entitie.get_property("entity_id"),
        summary: entitie.get_property("summary"),
        attributes_json: entitie.get_property("attributes_json"),
        created_at: entitie.get_property("created_at"),
        updated_at: entitie.get_property("updated_at"),
        is_deleted: entitie.get_property("is_deleted"),
        deleted_at: entitie.get_property("deleted_at"),
        identity_core: entitie.get_property("identity_core"),
        mat_tier: entitie.get_property("mat_tier"),
        recon_count: entitie.get_property("recon_count"),
        lexical_regime: entitie.get_property("lexical_regime"),
        identifier_label: entitie.get_property("identifier_label"),
        pii_detected: entitie.get_property("pii_detected"),
        pii_categories_json: entitie.get_property("pii_categories_json"),
        access_count: entitie.get_property("access_count"),
        last_accessed: entitie.get_property("last_accessed"),
        source_episode_ids: entitie.get_property("source_episode_ids"),
        evidence_count: entitie.get_property("evidence_count"),
        evidence_span_start: entitie.get_property("evidence_span_start"),
        evidence_span_end: entitie.get_property("evidence_span_end"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_outgoing_edges_by_predicateInput {

pub id: ID,
pub pred: String
}
#[derive(Serialize, Default)]
pub struct Get_outgoing_edges_by_predicateEdgesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler]
pub fn get_outgoing_edges_by_predicate (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_outgoing_edges_by_predicateInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let edges = G::new(&db, &txn, &arena)
.n_from_id(&data.id)

.out_e("RelatesTo")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("predicate")
                    .map_or(false, |v| *v == data.pred.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "edges": edges.iter().map(|edge| Get_outgoing_edges_by_predicateEdgesReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_identifier_reviewInput {

pub review_id: String,
pub cycle_id: String,
pub group_id: String,
pub entity_a_id: String,
pub entity_b_id: String,
pub entity_a_name: String,
pub entity_b_name: String,
pub entity_a_type: String,
pub entity_b_type: String,
pub raw_similarity: f64,
pub adjusted_similarity: f64,
pub decision_source: String,
pub decision_reason: String,
pub entity_a_regime: String,
pub entity_b_regime: String,
pub canonical_identifier_a: String,
pub canonical_identifier_b: String,
pub review_status: String,
pub metadata_json: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_identifier_reviewNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub review_id: Option<&'a Value>,
    pub entity_a_id: Option<&'a Value>,
    pub entity_b_id: Option<&'a Value>,
    pub entity_a_name: Option<&'a Value>,
    pub entity_b_name: Option<&'a Value>,
    pub entity_a_type: Option<&'a Value>,
    pub entity_b_type: Option<&'a Value>,
    pub raw_similarity: Option<&'a Value>,
    pub adjusted_similarity: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub decision_reason: Option<&'a Value>,
    pub entity_a_regime: Option<&'a Value>,
    pub entity_b_regime: Option<&'a Value>,
    pub canonical_identifier_a: Option<&'a Value>,
    pub canonical_identifier_b: Option<&'a Value>,
    pub review_status: Option<&'a Value>,
    pub metadata_json: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_identifier_review (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_identifier_reviewInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolIdentifierReview", Some(ImmutablePropertiesMap::new(20, vec![("entity_b_type", Value::from(&data.entity_b_type)), ("decision_reason", Value::from(&data.decision_reason)), ("entity_b_id", Value::from(&data.entity_b_id)), ("review_id", Value::from(&data.review_id)), ("entity_a_type", Value::from(&data.entity_a_type)), ("metadata_json", Value::from(&data.metadata_json)), ("review_status", Value::from(&data.review_status)), ("timestamp", Value::from(&data.timestamp)), ("canonical_identifier_a", Value::from(&data.canonical_identifier_a)), ("raw_similarity", Value::from(&data.raw_similarity)), ("entity_a_id", Value::from(&data.entity_a_id)), ("group_id", Value::from(&data.group_id)), ("entity_a_regime", Value::from(&data.entity_a_regime)), ("entity_b_name", Value::from(&data.entity_b_name)), ("entity_a_name", Value::from(&data.entity_a_name)), ("adjusted_similarity", Value::from(&data.adjusted_similarity)), ("entity_b_regime", Value::from(&data.entity_b_regime)), ("decision_source", Value::from(&data.decision_source)), ("canonical_identifier_b", Value::from(&data.canonical_identifier_b)), ("cycle_id", Value::from(&data.cycle_id))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_identifier_reviewNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        review_id: node.get_property("review_id"),
        entity_a_id: node.get_property("entity_a_id"),
        entity_b_id: node.get_property("entity_b_id"),
        entity_a_name: node.get_property("entity_a_name"),
        entity_b_name: node.get_property("entity_b_name"),
        entity_a_type: node.get_property("entity_a_type"),
        entity_b_type: node.get_property("entity_b_type"),
        raw_similarity: node.get_property("raw_similarity"),
        adjusted_similarity: node.get_property("adjusted_similarity"),
        decision_source: node.get_property("decision_source"),
        decision_reason: node.get_property("decision_reason"),
        entity_a_regime: node.get_property("entity_a_regime"),
        entity_b_regime: node.get_property("entity_b_regime"),
        canonical_identifier_a: node.get_property("canonical_identifier_a"),
        canonical_identifier_b: node.get_property("canonical_identifier_b"),
        review_status: node.get_property("review_status"),
        metadata_json: node.get_property("metadata_json"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_conversations_by_groupInput {

pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_conversations_by_groupConversationsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub group_id: Option<&'a Value>,
    pub conversation_id: Option<&'a Value>,
    pub title: Option<&'a Value>,
    pub session_date: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler]
pub fn find_conversations_by_group (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_conversations_by_groupInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let conversations = G::new(&db, &txn, &arena)
.n_from_type("Conversation")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok(val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone()))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "conversations": conversations.iter().map(|conversation| Find_conversations_by_groupConversationsReturnType {
        id: uuid_str(conversation.id(), &arena),
        label: conversation.label(),
        group_id: conversation.get_property("group_id"),
        conversation_id: conversation.get_property("conversation_id"),
        title: conversation.get_property("title"),
        session_date: conversation.get_property("session_date"),
        created_at: conversation.get_property("created_at"),
        updated_at: conversation.get_property("updated_at"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_maturations_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_maturations_by_cycleMaturationsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub mat_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub old_tier: Option<&'a Value>,
    pub new_tier: Option<&'a Value>,
    pub maturity_score: Option<&'a Value>,
    pub source_diversity: Option<&'a Value>,
    pub temporal_span_days: Option<&'a Value>,
    pub relationship_richness: Option<&'a Value>,
    pub access_regularity: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_maturations_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_maturations_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let maturations = G::new(&db, &txn, &arena)
.n_from_type("ConsolMaturation")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "maturations": maturations.iter().map(|maturation| Find_consol_maturations_by_cycleMaturationsReturnType {
        id: uuid_str(maturation.id(), &arena),
        label: maturation.label(),
        cycle_id: maturation.get_property("cycle_id"),
        group_id: maturation.get_property("group_id"),
        mat_id: maturation.get_property("mat_id"),
        entity_id: maturation.get_property("entity_id"),
        entity_name: maturation.get_property("entity_name"),
        old_tier: maturation.get_property("old_tier"),
        new_tier: maturation.get_property("new_tier"),
        maturity_score: maturation.get_property("maturity_score"),
        source_diversity: maturation.get_property("source_diversity"),
        temporal_span_days: maturation.get_property("temporal_span_days"),
        relationship_richness: maturation.get_property("relationship_richness"),
        access_regularity: maturation.get_property("access_regularity"),
        timestamp: maturation.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_inferred_edgeInput {

pub edge_id: String,
pub cycle_id: String,
pub group_id: String,
pub source_id: String,
pub target_id: String,
pub source_name: String,
pub target_name: String,
pub co_occurrence_count: i32,
pub confidence: f64,
pub infer_type: String,
pub pmi_score: f64,
pub llm_verdict: String,
pub relationship_id: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_inferred_edgeNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub edge_id: Option<&'a Value>,
    pub source_id: Option<&'a Value>,
    pub target_id: Option<&'a Value>,
    pub source_name: Option<&'a Value>,
    pub target_name: Option<&'a Value>,
    pub co_occurrence_count: Option<&'a Value>,
    pub confidence: Option<&'a Value>,
    pub infer_type: Option<&'a Value>,
    pub pmi_score: Option<&'a Value>,
    pub llm_verdict: Option<&'a Value>,
    pub relationship_id: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_inferred_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_inferred_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolInferredEdge", Some(ImmutablePropertiesMap::new(14, vec![("edge_id", Value::from(&data.edge_id)), ("co_occurrence_count", Value::from(&data.co_occurrence_count)), ("confidence", Value::from(&data.confidence)), ("pmi_score", Value::from(&data.pmi_score)), ("source_id", Value::from(&data.source_id)), ("target_name", Value::from(&data.target_name)), ("relationship_id", Value::from(&data.relationship_id)), ("infer_type", Value::from(&data.infer_type)), ("llm_verdict", Value::from(&data.llm_verdict)), ("timestamp", Value::from(&data.timestamp)), ("target_id", Value::from(&data.target_id)), ("group_id", Value::from(&data.group_id)), ("cycle_id", Value::from(&data.cycle_id)), ("source_name", Value::from(&data.source_name))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_inferred_edgeNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        edge_id: node.get_property("edge_id"),
        source_id: node.get_property("source_id"),
        target_id: node.get_property("target_id"),
        source_name: node.get_property("source_name"),
        target_name: node.get_property("target_name"),
        co_occurrence_count: node.get_property("co_occurrence_count"),
        confidence: node.get_property("confidence"),
        infer_type: node.get_property("infer_type"),
        pmi_score: node.get_property("pmi_score"),
        llm_verdict: node.get_property("llm_verdict"),
        relationship_id: node.get_property("relationship_id"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_schemaInput {

pub schema_id: String,
pub cycle_id: String,
pub group_id: String,
pub schema_entity_id: String,
pub schema_name: String,
pub instance_count: i32,
pub predicate_count: i32,
pub action: String,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_schemaNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub schema_id: Option<&'a Value>,
    pub schema_entity_id: Option<&'a Value>,
    pub schema_name: Option<&'a Value>,
    pub instance_count: Option<&'a Value>,
    pub predicate_count: Option<&'a Value>,
    pub action: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_schema (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_schemaInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolSchema", Some(ImmutablePropertiesMap::new(9, vec![("instance_count", Value::from(&data.instance_count)), ("schema_entity_id", Value::from(&data.schema_entity_id)), ("schema_id", Value::from(&data.schema_id)), ("group_id", Value::from(&data.group_id)), ("action", Value::from(&data.action)), ("cycle_id", Value::from(&data.cycle_id)), ("timestamp", Value::from(&data.timestamp)), ("schema_name", Value::from(&data.schema_name)), ("predicate_count", Value::from(&data.predicate_count))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_schemaNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        schema_id: node.get_property("schema_id"),
        schema_entity_id: node.get_property("schema_entity_id"),
        schema_name: node.get_property("schema_name"),
        instance_count: node.get_property("instance_count"),
        predicate_count: node.get_property("predicate_count"),
        action: node.get_property("action"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_complement_tagInput {

pub id: ID,
pub cycle_confirmed: i32,
pub cleared: bool,
pub updated_at: String
}
#[derive(Serialize, Default)]
pub struct Update_complement_tagTagReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub target_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub target_type: Option<&'a Value>,
    pub tag_type: Option<&'a Value>,
    pub cycle_tagged: Option<&'a Value>,
    pub cycle_confirmed: Option<&'a Value>,
    pub cleared: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_complement_tag (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_complement_tagInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let tag = {let update_tr = G::new(&db, &txn, &arena)
.n_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("cycle_confirmed", Value::from(&data.cycle_confirmed)), ("cleared", Value::from(&data.cleared)), ("updated_at", Value::from(&data.updated_at))])
    .collect_to_obj()?};
let response = json!({
    "tag": Update_complement_tagTagReturnType {
        id: uuid_str(tag.id(), &arena),
        label: tag.label(),
        target_id: tag.get_property("target_id"),
        group_id: tag.get_property("group_id"),
        target_type: tag.get_property("target_type"),
        tag_type: tag.get_property("tag_type"),
        cycle_tagged: tag.get_property("cycle_tagged"),
        cycle_confirmed: tag.get_property("cycle_confirmed"),
        cleared: tag.get_property("cleared"),
        created_at: tag.get_property("created_at"),
        updated_at: tag.get_property("updated_at"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct get_entityInput {

pub id: ID
}
#[derive(Serialize, Default)]
pub struct Get_entityEntityReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub name: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub canonical_identifier: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub summary: Option<&'a Value>,
    pub attributes_json: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub updated_at: Option<&'a Value>,
    pub is_deleted: Option<&'a Value>,
    pub deleted_at: Option<&'a Value>,
    pub identity_core: Option<&'a Value>,
    pub mat_tier: Option<&'a Value>,
    pub recon_count: Option<&'a Value>,
    pub lexical_regime: Option<&'a Value>,
    pub identifier_label: Option<&'a Value>,
    pub pii_detected: Option<&'a Value>,
    pub pii_categories_json: Option<&'a Value>,
    pub access_count: Option<&'a Value>,
    pub last_accessed: Option<&'a Value>,
    pub source_episode_ids: Option<&'a Value>,
    pub evidence_count: Option<&'a Value>,
    pub evidence_span_start: Option<&'a Value>,
    pub evidence_span_end: Option<&'a Value>,
}

#[handler]
pub fn get_entity (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<get_entityInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let entity = G::new(&db, &txn, &arena)
.n_from_id(&data.id).collect_to_obj()?;
let response = json!({
    "entity": Get_entityEntityReturnType {
        id: uuid_str(entity.id(), &arena),
        label: entity.label(),
        name: entity.get_property("name"),
        group_id: entity.get_property("group_id"),
        entity_type: entity.get_property("entity_type"),
        canonical_identifier: entity.get_property("canonical_identifier"),
        entity_id: entity.get_property("entity_id"),
        summary: entity.get_property("summary"),
        attributes_json: entity.get_property("attributes_json"),
        created_at: entity.get_property("created_at"),
        updated_at: entity.get_property("updated_at"),
        is_deleted: entity.get_property("is_deleted"),
        deleted_at: entity.get_property("deleted_at"),
        identity_core: entity.get_property("identity_core"),
        mat_tier: entity.get_property("mat_tier"),
        recon_count: entity.get_property("recon_count"),
        lexical_regime: entity.get_property("lexical_regime"),
        identifier_label: entity.get_property("identifier_label"),
        pii_detected: entity.get_property("pii_detected"),
        pii_categories_json: entity.get_property("pii_categories_json"),
        access_count: entity.get_property("access_count"),
        last_accessed: entity.get_property("last_accessed"),
        source_episode_ids: entity.get_property("source_episode_ids"),
        evidence_count: entity.get_property("evidence_count"),
        evidence_span_start: entity.get_property("evidence_span_start"),
        evidence_span_end: entity.get_property("evidence_span_end"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct update_edgeInput {

pub id: ID,
pub weight: f64,
pub is_expired: bool,
pub valid_to: String
}
#[derive(Serialize, Default)]
pub struct Update_edgeEdgeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub from_node: &'a str,
    pub to_node: &'a str,
    pub rel_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub predicate: Option<&'a Value>,
    pub weight: Option<&'a Value>,
    pub polarity: Option<&'a Value>,
    pub valid_from: Option<&'a Value>,
    pub valid_to: Option<&'a Value>,
    pub is_expired: Option<&'a Value>,
    pub created_at: Option<&'a Value>,
    pub source_episode_id: Option<&'a Value>,
}

#[handler(is_write)]
pub fn update_edge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<update_edgeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let edge = {let update_tr = G::new(&db, &txn, &arena)
.e_from_id(&data.id)
    .collect::<Result<Vec<_>, _>>()?;G::new_mut_from_iter(&db, &mut txn, update_tr.iter().cloned(), &arena)
    .update(&[("weight", Value::from(&data.weight)), ("is_expired", Value::from(&data.is_expired)), ("valid_to", Value::from(&data.valid_to))])
    .collect_to_obj()?};
let response = json!({
    "edge": Update_edgeEdgeReturnType {
        id: uuid_str(edge.id(), &arena),
        label: edge.label(),
        from_node: uuid_str(edge.from_node(), &arena),
        to_node: uuid_str(edge.to_node(), &arena),
        rel_id: edge.get_property("rel_id"),
        group_id: edge.get_property("group_id"),
        predicate: edge.get_property("predicate"),
        weight: edge.get_property("weight"),
        polarity: edge.get_property("polarity"),
        valid_from: edge.get_property("valid_from"),
        valid_to: edge.get_property("valid_to"),
        is_expired: edge.get_property("is_expired"),
        created_at: edge.get_property("created_at"),
        source_episode_id: edge.get_property("source_episode_id"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct find_consol_prunes_by_cycleInput {

pub cycle_id: String,
pub gid: String
}
#[derive(Serialize, Default)]
pub struct Find_consol_prunes_by_cyclePrunesReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub prune_id: Option<&'a Value>,
    pub entity_id: Option<&'a Value>,
    pub entity_name: Option<&'a Value>,
    pub entity_type: Option<&'a Value>,
    pub reason: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler]
pub fn find_consol_prunes_by_cycle (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<find_consol_prunes_by_cycleInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let prunes = G::new(&db, &txn, &arena)
.n_from_type("ConsolPrune")

.filter_ref(|val, txn|{
                if let Ok(val) = val {
                    Ok((val
                    .get_property("cycle_id")
                    .map_or(false, |v| *v == data.cycle_id.clone()) && val
                    .get_property("group_id")
                    .map_or(false, |v| *v == data.gid.clone())))
                } else {
                    Ok(false)
                }
            }).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "prunes": prunes.iter().map(|prune| Find_consol_prunes_by_cyclePrunesReturnType {
        id: uuid_str(prune.id(), &arena),
        label: prune.label(),
        cycle_id: prune.get_property("cycle_id"),
        group_id: prune.get_property("group_id"),
        prune_id: prune.get_property("prune_id"),
        entity_id: prune.get_property("entity_id"),
        entity_name: prune.get_property("entity_name"),
        entity_type: prune.get_property("entity_type"),
        reason: prune.get_property("reason"),
        timestamp: prune.get_property("timestamp"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct create_consol_mergeInput {

pub merge_id: String,
pub cycle_id: String,
pub group_id: String,
pub keep_id: String,
pub remove_id: String,
pub keep_name: String,
pub remove_name: String,
pub similarity: f64,
pub decision_confidence: f64,
pub decision_source: String,
pub decision_reason: String,
pub relationships_transferred: i32,
pub timestamp: f64
}
#[derive(Serialize, Default)]
pub struct Create_consol_mergeNodeReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub cycle_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub merge_id: Option<&'a Value>,
    pub keep_id: Option<&'a Value>,
    pub remove_id: Option<&'a Value>,
    pub keep_name: Option<&'a Value>,
    pub remove_name: Option<&'a Value>,
    pub similarity: Option<&'a Value>,
    pub decision_confidence: Option<&'a Value>,
    pub decision_source: Option<&'a Value>,
    pub decision_reason: Option<&'a Value>,
    pub relationships_transferred: Option<&'a Value>,
    pub timestamp: Option<&'a Value>,
}

#[handler(is_write)]
pub fn create_consol_merge (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<create_consol_mergeInput>(&input.request.body)?;
let arena = Bump::new();
let mut txn = db.graph_env.write_txn().map_err(|e| GraphError::New(format!("Failed to start write transaction: {:?}", e)))?;
    let node = G::new_mut(&db, &arena, &mut txn)
.add_n("ConsolMerge", Some(ImmutablePropertiesMap::new(13, vec![("decision_confidence", Value::from(&data.decision_confidence)), ("merge_id", Value::from(&data.merge_id)), ("timestamp", Value::from(&data.timestamp)), ("similarity", Value::from(&data.similarity)), ("keep_id", Value::from(&data.keep_id)), ("decision_reason", Value::from(&data.decision_reason)), ("relationships_transferred", Value::from(&data.relationships_transferred)), ("group_id", Value::from(&data.group_id)), ("cycle_id", Value::from(&data.cycle_id)), ("decision_source", Value::from(&data.decision_source)), ("keep_name", Value::from(&data.keep_name)), ("remove_id", Value::from(&data.remove_id)), ("remove_name", Value::from(&data.remove_name))].into_iter(), &arena)), Some(&["cycle_id", "group_id"])).collect_to_obj()?;
let response = json!({
    "node": Create_consol_mergeNodeReturnType {
        id: uuid_str(node.id(), &arena),
        label: node.label(),
        cycle_id: node.get_property("cycle_id"),
        group_id: node.get_property("group_id"),
        merge_id: node.get_property("merge_id"),
        keep_id: node.get_property("keep_id"),
        remove_id: node.get_property("remove_id"),
        keep_name: node.get_property("keep_name"),
        remove_name: node.get_property("remove_name"),
        similarity: node.get_property("similarity"),
        decision_confidence: node.get_property("decision_confidence"),
        decision_source: node.get_property("decision_source"),
        decision_reason: node.get_property("decision_reason"),
        relationships_transferred: node.get_property("relationships_transferred"),
        timestamp: node.get_property("timestamp"),
    }
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}

#[derive(Serialize, Deserialize, Clone)]
pub struct search_graph_embed_vectorsInput {

pub vec: Vec<f64>,
pub k: i32
}
#[derive(Serialize, Default)]
pub struct Search_graph_embed_vectorsResultsReturnType<'a> {
    pub id: &'a str,
    pub label: &'a str,
    pub data: &'a [f64],
    pub score: f64,
    pub entity_id: Option<&'a Value>,
    pub group_id: Option<&'a Value>,
    pub method: Option<&'a Value>,
    pub model_version: Option<&'a Value>,
}

#[handler]
pub fn search_graph_embed_vectors (input: HandlerInput) -> Result<Response, GraphError> {
let db = Arc::clone(&input.graph.storage);
let data = input.request.in_fmt.deserialize::<search_graph_embed_vectorsInput>(&input.request.body)?;
let arena = Bump::new();
let txn = db.graph_env.read_txn().map_err(|e| GraphError::New(format!("Failed to start read transaction: {:?}", e)))?;
    let results = G::new(&db, &txn, &arena)
.search_v::<fn(&HVector, &RoTxn) -> bool, _>(&data.vec, data.k.clone(), "GraphEmbedVec", None).collect::<Result<Vec<_>, _>>()?;
let response = json!({
    "results": results.iter().map(|result| Search_graph_embed_vectorsResultsReturnType {
        id: uuid_str(result.id(), &arena),
        label: result.label(),
        data: result.data(),
        score: result.score(),
        entity_id: result.get_property("entity_id"),
        group_id: result.get_property("group_id"),
        method: result.get_property("method"),
        model_version: result.get_property("model_version"),
    }).collect::<Vec<_>>()
});
txn.commit().map_err(|e| GraphError::New(format!("Failed to commit transaction: {:?}", e)))?;
Ok(input.request.out_fmt.create_response(&response))
}
