// ============================================================
// Engram HelixDB Schema
// Deploy with: helix deploy --path <config_path>
// ============================================================

// === Node Types ===

N::Entity {
    INDEX name: String,
    INDEX group_id: String,
    INDEX entity_type: String,
    INDEX canonical_identifier: String,
    entity_id: String,
    summary: String,
    attributes_json: String,
    created_at: String,
    updated_at: String,
    is_deleted: Boolean,
    deleted_at: String,
    identity_core: Boolean,
    mat_tier: String,
    recon_count: I32,
    lexical_regime: String,
    identifier_label: String,
    pii_detected: Boolean,
    pii_categories_json: String,
    access_count: I64,
    last_accessed: String,
    source_episode_ids: String,
    evidence_count: I64,
    evidence_span_start: String,
    evidence_span_end: String
}

N::Episode {
    INDEX group_id: String,
    INDEX status: String,
    INDEX session_id: String,
    episode_id: String,
    content: String,
    source: String,
    created_at: String,
    updated_at: String,
    error: String,
    retry_count: I32,
    processing_duration_ms: I64,
    skipped_meta: Boolean,
    skipped_triage: Boolean,
    encoding_context_json: String,
    memory_tier: String,
    consolidation_cycles: I32,
    entity_coverage: F64,
    projection_state: String,
    last_projection_reason: String,
    last_projected_at: String,
    conversation_date: String,
    attachments_json: String
}

N::EpisodeCue {
    INDEX episode_id: String,
    INDEX group_id: String,
    cue_text: String,
    supporting_spans_json: String,
    projection_state: String,
    created_at: String,
    updated_at: String
}

N::Intention {
    INDEX group_id: String,
    intention_id: String,
    trigger_text: String,
    action_text: String,
    entity_names_json: String,
    enabled: Boolean,
    fire_count: I32,
    max_fires: I32,
    created_at: String,
    updated_at: String,
    deleted_at: String,
    is_deleted: Boolean,
    context_json: String
}

N::Evidence {
    INDEX episode_id: String,
    INDEX group_id: String,
    INDEX status: String,
    evidence_id: String,
    fact_class: String,
    confidence: F64,
    source_type: String,
    extractor_name: String,
    payload_json: String,
    source_span: String,
    signals_json: String,
    ambiguity_tags_json: String,
    ambiguity_score: F64,
    adjudication_request_id: String,
    commit_reason: String,
    committed_id: String,
    deferred_cycles: I32,
    created_at: String,
    resolved_at: String
}

N::AdjudicationRequest {
    INDEX episode_id: String,
    INDEX group_id: String,
    INDEX status: String,
    request_id: String,
    ambiguity_tags_json: String,
    evidence_ids_json: String,
    selected_text: String,
    request_reason: String,
    resolution_source: String,
    resolution_payload_json: String,
    attempt_count: I32,
    created_at: String,
    resolved_at: String
}

N::SchemaMember {
    INDEX schema_entity_id: String,
    INDEX group_id: String,
    role_label: String,
    member_entity_id: String
}

// === Edge Types ===

E::RelatesTo {
    From: Entity,
    To: Entity,
    Properties: {
        rel_id: String,
        group_id: String,
        predicate: String,
        weight: F64,
        polarity: String,
        valid_from: String,
        valid_to: String,
        is_expired: Boolean,
        created_at: String,
        source_episode_id: String
    }
}

E::HasEntity {
    From: Episode,
    To: Entity,
    Properties: {
    }
}

E::HasSchemaMember {
    From: Entity,
    To: SchemaMember,
    Properties: {
    }
}

// === Vector Types ===

V::EntityVec {
    entity_id: String,
    group_id: String,
    content_type: String,
    embed_provider: String,
    embed_model: String
}

V::EpisodeVec {
    episode_id: String,
    group_id: String,
    content_type: String
}

V::CueVec {
    episode_id: String,
    group_id: String,
    content_type: String
}

V::GraphEmbedVec {
    entity_id: String,
    group_id: String,
    method: String,
    model_version: String
}

// ============================================================
// QUERIES — Entity CRUD
// ============================================================

QUERY create_entity(entity_id: String, name: String, group_id: String, entity_type: String, summary: String, attributes_json: String, created_at: String, updated_at: String, is_deleted: Boolean, deleted_at: String, identity_core: Boolean, mat_tier: String, recon_count: I32, lexical_regime: String, canonical_identifier: String, identifier_label: String, pii_detected: Boolean, pii_categories_json: String, access_count: I64, last_accessed: String, source_episode_ids: String, evidence_count: I64, evidence_span_start: String, evidence_span_end: String) =>
    node <- AddN<Entity>({entity_id: entity_id, name: name, group_id: group_id, entity_type: entity_type, summary: summary, attributes_json: attributes_json, created_at: created_at, updated_at: updated_at, is_deleted: is_deleted, deleted_at: deleted_at, identity_core: identity_core, mat_tier: mat_tier, recon_count: recon_count, lexical_regime: lexical_regime, canonical_identifier: canonical_identifier, identifier_label: identifier_label, pii_detected: pii_detected, pii_categories_json: pii_categories_json, access_count: access_count, last_accessed: last_accessed, source_episode_ids: source_episode_ids, evidence_count: evidence_count, evidence_span_start: evidence_span_start, evidence_span_end: evidence_span_end})
    RETURN node

QUERY get_entity(id: ID) =>
    entity <- N<Entity>(id)
    RETURN entity

QUERY find_entities_by_group(gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_entities_by_name(name_query: String, gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{name}::CONTAINS(name_query), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_entities_by_type(etype: String, gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{entity_type}::EQ(etype), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_entities_by_name_and_type(name_query: String, etype: String, gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{name}::CONTAINS(name_query), _::{entity_type}::EQ(etype), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_entities_exact_name(name_exact: String, gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{name}::EQ(name_exact), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_entities_by_canonical(canon: String, gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{canonical_identifier}::EQ(canon), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY find_identity_core_entities(gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{identity_core}::EQ(true), _::{is_deleted}::EQ(false)))
    RETURN entities

QUERY update_entity_full(id: ID, name: String, entity_type: String, summary: String, attributes_json: String, updated_at: String, is_deleted: Boolean, deleted_at: String, identity_core: Boolean, mat_tier: String, recon_count: I32, lexical_regime: String, canonical_identifier: String, identifier_label: String, pii_detected: Boolean, pii_categories_json: String, access_count: I64, last_accessed: String, source_episode_ids: String, evidence_count: I64, evidence_span_start: String, evidence_span_end: String) =>
    entity <- N<Entity>(id)
        ::UPDATE({name: name, entity_type: entity_type, summary: summary, attributes_json: attributes_json, updated_at: updated_at, is_deleted: is_deleted, deleted_at: deleted_at, identity_core: identity_core, mat_tier: mat_tier, recon_count: recon_count, lexical_regime: lexical_regime, canonical_identifier: canonical_identifier, identifier_label: identifier_label, pii_detected: pii_detected, pii_categories_json: pii_categories_json, access_count: access_count, last_accessed: last_accessed, source_episode_ids: source_episode_ids, evidence_count: evidence_count, evidence_span_start: evidence_span_start, evidence_span_end: evidence_span_end})
    RETURN entity

QUERY soft_delete_entity(id: ID, deleted_at: String) =>
    entity <- N<Entity>(id)
        ::UPDATE({is_deleted: true, deleted_at: deleted_at})
    RETURN entity

QUERY hard_delete_entity(id: ID) =>
    DROP N<Entity>(id)
    RETURN NONE

// ============================================================
// QUERIES — Relationship CRUD
// ============================================================

QUERY create_relationship(rel_id: String, group_id: String, predicate: String, weight: F64, polarity: String, valid_from: String, valid_to: String, is_expired: Boolean, created_at: String, source_episode_id: String, source_id: ID, target_id: ID) =>
    edge <- AddE<RelatesTo>({rel_id: rel_id, group_id: group_id, predicate: predicate, weight: weight, polarity: polarity, valid_from: valid_from, valid_to: valid_to, is_expired: is_expired, created_at: created_at, source_episode_id: source_episode_id})
        ::From(source_id)
        ::To(target_id)
    RETURN edge

QUERY get_outgoing_edges(id: ID) =>
    edges <- N<Entity>(id)
        ::OutE<RelatesTo>
    RETURN edges

QUERY get_incoming_edges(id: ID) =>
    edges <- N<Entity>(id)
        ::InE<RelatesTo>
    RETURN edges

QUERY get_outgoing_neighbors(id: ID) =>
    neighbors <- N<Entity>(id)
        ::Out<RelatesTo>
    RETURN neighbors

QUERY get_incoming_neighbors(id: ID) =>
    neighbors <- N<Entity>(id)
        ::In<RelatesTo>
    RETURN neighbors

QUERY get_outgoing_edges_by_predicate(id: ID, pred: String) =>
    edges <- N<Entity>(id)
        ::OutE<RelatesTo>
        ::WHERE(_::{predicate}::EQ(pred))
    RETURN edges

QUERY get_incoming_edges_by_predicate(id: ID, pred: String) =>
    edges <- N<Entity>(id)
        ::InE<RelatesTo>
        ::WHERE(_::{predicate}::EQ(pred))
    RETURN edges

QUERY update_edge(id: ID, weight: F64, is_expired: Boolean, valid_to: String) =>
    edge <- E<RelatesTo>(id)
        ::UPDATE({weight: weight, is_expired: is_expired, valid_to: valid_to})
    RETURN edge

QUERY invalidate_edge(id: ID, valid_to: String) =>
    edge <- E<RelatesTo>(id)
        ::UPDATE({is_expired: true, valid_to: valid_to})
    RETURN edge

QUERY get_edge(id: ID) =>
    edge <- E<RelatesTo>(id)
    RETURN edge

QUERY drop_edge(id: ID) =>
    DROP E<RelatesTo>(id)
    RETURN NONE

// ============================================================
// QUERIES — Episode CRUD
// ============================================================

QUERY create_episode(episode_id: String, group_id: String, content: String, source: String, session_id: String, status: String, created_at: String, updated_at: String, error: String, retry_count: I32, processing_duration_ms: I64, skipped_meta: Boolean, skipped_triage: Boolean, encoding_context_json: String, memory_tier: String, consolidation_cycles: I32, entity_coverage: F64, projection_state: String, last_projection_reason: String, last_projected_at: String, conversation_date: String, attachments_json: String) =>
    node <- AddN<Episode>({episode_id: episode_id, group_id: group_id, content: content, source: source, session_id: session_id, status: status, created_at: created_at, updated_at: updated_at, error: error, retry_count: retry_count, processing_duration_ms: processing_duration_ms, skipped_meta: skipped_meta, skipped_triage: skipped_triage, encoding_context_json: encoding_context_json, memory_tier: memory_tier, consolidation_cycles: consolidation_cycles, entity_coverage: entity_coverage, projection_state: projection_state, last_projection_reason: last_projection_reason, last_projected_at: last_projected_at, conversation_date: conversation_date, attachments_json: attachments_json})
    RETURN node

QUERY get_episode(id: ID) =>
    episode <- N<Episode>(id)
    RETURN episode

QUERY find_episodes_by_group(gid: String) =>
    episodes <- N<Episode>
        ::WHERE(_::{group_id}::EQ(gid))
    RETURN episodes

QUERY find_episodes_by_status(gid: String, st: String) =>
    episodes <- N<Episode>::WHERE(AND(_::{group_id}::EQ(gid), _::{status}::EQ(st)))
    RETURN episodes

QUERY find_episodes_by_source(gid: String, src: String) =>
    episodes <- N<Episode>::WHERE(AND(_::{group_id}::EQ(gid), _::{source}::EQ(src)))
    RETURN episodes

QUERY find_episodes_by_session(sid: String) =>
    episodes <- N<Episode>
        ::WHERE(_::{session_id}::EQ(sid))
    RETURN episodes

QUERY update_episode_full(id: ID, status: String, updated_at: String, error: String, retry_count: I32, processing_duration_ms: I64, content: String, skipped_meta: Boolean, skipped_triage: Boolean, encoding_context_json: String, memory_tier: String, consolidation_cycles: I32, entity_coverage: F64, projection_state: String, last_projection_reason: String, last_projected_at: String, conversation_date: String, attachments_json: String) =>
    episode <- N<Episode>(id)
        ::UPDATE({status: status, updated_at: updated_at, error: error, retry_count: retry_count, processing_duration_ms: processing_duration_ms, content: content, skipped_meta: skipped_meta, skipped_triage: skipped_triage, encoding_context_json: encoding_context_json, memory_tier: memory_tier, consolidation_cycles: consolidation_cycles, entity_coverage: entity_coverage, projection_state: projection_state, last_projection_reason: last_projection_reason, last_projected_at: last_projected_at, conversation_date: conversation_date, attachments_json: attachments_json})
    RETURN episode

QUERY get_episode_entities(id: ID) =>
    entities <- N<Episode>(id)
        ::Out<HasEntity>
    RETURN entities

QUERY get_episodes_for_entity(id: ID) =>
    episodes <- N<Entity>(id)
        ::In<HasEntity>
    RETURN episodes

QUERY link_episode_entity(episode_id: ID, entity_id: ID) =>
    edge <- AddE<HasEntity>
        ::From(episode_id)
        ::To(entity_id)
    RETURN edge

QUERY hard_delete_episode(id: ID) =>
    DROP N<Episode>(id)
    RETURN NONE

// ============================================================
// QUERIES — Episode Cues
// ============================================================

QUERY create_episode_cue(episode_id: String, group_id: String, cue_text: String, supporting_spans_json: String, projection_state: String, created_at: String, updated_at: String) =>
    node <- AddN<EpisodeCue>({episode_id: episode_id, group_id: group_id, cue_text: cue_text, supporting_spans_json: supporting_spans_json, projection_state: projection_state, created_at: created_at, updated_at: updated_at})
    RETURN node

QUERY find_cue_by_episode(ep_id: String, gid: String) =>
    cues <- N<EpisodeCue>::WHERE(AND(_::{episode_id}::EQ(ep_id), _::{group_id}::EQ(gid)))
    RETURN cues

QUERY update_cue(id: ID, cue_text: String, supporting_spans_json: String, projection_state: String, updated_at: String) =>
    cue <- N<EpisodeCue>(id)
        ::UPDATE({cue_text: cue_text, supporting_spans_json: supporting_spans_json, projection_state: projection_state, updated_at: updated_at})
    RETURN cue

QUERY hard_delete_cue(id: ID) =>
    DROP N<EpisodeCue>(id)
    RETURN NONE

// ============================================================
// QUERIES — Intentions
// ============================================================

QUERY create_intention(intention_id: String, group_id: String, trigger_text: String, action_text: String, entity_names_json: String, enabled: Boolean, fire_count: I32, max_fires: I32, created_at: String, updated_at: String, deleted_at: String, is_deleted: Boolean, context_json: String) =>
    node <- AddN<Intention>({intention_id: intention_id, group_id: group_id, trigger_text: trigger_text, action_text: action_text, entity_names_json: entity_names_json, enabled: enabled, fire_count: fire_count, max_fires: max_fires, created_at: created_at, updated_at: updated_at, deleted_at: deleted_at, is_deleted: is_deleted, context_json: context_json})
    RETURN node

QUERY get_intention(id: ID) =>
    intention <- N<Intention>(id)
    RETURN intention

QUERY find_intentions_by_group(gid: String) =>
    intentions <- N<Intention>::WHERE(AND(_::{group_id}::EQ(gid), _::{is_deleted}::EQ(false)))
    RETURN intentions

QUERY find_enabled_intentions(gid: String) =>
    intentions <- N<Intention>::WHERE(AND(_::{group_id}::EQ(gid), _::{enabled}::EQ(true), _::{is_deleted}::EQ(false)))
    RETURN intentions

QUERY update_intention_full(id: ID, trigger_text: String, action_text: String, entity_names_json: String, enabled: Boolean, fire_count: I32, max_fires: I32, updated_at: String, deleted_at: String, is_deleted: Boolean, context_json: String) =>
    intention <- N<Intention>(id)
        ::UPDATE({trigger_text: trigger_text, action_text: action_text, entity_names_json: entity_names_json, enabled: enabled, fire_count: fire_count, max_fires: max_fires, updated_at: updated_at, deleted_at: deleted_at, is_deleted: is_deleted, context_json: context_json})
    RETURN intention

QUERY soft_delete_intention(id: ID, deleted_at: String) =>
    intention <- N<Intention>(id)
        ::UPDATE({is_deleted: true, deleted_at: deleted_at})
    RETURN intention

QUERY hard_delete_intention(id: ID) =>
    DROP N<Intention>(id)
    RETURN NONE

// ============================================================
// QUERIES — Evidence
// ============================================================

QUERY create_evidence(evidence_id: String, episode_id: String, group_id: String, status: String, fact_class: String, confidence: F64, source_type: String, extractor_name: String, payload_json: String, source_span: String, signals_json: String, ambiguity_tags_json: String, ambiguity_score: F64, adjudication_request_id: String, commit_reason: String, committed_id: String, deferred_cycles: I32, created_at: String, resolved_at: String) =>
    node <- AddN<Evidence>({evidence_id: evidence_id, episode_id: episode_id, group_id: group_id, status: status, fact_class: fact_class, confidence: confidence, source_type: source_type, extractor_name: extractor_name, payload_json: payload_json, source_span: source_span, signals_json: signals_json, ambiguity_tags_json: ambiguity_tags_json, ambiguity_score: ambiguity_score, adjudication_request_id: adjudication_request_id, commit_reason: commit_reason, committed_id: committed_id, deferred_cycles: deferred_cycles, created_at: created_at, resolved_at: resolved_at})
    RETURN node

QUERY find_pending_evidence(gid: String) =>
    evidence <- N<Evidence>::WHERE(AND(_::{group_id}::EQ(gid), _::{status}::EQ("pending")))
    RETURN evidence

QUERY find_evidence_by_episode(ep_id: String, gid: String) =>
    evidence <- N<Evidence>::WHERE(AND(_::{episode_id}::EQ(ep_id), _::{group_id}::EQ(gid)))
    RETURN evidence

QUERY get_evidence(id: ID) =>
    evidence <- N<Evidence>(id)
    RETURN evidence

QUERY update_evidence(id: ID, status: String, resolved_at: String, commit_reason: String, committed_id: String) =>
    evidence <- N<Evidence>(id)
        ::UPDATE({status: status, resolved_at: resolved_at, commit_reason: commit_reason, committed_id: committed_id})
    RETURN evidence

QUERY hard_delete_evidence(id: ID) =>
    DROP N<Evidence>(id)
    RETURN NONE

// ============================================================
// QUERIES — Adjudication Requests
// ============================================================

QUERY create_adjudication(request_id: String, episode_id: String, group_id: String, status: String, ambiguity_tags_json: String, evidence_ids_json: String, selected_text: String, request_reason: String, resolution_source: String, resolution_payload_json: String, attempt_count: I32, created_at: String, resolved_at: String) =>
    node <- AddN<AdjudicationRequest>({request_id: request_id, episode_id: episode_id, group_id: group_id, status: status, ambiguity_tags_json: ambiguity_tags_json, evidence_ids_json: evidence_ids_json, selected_text: selected_text, request_reason: request_reason, resolution_source: resolution_source, resolution_payload_json: resolution_payload_json, attempt_count: attempt_count, created_at: created_at, resolved_at: resolved_at})
    RETURN node

QUERY find_pending_adjudications(gid: String) =>
    requests <- N<AdjudicationRequest>::WHERE(AND(_::{group_id}::EQ(gid), _::{status}::EQ("pending")))
    RETURN requests

QUERY find_adjudications_by_episode(ep_id: String, gid: String) =>
    requests <- N<AdjudicationRequest>::WHERE(AND(_::{episode_id}::EQ(ep_id), _::{group_id}::EQ(gid)))
    RETURN requests

QUERY get_adjudication(id: ID) =>
    request <- N<AdjudicationRequest>(id)
    RETURN request

QUERY update_adjudication(id: ID, status: String, resolution_source: String, resolution_payload_json: String, attempt_count: I32, resolved_at: String) =>
    request <- N<AdjudicationRequest>(id)
        ::UPDATE({status: status, resolution_source: resolution_source, resolution_payload_json: resolution_payload_json, attempt_count: attempt_count, resolved_at: resolved_at})
    RETURN request

QUERY hard_delete_adjudication(id: ID) =>
    DROP N<AdjudicationRequest>(id)
    RETURN NONE

// ============================================================
// QUERIES — Schema Members
// ============================================================

QUERY create_schema_member(schema_entity_id: String, group_id: String, role_label: String, member_entity_id: String) =>
    node <- AddN<SchemaMember>({schema_entity_id: schema_entity_id, group_id: group_id, role_label: role_label, member_entity_id: member_entity_id})
    RETURN node

QUERY find_schema_members(schema_id: String, gid: String) =>
    members <- N<SchemaMember>::WHERE(AND(_::{schema_entity_id}::EQ(schema_id), _::{group_id}::EQ(gid)))
    RETURN members

QUERY link_schema_member(entity_id: ID, member_id: ID) =>
    edge <- AddE<HasSchemaMember>
        ::From(entity_id)
        ::To(member_id)
    RETURN edge

QUERY hard_delete_schema_member(id: ID) =>
    DROP N<SchemaMember>(id)
    RETURN NONE

// ============================================================
// QUERIES — Vector Search
// ============================================================

QUERY search_entity_vectors(vec: [F64], k: I32) =>
    results <- SearchV<EntityVec>(vec, k)
    RETURN results

QUERY search_episode_vectors(vec: [F64], k: I32) =>
    results <- SearchV<EpisodeVec>(vec, k)
    RETURN results

QUERY search_cue_vectors(vec: [F64], k: I32) =>
    results <- SearchV<CueVec>(vec, k)
    RETURN results

QUERY search_graph_embed_vectors(vec: [F64], k: I32) =>
    results <- SearchV<GraphEmbedVec>(vec, k)
    RETURN results

QUERY add_entity_vector(entity_id: String, group_id: String, content_type: String, embed_provider: String, embed_model: String, vec: [F64]) =>
    v <- AddV<EntityVec>(vec, {entity_id: entity_id, group_id: group_id, content_type: content_type, embed_provider: embed_provider, embed_model: embed_model})
    RETURN v

QUERY add_episode_vector(episode_id: String, group_id: String, content_type: String, vec: [F64]) =>
    v <- AddV<EpisodeVec>(vec, {episode_id: episode_id, group_id: group_id, content_type: content_type})
    RETURN v

QUERY add_cue_vector(episode_id: String, group_id: String, content_type: String, vec: [F64]) =>
    v <- AddV<CueVec>(vec, {episode_id: episode_id, group_id: group_id, content_type: content_type})
    RETURN v

QUERY add_graph_embed_vector(entity_id: String, group_id: String, method: String, model_version: String, vec: [F64]) =>
    v <- AddV<GraphEmbedVec>(vec, {entity_id: entity_id, group_id: group_id, method: method, model_version: model_version})
    RETURN v

// ============================================================
// QUERIES — BM25 Text Search
// ============================================================

QUERY search_entities_bm25(query: String, k: I32) =>
    results <- SearchBM25<Entity>(query, k)
    RETURN results

QUERY search_episodes_bm25(query: String, k: I32) =>
    results <- SearchBM25<Episode>(query, k)
    RETURN results

QUERY search_cues_bm25(query: String, k: I32) =>
    results <- SearchBM25<EpisodeCue>(query, k)
    RETURN results

// ============================================================
// QUERIES — Analytics / Stats
// ============================================================

// NOTE: count_entities_by_group, count_episodes_by_group, count_edges_by_group,
// and get_entity_type_counts_by_group removed — Python counts results from
// find_entities_by_group / find_episodes_by_group queries instead.

// ============================================================
// === Conversation Store ===
// ============================================================

// --- Node Types ---

N::Conversation {
    INDEX group_id: String,
    conversation_id: String,
    title: String,
    session_date: String,
    created_at: String,
    updated_at: String
}

N::ConversationMessage {
    INDEX conversation_id: String,
    message_id: String,
    role: String,
    content: String,
    parts_json: String,
    created_at: String
}

// --- Edge Types ---

E::HasConversationEntity {
    From: Conversation,
    To: Entity,
    Properties: {
    }
}

// --- Queries ---

QUERY create_conversation(conversation_id: String, group_id: String, title: String, session_date: String, created_at: String, updated_at: String) =>
    node <- AddN<Conversation>({conversation_id: conversation_id, group_id: group_id, title: title, session_date: session_date, created_at: created_at, updated_at: updated_at})
    RETURN node

QUERY get_conversation(id: ID) =>
    conversation <- N<Conversation>(id)
    RETURN conversation

QUERY find_conversations_by_group(gid: String) =>
    conversations <- N<Conversation>
        ::WHERE(_::{group_id}::EQ(gid))
    RETURN conversations

QUERY create_conversation_message(conversation_id: String, message_id: String, role: String, content: String, parts_json: String, created_at: String) =>
    node <- AddN<ConversationMessage>({conversation_id: conversation_id, message_id: message_id, role: role, content: content, parts_json: parts_json, created_at: created_at})
    RETURN node

QUERY find_messages_by_conversation(conv_id: String) =>
    messages <- N<ConversationMessage>
        ::WHERE(_::{conversation_id}::EQ(conv_id))
    RETURN messages

QUERY update_conversation(id: ID, title: String, updated_at: String) =>
    conversation <- N<Conversation>(id)
        ::UPDATE({title: title, updated_at: updated_at})
    RETURN conversation

QUERY link_conversation_entity(conv_id: ID, entity_id: ID) =>
    edge <- AddE<HasConversationEntity>
        ::From(conv_id)
        ::To(entity_id)
    RETURN edge

QUERY find_conversation_entities(conv_id: ID) =>
    entities <- N<Conversation>(conv_id)
        ::Out<HasConversationEntity>
    RETURN entities

QUERY delete_conversation(id: ID) =>
    DROP N<Conversation>(id)
    RETURN NONE

QUERY hard_delete_conversation_message(id: ID) =>
    DROP N<ConversationMessage>(id)
    RETURN NONE

// ============================================================
// === Atlas Store ===
// ============================================================

// --- Node Types ---

N::AtlasSnapshot {
    INDEX group_id: String,
    snapshot_id: String,
    generated_at: String,
    represented_entity_count: I32,
    represented_edge_count: I32,
    displayed_node_count: I32,
    displayed_edge_count: I32,
    total_entities: I32,
    total_relationships: I32,
    total_regions: I32,
    hottest_region_id: String,
    fastest_growing_region_id: String,
    truncated: Boolean
}

N::AtlasRegion {
    INDEX snapshot_id: String,
    INDEX group_id: String,
    region_id: String,
    region_label: String,
    subtitle: String,
    kind: String,
    member_count: I32,
    represented_edge_count: I32,
    activation_score: F64,
    growth_7d: I32,
    growth_30d: I32,
    dominant_entity_types_json: String,
    hub_entity_ids_json: String,
    center_entity_id: String,
    latest_entity_created_at: String,
    x: F64,
    y: F64,
    z: F64
}

N::AtlasRegionEdge {
    INDEX snapshot_id: String,
    INDEX group_id: String,
    edge_id: String,
    source_region_id: String,
    target_region_id: String,
    weight: F64,
    relationship_count: I32
}

N::AtlasRegionMember {
    INDEX snapshot_id: String,
    INDEX group_id: String,
    INDEX region_id: String,
    entity_id: String
}

// --- Queries ---

QUERY create_atlas_snapshot(snapshot_id: String, group_id: String, generated_at: String, represented_entity_count: I32, represented_edge_count: I32, displayed_node_count: I32, displayed_edge_count: I32, total_entities: I32, total_relationships: I32, total_regions: I32, hottest_region_id: String, fastest_growing_region_id: String, truncated: Boolean) =>
    node <- AddN<AtlasSnapshot>({snapshot_id: snapshot_id, group_id: group_id, generated_at: generated_at, represented_entity_count: represented_entity_count, represented_edge_count: represented_edge_count, displayed_node_count: displayed_node_count, displayed_edge_count: displayed_edge_count, total_entities: total_entities, total_relationships: total_relationships, total_regions: total_regions, hottest_region_id: hottest_region_id, fastest_growing_region_id: fastest_growing_region_id, truncated: truncated})
    RETURN node

QUERY find_atlas_snapshots_by_group(gid: String) =>
    snapshots <- N<AtlasSnapshot>
        ::WHERE(_::{group_id}::EQ(gid))
    RETURN snapshots

QUERY get_atlas_snapshot(id: ID) =>
    snapshot <- N<AtlasSnapshot>(id)
    RETURN snapshot

QUERY create_atlas_region(snapshot_id: String, group_id: String, region_id: String, region_label: String, subtitle: String, kind: String, member_count: I32, represented_edge_count: I32, activation_score: F64, growth_7d: I32, growth_30d: I32, dominant_entity_types_json: String, hub_entity_ids_json: String, center_entity_id: String, latest_entity_created_at: String, x: F64, y: F64, z: F64) =>
    node <- AddN<AtlasRegion>({snapshot_id: snapshot_id, group_id: group_id, region_id: region_id, region_label: region_label, subtitle: subtitle, kind: kind, member_count: member_count, represented_edge_count: represented_edge_count, activation_score: activation_score, growth_7d: growth_7d, growth_30d: growth_30d, dominant_entity_types_json: dominant_entity_types_json, hub_entity_ids_json: hub_entity_ids_json, center_entity_id: center_entity_id, latest_entity_created_at: latest_entity_created_at, x: x, y: y, z: z})
    RETURN node

QUERY find_atlas_regions(snap_id: String, gid: String) =>
    regions <- N<AtlasRegion>::WHERE(AND(_::{snapshot_id}::EQ(snap_id), _::{group_id}::EQ(gid)))
    RETURN regions

QUERY create_atlas_region_edge(snapshot_id: String, group_id: String, edge_id: String, source_region_id: String, target_region_id: String, weight: F64, relationship_count: I32) =>
    node <- AddN<AtlasRegionEdge>({snapshot_id: snapshot_id, group_id: group_id, edge_id: edge_id, source_region_id: source_region_id, target_region_id: target_region_id, weight: weight, relationship_count: relationship_count})
    RETURN node

QUERY find_atlas_region_edges(snap_id: String, gid: String) =>
    edges <- N<AtlasRegionEdge>::WHERE(AND(_::{snapshot_id}::EQ(snap_id), _::{group_id}::EQ(gid)))
    RETURN edges

QUERY create_atlas_region_member(snapshot_id: String, group_id: String, region_id: String, entity_id: String) =>
    node <- AddN<AtlasRegionMember>({snapshot_id: snapshot_id, group_id: group_id, region_id: region_id, entity_id: entity_id})
    RETURN node

QUERY find_atlas_region_members(snap_id: String, region_id: String, gid: String) =>
    members <- N<AtlasRegionMember>::WHERE(AND(_::{snapshot_id}::EQ(snap_id), _::{region_id}::EQ(region_id), _::{group_id}::EQ(gid)))
    RETURN members

QUERY hard_delete_atlas_region(id: ID) =>
    DROP N<AtlasRegion>(id)
    RETURN NONE

QUERY hard_delete_atlas_region_edge(id: ID) =>
    DROP N<AtlasRegionEdge>(id)
    RETURN NONE

QUERY hard_delete_atlas_region_member(id: ID) =>
    DROP N<AtlasRegionMember>(id)
    RETURN NONE

QUERY delete_atlas_snapshot(id: ID) =>
    DROP N<AtlasSnapshot>(id)
    RETURN NONE

// ============================================================
// === Consolidation Store ===
// ============================================================

// --- Node Types ---

N::ConsolCycle {
    INDEX group_id: String,
    cycle_id: String,
    trigger: String,
    dry_run: Boolean,
    status: String,
    phase_results_json: String,
    started_at: F64,
    completed_at: F64,
    total_duration_ms: F64,
    error: String
}

N::ConsolMerge {
    INDEX cycle_id: String,
    INDEX group_id: String,
    merge_id: String,
    keep_id: String,
    remove_id: String,
    keep_name: String,
    remove_name: String,
    similarity: F64,
    decision_confidence: F64,
    decision_source: String,
    decision_reason: String,
    relationships_transferred: I32,
    timestamp: F64
}

N::ConsolIdentifierReview {
    INDEX cycle_id: String,
    INDEX group_id: String,
    review_id: String,
    entity_a_id: String,
    entity_b_id: String,
    entity_a_name: String,
    entity_b_name: String,
    entity_a_type: String,
    entity_b_type: String,
    raw_similarity: F64,
    adjusted_similarity: F64,
    decision_source: String,
    decision_reason: String,
    entity_a_regime: String,
    entity_b_regime: String,
    canonical_identifier_a: String,
    canonical_identifier_b: String,
    review_status: String,
    metadata_json: String,
    timestamp: F64
}

N::ConsolInferredEdge {
    INDEX cycle_id: String,
    INDEX group_id: String,
    edge_id: String,
    source_id: String,
    target_id: String,
    source_name: String,
    target_name: String,
    co_occurrence_count: I32,
    confidence: F64,
    infer_type: String,
    pmi_score: F64,
    llm_verdict: String,
    relationship_id: String,
    timestamp: F64
}

N::ConsolPrune {
    INDEX cycle_id: String,
    INDEX group_id: String,
    prune_id: String,
    entity_id: String,
    entity_name: String,
    entity_type: String,
    reason: String,
    timestamp: F64
}

N::ConsolReindex {
    INDEX cycle_id: String,
    INDEX group_id: String,
    reindex_id: String,
    entity_id: String,
    entity_name: String,
    source_phase: String,
    timestamp: F64
}

N::ConsolReplay {
    INDEX cycle_id: String,
    INDEX group_id: String,
    replay_id: String,
    episode_id: String,
    new_entities_found: I32,
    new_relationships_found: I32,
    entities_updated: I32,
    skipped_reason: String,
    timestamp: F64
}

N::ConsolDream {
    INDEX cycle_id: String,
    INDEX group_id: String,
    dream_id: String,
    source_entity_id: String,
    target_entity_id: String,
    weight_delta: F64,
    seed_entity_id: String,
    timestamp: F64
}

N::ConsolTriage {
    INDEX cycle_id: String,
    INDEX group_id: String,
    triage_id: String,
    episode_id: String,
    score: F64,
    decision: String,
    score_breakdown_json: String,
    timestamp: F64
}

N::ConsolDreamAssociation {
    INDEX cycle_id: String,
    INDEX group_id: String,
    assoc_id: String,
    source_entity_id: String,
    target_entity_id: String,
    source_entity_name: String,
    target_entity_name: String,
    source_domain: String,
    target_domain: String,
    surprise_score: F64,
    embedding_similarity: F64,
    structural_proximity: F64,
    relationship_id: String,
    timestamp: F64
}

N::ConsolGraphEmbed {
    INDEX cycle_id: String,
    INDEX group_id: String,
    embed_id: String,
    method: String,
    entities_trained: I32,
    dimensions: I32,
    training_duration_ms: F64,
    full_retrain: Boolean,
    timestamp: F64
}

N::ConsolMaturation {
    INDEX cycle_id: String,
    INDEX group_id: String,
    mat_id: String,
    entity_id: String,
    entity_name: String,
    old_tier: String,
    new_tier: String,
    maturity_score: F64,
    source_diversity: I32,
    temporal_span_days: F64,
    relationship_richness: I32,
    access_regularity: F64,
    timestamp: F64
}

N::ConsolSemanticTransition {
    INDEX cycle_id: String,
    INDEX group_id: String,
    trans_id: String,
    episode_id: String,
    old_tier: String,
    new_tier: String,
    entity_coverage: F64,
    consolidation_cycles: I32,
    timestamp: F64
}

N::ConsolSchema {
    INDEX cycle_id: String,
    INDEX group_id: String,
    schema_id: String,
    schema_entity_id: String,
    schema_name: String,
    instance_count: I32,
    predicate_count: I32,
    action: String,
    timestamp: F64
}

N::ConsolDecisionTrace {
    INDEX cycle_id: String,
    INDEX group_id: String,
    trace_id: String,
    phase: String,
    candidate_type: String,
    candidate_id: String,
    decision: String,
    decision_source: String,
    confidence: F64,
    threshold_band: String,
    features_json: String,
    constraints_json: String,
    policy_version: String,
    metadata_json: String,
    timestamp: F64
}

N::ConsolDecisionOutcome {
    INDEX cycle_id: String,
    INDEX group_id: String,
    outcome_id: String,
    phase: String,
    decision_trace_id: String,
    outcome_type: String,
    outcome_label: String,
    outcome_value: F64,
    metadata_json: String,
    timestamp: F64
}

N::ConsolDistillation {
    INDEX cycle_id: String,
    INDEX group_id: String,
    distill_id: String,
    phase: String,
    candidate_type: String,
    candidate_id: String,
    decision_trace_id: String,
    teacher_label: String,
    teacher_source: String,
    student_decision: String,
    student_confidence: F64,
    threshold_band: String,
    features_json: String,
    correct: Boolean,
    metadata_json: String,
    timestamp: F64
}

N::ConsolCalibration {
    INDEX cycle_id: String,
    INDEX group_id: String,
    calibration_id: String,
    phase: String,
    window_cycles: I32,
    total_traces: I32,
    labeled_examples: I32,
    oracle_examples: I32,
    abstain_count: I32,
    accuracy: F64,
    mean_confidence: F64,
    expected_calibration_error: F64,
    summary_json: String,
    timestamp: F64
}

N::ConsolEvidenceAdj {
    INDEX cycle_id: String,
    INDEX group_id: String,
    adj_id: String,
    evidence_id: String,
    action: String,
    new_confidence: F64,
    reason: String,
    timestamp: F64
}

N::ComplementTag {
    INDEX target_id: String,
    INDEX group_id: String,
    target_type: String,
    tag_type: String,
    score: F64,
    cycle_tagged: I32,
    cycle_confirmed: I32,
    cleared: Boolean,
    created_at: String,
    updated_at: String
}

N::ConsolMicroglia {
    INDEX cycle_id: String,
    INDEX group_id: String,
    microglia_id: String,
    target_type: String,
    target_id: String,
    action: String,
    tag_type: String,
    score: F64,
    detail: String,
    timestamp: F64
}

// --- Consolidation Queries: ConsolCycle ---

QUERY create_consol_cycle(cycle_id: String, group_id: String, trigger: String, dry_run: Boolean, status: String, phase_results_json: String, started_at: F64, completed_at: F64, total_duration_ms: F64, error: String) =>
    node <- AddN<ConsolCycle>({cycle_id: cycle_id, group_id: group_id, trigger: trigger, dry_run: dry_run, status: status, phase_results_json: phase_results_json, started_at: started_at, completed_at: completed_at, total_duration_ms: total_duration_ms, error: error})
    RETURN node

QUERY find_consol_cycles_by_group(gid: String) =>
    cycles <- N<ConsolCycle>
        ::WHERE(_::{group_id}::EQ(gid))
    RETURN cycles

QUERY get_consol_cycle(id: ID) =>
    cycle <- N<ConsolCycle>(id)
    RETURN cycle

QUERY update_consol_cycle(id: ID, status: String, phase_results_json: String, completed_at: F64, total_duration_ms: F64, error: String) =>
    cycle <- N<ConsolCycle>(id)
        ::UPDATE({status: status, phase_results_json: phase_results_json, completed_at: completed_at, total_duration_ms: total_duration_ms, error: error})
    RETURN cycle

QUERY find_consol_cycles_by_cycle(cycle_id: String, gid: String) =>
    cycles <- N<ConsolCycle>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN cycles

// --- Consolidation Queries: ConsolMerge ---

QUERY create_consol_merge(merge_id: String, cycle_id: String, group_id: String, keep_id: String, remove_id: String, keep_name: String, remove_name: String, similarity: F64, decision_confidence: F64, decision_source: String, decision_reason: String, relationships_transferred: I32, timestamp: F64) =>
    node <- AddN<ConsolMerge>({merge_id: merge_id, cycle_id: cycle_id, group_id: group_id, keep_id: keep_id, remove_id: remove_id, keep_name: keep_name, remove_name: remove_name, similarity: similarity, decision_confidence: decision_confidence, decision_source: decision_source, decision_reason: decision_reason, relationships_transferred: relationships_transferred, timestamp: timestamp})
    RETURN node

QUERY find_consol_merges_by_cycle(cycle_id: String, gid: String) =>
    merges <- N<ConsolMerge>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN merges

// --- Consolidation Queries: ConsolIdentifierReview ---

QUERY create_consol_identifier_review(review_id: String, cycle_id: String, group_id: String, entity_a_id: String, entity_b_id: String, entity_a_name: String, entity_b_name: String, entity_a_type: String, entity_b_type: String, raw_similarity: F64, adjusted_similarity: F64, decision_source: String, decision_reason: String, entity_a_regime: String, entity_b_regime: String, canonical_identifier_a: String, canonical_identifier_b: String, review_status: String, metadata_json: String, timestamp: F64) =>
    node <- AddN<ConsolIdentifierReview>({review_id: review_id, cycle_id: cycle_id, group_id: group_id, entity_a_id: entity_a_id, entity_b_id: entity_b_id, entity_a_name: entity_a_name, entity_b_name: entity_b_name, entity_a_type: entity_a_type, entity_b_type: entity_b_type, raw_similarity: raw_similarity, adjusted_similarity: adjusted_similarity, decision_source: decision_source, decision_reason: decision_reason, entity_a_regime: entity_a_regime, entity_b_regime: entity_b_regime, canonical_identifier_a: canonical_identifier_a, canonical_identifier_b: canonical_identifier_b, review_status: review_status, metadata_json: metadata_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_identifier_reviews_by_cycle(cycle_id: String, gid: String) =>
    reviews <- N<ConsolIdentifierReview>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN reviews

// --- Consolidation Queries: ConsolInferredEdge ---

QUERY create_consol_inferred_edge(edge_id: String, cycle_id: String, group_id: String, source_id: String, target_id: String, source_name: String, target_name: String, co_occurrence_count: I32, confidence: F64, infer_type: String, pmi_score: F64, llm_verdict: String, relationship_id: String, timestamp: F64) =>
    node <- AddN<ConsolInferredEdge>({edge_id: edge_id, cycle_id: cycle_id, group_id: group_id, source_id: source_id, target_id: target_id, source_name: source_name, target_name: target_name, co_occurrence_count: co_occurrence_count, confidence: confidence, infer_type: infer_type, pmi_score: pmi_score, llm_verdict: llm_verdict, relationship_id: relationship_id, timestamp: timestamp})
    RETURN node

QUERY find_consol_inferred_edges_by_cycle(cycle_id: String, gid: String) =>
    edges <- N<ConsolInferredEdge>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN edges

// --- Consolidation Queries: ConsolPrune ---

QUERY create_consol_prune(prune_id: String, cycle_id: String, group_id: String, entity_id: String, entity_name: String, entity_type: String, reason: String, timestamp: F64) =>
    node <- AddN<ConsolPrune>({prune_id: prune_id, cycle_id: cycle_id, group_id: group_id, entity_id: entity_id, entity_name: entity_name, entity_type: entity_type, reason: reason, timestamp: timestamp})
    RETURN node

QUERY find_consol_prunes_by_cycle(cycle_id: String, gid: String) =>
    prunes <- N<ConsolPrune>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN prunes

// --- Consolidation Queries: ConsolReindex ---

QUERY create_consol_reindex(reindex_id: String, cycle_id: String, group_id: String, entity_id: String, entity_name: String, source_phase: String, timestamp: F64) =>
    node <- AddN<ConsolReindex>({reindex_id: reindex_id, cycle_id: cycle_id, group_id: group_id, entity_id: entity_id, entity_name: entity_name, source_phase: source_phase, timestamp: timestamp})
    RETURN node

QUERY find_consol_reindexes_by_cycle(cycle_id: String, gid: String) =>
    reindexes <- N<ConsolReindex>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN reindexes

// --- Consolidation Queries: ConsolReplay ---

QUERY create_consol_replay(replay_id: String, cycle_id: String, group_id: String, episode_id: String, new_entities_found: I32, new_relationships_found: I32, entities_updated: I32, skipped_reason: String, timestamp: F64) =>
    node <- AddN<ConsolReplay>({replay_id: replay_id, cycle_id: cycle_id, group_id: group_id, episode_id: episode_id, new_entities_found: new_entities_found, new_relationships_found: new_relationships_found, entities_updated: entities_updated, skipped_reason: skipped_reason, timestamp: timestamp})
    RETURN node

QUERY find_consol_replays_by_cycle(cycle_id: String, gid: String) =>
    replays <- N<ConsolReplay>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN replays

// --- Consolidation Queries: ConsolDream ---

QUERY create_consol_dream(dream_id: String, cycle_id: String, group_id: String, source_entity_id: String, target_entity_id: String, weight_delta: F64, seed_entity_id: String, timestamp: F64) =>
    node <- AddN<ConsolDream>({dream_id: dream_id, cycle_id: cycle_id, group_id: group_id, source_entity_id: source_entity_id, target_entity_id: target_entity_id, weight_delta: weight_delta, seed_entity_id: seed_entity_id, timestamp: timestamp})
    RETURN node

QUERY find_consol_dreams_by_cycle(cycle_id: String, gid: String) =>
    dreams <- N<ConsolDream>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN dreams

// --- Consolidation Queries: ConsolTriage ---

QUERY create_consol_triage(triage_id: String, cycle_id: String, group_id: String, episode_id: String, score: F64, decision: String, score_breakdown_json: String, timestamp: F64) =>
    node <- AddN<ConsolTriage>({triage_id: triage_id, cycle_id: cycle_id, group_id: group_id, episode_id: episode_id, score: score, decision: decision, score_breakdown_json: score_breakdown_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_triages_by_cycle(cycle_id: String, gid: String) =>
    triages <- N<ConsolTriage>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN triages

// --- Consolidation Queries: ConsolDreamAssociation ---

QUERY create_consol_dream_association(assoc_id: String, cycle_id: String, group_id: String, source_entity_id: String, target_entity_id: String, source_entity_name: String, target_entity_name: String, source_domain: String, target_domain: String, surprise_score: F64, embedding_similarity: F64, structural_proximity: F64, relationship_id: String, timestamp: F64) =>
    node <- AddN<ConsolDreamAssociation>({assoc_id: assoc_id, cycle_id: cycle_id, group_id: group_id, source_entity_id: source_entity_id, target_entity_id: target_entity_id, source_entity_name: source_entity_name, target_entity_name: target_entity_name, source_domain: source_domain, target_domain: target_domain, surprise_score: surprise_score, embedding_similarity: embedding_similarity, structural_proximity: structural_proximity, relationship_id: relationship_id, timestamp: timestamp})
    RETURN node

QUERY find_consol_dream_associations_by_cycle(cycle_id: String, gid: String) =>
    assocs <- N<ConsolDreamAssociation>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN assocs

// --- Consolidation Queries: ConsolGraphEmbed ---

QUERY create_consol_graph_embed(embed_id: String, cycle_id: String, group_id: String, method: String, entities_trained: I32, dimensions: I32, training_duration_ms: F64, full_retrain: Boolean, timestamp: F64) =>
    node <- AddN<ConsolGraphEmbed>({embed_id: embed_id, cycle_id: cycle_id, group_id: group_id, method: method, entities_trained: entities_trained, dimensions: dimensions, training_duration_ms: training_duration_ms, full_retrain: full_retrain, timestamp: timestamp})
    RETURN node

QUERY find_consol_graph_embeds_by_cycle(cycle_id: String, gid: String) =>
    embeds <- N<ConsolGraphEmbed>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN embeds

// --- Consolidation Queries: ConsolMaturation ---

QUERY create_consol_maturation(mat_id: String, cycle_id: String, group_id: String, entity_id: String, entity_name: String, old_tier: String, new_tier: String, maturity_score: F64, source_diversity: I32, temporal_span_days: F64, relationship_richness: I32, access_regularity: F64, timestamp: F64) =>
    node <- AddN<ConsolMaturation>({mat_id: mat_id, cycle_id: cycle_id, group_id: group_id, entity_id: entity_id, entity_name: entity_name, old_tier: old_tier, new_tier: new_tier, maturity_score: maturity_score, source_diversity: source_diversity, temporal_span_days: temporal_span_days, relationship_richness: relationship_richness, access_regularity: access_regularity, timestamp: timestamp})
    RETURN node

QUERY find_consol_maturations_by_cycle(cycle_id: String, gid: String) =>
    maturations <- N<ConsolMaturation>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN maturations

// --- Consolidation Queries: ConsolSemanticTransition ---

QUERY create_consol_semantic_transition(trans_id: String, cycle_id: String, group_id: String, episode_id: String, old_tier: String, new_tier: String, entity_coverage: F64, consolidation_cycles: I32, timestamp: F64) =>
    node <- AddN<ConsolSemanticTransition>({trans_id: trans_id, cycle_id: cycle_id, group_id: group_id, episode_id: episode_id, old_tier: old_tier, new_tier: new_tier, entity_coverage: entity_coverage, consolidation_cycles: consolidation_cycles, timestamp: timestamp})
    RETURN node

QUERY find_consol_semantic_transitions_by_cycle(cycle_id: String, gid: String) =>
    transitions <- N<ConsolSemanticTransition>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN transitions

// --- Consolidation Queries: ConsolSchema ---

QUERY create_consol_schema(schema_id: String, cycle_id: String, group_id: String, schema_entity_id: String, schema_name: String, instance_count: I32, predicate_count: I32, action: String, timestamp: F64) =>
    node <- AddN<ConsolSchema>({schema_id: schema_id, cycle_id: cycle_id, group_id: group_id, schema_entity_id: schema_entity_id, schema_name: schema_name, instance_count: instance_count, predicate_count: predicate_count, action: action, timestamp: timestamp})
    RETURN node

QUERY find_consol_schemas_by_cycle(cycle_id: String, gid: String) =>
    schemas <- N<ConsolSchema>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN schemas

// --- Consolidation Queries: ConsolDecisionTrace ---

QUERY create_consol_decision_trace(trace_id: String, cycle_id: String, group_id: String, phase: String, candidate_type: String, candidate_id: String, decision: String, decision_source: String, confidence: F64, threshold_band: String, features_json: String, constraints_json: String, policy_version: String, metadata_json: String, timestamp: F64) =>
    node <- AddN<ConsolDecisionTrace>({trace_id: trace_id, cycle_id: cycle_id, group_id: group_id, phase: phase, candidate_type: candidate_type, candidate_id: candidate_id, decision: decision, decision_source: decision_source, confidence: confidence, threshold_band: threshold_band, features_json: features_json, constraints_json: constraints_json, policy_version: policy_version, metadata_json: metadata_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_decision_traces_by_cycle(cycle_id: String, gid: String) =>
    traces <- N<ConsolDecisionTrace>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN traces

// --- Consolidation Queries: ConsolDecisionOutcome ---

QUERY create_consol_decision_outcome(outcome_id: String, cycle_id: String, group_id: String, phase: String, decision_trace_id: String, outcome_type: String, outcome_label: String, outcome_value: F64, metadata_json: String, timestamp: F64) =>
    node <- AddN<ConsolDecisionOutcome>({outcome_id: outcome_id, cycle_id: cycle_id, group_id: group_id, phase: phase, decision_trace_id: decision_trace_id, outcome_type: outcome_type, outcome_label: outcome_label, outcome_value: outcome_value, metadata_json: metadata_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_decision_outcomes_by_cycle(cycle_id: String, gid: String) =>
    outcomes <- N<ConsolDecisionOutcome>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN outcomes

// --- Consolidation Queries: ConsolDistillation ---

QUERY create_consol_distillation(distill_id: String, cycle_id: String, group_id: String, phase: String, candidate_type: String, candidate_id: String, decision_trace_id: String, teacher_label: String, teacher_source: String, student_decision: String, student_confidence: F64, threshold_band: String, features_json: String, correct: Boolean, metadata_json: String, timestamp: F64) =>
    node <- AddN<ConsolDistillation>({distill_id: distill_id, cycle_id: cycle_id, group_id: group_id, phase: phase, candidate_type: candidate_type, candidate_id: candidate_id, decision_trace_id: decision_trace_id, teacher_label: teacher_label, teacher_source: teacher_source, student_decision: student_decision, student_confidence: student_confidence, threshold_band: threshold_band, features_json: features_json, correct: correct, metadata_json: metadata_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_distillations_by_cycle(cycle_id: String, gid: String) =>
    distillations <- N<ConsolDistillation>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN distillations

// --- Consolidation Queries: ConsolCalibration ---

QUERY create_consol_calibration(calibration_id: String, cycle_id: String, group_id: String, phase: String, window_cycles: I32, total_traces: I32, labeled_examples: I32, oracle_examples: I32, abstain_count: I32, accuracy: F64, mean_confidence: F64, expected_calibration_error: F64, summary_json: String, timestamp: F64) =>
    node <- AddN<ConsolCalibration>({calibration_id: calibration_id, cycle_id: cycle_id, group_id: group_id, phase: phase, window_cycles: window_cycles, total_traces: total_traces, labeled_examples: labeled_examples, oracle_examples: oracle_examples, abstain_count: abstain_count, accuracy: accuracy, mean_confidence: mean_confidence, expected_calibration_error: expected_calibration_error, summary_json: summary_json, timestamp: timestamp})
    RETURN node

QUERY find_consol_calibrations_by_cycle(cycle_id: String, gid: String) =>
    calibrations <- N<ConsolCalibration>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN calibrations

// --- Consolidation Queries: ConsolEvidenceAdj ---

QUERY create_consol_evidence_adj(adj_id: String, cycle_id: String, group_id: String, evidence_id: String, action: String, new_confidence: F64, reason: String, timestamp: F64) =>
    node <- AddN<ConsolEvidenceAdj>({adj_id: adj_id, cycle_id: cycle_id, group_id: group_id, evidence_id: evidence_id, action: action, new_confidence: new_confidence, reason: reason, timestamp: timestamp})
    RETURN node

QUERY find_consol_evidence_adjs_by_cycle(cycle_id: String, gid: String) =>
    adjs <- N<ConsolEvidenceAdj>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN adjs

// --- Consolidation Queries: ComplementTag ---

QUERY create_complement_tag(target_type: String, target_id: String, tag_type: String, score: F64, cycle_tagged: I32, cycle_confirmed: I32, cleared: Boolean, group_id: String, created_at: String, updated_at: String) =>
    node <- AddN<ComplementTag>({target_type: target_type, target_id: target_id, tag_type: tag_type, score: score, cycle_tagged: cycle_tagged, cycle_confirmed: cycle_confirmed, cleared: cleared, group_id: group_id, created_at: created_at, updated_at: updated_at})
    RETURN node

QUERY find_active_complement_tags(gid: String) =>
    tags <- N<ComplementTag>::WHERE(AND(_::{group_id}::EQ(gid), _::{cleared}::EQ(false)))
    RETURN tags

QUERY find_complement_tags_by_target(target_id: String) =>
    tags <- N<ComplementTag>
        ::WHERE(_::{target_id}::EQ(target_id))
    RETURN tags

QUERY update_complement_tag(id: ID, cycle_confirmed: I32, cleared: Boolean, updated_at: String) =>
    tag <- N<ComplementTag>(id)
        ::UPDATE({cycle_confirmed: cycle_confirmed, cleared: cleared, updated_at: updated_at})
    RETURN tag

QUERY find_confirmed_complement_tags(gid: String) =>
    tags <- N<ComplementTag>::WHERE(AND(_::{group_id}::EQ(gid), _::{cleared}::EQ(false)))
    RETURN tags

QUERY find_unconfirmed_complement_tags(gid: String) =>
    tags <- N<ComplementTag>::WHERE(AND(_::{group_id}::EQ(gid), _::{cleared}::EQ(false)))
    RETURN tags

// --- Consolidation Queries: ConsolMicroglia ---

QUERY create_consol_microglia(microglia_id: String, cycle_id: String, group_id: String, target_type: String, target_id: String, action: String, tag_type: String, score: F64, detail: String, timestamp: F64) =>
    node <- AddN<ConsolMicroglia>({microglia_id: microglia_id, cycle_id: cycle_id, group_id: group_id, target_type: target_type, target_id: target_id, action: action, tag_type: tag_type, score: score, detail: detail, timestamp: timestamp})
    RETURN node

QUERY find_consol_microglias_by_cycle(cycle_id: String, gid: String) =>
    records <- N<ConsolMicroglia>::WHERE(AND(_::{cycle_id}::EQ(cycle_id), _::{group_id}::EQ(gid)))
    RETURN records

// ============================================================
// QUERIES — Server-Side Embedding Search (Task 17)
// ============================================================

QUERY search_entities_embed(query: String, k: I32) =>
    results <- SearchV<EntityVec>(Embed(query), k)
    RETURN results

QUERY search_episodes_embed(query: String, k: I32) =>
    results <- SearchV<EpisodeVec>(Embed(query), k)
    RETURN results

QUERY search_cues_embed(query: String, k: I32) =>
    results <- SearchV<CueVec>(Embed(query), k)
    RETURN results

// ============================================================
// QUERIES — Post-Filtered Vector Search (Task 18)
// ============================================================

QUERY search_entity_vectors_filtered(vec: [F64], k: I32, gid: String) =>
    results <- SearchV<EntityVec>(vec, k)::WHERE(_::{group_id}::EQ(gid))
    RETURN results

QUERY search_episode_vectors_filtered(vec: [F64], k: I32, gid: String) =>
    results <- SearchV<EpisodeVec>(vec, k)::WHERE(_::{group_id}::EQ(gid))
    RETURN results

QUERY search_cue_vectors_filtered(vec: [F64], k: I32, gid: String) =>
    results <- SearchV<CueVec>(vec, k)::WHERE(_::{group_id}::EQ(gid))
    RETURN results

// ============================================================
// QUERIES — ShortestPath Graph Algorithms (Task 19)
// ============================================================

QUERY shortest_path_bfs(start: ID, end: ID) =>
    path <- N<Entity>(start)::ShortestPathBFS<RelatesTo>::To(end)
    RETURN path

QUERY shortest_path_weighted(start: ID, end: ID) =>
    path <- N<Entity>(start)::ShortestPathDijkstras<RelatesTo>(_::{weight})::To(end)
    RETURN path

// ============================================================
// QUERIES — Multi-Hop Chained Traversals (Task 21)
// ============================================================

QUERY get_two_hop_neighbors(id: ID) =>
    neighbors <- N<Entity>(id)::Out<RelatesTo>::Out<RelatesTo>
    RETURN neighbors

QUERY get_entity_cooccurrences(id: ID) =>
    cooccurring <- N<Entity>(id)::In<HasEntity>::Out<HasEntity>
    RETURN cooccurring

QUERY get_entity_neighborhood(id: ID) =>
    edges <- N<Entity>(id)::OutE<RelatesTo>
    neighbors <- N<Entity>(id)::Out<RelatesTo>
    RETURN {edges: edges, neighbors: neighbors}

// ============================================================
// QUERIES — RANGE Pagination & Property Projection (Task 22)
// ============================================================

QUERY find_entities_by_group_limited(gid: String, limit: I64) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{is_deleted}::EQ(false)))::RANGE(0, limit)
    RETURN entities

QUERY find_episodes_by_group_limited(gid: String, limit: I64) =>
    episodes <- N<Episode>::WHERE(_::{group_id}::EQ(gid))::RANGE(0, limit)
    RETURN episodes

QUERY find_entity_ids_by_group(gid: String) =>
    entities <- N<Entity>::WHERE(AND(_::{group_id}::EQ(gid), _::{is_deleted}::EQ(false)))
    RETURN entities::{entity_id: entity_id, name: name, entity_type: entity_type}

// ============================================================
// Episode Chunking — Types
// ============================================================

V::EpisodeChunk {
    episode_id: String,
    group_id: String,
    chunk_text: String,
    chunk_index: I32,
    content_type: String
}

E::HasEpisodeChunk {
    From: Episode,
    To: EpisodeChunk,
    Properties: {
    }
}

// ============================================================
// QUERIES — Episode Chunking (server-side Embed + Chunk)
// ============================================================

// Create a chunk with server-side embedding via Embed()
QUERY create_episode_chunk_embed(episode_id: String, group_id: String, chunk_text: String, chunk_index: I32, content_type: String) =>
    chunk <- AddV<EpisodeChunk>(Embed(chunk_text), {episode_id: episode_id, group_id: group_id, chunk_text: chunk_text, chunk_index: chunk_index, content_type: content_type})
    RETURN chunk

// Create a chunk with pre-computed vector (for client-side Gemini embedding)
QUERY create_episode_chunk_vec(episode_id: String, group_id: String, chunk_text: String, chunk_index: I32, content_type: String, vec: [F64]) =>
    chunk <- AddV<EpisodeChunk>(vec, {episode_id: episode_id, group_id: group_id, chunk_text: chunk_text, chunk_index: chunk_index, content_type: content_type})
    RETURN chunk

// Link a chunk to its parent episode
QUERY link_episode_chunk(episode_id: ID, chunk_id: ID) =>
    edge <- AddE<HasEpisodeChunk>::From(episode_id)::To(chunk_id)
    RETURN edge

// Search chunks with server-side embedding
QUERY search_episode_chunks_embed(query: String, k: I32) =>
    results <- SearchV<EpisodeChunk>(Embed(query), k)
    RETURN results

// Search chunks with pre-computed vector
QUERY search_episode_chunks_vec(vec: [F64], k: I32) =>
    results <- SearchV<EpisodeChunk>(vec, k)
    RETURN results

// Search chunks with group filtering
QUERY search_episode_chunks_filtered(vec: [F64], k: I32, gid: String) =>
    results <- SearchV<EpisodeChunk>(vec, k)::WHERE(_::{group_id}::EQ(gid))
    RETURN results

// Search chunks with server-side embed + group filtering
QUERY search_episode_chunks_embed_filtered(query: String, k: I32, gid: String) =>
    results <- SearchV<EpisodeChunk>(Embed(query), k)::WHERE(_::{group_id}::EQ(gid))
    RETURN results

// Get all chunks for an episode
QUERY get_episode_chunks(ep_id: String) =>
    chunks <- N<Episode>::WHERE(_::{episode_id}::EQ(ep_id))::Out<HasEpisodeChunk>
    RETURN chunks

// Delete a single chunk
QUERY hard_delete_episode_chunk(id: ID) =>
    DROP V<EpisodeChunk>(id)
    RETURN NONE
