-- Engram SQLite Schema (Lite Mode)

CREATE TABLE IF NOT EXISTS entities (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    entity_type     TEXT NOT NULL,
    summary         TEXT,
    attributes      TEXT,            -- JSON blob
    group_id        TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,    -- ISO 8601
    updated_at      TEXT NOT NULL,
    deleted_at      TEXT,            -- soft delete
    activation_base REAL NOT NULL DEFAULT 0.5,
    activation_current REAL NOT NULL DEFAULT 0.5,
    access_count    INTEGER NOT NULL DEFAULT 0,
    last_accessed   TEXT,
    lexical_regime  TEXT,
    canonical_identifier TEXT,
    identifier_label INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id);
CREATE INDEX IF NOT EXISTS idx_entities_lexical_regime ON entities(lexical_regime);
CREATE INDEX IF NOT EXISTS idx_entities_canonical_identifier ON entities(canonical_identifier);

CREATE TABLE IF NOT EXISTS relationships (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES entities(id),
    target_id       TEXT NOT NULL REFERENCES entities(id),
    predicate       TEXT NOT NULL,
    weight          REAL NOT NULL DEFAULT 1.0,
    valid_from      TEXT,
    valid_to        TEXT,
    created_at      TEXT NOT NULL,
    source_episode  TEXT,
    group_id        TEXT NOT NULL DEFAULT 'default'
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_predicate ON relationships(predicate);
CREATE INDEX IF NOT EXISTS idx_rel_group ON relationships(group_id);

CREATE TABLE IF NOT EXISTS episodes (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    source          TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    group_id        TEXT NOT NULL DEFAULT 'default',
    session_id      TEXT,
    created_at      TEXT NOT NULL,
    encoding_context TEXT,
    memory_tier     TEXT DEFAULT 'episodic',
    consolidation_cycles INTEGER DEFAULT 0,
    entity_coverage REAL DEFAULT 0.0,
    projection_state TEXT DEFAULT 'queued',
    last_projection_reason TEXT,
    last_projected_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_episodes_group ON episodes(group_id);

CREATE TABLE IF NOT EXISTS episode_cues (
    episode_id              TEXT PRIMARY KEY REFERENCES episodes(id) ON DELETE CASCADE,
    group_id                TEXT NOT NULL DEFAULT 'default',
    cue_version             INTEGER NOT NULL DEFAULT 1,
    discourse_class         TEXT NOT NULL DEFAULT 'world',
    projection_state        TEXT NOT NULL DEFAULT 'cued',
    cue_score               REAL NOT NULL DEFAULT 0.0,
    salience_score          REAL NOT NULL DEFAULT 0.0,
    projection_priority     REAL NOT NULL DEFAULT 0.0,
    route_reason            TEXT,
    cue_text                TEXT NOT NULL,
    entity_mentions_json    TEXT NOT NULL DEFAULT '[]',
    temporal_markers_json   TEXT NOT NULL DEFAULT '[]',
    quote_spans_json        TEXT NOT NULL DEFAULT '[]',
    contradiction_keys_json TEXT NOT NULL DEFAULT '[]',
    first_spans_json        TEXT NOT NULL DEFAULT '[]',
    hit_count               INTEGER NOT NULL DEFAULT 0,
    surfaced_count          INTEGER NOT NULL DEFAULT 0,
    selected_count          INTEGER NOT NULL DEFAULT 0,
    used_count              INTEGER NOT NULL DEFAULT 0,
    near_miss_count         INTEGER NOT NULL DEFAULT 0,
    policy_score            REAL NOT NULL DEFAULT 0.0,
    projection_attempts     INTEGER NOT NULL DEFAULT 0,
    last_hit_at             TEXT,
    last_feedback_at        TEXT,
    last_projected_at       TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episode_cues_group ON episode_cues(group_id);
CREATE INDEX IF NOT EXISTS idx_episode_cues_projection_state ON episode_cues(projection_state);
CREATE INDEX IF NOT EXISTS idx_episode_cues_priority ON episode_cues(projection_priority);
CREATE INDEX IF NOT EXISTS idx_episode_cues_policy_score ON episode_cues(policy_score);

CREATE TABLE IF NOT EXISTS episode_entities (
    episode_id      TEXT NOT NULL REFERENCES episodes(id),
    entity_id       TEXT NOT NULL REFERENCES entities(id),
    PRIMARY KEY (episode_id, entity_id)
);

-- FTS5 virtual table for text search
CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
    name,
    summary,
    entity_type,
    content=entities,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    content,
    content=episodes,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episode_cues_fts USING fts5(
    cue_text,
    content=episode_cues,
    content_rowid=rowid,
    tokenize='porter unicode61'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
    INSERT INTO entities_fts(rowid, name, summary, entity_type)
    VALUES (new.rowid, new.name, new.summary, new.entity_type);
END;

CREATE TRIGGER IF NOT EXISTS entities_au AFTER UPDATE ON entities BEGIN
    INSERT INTO entities_fts(entities_fts, rowid, name, summary, entity_type)
    VALUES ('delete', old.rowid, old.name, old.summary, old.entity_type);
    INSERT INTO entities_fts(rowid, name, summary, entity_type)
    VALUES (new.rowid, new.name, new.summary, new.entity_type);
END;

-- Prospective memory (Wave 4)
CREATE TABLE IF NOT EXISTS intentions (
    id              TEXT PRIMARY KEY,
    trigger_text    TEXT NOT NULL,
    action_text     TEXT NOT NULL,
    trigger_type    TEXT NOT NULL DEFAULT 'semantic',
    entity_name     TEXT,
    threshold       REAL NOT NULL DEFAULT 0.7,
    max_fires       INTEGER NOT NULL DEFAULT 5,
    fire_count      INTEGER NOT NULL DEFAULT 0,
    enabled         INTEGER NOT NULL DEFAULT 1,
    group_id        TEXT NOT NULL DEFAULT 'default',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    expires_at      TEXT
);

CREATE INDEX IF NOT EXISTS idx_intentions_group ON intentions(group_id);
CREATE INDEX IF NOT EXISTS idx_intentions_entity ON intentions(entity_name);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS episode_cues_ai AFTER INSERT ON episode_cues BEGIN
    INSERT INTO episode_cues_fts(rowid, cue_text)
    VALUES (new.rowid, new.cue_text);
END;

CREATE TRIGGER IF NOT EXISTS episode_cues_au AFTER UPDATE ON episode_cues BEGIN
    INSERT INTO episode_cues_fts(episode_cues_fts, rowid, cue_text)
    VALUES ('delete', old.rowid, old.cue_text);
    INSERT INTO episode_cues_fts(rowid, cue_text)
    VALUES (new.rowid, new.cue_text);
END;

-- Conversation persistence (Knowledge tab)
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    group_id    TEXT NOT NULL DEFAULT 'default',
    title       TEXT,
    session_date TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conversations_group_date
    ON conversations(group_id, session_date);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user','assistant')),
    content         TEXT NOT NULL,
    parts_json      TEXT,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conv_messages_conv
    ON conversation_messages(conversation_id);

CREATE TABLE IF NOT EXISTS conversation_entities (
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    entity_id       TEXT NOT NULL REFERENCES entities(id),
    PRIMARY KEY (conversation_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_conv_entities_entity
    ON conversation_entities(entity_id);

-- Graph structural embeddings (Node2Vec, TransE, GNN)
CREATE TABLE IF NOT EXISTS graph_embeddings (
    id              TEXT NOT NULL,
    group_id        TEXT NOT NULL,
    embedding       BLOB NOT NULL,
    dimensions      INTEGER NOT NULL,
    method          TEXT NOT NULL,
    model_version   TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (id, method, group_id)
);

CREATE INDEX IF NOT EXISTS idx_graph_emb_group ON graph_embeddings(group_id);
CREATE INDEX IF NOT EXISTS idx_graph_emb_method ON graph_embeddings(method);

-- Schema formation (Brain Architecture Phase 3)
CREATE TABLE IF NOT EXISTS schema_members (
    schema_entity_id TEXT NOT NULL REFERENCES entities(id),
    role_label       TEXT NOT NULL,
    member_type      TEXT NOT NULL,
    member_predicate TEXT NOT NULL,
    group_id         TEXT NOT NULL DEFAULT 'default',
    PRIMARY KEY (schema_entity_id, role_label, group_id)
);

-- Evidence-based extraction (v2)
CREATE TABLE IF NOT EXISTS episode_evidence (
    evidence_id     TEXT PRIMARY KEY,
    episode_id      TEXT NOT NULL REFERENCES episodes(id),
    group_id        TEXT NOT NULL DEFAULT 'default',
    fact_class      TEXT NOT NULL,
    confidence      REAL NOT NULL DEFAULT 0.0,
    source_type     TEXT NOT NULL,
    extractor_name  TEXT NOT NULL DEFAULT '',
    payload_json    TEXT NOT NULL DEFAULT '{}',
    source_span     TEXT,
    signals_json    TEXT NOT NULL DEFAULT '[]',
    ambiguity_tags_json TEXT NOT NULL DEFAULT '[]',
    ambiguity_score REAL NOT NULL DEFAULT 0.0,
    adjudication_request_id TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    commit_reason   TEXT,
    committed_id    TEXT,
    deferred_cycles INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL,
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_episode ON episode_evidence(episode_id);
CREATE INDEX IF NOT EXISTS idx_evidence_status ON episode_evidence(status);
CREATE INDEX IF NOT EXISTS idx_evidence_adjudication_request ON episode_evidence(adjudication_request_id);

CREATE TABLE IF NOT EXISTS episode_adjudications (
    request_id       TEXT PRIMARY KEY,
    episode_id       TEXT NOT NULL REFERENCES episodes(id),
    group_id         TEXT NOT NULL DEFAULT 'default',
    status           TEXT NOT NULL DEFAULT 'pending',
    ambiguity_tags_json TEXT NOT NULL DEFAULT '[]',
    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
    selected_text    TEXT NOT NULL DEFAULT '',
    request_reason   TEXT NOT NULL DEFAULT '',
    resolution_source TEXT,
    resolution_payload_json TEXT,
    attempt_count    INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT NOT NULL,
    resolved_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_episode_adjudications_episode ON episode_adjudications(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_adjudications_status ON episode_adjudications(status);

-- Complement tags (Microglia phase — graph immune surveillance)
CREATE TABLE IF NOT EXISTS complement_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_type TEXT NOT NULL CHECK(target_type IN ('edge', 'entity')),
    target_id TEXT NOT NULL,
    tag_type TEXT NOT NULL,
    score REAL NOT NULL,
    cycle_tagged INTEGER NOT NULL,
    cycle_confirmed INTEGER,
    cleared INTEGER NOT NULL DEFAULT 0,
    group_id TEXT NOT NULL DEFAULT 'default',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(target_id, tag_type)
);
CREATE INDEX IF NOT EXISTS idx_complement_tags_target ON complement_tags(target_id, cleared);
