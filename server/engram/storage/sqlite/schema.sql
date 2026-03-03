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
    last_accessed   TEXT
);

CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_group ON entities(group_id);

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
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_group ON episodes(group_id);

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

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, content)
    VALUES (new.rowid, new.content);
END;
