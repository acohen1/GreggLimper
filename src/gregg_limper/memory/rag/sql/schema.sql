-- Core fragments table
CREATE TABLE IF NOT EXISTS fragments (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  server_id     INTEGER NOT NULL,
  channel_id    INTEGER NOT NULL,
  message_id    INTEGER NOT NULL,
  author_id     INTEGER NOT NULL,
  ts            REAL    NOT NULL,

  content       TEXT    NOT NULL,
  type          TEXT    NOT NULL, -- mirrors cache fragment.type
  title         TEXT,
  url           TEXT,
  media_id      TEXT    NOT NULL, -- unique ID generated in ingest.py

  embedding     BLOB    NOT NULL, -- float32 bytes
  emb_model     TEXT    NOT NULL, -- e.g., "text-embedding-3-small"
  emb_dim       INTEGER NOT NULL, -- e.g., 1536
  last_embedded_ts REAL NOT NULL DEFAULT 0,

  source_idx    INTEGER NOT NULL,
  content_hash  TEXT    NOT NULL,

  UNIQUE(message_id, source_idx, type, content_hash)
);

CREATE INDEX IF NOT EXISTS idx_frag_sc  ON fragments(server_id, channel_id);
CREATE INDEX IF NOT EXISTS idx_frag_ts  ON fragments(ts);
CREATE INDEX IF NOT EXISTS idx_frag_msg ON fragments(message_id);
CREATE INDEX IF NOT EXISTS idx_frag_media ON fragments(media_id);

-- Simple metadata tables (JSON blobs)
-- Consent allowlist
CREATE TABLE IF NOT EXISTS rag_consent (
  user_id INTEGER PRIMARY KEY,
  ts      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_profiles (
  user_id   INTEGER PRIMARY KEY,
  blob      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS server_styles (
  server_id INTEGER PRIMARY KEY,
  blob      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS channel_summaries (
  channel_id INTEGER PRIMARY KEY,
  summary    TEXT NOT NULL
);
