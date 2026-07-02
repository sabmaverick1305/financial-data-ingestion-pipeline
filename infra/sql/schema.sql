-- AMFI pipeline schema  v2
-- Run once against your PostgreSQL database before the first pipeline execution.
-- Requires: PostgreSQL 15+ with pgvector extension available (supported on AWS RDS)

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS vector;         -- pgvector: vector similarity search

-- ── document_metadata ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_metadata (
  document_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source            TEXT NOT NULL,
  provider          TEXT NOT NULL,
  document_type     TEXT NOT NULL,
  category          TEXT,
  title             TEXT,
  original_url      TEXT,
  s3_raw_key        TEXT NOT NULL,
  s3_processed_key  TEXT,
  file_name         TEXT UNIQUE,
  file_type         TEXT,
  file_size_bytes   BIGINT,
  file_hash         TEXT,
  publication_date  DATE,
  period_year       INT,
  period_month      INT,
  period_quarter    TEXT,
  volume            TEXT,
  issue             TEXT,
  page_count        INT,
  language          TEXT    DEFAULT 'en',

  -- Pipeline status
  -- uploaded → text_extracted → tables_extracted → processed → embedded
  processing_status TEXT    DEFAULT 'uploaded',
  has_text_layer    BOOLEAN,

  -- v2: production-grade reliability columns
  attempt_count     INT     DEFAULT 0,
  last_error        TEXT,
  claim_expires_at  TIMESTAMPTZ,          -- TTL for zombie claim recovery
  schema_version    TEXT    DEFAULT 'v1', -- artifact schema version in S3

  created_at        TIMESTAMPTZ DEFAULT NOW(),
  updated_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── document_processing_log ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_processing_log (
  log_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id  UUID REFERENCES document_metadata(document_id) ON DELETE CASCADE,
  stage        TEXT,
  status       TEXT,
  message      TEXT,
  started_at   TIMESTAMPTZ,
  completed_at TIMESTAMPTZ
);

-- ── document_chunks ───────────────────────────────────────────────────────────
-- One row per text chunk produced by chunk-worker.
-- Embedding column uses pgvector (384 dims = all-MiniLM-L6-v2).
-- Bump EMBED_DIM and rebuild index if switching models.
CREATE TABLE IF NOT EXISTS document_chunks (
  chunk_id        UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id     UUID    NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
  chunk_index     INT     NOT NULL,          -- 0-based position within document
  text            TEXT    NOT NULL,
  char_start      INT,                       -- character offset in original full_text
  char_end        INT,
  token_count     INT,                       -- approximate word count
  embedding       vector(384),               -- all-MiniLM-L6-v2 (384 dims)
  embedding_model TEXT    DEFAULT 'all-MiniLM-L6-v2',
  period_year     INT,                       -- denormalised for fast date-range filter
  period_month    INT,
  category        TEXT,                      -- 'monthly' | 'quarterly' | 'unknown'
  created_at      TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE (document_id, chunk_index)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

-- document_metadata: fast status queue queries
CREATE INDEX IF NOT EXISTS idx_doc_status
  ON document_metadata (processing_status, has_text_layer, created_at);

-- document_metadata: stale claim detection
CREATE INDEX IF NOT EXISTS idx_doc_claim_expires
  ON document_metadata (claim_expires_at)
  WHERE claim_expires_at IS NOT NULL;

-- document_chunks: HNSW index for approximate nearest-neighbour vector search
--
-- Tune m and ef_construction based on corpus size:
--   <  100 K chunks  (< 3 K docs)   : m=16,  ef_construction=64   (current)
--   <  1 M  chunks  (<30 K docs)   : m=32,  ef_construction=128
--   < 15 M  chunks  (<500K docs)   : m=48,  ef_construction=200
--
-- At query time, raise hnsw.ef_search for better recall at the cost of latency:
--   SET hnsw.ef_search = 100;  -- default 40
--
-- To rebuild for a larger corpus (downtime required):
--   DROP INDEX idx_chunks_embedding_hnsw;
--   SET maintenance_work_mem = '4GB';
--   CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks
--     USING hnsw (embedding vector_cosine_ops) WITH (m=48, ef_construction=200);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
  ON document_chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);

-- document_chunks: full-text search (BM25-style keyword search via GIN)
CREATE INDEX IF NOT EXISTS idx_chunks_fts
  ON document_chunks USING gin (to_tsvector('english', text));

-- document_chunks: lookup all chunks for a document (used by embed-worker upsert)
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
  ON document_chunks (document_id);

-- document_chunks: time-range filters (e.g. "last 12 months")
CREATE INDEX IF NOT EXISTS idx_chunks_period
  ON document_chunks (period_year, period_month);

-- document_processing_log: lookup log entries for a document
CREATE INDEX IF NOT EXISTS idx_log_document_id
  ON document_processing_log (document_id, stage);

-- ── document_table_assets ─────────────────────────────────────────────────────
-- One row per table extracted from a document (by table-worker / ocr-worker).
-- Tracks the S3 parquet key, shape, and JSON schema for each table so
-- downstream consumers can discover and query individual tables without
-- scanning the full parquet file list.
CREATE TABLE IF NOT EXISTS document_table_assets (
  table_asset_id  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id     UUID    NOT NULL REFERENCES document_metadata(document_id) ON DELETE CASCADE,
  table_index     INT     NOT NULL,    -- 1-based, matches table_{index:03d}.parquet filename
  table_name      TEXT,               -- inferred or labelled caption (if available)
  page_number     INT,                -- page the table appeared on (NULL for spreadsheets)
  section_title   TEXT,               -- nearest heading above the table
  parquet_s3_key  TEXT    NOT NULL,   -- e.g. processed/v1/amfi/monthly/2024/06/tables/table_001.parquet
  row_count       INT,
  column_count    INT,
  schema_json     JSONB,              -- [{name, dtype}, …] column schema
  created_at      TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (document_id, table_index)
);

-- Fast lookup: all tables for a document (used when building table index)
CREATE INDEX IF NOT EXISTS idx_table_assets_document_id
  ON document_table_assets (document_id);

-- Filter by page (useful for PDF layout tools)
CREATE INDEX IF NOT EXISTS idx_table_assets_page
  ON document_table_assets (document_id, page_number);

-- ── Idempotent migrations (for existing databases) ────────────────────────────
-- Safe to run against a database that was created with the v1 schema.
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS has_text_layer   BOOLEAN;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS attempt_count    INT DEFAULT 0;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS last_error       TEXT;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS schema_version   TEXT DEFAULT 'v1';
