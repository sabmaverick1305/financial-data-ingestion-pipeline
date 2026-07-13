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

-- ── /api/ask request/answer log — keyed by query_id, for feedback and eval
-- linkage. Managed by src/financial_pipeline/storage/query_log.py, created
-- at API startup, not part of the ingestion pipeline's tables above.
-- Separate from LangGraph's own checkpoint_* tables (created by
-- PostgresSaver.setup() — see storage/checkpointer.py), which persist
-- node-by-node graph STATE keyed by thread_id, not this flat summary.
CREATE TABLE IF NOT EXISTS query_log (
  query_id          UUID PRIMARY KEY,
  thread_id         UUID NOT NULL,
  question          TEXT NOT NULL,
  answer            TEXT,
  intent_type       TEXT,
  route             TEXT,
  citations_count   INT,
  latency_ms        INT,
  blocked           BOOLEAN DEFAULT FALSE,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  feedback_rating   INT,
  feedback_comment  TEXT,
  feedback_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_query_log_thread_id
  ON query_log (thread_id);

CREATE INDEX IF NOT EXISTS idx_query_log_created_at
  ON query_log (created_at);

-- ── mf_scheme_master / mf_nav_history / mf_scheme_sync_status ─────────────────
-- Per-scheme master data, daily NAV time series, and per-scheme sync/backfill
-- tracking, sourced from mfapi.in. Separate from document_metadata/
-- document_chunks (AMFI monthly/quarterly PDF ingestion) — this is a distinct
-- scheme-level dataset keyed by scheme_code (mfapi's TEXT scheme code, not the
-- AMFI PDF pipeline's document_id).
CREATE TABLE IF NOT EXISTS mf_scheme_master (
  scheme_code   TEXT PRIMARY KEY,
  scheme_name   TEXT NOT NULL,
  amc_name      TEXT,
  category      TEXT,
  scheme_type   TEXT,
  is_active     BOOLEAN DEFAULT TRUE,
  source        TEXT DEFAULT 'mfapi',
  raw_s3_key    TEXT,
  created_at    TIMESTAMP DEFAULT NOW(),
  updated_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mf_nav_history (
  scheme_code  TEXT REFERENCES mf_scheme_master(scheme_code),
  nav_date     DATE NOT NULL,
  nav          NUMERIC NOT NULL,
  source       TEXT DEFAULT 'mfapi',
  raw_s3_key   TEXT,
  created_at   TIMESTAMP DEFAULT NOW(),
  updated_at   TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (scheme_code, nav_date)
);

-- Per-scheme sync/backfill bookkeeping — one row per scheme, updated on every
-- fetch attempt so a resumable ingestion job can pick up where it left off.
CREATE TABLE IF NOT EXISTS mf_scheme_sync_status (
  scheme_code  TEXT PRIMARY KEY REFERENCES mf_scheme_master(scheme_code),

  sync_status  TEXT NOT NULL DEFAULT 'pending',
  -- pending | in_progress | success | failed | rate_limited | no_data | inactive

  requested_start_date  DATE,
  requested_end_date    DATE,

  last_success_at  TIMESTAMP,
  last_failed_at   TIMESTAMP,
  last_attempt_at  TIMESTAMP,

  latest_nav_date   DATE,
  latest_nav_value  NUMERIC,

  records_fetched   INT DEFAULT 0,
  records_inserted  INT DEFAULT 0,
  records_updated   INT DEFAULT 0,

  retry_count    INT DEFAULT 0,
  next_retry_at  TIMESTAMP,

  http_status    INT,
  error_message  TEXT,

  raw_s3_key  TEXT,

  created_at  TIMESTAMP DEFAULT NOW(),
  updated_at  TIMESTAMP DEFAULT NOW()
);

-- One row per mf-nav-sync job run — overall summary, separate from the
-- per-scheme detail in mf_scheme_sync_status above.
CREATE TABLE IF NOT EXISTS mf_ingestion_log (
  run_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  source_s3_key  TEXT NOT NULL,

  requested_start_date  DATE,
  requested_end_date    DATE,

  total_schemes    INT DEFAULT 0,
  valid_schemes    INT DEFAULT 0,
  invalid_schemes  INT DEFAULT 0,

  success_count       INT DEFAULT 0,
  failed_count        INT DEFAULT 0,
  rate_limited_count  INT DEFAULT 0,
  no_data_count       INT DEFAULT 0,
  inactive_marked     INT DEFAULT 0,

  nav_rows_inserted  INT DEFAULT 0,
  nav_rows_updated   INT DEFAULT 0,

  status  TEXT NOT NULL DEFAULT 'running',
  -- running | completed | completed_with_errors

  error_summary  TEXT,

  started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_mf_ingestion_log_started_at
  ON mf_ingestion_log (started_at);

-- ── mf_scheme_performance ──────────────────────────────────────────────────────
-- Derived/computed metrics per scheme (latest NAV, trailing returns,
-- rolling volatility, 52-week high/low), refreshed from mf_nav_history.
-- Not written by mf_ingestion/ directly — a downstream computation layer.
CREATE TABLE IF NOT EXISTS mf_scheme_performance (
  scheme_code  TEXT PRIMARY KEY REFERENCES mf_scheme_master(scheme_code),

  latest_nav       NUMERIC,
  latest_nav_date  DATE,

  return_1d  NUMERIC,
  return_1w  NUMERIC,

  return_1m  NUMERIC,
  return_3m  NUMERIC,
  return_6m  NUMERIC,

  return_1y        NUMERIC,
  return_3y_cagr    NUMERIC,
  return_5y_cagr    NUMERIC,
  return_10y_cagr   NUMERIC,

  all_time_return  NUMERIC,

  rolling_volatility  NUMERIC,
  rolling_stddev      NUMERIC,

  nav_high_52w  NUMERIC,
  nav_low_52w   NUMERIC,

  updated_at  TIMESTAMP
);

-- ── financial_entity_master / _identifier / _relationship ─────────────────────
-- The canonical entity graph (scheme / scheme_plan / organization / category
-- nodes, connected by has_plan / manages / belongs_to edges). Originally
-- created out-of-band against the live DB during an earlier session (bulk
-- backfill via scratchpad's populate_entity_{master,identifier,
-- relationship}.py) and never codified here — every column below is taken
-- directly from the exact queries services/entity_store.py already runs
-- against these tables, not newly designed. Codifying it here so a fresh
-- database can be provisioned from this file alone, matching what services/
-- entity_store.py, entity_ingestion.py, and entity_reconciliation.py expect.
CREATE TABLE IF NOT EXISTS financial_entity_master (
  entity_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_type        TEXT NOT NULL,            -- scheme | scheme_plan | organization | category
  entity_subtype     TEXT,                     -- e.g. 'mutual_fund_scheme', 'amc', taxonomy leaf id
  canonical_name     TEXT NOT NULL,
  normalized_name    TEXT NOT NULL,
  canonical_source   TEXT,                     -- mfapi | amfi | fies_taxonomy
  source_confidence  NUMERIC,
  metadata           JSONB DEFAULT '{}'::jsonb,
  lifecycle_status   TEXT NOT NULL DEFAULT 'pending',
  -- pending | active | renamed | merged | inactive | closed | archived
  created_at         TIMESTAMPTZ DEFAULT NOW(),
  updated_at         TIMESTAMPTZ DEFAULT NOW()
);

-- Natural-key uniqueness, scoped to the lifecycle statuses that count as
-- "existing" for resolution purposes — see entity_store.lookup_entity's
-- own docstring for why merged/inactive/closed/archived rows are excluded.
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_entity_identity
  ON financial_entity_master (entity_type, entity_subtype, normalized_name)
  WHERE lifecycle_status IN ('pending', 'active', 'renamed');

CREATE INDEX IF NOT EXISTS idx_entity_master_type
  ON financial_entity_master (entity_type, entity_subtype);

CREATE TABLE IF NOT EXISTS financial_entity_identifier (
  identifier_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  entity_id              UUID NOT NULL REFERENCES financial_entity_master(entity_id) ON DELETE CASCADE,
  source_system          TEXT NOT NULL,        -- mfapi | amfi | fies_taxonomy | sebi
  identifier_type        TEXT NOT NULL,        -- scheme_code | amc_entity_id | category_taxonomy_id | ...
  identifier_value       TEXT NOT NULL,
  is_primary_for_source  BOOLEAN DEFAULT TRUE,
  match_method           TEXT,                 -- deterministic | exact_identifier | ...
  match_confidence       NUMERIC,
  source_record_name     TEXT,
  created_at             TIMESTAMPTZ DEFAULT NOW(),
  updated_at             TIMESTAMPTZ DEFAULT NOW(),

  UNIQUE (source_system, identifier_type, identifier_value)
);

CREATE INDEX IF NOT EXISTS idx_entity_identifier_entity_id
  ON financial_entity_identifier (entity_id);

CREATE TABLE IF NOT EXISTS financial_entity_relationship (
  relationship_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_entity_id          UUID NOT NULL REFERENCES financial_entity_master(entity_id) ON DELETE CASCADE,
  relationship_type         TEXT NOT NULL,     -- has_plan | manages | belongs_to
  target_entity_id          UUID NOT NULL REFERENCES financial_entity_master(entity_id) ON DELETE CASCADE,
  source_system             TEXT,
  relationship_confidence   NUMERIC,
  is_inferred                BOOLEAN DEFAULT FALSE,
  is_active                  BOOLEAN DEFAULT TRUE,
  valid_from                 DATE DEFAULT CURRENT_DATE,
  valid_to                   DATE,
  created_at                 TIMESTAMPTZ DEFAULT NOW(),
  updated_at                 TIMESTAMPTZ DEFAULT NOW()
);

-- Soft-supersede pattern: only one *active* edge per (source, type, target)
-- triple. deactivate_relationship() flips is_active rather than deleting, so
-- history is preserved and a new active edge can be inserted immediately
-- after superseding an old one — see entity_store.py's own docstrings.
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_entity_relationship
  ON financial_entity_relationship (source_entity_id, relationship_type, target_entity_id)
  WHERE is_active = TRUE;

-- relationship_exists_for_source's access pattern (source_entity_id, relationship_type)
-- is already the unique index's left prefix; target-side lookups
-- (relationship_exists_for_target, used for manages) need their own index.
CREATE INDEX IF NOT EXISTS idx_entity_relationship_target
  ON financial_entity_relationship (target_entity_id, relationship_type)
  WHERE is_active = TRUE;

-- ── pipeline_run / ingestion_lineage ───────────────────────────────────────────
-- Generalizes mf_ingestion_log's run-tracking pattern to every ingestion
-- pipeline (AMFI NAV fetch, mf_ingestion sync, SEBI SID scrape, mf_performance,
-- the document worker chain), so "which job run produced this artifact" is
-- answerable for all of them, not just mf_ingestion. mf_ingestion_log itself
-- is left in place unchanged (mf_ingestion/sync.py still writes it) — this is
-- an additive, cross-pipeline layer above it, not a replacement.
CREATE TABLE IF NOT EXISTS pipeline_run (
  run_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  pipeline_name  TEXT NOT NULL,     -- e.g. 'amfi_nav_fetch', 'mf_ingestion_sync', 'sebi_sid_scrape'
  started_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at   TIMESTAMPTZ,
  status         TEXT NOT NULL DEFAULT 'running',
  -- running | completed | completed_with_errors | failed
  summary        JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_pipeline_run_name_started
  ON pipeline_run (pipeline_name, started_at);

-- One row per artifact-to-artifact transformation event within a run — the
-- Bronze -> Silver -> Gold trail. source_ref/target_ref are S3 keys, HTTP
-- URLs, or "table:primary_key" strings (e.g. 'mf_scheme_master:118825'),
-- matching the raw_s3_key convention mf_scheme_master/mf_nav_history already
-- use, generalized into a queryable log instead of a single overwritable
-- column.
CREATE TABLE IF NOT EXISTS ingestion_lineage (
  lineage_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  run_id         UUID NOT NULL REFERENCES pipeline_run(run_id) ON DELETE CASCADE,
  stage          TEXT NOT NULL,
  -- bronze_ingest | bronze_to_silver | silver_to_gold | entity_resolution
  source_type    TEXT,              -- http | s3 | postgres
  source_ref     TEXT,
  target_type    TEXT,              -- s3 | postgres
  target_ref     TEXT,
  record_count   INT,
  status         TEXT NOT NULL DEFAULT 'success',  -- success | failed
  error_message  TEXT,
  created_at     TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_lineage_run_id
  ON ingestion_lineage (run_id);

-- "What produced this row/file" and "what did this row/file produce" lookups
CREATE INDEX IF NOT EXISTS idx_ingestion_lineage_target_ref
  ON ingestion_lineage (target_ref);

CREATE INDEX IF NOT EXISTS idx_ingestion_lineage_source_ref
  ON ingestion_lineage (source_ref);

-- ── Idempotent migrations (for existing databases) ────────────────────────────
-- Safe to run against a database that was created with the v1 schema.
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS has_text_layer   BOOLEAN;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS attempt_count    INT DEFAULT 0;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS last_error       TEXT;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ;
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS schema_version   TEXT DEFAULT 'v1';

-- v3: SEBI SID documents are linked to their canonical AMC entity so lineage
-- can trace a document back through entity resolution, not just to its raw
-- S3 key — see services/sebi_ingestion/sync.py and services/lineage.py.
ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS amc_entity_id UUID
  REFERENCES financial_entity_master(entity_id);
