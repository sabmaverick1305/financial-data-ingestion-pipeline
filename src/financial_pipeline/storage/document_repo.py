from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

log = structlog.get_logger()

CLAIM_TTL_MINUTES = 30
MAX_ATTEMPTS = 3


class Status:
    UPLOADED = "uploaded"
    CLASSIFIED = "classified"
    PROCESSING_STARTED = "processing_started"
    TEXT_EXTRACTED = "text_extracted"
    TABLES_EXTRACTED = "tables_extracted"
    PROCESSED = "processed"
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"

    # Transient claim states — a row sits here only while a specific worker
    # invocation is actively processing it. claim_expires_at is set at claim
    # time so reset_stale_claims() can recover rows whose worker died without
    # reverting (SIGKILL, OOM, Fargate preemption).
    TEXT_PROCESSING = "text_processing"
    TABLE_PROCESSING = "table_processing"
    OCR_PROCESSING = "ocr_processing"
    CHUNK_PROCESSING = "chunk_processing"
    EMBED_PROCESSING = "embed_processing"

    PENDING = (UPLOADED, CLASSIFIED)

    # Worker pipeline stages:
    #   text-worker  (low memory,  PyMuPDF/pandas) : PENDING            -> TEXT_EXTRACTED
    #   table-worker (high memory, Docling text)   : TEXT_EXTRACTED (+text layer)   -> TABLES_EXTRACTED
    #   ocr-worker   (high memory, Docling OCR)    : TEXT_EXTRACTED (no text layer) -> TABLES_EXTRACTED
    #   chunk-worker (low memory,  pure python)    : TABLES_EXTRACTED  -> PROCESSED
    #   embed-worker (low memory,  sentence-transformers CPU): PROCESSED -> EMBEDDED
    TEXT_PENDING = PENDING
    TABLE_PENDING = (TEXT_EXTRACTED,)
    OCR_PENDING = (TEXT_EXTRACTED,)
    CHUNK_PENDING = (TABLES_EXTRACTED,)
    EMBED_PENDING = (PROCESSED,)

    # Maps each claim state back to the status to revert to on expiry/failure
    CLAIM_REVERT = {
        TEXT_PROCESSING: UPLOADED,
        TABLE_PROCESSING: TEXT_EXTRACTED,
        OCR_PROCESSING: TEXT_EXTRACTED,
        CHUNK_PROCESSING: TABLES_EXTRACTED,
        EMBED_PROCESSING: PROCESSED,
    }


class DocumentRepository:
    """Owns document_metadata and document_processing_log tables."""

    def __init__(self, connection_url: str) -> None:
        self._engine: Engine = create_engine(connection_url, pool_pre_ping=True)
        self._log = log.bind(backend="postgres")

    # ------------------------------------------------------------------
    # Schema bootstrap
    # ------------------------------------------------------------------

    def create_tables(self) -> None:
        """Idempotently create and migrate both tables."""
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    document_id       UUID PRIMARY KEY,
                    source            TEXT NOT NULL,
                    provider          TEXT NOT NULL,
                    document_type     TEXT NOT NULL,
                    category          TEXT,
                    title             TEXT,
                    original_url      TEXT,
                    s3_raw_key        TEXT NOT NULL,
                    s3_processed_key  TEXT,
                    file_name         TEXT,
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
                    language          TEXT DEFAULT 'en',
                    processing_status TEXT DEFAULT 'uploaded',
                    has_text_layer    BOOLEAN,
                    attempt_count     INT DEFAULT 0,
                    last_error        TEXT,
                    claim_expires_at  TIMESTAMPTZ,
                    schema_version    TEXT DEFAULT 'v1',
                    created_at        TIMESTAMPTZ DEFAULT NOW(),
                    updated_at        TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            )
            # Idempotent migrations for tables created before these columns existed.
            for migration in [
                "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS has_text_layer BOOLEAN",
                "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS attempt_count INT DEFAULT 0",
                "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS last_error TEXT",
                "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ",
                "ALTER TABLE document_metadata ADD COLUMN IF NOT EXISTS schema_version TEXT DEFAULT 'v1'",
            ]:
                conn.execute(text(migration))

            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS document_processing_log (
                    log_id       UUID PRIMARY KEY,
                    document_id  UUID REFERENCES document_metadata(document_id),
                    stage        TEXT,
                    status       TEXT,
                    message      TEXT,
                    started_at   TIMESTAMPTZ,
                    completed_at TIMESTAMPTZ
                )
            """)
            )

            # Index for fast queue queries
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_doc_status
                ON document_metadata (processing_status, has_text_layer, created_at)
            """)
            )
            # Index for stale claim detection
            conn.execute(
                text("""
                CREATE INDEX IF NOT EXISTS idx_doc_claim_expires
                ON document_metadata (claim_expires_at)
                WHERE claim_expires_at IS NOT NULL
            """)
            )

        self._log.info("db.tables_ready")

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_metadata(
        self,
        *,
        source: str,
        provider: str,
        document_type: str,
        s3_raw_key: str,
        original_url: str,
        file_name: str,
        file_size_bytes: int,
        file_hash: str,
        category: str | None = None,
        title: str | None = None,
        file_type: str | None = None,
        period_year: int | None = None,
        period_month: int | None = None,
        period_quarter: str | None = None,
        volume: str | None = None,
        issue: str | None = None,
    ) -> tuple[str, str]:
        """Insert or update a document_metadata row keyed on file_name.

        Returns (document_id, action) where action is 'inserted' or 'updated'.
        On conflict (same file_name) resets status to 'uploaded' and refreshes
        file_hash/size so the document re-enters the pipeline.
        """
        document_id = str(uuid.uuid4())
        with self._engine.begin() as conn:
            row = conn.execute(
                text("""
                    INSERT INTO document_metadata (
                        document_id, source, provider, document_type, category, title,
                        original_url, s3_raw_key, file_name, file_type,
                        file_size_bytes, file_hash,
                        period_year, period_month, period_quarter, volume, issue,
                        processing_status, attempt_count
                    ) VALUES (
                        :document_id, :source, :provider, :document_type, :category, :title,
                        :original_url, :s3_raw_key, :file_name, :file_type,
                        :file_size_bytes, :file_hash,
                        :period_year, :period_month, :period_quarter, :volume, :issue,
                        'uploaded', 0
                    )
                    ON CONFLICT (file_name) DO UPDATE SET
                        s3_raw_key        = EXCLUDED.s3_raw_key,
                        file_hash         = EXCLUDED.file_hash,
                        file_size_bytes   = EXCLUDED.file_size_bytes,
                        processing_status = 'uploaded',
                        attempt_count     = 0,
                        last_error        = NULL,
                        claim_expires_at  = NULL,
                        updated_at        = NOW()
                    RETURNING document_id, (xmax = 0) AS inserted
                """),
                {
                    "document_id": document_id,
                    "source": source,
                    "provider": provider,
                    "document_type": document_type,
                    "category": category,
                    "title": title,
                    "original_url": original_url,
                    "s3_raw_key": s3_raw_key,
                    "file_name": file_name,
                    "file_type": file_type,
                    "file_size_bytes": file_size_bytes,
                    "file_hash": file_hash,
                    "period_year": period_year,
                    "period_month": period_month,
                    "period_quarter": period_quarter,
                    "volume": volume,
                    "issue": issue,
                },
            ).fetchone()

        if row is None:
            raise RuntimeError(f"Upsert returned no row for file_name={file_name!r}")

        actual_id = str(row.document_id)
        action = "inserted" if row.inserted else "updated"
        self._log.info("db.metadata_upserted", document_id=actual_id, file_name=file_name, action=action)
        return actual_id, action

    def log_stage(
        self,
        document_id: str,
        stage: str,
        status: str,
        message: str = "",
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> None:
        now = datetime.now(tz=UTC)
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO document_processing_log
                        (log_id, document_id, stage, status, message, started_at, completed_at)
                    VALUES
                        (:log_id, :document_id, :stage, :status, :message, :started_at, :completed_at)
                """),
                {
                    "log_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "stage": stage,
                    "status": status,
                    "message": message,
                    "started_at": started_at or now,
                    "completed_at": completed_at or now,
                },
            )

    def get_documents_by_status(
        self,
        statuses: tuple[str, ...] | list[str],
        has_text_layer: bool | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        placeholders = ", ".join(f"'{s}'" for s in statuses)
        sql = f"""
            SELECT document_id, file_name, file_type, category,
                   s3_raw_key, s3_processed_key, period_year, period_month,
                   period_quarter, volume, issue, processing_status,
                   has_text_layer, attempt_count, schema_version
              FROM document_metadata
             WHERE processing_status IN ({placeholders})
        """
        params: dict = {}
        if has_text_layer is not None:
            sql += " AND has_text_layer = :has_text_layer"
            params["has_text_layer"] = has_text_layer
        sql += " ORDER BY created_at"
        if limit:
            sql += " LIMIT :limit"
            params["limit"] = limit
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

    def get_pending_documents(self, limit: int | None = None) -> list[dict]:
        return self.get_documents_by_status(Status.PENDING, limit=limit)

    def claim_documents(
        self,
        from_statuses: tuple[str, ...] | list[str],
        claim_status: str,
        has_text_layer: bool | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """Atomically claim documents using SELECT ... FOR UPDATE SKIP LOCKED.

        Sets claim_expires_at = NOW() + CLAIM_TTL_MINUTES so that
        reset_stale_claims() can recover rows whose worker died without
        reverting (SIGKILL, OOM, Fargate preemption).
        """
        placeholders = ", ".join(f"'{s}'" for s in from_statuses)
        text_filter = ""
        params: dict = {"claim_status": claim_status}
        if has_text_layer is not None:
            text_filter = "AND has_text_layer = :has_text_layer"
            params["has_text_layer"] = has_text_layer
        limit_clause = ""
        if limit:
            limit_clause = "LIMIT :limit"
            params["limit"] = limit

        sql = f"""
            WITH claimed AS (
                SELECT document_id
                  FROM document_metadata
                 WHERE processing_status IN ({placeholders})
                   {text_filter}
                 ORDER BY created_at
                 {limit_clause}
                   FOR UPDATE SKIP LOCKED
            )
            UPDATE document_metadata d
               SET processing_status = :claim_status,
                   claim_expires_at  = NOW() + INTERVAL '{CLAIM_TTL_MINUTES} minutes',
                   updated_at        = NOW()
              FROM claimed
             WHERE d.document_id = claimed.document_id
            RETURNING d.document_id, d.file_name, d.file_type, d.category,
                      d.s3_raw_key, d.s3_processed_key, d.period_year, d.period_month,
                      d.period_quarter, d.volume, d.issue, d.processing_status,
                      d.has_text_layer, d.attempt_count, d.schema_version
        """
        with self._engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        self._log.info("db.documents_claimed", count=len(rows), claim_status=claim_status)
        return [dict(r) for r in rows]

    def update_status(
        self,
        document_id: str,
        status: str,
        s3_processed_key: str | None = None,
        has_text_layer: bool | None = None,
        schema_version: str | None = None,
    ) -> None:
        """Advance status on success; clears claim_expires_at."""
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE document_metadata
                       SET processing_status  = :status,
                           s3_processed_key   = COALESCE(:s3_processed_key, s3_processed_key),
                           has_text_layer     = COALESCE(:has_text_layer, has_text_layer),
                           schema_version     = COALESCE(:schema_version, schema_version),
                           claim_expires_at   = NULL,
                           last_error         = NULL,
                           updated_at         = NOW()
                     WHERE document_id = :document_id
                """),
                {
                    "document_id": document_id,
                    "status": status,
                    "s3_processed_key": s3_processed_key,
                    "has_text_layer": has_text_layer,
                    "schema_version": schema_version,
                },
            )
        self._log.info("db.status_updated", document_id=document_id, status=status)

    def record_failure(
        self,
        document_id: str,
        error: str,
        revert_status: str,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> str:
        """Increment attempt_count and either revert for retry or mark failed.

        Returns the new processing_status ('failed' or revert_status).
        Workers should call this instead of update_status() on exception so
        documents that fail repeatedly don't re-queue forever.
        """
        with self._engine.begin() as conn:
            row = conn.execute(
                text("""
                    UPDATE document_metadata
                       SET attempt_count    = attempt_count + 1,
                           last_error       = :error,
                           claim_expires_at = NULL,
                           processing_status = CASE
                               WHEN attempt_count + 1 >= :max_attempts THEN 'failed'
                               ELSE :revert_status
                           END,
                           updated_at = NOW()
                     WHERE document_id = :document_id
                    RETURNING processing_status, attempt_count
                """),
                {
                    "document_id": document_id,
                    "error": error[:2000],  # guard against huge stack traces
                    "revert_status": revert_status,
                    "max_attempts": max_attempts,
                },
            ).fetchone()

        new_status = row.processing_status if row else revert_status
        attempt = row.attempt_count if row else "?"
        self._log.warning(
            "db.failure_recorded",
            document_id=document_id,
            new_status=new_status,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        return new_status

    # ------------------------------------------------------------------
    # Operational helpers
    # ------------------------------------------------------------------

    def reset_stale_claims(self, dry_run: bool = False) -> list[dict]:
        """Reset documents whose claim TTL has expired (worker died mid-process).

        Safe to call from any process — uses a single atomic UPDATE so
        concurrent callers don't double-reset. Returns the list of reset rows.
        Run this on worker startup and via EventBridge every 5 minutes.
        """
        claim_states = list(Status.CLAIM_REVERT.keys())
        placeholders = ", ".join(f"'{s}'" for s in claim_states)

        # Build a CASE expression for per-state revert targets
        case_branches = " ".join(f"WHEN '{claim}' THEN '{revert}'" for claim, revert in Status.CLAIM_REVERT.items())

        if dry_run:
            sql = f"""
                SELECT document_id, file_name, processing_status, claim_expires_at
                  FROM document_metadata
                 WHERE processing_status IN ({placeholders})
                   AND claim_expires_at < NOW()
            """
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql)).mappings().all()
            result = [dict(r) for r in rows]
        else:
            sql = f"""
                UPDATE document_metadata
                   SET processing_status = CASE processing_status {case_branches} END,
                       claim_expires_at  = NULL,
                       updated_at        = NOW()
                 WHERE processing_status IN ({placeholders})
                   AND claim_expires_at < NOW()
                RETURNING document_id, file_name, processing_status
            """
            with self._engine.begin() as conn:
                rows = conn.execute(text(sql)).mappings().all()
            result = [dict(r) for r in rows]

        if result:
            self._log.warning(
                "db.stale_claims_reset",
                count=len(result),
                dry_run=dry_run,
                files=[r["file_name"] for r in result[:10]],
            )
        return result

    def queue_depths(self) -> dict[str, int]:
        """Return pending count for each pipeline stage — use for CloudWatch metrics."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                SELECT
                    SUM(CASE WHEN processing_status IN ('uploaded','classified') THEN 1 ELSE 0 END)                  AS text_pending, # noqa: E501
                    SUM(CASE WHEN processing_status = 'text_extracted' AND has_text_layer = true  THEN 1 ELSE 0 END) AS table_pending, # noqa: E501
                    SUM(CASE WHEN processing_status = 'text_extracted' AND has_text_layer = false THEN 1 ELSE 0 END) AS ocr_pending, # noqa: E501
                    SUM(CASE WHEN processing_status = 'tables_extracted' THEN 1 ELSE 0 END)                          AS chunk_pending, # noqa: E501
                    SUM(CASE WHEN processing_status = 'processed'        THEN 1 ELSE 0 END)                          AS embed_pending, # noqa: E501
                    SUM(CASE WHEN processing_status = 'embedded'         THEN 1 ELSE 0 END)                          AS embedded,
                    SUM(CASE WHEN processing_status = 'failed'           THEN 1 ELSE 0 END)                          AS failed
                FROM document_metadata
            """)
            ).fetchone()
        return {
            "text_pending": int(rows[0] or 0),
            "table_pending": int(rows[1] or 0),
            "ocr_pending": int(rows[2] or 0),
            "chunk_pending": int(rows[3] or 0),
            "embed_pending": int(rows[4] or 0),
            "embedded": int(rows[5] or 0),
            "failed": int(rows[6] or 0),
        }

    def get_failed_documents(self, limit: int = 100) -> list[dict]:
        """Return documents in terminal 'failed' state for review."""
        with self._engine.connect() as conn:
            rows = (
                conn.execute(
                    text("""
                SELECT document_id, file_name, attempt_count, last_error, updated_at
                  FROM document_metadata
                 WHERE processing_status = 'failed'
                 ORDER BY updated_at DESC
                 LIMIT :limit
            """),
                    {"limit": limit},
                )
                .mappings()
                .all()
            )
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Indexing layer — pgvector
    # ------------------------------------------------------------------

    def store_chunk_embeddings(
        self,
        document_id: str,
        chunks: list[dict],
        embeddings: list[list[float]],
        model_name: str,
        period_year: int | None = None,
        period_month: int | None = None,
        category: str | None = None,
    ) -> int:
        """Upsert chunks + their vector embeddings into document_chunks.

        Deletes any existing chunks for the document first (idempotent re-run).
        Returns the number of rows inserted.
        """
        if not chunks:
            return 0

        with self._engine.begin() as conn:
            # Register pgvector type so psycopg2 accepts Python lists as vector
            from pgvector.psycopg2 import register_vector

            register_vector(conn.connection.dbapi_connection)

            # Delete stale chunks (idempotent re-embedding)
            conn.execute(
                text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )

            rows = [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "chunk_index": chunk.get("chunk_id", i),
                    "text": chunk["text"],
                    "char_start": chunk.get("start"),
                    "char_end": chunk.get("end"),
                    "token_count": len(chunk["text"].split()),
                    "embedding": embeddings[i],
                    "embedding_model": model_name,
                    "period_year": period_year,
                    "period_month": period_month,
                    "category": category,
                }
                for i, chunk in enumerate(chunks)
            ]

            conn.execute(
                text("""
                    INSERT INTO document_chunks
                        (chunk_id, document_id, chunk_index, text,
                         char_start, char_end, token_count,
                         embedding, embedding_model,
                         period_year, period_month, category)
                    VALUES
                        (:chunk_id, :document_id, :chunk_index, :text,
                         :char_start, :char_end, :token_count,
                         :embedding, :embedding_model,
                         :period_year, :period_month, :category)
                """),
                rows,
            )

        self._log.info(
            "db.chunks_stored",
            document_id=document_id,
            count=len(rows),
            model=model_name,
        )
        return len(rows)

    def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        period_year: int | None = None,
        period_month: int | None = None,
        category: str | None = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """Vector similarity search using pgvector cosine distance.

        Returns chunks ranked by cosine similarity to the query embedding,
        with optional filters on year, month, and document category.
        """
        filters = []
        params: dict = {
            "embedding": query_embedding,
            "limit": limit,
            "min_sim": 1.0 - min_similarity,  # cosine distance threshold
        }
        if period_year is not None:
            filters.append("dc.period_year = :period_year")
            params["period_year"] = period_year
        if period_month is not None:
            filters.append("dc.period_month = :period_month")
            params["period_month"] = period_month
        if category is not None:
            filters.append("dc.category = :category")
            params["category"] = category

        where = ("WHERE " + " AND ".join(filters)) if filters else ""

        sql = f"""
            SELECT
                dc.chunk_id,
                dc.chunk_index,
                dc.text,
                dc.char_start,
                dc.char_end,
                dc.period_year,
                dc.period_month,
                dc.category,
                dm.file_name,
                dm.document_type,
                dm.s3_processed_key,
                1 - (dc.embedding <=> CAST(:embedding AS vector)) AS similarity
            FROM document_chunks dc
            JOIN document_metadata dm ON dc.document_id = dm.document_id
            {where}
            ORDER BY dc.embedding <=> CAST(:embedding AS vector)
            LIMIT :limit
        """
        with self._engine.connect() as conn:
            from pgvector.psycopg2 import register_vector

            register_vector(conn.connection.dbapi_connection)
            rows = conn.execute(text(sql), params).mappings().all()

        return [dict(r) for r in rows if r["similarity"] >= min_similarity]

    def search_fulltext(
        self,
        query: str,
        limit: int = 10,
        period_year: int | None = None,
        period_month: int | None = None,
        category: str | None = None,
    ) -> list[dict]:
        """Full-text keyword search via PostgreSQL tsvector / GIN index."""
        filters = ["to_tsvector('english', dc.text) @@ plainto_tsquery('english', :query)"]
        params: dict = {"query": query, "limit": limit}

        if period_year is not None:
            filters.append("dc.period_year = :period_year")
            params["period_year"] = period_year
        if period_month is not None:
            filters.append("dc.period_month = :period_month")
            params["period_month"] = period_month
        if category is not None:
            filters.append("dc.category = :category")
            params["category"] = category

        where = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT
                dc.chunk_id,
                dc.chunk_index,
                dc.text,
                dc.period_year,
                dc.period_month,
                dc.category,
                dm.file_name,
                dm.document_type,
                dm.s3_processed_key,
                ts_rank(to_tsvector('english', dc.text),
                        plainto_tsquery('english', :query)) AS rank
            FROM document_chunks dc
            JOIN document_metadata dm ON dc.document_id = dm.document_id
            {where}
            ORDER BY rank DESC
            LIMIT :limit
        """
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(r) for r in rows]

    def chunk_count(self) -> int:
        """Total number of embedded chunks across all documents."""
        with self._engine.connect() as conn:
            return conn.execute(text("SELECT count(*) FROM document_chunks")).scalar() or 0

    # ------------------------------------------------------------------
    # Table asset registry
    # ------------------------------------------------------------------

    def register_table_assets(
        self,
        document_id: str,
        tables: list,  # list[pd.DataFrame]
        table_meta: list,  # list[TableMeta] — parallel to tables
        prefix: str,
    ) -> int:
        """Upsert one document_table_assets row per extracted table.

        Idempotent: deletes existing rows for this document first so re-runs
        after a table-worker retry don't create duplicates.
        Returns the number of rows inserted.
        """
        import json as _json

        if not tables:
            return 0

        rows = []
        for i, (df, meta) in enumerate(zip(tables, table_meta or [{}] * len(tables)), start=1):
            page_no = getattr(meta, "page_number", None)
            table_name = getattr(meta, "table_name", None)
            section_title = getattr(meta, "section_title", None)
            schema_json = _json.dumps([{"name": str(col), "dtype": str(df[col].dtype)} for col in df.columns])
            rows.append(
                {
                    "table_asset_id": str(uuid.uuid4()),
                    "document_id": document_id,
                    "table_index": i,
                    "table_name": table_name,
                    "page_number": page_no,
                    "section_title": section_title,
                    "parquet_s3_key": f"{prefix}/tables/table_{i:03d}.parquet",
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "schema_json": schema_json,
                }
            )

        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM document_table_assets WHERE document_id = :doc_id"),
                {"doc_id": document_id},
            )
            conn.execute(
                text("""
                    INSERT INTO document_table_assets
                        (table_asset_id, document_id, table_index, table_name,
                         page_number, section_title, parquet_s3_key,
                         row_count, column_count, schema_json)
                    VALUES
                        (:table_asset_id, :document_id, :table_index, :table_name,
                         :page_number, :section_title, :parquet_s3_key,
                         :row_count, :column_count, CAST(:schema_json AS JSONB))
                """),
                rows,
            )

        self._log.info(
            "db.table_assets_registered",
            document_id=document_id,
            count=len(rows),
        )
        return len(rows)
