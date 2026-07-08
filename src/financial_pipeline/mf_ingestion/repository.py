"""Postgres repository for the mfapi.in scheme/NAV ingestion pipeline.

Owns four tables — mf_scheme_master, mf_nav_history, mf_scheme_sync_status,
mf_ingestion_log — all self-contained and separate from the AMFI PDF
pipeline's document_metadata/document_chunks and from LangGraph's query_log.
"""

from __future__ import annotations

from datetime import date

import structlog
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Engine

from financial_pipeline.mf_ingestion.models import NavRecord, SchemeRecord

log = structlog.get_logger()

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS mf_scheme_master (
  scheme_code TEXT PRIMARY KEY,
  scheme_name TEXT NOT NULL,
  amc_name TEXT,
  category TEXT,
  scheme_type TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  source TEXT DEFAULT 'mfapi',
  raw_s3_key TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mf_nav_history (
  scheme_code TEXT REFERENCES mf_scheme_master(scheme_code),
  nav_date DATE NOT NULL,
  nav NUMERIC NOT NULL,
  source TEXT DEFAULT 'mfapi',
  raw_s3_key TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (scheme_code, nav_date)
);

CREATE TABLE IF NOT EXISTS mf_scheme_sync_status (
  scheme_code TEXT PRIMARY KEY REFERENCES mf_scheme_master(scheme_code),
  sync_status TEXT NOT NULL DEFAULT 'pending',
  requested_start_date DATE,
  requested_end_date DATE,
  last_success_at TIMESTAMP,
  last_failed_at TIMESTAMP,
  last_attempt_at TIMESTAMP,
  latest_nav_date DATE,
  latest_nav_value NUMERIC,
  records_fetched INT DEFAULT 0,
  records_inserted INT DEFAULT 0,
  records_updated INT DEFAULT 0,
  retry_count INT DEFAULT 0,
  next_retry_at TIMESTAMP,
  http_status INT,
  error_message TEXT,
  raw_s3_key TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS mf_ingestion_log (
  run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_s3_key TEXT NOT NULL,
  requested_start_date DATE,
  requested_end_date DATE,
  total_schemes INT DEFAULT 0,
  valid_schemes INT DEFAULT 0,
  invalid_schemes INT DEFAULT 0,
  success_count INT DEFAULT 0,
  failed_count INT DEFAULT 0,
  rate_limited_count INT DEFAULT 0,
  no_data_count INT DEFAULT 0,
  inactive_marked INT DEFAULT 0,
  nav_rows_inserted INT DEFAULT 0,
  nav_rows_updated INT DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'running',
  error_summary TEXT,
  started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);
"""


class MFIngestionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._log = log.bind(component="mf_ingestion_repo")

    def create_tables(self) -> None:
        with self._engine.begin() as conn:
            for stmt in _CREATE_TABLES_SQL.strip().split(";\n\n"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
        self._log.info("mf_ingestion.tables_ready")

    def upsert_scheme(self, scheme: SchemeRecord, raw_s3_key: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO mf_scheme_master
                        (scheme_code, scheme_name, amc_name, category, scheme_type,
                         is_active, source, raw_s3_key, created_at, updated_at)
                    VALUES
                        (:scheme_code, :scheme_name, :amc_name, :category, :scheme_type,
                         TRUE, 'mfapi', :raw_s3_key, NOW(), NOW())
                    ON CONFLICT (scheme_code) DO UPDATE SET
                        scheme_name = EXCLUDED.scheme_name,
                        amc_name    = COALESCE(EXCLUDED.amc_name, mf_scheme_master.amc_name),
                        category    = COALESCE(EXCLUDED.category, mf_scheme_master.category),
                        scheme_type = COALESCE(EXCLUDED.scheme_type, mf_scheme_master.scheme_type),
                        is_active   = TRUE,
                        raw_s3_key  = :raw_s3_key,
                        updated_at  = NOW()
                """),
                {
                    "scheme_code": scheme.scheme_code,
                    "scheme_name": scheme.scheme_name,
                    "amc_name": scheme.amc_name,
                    "category": scheme.category,
                    "scheme_type": scheme.scheme_type,
                    "raw_s3_key": raw_s3_key,
                },
            )

    def upsert_nav_history(self, records: list[NavRecord], raw_s3_key: str | None = None) -> tuple[int, int]:
        """Bulk upsert for one scheme's NAV rows. Returns (inserted, updated)."""
        if not records:
            return (0, 0)
        scheme_code = records[0].scheme_code
        with self._engine.begin() as conn:
            existing = conn.execute(
                text("SELECT nav_date FROM mf_nav_history WHERE scheme_code = :scheme_code"),
                {"scheme_code": scheme_code},
            )
            existing_dates = {row[0] for row in existing}
            new_dates = {r.nav_date for r in records}
            updated = len(new_dates & existing_dates)
            inserted = len(new_dates - existing_dates)

            conn.execute(
                text("""
                    INSERT INTO mf_nav_history (scheme_code, nav_date, nav, source, raw_s3_key, created_at, updated_at)
                    VALUES (:scheme_code, :nav_date, :nav, 'mfapi', :raw_s3_key, NOW(), NOW())
                    ON CONFLICT (scheme_code, nav_date) DO UPDATE SET
                        nav = EXCLUDED.nav,
                        raw_s3_key = EXCLUDED.raw_s3_key,
                        updated_at = NOW()
                """),
                [
                    {"scheme_code": r.scheme_code, "nav_date": r.nav_date, "nav": r.nav, "raw_s3_key": raw_s3_key}
                    for r in records
                ],
            )
        return (inserted, updated)

    def upsert_sync_status(
        self,
        *,
        scheme_code: str,
        sync_status: str,
        requested_start_date: date,
        requested_end_date: date,
        latest_nav_date: date | None,
        latest_nav_value: float | None,
        records_fetched: int,
        records_inserted: int,
        records_updated: int,
        retry_count: int,
        http_status: int | None,
        error_message: str | None,
        raw_s3_key: str | None,
    ) -> None:
        timestamp_field = "last_success_at" if sync_status == "success" else "last_failed_at"
        with self._engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO mf_scheme_sync_status (
                        scheme_code, sync_status, requested_start_date, requested_end_date,
                        last_attempt_at, {timestamp_field},
                        latest_nav_date, latest_nav_value,
                        records_fetched, records_inserted, records_updated,
                        retry_count, http_status, error_message, raw_s3_key,
                        created_at, updated_at
                    ) VALUES (
                        :scheme_code, :sync_status, :requested_start_date, :requested_end_date,
                        NOW(), NOW(),
                        :latest_nav_date, :latest_nav_value,
                        :records_fetched, :records_inserted, :records_updated,
                        :retry_count, :http_status, :error_message, :raw_s3_key,
                        NOW(), NOW()
                    )
                    ON CONFLICT (scheme_code) DO UPDATE SET
                        sync_status = EXCLUDED.sync_status,
                        requested_start_date = EXCLUDED.requested_start_date,
                        requested_end_date = EXCLUDED.requested_end_date,
                        last_attempt_at = NOW(),
                        {timestamp_field} = NOW(),
                        latest_nav_date = COALESCE(EXCLUDED.latest_nav_date, mf_scheme_sync_status.latest_nav_date),
                        latest_nav_value = COALESCE(EXCLUDED.latest_nav_value, mf_scheme_sync_status.latest_nav_value),
                        records_fetched = EXCLUDED.records_fetched,
                        records_inserted = EXCLUDED.records_inserted,
                        records_updated = EXCLUDED.records_updated,
                        retry_count = EXCLUDED.retry_count,
                        http_status = EXCLUDED.http_status,
                        error_message = EXCLUDED.error_message,
                        raw_s3_key = EXCLUDED.raw_s3_key,
                        updated_at = NOW()
                """),
                {
                    "scheme_code": scheme_code,
                    "sync_status": sync_status,
                    "requested_start_date": requested_start_date,
                    "requested_end_date": requested_end_date,
                    "latest_nav_date": latest_nav_date,
                    "latest_nav_value": latest_nav_value,
                    "records_fetched": records_fetched,
                    "records_inserted": records_inserted,
                    "records_updated": records_updated,
                    "retry_count": retry_count,
                    "http_status": http_status,
                    "error_message": error_message,
                    "raw_s3_key": raw_s3_key,
                },
            )

    def mark_missing_inactive(self, active_scheme_codes: set[str]) -> int:
        """Mark schemes absent from the latest master list as inactive. Returns rows affected."""
        with self._engine.begin() as conn:
            if active_scheme_codes:
                stmt = text(
                    "UPDATE mf_scheme_master SET is_active = FALSE, updated_at = NOW() "
                    "WHERE is_active = TRUE AND scheme_code NOT IN :active_codes"
                ).bindparams(bindparam("active_codes", expanding=True))
                result = conn.execute(stmt, {"active_codes": list(active_scheme_codes)})
            else:
                result = conn.execute(
                    text("UPDATE mf_scheme_master SET is_active = FALSE, updated_at = NOW() WHERE is_active = TRUE")
                )
        return result.rowcount

    def start_ingestion_log(self, source_s3_key: str, requested_start_date: date, requested_end_date: date) -> str:
        with self._engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO mf_ingestion_log (source_s3_key, requested_start_date, requested_end_date, status)
                    VALUES (:source_s3_key, :start, :end, 'running')
                    RETURNING run_id
                """),
                {"source_s3_key": source_s3_key, "start": requested_start_date, "end": requested_end_date},
            )
            run_id = result.scalar_one()
        return str(run_id)

    def complete_ingestion_log(
        self,
        run_id: str,
        *,
        total_schemes: int,
        valid_schemes: int,
        invalid_schemes: int,
        success_count: int,
        failed_count: int,
        rate_limited_count: int,
        no_data_count: int,
        inactive_marked: int,
        nav_rows_inserted: int,
        nav_rows_updated: int,
        status: str,
        error_summary: str | None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE mf_ingestion_log SET
                        total_schemes = :total_schemes,
                        valid_schemes = :valid_schemes,
                        invalid_schemes = :invalid_schemes,
                        success_count = :success_count,
                        failed_count = :failed_count,
                        rate_limited_count = :rate_limited_count,
                        no_data_count = :no_data_count,
                        inactive_marked = :inactive_marked,
                        nav_rows_inserted = :nav_rows_inserted,
                        nav_rows_updated = :nav_rows_updated,
                        status = :status,
                        error_summary = :error_summary,
                        completed_at = NOW()
                    WHERE run_id = :run_id
                """),
                {
                    "run_id": run_id,
                    "total_schemes": total_schemes,
                    "valid_schemes": valid_schemes,
                    "invalid_schemes": invalid_schemes,
                    "success_count": success_count,
                    "failed_count": failed_count,
                    "rate_limited_count": rate_limited_count,
                    "no_data_count": no_data_count,
                    "inactive_marked": inactive_marked,
                    "nav_rows_inserted": nav_rows_inserted,
                    "nav_rows_updated": nav_rows_updated,
                    "status": status,
                    "error_summary": error_summary,
                },
            )
