"""Orchestrates the mf_scheme_performance calculation job.

Flow: for every scheme_code in mf_scheme_master -> fetch its full NAV
history (one query) -> compute returns/CAGR/volatility/52w-high-low in
Python -> upsert into mf_scheme_performance (one write). Schemes are
processed concurrently via a thread pool — this is pure DB I/O (no external
API calls), so there's no rate-limit concern, only DB connection pool sizing.

Intended to run immediately after mf_ingestion.sync.run_sync() completes,
in the same nightly job (see graph: ingestion -> performance calc -> table).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from sqlalchemy import create_engine

from financial_pipeline.mf_performance.calculator import compute_performance
from financial_pipeline.mf_performance.repository import MFPerformanceRepository
from financial_pipeline.services import entity_store, lineage

log = structlog.get_logger()

# One psycopg2 connection per worker thread for lineage writes, reused across
# every scheme that thread processes — same rationale as mf_ingestion/sync.py's
# _entity_conn (avoids a fresh cross-region-RDS connection per scheme).
_lineage_conn_local = threading.local()


def _lineage_conn():
    conn = getattr(_lineage_conn_local, "conn", None)
    if conn is None or conn.closed:
        conn = entity_store.connect()
        _lineage_conn_local.conn = conn
    return conn


def _process_one(repo: MFPerformanceRepository, scheme_code: str, run_id: str) -> bool:
    history = repo.fetch_history(scheme_code)
    metrics = compute_performance(scheme_code, history)
    if metrics is None:
        return False
    repo.upsert_performance(metrics)

    conn = _lineage_conn()
    cur = conn.cursor()
    lineage.record_transform(
        cur,
        run_id=run_id,
        stage="silver_to_gold",
        source_type="postgres",
        source_ref=f"mf_nav_history:{scheme_code}",
        target_type="postgres",
        target_ref=f"mf_scheme_performance:{scheme_code}",
        record_count=1,
    )
    conn.commit()
    cur.close()
    return True


def calculate_all_performance(*, postgres_url: str, max_workers: int = 15) -> dict:
    engine = create_engine(postgres_url, pool_pre_ping=True, pool_size=max_workers, max_overflow=max_workers)
    repo = MFPerformanceRepository(engine)
    repo.create_table()

    scheme_codes = repo.fetch_all_scheme_codes()
    run_log = log.bind(job="mf_performance_calc")
    run_log.info("mf_performance.started", total_schemes=len(scheme_codes), max_workers=max_workers)

    lineage_conn = entity_store.connect()
    lineage_cur = lineage_conn.cursor()
    run_id = lineage.start_run(lineage_cur, "mf_performance_calc")
    lineage_conn.commit()
    lineage_cur.close()

    computed = 0
    skipped = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, repo, code, run_id): code for code in scheme_codes}
        for future in as_completed(futures):
            try:
                ok = future.result()
            except Exception:
                run_log.exception("mf_performance.scheme_failed", scheme_code=futures[future])
                ok = False
            if ok:
                computed += 1
            else:
                skipped += 1
            done += 1
            if done % 2000 == 0:
                run_log.info("mf_performance.progress", done=done, total=len(scheme_codes))

    result = {"total_schemes": len(scheme_codes), "computed": computed, "skipped": skipped}
    run_log.info("mf_performance.completed", **result)

    lineage_cur = lineage_conn.cursor()
    lineage.complete_run(lineage_cur, run_id, "completed", **result)
    lineage_conn.commit()
    lineage_cur.close()
    lineage_conn.close()

    return result
