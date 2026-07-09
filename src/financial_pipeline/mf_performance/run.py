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

from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from sqlalchemy import create_engine

from financial_pipeline.mf_performance.calculator import compute_performance
from financial_pipeline.mf_performance.repository import MFPerformanceRepository

log = structlog.get_logger()


def _process_one(repo: MFPerformanceRepository, scheme_code: str) -> bool:
    history = repo.fetch_history(scheme_code)
    metrics = compute_performance(scheme_code, history)
    if metrics is None:
        return False
    repo.upsert_performance(metrics)
    return True


def calculate_all_performance(*, postgres_url: str, max_workers: int = 15) -> dict:
    engine = create_engine(postgres_url, pool_pre_ping=True, pool_size=max_workers, max_overflow=max_workers)
    repo = MFPerformanceRepository(engine)
    repo.create_table()

    scheme_codes = repo.fetch_all_scheme_codes()
    run_log = log.bind(job="mf_performance_calc")
    run_log.info("mf_performance.started", total_schemes=len(scheme_codes), max_workers=max_workers)

    computed = 0
    skipped = 0
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, repo, code): code for code in scheme_codes}
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
    return result
