"""Postgres repository for the mf_scheme_performance calculation job.

Reads from mf_scheme_master/mf_nav_history (owned by mf_ingestion/), writes
to mf_scheme_performance. Self-contained — a downstream computation layer,
not part of the ingestion pipeline itself.
"""

from __future__ import annotations

from datetime import date

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

from financial_pipeline.mf_performance.calculator import PerformanceMetrics

log = structlog.get_logger()

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS mf_scheme_performance (
  scheme_code TEXT PRIMARY KEY REFERENCES mf_scheme_master(scheme_code),
  latest_nav NUMERIC,
  latest_nav_date DATE,
  return_1d NUMERIC,
  return_1w NUMERIC,
  return_1m NUMERIC,
  return_3m NUMERIC,
  return_6m NUMERIC,
  return_1y NUMERIC,
  return_3y_cagr NUMERIC,
  return_5y_cagr NUMERIC,
  return_10y_cagr NUMERIC,
  all_time_return NUMERIC,
  rolling_volatility NUMERIC,
  rolling_stddev NUMERIC,
  nav_high_52w NUMERIC,
  nav_low_52w NUMERIC,
  updated_at TIMESTAMP
);
"""


class MFPerformanceRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._log = log.bind(component="mf_performance_repo")

    def create_table(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(_CREATE_TABLE_SQL))
        self._log.info("mf_performance.table_ready")

    def fetch_all_scheme_codes(self) -> list[str]:
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT scheme_code FROM mf_scheme_master"))
            return [row[0] for row in result]

    def fetch_history(self, scheme_code: str) -> list[tuple[date, float]]:
        with self._engine.connect() as conn:
            result = conn.execute(
                text("SELECT nav_date, nav FROM mf_nav_history WHERE scheme_code = :scheme_code ORDER BY nav_date"),
                {"scheme_code": scheme_code},
            )
            return [(row[0], float(row[1])) for row in result]

    def upsert_performance(self, metrics: PerformanceMetrics) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO mf_scheme_performance (
                        scheme_code, latest_nav, latest_nav_date,
                        return_1d, return_1w, return_1m, return_3m, return_6m, return_1y,
                        return_3y_cagr, return_5y_cagr, return_10y_cagr,
                        all_time_return, rolling_volatility, rolling_stddev,
                        nav_high_52w, nav_low_52w, updated_at
                    ) VALUES (
                        :scheme_code, :latest_nav, :latest_nav_date,
                        :return_1d, :return_1w, :return_1m, :return_3m, :return_6m, :return_1y,
                        :return_3y_cagr, :return_5y_cagr, :return_10y_cagr,
                        :all_time_return, :rolling_volatility, :rolling_stddev,
                        :nav_high_52w, :nav_low_52w, NOW()
                    )
                    ON CONFLICT (scheme_code) DO UPDATE SET
                        latest_nav = EXCLUDED.latest_nav,
                        latest_nav_date = EXCLUDED.latest_nav_date,
                        return_1d = EXCLUDED.return_1d,
                        return_1w = EXCLUDED.return_1w,
                        return_1m = EXCLUDED.return_1m,
                        return_3m = EXCLUDED.return_3m,
                        return_6m = EXCLUDED.return_6m,
                        return_1y = EXCLUDED.return_1y,
                        return_3y_cagr = EXCLUDED.return_3y_cagr,
                        return_5y_cagr = EXCLUDED.return_5y_cagr,
                        return_10y_cagr = EXCLUDED.return_10y_cagr,
                        all_time_return = EXCLUDED.all_time_return,
                        rolling_volatility = EXCLUDED.rolling_volatility,
                        rolling_stddev = EXCLUDED.rolling_stddev,
                        nav_high_52w = EXCLUDED.nav_high_52w,
                        nav_low_52w = EXCLUDED.nav_low_52w,
                        updated_at = NOW()
                """),
                {
                    "scheme_code": metrics.scheme_code,
                    "latest_nav": metrics.latest_nav,
                    "latest_nav_date": metrics.latest_nav_date,
                    "return_1d": metrics.return_1d,
                    "return_1w": metrics.return_1w,
                    "return_1m": metrics.return_1m,
                    "return_3m": metrics.return_3m,
                    "return_6m": metrics.return_6m,
                    "return_1y": metrics.return_1y,
                    "return_3y_cagr": metrics.return_3y_cagr,
                    "return_5y_cagr": metrics.return_5y_cagr,
                    "return_10y_cagr": metrics.return_10y_cagr,
                    "all_time_return": metrics.all_time_return,
                    "rolling_volatility": metrics.rolling_volatility,
                    "rolling_stddev": metrics.rolling_stddev,
                    "nav_high_52w": metrics.nav_high_52w,
                    "nav_low_52w": metrics.nav_low_52w,
                },
            )
