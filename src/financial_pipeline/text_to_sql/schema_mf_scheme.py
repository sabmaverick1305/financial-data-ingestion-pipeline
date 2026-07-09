"""DDL references for the per-scheme mutual fund tables (mfapi.in dataset).

mf_scheme_master / mf_nav_history / mf_scheme_performance are owned and
created by mf_ingestion/ and mf_performance/ respectively (see their
repository.py modules) — this module only re-declares the DDL as plain
strings for Vanna's training corpus (scripts/train_vanna.py), so the SQL
agent has schema context without importing the ingestion/performance code.
"""
from __future__ import annotations

MF_SCHEME_MASTER_DDL = """
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
"""

MF_NAV_HISTORY_DDL = """
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
"""

MF_SCHEME_PERFORMANCE_DDL = """
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
