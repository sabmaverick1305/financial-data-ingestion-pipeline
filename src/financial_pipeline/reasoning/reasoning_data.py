"""Fetches real MetricObservations for the reasoning engine from
amfi_fund_stats — the DB-first pattern already proven in
graph/nodes_analytical.py's query_db_stats (same ILIKE fund_category
matching, same column mapping), reused here rather than re-derived, since
that pattern is what's actually tested against the live schema.

Scoped to period_year >= 2020 (amfi_fund_stats) only. amfi_amc_stats
(pre-2020) uses a different, coarser scheme_type taxonomy — SEBI's fund
category reclassification only happened in Oct 2017, so "large cap"/"mid
cap"-level causal reasoning isn't meaningful before 2020 anyway.
"""
from __future__ import annotations

import structlog
from sqlalchemy import text as sa_text

from financial_pipeline.reasoning.reasoning_engine import MetricObservation

log = structlog.get_logger()

# amfi_fund_stats column per metric_id — same mapping as
# nodes_analytical.py's _FUND_STATS_COLS, duplicated rather than imported to
# keep the reasoning package independent of the analytical-agent module.
_FUND_STATS_COLS: dict[str, str] = {
    "aum":             "aum",
    "avg_aum":         "avg_aum",
    "funds_mobilized": "funds_mobilized",
    "net_inflow":      "net_inflow",
    "redemption":      "redemption",
    "folios":          "no_of_folios",
}

MIN_SUPPORTED_YEAR = 2020


def fetch_observations(
    repo,
    scheme_type_display: str,
    metric_ids: list[str],
    year_from: int,
    year_to: int,
) -> dict[str, MetricObservation]:
    """Fetch each metric's first and last available monthly value in
    [year_from, year_to] for a fund category (ILIKE match on fund_category,
    e.g. scheme_type_display="large cap" matches "Large Cap Fund").

    Returns only metrics that had at least 2 distinct data points (so a
    period-over-period comparison is actually meaningful) — a metric with
    zero/one data points is silently omitted rather than fabricating a
    change from a single number.
    """
    if year_to < MIN_SUPPORTED_YEAR:
        log.info("reasoning_data.pre_2020_unsupported", year_to=year_to)
        return {}
    year_from = max(year_from, MIN_SUPPORTED_YEAR)

    observations: dict[str, MetricObservation] = {}
    scheme_pattern = f"%{scheme_type_display}%"

    for metric_id in metric_ids:
        col = _FUND_STATS_COLS.get(metric_id)
        if col is None:
            continue
        sql = f"""
            SELECT period_year, period_month, {col}
            FROM   amfi_fund_stats
            WHERE  fund_category ILIKE :scheme
              AND  period_year BETWEEN :year_from AND :year_to
              AND  {col} IS NOT NULL
            ORDER BY period_year, period_month
        """
        try:
            with repo._engine.connect() as conn:
                rows = conn.execute(
                    sa_text(sql),
                    {"scheme": scheme_pattern, "year_from": year_from, "year_to": year_to},
                ).fetchall()
        except Exception as exc:
            log.warning("reasoning_data.fetch_failed", metric=metric_id, error=str(exc))
            continue

        if len(rows) < 2:
            continue

        first_yr, first_mo, first_val = rows[0]
        last_yr, last_mo, last_val = rows[-1]
        observations[metric_id] = MetricObservation(
            metric_id=metric_id,
            start_value=float(first_val),
            end_value=float(last_val),
            start_label=f"{first_mo:02d}/{first_yr}",
            end_label=f"{last_mo:02d}/{last_yr}",
        )

    return observations
